using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>The user-selected Draven rule: private command look, public tokens,
/// then an optional matching raid after each enemy command has completed.</summary>
public static class DravenAbilityEffects
{
    public const string CardId = "general-draven-commander";
    public const string DravenLookAction = "commander.draven-look";
    public const string NoLookAction = "commander.draven-no-look";
    public const string TokenRule = "draven.tokens";
    public const string TokenAction = "commander.draven-token";
    public const string PairAction = "commander.draven-token-pair";
    public const string SkipTokensAction = "commander.draven-skip-tokens";
    public const string RaidAction = "commander.draven-raid";
    public const string RaidRule = "draven.raid";
    public const string PassRaidAction = "commander.draven-pass-raid";
    public static bool Handles(string rule) => rule is TokenRule or RaidRule;

    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        if (timing != "end-command-phase") yield break;
        foreach (var upgrade in game.Topology.Upgrades)
        {
            if (upgrade.IsSquadronLeader || upgrade.DefinitionId != CardId) continue;
            int carrier = checked((int)game.State.Registers[game.Topology.Registers.UpgradeOwner(upgrade.UpgradeIndex)] - 1);
            if (!game.IsShip(carrier) || game.IsDestroyed(carrier) || game.IsReserved(carrier)) continue;
            int owner = game.State.Owners[carrier];
            var card = game.State.Upgrades[upgrade.UpgradeIndex];
            if (!card.Active) continue;
            if (timing == "end-command-phase")
            {
                // "Then you may" is independent of taking the optional look.
                yield return new(new(GameActionKind.UseAbility, carrier, Index: upgrade.UpgradeIndex, AbilityId: NoLookAction),
                    "Skip the private look and choose whether to place Draven tokens");
                foreach (var enemy in game.Topology.Ships)
                    if (enemy.Owner != owner && !game.IsDestroyed(enemy.EntityIndex) && !game.IsReserved(enemy.EntityIndex))
                        yield return new(new(GameActionKind.UseAbility, carrier, enemy.EntityIndex,
                            Index: upgrade.UpgradeIndex, AbilityId: DravenLookAction),
                            "Look at enemy command dials with General Draven");
            }
        }
    }

    public static bool HasAfterCommandEffect(GameContext game, int ship) =>
        game.Flow.ShipActivations[ship].CurrentResolvedCommand is { } command && FindReactionCard(game, ship, command) >= 0;

    private static int FindReactionCard(GameContext game, int ship, ShipCommand command)
    {
        // Most search positions have no Draven. Avoid opening and enumerating a
        // whole general timing window for every resolved command in those games.
        if (!game.IsShip(ship) || game.IsDestroyed(ship) || game.IsReserved(ship)) return -1;
        foreach (var upgrade in game.Topology.Upgrades)
        {
            if (upgrade.IsSquadronLeader || upgrade.DefinitionId != CardId) continue;
            int carrier = checked((int)game.State.Registers[game.Topology.Registers.UpgradeOwner(upgrade.UpgradeIndex)] - 1);
            if (!game.IsShip(carrier) || game.State.Owners[carrier] == game.State.Owners[ship] ||
                game.IsDestroyed(carrier) || game.IsReserved(carrier)) continue;
            var card = game.State.Upgrades[upgrade.UpgradeIndex];
            if (card.Active && card.GetTypedCommandTokenCount(command) > 0) return upgrade.UpgradeIndex;
        }
        return -1;
    }

    public static bool BeginReaction(GameContext game, int ship, ShipCommand command, GamePhase resumePhase)
    {
        int upgrade = FindReactionCard(game, ship, command);
        if (upgrade < 0) return false;
        int carrier = checked((int)game.State.Registers[game.Topology.Registers.UpgradeOwner(upgrade)] - 1);
        EngineEffects.PushChoice(game, new(RaidRule, carrier, ship, Stage: (int)resumePhase,
            Value: (int)command, Indices: [upgrade]), []);
        return true;
    }

    private static string PendingKey(int ship) => "activation.draven-pending." + ship;
    public static bool DeferCompletion(GameContext game, int ship, ShipCommand command)
    {
        if (command is not (ShipCommand.Navigate or ShipCommand.ConcentrateFire)) return false;
        ArmDeferred(game, ship, command); return true;
    }
    public static void ArmDeferred(GameContext game, int ship, ShipCommand command)
    {
        if (FindReactionCard(game, ship, command) < 0) return;
        string key = PendingKey(ship);
        game.Flow.Flags[key] = game.Flow.Flags.GetValueOrDefault(key) | (1L << (int)command);
    }
    public static bool CompleteDeferred(GameContext game, int ship, ShipCommand command, GamePhase resumePhase)
    {
        string key = PendingKey(ship); long pending = game.Flow.Flags.GetValueOrDefault(key), bit = 1L << (int)command;
        if ((pending & bit) == 0) return false;
        pending &= ~bit;
        if (pending == 0) game.Flow.Flags.Remove(key); else game.Flow.Flags[key] = pending;
        return BeginReaction(game, ship, command, resumePhase);
    }
    public static void ResetActivation(GameContext game, int ship)
    {
        game.Flow.Flags.Remove(PendingKey(ship));
        game.Flow.Flags.Remove("activation.cf-mode." + ship);
    }

    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (action.AbilityId is DravenLookAction or NoLookAction)
        {
            // SearchKnowledge records the legal private look from this committed
            // action. The native game neither predicts nor reveals it publicly.
            EngineEffects.PushChoice(game, new(TokenRule, action.Source, action.Target, Value: action.Index), []);
            return true;
        }
        return false;
    }

    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = frame.Rule == RaidRule ? RaidOptions(game, frame).ToArray() : Options(frame).ToArray();
        bool forced = options.Length == 1;
        int actor = forced ? 0 : game.State.Owners[options[0].Template.Source];
        return new(forced ? DecisionKind.Forced : DecisionKind.Player, actor,
            GamePhase.AbilityChoice, options);
    }

    private static IEnumerable<GameActionOption> RaidOptions(GameContext game, GameContinuation frame)
    {
        int index = frame.Indices![0], carrier = checked((int)game.State.Registers[game.Topology.Registers.UpgradeOwner(index)] - 1);
        var command = (ShipCommand)frame.Value; var card = game.State.Upgrades[index];
        if (game.IsShip(carrier) && game.State.Owners[carrier] != game.State.Owners[frame.Target] &&
            !game.IsDestroyed(carrier) && !game.IsReserved(carrier) && !game.IsDestroyed(frame.Target) && !game.IsReserved(frame.Target) &&
            card.Active && card.GetTypedCommandTokenCount(command) > 0)
            yield return new(new(GameActionKind.UseAbility, carrier, frame.Target, Command: command, Index: index, AbilityId: RaidAction),
                "Spend Draven's matching token to assign raid");
        yield return new(new(GameActionKind.Pass, frame.Source, frame.Target, Command: command, Index: index, AbilityId: PassRaidAction),
            "Do not spend Draven's command token");
    }

    private static IEnumerable<GameActionOption> Options(GameContinuation frame)
    {
        if (frame.Stage == 0)
        {
            foreach (var first in Enum.GetValues<ShipCommand>())
                foreach (var second in Enum.GetValues<ShipCommand>())
                    if ((int)second > (int)first)
                        yield return new(new(GameActionKind.ChooseCommand, frame.Source, frame.Target, Command: first,
                            Amount: (int)second + 1, Index: checked((int)frame.Value), AbilityId: PairAction),
                            "Place " + first + " and " + second + " tokens on General Draven");
            yield return new(new(GameActionKind.Pass, frame.Source, frame.Target,
                Index: checked((int)frame.Value), AbilityId: SkipTokensAction), "Do not place command tokens");
            yield break;
        }
        // Legacy recovery for a previously staged first token. New games expose
        // unordered pairs as one native commit, factorized by the policy adapter.
        foreach (var command in Enum.GetValues<ShipCommand>())
            if (frame.Stage == 0 || frame.Indices![0] != (int)command)
                yield return new(new(GameActionKind.ChooseCommand, frame.Source, frame.Target, Command: command,
                    Index: checked((int)frame.Value), AbilityId: TokenAction), "Place " + command + " token on General Draven");
    }

    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (frame.Rule == RaidRule)
        {
            if (!RaidOptions(game, frame).Any(option => GameSession.Matches(option, action))) throw new ArgumentException("Illegal Draven reaction.");
            if (action.AbilityId == RaidAction)
            {
                if (!game.State.Upgrades[action.Index].TrySpendTypedCommandToken(action.Command)) throw new InvalidOperationException("Draven's token is unavailable.");
                game.State.Commands[action.Target].TryGainRaidToken(action.Command);
            }
            game.Flow.Continuations.RemoveAt(game.Flow.Continuations.Count - 1); game.Flow.PendingOptions = [];
            game.Flow.Phase = (GamePhase)frame.Stage; return;
        }
        if (!Options(frame).Any(option => GameSession.Matches(option, action))) throw new ArgumentException("Illegal Draven token choice.");
        if (action.Kind == GameActionKind.Pass) { EngineEffects.ReturnFromChoice(game); return; }
        if (action.AbilityId == PairAction)
        {
            var selected = game.State.Upgrades[checked((int)frame.Value)];
            selected.AddTypedCommandToken(action.Command);
            selected.AddTypedCommandToken((ShipCommand)(action.Amount - 1));
            EngineEffects.ReturnFromChoice(game); return;
        }
        var card = game.State.Upgrades[checked((int)frame.Value)];
        card.AddTypedCommandToken((ShipCommand)frame.Indices![0]);
        card.AddTypedCommandToken(action.Command);
        EngineEffects.ReturnFromChoice(game);
    }
}
