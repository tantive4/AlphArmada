using Armada.Rules;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public static class RaidAbilityEffects
{
    public static bool SupportsUpgrade(string id) => id is "b2-rocket-troopers" or "jyn-erso-boarding-team" or "iden-versio-officer" or "asajj-ventress-officer";
    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        bool iden = action.AbilityId == UnitAbilityEngine.IdenVersioRaidAction;
        bool dial = action.AbilityId == UnitAbilityEngine.JynErsoRaidDialAction;
        bool token = action.AbilityId == UnitAbilityEngine.JynErsoRaidTokenAction;
        if (!iden && !dial && !token) return false;
        int source = action.Source; var ship = game.ShipViews[source]; var activation = game.Flow.ShipActivations[source];
        string id = iden ? "iden-versio-officer" : "jyn-erso-boarding-team";
        if (ship.Loadout.Find(id)?.Active != true || (iden || dial) && activation.RevealedCommand != ShipCommand.Squadron
            || token && !ship.Commands.HasToken(ShipCommand.Squadron)) throw new ArgumentException("Squadron boarding raid cannot be paid.");
        if (token) GeneralAbilityEffects.SpendCommandToken(game, source, ShipCommand.Squadron);
        if (!ship.Loadout.Find(id)!.Discard()) throw new InvalidOperationException("Raid boarding card unavailable.");
        if (dial) { activation.RevealedDialAvailable = false; activation.NativeDialConsumedBeforePrelude = true; }
        PushRaid(game, source, -1, iden ? 1 : 2, id, GamePhase.AbilityChoice, requireUnraided: !iden); return true;
    }
    public static bool TryCombatApply(GameContext game, AttackState attack, CombatWindow window, GameAction action)
    {
        if (action.AbilityId == UnitAbilityEngine.B2RocketTroopersRaidAction)
        {
            if (window != CombatWindow.BeforeRoll || !game.IsShip(attack.Attacker) || !game.IsShip(attack.Defender)
                || attack.Range is not (RangeBand.Close or RangeBand.Medium) || game.ShipViews[attack.Attacker].Loadout.Find("b2-rocket-troopers")?.Exhaust() != true)
                throw new ArgumentException("B2 Rocket Troopers cannot be used for this attack.");
            PushRaid(game, attack.Attacker, attack.Defender, 1, "b2-rocket-troopers", game.Flow.Phase, requireUnraided: false); return true;
        }
        if (action.AbilityId.StartsWith("raid.asajj.", StringComparison.Ordinal))
        {
            if (!GetCombatActions(game, attack, window).Any(option => GameSession.Matches(option, action))) throw new ArgumentException("Asajj cannot steal these tokens.");
            var defender = game.State.Commands[attack.Defender];
            if (!defender.DiscardRaidToken(action.Command) || !defender.TrySpendToken(action.Command) || !TokenAssignments.TryAssign(game, attack.Attacker, action.Command))
                throw new InvalidOperationException("Asajj token transfer failed.");
            attack.EffectFlags["raid.asajj.used"] = 1; return true;
        }
        return false;
    }
    public static IEnumerable<GameActionOption> GetCombatActions(GameContext game, AttackState attack, CombatWindow window)
    {
        if (window != CombatWindow.AfterAttack || !game.IsShip(attack.Attacker) || !game.IsShip(attack.Defender)
            || game.IsDestroyed(attack.Defender) || !game.HasUpgrade(attack.Attacker, "asajj-ventress-officer")
            || attack.EffectFlags.ContainsKey("raid.asajj.used") || attack.AppliedDamage <= 0) yield break;
        foreach (var command in Enum.GetValues<ShipCommand>())
            if (game.State.Commands[attack.Defender].RaidTokenCountOf(command) > 0 && game.State.Commands[attack.Defender].HasToken(command)
                && game.State.Ships[attack.Attacker].CanHaveCommandTokens && TokenAssignments.CanAssign(game, attack.Attacker, command))
                yield return new(new(GameActionKind.UseAbility, attack.Attacker, attack.Defender, Command: command,
                    AbilityId: "raid.asajj." + (int)command), "Steal matching " + command + " raid and command token");
    }
    public static bool Handles(string rule) => rule == "general.raid";
    private static void PushRaid(GameContext game, int source, int target, int quantity, string card, GamePhase resume, bool requireUnraided)
    {
        var frame = new GameContinuation("general.raid", source, target, (int)resume, quantity, card, [requireUnraided ? 1 : 0]);
        EngineEffects.PushChoice(game, frame, Options(game, frame));
    }
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = Options(game, frame).ToArray(); bool forced = options.Length == 1 && options[0].Template.Kind == GameActionKind.Pass;
        return new(forced ? DecisionKind.Forced : DecisionKind.Player, forced ? 0 : game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
    }
    private static IEnumerable<GameActionOption> Options(GameContext game, GameContinuation frame)
    {
        bool any = false;
        if (frame.Value > 0)
        {
            if (frame.Target < 0)
                foreach (var target in game.SceneShipViews.Where(target => target.OwnerId != game.State.Owners[frame.Source]
                    && RaidRules.FootprintDistance(game.ShipViews[frame.Source], target) <= 5.426696f
                    && (frame.Indices![0] == 0 || target.Commands.RaidTokenCount == 0)
                    && Enum.GetValues<ShipCommand>().Any(target.Commands.CanGainRaidToken)))
                {
                    any = true; yield return new(new(GameActionKind.ChooseEntity, frame.Source, target.Index, AbilityId: "raid.target"), "Raid " + target.Definition.DisplayName);
                }
            else if (!game.IsDestroyed(frame.Target))
                foreach (var command in Enum.GetValues<ShipCommand>().Where(game.State.Commands[frame.Target].CanGainRaidToken))
                {
                    any = true; yield return new(new(GameActionKind.ChooseCommand, frame.Source, frame.Target, Command: command, AbilityId: "raid.command"), "Place " + command + " raid token");
                }
        }
        if (!any) yield return new(new(GameActionKind.Pass, frame.Source, AbilityId: "raid.finish"), "Finish paid raid effect");
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!Options(game, frame).Any(option => GameSession.Matches(option, action))) throw new ArgumentException("Illegal paid raid choice.");
        if (action.Kind == GameActionKind.ChooseEntity) { game.Flow.Continuations[^1] = frame with { Target = action.Target }; return; }
        if (action.Kind == GameActionKind.ChooseCommand)
        {
            if (!game.State.Commands[frame.Target].TryGainRaidToken(action.Command)) throw new InvalidOperationException("Chosen raid type is unavailable.");
            if (frame.Value > 1) { game.Flow.Continuations[^1] = frame with { Value = frame.Value - 1 }; return; }
        }
        game.Flow.Continuations.RemoveAt(game.Flow.Continuations.Count - 1); game.Flow.PendingOptions = []; game.Flow.Phase = (GamePhase)frame.Stage;
    }
}
