using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>The five source boarding-team interactions, including selection of exact
/// physical tokens, known facedown cards, enemy upgrades and individual command dials.</summary>
public static class BoardingAbilityEffects
{
    private static readonly HashSet<string> Cards = new(StringComparer.Ordinal)
    { "boarding-engineers-boarding-team", "boarding-troopers-boarding-team", "cham-syndulla-boarding-team", "darth-vader-boarding-team", "shriv-surgaav-boarding-team" };
    public static bool SupportsUpgrade(string id) => Cards.Contains(id);
    public static bool Handles(string rule) => rule == "general.boarding";
    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        const string prefix = "remaining.command-reveal.boarding.";
        if (!action.AbilityId.StartsWith(prefix, StringComparison.Ordinal)) return false;
        string body = action.AbilityId[prefix.Length..]; int split = body.IndexOf('.');
        if (split <= 0 || !Cards.Contains(body[..split]) || !game.Topology.EntityByRuntimeId.TryGetValue(body[(split + 1)..], out int target))
            throw new ArgumentException("Unknown boarding team or target.");
        string cardId = body[..split]; var source = game.ShipViews[action.Source]; var defender = game.ShipViews[target];
        var activation = game.Flow.ShipActivations[action.Source]; bool spendDial = activation.RevealedCommand == ShipCommand.Squadron && activation.RevealedDialAvailable;
        if (defender.OwnerId == source.OwnerId || defender.IsDestroyed || game.IsReserved(target)
            || RaidRules.FootprintDistance(source, defender) > 5.426696f || source.Loadout.Find(cardId)?.Active != true
            || (!spendDial && !source.Commands.HasToken(ShipCommand.Squadron))) throw new ArgumentException("Illegal boarding target.");
        if (!spendDial) GeneralAbilityEffects.SpendCommandToken(game, action.Source, ShipCommand.Squadron);
        source.Loadout.Find(cardId)!.Discard();
        if (spendDial)
        {
            activation.RevealedDialAvailable = false;
            int index = game.Flow.Continuations.FindLastIndex(x => x == window);
            game.Flow.Continuations[index] = window with { Stage = (int)GamePhase.SelectAttack };
        }
        int maximum = cardId switch
        {
            "cham-syndulla-boarding-team" => defender.Commands.Count,
            "boarding-engineers-boarding-team" => Math.Max(0, source.Definition.Engineering),
            "boarding-troopers-boarding-team" => Math.Max(0, CommandUpgradeRules.SquadronValue(source)), _ => 1
        };
        var frame = new GameContinuation("general.boarding", action.Source, target, Value: maximum, Text: cardId);
        EngineEffects.PushChoice(game, frame, Options(game, frame)); return true;
    }
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = Options(game, frame).ToArray();
        bool forced = options.Length == 1 && options[0].Template.Kind == GameActionKind.Pass;
        return new(forced ? DecisionKind.Forced : DecisionKind.Player, forced ? 0 : game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
    }
    private static IEnumerable<GameActionOption> Options(GameContext game, GameContinuation frame)
    {
        var target = game.ShipViews[frame.Target];
        if (frame.Value <= 0 || target.IsDestroyed) { yield return Finish(frame); yield break; }
        switch (frame.Text)
        {
            case "boarding-engineers-boarding-team":
                for (int index = 0; index < target.State.DamageCards.Count; index++)
                {
                    var card = target.State.DamageCards[index];
                    if (!card.FaceUp && card.CriticalCard.HasValue)
                        yield return new(new(GameActionKind.ChooseCritical, frame.Source, frame.Target, Index: index, AbilityId: "boarding.flip"), "Flip " + card.CriticalCard.Value);
                }
                yield return Finish(frame); yield break;
            case "boarding-troopers-boarding-team":
                foreach (var token in ForcedDefenseTokenDiscardRules.GetAvailableInstances(target.State))
                    if (!token.Exhausted || target.State.CanSpendExhaustedDefenseTokens)
                        yield return new(new(GameActionKind.ChooseToken, frame.Source, frame.Target, Token: token.Token, Index: token.Slot,
                            Amount: token.Exhausted ? 1 : 0, AbilityId: "boarding.spend"), "Spend " + token.Token + " token " + token.Slot);
                yield return Finish(frame); yield break;
            case "cham-syndulla-boarding-team":
                foreach (var command in Enum.GetValues<ShipCommand>())
                    yield return new(new(GameActionKind.ChooseCommand, frame.Source, frame.Target, Command: command, Index: frame.Stage, AbilityId: "boarding.command"),
                        "Set dial " + (frame.Stage + 1) + " to " + command);
                yield break;
            default:
                bool any = false;
                foreach (var card in target.Loadout.Upgrades.Where(card => card.Active && EligibleDiscard(frame.Text, card)))
                {
                    any = true;
                    yield return new(new(GameActionKind.UseAbility, frame.Source, frame.Target, Index: Array.IndexOf(game.State.Upgrades, card), AbilityId: "boarding.discard"), "Discard " + card.Definition.name);
                }
                if (!any) yield return Finish(frame);
                yield break;
        }
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!Options(game, frame).Any(option => GameSession.Matches(option, action))) throw new ArgumentException("Illegal boarding choice.");
        if (action.Kind == GameActionKind.Pass) { EngineEffects.ReturnFromChoice(game); RoundModule.CheckTerminal(game); return; }
        var target = game.ShipViews[frame.Target];
        var next = frame with { Stage = frame.Stage + 1, Value = frame.Value - 1 };
        game.Flow.Continuations[^1] = next;
        if (action.AbilityId == "boarding.flip")
        {
            if (!target.State.FlipFacedownDamageCardFaceupAt(action.Index, out var card)) throw new InvalidOperationException("The chosen facedown card is absent.");
            CombatModule.ResolveNewFaceupCard(game, frame.Target, card, GamePhase.AbilityChoice); return;
        }
        if (action.AbilityId == "boarding.spend")
        {
            if (!target.State.TryResolveDefenseToken(new DefenseTokenInstance(action.Token, action.Index, action.Amount != 0), out _)) throw new InvalidOperationException("Cannot spend selected token.");
        }
        else if (action.AbilityId == "boarding.command") target.Commands.ReplaceAt(action.Index, action.Command);
        else if (action.AbilityId == "boarding.discard")
            UnitAbilityEngine.DiscardUpgradeFromShip(target, game.State.Upgrades[action.Index], out _);
        if (next.Value <= 0) EngineEffects.ReturnFromChoice(game);
        RoundModule.ObserveObjectives(game); RoundModule.CheckTerminal(game);
    }
    private static bool EligibleDiscard(string id, UpgradeCardRuntimeState card) => id == "darth-vader-boarding-team" ? card.Definition.slot != "commander" :
        card.Definition.slot is "weapons-team" or "support-team" or "offensive-retro" or "defensive-retro" or "ordnance" or "ion-cannon" or "turbolaser";
    private static GameActionOption Finish(GameContinuation frame) => new(new(GameActionKind.Pass, frame.Source, frame.Target, AbilityId: "boarding.finish"), "Finish boarding");
}
