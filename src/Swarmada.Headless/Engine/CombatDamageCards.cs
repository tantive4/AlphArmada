using Armada.Core;
using Armada.Rules;

namespace Swarmada.Headless.Engine;

public sealed partial class CombatModule
{
    private static IEnumerable<GameActionOption> IncomingCardOptions(GameContext game, AttackState a)
    {
        bool crew = ShipRuntimeState.IsCrewDamageCard(a.IncomingDamageCard!.Value);
        if (crew && game.HasUpgrade(a.Defender, "medical-team"))
            yield return new(new(GameActionKind.UseAbility, a.Defender, AbilityId: "faceup-prevention.medical"), "Discard Medical Team and this incoming Crew card");
        if (crew && game.HasUpgrade(a.Defender, "lira-wessex-officer") && game.State.Commands[a.Defender].HasToken(ShipCommand.Engineering))
            yield return new(new(GameActionKind.UseAbility, a.Defender, AbilityId: "faceup-prevention.lira"), "Spend Engineering token; deal Crew card facedown");
        if (!crew && game.UpgradesFor(a.Defender).Any(card => card.Definition.id == "tide-of-progress-xii-title" && card.Active && !card.Exhausted))
            yield return new(new(GameActionKind.UseAbility, a.Defender, AbilityId: "faceup-prevention.tide"), "Exhaust Tide of Progress XII; discard incoming Ship card");
        yield return new(new(GameActionKind.UseAbility, a.Defender, AbilityId: "faceup-prevention.skip"), "Accept incoming faceup " + a.IncomingDamageCard.Value);
    }
    private void ApplyIncomingCardChoice(GameContext game, AttackState a, GameAction action)
    {
        var incoming = a.IncomingDamageCard!.Value; a.IncomingDamageCard = null;
        if (action.AbilityId == "faceup-prevention.lira")
        {
            if (!game.State.Commands[a.Defender].TrySpendToken(ShipCommand.Engineering)) throw new InvalidOperationException("Lira Wessex requires an Engineering token.");
            CommitDamageCard(game, a, incoming, false); return;
        }
        if (action.AbilityId == "faceup-prevention.skip") { CommitDamageCard(game, a, incoming, true); return; }
        string id = action.AbilityId == "faceup-prevention.medical" ? "medical-team" : "tide-of-progress-xii-title";
        var card = game.UpgradesFor(a.Defender).Single(card => card.Definition.id == id && card.Active);
        if (!(id == "medical-team" ? card.Discard() : card.Exhaust())) throw new InvalidOperationException("Damage prevention card is unavailable.");
        Deck(game, a.Defender).Discard(incoming); ContinueHullDamage(game, a);
    }
}
