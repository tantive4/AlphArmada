using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Snapshot-owned decisions for round resources, reserve entry and upgrade ready payments.</summary>
public static class RoundAbilityEffects
{
    public static bool SupportsUpgrade(string id) => id is
        "garm-bel-iblis-commander" or "general-tagge-commander" or "admiral-motti-commander" or
        "admiral-trench-commander" or "general-draven-commander" or "grand-admiral-thrawn-commander" or
        "admiral-tarkin-commander" or "count-dooku-commander" or "grand-moff-tarkin-commander" or
        "emperor-palpatine-commander" or "sovereign-title" or "hondo-ohnaka-officer" or
        "early-warning-system" or "reinforced-blast-doors" or "redundant-shields" or "admiral-konstantine-commander" or
        "all-fighters-follow-me" or "entrapment-formation" or "intensify-firepower" or "shields-to-maximum" or "take-evasive-action" or
        "amity-title" or "concord-title" or "endeavor-title" or "tb-73-title" or "unity-title" or
        "executor-title" or "phoenix-home-title" or "relentless-title" or "nova-defiant-title" or
        "captain-needa-officer" or "vanguard-title" or "local-fire-control" or "kyrsta-agate-commander" or
        "taskmaster-grint-officer" or "admiral-titus-officer" or "aspiration-title" or
        "wullf-yularen-officer" or "engineering-team" or "redemption-title" or "expanded-hangar-bay" or "boosted-comms" or "gilded-aegis-title" or
        "admiral-yularen-commander" or "auxiliary-shields-team" or "liberty-title" or "centicore-title" or "triumphant-title";
    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        if (timing is "end-command-phase" or "end-ship-phase")
        {
            foreach (var ship in game.Topology.Ships.Where(x => !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex) && game.HasUpgrade(x.EntityIndex, "gilded-aegis-title")))
                foreach (var zone in game.State.Ships[ship.EntityIndex].HullZones.Where(zone => game.State.Ships[ship.EntityIndex].GetShields(zone) < 6 &&
                    game.State.Ships[ship.EntityIndex].HullZones.Any(donor => donor != zone && game.State.Ships[ship.EntityIndex].GetShields(donor) > 0)))
                    foreach (var token in game.State.Ships[ship.EntityIndex].GetDefenseTokenInstances(DefenseTokenType.Redirect))
                        yield return new(new(GameActionKind.ChooseZone, ship.EntityIndex, ToZone: zone, Token: token.Token, Index: token.Slot,
                            Amount: token.Exhausted ? 1 : 0, AbilityId: "round.gilded.begin"), "Discard Redirect token and move shields into " + zone);
            yield break;
        }
        if (timing != "status") yield break;
        foreach (var upgrade in game.Topology.Upgrades.Where(x => !x.IsSquadronLeader))
        {
            int ownerEntity = (int)game.State.Registers[game.Topology.Registers.UpgradeOwner(upgrade.UpgradeIndex)] - 1;
            if (!game.IsShip(ownerEntity) || game.IsDestroyed(ownerEntity) || game.IsReserved(ownerEntity)) continue;
            var card = game.State.Upgrades[upgrade.UpgradeIndex];
            foreach (var payment in UpgradeReadyRules.GetPayments(card, game.State.Commands[ownerEntity]))
                yield return new(new GameAction(GameActionKind.UseAbility, ownerEntity,
                    Command: payment.Token, Index: upgrade.UpgradeIndex, Amount: payment.Amount,
                    AbilityId: payment.FromCard ? "round.ready-from-card" : "round.ready-from-ship"),
                    "Pay " + payment.Amount + " " + payment.Token + " to ready " + card.Definition.name);
        }
    }

    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (action.AbilityId == "round.gilded.begin")
        {
            if (!game.State.Ships[action.Source].DiscardDefenseToken(new DefenseTokenInstance(action.Token, action.Index, action.Amount != 0))) throw new ArgumentException("Gilded Aegis Redirect token is unavailable.");
            EngineEffects.PushChoice(game, new("round.gilded", action.Source, Stage: (int)action.ToZone), []); return true;
        }
        if (action.AbilityId is not ("round.ready-from-card" or "round.ready-from-ship")) return false;
        if (!UpgradeReadyRules.TryReady(game.State.Upgrades[action.Index], game.State.Commands[action.Source],
            new UpgradeReadyPayment(action.Command, action.AbilityId == "round.ready-from-card", action.Amount)))
            throw new ArgumentException("Upgrade ready payment is no longer available.");
        return true;
    }

    public static GameDecision GetDecision(GameContext game, GameContinuation frame) => frame.Rule == "round.gilded"
        ? new(DecisionKind.Player, game.State.Owners[frame.Source], GamePhase.AbilityChoice, GildedOptions(game, frame).ToArray()) : RoundModule.RoundChoiceDecision(game, frame);
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (frame.Rule != "round.gilded") { RoundModule.ApplyRoundChoice(game, frame, action); return; }
        if (!GildedOptions(game, frame).Any(x => x.Template == action)) throw new ArgumentException("Illegal Gilded Aegis shield transfer.");
        if (action.Kind == GameActionKind.Pass) EngineEffects.ReturnFromChoice(game);
        else if (game.State.Ships[frame.Source].TransferShield(action.FromZone, action.ToZone, maximumOverride: 6) != 1) throw new ArgumentException("Gilded Aegis transfer failed.");
    }
    private static IEnumerable<GameActionOption> GildedOptions(GameContext game, GameContinuation frame)
    {
        var ship = game.State.Ships[frame.Source]; var destination = (HullZone)frame.Stage;
        if (ship.GetShields(destination) < 6 && !(ship.GetShields(destination) == 0 && ship.HasFaceupCritical(CriticalDamageCard.CapacitorFailure)))
            foreach (var source in ship.HullZones.Where(x => x != destination && ship.GetShields(x) > 0))
                yield return new(new(GameActionKind.TransferShield, frame.Source, FromZone: source, ToZone: destination, AbilityId: "round.gilded.move"), "Move one shield from " + source);
        yield return new(new(GameActionKind.Pass, frame.Source, AbilityId: "round.gilded.finish"), "Finish moving shields");
    }
}
