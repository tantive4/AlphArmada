using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public static class RepairAbilityEffects
{
    public static bool SupportsUpgrade(string id) => id is "projection-experts" or "repair-crews" or "wat-tambor-officer" or "battle-droid-reserves" or "mon-calamari-exodus-fleet-title";
    public static bool BeginEngineering(GameContext game, int ship) => EngineEffects.BeginWindow(game, "engineering-start", ship, resumePhase: GamePhase.Repair);
    public static IEnumerable<GameActionOption> GetWindowActions(GameContext game, string timing, int source, int target)
    {
        if (timing != "engineering-start" || !game.IsShip(source) || !game.HasUpgrade(source, "mon-calamari-exodus-fleet-title")) yield break;
        foreach (var friend in game.SceneShipViews.Where(friend => friend.Index != source && friend.OwnerId == game.State.Owners[source]
            && RaidRules.FootprintDistance(game.ShipViews[source], friend) <= 10.7221f
            && friend.Loadout.Find("mon-calamari-exodus-fleet-title") is { Active: true, Exhausted: false }))
            yield return new(new(GameActionKind.UseAbility, source, friend.Index, AbilityId: "repair.exodus"), "Exhaust " + friend.Definition.DisplayName + " Exodus Fleet for +2 engineering");
    }
    public static bool TryWindowApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (action.AbilityId != "repair.exodus") return false;
        if (game.ShipViews[action.Target].Loadout.Find("mon-calamari-exodus-fleet-title")?.Exhaust() != true) throw new InvalidOperationException("Exodus Fleet is unavailable.");
        game.Flow.ShipActivations[action.Source].EngineeringPoints += 2; return true;
    }
    public static IEnumerable<GameActionOption> GetActions(GameContext game, int ship)
    {
        using var scope = new GameScope(game);
        var activation = game.Flow.ShipActivations[ship]; var source = game.ShipViews[ship];
        var friends = game.SceneShipViews.Where(friend => friend.OwnerId == source.OwnerId).ToArray();
        if (game.HasUpgrade(ship, "projection-experts") && activation.EngineeringPoints > 0 && game.Flow.Flags.GetValueOrDefault(Key(game, ship, "projection")) < 2)
            foreach (var friend in friends.Where(friend => friend.Index != ship && RaidRules.FootprintDistance(source, friend) <= 13.260394f))
                foreach (var fromZone in source.State.HullZones.Where(zone => source.State.GetShields(zone) > 0))
                    foreach (var toZone in friend.State.HullZones.Where(zone => friend.State.GetShields(zone) < UnitAbilityEngine.GetEffectiveShieldMaximum(friend, zone)
                        && !(friend.State.GetShields(zone) == 0 && friend.State.HasFaceupCritical(CriticalDamageCard.CapacitorFailure))))
                        yield return new(new(GameActionKind.TransferShield, ship, friend.Index, FromZone: fromZone, ToZone: toZone, Amount: 1,
                            AbilityId: "repair.projection"), "Project " + fromZone + " shield to " + friend.Definition.DisplayName + " " + toZone);
        if (game.HasUpgrade(ship, "repair-crews"))
            foreach (var friend in friends.Where(friend => friend.Index != ship && RaidRules.FootprintDistance(source, friend) <= 5.4704595f))
            {
                if (friend.State.FacedownDamageCardCount > 0)
                    yield return new(new(GameActionKind.RepairHull, ship, friend.Index, Index: -1, AbilityId: "repair.crews"), "Spend Repair command to discard friendly facedown damage");
                foreach (var critical in friend.State.GetFaceupCriticalCards().Distinct())
                    yield return new(new(GameActionKind.RepairHull, ship, friend.Index, Index: (int)critical, AbilityId: "repair.crews"), "Spend Repair command to discard friendly " + critical);
            }
        if (game.HasUpgrade(ship, "wat-tambor-officer") && game.Flow.Flags.GetValueOrDefault(Key(game, ship, "wat-uses")) < 2)
        {
            long mode = game.Flow.Flags.GetValueOrDefault(Key(game, ship, "wat-mode"));
            foreach (var friend in friends.Where(friend => friend.Index == ship || RaidRules.FootprintDistance(source, friend) <= 13.260394f))
            {
                if (mode == 1 && friend.Index != ship || mode == 2 && game.Flow.Flags.GetValueOrDefault(Key(game, ship, "wat-external")) != friend.Index + 1) continue;
                foreach (var zone in friend.State.HullZones.Where(zone => friend.State.GetShields(zone) > 0))
                {
                    if (mode == 2 && game.Flow.Flags.GetValueOrDefault(Key(game, ship, "wat-zone")) != (int)zone) continue;
                    yield return new(new(GameActionKind.UseAbility, ship, friend.Index, FromZone: zone, AbilityId: "repair.wat"), "Spend " + friend.Definition.DisplayName + " " + zone + " shield for +2 engineering");
                }
            }
        }
        if (source.Loadout.Find("battle-droid-reserves") is { Active: true, Exhausted: false })
            yield return new(new(GameActionKind.UseAbility, ship, AbilityId: "repair.droids"), "Select Crew cards for Battle Droid Reserves");
    }
    public static bool TryApply(GameContext game, GameAction action)
    {
        if (!action.AbilityId.StartsWith("repair.", StringComparison.Ordinal) || action.AbilityId == "repair.yularen") return false;
        if (!GetActions(game, action.Source).Any(option => GameSession.Matches(option, action))) throw new ArgumentException("Illegal Repair upgrade action.");
        var activation = game.Flow.ShipActivations[action.Source];
        switch (action.AbilityId)
        {
            case "repair.projection":
                if (game.State.Ships[action.Target].RestoreShield(action.ToZone, 1, game.HasUpgrade(action.Target, "auxiliary-shields-team")) != 1
                    || game.State.Ships[action.Source].LoseShields(action.FromZone, 1) != 1) throw new InvalidOperationException("Projection shield transfer failed.");
                activation.EngineeringPoints--; game.Flow.Flags[Key(game, action.Source, "projection")] = game.Flow.Flags.GetValueOrDefault(Key(game, action.Source, "projection")) + 1; return true;
            case "repair.crews":
                var target = game.State.Ships[action.Target]; CriticalDamageCard removed;
                if (action.Index < 0) { if (!target.TryRepairFacedownDamage(out removed)) throw new InvalidOperationException("No facedown card to repair."); }
                else { removed = (CriticalDamageCard)action.Index; if (target.RepairFaceupDamage(removed) != 1) throw new InvalidOperationException("No selected faceup card to repair."); }
                game.State.Decks[game.State.Owners[action.Target] - 1].Discard(removed); activation.EngineeringPoints = 0;
                game.Flow.Phase = game.Flow.CommandReturnPhase; game.Flow.CommandReturnPhase = GamePhase.SelectAttack; activation.CommandResolutionCompleted = true; return true;
            case "repair.wat":
                if (game.State.Ships[action.Target].LoseShields(action.FromZone, 1) != 1) throw new InvalidOperationException("No Wat Tambor shield available.");
                if (game.Flow.Flags.GetValueOrDefault(Key(game, action.Source, "wat-mode")) == 0)
                {
                    bool external = action.Target != action.Source;
                    game.Flow.Flags[Key(game, action.Source, "wat-mode")] = external ? 2 : 1;
                    if (external) { game.Flow.Flags[Key(game, action.Source, "wat-external")] = action.Target + 1; game.Flow.Flags[Key(game, action.Source, "wat-zone")] = (int)action.FromZone; }
                }
                game.Flow.Flags[Key(game, action.Source, "wat-uses")] = game.Flow.Flags.GetValueOrDefault(Key(game, action.Source, "wat-uses")) + 1;
                activation.EngineeringPoints += 2; return true;
            case "repair.droids":
                var frame = new GameContinuation("general.repair.droids", action.Source, Indices: []);
                EngineEffects.PushChoice(game, frame, DroidOptions(game, frame)); return true;
            default: return false;
        }
    }
    public static bool Handles(string rule) => rule == "general.repair.droids";
    public static GameDecision GetDecision(GameContext game, GameContinuation frame) => new(DecisionKind.Player,
        game.State.Owners[frame.Source], GamePhase.AbilityChoice, DroidOptions(game, frame).ToArray());
    private static IEnumerable<GameActionOption> DroidOptions(GameContext game, GameContinuation frame)
    {
        var selected = frame.Indices ?? [];
        var cards = game.State.Ships[frame.Source].DamageCards;
        for (int index = 0; index < cards.Count; index++)
        {
            var card = cards[index];
            if (card.FaceUp && card.CriticalCard.HasValue && ShipRuntimeState.IsCrewDamageCard(card.CriticalCard.Value))
                yield return new(new(GameActionKind.ChooseCritical, frame.Source, Index: index, AbilityId: selected.Contains(index) ? "repair.droids.deselect" : "repair.droids.select"),
                    (selected.Contains(index) ? "Deselect " : "Select ") + card.CriticalCard.Value);
        }
        yield return new(new(GameActionKind.UseAbility, frame.Source, AbilityId: "repair.droids.apply"), "Exhaust and flip selected Crew cards facedown");
        yield return new(new(GameActionKind.Pass, frame.Source, AbilityId: "repair.droids.cancel"), "Cancel without exhausting");
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!DroidOptions(game, frame).Any(option => GameSession.Matches(option, action))) throw new ArgumentException("Illegal Battle Droid Reserves choice.");
        if (action.AbilityId is "repair.droids.select" or "repair.droids.deselect")
        {
            game.Flow.Continuations[^1] = frame with { Indices = action.AbilityId == "repair.droids.select" ? (frame.Indices ?? []).Append(action.Index).ToArray()
                : (frame.Indices ?? []).Where(index => index != action.Index).ToArray() }; return;
        }
        if (action.AbilityId == "repair.droids.apply")
        {
            if (game.ShipViews[frame.Source].Loadout.Find("battle-droid-reserves")?.Exhaust() != true) throw new InvalidOperationException("Battle Droid Reserves unavailable.");
            foreach (int index in frame.Indices ?? [])
                if (game.State.Ships[frame.Source].DamageCards[index].CriticalCard is { } critical) game.State.Ships[frame.Source].FlipCriticalFacedown(critical);
        }
        game.Flow.Continuations.RemoveAt(game.Flow.Continuations.Count - 1); game.Flow.PendingOptions = []; game.Flow.Phase = GamePhase.Repair;
    }
    private static string Key(GameContext game, int ship, string field) => $"repair.{field}.{game.State.Round}.{ship}";
}
