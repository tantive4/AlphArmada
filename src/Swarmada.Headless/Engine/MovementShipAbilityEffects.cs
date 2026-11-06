using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public static partial class MovementAbilityEffects
{
    private static IEnumerable<GameActionOption> GetAdditionalShipActions(GameContext game, string timing, int source)
    {
        if (!game.IsShip(source) || game.IsDestroyed(source) || game.IsReserved(source)) yield break;
        if (timing == "after-maneuver" && game.Flow.ShipActivations[source].DeferredSquadronActivations > 0)
            yield return Option(source, "remaining.after-maneuver.flight-commander", "Resolve held Squadron command after maneuver");
        if (timing == "end-attack-step")
        {
            if (Card(game, source, "ordnance-pods") is { Active: true, Exhausted: false } && RemainingUpgradeRules.SizeRank(game.ShipViews[source]) >= 2
                && ExtraFlakTargets(game, source, TabletopMeasurements.CloseRange).Any())
                yield return Option(source, "remaining.end-activation.ordnance-pods", "Ordnance Pods: one-black-die anti-squadron attack from one arc");
            if (Card(game, source, "impetuous-title") is { Active: true, UsesThisActivation: 0 }
                && game.Topology.Ships[source].CatalogEntry.Definition.AntiSquadronArmament.TotalDice > 0 && ExtraFlakTargets(game, source, TabletopMeasurements.LongRange).Any())
                yield return Option(source, "remaining.end-activation.impetuous", "Impetuous: additional attack against one squadron");
        }
        if (timing == "after-maneuver-final" && Card(game, source, "slicer-tools") is { Active: true, Exhausted: false } && SlicerTargets(game, source).Any())
            yield return Option(source, "slicer.use", "Exhaust Slicer Tools and interfere with an enemy command stack");
        if (timing == "end-activation" && game.State.Ships[source].CurrentSpeed > 0 && Card(game, source, "magnite-crystal-tractor-beam-array") is { Active: true, Exhausted: false })
            foreach (var target in game.Topology.Ships)
                if (target.Owner != game.State.Owners[source] && !game.IsDestroyed(target.EntityIndex) && !game.IsReserved(target.EntityIndex)
                    && Geometry2D.Distance(game.ShipBase(source), game.ShipBase(target.EntityIndex)) <= TabletopMeasurements.LongRange)
                    yield return new(new(GameActionKind.UseAbility, source, target.EntityIndex, AbilityId: "remaining.end-activation.magnite." + target.RuntimeId), "Exhaust Magnite and set enemy speed to this ship's speed");
        if (timing == "after-maneuver" && game.HasUpgrade(source, "demolisher-title")
            && game.Flow.Flags.GetValueOrDefault($"movement.maneuvers.{game.State.Round}.{source}") == 1
            && !game.Flow.Flags.ContainsKey($"combat.demolisher.used.{game.State.Round}.{source}")
            && game.Flow.ShipActivations[source].AttacksResolved < 2 && game.State.Ships[source].HullZones.Any(zone => (game.Flow.ShipActivations[source].UsedHullZones & (1 << (int)zone)) == 0))
            yield return Option(source, "remaining.after-maneuver.demolisher", "Demolisher: perform one remaining attack after first maneuver");
    }
    private static bool TryApplyAdditionalShipEffect(GameContext game, GameAction action)
    {
        if (action.AbilityId is "remaining.end-activation.ordnance-pods" or "remaining.end-activation.impetuous")
        {
            bool pods = action.AbilityId.EndsWith("ordnance-pods", StringComparison.Ordinal); string id = pods ? "ordnance-pods" : "impetuous-title";
            var card = Card(game, action.Source, id) ?? throw new ArgumentException("Extra attack card unavailable.");
            int upgradeIndex = Array.FindIndex(game.State.Upgrades, c => ReferenceEquals(c, card));
            EngineEffects.PushChoice(game, new("movement-choice.extra-flak", action.Source, game.Flow.ActiveSquadron, Text: id), []);
            game.Flow.ActiveSquadron = -1;
            CombatModule.QueueGrantedAttack(game, action.Source, returnPhase: GamePhase.AbilityChoice, flakOnly: true,
                forcedBlackDice: pods ? 1 : null, maximumRange: pods ? RangeBand.Close : null);
            game.Flow.Flags["combat.granted.upgrade-index"] = upgradeIndex;
            if (pods) game.Flow.Flags["combat.granted.exhaust"] = 1; else game.Flow.Flags["combat.granted.single-target"] = 1;
            return true;
        }
        if (action.AbilityId == "remaining.after-maneuver.flight-commander")
        {
            if (!ActivationAbilityEffects.ResumeDeferredSquadrons(game, GamePhase.AbilityChoice)) throw new ArgumentException("No held Squadron command.");
            return true;
        }
        if (action.AbilityId == "slicer.use")
        {
            if (Card(game, action.Source, "slicer-tools")?.Exhaust() != true) throw new ArgumentException("Slicer Tools unavailable.");
            EngineEffects.PushChoice(game, new("movement-choice.slicer", action.Source), []); return true;
        }
        if (action.AbilityId.StartsWith("remaining.end-activation.magnite.", StringComparison.Ordinal))
        {
            if (Card(game, action.Source, "magnite-crystal-tractor-beam-array")?.Exhaust() != true) throw new ArgumentException("Magnite Array unavailable.");
            game.State.Ships[action.Target].SetSpeed(game.State.Ships[action.Source].CurrentSpeed, MaximumSpeed(game, action.Target));
            RoundModule.ObserveObjectives(game); return true;
        }
        if (action.AbilityId == "remaining.after-maneuver.demolisher")
        {
            string suffix = $"{game.State.Round}.{action.Source}";
            game.Flow.Flags["combat.demolisher.used." + suffix] = 1;
            game.Flow.Flags["combat.demolisher.after-maneuver." + suffix] = 1;
            game.Flow.Flags["combat.demolisher.start-count." + suffix] = game.Flow.ShipActivations[action.Source].AttacksResolved;
            EngineEffects.PushChoice(game, new("movement-choice.demolisher", action.Source), []); game.Flow.Phase = GamePhase.SelectAttack; return true;
        }
        return false;
    }
    private static GameDecision? GetAdditionalShipDecision(GameContext game, GameContinuation frame)
    {
        if (frame.Rule == "movement-choice.extra-flak") return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), "Return from additional anti-squadron attack")]);
        if (frame.Rule == "movement-choice.demolisher") return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), "Resume after-maneuver choices")]);
        if (frame.Rule != "movement-choice.slicer") return null;
        var options = new List<GameActionOption>();
        if (frame.Target < 0)
        {
            foreach (int target in SlicerTargets(game, frame.Source)) options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, target), "Choose enemy ship command stack"));
        }
        else
        {
            bool hasTop = game.State.Commands[frame.Target].TryPeekTop(out var top);
            foreach (var command in Enum.GetValues<ShipCommand>())
            {
                if (hasTop) options.Add(new(new(GameActionKind.ChooseCommand, frame.Source, frame.Target, Command: command, AbilityId: "slicer.command"), "Replace top dial with " + command));
                if ((!hasTop || command != top) && game.State.Commands[frame.Target].CanGainRaidToken(command))
                    options.Add(new(new(GameActionKind.ChooseCommand, frame.Source, frame.Target, Command: command, AbilityId: "slicer.raid"), "Assign " + command + " raid token"));
            }
        }
        if (options.Count == 0) options.Add(new(new(GameActionKind.Pass), "No legal Slicer target or command remains"));
        return new(options[0].Template.Kind == GameActionKind.Pass ? DecisionKind.Forced : DecisionKind.Player,
            options[0].Template.Kind == GameActionKind.Pass ? 0 : game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
    }
    private static bool ApplyAdditionalShipChoice(GameContext game, GameContinuation frame, GameAction action)
    {
        if (frame.Rule == "movement-choice.extra-flak")
        {
            game.Flow.ActiveSquadron = frame.Target; game.Flow.Flags.Remove("combat.granted.completed");
            EngineEffects.ReturnFromChoice(game); return true;
        }
        if (frame.Rule == "movement-choice.demolisher")
        {
            game.Flow.Flags.Remove($"combat.demolisher.after-maneuver.{game.State.Round}.{frame.Source}");
            game.Flow.Flags.Remove($"combat.demolisher.start-count.{game.State.Round}.{frame.Source}");
            EngineEffects.ReturnFromChoice(game); return true;
        }
        if (frame.Rule != "movement-choice.slicer") return false;
        if (action.Kind == GameActionKind.ChooseEntity) { game.Flow.Continuations[^1] = frame with { Target = action.Target }; return true; }
        if (action.AbilityId == "slicer.command" && !game.State.Commands[frame.Target].ReplaceTop(action.Command)) throw new ArgumentException("Target has no top command dial.");
        if (action.AbilityId == "slicer.raid" && !game.State.Commands[frame.Target].TryGainRaidToken(action.Command)) throw new ArgumentException("Target cannot gain raid token.");
        EngineEffects.ReturnFromChoice(game); return true;
    }
    private static IEnumerable<int> SlicerTargets(GameContext game, int source) => game.Topology.Ships
        .Where(s => s.Owner != game.State.Owners[source] && !game.IsDestroyed(s.EntityIndex) && !game.IsReserved(s.EntityIndex)
            && Geometry2D.Distance(game.ShipBase(source), game.ShipBase(s.EntityIndex)) <= TabletopMeasurements.MediumRange).Select(s => s.EntityIndex);
    private static IEnumerable<int> ExtraFlakTargets(GameContext game, int source, float range) => game.Topology.Squadrons
        .Where(s => s.Owner != game.State.Owners[source] && !game.IsDestroyed(s.EntityIndex) && !game.IsReserved(s.EntityIndex)
            && Geometry2D.Distance(game.SquadronBase(s.EntityIndex), game.ShipBase(source)) <= range).Select(s => s.EntityIndex);
}
