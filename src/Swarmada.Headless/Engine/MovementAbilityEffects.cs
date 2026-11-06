using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Armada.Session;
using Swarmada.Headless.Effects;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Controller-owned maneuver and squadron timing effects. All temporary choices are
/// serialized continuations; handlers do not retain game-specific state.</summary>
public static partial class MovementAbilityEffects
{
    private static readonly HashSet<string> Supported = new(StringComparer.Ordinal)
    {
        "hardened-bulkheads", "jainas-light-title", "eclipse-title", "garels-honor-title", "admiral-montferrat-officer",
        "nav-team", "harrow-title", "rakehell-title", "general-madine-commander", "admiral-ozzel-commander",
        "engine-techs", "quantum-storm-title", "fighter-coordination-team", "squall-title", "wedge-antilles-officer",
        "independence-title", "yavaris-title", "corrupter-title", "vector-title", "admiral-chiraneau-officer",
        "plo-koon-commander", "g-8-experimental-projector", "clone-captain-silver-officer", "swift-return-title",
        "moff-jerjerrod-commander", "chart-officer", "ahsoka-tano-officer-republic", "nevoota-bee-title", "slicer-tools", "magnite-crystal-tractor-beam-array", "demolisher-title",
        "hyperwave-signal-boost", "instigator-title", "ordnance-pods", "impetuous-title", "advanced-transponder-net"
    };
    public static bool SupportsUpgrade(string id) => Supported.Contains(id);
    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        using var scope = new GameScope(game);
        foreach (var option in GetSquadronEffectActions(game, timing, source, target)) yield return option;
        foreach (var option in GetAdditionalShipActions(game, timing, source)) yield return option;
        if (game.IsShip(source) && !game.IsDestroyed(source))
        {
            if (timing == "determine-course" && !game.Flow.Flags.ContainsKey(RemainingCourseKey(game, source)))
            {
                if (RemainingUpgradeRules.FleetHas(game.ShipViews[source], "moff-jerjerrod-commander"))
                    yield return Option(source, "remaining.determine-course.jerjerrod", "Suffer one damage for extra yaw");
                var silver = Card(game, source, "clone-captain-silver-officer");
                if (silver is { Active: true, Exhausted: false } && !NavigateResolved(game, source))
                    for (var change = -3; change <= 3; change++)
                        if (change != 0 && MovementModule.PhysicalDialSpeed(game, source) + change >= 0
                            && MovementModule.PhysicalDialSpeed(game, source) + change <= MaximumSpeed(game, source))
                            yield return Option(source, "remaining.determine-course.silver." + change, "Exhaust Silver and change speed " + change);
                if (game.HasUpgrade(source, "swift-return-title") && OfficialObjectives.Obstacles(game).Any(index =>
                    OfficialObjectives.DistanceToPiece(game, source, index) <= TabletopMeasurements.DistanceBand(2)))
                {
                    if (MovementModule.PhysicalDialSpeed(game, source) > 0) yield return Option(source, "remaining.determine-course.swift.-1", "Swift Return: speed -1");
                    if (MovementModule.PhysicalDialSpeed(game, source) < MaximumSpeed(game, source)) yield return Option(source, "remaining.determine-course.swift.1", "Swift Return: speed +1");
                    yield return Option(source, "remaining.determine-course.swift.yaw", "Swift Return: extra yaw");
                }
            }
            if (timing == "after-maneuver")
            {
                if (NavigateResolved(game, source))
                {
                    if (Card(game, source, "engine-techs") is { Active: true, Exhausted: false } && game.Flow.Movement is { CollidedShip: < 0 })
                        yield return Option(source, "remaining.after-maneuver.engine-techs", "Exhaust Engine Techs for a speed-one maneuver");
                    if (Card(game, source, "quantum-storm-title") is { Active: true, Exhausted: false })
                        yield return Option(source, "remaining.after-maneuver.quantum-storm", "Exhaust Quantum Storm for a speed-one maneuver");
                }
                if (game.HasUpgrade(source, "fighter-coordination-team") && CoordinationCandidates(game, source, "fighter-coordination-team", []).Any())
                    yield return Option(source, "remaining.after-maneuver.fighter-coordination-team", "Move eligible squadrons with Fighter Coordination Team");
            }
        }
        if (timing == "before-end-squadron-phase")
            foreach (var ship in game.Topology.Ships)
                if (!game.IsDestroyed(ship.EntityIndex) && !game.IsReserved(ship.EntityIndex)
                    && game.HasUpgrade(ship.EntityIndex, "wedge-antilles-officer") && game.State.Commands[ship.EntityIndex].HasToken(ShipCommand.Squadron)
                    && CoordinationCandidates(game, ship.EntityIndex, "wedge-antilles-officer", []).Any())
                    yield return Option(ship.EntityIndex, "movement.wedge-antilles", "Wedge: grant Cloak to up to three eligible squadrons");
    }
    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (TryApplySquadronEffect(game, window, action) || TryApplyAdditionalShipEffect(game, action)) return true;
        var id = action.AbilityId; int source = action.Source;
        if (id.StartsWith("remaining.determine-course.", StringComparison.Ordinal))
        {
            game.Flow.Flags[RemainingCourseKey(game, source)] = 1;
            var effect = id["remaining.determine-course.".Length..];
            if (effect == "jerjerrod")
            {
                game.Flow.Flags[$"movement.jerjerrod.{game.State.Round}.{source}"] = 1;
                CombatModule.QueueDamage(game, source, 1, false, false, GamePhase.AbilityChoice, rule: "moff-jerjerrod");
                return true;
            }
            if (effect == "swift.yaw") { game.Flow.Flags[$"movement.extra-yaw.{game.State.Round}.{source}"] = 1; return true; }
            int change;
            if (effect.StartsWith("silver.", StringComparison.Ordinal) && int.TryParse(effect[7..], out change))
            {
                if (Card(game, source, "clone-captain-silver-officer")?.Exhaust() != true) throw new ArgumentException("Silver is unavailable.");
                game.Flow.ShipActivations[source].ResolvedCommands |= 1 << (int)ShipCommand.Navigate;
            }
            else if (effect.StartsWith("swift.", StringComparison.Ordinal) && int.TryParse(effect[6..], out change)) { }
            else return false;
            ChangeSpeed(game, source, change); return true;
        }
        if (id is "remaining.after-maneuver.engine-techs" or "remaining.after-maneuver.quantum-storm")
        {
            string cardId = id.EndsWith("engine-techs", StringComparison.Ordinal) ? "engine-techs" : "quantum-storm-title";
            if (Card(game, source, cardId)?.Exhaust() != true) throw new ArgumentException("Extra maneuver card is unavailable.");
            game.Flow.ExtraManeuvers.Add(new() { Ship = source, OriginalSpeed = game.State.Ships[source].CurrentSpeed,
                ParentMovement = game.Flow.Movement?.DeepClone() });
            game.State.Ships[source].SetSpeed(1, MaximumSpeed(game, source));
            game.Flow.Movement = null;
            game.Flow.Flags.Remove($"movement.determine-course.{game.State.Round}.{source}");
            game.Flow.Flags.Remove($"movement.before-determine.{game.State.Round}.{source}");
            game.Flow.Flags.Remove(RemainingCourseKey(game, source));
            game.Flow.Flags.Remove(G8Key(game, source));
            game.Flow.Flags.Remove($"movement.extra-yaw.{game.State.Round}.{source}");
            game.Flow.Flags.Remove($"movement.jerjerrod.{game.State.Round}.{source}");
            EngineEffects.PushChoice(game, new("movement-choice.extra-maneuver", source), []);
            game.Flow.Phase = GamePhase.SelectCourse; return true;
        }
        if (id is "remaining.after-maneuver.fighter-coordination-team" or "remaining.start-activation.squall" or "movement.wedge-antilles")
        {
            var card = id == "remaining.start-activation.squall" ? "squall-title"
                : id == "movement.wedge-antilles" ? "wedge-antilles-officer" : "fighter-coordination-team";
            var remaining = card == "fighter-coordination-team" ? CommandUpgradeEffects.SquadronValue(game.Topology.Ships[source].CatalogEntry.Definition.Squadron,
                game.HasUpgrade(source, "expanded-hangar-bay"), game.HasUpgrade(source, "spha-t")) : 3;
            EngineEffects.PushChoice(game, new("movement-choice.coordination", source, Value: remaining, Text: card, Indices: []), []);
            return true;
        }
        return false;
    }
    public static bool Handles(string rule) => rule.StartsWith("movement-choice.", StringComparison.Ordinal);
    public static GameDecision GetDecision(GameContext game, GameContinuation continuation)
    {
        if (GetSquadronChoiceDecision(game, continuation) is { } squadronDecision) return squadronDecision;
        if (GetAdditionalShipDecision(game, continuation) is { } shipDecision) return shipDecision;
        if (continuation.Rule == "movement-choice.g8")
        {
            var options = G8Candidates(game, continuation.Source).Select(ship => new GameActionOption(
                new(GameActionKind.UseAbility, ship, continuation.Source, AbilityId: "g8-projector.use"), "Exhaust G-8 Projector")).ToList();
            options.Add(new(new(GameActionKind.Pass, continuation.Source, AbilityId: "movement.g8.skip"), "Skip G-8 Projector"));
            return new(options.Count == 1 ? DecisionKind.Forced : DecisionKind.Player, options.Count == 1 ? 0 : 3 - game.State.Owners[continuation.Source], GamePhase.AbilityChoice, options);
        }
        if (continuation.Rule == "movement-choice.coordination")
        {
            var options = new List<GameActionOption>();
            if (continuation.Value > 0)
                foreach (var squad in CoordinationCandidates(game, continuation.Source, continuation.Text, continuation.Indices ?? []))
                    options.Add(continuation.Text == "wedge-antilles-officer"
                        ? new(new(GameActionKind.ChooseEntity, continuation.Source, squad, AbilityId: continuation.Text), "Grant Cloak to this squadron")
                        : new(new(GameActionKind.MoveSquadron, continuation.Source, squad, AbilityId: continuation.Text), "Reposition eligible squadron", AcceptsPosition: true));
            options.Add(new(new(GameActionKind.Pass, continuation.Source, AbilityId: "movement.coordination.finish"), "Finish squadron coordination"));
            return new(options.Count == 1 ? DecisionKind.Forced : DecisionKind.Player, options.Count == 1 ? 0 : game.State.Owners[continuation.Source], GamePhase.AbilityChoice, options);
        }
        throw new NotSupportedException("Unknown movement continuation: " + continuation.Rule);
    }
    public static void Apply(GameContext game, GameContinuation continuation, GameAction action)
    {
        if (!GetDecision(game, continuation).Actions.Any(o => GameSession.Matches(o, action))) throw new ArgumentException("Illegal movement ability choice.", nameof(action));
        if (ApplySquadronChoice(game, continuation, action)) return;
        if (ApplyAdditionalShipChoice(game, continuation, action)) return;
        if (continuation.Rule == "movement-choice.g8")
        {
            if (action.Kind != GameActionKind.Pass)
            {
                if (Card(game, action.Source, "g-8-experimental-projector")?.Exhaust() != true) throw new ArgumentException("G-8 is unavailable.");
                int mover = continuation.Source;
                game.Flow.Flags[game.Topology.ProjectorRestoreKeys[mover]] = game.State.Ships[mover].CurrentSpeed + 1;
                game.State.Ships[mover].SetSpeed(game.State.Ships[mover].CurrentSpeed - 1, MaximumSpeed(game, mover));
            }
            game.Flow.Flags[G8Key(game, continuation.Source)] = 1;
            game.Flow.Continuations.RemoveAt(game.Flow.Continuations.Count - 1); game.Flow.Phase = GamePhase.SelectCourse; return;
        }
        if (continuation.Rule == "movement-choice.coordination")
        {
            if (action.Kind == GameActionKind.Pass) { EngineEffects.ReturnFromChoice(game); return; }
            if (continuation.Text == "wedge-antilles-officer")
            {
                if ((continuation.Indices?.Length ?? 0) == 0 && !game.State.Commands[continuation.Source].TrySpendToken(ShipCommand.Squadron))
                    throw new ArgumentException("Wedge requires a Squadron token.");
                game.State.Squadrons[game.SquadronIndex(action.Target)].GrantTemporaryCloak();
                game.Flow.Continuations[^1] = continuation with { Value = continuation.Value - 1, Indices = [..continuation.Indices ?? [], action.Target] }; return;
            }
            int squad = action.Target; var position = action.Position ?? throw new ArgumentException("A destination is required.");
            var range = continuation.Text == "squall-title" ? TabletopMeasurements.DistanceBand(2) : TabletopMeasurements.DistanceOne;
            if (Vector2.Distance(position, game.State.Poses[squad].Position) > range + Geometry2D.DefaultTolerance
                || !SquadronModule.IsLegalMoveDestination(game, squad, position)) throw new ArgumentException("Coordination destination is illegal.");
            if (continuation.Text == "squall-title" && EngagedAt(game, squad, position)) throw new ArgumentException("Squall cannot end engaged.");
            game.State.Poses[squad] = new(position, game.State.Poses[squad].YawDegrees);
            PlayAreaDestruction.DestroyIfOutside(game, squad);
            game.Flow.Continuations[^1] = continuation with { Value = continuation.Value - 1, Indices = [..continuation.Indices ?? [], squad] };
            RoundModule.ObserveObjectives(game); RoundModule.CheckTerminal(game);
            MissingSquadronLifecycleEffects.BeginBonusMoveConsequences(game, squad, GamePhase.AbilityChoice); return;
        }
        throw new NotSupportedException("Unknown movement continuation: " + continuation.Rule);
    }
    public static bool BeginG8Choice(GameContext game, int mover)
    {
        if (game.Flow.Flags.ContainsKey(G8Key(game, mover))) return false;
        if (!G8Candidates(game, mover).Any()) { game.Flow.Flags[G8Key(game, mover)] = 1; return false; }
        EngineEffects.PushChoice(game, new("movement-choice.g8", mover), []); return true;
    }
    public static void RestoreTemporarySpeedAfterMove(GameContext game, int ship)
    {
        OfficialCreatureObjectives.RestoreGravitySpeed(game, ship);
        if (game.Flow.Flags.Remove(game.Topology.ProjectorRestoreKeys[ship], out var original)) game.State.Ships[ship].SetSpeed((int)original - 1, MaximumSpeed(game, ship));
        if (game.Flow.ForcedManeuvers.LastOrDefault() is { } forced && forced.Ship == ship)
            game.State.Ships[ship].SetSpeed(forced.OriginalSpeed, MaximumSpeed(game, ship));
        if (game.Flow.ExtraManeuvers.LastOrDefault() is { } extra && extra.Ship == ship)
            game.State.Ships[ship].SetSpeed(extra.OriginalSpeed, MaximumSpeed(game, ship));
    }
    public static bool FinishExtraManeuver(GameContext game)
    {
        if (game.Flow.ExtraManeuvers.Count == 0) return false;
        var extra = game.Flow.ExtraManeuvers[^1];
        if (game.Flow.Continuations.Count == 0 || game.Flow.Continuations[^1].Rule != "movement-choice.extra-maneuver")
            throw new InvalidOperationException("Extra maneuver continuation mismatch.");
        game.State.Ships[extra.Ship].SetSpeed(extra.OriginalSpeed, MaximumSpeed(game, extra.Ship));
        game.Flow.Movement = extra.ParentMovement?.DeepClone(); game.Flow.ExtraManeuvers.RemoveAt(game.Flow.ExtraManeuvers.Count - 1);
        EngineEffects.ReturnFromChoice(game);
        // Upstream replaces the old post-maneuver action bar with the additional maneuver's
        // action bar. Its completed window must not offer the old FCT/Slicer options again.
        if (game.Flow.Continuations.LastOrDefault() is { Rule: "effects.window", Text: "after-maneuver" } parent)
            EngineEffects.CloseWindow(game, parent);
        if (game.Flow.Movement is { } movement) movement.Stage = 7;
        game.Flow.Phase = GamePhase.SelectCourse; return true;
    }
    private static IEnumerable<int> G8Candidates(GameContext game, int mover)
    {
        if (game.State.Ships[mover].CurrentSpeed <= 0) yield break;
        foreach (var source in game.Topology.Ships)
            if (source.Owner != game.State.Owners[mover] && !game.IsDestroyed(source.EntityIndex) && !game.IsReserved(source.EntityIndex)
                && Card(game, source.EntityIndex, "g-8-experimental-projector") is { Active: true, Exhausted: false }
                && Geometry2D.Distance(game.ShipBase(mover), game.ShipBase(source.EntityIndex)) <= TabletopMeasurements.LongRange) yield return source.EntityIndex;
    }
    private static IEnumerable<int> CoordinationCandidates(GameContext game, int ship, string card, int[] moved)
    {
        foreach (var squad in game.Topology.Squadrons)
        {
            int entity = squad.EntityIndex;
            if (squad.Owner != game.State.Owners[ship] || game.IsDestroyed(entity) || game.IsReserved(entity) || moved.Contains(entity)
                || game.State.Squadrons[squad.SquadronIndex].TemporaryAttackOnly) continue;
            if (card == "wedge-antilles-officer")
            {
                if (squad.CatalogEntry.Definition.IsUnique || SquadronAbilityEngine.HasKeyword(squad.CatalogEntry.Definition, "STRATEGIC")
                    || Geometry2D.Distance(game.SquadronBase(entity), game.ShipBase(ship)) > TabletopMeasurements.LongRange) continue;
            }
            else if (Geometry2D.Distance(game.SquadronBase(entity), game.ShipBase(ship)) > TabletopMeasurements.MediumRange
                || EngagedAt(game, entity, game.State.Poses[entity].Position)) continue;
            yield return entity;
        }
    }
    private static bool EngagedAt(GameContext game, int squad, Vector2 position)
    {
        var circle = new Circle(position, game.SquadronBase(squad).Radius);
        return game.Topology.Squadrons.Any(other => other.Owner != game.State.Owners[squad]
            && !game.IsDestroyed(other.EntityIndex) && !game.IsReserved(other.EntityIndex)
            && Geometry2D.Distance(circle, game.SquadronBase(other.EntityIndex)) <= TabletopMeasurements.DistanceOne);
    }
    private static GameActionOption Option(int source, string id, string label) => new(new(GameActionKind.UseAbility, source, AbilityId: id), label);
    private static UpgradeCardRuntimeState? Card(GameContext game, int entity, string id)
    {
        for (int i = 0; i < game.State.Upgrades.Length; i++)
            if (game.State.Upgrades[i].Definition.id == id && game.State.Registers[game.Topology.Registers.UpgradeOwner(i)] == entity + 1) return game.State.Upgrades[i];
        return null;
    }
    private static bool NavigateResolved(GameContext game, int ship) => (game.Flow.ShipActivations[ship].ResolvedCommands & (1 << (int)ShipCommand.Navigate)) != 0;
    private static int MaximumSpeed(GameContext game, int ship) => game.Topology.Ships[ship].CatalogEntry.Definition.MaximumSpeed;
    private static string RemainingCourseKey(GameContext game, int ship) => $"movement.remaining-course.{game.State.Round}.{ship}";
    private static string G8Key(GameContext game, int ship) => $"movement.g8-resolved.{game.State.Round}.{ship}";
    private static void ChangeSpeed(GameContext game, int ship, int delta)
    {
        int previous = MovementModule.PhysicalDialSpeed(game, ship);
        if (previous + delta < 0 || previous + delta > MaximumSpeed(game, ship)) throw new ArgumentException("Illegal speed change.");
        MovementModule.SetPhysicalDialSpeed(game, ship, previous + delta); RoundModule.ObserveObjectives(game);
        if (delta != 0 && game.State.Ships[ship].HasFaceupCritical(CriticalDamageCard.ThrusterFissure))
            CombatModule.QueueDamage(game, ship, 1, false, false, GamePhase.AbilityChoice, rule: "thruster-fissure");
    }
}
