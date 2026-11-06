using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Armada.Session;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Map choices use continuous physical placement. Obstacle movement shares
/// the objective-aware validator and moves attached tokens with the obstacle.</summary>
public static partial class MapAbilityEffects
{
    private const float MineRadius = .26f * TabletopMeasurements.MillimetresPerWorldUnit;
    public static bool SupportsUpgrade(string id) => id is "ezra-bridger-officer" or "sabine-wren-officer" or "cataclysm-title" or "spha-t"
        or "superheavy-composite-beam-turbolasers" or "orbital-bombardment-particle-cannons" or "rapid-launch-bays" or "invisible-hand-title" or "mercy-mission";
    public static bool Handles(string rule) => rule.StartsWith("map-choice.", StringComparison.Ordinal);
    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        using var scope = new GameScope(game);
        if (timing == "before-start-squadron-phase")
            foreach (var squad in game.Topology.Squadrons)
                if (!game.IsDestroyed(squad.EntityIndex) && !game.IsReserved(squad.EntityIndex) && !game.State.Squadrons[squad.SquadronIndex].Activated
                    && squad.CatalogEntry.Definition.SquadronId == "mart-mattin-satos-hammer")
                    yield return Ability(squad.EntityIndex, "mart-mattin.place", "Give up Mart Mattin's activation to place a proximity mine");
        if (timing == "start-ship-phase")
            foreach (var ship in game.Topology.Ships)
                if (AvailableShip(game, ship.EntityIndex) && game.HasUpgrade(ship.EntityIndex, "sabine-wren-officer") && Obstacles(game).Any())
                    yield return Ability(ship.EntityIndex, "start-placement.sabine.place", "Discard Sabine to place a proximity mine");
        if (timing == "after-start-ship-phase")
            foreach (var ship in game.Topology.Ships)
                if (AvailableShip(game, ship.EntityIndex) && game.HasUpgrade(ship.EntityIndex, "cataclysm-title")
                    && game.State.Commands[ship.EntityIndex].HasToken(ShipCommand.ConcentrateFire)
                    && RemainingUpgradeRules.CanPlaceEffectiveIgnitionTarget(game.ShipViews[ship.EntityIndex]))
                    foreach (var profile in RemainingUpgradeRules.EffectiveIgnitionProfiles(game.ShipViews[ship.EntityIndex]))
                        foreach (var range in PlacementBands(game, ship.EntityIndex, profile.AssociatedHullZone))
                            yield return new(new(GameActionKind.UseAbility, ship.EntityIndex, FromZone: profile.AssociatedHullZone, Index: (int)range,
                                AbilityId: "start-placement.cataclysm." + range.ToString().ToLowerInvariant()), "Spend Concentrate Fire token and place targeting token at " + range);
        if (!AvailableShip(game, source)) yield break;
        if (MapMovementRules.Enabled(game) && timing == "command-reveal" && game.HasUpgrade(source, "ezra-bridger-officer") && EzraSources(game, source).Any())
            yield return Ability(source, "map.ezra", "Ezra: choose a nearby obstacle to move");
        if (timing == "end-activation" && RemainingUpgradeRules.CanPlaceEffectiveIgnitionTarget(game.ShipViews[source]) && !IgnitionPlacedInWindow(game))
            foreach (var profile in RemainingUpgradeRules.EffectiveIgnitionProfiles(game.ShipViews[source]))
                foreach (var range in PlacementBands(game, source, profile.AssociatedHullZone))
                    yield return new(new(GameActionKind.UseAbility, source, FromZone: profile.AssociatedHullZone, Index: (int)range,
                        AbilityId: "remaining.end-activation.ignition." + (int)profile.AssociatedHullZone + "." + range.ToString().ToLowerInvariant()),
                        "Place " + profile.AssociatedHullZone + " targeting token at " + range);
    }
    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (action.AbilityId == "mart-mattin.place")
        {
            game.State.Squadrons[game.SquadronIndex(action.Source)].MarkActivated(false); game.State.Activated[action.Source] = true;
            EngineEffects.PushChoice(game, new("map-choice.mart-mattin", action.Source), []); return true;
        }
        if (action.AbilityId == "map.ezra")
        {
            if (!MapMovementRules.Enabled(game)) throw new ArgumentException("Obstacle relocation is disabled for this match.", nameof(action));
            EngineEffects.PushChoice(game, new("map-choice.ezra-source", action.Source), []); return true;
        }
        if (action.AbilityId == "start-placement.sabine.place")
        { EngineEffects.PushChoice(game, new("map-choice.sabine", action.Source), []); return true; }
        bool cataclysm = action.AbilityId.StartsWith("start-placement.cataclysm.", StringComparison.Ordinal);
        if (cataclysm || action.AbilityId.StartsWith("remaining.end-activation.ignition.", StringComparison.Ordinal))
        {
            EngineEffects.PushChoice(game, new("map-choice.ignition", action.Source, Target: (int)action.FromZone,
                Value: action.Index, Text: cataclysm ? "cataclysm" : "normal"), []); return true;
        }
        return false;
    }
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        if (GetReservedReleaseDecision(game, frame) is { } reservedDecision) return reservedDecision;
        var options = new List<GameActionOption>();
        switch (frame.Rule)
        {
            case "map-choice.ezra-source":
                if (MapMovementRules.Enabled(game))
                    foreach (int obstacle in EzraSources(game, frame.Source)) options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, Index: obstacle), "Choose obstacle"));
                break;
            case "map-choice.ezra-destination":
                if (MapMovementRules.Enabled(game))
                    options.Add(new(new(GameActionKind.UseAbility, frame.Source, AbilityId: "map.ezra.move", Flipped: game.State.SpatialObjects[frame.Target].Flipped),
                        "Choose obstacle destination within distance two", AcceptsPosition: true, AcceptsYaw: true, AcceptsFlip: true));
                break;
            case "map-choice.sabine":
                options.Add(new(new(GameActionKind.PlaceToken, frame.Source, AbilityId: "map.sabine.place"), "Choose legal proximity mine position", AcceptsPosition: true)); break;
            case "map-choice.mart-mattin":
                options.Add(new(new(GameActionKind.PlaceToken, frame.Source, AbilityId: "map.mart-mattin.place"), "Choose mine position within distance one of Mart Mattin", AcceptsPosition: true)); break;
            case "map-choice.ignition":
                options.Add(new(new(GameActionKind.PlaceToken, frame.Source, FromZone: (HullZone)frame.Target, Index: (int)frame.Value, AbilityId: "map.ignition.place"),
                    "Choose position in special arc and selected range band", AcceptsPosition: true)); break;
            default: throw new NotSupportedException("Unknown map ability continuation: " + frame.Rule);
        }
        options.Add(new(new(GameActionKind.Pass, frame.Source), "Cancel placement and keep card or token"));
        return new(options.Count == 1 ? DecisionKind.Forced : DecisionKind.Player, options.Count == 1 ? 0 : game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!GetDecision(game, frame).Actions.Any(o => GameSession.Matches(o, action))) throw new ArgumentException("Illegal map ability choice.");
        if (ApplyReservedRelease(game, frame, action)) return;
        if (action.Kind == GameActionKind.Pass) { EngineEffects.ReturnFromChoice(game); return; }
        switch (frame.Rule)
        {
            case "map-choice.ezra-source":
                game.Flow.Continuations[^1] = frame with { Rule = "map-choice.ezra-destination", Target = action.Index }; return;
            case "map-choice.ezra-destination":
                var destination = new Pose(action.Position ?? throw new ArgumentException("Ezra requires a destination position."), action.YawDegrees);
                var ezra = Card(game, frame.Source, "ezra-bridger-officer");
                if (ezra is not { Active: true }) throw new ArgumentException("Ezra obstacle movement unavailable.");
                OfficialObjectives.ValidateObstacleMove(game, frame.Target, destination, TabletopMeasurements.DistanceBand(2), flipped: action.Flipped);
                if (!ezra.Discard()) throw new ArgumentException("Ezra obstacle movement unavailable.");
                OfficialObjectives.MoveObstacle(game, frame.Target, destination, TabletopMeasurements.DistanceBand(2), flipped: action.Flipped);
                RoundModule.ObserveObjectives(game);
                EngineEffects.ReturnFromChoice(game);
                OfficialObjectives.QueueChoices(game, OfficialAssaultObjectives.PurrgilMovedChoices(game, frame.Target, game.State.Owners[frame.Source]), GamePhase.AbilityChoice); return;
            case "map-choice.sabine":
                var minePosition = action.Position ?? throw new ArgumentException("Mine position required.");
                if (!IsLegalSabinePlacement(game, frame.Source, minePosition)) throw new ArgumentException("Illegal Sabine mine position.");
                if (Card(game, frame.Source, "sabine-wren-officer")?.Discard() != true) throw new ArgumentException("Sabine unavailable.");
                game.State.SpatialObjects.Add(new(SetupModule.ProximityMineKind, new(minePosition), new(MineRadius * 2), game.State.Owners[frame.Source], frame.Source,
                    Id: "mine|sabine|" + game.Topology.Ships[frame.Source].RuntimeId));
                TokenObstacleAttachments.RecordPlacement(game, game.State.SpatialObjects.Count - 1); break;
            case "map-choice.mart-mattin":
                var martPosition = action.Position ?? throw new ArgumentException("Mine position required.");
                if (!IsLegalMineSpacing(game, martPosition) || Vector2.Distance(martPosition, game.State.Poses[frame.Source].Position) - MineRadius - game.SquadronBase(frame.Source).Radius > TabletopMeasurements.DistanceOne)
                    throw new ArgumentException("Illegal Mart Mattin mine position.");
                game.State.SpatialObjects.Add(new(SetupModule.ProximityMineKind, new(martPosition), new(MineRadius * 2), game.State.Owners[frame.Source], frame.Source,
                    Id: "mine|mart-mattin|" + game.Topology.Squadrons[game.SquadronIndex(frame.Source)].RuntimeId));
                TokenObstacleAttachments.RecordPlacement(game, game.State.SpatialObjects.Count - 1); break;
            case "map-choice.ignition":
                var position = action.Position ?? throw new ArgumentException("Targeting token position required.");
                if (!IsLegalIgnitionPlacement(game, frame.Source, (HullZone)frame.Target, position, (RangeBand)frame.Value)) throw new ArgumentException("Illegal ignition target position.");
                using (var scope = new GameScope(game))
                {
                    if (frame.Text == "cataclysm" && !game.State.Commands[frame.Source].TrySpendToken(ShipCommand.ConcentrateFire)) throw new ArgumentException("Cataclysm requires a Concentrate Fire token.");
                    if (!RemainingUpgradeRules.CommitEffectiveIgnitionTargetPlacement(game.ShipViews[frame.Source], (HullZone)frame.Target)) throw new ArgumentException("Ignition battery unavailable.");
                }
                IgnitionAttackGeometry.PlaceTarget(game, frame.Source, (HullZone)frame.Target, position);
                var window = game.Flow.Continuations.LastOrDefault(c => c.Rule == "effects.window");
                if (window is not null) game.Flow.Flags[$"effects.used.{window.Value}.ignition.placed"] = 1;
                break;
        }
        EngineEffects.ReturnFromChoice(game);
    }
    public static bool IsLegalSabinePlacement(GameContext game, int source, Vector2 position)
    {
        if (!IsLegalMineSpacing(game, position)) return false;
        if (!Obstacles(game).Any(i => Geometry2D.Distance(new Circle(position, MineRadius), ObstacleGeometry.Shape(game, i)) <= TabletopMeasurements.DistanceBand(2))) return false;
        return !game.Topology.Ships.Any(s => s.Owner != game.State.Owners[source] && !game.IsDestroyed(s.EntityIndex) && !game.IsReserved(s.EntityIndex)
            && (s.IsStation ? Geometry2D.Distance(new Circle(position, MineRadius), CombatModule.StationShape(game, s.EntityIndex))
                : Math.Max(0, Vector2.Distance(position, game.ShipBase(s.EntityIndex).ClosestPoint(position)) - MineRadius)) <= TabletopMeasurements.DistanceBand(3));
    }
    public static bool IsLegalMineSpacing(GameContext game, Vector2 position)
    {
        if (!float.IsFinite(position.X) || !float.IsFinite(position.Y)) return false;
        var board = MovementModule.Board(game); var local = board.Pose.InverseTransformPoint(position);
        if (MathF.Abs(local.X) + MineRadius > board.HalfSize.X || MathF.Abs(local.Y) + MineRadius > board.HalfSize.Y) return false;
        return !game.State.SpatialObjects.Any(x => x.Active && x.Kind == SetupModule.ProximityMineKind && Vector2.Distance(position, x.Pose.Position) - MineRadius * 2 <= TabletopMeasurements.DistanceOne);
    }
    public static bool IsLegalIgnitionPlacement(GameContext game, int ship, HullZone zone, Vector2 position, RangeBand range)
    {
        if (!float.IsFinite(position.X) || !float.IsFinite(position.Y) || !MovementModule.Board(game).Contains(position)) return false;
        using var scope = new GameScope(game);
        return RemainingUpgradeRules.CanPlaceEffectiveIgnitionTarget(game.ShipViews[ship]) && PlacementBands(game, ship, zone).Contains(range)
            && IgnitionAttackGeometry.MeasurePlacement(game, ship, zone, position, range).IsLegal;
    }
    public static bool IsLegalEzraDestination(GameContext game, int obstacle, Vector2 destination)
    {
        if (!MapMovementRules.Enabled(game)) return false;
        if (!Obstacles(game).Contains(obstacle)) return false;
        try { OfficialObjectives.ValidateObstacleMove(game, obstacle, new(destination, game.State.SpatialObjects[obstacle].Pose.YawDegrees), TabletopMeasurements.DistanceBand(2)); return true; }
        catch (ArgumentException) { return false; }
    }
    private static IEnumerable<int> EzraSources(GameContext game, int source) => Obstacles(game)
        .Where(i => OfficialObjectives.DistanceToPiece(game, source, i) <= TabletopMeasurements.DistanceBand(2));
    private static IEnumerable<int> Obstacles(GameContext game) => Enumerable.Range(0, game.State.SpatialObjects.Count)
        .Where(i => game.State.SpatialObjects[i].Active && game.State.SpatialObjects[i].Kind == (int)MapPlacementKind.Obstacle);
    private static IReadOnlyList<RangeBand> PlacementBands(GameContext game, int ship, HullZone zone)
    {
        if (!RemainingUpgradeRules.TryGetEffectiveIgnitionProfile(game.ShipViews[ship], zone, out var profile)) return [];
        return IgnitionAttackGeometry.AvailablePlacementBands(profile, game.HasUpgrade(ship, "superheavy-composite-beam-turbolasers"), game.HasUpgrade(ship, "orbital-bombardment-particle-cannons"));
    }
    private static bool IgnitionPlacedInWindow(GameContext game)
    {
        var window = game.Flow.Continuations.LastOrDefault(c => c.Rule == "effects.window");
        return window is not null && game.Flow.Flags.ContainsKey($"effects.used.{window.Value}.ignition.placed");
    }
    private static bool AvailableShip(GameContext game, int ship) => game.IsShip(ship) && !game.IsDestroyed(ship) && !game.IsReserved(ship);
    private static UpgradeCardRuntimeState? Card(GameContext game, int ship, string id) => game.UpgradesFor(ship).FirstOrDefault(c => c.Definition.id == id);
    private static GameActionOption Ability(int source, string id, string label) => new(new(GameActionKind.UseAbility, source, AbilityId: id), label);
}
