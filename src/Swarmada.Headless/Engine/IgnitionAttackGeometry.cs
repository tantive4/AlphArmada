using System.Numerics;
using Armada.Core;
using Armada.Data;
using Armada.Rules;
using Armada.Session;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public readonly record struct IgnitionPlacementMeasurement(RangeBand Range, float Distance,
    bool InSpecialArc, bool MatchesRequiredBand, Vector2 RangeOrigin, Vector2 RangeDestination)
{ public bool IsLegal => InSpecialArc && MatchesRequiredBand; }
public readonly record struct IgnitionAttackMeasurement(HullZone AssociatedHullZone, RangeBand AttackRange,
    RangeBand DiceRange, float AttackDistance, float DiceDistance, bool Obstructed, bool IsExtremeRange,
    BatteryArmament Battery, Vector2 AttackOrigin, Vector2 AttackDestination, Vector2 DiceOrigin, Vector2 DiceDestination);

/// <summary>Exact source ignition dual-distance model: hull-edge range controls defense effects,
/// while token-to-entire-defender-base range determines usable special-battery dice.</summary>
public static partial class IgnitionAttackGeometry
{
    public const int TargetTokenKind = 10001;
    public static IReadOnlyList<RangeBand> AvailablePlacementBands(IgnitionProfile? profile, bool grantsMedium, bool grantsLong)
    {
        var result = (profile?.BasePlacementBands ?? []).Where(b => b != RangeBand.OutOfRange).Distinct().ToList();
        if (grantsMedium && !result.Contains(RangeBand.Medium)) result.Add(RangeBand.Medium);
        if (grantsLong && !result.Contains(RangeBand.Long)) result.Add(RangeBand.Long);
        return result;
    }
    public static IgnitionPlacementMeasurement MeasurePlacement(GameContext game, int ship, HullZone zone, Vector2 position, RangeBand requiredBand)
    {
        using var scope = new GameScope(game);
        if (!RemainingUpgradeRules.TryGetEffectiveIgnitionProfile(game.ShipViews[ship], zone, out var profile))
            return new(RangeBand.OutOfRange, float.PositiveInfinity, false, false, default, position);
        var hull = Hull(game, ship, zone);
        var origin = hull.Edges.ToArray().Select(edge => Geometry2D.ClosestPoint(edge, position)).MinBy(point => Vector2.DistanceSquared(point, position));
        float distance = Vector2.Distance(origin, position); var range = HullGeometry.ClassifyRange(distance);
        var pose = game.State.Poses[ship];
        bool arc = ContainsPointInSpecialArc(new(pose.Position.X, 0, pose.Position.Y),
            Quaternion.CreateFromAxisAngle(Vector3.UnitY, pose.YawDegrees * MathF.PI / 180), profile, new(position.X, 0, position.Y));
        return new(range, distance, arc, range == requiredBand, origin, position);
    }
    public static bool TryGetTarget(GameContext game, int ship, out SpatialObjectState token, out int index)
    {
        using var scope = new GameScope(game);
        for (int i = 0; i < game.State.SpatialObjects.Count; i++)
        {
            var item = game.State.SpatialObjects[i];
            if (item.Kind == TargetTokenKind && item.Active && item.SourceEntity == ship && item.Owner == game.State.Owners[ship]
                && RemainingUpgradeRules.TryGetEffectiveIgnitionProfile(game.ShipViews[ship], (HullZone)item.Payload, out _))
            { token = item; index = i; return true; }
        }
        token = default; index = -1; return false;
    }
    public static void PlaceTarget(GameContext game, int ship, HullZone zone, Vector2 position)
    {
        int index = game.State.SpatialObjects.FindIndex(t => t.Kind == TargetTokenKind && t.SourceEntity == ship);
        var token = new SpatialObjectState(TargetTokenKind, new(position, game.State.Poses[ship].YawDegrees), Vector2.One,
            game.State.Owners[ship], ship, Payload: (int)zone);
        if (index >= 0) game.State.SpatialObjects[index] = token;
        else { index = game.State.SpatialObjects.Count; game.State.SpatialObjects.Add(token); }
        TokenObstacleAttachments.RecordPlacement(game, index);
    }
    public static void ConsumeTarget(GameContext game, int ship)
    {
        for (int i = 0; i < game.State.SpatialObjects.Count; i++)
            if (game.State.SpatialObjects[i].Kind == TargetTokenKind && game.State.SpatialObjects[i].SourceEntity == ship)
            {
                game.State.SpatialObjects[i] = game.State.SpatialObjects[i] with { Active = false };
                TokenObstacleAttachments.ForgetToken(game, i);
            }
    }
    public static bool TryMeasureAttack(GameContext game, int attacker, int defender, HullZone targetZone, out IgnitionAttackMeasurement measurement)
    {
        measurement = default;
        if (!game.IsShip(attacker) || !game.IsShip(defender) || game.IsDestroyed(attacker) || game.IsDestroyed(defender)
            || game.IsReserved(attacker) || game.IsReserved(defender) || !TryGetTarget(game, attacker, out var token, out _)) return false;
        using var scope = new GameScope(game);
        if (!RemainingUpgradeRules.TryGetEffectiveIgnitionProfile(game.ShipViews[attacker], (HullZone)token.Payload, out var profile)) return false;
        var origin = game.State.Poses[attacker]; var ordinary = Hull(game, attacker, profile.AssociatedHullZone);
        var special = new HullZoneGeometry(profile.AssociatedHullZone, origin.Position,
            origin.TransformDirection(profile.LeftBoundaryLocal), origin.TransformDirection(profile.RightBoundaryLocal), ordinary.TargetingPoint, ordinary.Edges);
        bool station = game.Topology.Ships[defender].IsStation;
        if (station && (targetZone != HullZone.Front || Geometry2D.Overlaps(game.ShipBase(attacker), CombatModule.StationShape(game, defender)))) return false;
        var target = station ? CombatModule.StationZone(game, defender, ordinary.TargetingPoint) : Hull(game, defender, targetZone);
        var attack = HullGeometry.MeasureAttack(special, target);
        if (!attack.HasEligibleDefenderEdge || !station && !(HullGeometry.IsTargetZoneExposed(ordinary.TargetingPoint, target, game.ShipBase(defender))
            || game.ShipBase(defender).Contains(ordinary.TargetingPoint) && target.ContainsInArc(ordinary.TargetingPoint))) return false;
        var attackRange = HullGeometry.ClassifyRange(attack.Distance);
        var diceDestination = station ? CombatModule.StationShape(game, defender).ClosestPoint(token.Pose.Position) : game.ShipBase(defender).ClosestPoint(token.Pose.Position);
        float diceDistance = Vector2.Distance(token.Pose.Position, diceDestination); var diceRange = HullGeometry.ClassifyRange(diceDistance);
        var battery = profile.SpecialBattery;
        if (!AttackResolver.HasUsableDiceAtRange(battery.RedDice, battery.BlueDice, battery.BlackDice, diceRange)
            || !game.State.Ships[attacker].CanAttackAtRange(attackRange)) return false;
        var blockers = new List<LineOfSightObstacle>();
        foreach (var ship in game.Topology.Ships)
            if (ship.EntityIndex != attacker && ship.EntityIndex != defender && !ship.IsStation && !game.IsDestroyed(ship.EntityIndex) && !game.IsReserved(ship.EntityIndex)) blockers.Add(new(game.ShipBase(ship.EntityIndex)));
        for (int index = 0; index < game.State.SpatialObjects.Count; index++)
        {
            var obstacle = game.State.SpatialObjects[index];
            if (OfficialObjectives.Is(game, "contested-outpost") && OfficialObjectives.ObstacleId(game, index) == "station") continue;
            if (OfficialObjectives.State(game)?.PieceEntities.TryGetValue(index, out int entity) == true && (entity == attacker || entity == defender)) continue;
            if (obstacle.Active && obstacle.Kind == (int)MapPlacementKind.Obstacle) blockers.Add(new(ObstacleGeometry.Shape(game, index)));
        }
        bool obstructed = LineOfSightGeometry.Evaluate(new(ordinary.TargetingPoint, target.TargetingPoint), blockers.ToArray()) == LineOfSightResult.Obstructed;
        obstructed |= CommanderRules.DefenderIsTreatedAsObstructed(game.ShipViews[defender]).Applied || OfficerCombatRules.ObstructionNotice(game.ShipViews[defender]).Applied
            || game.Flow.Flags.GetValueOrDefault($"round.early-warning.{defender}") == (int)targetZone + 1;
        if (UnitAbilityEngine.AttacksCannotBeObstructed(game.ShipViews[attacker])) obstructed = false;
        if (obstructed && !game.State.Ships[attacker].CanChooseObstructedTarget) return false;
        measurement = new(profile.AssociatedHullZone, attackRange, diceRange, attack.Distance, diceDistance, obstructed,
            attackRange == RangeBand.OutOfRange, battery, attack.Origin, attack.Destination, token.Pose.Position, diceDestination);
        return true;
    }
    private static HullZoneGeometry Hull(GameContext game, int ship, HullZone zone) => game.State.Ships[ship].HasAuxiliaryHullZones
        ? HullGeometry.Huge(game.ShipBase(ship), zone, game.Topology.Ships[ship].CatalogEntry.Definition.ShipType == "prototype-super-star-destroyer")
        : HullGeometry.Standard(game.ShipBase(ship), zone);
}
