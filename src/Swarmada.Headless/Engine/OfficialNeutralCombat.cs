using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

public sealed partial class CombatModule
{
    /// <summary>The objective attack is a ship attack, at medium range, and cannot be
    /// obstructed. It has no friendly fleet and no physical attacker for Counter/Salvo.</summary>
    public static void QueuePlanetaryAttack(GameContext game, int tokenIndex, int defender, HullZone zone, GamePhase returnPhase)
    {
        if (!game.IsShip(defender) || !OfficialObjectives.Live(game, defender)) throw new ArgumentException("Planetary Ion Cannon requires a live defending ship.");
        var attack = new AttackState { Attacker = -1, Defender = defender, ToZone = zone,
            Range = RangeBand.Medium, DiceRange = RangeBand.Medium, Dice = Unrolled(new(0, 4, 0)), BlueDice = 4, ResumePhase = returnPhase };
        attack.EffectFlags["neutral.ship"] = 1; attack.EffectFlags["neutral.planetary"] = tokenIndex + 1;
        attack.EffectFlags["neutral.owner"] = OfficialObjectives.SecondPlayer(game);
        game.Flow.Attacks.Add(attack); new CombatModule().Enter(game, attack, GamePhase.ModifyAttack, CombatWindow.BeforeRoll);
    }
    /// <summary>Exogorth LOS starts at its closest token point, ignoring the token itself
    /// and every obstacle it touches. Other intervening ships and obstacles can obstruct.</summary>
    public static void QueueExogorthAttack(GameContext game, int obstacleIndex, int defender, GamePhase returnPhase = GamePhase.AbilityChoice)
    {
        if (game.IsShip(defender) || !OfficialObjectives.Live(game, defender)) throw new ArgumentException("Exogorth requires a live defending squadron.");
        var shape = ObstacleGeometry.Shape(game, obstacleIndex);
        var target = game.SquadronBase(defender); var origin = shape.ClosestBoundaryPoint(target.Center);
        var blockers = new List<LineOfSightObstacle>();
        foreach (var ship in game.SceneShipViews) if (!Station(game, ship.Index)) blockers.Add(new(game.ShipBase(ship.Index)));
        foreach (int index in OfficialObjectives.Obstacles(game))
        {
            if (index == obstacleIndex) continue;
            if (OfficialObjectives.Is(game, "contested-outpost") && OfficialObjectives.ObstacleId(game, index) == "station") continue;
            var other = ObstacleGeometry.Shape(game, index);
            if (Geometry2D.Distance(shape, other) <= .0115f) continue;
            blockers.Add(new(other));
        }
        bool obstructed = LineOfSightGeometry.Evaluate(new(origin, target.Center), blockers.ToArray()) == LineOfSightResult.Obstructed;
        var attack = new AttackState { Attacker = -1, Defender = defender, Range = RangeBand.Close, DiceRange = RangeBand.Close,
            Dice = Unrolled(new(0, 3, 0)), BlueDice = 3, Obstructed = obstructed, MandatoryRemovals = obstructed ? 1 : 0, ResumePhase = returnPhase };
        attack.EffectFlags["neutral.ship"] = 1; attack.EffectFlags["neutral.exogorth"] = obstacleIndex + 1;
        attack.EffectFlags["neutral.owner"] = 3 - game.State.Owners[defender];
        game.Flow.Attacks.Add(attack); new CombatModule().Enter(game, attack, GamePhase.ModifyAttack, CombatWindow.BeforeRoll);
    }
    public static void QueueStationDamage(GameContext game, int pieceIndex, int amount, GamePhase returnPhase = GamePhase.AbilityChoice)
    {
        if (OfficialObjectives.State(game)?.PieceEntities.TryGetValue(pieceIndex, out int station) != true)
            throw new ArgumentException("Station piece has no prepared station entity.");
        QueueDamage(game, station, amount, false, true, returnPhase, rule: "objective.station.damage");
    }
    private static Vector2 ClosestPointOnRectangle(OrientedRectangle rectangle, Vector2 reference)
    {
        Span<Vector2> corners = stackalloc Vector2[4]; rectangle.CopyCorners(corners);
        Vector2 result = corners[0]; float distance = float.PositiveInfinity;
        for (int i = 0; i < 4; i++)
        {
            var point = Geometry2D.ClosestPoint(new(corners[i], corners[(i + 1) % 4]), reference);
            float squared = Vector2.DistanceSquared(point, reference);
            if (squared < distance) { result = point; distance = squared; }
        }
        return result;
    }
    internal static HullZoneGeometry StationZone(GameContext game, int station, Vector2 reference)
    {
        var shape = StationShape(game, station);
        var edges = new Segment[shape.Count]; for (int i = 0; i < edges.Length; i++) edges[i] = shape.Edge(i);
        return new(HullZone.Front, shape.Center, new(-1, 1), new(1, 1), shape.ClosestBoundaryPoint(reference), edges);
    }
    internal static PolygonFootprint StationShape(GameContext game, int station)
    {
        if (OfficialObjectives.State(game) is { } state)
            foreach (var piece in state.PieceEntities)
                if (piece.Value == station) return ObstacleGeometry.Shape(game, piece.Key);
        if (game.IsShip(station) && game.Topology.Ships[station].IsStation)
            return new(ObstacleShapeCatalog.Get("station"), game.State.Poses[station], game.Topology.Ships[station].FootprintMillimetres);
        throw new InvalidOperationException("A station entity requires its prepared physical obstacle.");
    }
    private static bool Station(GameContext game, int entity) => game.IsShip(entity) && game.Topology.Ships[entity].IsStation;
    internal static IEnumerable<HullZone> TargetZones(GameContext game, int entity) => Station(game, entity) ? [HullZone.Front]
        : game.IsShip(entity) ? game.State.Ships[entity].HullZones : [HullZone.Front];
}
