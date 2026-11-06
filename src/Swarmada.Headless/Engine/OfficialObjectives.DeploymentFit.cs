using System.Numerics;
using Armada.Core;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

public static partial class OfficialObjectives
{
    /// <summary>Caller ruling: reject and retry a setup when no complete Ambush fit
    /// exists. An invalid proposed pose alone cannot invalidate an otherwise legal setup.</summary>
    public static bool ValidateAmbushFit(GameContext game, int entity, OrientedRectangle shape)
    {
        if (State(game)?.InvalidSetup is { } rejected) throw new InvalidSetupException(rejected);
        var context = new DeploymentFitContext(game, entity, AmbushZone(game));
        if (context.Valid(shape)) return true;
        if (context.MayHavePlacement()) return false;
        var result = new InvalidSetupResult(InvalidSetupCode.FleetAmbushNoLegalDeployment, "fleet-ambush",
            game.State.Owners[entity], entity,
            "Fleet Ambush has no legal complete deployment for the selected ship. Discard this setup attempt and restart with a new setup seed.");
        State(game)!.InvalidSetup = result;
        game.State.Revision++;
        throw new InvalidSetupException(result);
    }

    /// <summary>Huge Surprise Attack flagship placement: contact with its own edge takes
    /// priority; only if impossible may its rear hull zone be farther from that edge.</summary>
    public static bool ValidateHugeSurpriseFit(GameContext game, int entity, OrientedRectangle shape)
    {
        if (!game.Topology.Ships[entity].CatalogEntry.Definition.HasAuxiliaryHullZones) return false;
        foreach (int station in Obstacles(game).Where(i => ObstacleId(game, i) == "station"))
        {
            var context = new DeploymentFitContext(game, entity, null, station);
            if (!context.Valid(shape)) continue;
            if (DistanceFromPlayerEdge(game, shape, game.State.Owners[entity]) <= DeploymentFitTolerance) return true;
            if (context.HasPlacement(DeploymentFitGoal.OwnEdge)) return false;
            float rearDistance = context.RearDistance(shape);
            return !context.HasPlacement(DeploymentFitGoal.Rear, rearDistance - DeploymentFitTolerance * 4);
        }
        return false;
    }

    private const float DeploymentFitTolerance = Geometry2D.DefaultTolerance;
    private enum DeploymentFitGoal { Any, OwnEdge, Rear }
    private readonly record struct DeploymentFitInterval(float Low, float High);
    private readonly record struct DeploymentFitCell(float YawLow, float YawHigh, float XLow, float XHigh);
    private sealed class DeploymentFitContext
    {
        private readonly GameContext game;
        private readonly int owner;
        private readonly Vector2 size, inward;
        private readonly float radius, edgeOffset;
        private readonly OrientedRectangle area;
        private readonly OrientedRectangle? requiredZone;
        private readonly PolygonFootprint? station;
        private readonly List<OrientedRectangle> rectangles = [];
        private readonly List<PolygonFootprint> obstacles = [];
        private readonly List<Circle> circles = [];
        private readonly bool prototype;
        public DeploymentFitContext(GameContext game, int entity, OrientedRectangle? requiredZone, int stationIndex = -1)
        {
            this.game = game; this.requiredZone = requiredZone; owner = game.State.Owners[entity];
            size = game.Topology.Ships[entity].FootprintMillimetres; radius = size.Length() / 2; area = SetupArea(game);
            var zone = game.Topology.DeploymentZones.Single(z => z.Owner == owner);
            inward = (zone.MinimumMillimetres.Y + zone.MaximumMillimetres.Y) < 0 ? Vector2.UnitY : -Vector2.UnitY;
            edgeOffset = game.Topology.BoardSizeMillimetres.Y / 2;
            prototype = game.Topology.Ships[entity].CatalogEntry.Definition.ShipId.StartsWith("prototype", StringComparison.Ordinal);
            if (stationIndex >= 0) station = ObstacleGeometry.Shape(game, stationIndex);
            foreach (var ship in game.Topology.Ships)
                if (ship.EntityIndex != entity && !ship.IsStation && Placed(game, ship.EntityIndex) && Live(game, ship.EntityIndex)) rectangles.Add(game.ShipBase(ship.EntityIndex));
            foreach (var squad in game.Topology.Squadrons)
                if (Placed(game, squad.EntityIndex) && Live(game, squad.EntityIndex)) circles.Add(game.SquadronBase(squad.EntityIndex));
            foreach (int piece in Obstacles(game))
                if (piece != stationIndex) obstacles.Add(ObstacleGeometry.Shape(game, piece));
        }
        public bool Valid(OrientedRectangle shape)
        {
            if (!PlayAreaGeometry.Contains(MovementModule.Board(game), shape) || !PlayAreaGeometry.Contains(area, shape) ||
                requiredZone is { } zone && !PlayAreaGeometry.Contains(zone, shape) || station is { } obstacle && !Geometry2D.Overlaps(shape, obstacle)) return false;
            foreach (var rectangle in rectangles) if (Geometry2D.Overlaps(shape, rectangle)) return false;
            foreach (var polygon in obstacles) if (Geometry2D.Overlaps(shape, polygon)) return false;
            foreach (var circle in circles) if (Geometry2D.Overlaps(circle, shape)) return false;
            return true;
        }
        public float RearDistance(OrientedRectangle shape)
        {
            float result = float.PositiveInfinity;
            foreach (var edge in HullGeometry.Huge(shape, HullZone.Rear, prototype).Edges)
            { result = MathF.Min(result, DistanceFromPlayerEdge(game, edge.Start, owner)); result = MathF.Min(result, DistanceFromPlayerEdge(game, edge.End, owner)); }
            return result;
        }
        private float RearOffset(OrientedRectangle shape) => RearDistance(shape) - edgeOffset - Vector2.Dot(shape.Center, inward);
        public bool MayHavePlacement() => HasPlacement(DeploymentFitGoal.Any, conservative: true);
        public bool HasPlacement(DeploymentFitGoal goal, float rearMaximum = 0, bool conservative = false)
        {
            if (station is { } obstacle)
            {
                // No two base points can be farther apart than the diagonal. This
                // orientation-independent lower bound settles the common distant-station
                // exception without subdividing an entire row of equivalent translations.
                float nearestStation = float.PositiveInfinity;
                for (int i = 0; i < obstacle.Count; i++) nearestStation = MathF.Min(nearestStation, DistanceFromPlayerEdge(game, obstacle.Vertex(i), owner));
                if (goal == DeploymentFitGoal.OwnEdge && nearestStation > size.Length() + DeploymentFitTolerance) return false;
                if (goal == DeploymentFitGoal.Rear && rearMaximum < nearestStation - size.Length() - 1e-5f) return false;
            }
            var cells = new Stack<DeploymentFitCell>(); cells.Push(new(0, goal == DeploymentFitGoal.Any ? 180 : 360, -area.HalfSize.X, area.HalfSize.X));
            // Exact-touching fits can lie only on a yaw boundary. A failed midpoint
            // does not prove those poses absent, even at very small physical tolerance.
            if (conservative && (HasWitness(0, 0, goal, rearMaximum) || HasWitness(90, 0, goal, rearMaximum))) return true;
            while (cells.TryPop(out var cell))
            {
                float yaw = (cell.YawLow + cell.YawHigh) / 2, x = (cell.XLow + cell.XHigh) / 2;
                float angularError = radius * 2 * MathF.Sin((cell.YawHigh - cell.YawLow) * MathF.PI / 720);
                float lateralError = (cell.XHigh - cell.XLow) / 2, uncertainty = angularError + lateralError;
                var middle = new OrientedRectangle(new(new Vector2(x, 0), yaw), size);
                if (Intervals(middle, goal, rearMaximum, uncertainty).Count == 0) continue;
                // At fixed yaw and lateral position the full continuum of longitudinal
                // translations is a union of analytic intervals. Include tangent endpoints.
                if (HasWitness(yaw, x, goal, rearMaximum)) return true;
                // This is the engine's physical geometry tolerance, not a movement grid.
                // Conservative interval bounds discard whole blocked pose cells.
                if (uncertainty <= DeploymentFitTolerance / 4)
                {
                    // Only an empty conservative interval is an absence certificate.
                    // Retain ambiguous leaves for Ambush rejection: a caller may still
                    // supply their exact continuous fit, which Valid accepts directly.
                    if (conservative) return true;
                    continue;
                }
                if (angularError >= lateralError)
                { cells.Push(cell with { YawLow = yaw }); cells.Push(cell with { YawHigh = yaw }); }
                else
                { cells.Push(cell with { XLow = x }); cells.Push(cell with { XHigh = x }); }
            }
            return false;
        }
        private bool HasWitness(float yaw, float x, DeploymentFitGoal goal, float rearMaximum)
        {
            var middle = new OrientedRectangle(new(new Vector2(x, 0), yaw), size);
            foreach (var interval in Intervals(middle, goal, rearMaximum, 0))
                foreach (float y in new[] { interval.Low, interval.High, (interval.Low + interval.High) / 2,
                    MathF.Min(interval.High, interval.Low + DeploymentFitTolerance * 4), MathF.Max(interval.Low, interval.High - DeploymentFitTolerance * 4) })
                {
                    var shape = new OrientedRectangle(new(new Vector2(x, 0) + inward * y, yaw), size);
                    if (Valid(shape) && (goal != DeploymentFitGoal.OwnEdge || DistanceFromPlayerEdge(game, shape, owner) <= DeploymentFitTolerance) &&
                        (goal != DeploymentFitGoal.Rear || RearDistance(shape) <= rearMaximum)) return true;
                }
            return false;
        }
        private List<DeploymentFitInterval> Intervals(OrientedRectangle shape, DeploymentFitGoal goal, float rearMaximum, float relaxation)
        {
            float low = -edgeOffset - radius, high = edgeOffset + radius;
            if (!ClipContainment(area, shape, relaxation, ref low, ref high) || !ClipContainment(MovementModule.Board(game), shape, relaxation, ref low, ref high) ||
                requiredZone is { } zone && !ClipContainment(zone, shape, relaxation, ref low, ref high)) return [];
            if (goal == DeploymentFitGoal.OwnEdge) high = MathF.Min(high, DeploymentFitTolerance - edgeOffset + shape.ProjectedRadius(inward) + relaxation);
            if (goal == DeploymentFitGoal.Rear) high = MathF.Min(high, rearMaximum - edgeOffset - RearOffset(shape) + relaxation);
            if (low > high) return [];
            var allowed = new List<DeploymentFitInterval> { new(low, high) };
            if (station is { } obstacle)
            {
                var grown = new OrientedRectangle(shape.Pose, shape.Size + new Vector2(relaxation * 2));
                var contact = PolygonCollisionIntervals(grown, obstacle, inward, -DeploymentFitTolerance);
                IntersectFitIntervals(allowed, contact);
                if (allowed.Count == 0) return allowed;
            }
            var half = shape.HalfSize - new Vector2(relaxation);
            if (half.X <= 0 || half.Y <= 0) return allowed;
            var inner = new OrientedRectangle(shape.Pose, half * 2);
            foreach (var rectangle in rectangles)
                if (CollisionInterval(inner, rectangle, inward, DeploymentFitTolerance, out var forbidden)) SubtractFitInterval(allowed, forbidden);
            foreach (var polygon in obstacles)
                foreach (var forbidden in PolygonCollisionIntervals(inner, polygon, inward, DeploymentFitTolerance)) SubtractFitInterval(allowed, forbidden);
            foreach (var circle in circles)
                if (CircleInterval(inner, circle, inward, out var forbidden)) SubtractFitInterval(allowed, forbidden);
            return allowed;
        }
        private bool ClipContainment(OrientedRectangle outer, OrientedRectangle shape, float relaxation, ref float low, ref float high)
        {
            Span<Vector2> corners = stackalloc Vector2[4]; shape.CopyCorners(corners); var slope = outer.Pose.InverseTransformDirection(inward);
            foreach (var corner in corners)
            {
                var point = outer.Pose.InverseTransformPoint(corner);
                if (!ClipFitLinear(point.X, slope.X, outer.HalfSize.X + PlayAreaGeometry.SourceContainmentTolerance + relaxation, ref low, ref high) ||
                    !ClipFitLinear(point.Y, slope.Y, outer.HalfSize.Y + PlayAreaGeometry.SourceContainmentTolerance + relaxation, ref low, ref high)) return false;
            }
            return true;
        }
        private static bool CollisionInterval(OrientedRectangle moving, OrientedRectangle other, Vector2 direction, float clearance, out DeploymentFitInterval result)
        {
            float low = float.NegativeInfinity, high = float.PositiveInfinity;
            Span<Vector2> axes = stackalloc Vector2[4] { moving.Pose.Right, moving.Pose.Forward, other.Pose.Right, other.Pose.Forward };
            foreach (var axis in axes)
                if (!ClipFitLinear(Vector2.Dot(moving.Center - other.Center, axis), Vector2.Dot(direction, axis), moving.ProjectedRadius(axis) + other.ProjectedRadius(axis) - clearance, ref low, ref high))
                { result = default; return false; }
            result = new(low, high); return true;
        }
        private static bool CircleInterval(OrientedRectangle shape, Circle circle, Vector2 direction, out DeploymentFitInterval result)
        {
            var point = shape.Pose.InverseTransformPoint(circle.Center); var delta = -shape.Pose.InverseTransformDirection(direction);
            float expansion = MathF.Max(0, circle.Radius - DeploymentFitTolerance), minimum = float.PositiveInfinity, maximum = float.NegativeInfinity;
            IncludeBox(shape.HalfSize + new Vector2(expansion, 0)); IncludeBox(shape.HalfSize + new Vector2(0, expansion));
            for (int x = -1; x <= 1; x += 2) for (int y = -1; y <= 1; y += 2)
            {
                var v = point - shape.HalfSize * new Vector2(x, y); float b = Vector2.Dot(v, delta), c = v.LengthSquared() - expansion * expansion, discriminant = b * b - c;
                if (discriminant < 0) continue;
                float root = MathF.Sqrt(MathF.Max(0, discriminant)); minimum = MathF.Min(minimum, -b - root); maximum = MathF.Max(maximum, -b + root);
            }
            result = new(minimum, maximum); return minimum <= maximum;
            void IncludeBox(Vector2 half)
            {
                float low = float.NegativeInfinity, high = float.PositiveInfinity;
                if (!ClipFitLinear(point.X, delta.X, half.X, ref low, ref high) || !ClipFitLinear(point.Y, delta.Y, half.Y, ref low, ref high)) return;
                minimum = MathF.Min(minimum, low); maximum = MathF.Max(maximum, high);
            }
        }
        private static List<DeploymentFitInterval> PolygonCollisionIntervals(OrientedRectangle moving, PolygonFootprint other, Vector2 direction, float clearance)
        {
            var result = new List<DeploymentFitInterval>(); Span<Vector2> triangle = stackalloc Vector2[3];
            for (int index = 0; index < other.TriangleCount; index++)
            {
                other.CopyTriangle(index, triangle); float low = float.NegativeInfinity, high = float.PositiveInfinity; bool intersects = true;
                for (int axisIndex = 0; axisIndex < 5; axisIndex++)
                {
                    var edge = triangle[(axisIndex - 2 + 1 + 3) % 3] - triangle[(axisIndex - 2 + 3) % 3];
                    var axis = axisIndex == 0 ? moving.Pose.Right : axisIndex == 1 ? moving.Pose.Forward : Vector2.Normalize(new Vector2(-edge.Y, edge.X));
                    float minimum = float.PositiveInfinity, maximum = float.NegativeInfinity;
                    foreach (var vertex in triangle) { float projection = Vector2.Dot(vertex, axis); minimum = MathF.Min(minimum, projection); maximum = MathF.Max(maximum, projection); }
                    if (!ClipFitLinear(Vector2.Dot(moving.Center, axis) - (minimum + maximum) / 2, Vector2.Dot(direction, axis),
                        moving.ProjectedRadius(axis) + (maximum - minimum) / 2 - clearance, ref low, ref high)) { intersects = false; break; }
                }
                if (intersects) result.Add(new(low, high));
            }
            return result;
        }
        private static void IntersectFitIntervals(List<DeploymentFitInterval> allowed, List<DeploymentFitInterval> other)
        {
            var result = new List<DeploymentFitInterval>();
            foreach (var interval in allowed) foreach (var contact in other)
            {
                float low = MathF.Max(interval.Low, contact.Low), high = MathF.Min(interval.High, contact.High);
                if (low <= high) result.Add(new(low, high));
            }
            allowed.Clear(); allowed.AddRange(result);
        }
        private static bool ClipFitLinear(float value, float slope, float bound, ref float low, ref float high)
        {
            if (bound < 0) return false;
            if (MathF.Abs(slope) < 1e-8f) return MathF.Abs(value) <= bound;
            float first = (-bound - value) / slope, second = (bound - value) / slope;
            low = MathF.Max(low, MathF.Min(first, second)); high = MathF.Min(high, MathF.Max(first, second)); return low <= high;
        }
        private static void SubtractFitInterval(List<DeploymentFitInterval> allowed, DeploymentFitInterval forbidden)
        {
            for (int i = allowed.Count - 1; i >= 0; i--)
            {
                var item = allowed[i]; if (forbidden.High <= item.Low || forbidden.Low >= item.High) continue;
                allowed.RemoveAt(i);
                if (forbidden.Low >= item.Low) allowed.Insert(i, new(item.Low, forbidden.Low));
                if (forbidden.High <= item.High) allowed.Insert(i, new(forbidden.High, item.High));
            }
        }
    }
}
