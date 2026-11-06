using System.Numerics;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

public static partial class OfficialObjectives
{
    private const float ObstacleRangeTolerance = Geometry2D.DefaultTolerance / 4;

    private readonly record struct PhysicalRuler(Vector2 Origin, Vector2 Direction, float Low, float High)
    {
        public Vector2 Normal => new(-Direction.Y, Direction.X);
    }

    /// <summary>Geometry for checking one caller-supplied destination. This context
    /// never searches, compares destinations, or changes the supplied position.</summary>
    private sealed class PhysicalObstacleMoveContext
    {
        public readonly PolygonFootprint Start;
        public readonly bool Flipped;
        public readonly List<PhysicalRuler> Rulers = [];
        private readonly float maximum;
        private readonly OrientedRectangle board;
        private readonly List<PolygonFootprint> polygons = [];
        private readonly List<OrientedRectangle> rectangles = [];
        private readonly List<Circle> circles = [], originalCarried = [], movingCarried = [];

        public PhysicalObstacleMoveContext(GameContext game, int index, float maximum, int toward, bool? flipped, bool allowSquadronOverlap)
        {
            Start = ObstacleGeometry.Shape(game, index); Flipped = flipped ?? Start.Flipped; this.maximum = maximum;
            board = MovementModule.Board(game);
            foreach (var pair in game.Flow.TokenHosts)
                if (game.State.SpatialObjects[pair.Key].Active && Array.IndexOf(pair.Value, index) >= 0)
                {
                    var circle = PieceCircle(game, pair.Key);
                    if (Geometry2D.Contains(Start, circle, 0)) continue;
                    originalCarried.Add(circle);
                    var local = Start.Pose.InverseTransformPoint(circle.Center);
                    if (Flipped != Start.Flipped) local.X = -local.X;
                    movingCarried.Add(new(local, circle.Radius));
                }
            if (toward >= 0) BuildRulers(game, toward);
            int stationEntity = State(game)?.PieceEntities.GetValueOrDefault(index, -1) ?? -1;
            foreach (var ship in game.Topology.Ships)
                if (!ship.IsStation && ship.EntityIndex != stationEntity && Live(game, ship.EntityIndex)) rectangles.Add(game.ShipBase(ship.EntityIndex));
            if (!allowSquadronOverlap)
                foreach (var squad in game.Topology.Squadrons)
                    if (Live(game, squad.EntityIndex)) circles.Add(game.SquadronBase(squad.EntityIndex));
            for (int i = 0; i < game.State.SpatialObjects.Count; i++)
            {
                var piece = game.State.SpatialObjects[i]; if (!piece.Active || i == index) continue;
                if (piece.Kind == 0) polygons.Add(ObstacleGeometry.Shape(game, i));
                else if (TokenObstacleAttachments.IsTokenKind(piece.Kind) && !TokenObstacleAttachments.IsCarriedBy(game, i, index)) circles.Add(PieceCircle(game, i));
            }
        }

        private void BuildRulers(GameContext game, int toward)
        {
            var target = game.State.SpatialObjects[toward]; var targetCircles = new List<Circle>();
            float best = float.PositiveInfinity; var candidates = new List<(float Distance, PhysicalRuler Ruler)>();
            PolygonFootprint? targetPolygon = target.Kind == 0 ? ObstacleGeometry.Shape(game, toward) : null;
            if (targetPolygon is null) targetCircles.Add(PieceCircle(game, toward));
            else
                foreach (var pair in game.Flow.TokenHosts)
                    if (game.State.SpatialObjects[pair.Key].Active && Array.IndexOf(pair.Value, toward) >= 0)
                    { var circle = PieceCircle(game, pair.Key); if (!Geometry2D.Contains(targetPolygon.Value, circle, 0)) targetCircles.Add(circle); }
            if (targetPolygon is { } polygon)
            {
                EdgePairs(polygon, 0, 0);
                foreach (var circle in originalCarried) for (int j = 0; j < polygon.Count; j++)
                { var second = Geometry2D.ClosestPoint(polygon.Edge(j), circle.Center); Consider(CircleToward(circle, second), second); }
            }
            foreach (var targetCircle in targetCircles)
            {
                for (int i = 0; i < Start.Count; i++)
                { var first = Geometry2D.ClosestPoint(Start.Edge(i), targetCircle.Center); Consider(first, CircleToward(targetCircle, first)); }
                foreach (var sourceCircle in originalCarried)
                {
                    var second = CircleToward(targetCircle, sourceCircle.Center); var first = CircleToward(sourceCircle, second);
                    Consider(first, second);
                }
            }
            foreach (var candidate in candidates)
                if (candidate.Distance <= best + Geometry2D.DefaultTolerance && !Rulers.Any(r => Equivalent(r, candidate.Ruler))) Rulers.Add(candidate.Ruler);
            if (Rulers.Count == 0) throw new ArgumentException("Physical outlines have no closest ruler.");

            void EdgePairs(PolygonFootprint targetShape, int sourceNode, int targetNode)
            {
                var a = Start.Definition.EdgeNode(sourceNode); var b = targetShape.Definition.EdgeNode(targetNode);
                Start.NodeWorldBounds(a, out var amin, out var amax); targetShape.NodeWorldBounds(b, out var bmin, out var bmax);
                var gap = Vector2.Max(Vector2.Zero, Vector2.Max(amin - bmax, bmin - amax));
                if (gap.LengthSquared() > (best + Geometry2D.DefaultTolerance) * (best + Geometry2D.DefaultTolerance)) return;
                if (a.IsLeaf && b.IsLeaf)
                {
                    for (int i = 0; i < a.Count; i++) for (int j = 0; j < b.Count; j++)
                    {
                        var first = Start.Edge(Start.Definition.EdgeIndex(a.First + i)); var second = targetShape.Edge(targetShape.Definition.EdgeIndex(b.First + j));
                        var pair = Geometry2D.ClosestPoints(first, second); Consider(pair.First, pair.Second, first, second);
                    }
                }
                else if (b.IsLeaf || !a.IsLeaf && (amax - amin).LengthSquared() >= (bmax - bmin).LengthSquared())
                { EdgePairs(targetShape, a.Left, targetNode); EdgePairs(targetShape, a.Right, targetNode); }
                else { EdgePairs(targetShape, sourceNode, b.Left); EdgePairs(targetShape, sourceNode, b.Right); }
            }

            void Consider(Vector2 first, Vector2 second, Segment? sourceEdge = null, Segment? targetEdge = null)
            {
                float distance = Vector2.Distance(first, second); if (distance > best + Geometry2D.DefaultTolerance) return;
                if (distance < best - Geometry2D.DefaultTolerance) candidates.Clear(); best = MathF.Min(best, distance);
                var delta = second - first;
                var direction = delta.LengthSquared() > 1e-10f ? Vector2.Normalize(delta) : Vector2.Normalize(target.Pose.Position - Start.Center);
                if (!float.IsFinite(direction.X)) direction = Vector2.UnitX;
                var normal = new Vector2(-direction.Y, direction.X); float low = 0, high = 0;
                if (sourceEdge is { } a && targetEdge is { } b &&
                    MathF.Abs(Vector2.Dot(a.Delta, direction)) <= Geometry2D.DefaultTolerance && MathF.Abs(Vector2.Dot(b.Delta, direction)) <= Geometry2D.DefaultTolerance)
                {
                    float a0 = Vector2.Dot(a.Start - first, normal), a1 = Vector2.Dot(a.End - first, normal);
                    float b0 = Vector2.Dot(b.Start - first, normal), b1 = Vector2.Dot(b.End - first, normal);
                    low = MathF.Max(MathF.Min(a0, a1), MathF.Min(b0, b1)); high = MathF.Min(MathF.Max(a0, a1), MathF.Max(b0, b1));
                    if (low > high) low = high = 0;
                }
                candidates.Add((distance, new(first, direction, low, high)));
            }
        }

        private static bool Equivalent(PhysicalRuler a, PhysicalRuler b) => Vector2.Distance(a.Direction, b.Direction) < 1e-6f &&
            Vector2.Distance(a.Origin + a.Normal * a.Low, b.Origin + b.Normal * b.Low) <= Geometry2D.DefaultTolerance &&
            Vector2.Distance(a.Origin + a.Normal * a.High, b.Origin + b.Normal * b.High) <= Geometry2D.DefaultTolerance;
        private static Vector2 CircleToward(Circle circle, Vector2 point)
        { var delta = point - circle.Center; float length = delta.Length(); return length <= circle.Radius ? point : circle.Center + delta * (circle.Radius / length); }
        private Circle MovingCircle(Circle local, Pose pose) => new(pose.TransformPoint(local.Center), local.Radius);

        public float Progress(PhysicalRuler ruler, Pose pose)
        {
            var shape = Start.At(pose, Flipped); float highest = float.NegativeInfinity;
            for (int i = 0; i < shape.Count; i++) highest = MathF.Max(highest, Vector2.Dot(shape.Vertex(i) - ruler.Origin, ruler.Direction));
            foreach (var local in movingCarried)
            {
                var circle = MovingCircle(local, pose);
                highest = MathF.Max(highest, Vector2.Dot(circle.Center - ruler.Origin, ruler.Direction) + circle.Radius);
            }
            return highest;
        }

        public bool Valid(Pose pose, PhysicalRuler? ruler, out Vector2? rangeViolation)
        {
            rangeViolation = null; var shape = Start.At(pose, Flipped);
            if (!PlayAreaGeometry.Contains(board, shape) || ruler is { } line && Progress(line, pose) > maximum + ObstacleMoveTolerance) return false;
            foreach (var rectangle in rectangles) if (Geometry2D.Overlaps(rectangle, shape)) return false;
            foreach (var polygon in polygons) if (Geometry2D.Overlaps(shape, polygon)) return false;
            foreach (var circle in circles) if (Geometry2D.Overlaps(circle, shape)) return false;
            if (!PolygonWithinRange(shape, out var violation)) { rangeViolation = violation; return false; }
            foreach (var local in movingCarried)
            {
                var circle = MovingCircle(local, pose);
                if (!PlayAreaGeometry.Contains(board, circle)) return false;
                foreach (var rectangle in rectangles) if (Geometry2D.Overlaps(circle, rectangle)) return false;
                foreach (var polygon in polygons) if (Geometry2D.Overlaps(circle, polygon)) return false;
                foreach (var other in circles) if (Geometry2D.Overlaps(circle, other)) return false;
                if (!CircleWithinRange(circle, out violation)) { rangeViolation = violation; return false; }
            }
            return true;
        }

        private float OriginalDistance(Vector2 point)
        {
            float distance = Vector2.Distance(point, Start.ClosestPoint(point));
            foreach (var circle in originalCarried) distance = MathF.Min(distance, MathF.Max(0, Vector2.Distance(point, circle.Center) - circle.Radius));
            return distance;
        }
        private bool PolygonWithinRange(PolygonFootprint shape, out Vector2 violation)
        {
            violation = default; float limit = maximum + ObstacleMoveTolerance; bool displacementBound = true;
            for (int i = 0; i < shape.Count; i++) if (Vector2.Distance(shape.Vertex(i), Start.Vertex(i)) > limit) { displacementBound = false; break; }
            if (displacementBound) return true;
            // An interior disk is a guaranteed subset of the original contour.
            // Its distance-expanded disk is convex: enclosing every destination
            // vertex therefore certifies the entire concave footprint at once.
            // This only accelerates a successful proof; ambiguous cases retain the
            // exact original-polygon certification below.
            var interior = Start.InteriorCircle; float coverRadius = limit + interior.Radius;
            bool covered = float.IsFinite(coverRadius);
            for (int i = 0; covered && i < shape.Count; i++)
                if (Vector2.Distance(shape.Vertex(i), interior.Center) > coverRadius) covered = false;
            if (covered) return true;
            Span<Vector2> triangle = stackalloc Vector2[3]; var pending = new Stack<(Vector2 A, Vector2 B, Vector2 C)>();
            for (int i = 0; i < shape.TriangleCount; i++) { shape.CopyTriangle(i, triangle); pending.Push((triangle[0], triangle[1], triangle[2])); }
            while (pending.TryPop(out var cell))
            {
                float da = OriginalDistance(cell.A), db = OriginalDistance(cell.B), dc = OriginalDistance(cell.C);
                if (da > limit) { violation = cell.A; return false; } if (db > limit) { violation = cell.B; return false; } if (dc > limit) { violation = cell.C; return false; }
                var center = (cell.A + cell.B + cell.C) / 3; float d = OriginalDistance(center);
                if (d > limit) { violation = center; return false; }
                float r = MathF.Max(Vector2.Distance(center, cell.A), MathF.Max(Vector2.Distance(center, cell.B), Vector2.Distance(center, cell.C)));
                if (d + r <= limit || r <= ObstacleRangeTolerance) continue;
                float ab = Vector2.DistanceSquared(cell.A, cell.B), bc = Vector2.DistanceSquared(cell.B, cell.C), ca = Vector2.DistanceSquared(cell.C, cell.A);
                if (ab >= bc && ab >= ca) { var mid = (cell.A + cell.B) / 2; pending.Push((cell.A, mid, cell.C)); pending.Push((mid, cell.B, cell.C)); }
                else if (bc >= ca) { var mid = (cell.B + cell.C) / 2; pending.Push((cell.A, cell.B, mid)); pending.Push((cell.A, mid, cell.C)); }
                else { var mid = (cell.C + cell.A) / 2; pending.Push((cell.A, cell.B, mid)); pending.Push((mid, cell.B, cell.C)); }
            }
            return true;
        }
        private bool CircleWithinRange(Circle circle, out Vector2 violation)
        {
            violation = default; float limit = maximum + ObstacleMoveTolerance;
            var interior = Start.InteriorCircle;
            float coverRadius = limit + interior.Radius;
            if (float.IsFinite(coverRadius) && Vector2.Distance(circle.Center, interior.Center) + circle.Radius <= coverRadius) return true;
            foreach (var original in originalCarried) if (Vector2.Distance(circle.Center, original.Center) + circle.Radius - original.Radius <= limit) return true;
            var arcs = new Stack<(float Angle, float Half)>(); arcs.Push((0, MathF.PI));
            while (arcs.TryPop(out var arc))
            {
                var point = circle.Center + circle.Radius * new Vector2(MathF.Cos(arc.Angle), MathF.Sin(arc.Angle)); float distance = OriginalDistance(point);
                if (distance > limit) { violation = point; return false; }
                float error = circle.Radius * 2 * MathF.Sin(arc.Half / 2); if (distance + error <= limit || error <= ObstacleRangeTolerance) continue;
                arcs.Push((arc.Angle - arc.Half / 2, arc.Half / 2)); arcs.Push((arc.Angle + arc.Half / 2, arc.Half / 2));
            }
            return true;
        }

    }
}
