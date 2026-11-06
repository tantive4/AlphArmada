using System.Numerics;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

/// <summary>Continuous overlap placement. Pending bases are lifted; contact feasibility is
/// solved on line/circle boundaries rather than a policy grid or angular sampling.</summary>
public static class SquadronDisplacementGeometry
{
    private const float Tolerance = Geometry2D.DefaultTolerance;
    public static bool IsLegalPosition(GameContext game, int squadron, Vector2 position, IReadOnlyList<int> pending)
    {
        if (!float.IsFinite(position.X) || !float.IsFinite(position.Y)) return false;
        var circle = new Circle(position, game.SquadronBase(squadron).Radius);
        if (!PlayAreaGeometry.Contains(MovementModule.Board(game), circle)) return false;
        foreach (var ship in game.Topology.Ships)
            if (!ship.IsStation && OfficialObjectives.Live(game, ship.EntityIndex) && Geometry2D.Overlaps(circle, game.ShipBase(ship.EntityIndex))) return false;
        foreach (var squad in game.Topology.Squadrons)
            if (squad.EntityIndex != squadron && !pending.Contains(squad.EntityIndex) && OfficialObjectives.Live(game, squad.EntityIndex)
                && Geometry2D.Overlaps(circle, game.SquadronBase(squad.EntityIndex))) return false;
        return true;
    }
    public static bool Touches(Circle squadron, OrientedRectangle piece, bool flatObstacle = false)
        => flatObstacle ? Geometry2D.Distance(squadron, piece) <= Tolerance
            : Geometry2D.Contact(squadron, piece) == ContactKind.Touching;

    public static bool Touches(Circle squadron, PolygonFootprint piece, bool flatObstacle = false)
        => flatObstacle ? Geometry2D.Distance(squadron, piece) <= Tolerance : Geometry2D.Contact(squadron, piece) == ContactKind.Touching;

    public static bool CanPlaceTouching(GameContext game, int squadron, PolygonFootprint piece, IReadOnlyList<int> pending, bool flatObstacle = false)
    {
        float radius = game.SquadronBase(squadron).Radius;
        var contact = RoundedPolygon(piece, radius); var blockers = new List<Boundary>();
        var board = MovementModule.Board(game); AddRectangle(blockers, new(board.Pose, board.Size - new Vector2(radius * 2)));
        foreach (var ship in game.Topology.Ships)
            if (!ship.IsStation && OfficialObjectives.Live(game, ship.EntityIndex)) blockers.AddRange(RoundedRectangle(game.ShipBase(ship.EntityIndex), radius));
        foreach (var squad in game.Topology.Squadrons)
            if (squad.EntityIndex != squadron && !pending.Contains(squad.EntityIndex) && OfficialObjectives.Live(game, squad.EntityIndex))
                blockers.Add(Boundary.Arc(game.SquadronBase(squad.EntityIndex).Center, radius + game.SquadronBase(squad.EntityIndex).Radius, 0, MathF.Tau));
        bool Legal(Vector2 point) => Touches(new(point, radius), piece, flatObstacle) && IsLegalPosition(game, squadron, point, pending);
        if (flatObstacle)
            for (int i = 0; i < piece.Count; i++) if (Legal(piece.Vertex(i))) return true;
        var all = contact.Concat(blockers).ToArray(); var curves = flatObstacle ? all : contact;
        foreach (var curve in curves)
        {
            var cuts = new List<float> { 0, 1 };
            foreach (var boundary in all) Intersections(curve, boundary, cuts);
            cuts.Sort();
            for (int i = 0; i < cuts.Count; i++)
            {
                if (Legal(curve.At(cuts[i]))) return true;
                if (i > 0 && cuts[i] - cuts[i - 1] > 1e-7f && Legal(curve.At((cuts[i] + cuts[i - 1]) * .5f))) return true;
            }
        }
        return false;
    }

    public static Vector2 TouchingPosition(PolygonFootprint piece, Circle squadron, Vector2 desired)
    {
        var nearest = piece.ClosestBoundaryPoint(desired); var direction = desired - nearest;
        if (direction.LengthSquared() > 1e-10f)
        { direction = Vector2.Normalize(direction); if (piece.Contains(desired, 0)) direction = -direction; }
        else
        {
            direction = Vector2.UnitX;
            for (int i = 0; i < piece.Count; i++)
                if (Vector2.DistanceSquared(Geometry2D.ClosestPoint(piece.Edge(i), desired), desired) < 1e-8f)
                { var edge = piece.Edge(i).Delta; direction = Vector2.Normalize(new Vector2(edge.Y, -edge.X)) * (piece.Flipped ? -1 : 1); break; }
        }
        return nearest + direction * squadron.Radius;
    }

    private static Boundary[] RoundedPolygon(PolygonFootprint polygon, float radius)
    {
        var result = new List<Boundary>(polygon.Count * 2);
        for (int i = 0; i < polygon.Count; i++)
        {
            var edge = polygon.Edge(i); var outward = Vector2.Normalize(new Vector2(edge.Delta.Y, -edge.Delta.X)) * (polygon.Flipped ? -1 : 1);
            result.Add(Boundary.Line(edge.Start + outward * radius, edge.End + outward * radius));
            result.Add(Boundary.Arc(edge.Start, radius, 0, MathF.Tau));
        }
        return result.ToArray();
    }

    public static bool CanPlaceTouching(GameContext game, int squadron, OrientedRectangle piece, IReadOnlyList<int> pending, bool flatObstacle = false)
    {
        float radius = game.SquadronBase(squadron).Radius;
        var contact = RoundedRectangle(piece, radius);
        var blockers = new List<Boundary>();
        var board = MovementModule.Board(game);
        AddRectangle(blockers, new(board.Pose, board.Size - new Vector2(radius * 2)));
        foreach (var ship in game.Topology.Ships)
            if (!ship.IsStation && OfficialObjectives.Live(game, ship.EntityIndex)) blockers.AddRange(RoundedRectangle(game.ShipBase(ship.EntityIndex), radius));
        foreach (var squad in game.Topology.Squadrons)
            if (squad.EntityIndex != squadron && !pending.Contains(squad.EntityIndex) && OfficialObjectives.Live(game, squad.EntityIndex))
                blockers.Add(Boundary.Arc(game.SquadronBase(squad.EntityIndex).Center, radius + game.SquadronBase(squad.EntityIndex).Radius, 0, MathF.Tau));
        bool Legal(Vector2 point) => Touches(new(point, radius), piece, flatObstacle) && IsLegalPosition(game, squadron, point, pending);
        if (flatObstacle && Legal(piece.Center)) return true;
        // A legal interval changes only at a boundary intersection. Test each endpoint and
        // the interior of every interval. Flat obstacles also allow centres inside the token.
        var curves = flatObstacle ? contact.Concat(blockers).ToArray() : contact;
        var all = contact.Concat(blockers).ToArray();
        foreach (var curve in curves)
        {
            var cuts = new List<float> { 0, 1 };
            foreach (var boundary in all) Intersections(curve, boundary, cuts);
            cuts.Sort();
            for (int i = 0; i < cuts.Count; i++)
            {
                if (Legal(curve.At(cuts[i]))) return true;
                if (i > 0 && cuts[i] - cuts[i - 1] > 1e-7f && Legal(curve.At((cuts[i] + cuts[i - 1]) * .5f))) return true;
            }
        }
        return false;
    }
    /// <summary>Nearest exact base contact to a requested centre, including rounded corners.</summary>
    public static Vector2 TouchingPosition(OrientedRectangle piece, Circle squadron, Vector2 desired)
    {
        var local = piece.Pose.InverseTransformPoint(desired); var half = piece.HalfSize;
        var nearest = Vector2.Clamp(local, -half, half); var direction = local - nearest;
        if (direction.LengthSquared() <= 1e-10f)
        {
            float x = half.X - MathF.Abs(local.X), y = half.Y - MathF.Abs(local.Y);
            if (x < y) { direction = new(local.X < 0 ? -1 : 1, 0); nearest.X = direction.X * half.X; }
            else { direction = new(0, local.Y < 0 ? -1 : 1); nearest.Y = direction.Y * half.Y; }
        }
        else direction = Vector2.Normalize(direction);
        return piece.Pose.TransformPoint(nearest + direction * squadron.Radius);
    }
    private readonly record struct Boundary(Vector2 Start, Vector2 End, Vector2 Centre, float Radius, float Angle, float Sweep)
    {
        public bool IsArc => Radius > 0;
        public static Boundary Line(Vector2 start, Vector2 end) => new(start, end, default, 0, 0, 0);
        public static Boundary Arc(Vector2 centre, float radius, float angle, float sweep) => new(default, default, centre, radius, angle, sweep);
        public Vector2 At(float t) => IsArc ? Centre + Radius * new Vector2(MathF.Cos(Angle + Sweep * t), MathF.Sin(Angle + Sweep * t)) : Vector2.Lerp(Start, End, t);
        public void AddCut(Vector2 point, List<float> cuts)
        {
            float t;
            if (IsArc)
            {
                float angle = MathF.Atan2(point.Y - Centre.Y, point.X - Centre.X) - Angle;
                angle %= MathF.Tau; if (angle < 0) angle += MathF.Tau;
                t = angle / Sweep;
                if (t > 1 + 1e-5f) return;
            }
            else
            {
                var delta = End - Start; float length = delta.LengthSquared();
                if (length < 1e-10f) return;
                t = Vector2.Dot(point - Start, delta) / length;
                if (t < -1e-5f || t > 1 + 1e-5f) return;
            }
            cuts.Add(Math.Clamp(t, 0, 1));
        }
    }
    private static Boundary[] RoundedRectangle(OrientedRectangle rectangle, float radius)
    {
        var result = new List<Boundary>(8); var h = rectangle.HalfSize; var pose = rectangle.Pose;
        result.Add(Boundary.Line(pose.TransformPoint(new(h.X + radius, -h.Y)), pose.TransformPoint(new(h.X + radius, h.Y))));
        result.Add(Boundary.Line(pose.TransformPoint(new(-h.X, h.Y + radius)), pose.TransformPoint(new(h.X, h.Y + radius))));
        result.Add(Boundary.Line(pose.TransformPoint(new(-h.X - radius, -h.Y)), pose.TransformPoint(new(-h.X - radius, h.Y))));
        result.Add(Boundary.Line(pose.TransformPoint(new(-h.X, -h.Y - radius)), pose.TransformPoint(new(h.X, -h.Y - radius))));
        float rotation = MathF.Atan2(pose.Right.Y, pose.Right.X);
        result.Add(Boundary.Arc(pose.TransformPoint(new(h.X, h.Y)), radius, rotation, MathF.PI / 2));
        result.Add(Boundary.Arc(pose.TransformPoint(new(-h.X, h.Y)), radius, rotation + MathF.PI / 2, MathF.PI / 2));
        result.Add(Boundary.Arc(pose.TransformPoint(new(-h.X, -h.Y)), radius, rotation + MathF.PI, MathF.PI / 2));
        result.Add(Boundary.Arc(pose.TransformPoint(new(h.X, -h.Y)), radius, rotation + MathF.PI * 1.5f, MathF.PI / 2));
        return result.ToArray();
    }
    private static void AddRectangle(List<Boundary> result, OrientedRectangle rectangle)
    {
        Span<Vector2> corners = stackalloc Vector2[4]; rectangle.CopyCorners(corners);
        for (int i = 0; i < 4; i++) result.Add(Boundary.Line(corners[i], corners[(i + 1) % 4]));
    }
    private static void Intersections(Boundary first, Boundary second, List<float> cuts)
    {
        if (!first.IsArc && !second.IsArc)
        {
            if (Geometry2D.TryIntersect(new(first.Start, first.End), new(second.Start, second.End), out var point)) first.AddCut(point, cuts);
            // Collinear curves may share an interval rather than a single intersection.
            var delta = first.End - first.Start;
            if (MathF.Abs(Geometry2D.Cross(delta, second.Start - first.Start)) < 1e-3f)
            { first.AddCut(second.Start, cuts); first.AddCut(second.End, cuts); }
            return;
        }
        if (!first.IsArc || !second.IsArc)
        {
            var line = first.IsArc ? second : first; var arc = first.IsArc ? first : second;
            var delta = line.End - line.Start; var offset = line.Start - arc.Centre;
            double a = delta.LengthSquared(), b = 2 * Vector2.Dot(offset, delta), c = offset.LengthSquared() - arc.Radius * arc.Radius;
            double determinant = b * b - 4 * a * c;
            if (a < 1e-12 || determinant < -1e-5) return;
            double root = Math.Sqrt(Math.Max(0, determinant));
            foreach (double t in new[] { (-b - root) / (2 * a), (-b + root) / (2 * a) })
                if (t >= -1e-6 && t <= 1 + 1e-6) first.AddCut(line.At((float)Math.Clamp(t, 0, 1)), cuts);
            return;
        }
        var between = second.Centre - first.Centre; double distance = between.Length();
        if (distance < 1e-7)
        {
            if (Math.Abs(first.Radius - second.Radius) < 1e-4)
            { first.AddCut(second.At(0), cuts); first.AddCut(second.At(1), cuts); }
            return;
        }
        if (distance > first.Radius + second.Radius + 1e-5 || distance < Math.Abs(first.Radius - second.Radius) - 1e-5) return;
        double along = (first.Radius * first.Radius - second.Radius * second.Radius + distance * distance) / (2 * distance);
        double height = Math.Sqrt(Math.Max(0, first.Radius * first.Radius - along * along));
        var unit = between / (float)distance; var middle = first.Centre + unit * (float)along; var normal = new Vector2(-unit.Y, unit.X) * (float)height;
        first.AddCut(middle + normal, cuts); first.AddCut(middle - normal, cuts);
    }
}
