using System.Numerics;

namespace Swarmada.Headless.Geometry;

public enum ContactKind { Separated, Touching, Overlapping }

public static partial class Geometry2D
{
    // Recovered GameplayFootprintView uses 0.0001 Unity units, or 0.002285 mm.
    public const float DefaultTolerance = 0.002285f;
    private const float ArithmeticEpsilon = 1e-7f;
    public static float Cross(Vector2 a, Vector2 b) => a.X * b.Y - a.Y * b.X;
    public static void RequireFinite(Vector2 value, string parameterName)
    {
        if (!float.IsFinite(value.X) || !float.IsFinite(value.Y)) throw new ArgumentOutOfRangeException(parameterName);
    }
    internal static void RequireTolerance(float tolerance)
    {
        if (!float.IsFinite(tolerance) || tolerance < 0) throw new ArgumentOutOfRangeException(nameof(tolerance));
    }
    public static Vector2 ClosestPoint(Segment segment, Vector2 point)
    {
        var squared = segment.Delta.LengthSquared();
        return squared <= ArithmeticEpsilon * ArithmeticEpsilon ? segment.Start
            : segment.PointAt(Math.Clamp(Vector2.Dot(point - segment.Start, segment.Delta) / squared, 0, 1));
    }
    public static ClosestPair ClosestPoints(Segment first, Segment second)
    {
        if (TryIntersect(first, second, out var intersection)) return new(0, intersection, intersection);
        var best = new ClosestPair(float.PositiveInfinity, first.Start, second.Start);
        Consider(ref best, first.Start, ClosestPoint(second, first.Start));
        Consider(ref best, first.End, ClosestPoint(second, first.End));
        Consider(ref best, ClosestPoint(first, second.Start), second.Start);
        Consider(ref best, ClosestPoint(first, second.End), second.End);
        return best;
    }
    internal static void Consider(ref ClosestPair best, Vector2 first, Vector2 second)
    {
        var distance = Vector2.Distance(first, second);
        if (distance < best.Distance) best = new(distance, first, second);
    }
    /// <summary>Includes endpoint contact and collinear overlap. Returns a deterministic first point.</summary>
    public static bool TryIntersect(Segment first, Segment second, out Vector2 intersection)
    {
        var r = first.Delta;
        var s = second.Delta;
        var cross = (double)r.X * s.Y - (double)r.Y * s.X;
        var offset = second.Start - first.Start;
        if (Math.Abs(cross) > 1e-12 * Math.Max(1, Math.Sqrt((double)r.LengthSquared() * s.LengthSquared())))
        {
            var t = ((double)offset.X * s.Y - (double)offset.Y * s.X) / cross;
            var u = ((double)offset.X * r.Y - (double)offset.Y * r.X) / cross;
            if (t >= 0 && t <= 1 && u >= 0 && u <= 1)
            {
                intersection = first.PointAt((float)t);
                return true;
            }
        }
        else
        {
            // Endpoint-on-segment tests cover collinear and zero-length segments.
            Span<Vector2> candidates = stackalloc Vector2[4] { first.Start, first.End, second.Start, second.End };
            foreach (var candidate in candidates)
                if (Vector2.DistanceSquared(candidate, ClosestPoint(first, candidate)) < 1e-12f
                    && Vector2.DistanceSquared(candidate, ClosestPoint(second, candidate)) < 1e-12f)
                {
                    intersection = candidate;
                    return true;
                }
        }
        intersection = default;
        return false;
    }
    public static ContactKind Contact(OrientedRectangle first, OrientedRectangle second, float tolerance = DefaultTolerance)
    {
        RequireTolerance(tolerance);
        // The basis is unchanged across the four SAT projections. Cache it here
        // rather than recomputing each pose's sin/cos in ProjectedRadius eight times.
        var firstRight = first.Pose.Right; var firstForward = first.Pose.Forward;
        var secondRight = second.Pose.Right; var secondForward = second.Pose.Forward;
        var firstHalf = first.HalfSize; var secondHalf = second.HalfSize;
        Span<Vector2> axes = stackalloc Vector2[4] { firstRight, firstForward, secondRight, secondForward };
        var delta = second.Center - first.Center;
        var penetration = float.PositiveInfinity;
        foreach (var axis in axes)
        {
            float firstRadius = MathF.Abs(Vector2.Dot(firstRight, axis)) * firstHalf.X
                + MathF.Abs(Vector2.Dot(firstForward, axis)) * firstHalf.Y;
            float secondRadius = MathF.Abs(Vector2.Dot(secondRight, axis)) * secondHalf.X
                + MathF.Abs(Vector2.Dot(secondForward, axis)) * secondHalf.Y;
            var overlap = firstRadius + secondRadius - MathF.Abs(Vector2.Dot(delta, axis));
            if (overlap < -tolerance) return ContactKind.Separated;
            penetration = MathF.Min(penetration, overlap);
        }
        return penetration <= tolerance ? ContactKind.Touching : ContactKind.Overlapping;
    }
    public static ContactKind Contact(Circle first, Circle second, float tolerance = DefaultTolerance)
    {
        RequireTolerance(tolerance);
        var penetration = first.Radius + second.Radius - Vector2.Distance(first.Center, second.Center);
        return penetration < -tolerance ? ContactKind.Separated : penetration <= tolerance ? ContactKind.Touching : ContactKind.Overlapping;
    }
    public static ContactKind Contact(Circle circle, OrientedRectangle rectangle, float tolerance = DefaultTolerance)
    {
        RequireTolerance(tolerance);
        var penetration = circle.Radius - Vector2.Distance(circle.Center, rectangle.ClosestPoint(circle.Center));
        return penetration < -tolerance ? ContactKind.Separated : penetration <= tolerance ? ContactKind.Touching : ContactKind.Overlapping;
    }
    public static ContactKind Contact(ConvexPolygon first, ConvexPolygon second, float tolerance = DefaultTolerance)
        => Contact(first.Vertices, second.Vertices, tolerance);
    public static ContactKind Contact(OrientedRectangle rectangle, ConvexPolygon polygon, float tolerance = DefaultTolerance)
    {
        Span<Vector2> corners = stackalloc Vector2[4];
        rectangle.CopyCorners(corners);
        return Contact(corners, polygon.Vertices, tolerance);
    }
    public static ContactKind Contact(Circle circle, ConvexPolygon polygon, float tolerance = DefaultTolerance)
    {
        RequireTolerance(tolerance);
        var penetration = circle.Radius - Vector2.Distance(circle.Center, polygon.ClosestPoint(circle.Center));
        return penetration < -tolerance ? ContactKind.Separated : penetration <= tolerance ? ContactKind.Touching : ContactKind.Overlapping;
    }
    public static ContactKind Contact(ReadOnlySpan<Vector2> first, ReadOnlySpan<Vector2> second, float tolerance = DefaultTolerance)
    {
        RequireTolerance(tolerance);
        if (first.Length < 3 || second.Length < 3) throw new ArgumentException("Convex polygons need at least three vertices.");
        var minimum = float.PositiveInfinity;
        if (!TestAxes(first, second, first, tolerance, ref minimum)
            || !TestAxes(first, second, second, tolerance, ref minimum)) return ContactKind.Separated;
        return minimum <= tolerance ? ContactKind.Touching : ContactKind.Overlapping;
    }
    private static bool TestAxes(ReadOnlySpan<Vector2> first, ReadOnlySpan<Vector2> second,
        ReadOnlySpan<Vector2> source, float tolerance, ref float minimum)
    {
        for (var i = 0; i < source.Length; i++)
        {
            var edge = source[(i + 1) % source.Length] - source[i];
            if (edge.LengthSquared() < 1e-12f) throw new ArgumentException("Degenerate polygon edge.");
            var axis = Vector2.Normalize(new Vector2(-edge.Y, edge.X));
            Project(first, axis, out var minA, out var maxA);
            Project(second, axis, out var minB, out var maxB);
            var overlap = MathF.Min(maxA - minB, maxB - minA);
            if (overlap < -tolerance) return false;
            minimum = MathF.Min(minimum, overlap);
        }
        return true;
    }
    private static void Project(ReadOnlySpan<Vector2> vertices, Vector2 axis, out float minimum, out float maximum)
    {
        minimum = maximum = Vector2.Dot(vertices[0], axis);
        for (var i = 1; i < vertices.Length; i++)
        {
            var dot = Vector2.Dot(vertices[i], axis);
            minimum = MathF.Min(minimum, dot);
            maximum = MathF.Max(maximum, dot);
        }
    }
    public static bool Overlaps(OrientedRectangle first, OrientedRectangle second, float tolerance = DefaultTolerance)
        => Contact(first, second, tolerance) == ContactKind.Overlapping;
    public static bool Overlaps(Circle first, Circle second, float tolerance = DefaultTolerance)
        => Contact(first, second, tolerance) == ContactKind.Overlapping;
    public static bool Overlaps(Circle circle, OrientedRectangle rectangle, float tolerance = DefaultTolerance)
        => Contact(circle, rectangle, tolerance) == ContactKind.Overlapping;
    public static bool Overlaps(ConvexPolygon first, ConvexPolygon second, float tolerance = DefaultTolerance)
        => Contact(first, second, tolerance) == ContactKind.Overlapping;
    public static bool Overlaps(OrientedRectangle rectangle, ConvexPolygon polygon, float tolerance = DefaultTolerance)
        => Contact(rectangle, polygon, tolerance) == ContactKind.Overlapping;
    public static bool Overlaps(Circle circle, ConvexPolygon polygon, float tolerance = DefaultTolerance)
        => Contact(circle, polygon, tolerance) == ContactKind.Overlapping;
    public static float Distance(Circle first, Circle second) => MathF.Max(0, Vector2.Distance(first.Center, second.Center) - first.Radius - second.Radius);
    public static float Distance(Circle circle, OrientedRectangle rectangle) => MathF.Max(0, Vector2.Distance(circle.Center, rectangle.ClosestPoint(circle.Center)) - circle.Radius);
    public static float Distance(Circle circle, ConvexPolygon polygon) => MathF.Max(0, Vector2.Distance(circle.Center, polygon.ClosestPoint(circle.Center)) - circle.Radius);
    public static float Distance(ConvexPolygon first, ConvexPolygon second) => ClosestPoints(first, second).Distance;
    public static float Distance(OrientedRectangle first, OrientedRectangle second) => ClosestPoints(first, second).Distance;
    public static ClosestPair ClosestPoints(OrientedRectangle first, OrientedRectangle second)
    {
        Span<Vector2> a = stackalloc Vector2[4];
        Span<Vector2> b = stackalloc Vector2[4];
        first.CopyCorners(a);
        second.CopyCorners(b);
        if (Contact(first, second, 0) != ContactKind.Separated)
        {
            // A zero distance witness may be on any shared point, not necessarily a perimeter intersection.
            for (var i = 0; i < 4; i++)
            {
                if (second.Contains(a[i], 0)) return new(0, a[i], a[i]);
                if (first.Contains(b[i], 0)) return new(0, b[i], b[i]);
            }
        }
        return ClosestPerimeters(a, b);
    }
    public static ClosestPair ClosestPoints(ConvexPolygon first, ConvexPolygon second)
    {
        foreach (var v in first.Vertices) if (second.Contains(v, 0)) return new(0, v, v);
        foreach (var v in second.Vertices) if (first.Contains(v, 0)) return new(0, v, v);
        return ClosestPerimeters(first.Vertices, second.Vertices);
    }
    private static ClosestPair ClosestPerimeters(ReadOnlySpan<Vector2> first, ReadOnlySpan<Vector2> second)
    {
        var best = new ClosestPair(float.PositiveInfinity, first[0], second[0]);
        for (var i = 0; i < first.Length; i++)
            for (var j = 0; j < second.Length; j++)
            {
                var pair = ClosestPoints(new Segment(first[i], first[(i + 1) % first.Length]), new Segment(second[j], second[(j + 1) % second.Length]));
                if (pair.Distance < best.Distance) best = pair;
                if (best.Distance == 0) return best;
            }
        return best;
    }
    /// <summary>Liang-Barsky clipping. Parameters are measured on the original segment; tangent contact succeeds.</summary>
    public static bool Clip(Segment segment, OrientedRectangle rectangle, out float enter, out float exit, float inset = 0)
    {
        RequireTolerance(inset);
        var h = rectangle.HalfSize - new Vector2(inset);
        enter = 0;
        exit = 1;
        if (h.X < 0 || h.Y < 0) return false;
        var start = rectangle.Pose.InverseTransformPoint(segment.Start);
        var delta = rectangle.Pose.InverseTransformDirection(segment.Delta);
        return ClipPositive(start.X + h.X, delta.X, ref enter, ref exit)
            && ClipPositive(h.X - start.X, -delta.X, ref enter, ref exit)
            && ClipPositive(start.Y + h.Y, delta.Y, ref enter, ref exit)
            && ClipPositive(h.Y - start.Y, -delta.Y, ref enter, ref exit);
    }
    public static bool Clip(Segment segment, ConvexPolygon polygon, out float enter, out float exit, float inset = 0)
    {
        RequireTolerance(inset);
        enter = 0;
        exit = 1;
        var vertices = polygon.Vertices;
        for (var i = 0; i < vertices.Length; i++)
        {
            var edge = vertices[(i + 1) % vertices.Length] - vertices[i];
            var value = Cross(edge, segment.Start - vertices[i]) - inset * edge.Length();
            if (!ClipPositive(value, Cross(edge, segment.Delta), ref enter, ref exit)) return false;
        }
        return true;
    }
    internal static bool ClipPositive(float value, float delta, ref float enter, ref float exit)
    {
        if (MathF.Abs(delta) <= ArithmeticEpsilon) return value >= 0;
        var parameter = -value / delta;
        if (delta > 0) enter = MathF.Max(enter, parameter); else exit = MathF.Min(exit, parameter);
        return enter <= exit;
    }
    public static bool SegmentCrossesInterior(Segment segment, OrientedRectangle rectangle, float tolerance = DefaultTolerance)
        => segment.Length > tolerance && Clip(segment, rectangle, out var enter, out var exit, tolerance) && (exit - enter) * segment.Length > tolerance;
    public static bool SegmentCrossesInterior(Segment segment, ConvexPolygon polygon, float tolerance = DefaultTolerance)
        => segment.Length > tolerance && Clip(segment, polygon, out var enter, out var exit, tolerance) && (exit - enter) * segment.Length > tolerance;
    public static bool SegmentCrossesInterior(Segment segment, Circle circle, float tolerance = DefaultTolerance)
    {
        RequireTolerance(tolerance);
        var radius = circle.Radius - tolerance;
        return segment.Length > tolerance && radius > 0 && Vector2.DistanceSquared(circle.Center, ClosestPoint(segment, circle.Center)) < radius * radius;
    }
    /// <summary>Remove only portions of the LOS inside its own endpoint tokens. Third-party obstruction
    /// is tested against the returned segment. Token overlap returns an empty segment intentionally.</summary>
    public static Segment TrimEndpointTokens(Segment line, OrientedRectangle sourceToken, OrientedRectangle targetToken)
    {
        var start = 0f;
        var end = 1f;
        if (sourceToken.Contains(line.Start, 0) && Clip(line, sourceToken, out _, out var sourceExit)) start = sourceExit;
        if (targetToken.Contains(line.End, 0) && Clip(line, targetToken, out var targetEnter, out _)) end = targetEnter;
        if (end < start) end = start;
        return new Segment(line.PointAt(start), line.PointAt(end));
    }
}
