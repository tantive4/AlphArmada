using System.Numerics;
using System.Text.Json.Serialization;

namespace Swarmada.Headless.Geometry;

/// <summary>All geometry uses millimetres. X is right, Y is forward; yaw increases clockwise,
/// matching the source game's Unity X/Z plane rather than mathematical rotation.</summary>
public readonly record struct Pose
{
    public Vector2 Position { get; }
    public float YawDegrees { get; }
    [JsonConstructor]
    public Pose(Vector2 position, float yawDegrees = 0)
    {
        Geometry2D.RequireFinite(position, nameof(position));
        if (!float.IsFinite(yawDegrees)) throw new ArgumentOutOfRangeException(nameof(yawDegrees));
        Position = position;
        YawDegrees = NormalizeDegrees(yawDegrees);
    }
    public Vector2 Right => TransformDirection(Vector2.UnitX);
    public Vector2 Forward => TransformDirection(Vector2.UnitY);
    public Vector2 TransformDirection(Vector2 local)
    {
        var radians = YawDegrees * (MathF.PI / 180);
        var c = MathF.Cos(radians);
        var s = MathF.Sin(radians);
        return new Vector2(c * local.X + s * local.Y, -s * local.X + c * local.Y);
    }
    public Vector2 TransformPoint(Vector2 local) => Position + TransformDirection(local);
    public Vector2 InverseTransformDirection(Vector2 world)
    {
        var radians = YawDegrees * (MathF.PI / 180);
        var c = MathF.Cos(radians);
        var s = MathF.Sin(radians);
        return new Vector2(c * world.X - s * world.Y, s * world.X + c * world.Y);
    }
    public Vector2 InverseTransformPoint(Vector2 world) => InverseTransformDirection(world - Position);
    public static float NormalizeDegrees(float degrees)
    {
        var value = degrees % 360;
        return value < 0 ? value + 360 : value;
    }
}

public readonly record struct Segment(Vector2 Start, Vector2 End)
{
    public Vector2 Delta => End - Start;
    public float Length => Vector2.Distance(Start, End);
    public Vector2 PointAt(float parameter) => Vector2.Lerp(Start, End, parameter);
    public Segment Transform(Pose pose) => new(pose.TransformPoint(Start), pose.TransformPoint(End));
}

public readonly record struct OrientedRectangle
{
    public Pose Pose { get; }
    public Vector2 Size { get; }
    public Vector2 HalfSize => Size * 0.5f;
    public Vector2 Center => Pose.Position;
    [JsonConstructor]
    public OrientedRectangle(Pose pose, Vector2 size)
    {
        Geometry2D.RequireFinite(size, nameof(size));
        if (size.X <= 0 || size.Y <= 0) throw new ArgumentOutOfRangeException(nameof(size));
        Pose = pose;
        Size = size;
    }
    /// <summary>Corners in counterclockwise order; destination must have length at least four.</summary>
    public void CopyCorners(Span<Vector2> destination)
    {
        if (destination.Length < 4) throw new ArgumentException("Four corner slots are required.", nameof(destination));
        var h = HalfSize;
        destination[0] = Pose.TransformPoint(new(-h.X, -h.Y));
        destination[1] = Pose.TransformPoint(new(h.X, -h.Y));
        destination[2] = Pose.TransformPoint(new(h.X, h.Y));
        destination[3] = Pose.TransformPoint(new(-h.X, h.Y));
    }
    public Vector2 ClosestPoint(Vector2 point)
    {
        var local = Pose.InverseTransformPoint(point);
        return Pose.TransformPoint(Vector2.Clamp(local, -HalfSize, HalfSize));
    }
    public Vector2 ClosestBoundaryPoint(Vector2 point)
    {
        var local = Pose.InverseTransformPoint(point);
        var closest = Vector2.Clamp(local, -HalfSize, HalfSize);
        if (MathF.Abs(local.X) < HalfSize.X && MathF.Abs(local.Y) < HalfSize.Y)
        {
            if (HalfSize.X - MathF.Abs(local.X) <= HalfSize.Y - MathF.Abs(local.Y))
                closest.X = local.X < 0 ? -HalfSize.X : HalfSize.X;
            else closest.Y = local.Y < 0 ? -HalfSize.Y : HalfSize.Y;
        }
        return Pose.TransformPoint(closest);
    }
    public float ProjectedRadius(Vector2 unitAxis) => MathF.Abs(Vector2.Dot(Pose.Right, unitAxis)) * HalfSize.X
        + MathF.Abs(Vector2.Dot(Pose.Forward, unitAxis)) * HalfSize.Y;
    public bool Contains(Vector2 point, float tolerance = Geometry2D.DefaultTolerance)
    {
        var p = Pose.InverseTransformPoint(point);
        return MathF.Abs(p.X) <= HalfSize.X + tolerance && MathF.Abs(p.Y) <= HalfSize.Y + tolerance;
    }
}

public readonly record struct Circle
{
    public Vector2 Center { get; }
    public float Radius { get; }
    [JsonConstructor]
    public Circle(Vector2 center, float radius)
    {
        Geometry2D.RequireFinite(center, nameof(center));
        if (!float.IsFinite(radius) || radius <= 0) throw new ArgumentOutOfRangeException(nameof(radius));
        Center = center;
        Radius = radius;
    }
}

/// <summary>Immutable convex polygon. Constructor copies vertices and rejects concave/degenerate input.
/// Boundary vertices may be clockwise or counterclockwise; no closing duplicate vertex is accepted.</summary>
public sealed class ConvexPolygon
{
    private readonly Vector2[] vertices;
    public ReadOnlySpan<Vector2> Vertices => vertices;
    public int Count => vertices.Length;
    public ConvexPolygon(ReadOnlySpan<Vector2> vertices)
    {
        if (vertices.Length < 3) throw new ArgumentException("At least three vertices are required.", nameof(vertices));
        this.vertices = vertices.ToArray();
        double area = 0;
        for (var i = 0; i < vertices.Length; i++)
        {
            Geometry2D.RequireFinite(vertices[i], nameof(vertices));
            if (Vector2.DistanceSquared(vertices[i], vertices[(i + 1) % vertices.Length]) < 1e-10f)
                throw new ArgumentException("Adjacent vertices must differ.", nameof(vertices));
            area += Geometry2D.Cross(vertices[i], vertices[(i + 1) % vertices.Length]);
        }
        if (Math.Abs(area) < 1e-8) throw new ArgumentException("Polygon has zero area.", nameof(vertices));
        var sign = Math.Sign(area);
        // Every vertex must be in the same closed half-plane of every edge. This also rejects
        // self-intersecting polygons, which a consecutive-triple turn check alone can accept.
        for (var i = 0; i < vertices.Length; i++)
        {
            var a = vertices[i];
            var edge = vertices[(i + 1) % vertices.Length] - a;
            for (var j = 0; j < vertices.Length; j++)
                if (Geometry2D.Cross(edge, vertices[j] - a) * sign < -1e-5f)
                    throw new ArgumentException("Vertices must describe a convex perimeter.", nameof(vertices));
        }
        if (area < 0) Array.Reverse(this.vertices);
    }
    public ConvexPolygon Transform(Pose pose)
    {
        var transformed = new Vector2[vertices.Length];
        for (var i = 0; i < transformed.Length; i++) transformed[i] = pose.TransformPoint(vertices[i]);
        return new ConvexPolygon(transformed);
    }
    public bool Contains(Vector2 point, float tolerance = Geometry2D.DefaultTolerance)
    {
        for (var i = 0; i < vertices.Length; i++)
        {
            var edge = vertices[(i + 1) % vertices.Length] - vertices[i];
            if (Geometry2D.Cross(edge, point - vertices[i]) < -tolerance * edge.Length()) return false;
        }
        return true;
    }
    public Vector2 ClosestBoundaryPoint(Vector2 point)
    {
        var closest = vertices[0];
        var bestSquared = float.PositiveInfinity;
        for (var i = 0; i < vertices.Length; i++)
        {
            var candidate = Geometry2D.ClosestPoint(new Segment(vertices[i], vertices[(i + 1) % vertices.Length]), point);
            var squared = Vector2.DistanceSquared(point, candidate);
            if (squared < bestSquared) { bestSquared = squared; closest = candidate; }
        }
        return closest;
    }
    public Vector2 ClosestPoint(Vector2 point) => Contains(point, 0) ? point : ClosestBoundaryPoint(point);
}

public readonly record struct ClosestPair(float Distance, Vector2 First, Vector2 Second);

/// <summary>SWArmada 0.2.7's measured rulers, converted from its 22.85 mm/world-unit scale.
/// These deliberately preserve source dimensions instead of substituting nominal inches.</summary>
public static class TabletopMeasurements
{
    public const float MillimetresPerWorldUnit = 22.85f;
    public const float ManeuverSegment = 66.7131f;
    public const float ManeuverClickDegrees = 22.5f;
    public const float CloseRange = 124;
    public const float MediumRange = 188;
    public const float LongRange = 303;
    public const float DistanceOne = 77;
    public const float StandardBoardWidth = 1828.8f;
    public const float StandardBoardLength = 914.4f;
    public static float DistanceBand(int distance) => Math.Max(0, distance) switch
    {
        0 => 0, 1 => 77, 2 => 125, 3 => 185, 4 => 245, _ => 303
    };
    public static float FromWorldUnits(float value) => value * MillimetresPerWorldUnit;
    public static float ToWorldUnits(float millimetres) => millimetres / MillimetresPerWorldUnit;
}
