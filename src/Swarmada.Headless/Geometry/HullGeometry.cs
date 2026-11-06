using System.Numerics;
using Armada.Core;
using Armada.Rules;

namespace Swarmada.Headless.Geometry;

/// <summary>World-space geometry for one hull zone. Edges are the zone's exterior perimeter,
/// not its internal firing-arc divisions. Boundary rays run clockwise and span less than 180 degrees.</summary>
public sealed class HullZoneGeometry
{
    private readonly Segment[] edges;
    public HullZone Zone { get; }
    public Vector2 ArcOrigin { get; }
    public Vector2 FirstBoundary { get; }
    public Vector2 SecondBoundary { get; }
    public Vector2 TargetingPoint { get; }
    public ReadOnlySpan<Segment> Edges => edges;
    public HullZoneGeometry(HullZone zone, Vector2 arcOrigin, Vector2 firstBoundary, Vector2 secondBoundary,
        Vector2 targetingPoint, ReadOnlySpan<Segment> edges)
    {
        Geometry2D.RequireFinite(arcOrigin, nameof(arcOrigin));
        Geometry2D.RequireFinite(firstBoundary, nameof(firstBoundary));
        Geometry2D.RequireFinite(secondBoundary, nameof(secondBoundary));
        Geometry2D.RequireFinite(targetingPoint, nameof(targetingPoint));
        if (firstBoundary.LengthSquared() == 0 || secondBoundary.LengthSquared() == 0
            || Geometry2D.Cross(firstBoundary, secondBoundary) >= 0)
            throw new ArgumentException("Boundary rays must span a clockwise convex cone.");
        if (edges.Length == 0) throw new ArgumentException("A hull zone needs at least one edge.", nameof(edges));
        this.edges = edges.ToArray();
        foreach (var edge in this.edges)
        {
            Geometry2D.RequireFinite(edge.Start, nameof(edges));
            Geometry2D.RequireFinite(edge.End, nameof(edges));
            if (edge.Length == 0) throw new ArgumentException("Hull edge cannot be degenerate.", nameof(edges));
        }
        Zone = zone;
        ArcOrigin = arcOrigin;
        FirstBoundary = Vector2.Normalize(firstBoundary);
        SecondBoundary = Vector2.Normalize(secondBoundary);
        TargetingPoint = targetingPoint;
    }
    public bool ContainsInArc(Vector2 point, float tolerance = Geometry2D.DefaultTolerance)
    {
        Geometry2D.RequireTolerance(tolerance);
        var relative = point - ArcOrigin;
        return Geometry2D.Cross(FirstBoundary, relative) <= tolerance
            && Geometry2D.Cross(SecondBoundary, relative) >= -tolerance;
    }
}

public readonly record struct AttackRangeMeasurement(float Distance, Vector2 Origin, Vector2 Destination, bool HasEligibleDefenderEdge)
{
    public RangeBand Range => HullGeometry.ClassifyRange(Distance);
}

public static class HullGeometry
{
    public static RangeBand ClassifyRange(float distanceMillimetres)
    {
        if (float.IsNaN(distanceMillimetres) || distanceMillimetres < 0) throw new ArgumentOutOfRangeException(nameof(distanceMillimetres));
        return distanceMillimetres <= TabletopMeasurements.CloseRange ? RangeBand.Close
            : distanceMillimetres <= TabletopMeasurements.MediumRange ? RangeBand.Medium
            : distanceMillimetres <= TabletopMeasurements.LongRange ? RangeBand.Long : RangeBand.OutOfRange;
    }
    public static HullZoneGeometry Standard(OrientedRectangle footprint, HullZone zone)
    {
        var h = footprint.HalfSize;
        Vector2 first, second, target;
        switch (zone)
        {
            case HullZone.Front: first = new(-h.X, h.Y); second = new(h.X, h.Y); target = new(0, footprint.Size.Y / 3); break;
            case HullZone.Rear: first = new(h.X, -h.Y); second = new(-h.X, -h.Y); target = new(0, -footprint.Size.Y / 3); break;
            case HullZone.Left: first = new(-h.X, -h.Y); second = new(-h.X, h.Y); target = new(-footprint.Size.X / 3, 0); break;
            case HullZone.Right: first = new(h.X, h.Y); second = new(h.X, -h.Y); target = new(footprint.Size.X / 3, 0); break;
            default: throw new ArgumentOutOfRangeException(nameof(zone), "Auxiliary zones require a huge hull profile.");
        }
        var pose = footprint.Pose;
        return new HullZoneGeometry(zone, pose.Position, pose.TransformDirection(first), pose.TransformDirection(second),
            pose.TransformPoint(target), new[] { new Segment(pose.TransformPoint(first), pose.TransformPoint(second)) });
    }
    /// <summary>SWArmada's source measured Super Star Destroyer hull profiles, including all six zones.
    /// prototype selects prototype-super-star-destroyer instead of the standard huge profile.</summary>
    public static HullZoneGeometry Huge(OrientedRectangle footprint, HullZone zone, bool prototype = false)
    {
        var h = footprint.HalfSize;
        Vector2 Scale(float x, float y) => new(x * h.X, y * h.Y);
        var frontLeft = Scale(0.00024143f, prototype ? 0.5485351f : 0.54950696f);
        var frontRight = Scale(0.00024143f, prototype ? 0.54416186f : 0.54197526f);
        var rearLeft = Scale(0.00024143f, prototype ? -0.7576053f : -0.75784826f);
        var rearRight = Scale(0.00024143f, prototype ? -0.7656229f : -0.765137f);
        var fl = Scale(-0.8240397f, prototype ? 0.70676005f : 0.70590967f) - frontLeft;
        var fr = Scale(0.82452255f, prototype ? 0.712958f : 0.7115002f) - frontRight;
        var rl = Scale(-0.8240397f, -0.81391203f) - rearLeft;
        var rr = Scale(0.82452255f, prototype ? -0.8073496f : -0.8071067f) - rearRight;
        var middle = (prototype ? -0.14340834f : -0.1424365f) * h.Y;
        Vector2 a, b, first, second, target;
        Vector2[] perimeter;
        var zfl = AtX(frontLeft, fl, -h.X);
        var zfr = AtX(frontRight, fr, h.X);
        var zrl = AtX(rearLeft, rl, -h.X);
        var zrr = AtX(rearRight, rr, h.X);
        switch (zone)
        {
            case HullZone.Front:
                a = frontLeft; b = frontRight; first = fl; second = fr;
                target = prototype ? Scale(0.00093f, 0.87792f) : Scale(0.00235f, 0.87834f);
                perimeter = new[] { new Vector2(-h.X, zfl), new(-h.X, h.Y), new(h.X, h.Y), new(h.X, zfr) }; break;
            case HullZone.Rear:
                a = rearRight; b = rearLeft; first = rr; second = rl;
                target = prototype ? Scale(-0.00264f, -0.91046f) : Scale(0.00329f, -0.91082f);
                perimeter = new[] { new Vector2(h.X, zrr), new(h.X, -h.Y), new(-h.X, -h.Y), new(-h.X, zrl) }; break;
            case HullZone.Left:
                a = new(0, middle); b = frontLeft; first = -Vector2.UnitX; second = fl;
                target = prototype ? Scale(-0.61101f, 0.23789f) : Scale(-0.60776f, 0.2393f);
                perimeter = new[] { new Vector2(-h.X, middle), new(-h.X, zfl) }; break;
            case HullZone.Right:
                a = frontRight; b = new(0, middle); first = fr; second = Vector2.UnitX;
                target = prototype ? Scale(0.59981f, 0.23853f) : Scale(0.6147f, 0.23939f);
                perimeter = new[] { new Vector2(h.X, zfr), new(h.X, middle) }; break;
            case HullZone.LeftAuxiliary:
                a = rearLeft; b = new(0, middle); first = rl; second = -Vector2.UnitX;
                target = prototype ? Scale(-0.6125f, -0.48476f) : Scale(-0.61089f, -0.48455f);
                perimeter = new[] { new Vector2(-h.X, zrl), new(-h.X, middle) }; break;
            case HullZone.RightAuxiliary:
                a = new(0, middle); b = rearRight; first = Vector2.UnitX; second = rr;
                target = prototype ? Scale(0.60561f, -0.48453f) : Scale(0.61975f, -0.48457f);
                perimeter = new[] { new Vector2(h.X, middle), new(h.X, zrr) }; break;
            default: throw new ArgumentOutOfRangeException(nameof(zone));
        }
        var origin = a + first * (Geometry2D.Cross(b - a, second) / Geometry2D.Cross(first, second));
        var pose = footprint.Pose;
        var edges = new Segment[perimeter.Length - 1];
        for (var i = 0; i < edges.Length; i++) edges[i] = new Segment(pose.TransformPoint(perimeter[i]), pose.TransformPoint(perimeter[i + 1]));
        return new HullZoneGeometry(zone, pose.TransformPoint(origin), pose.TransformDirection(first), pose.TransformDirection(second), pose.TransformPoint(target), edges);
    }
    private static float AtX(Vector2 point, Vector2 direction, float x) => point.Y + (x - point.X) * direction.Y / direction.X;
    /// <summary>Range measures to the portion of the selected defending zone lying inside the firing arc,
    /// rather than measuring whole-base distance or distance between targeting dots.</summary>
    public static AttackRangeMeasurement MeasureAttack(HullZoneGeometry attacker, HullZoneGeometry defender)
    {
        var best = new AttackRangeMeasurement(float.PositiveInfinity, attacker.Edges[0].Start, defender.Edges[0].Start, false);
        foreach (var defendingEdge in defender.Edges)
        {
            if (!ClipToArc(defendingEdge, attacker, out var clipped)) continue;
            foreach (var attackingEdge in attacker.Edges)
            {
                var pair = Geometry2D.ClosestPoints(attackingEdge, clipped);
                if (pair.Distance < best.Distance) best = new(pair.Distance, pair.First, pair.Second, true);
            }
        }
        return best;
    }
    public static bool ClipToArc(Segment edge, HullZoneGeometry arc, out Segment clipped)
    {
        var relative = edge.Start - arc.ArcOrigin;
        var enter = 0f;
        var exit = 1f;
        if (Geometry2D.ClipPositive(-Geometry2D.Cross(arc.FirstBoundary, relative), -Geometry2D.Cross(arc.FirstBoundary, edge.Delta), ref enter, ref exit)
            && Geometry2D.ClipPositive(Geometry2D.Cross(arc.SecondBoundary, relative), Geometry2D.Cross(arc.SecondBoundary, edge.Delta), ref enter, ref exit))
        {
            clipped = new(edge.PointAt(enter), edge.PointAt(exit));
            // The source disallows a defending edge touching only one arc-boundary point.
            return clipped.Length > 0.0002285f;
        }
        clipped = default;
        return false;
    }
    public static ClosestPair MeasureToSquadron(HullZoneGeometry attacker, Circle target)
    {
        var best = new ClosestPair(float.PositiveInfinity, default, default);
        foreach (var edge in attacker.Edges)
        {
            var origin = Geometry2D.ClosestPoint(edge, target.Center);
            var delta = target.Center - origin;
            var length = delta.Length();
            var destination = length <= target.Radius ? origin : target.Center - delta * (target.Radius / length);
            Geometry2D.Consider(ref best, origin, destination);
        }
        return best;
    }
    /// <summary>LOS must first enter an exterior edge belonging to the selected target zone.
    /// The default corner tolerance reproduces the source's 0.02-world-unit adjacent-zone allowance.</summary>
    public static bool IsTargetZoneExposed(Vector2 from, HullZoneGeometry target, OrientedRectangle targetBase,
        float cornerTolerance = 0.457f)
    {
        Geometry2D.RequireTolerance(cornerTolerance);
        if (targetBase.Contains(from, 0)) return false;
        var line = new Segment(from, target.TargetingPoint);
        if (!Geometry2D.Clip(line, targetBase, out var entry, out _)) return false;
        var point = line.PointAt(entry);
        foreach (var edge in target.Edges)
            if (Vector2.Distance(point, Geometry2D.ClosestPoint(edge, point)) <= cornerTolerance) return true;
        return false;
    }
}

public enum LineOfSightResult { Clear, Obstructed, Blocked }
public enum LineContactPolicy { SourceIncludesTangency, InteriorOnly }
public readonly record struct LineOfSightObstacle(OrientedRectangle Footprint, bool BlocksAbsolutely = false)
{
    public PolygonFootprint? Polygon { get; init; }
    public LineOfSightObstacle(PolygonFootprint footprint, bool BlocksAbsolutely = false)
        : this(footprint.Bounds, BlocksAbsolutely) { Polygon = footprint; }
}

public static class LineOfSightGeometry
{
    /// <summary>Only third-party ships/obstacles should be supplied. SourceIncludesTangency preserves
    /// SWArmada's closed rectangle segment test; InteriorOnly is an explicit geometry policy change.
    /// Physical obstacle outlines retain their concave boundaries.</summary>
    public static LineOfSightResult Evaluate(Segment line, ReadOnlySpan<LineOfSightObstacle> blockers,
        bool ignoresObstruction = false, LineContactPolicy contactPolicy = LineContactPolicy.SourceIncludesTangency)
    {
        if (line.Length <= 0.0002285f) return LineOfSightResult.Clear;
        var obstructed = false;
        foreach (var blocker in blockers)
        {
            var crosses = blocker.Polygon is { } polygon
                ? contactPolicy == LineContactPolicy.SourceIncludesTangency ? TouchesPolygon(line, polygon) : Geometry2D.SegmentCrossesInterior(line, polygon)
                : contactPolicy == LineContactPolicy.SourceIncludesTangency ? Geometry2D.Clip(line, blocker.Footprint, out _, out _)
                    : Geometry2D.SegmentCrossesInterior(line, blocker.Footprint);
            if (!crosses) continue;
            if (blocker.BlocksAbsolutely) return LineOfSightResult.Blocked;
            obstructed = true;
        }
        return obstructed && !ignoresObstruction ? LineOfSightResult.Obstructed : LineOfSightResult.Clear;
    }
    private static bool TouchesPolygon(Segment line, PolygonFootprint polygon)
    {
        if (!Geometry2D.Clip(line, polygon.Bounds, out _, out _)) return false;
        if (polygon.Contains(line.Start, 0) || polygon.Contains(line.End, 0)) return true;
        for (int edge = 0; edge < polygon.Count; edge++)
            if (Geometry2D.TryIntersect(line, polygon.Edge(edge), out _)) return true;
        return false;
    }
}
