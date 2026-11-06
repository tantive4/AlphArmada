using System.Numerics;

namespace Swarmada.Headless.Geometry;

public enum GuideSide { Auto, Center, Port, Starboard }

public static class ManeuverGeometry
{
    public static GuideSide GuideSideForCourse(ReadOnlySpan<int> clicks)
    {
        foreach (var click in clicks)
        {
            if (click < 0) return GuideSide.Port;
            if (click > 0) return GuideSide.Starboard;
        }
        return GuideSide.Center;
    }
    private static Vector2 GuideOffset(Vector2 footprint, GuideSide side, bool rearBase)
    {
        if (footprint.X <= 0 || footprint.Y <= 0) throw new ArgumentOutOfRangeException(nameof(footprint));
        Geometry2D.RequireFinite(footprint, nameof(footprint));
        var sign = side switch { GuideSide.Port => -1, GuideSide.Starboard => 1, _ => 0 };
        return new(footprint.X * 0.5f * sign, footprint.Y * (rearBase ? -0.14657535f : 0.5f));
    }
    /// <summary>Exact planar counterpart of SWArmada ManeuverGeometry. Each link first advances
    /// 66.7131 mm, then rotates by click*22.5 degrees, and the final guide offset is removed.</summary>
    public static Pose CalculateEndPose(Pose start, ReadOnlySpan<int> clicks, Vector2 footprint,
        GuideSide side = GuideSide.Auto, bool rearBase = false)
        => SamplePose(start, clicks, footprint, 1, side, rearBase);
    public static Pose SamplePose(Pose start, ReadOnlySpan<int> clicks, Vector2 footprint, float progress,
        GuideSide side = GuideSide.Auto, bool rearBase = false)
    {
        if (!float.IsFinite(progress)) throw new ArgumentOutOfRangeException(nameof(progress));
        ValidateClicks(clicks);
        if (side == GuideSide.Auto) side = GuideSideForCourse(clicks);
        var offset = GuideOffset(footprint, side, rearBase);
        if (clicks.Length == 0 || progress <= 0) return start;
        progress = Math.Clamp(progress, 0, 1);
        var guide = start.TransformPoint(offset);
        var yaw = start.YawDegrees;
        var scaled = progress * clicks.Length;
        var joint = Math.Min(clicks.Length - 1, (int)scaled);
        for (var i = 0; i < joint; i++)
        {
            guide += new Pose(Vector2.Zero, yaw).Forward * TabletopMeasurements.ManeuverSegment;
            yaw += clicks[i] * TabletopMeasurements.ManeuverClickDegrees;
        }
        var fraction = progress >= 1 ? 1 : scaled - joint;
        guide += new Pose(Vector2.Zero, yaw).Forward * (TabletopMeasurements.ManeuverSegment * fraction);
        yaw += clicks[joint] * TabletopMeasurements.ManeuverClickDegrees * fraction;
        var rotation = new Pose(Vector2.Zero, yaw);
        return new Pose(guide - rotation.TransformDirection(offset), yaw);
    }
    public static Vector2[] CalculateGuidePoints(Pose start, ReadOnlySpan<int> clicks, Vector2 footprint,
        GuideSide side = GuideSide.Auto, bool rearBase = false)
    {
        ValidateClicks(clicks);
        if (side == GuideSide.Auto) side = GuideSideForCourse(clicks);
        var offset = GuideOffset(footprint, side, rearBase);
        if (clicks.Length == 0) return new[] { start.Position };
        var points = new Vector2[clicks.Length + 1];
        var guide = start.TransformPoint(offset);
        var yaw = start.YawDegrees;
        points[0] = guide;
        for (var i = 0; i < clicks.Length; i++)
        {
            guide += new Pose(Vector2.Zero, yaw).Forward * TabletopMeasurements.ManeuverSegment;
            points[i + 1] = guide;
            yaw += clicks[i] * TabletopMeasurements.ManeuverClickDegrees;
        }
        return points;
    }
    public static bool HugeShipOverlapsTool(Pose start, ReadOnlySpan<int> clicks, Vector2 footprint, GuideSide side = GuideSide.Auto)
    {
        if (clicks.Length == 0) return false;
        var points = CalculateGuidePoints(start, clicks, footprint, side, rearBase: true);
        var end = CalculateEndPose(start, clicks, footprint, side, rearBase: true);
        var startBase = new OrientedRectangle(start, footprint);
        var endBase = new OrientedRectangle(end, footprint);
        for (var i = 1; i < points.Length; i++)
        {
            var segment = new Segment(points[i - 1], points[i]);
            // The source deliberately uses a larger inset here than for base overlap.
            if (Geometry2D.SegmentCrossesInterior(segment, startBase, 0.02285f)
                || Geometry2D.SegmentCrossesInterior(segment, endBase, 0.02285f)) return true;
        }
        return false;
    }
    /// <summary>Complete Cartesian product of chart-permitted yaw values, independent of UI previews.
    /// Additional yaw from navigation/upgrades must be applied to the chart before calling.</summary>
    public static IReadOnlyList<int[]> EnumerateCourses(ReadOnlySpan<int> maximumYawPerJoint)
    {
        if (maximumYawPerJoint.Length > 4) throw new ArgumentOutOfRangeException(nameof(maximumYawPerJoint));
        var chart = maximumYawPerJoint.ToArray();
        foreach (var maximum in chart) if (maximum < 0 || maximum > 2) throw new ArgumentOutOfRangeException(nameof(maximumYawPerJoint));
        var result = new List<int[]>();
        EnumerateJoint(chart, new int[chart.Length], 0, result);
        return result;
    }
    private static void EnumerateJoint(int[] chart, int[] working, int index, List<int[]> result)
    {
        if (index == chart.Length) { result.Add((int[])working.Clone()); return; }
        for (var click = -chart[index]; click <= chart[index]; click++)
        {
            working[index] = click;
            EnumerateJoint(chart, working, index + 1, result);
        }
    }
    private static void ValidateClicks(ReadOnlySpan<int> clicks)
    {
        if (clicks.Length > 4) throw new ArgumentOutOfRangeException(nameof(clicks));
        foreach (var click in clicks) if (click < -2 || click > 2) throw new ArgumentOutOfRangeException(nameof(clicks));
    }
}

public static partial class PlayAreaGeometry
{
    public const float SourceContainmentTolerance = 0.011425f;
    public static bool Contains(OrientedRectangle board, OrientedRectangle footprint, float tolerance = SourceContainmentTolerance)
    {
        Geometry2D.RequireTolerance(tolerance);
        Span<Vector2> corners = stackalloc Vector2[4];
        footprint.CopyCorners(corners);
        foreach (var corner in corners) if (!board.Contains(corner, tolerance)) return false;
        return true;
    }
    public static bool Contains(OrientedRectangle board, Circle footprint, float tolerance = SourceContainmentTolerance)
    {
        Geometry2D.RequireTolerance(tolerance);
        var local = board.Pose.InverseTransformPoint(footprint.Center);
        return MathF.Abs(local.X) + footprint.Radius <= board.HalfSize.X + tolerance
            && MathF.Abs(local.Y) + footprint.Radius <= board.HalfSize.Y + tolerance;
    }
    public static Vector2 ClampPoint(OrientedRectangle board, Vector2 point, float inset = 0)
    {
        Geometry2D.RequireTolerance(inset);
        var h = Vector2.Max(Vector2.Zero, board.HalfSize - new Vector2(inset));
        return board.Pose.TransformPoint(Vector2.Clamp(board.Pose.InverseTransformPoint(point), -h, h));
    }
    public static float Clearance(OrientedRectangle board, OrientedRectangle footprint)
    {
        var local = board.Pose.InverseTransformPoint(footprint.Center);
        return MathF.Min(board.HalfSize.X - MathF.Abs(local.X) - footprint.ProjectedRadius(board.Pose.Right),
            board.HalfSize.Y - MathF.Abs(local.Y) - footprint.ProjectedRadius(board.Pose.Forward));
    }
}

/// <summary>The source tests squadron containment using its bounding square. TrueCircle is an explicit
/// opt-in correction, permitting circle placements near a rotated board edge that the square rejects.</summary>
public enum SquadronBoardPolicy { SourceBoundingSquare, TrueCircle }

public static class SquadronGeometry
{
    /// <summary>SWArmada clamps centre displacement to the speed ruler; intervening bases do not block movement.</summary>
    public static bool CanReach(Circle source, Vector2 destination, int speed, float tolerance = Geometry2D.DefaultTolerance)
    {
        if (speed < 0 || speed > 5) throw new ArgumentOutOfRangeException(nameof(speed));
        Geometry2D.RequireFinite(destination, nameof(destination));
        Geometry2D.RequireTolerance(tolerance);
        return Vector2.Distance(source.Center, destination) <= TabletopMeasurements.DistanceBand(speed) + tolerance;
    }
    public static bool CanPlace(Circle destination, OrientedRectangle board,
        ReadOnlySpan<OrientedRectangle> ships, ReadOnlySpan<Circle> squadrons,
        SquadronBoardPolicy boardPolicy = SquadronBoardPolicy.SourceBoundingSquare, float squadronYaw = 0)
    {
        var inside = boardPolicy == SquadronBoardPolicy.TrueCircle ? PlayAreaGeometry.Contains(board, destination)
            : PlayAreaGeometry.Contains(board, new OrientedRectangle(new Pose(destination.Center, squadronYaw), new Vector2(destination.Radius * 2)));
        if (!inside) return false;
        foreach (var ship in ships) if (Geometry2D.Overlaps(destination, ship)) return false;
        foreach (var squadron in squadrons) if (Geometry2D.Overlaps(destination, squadron)) return false;
        return true;
    }
    /// <summary>Heavy affects which other rules engagement restricts, not whether a squadron is engaged.
    /// Caller supplies ownership and non-geometric effects (Grit, Intel, Cloak, etc.).</summary>
    public static bool IsEngaged(Circle first, Circle second, bool enemies, bool hasLineOfSight = true,
        float engagementDistance = TabletopMeasurements.DistanceOne)
        => enemies && hasLineOfSight && Geometry2D.Distance(first, second) <= engagementDistance + Geometry2D.DefaultTolerance;
}
