using System.Numerics;
using System.Text.Json.Serialization;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Objective;

[JsonPolymorphic(TypeDiscriminatorPropertyName = "scenario")]
[JsonDerivedType(typeof(EscortVipSnapshot), "escort-the-vip-v1")]
[JsonDerivedType(typeof(SeizeStationSnapshot), "seize-the-station-v1")]
[JsonDerivedType(typeof(IntelRetrievalSnapshot), "intel-retrieval-v1")]
[JsonDerivedType(typeof(OfficialObjectiveSnapshot), "official-objective-v1")]
public abstract record ObjectiveSnapshot;

public interface IObjectiveRuntime
{
    ObjectiveSnapshot CaptureSnapshot();
    void RestoreSnapshot(ObjectiveSnapshot snapshot);
    IObjectiveRuntime DeepClone();
    void CopyFrom(IObjectiveRuntime source);
}

public static class ObjectiveRuntime
{
    public static IObjectiveRuntime FromSnapshot(ObjectiveSnapshot snapshot) => snapshot switch
    {
        EscortVipSnapshot escort => EscortVipObjective.FromSnapshot(escort),
        SeizeStationSnapshot station => SeizeStationObjective.FromSnapshot(station),
        IntelRetrievalSnapshot intel => IntelRetrievalObjective.FromSnapshot(intel),
        OfficialObjectiveSnapshot official => OfficialObjectiveState.FromSnapshot(official),
        _ => throw new ArgumentException("Unsupported objective snapshot type.", nameof(snapshot))
    };
}

/// <summary>Explicit live capital-ship inputs; squadrons do not contest these source scenarios.</summary>
public readonly record struct ObjectiveShip(int Id, int Owner, OrientedRectangle Base, int Speed,
    bool Destroyed = false, bool Reserved = false, string SizeClass = "small");
public sealed record ObjectiveOutcome(int Winner, string Reason);
/// <summary>On acceptance, the game handler must commit RequiredSpeed to the real ship. These
/// objective classes do not own or mutate the caller's combat state.</summary>
public readonly record struct ObjectiveChoiceResult(bool Accepted, int? RequiredSpeed = null);

public static class ObjectiveGeometry
{
    public const float RequiredBaseOverlap = 0.5f;
    public const float FullBaseContainment = 1;
    // This is a dimensionless fraction tolerance in both source controllers, not Unity distance.
    public const float OverlapFractionTolerance = 0.0005f;
    public const int CircleIntegrationSteps = 96;
    public static bool MeetsOverlap(float fraction, float required = RequiredBaseOverlap)
        => fraction + OverlapFractionTolerance >= Math.Clamp(required, 0, 1);
    /// <summary>Port of IntelRetrievalObjectiveController.CalculateIntelZoneOverlap: Simpson's
    /// rule with 96 slices, a distance-one circle, and the ship's full rectangular base area.
    /// This intentionally retains the source's numerical integration instead of claiming exact area.</summary>
    public static float IntelZoneOverlap(OrientedRectangle footprint, Vector2 zoneCenter)
    {
        var half = Vector2.Max(new Vector2(0.02285f), footprint.HalfSize);
        var center = footprint.Pose.InverseTransformPoint(zoneCenter);
        var radius = TabletopMeasurements.DistanceOne;
        var low = MathF.Max(-half.X, center.X - radius);
        var high = MathF.Min(half.X, center.X + radius);
        if (high <= low) return 0;
        var step = (high - low) / CircleIntegrationSteps;
        var sum = CircleSlice(low, center, radius, half.Y) + CircleSlice(high, center, radius, half.Y);
        for (var i = 1; i < CircleIntegrationSteps; i++)
            sum += CircleSlice(low + step * i, center, radius, half.Y) * (i % 2 == 0 ? 2 : 4);
        return Math.Clamp(sum * step / 3 / (footprint.Size.X * footprint.Size.Y), 0, 1);
    }
    private static float CircleSlice(float x, Vector2 center, float radius, float halfLength)
    {
        var delta = x - center.X;
        var squared = radius * radius - delta * delta;
        if (squared <= 0) return 0;
        var reach = MathF.Sqrt(squared);
        return MathF.Max(0, MathF.Min(halfLength, center.Y + reach) - MathF.Max(-halfLength, center.Y - reach));
    }
    /// <summary>Polygon clipping counterpart of the source's starting-area overlap calculation.
    /// Rotated zones are additionally supported by doing the same clipping in zone-local axes.</summary>
    public static float RectangleOverlap(OrientedRectangle footprint, OrientedRectangle zone)
    {
        Span<Vector2> corners = stackalloc Vector2[4];
        footprint.CopyCorners(corners);
        Span<Vector2> points = stackalloc Vector2[16];
        Span<Vector2> scratch = stackalloc Vector2[16];
        for (int i = 0; i < 4; i++) points[i] = zone.Pose.InverseTransformPoint(corners[i]);
        int count = Clip(points[..4], scratch, 0, -zone.HalfSize.X, true);
        count = Clip(scratch[..count], points, 0, zone.HalfSize.X, false);
        count = Clip(points[..count], scratch, 1, -zone.HalfSize.Y, true);
        count = Clip(scratch[..count], points, 1, zone.HalfSize.Y, false);
        if (count < 3) return 0;
        double area = 0;
        for (var i = 0; i < count; i++)
        {
            var a = points[i]; var b = points[(i + 1) % count];
            area += (double)a.X * b.Y - (double)b.X * a.Y;
        }
        return Math.Clamp((float)(Math.Abs(area) * 0.5 / (footprint.Size.X * footprint.Size.Y)), 0, 1);
    }
    private static int Clip(ReadOnlySpan<Vector2> points, Span<Vector2> result, int axis, float bound, bool minimum)
    {
        int count = 0;
        if (points.Length == 0) return 0;
        float Coordinate(Vector2 v) => axis == 0 ? v.X : v.Y;
        bool Inside(Vector2 v) => minimum ? Coordinate(v) >= bound : Coordinate(v) <= bound;
        var previous = points[^1]; var previousInside = Inside(previous);
        foreach (var point in points)
        {
            var inside = Inside(point);
            if (inside != previousInside)
            {
                var delta = Coordinate(point) - Coordinate(previous);
                var t = MathF.Abs(delta) < 0.00002285f ? 0 : (bound - Coordinate(previous)) / delta;
                result[count++] = Vector2.Lerp(previous, point, t);
            }
            if (inside) result[count++] = point;
            previous = point; previousInside = inside;
        }
        return count;
    }
    internal static ObjectiveShip? Find(ReadOnlySpan<ObjectiveShip> ships, int id)
    {
        foreach (var ship in ships) if (ship.Id == id) return ship;
        return null;
    }
}
