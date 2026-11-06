using System.Numerics;
using Armada.Data;

namespace Swarmada.Headless.Engine;

public readonly record struct SourceRangeMeasurement(float Distance, Vector3 Origin, Vector3 Destination, bool HasEligibleDefenderEdge);
public static class AttackRangeGeometry
{
    public static SourceRangeMeasurement MeasureSegmentToCircle(Vector3 a, Vector3 b, Vector3 center, float radius)
    {
        var delta = b - a;
        float t = delta.LengthSquared() < 0.0001f ? 0 : Math.Clamp(Vector3.Dot(center - a, delta) / delta.LengthSquared(), 0, 1);
        var origin = a + delta * t;
        var direction = EngineVectors.ProjectOnPlane(center - origin, Vector3.UnitY);
        float length = direction.Length(), r = Math.Max(0, radius);
        var destination = length > 1E-5f ? center - direction / length * Math.Min(r, length) : origin;
        return new(Math.Max(0, length - r), origin, destination, true);
    }
}
public static partial class IgnitionAttackGeometry
{
    public static void GetWorldSpecialArc(Quaternion rotation, IgnitionProfile? profile,
        out Vector3 left, out Vector3 right, out Vector3 forward, out float halfAngle)
    {
        Vector2 l = profile?.LeftBoundaryLocal ?? new(-0.3826834f, 0.9238795f);
        Vector2 r = profile?.RightBoundaryLocal ?? new(0.3826834f, 0.9238795f);
        left = EngineVectors.ProjectOnPlane(Vector3.Transform(new(l.X, 0, l.Y), rotation), Vector3.UnitY).NormalizeSafe();
        right = EngineVectors.ProjectOnPlane(Vector3.Transform(new(r.X, 0, r.Y), rotation), Vector3.UnitY).NormalizeSafe();
        forward = (left + right).NormalizeSafe();
        halfAngle = Math.Max(EngineVectors.Angle(forward, left), EngineVectors.Angle(forward, right));
    }
    public static bool ContainsPointInSpecialArc(Vector3 center, Quaternion rotation, IgnitionProfile profile, Vector3 point)
    {
        var direction = EngineVectors.ProjectOnPlane(point - center, Vector3.UnitY);
        if (direction.LengthSquared() < 0.0001f) return false;
        GetWorldSpecialArc(rotation, profile, out _, out _, out var forward, out float halfAngle);
        return forward.LengthSquared() >= 0.0001f && EngineVectors.Angle(forward, direction.NormalizeSafe()) <= halfAngle + 0.2f;
    }
}
