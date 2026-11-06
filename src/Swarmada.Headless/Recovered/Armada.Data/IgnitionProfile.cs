using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json;
using Armada.Core;
using Armada.Rules;

namespace Armada.Data;

/// <summary>Original ignition profile with native value-vector geometry and immutable placement bands.</summary>
public sealed class IgnitionProfile
{
    public HullZone AssociatedHullZone { get; }
    public BatteryArmament SpecialBattery { get; }
    public Vector2 LeftBoundaryLocal { get; }
    public Vector2 RightBoundaryLocal { get; }
    public IReadOnlyList<RangeBand> BasePlacementBands { get; }
    public string TargetingTokenPresentationId { get; }
    public bool IsConfigured => SpecialBattery.TotalDice > 0 && LeftBoundaryLocal.LengthSquared() > 0.0001f &&
        RightBoundaryLocal.LengthSquared() > 0.0001f;

    public IgnitionProfile(HullZone hullZone, BatteryArmament battery, Vector2 leftBoundary,
        Vector2 rightBoundary, RangeBand[]? placementBands, string? presentationId)
    {
        AssociatedHullZone = hullZone;
        SpecialBattery = battery;
        LeftBoundaryLocal = leftBoundary;
        RightBoundaryLocal = rightBoundary;
        BasePlacementBands = Array.AsReadOnly((RangeBand[])(placementBands?.Clone() ?? Array.Empty<RangeBand>()));
        TargetingTokenPresentationId = presentationId ?? string.Empty;
    }

    internal static IgnitionProfile Read(JsonElement value) => new(
        (HullZone)value.GetProperty("associatedHullZone").GetInt32(),
        BatteryArmament.Read(value.GetProperty("specialBattery")),
        ReadVector(value.GetProperty("leftBoundaryLocal")), ReadVector(value.GetProperty("rightBoundaryLocal")),
        value.GetProperty("basePlacementBands").EnumerateArray().Select(x => (RangeBand)x.GetInt32()).ToArray(),
        value.GetProperty("targetingTokenPresentationId").GetString());

    internal static Vector2 ReadVector(JsonElement value) => new(value.GetProperty("x").GetSingle(), value.GetProperty("y").GetSingle());
}
