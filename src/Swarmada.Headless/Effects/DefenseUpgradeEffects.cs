using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Effects;

public readonly record struct DamageReductionResult(DefenseResolution Defense, int SelectedReduction, int BrightHopeReduction);

/// <summary>Public functional behavior ported from DefenseUpgradeRules. Its unused private legacy
/// auto-choice routines (Adi Gallia/Tranquility/nearby reducers) are deliberately not activated here.
/// Their player choices and timing still require dedicated rule handlers.</summary>
public static class DefenseUpgradeEffects
{
    public static bool HasStartOfShipPhaseChoice(string cardId) => cardId is "early-warning-system" or "reinforced-blast-doors";
    public static int MaximumRedirectZones(bool advancedProjectorsActive, bool foresightActive)
        => advancedProjectorsActive ? 3 : foresightActive ? 2 : 1;
    public static bool AllowsNonAdjacentRedirect(bool advancedProjectorsActive) => advancedProjectorsActive;
    public static int ThermalShieldsRemovalCount(int initialGatheredDiceCount, int remainingDiceCount)
        => Math.Min(Math.Max(0, remainingDiceCount), Math.Max(0, initialGatheredDiceCount) / 2);
    public static bool CanUseThermalShields(ShipRuntimeState? defender, UpgradeCardRuntimeState? thermalShields,
        int initialGatheredDiceCount, int remainingDiceCount)
        => defender is { CurrentHull: > 0, CurrentSpeed: > 0 }
            && ThermalShieldsRemovalCount(initialGatheredDiceCount, remainingDiceCount) > 0
            && thermalShields is { Active: true, Exhausted: false, UsesThisAttack: 0 }
            && defender.CanResolveDefenseToken(DefenseTokenType.Brace);
    /// <summary>Eligibility only. It neither exhausts Thermal Shields nor spends a Brace; the attack
    /// handler must atomically commit the selected cost and removals at the proper timing window.</summary>
    public static bool CanUseThermalShields(ShipRuntimeState? defender, UpgradeCardRuntimeState? thermalShields, int gatheredDiceCount)
        => CanUseThermalShields(defender, thermalShields, gatheredDiceCount, gatheredDiceCount);
    public static DamageReductionResult ApplyBeforeDamageReductions(DefenseResolution defense, HullZone defendingZone,
        int selectedReduction, bool brightHopeActive)
    {
        var selected = Math.Min(Math.Max(0, selectedReduction), defense.TotalDamage);
        if (selected > 0) defense = DefenseResolver.ReduceTotalDamage(defense, selected);
        var brightHope = brightHopeActive && defendingZone != HullZone.Rear && defense.TotalDamage > 0 ? 1 : 0;
        if (brightHope > 0) defense = DefenseResolver.ReduceTotalDamage(defense, brightHope);
        return new(defense, selected, brightHope);
    }
    /// <summary>Returns the first eligible source in stable caller order, matching the source query's
    /// first-match behavior. Distance is footprint-to-footprint, not centre distance.</summary>
    public static int? TargetingScramblerSource(int defenderOwner, ReadOnlySpan<NearbyUpgradeSource> candidates)
    {
        foreach (var source in candidates)
            if (!source.Destroyed && source.Owner == defenderOwner && source.Card is { Active: true, Exhausted: false }
                && string.Equals(source.Card.Definition.id, "targeting-scrambler", StringComparison.OrdinalIgnoreCase)
                && source.DistanceToSubjectMillimetres <= TabletopMeasurements.MediumRange)
                return source.RuntimeIndex;
        return null;
    }
}

/// <summary>Explicit replacement for scene-object discovery. Supply all potential sources in stable
/// entity order and compute distances from the actual bases, with destroyed status from live state.</summary>
public readonly record struct NearbyUpgradeSource(int RuntimeIndex, int Owner, bool Destroyed,
    float DistanceToSubjectMillimetres, UpgradeCardRuntimeState? Card, float DistanceToOtherMillimetres = float.PositiveInfinity);

/// <summary>Port of OfficerCombatRules. The caller is responsible for attack/ram timing.</summary>
public static class OfficerCombatEffects
{
    public static bool TreatsAttackAsObstructed(bool montferratActive, int currentSpeed) => montferratActive && currentSpeed >= 3;
    public static bool DiscardMontferratAfterRam(UpgradeCardRuntimeState? montferrat)
        => montferrat is { Active: true } && montferrat.Discard();
    public static bool PreventsCriticalEffects(bool commanderGherantActive) => commanderGherantActive;
    public static bool DiscardGherantAfterShipAttack(UpgradeCardRuntimeState? gherant)
        => gherant is { Active: true } && gherant.Discard();
    public static bool ContainPreventsAllCriticalEffects(bool damageControlOfficerActive) => damageControlOfficerActive;
    /// <summary>The recovered Captain Brunson helper uses centre distance, unlike footprint-based
    /// squadron/support ranges. This API retains that source behavior explicitly.</summary>
    public static bool IsNearObstacle(ReadOnlySpan<float> activeObstacleCentreDistancesMillimetres)
    {
        foreach (var distance in activeObstacleCentreDistancesMillimetres)
            if (distance <= TabletopMeasurements.MediumRange) return true;
        return false;
    }
}
