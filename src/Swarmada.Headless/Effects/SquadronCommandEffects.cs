using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Effects;

/// <summary>Port of CommandUpgradeRules. Base values and active upgrade flags come from the
/// commander's definition/loadout; Flight Controllers requires a currently live commanding ship.</summary>
public static class CommandUpgradeEffects
{
    public static int SquadronValue(int baseSquadronValue, bool expandedHangarBayActive, bool sphaTActive)
        => Math.Max(0, checked(baseSquadronValue + (expandedHangarBayActive ? 1 : 0) - (sphaTActive ? 2 : 0)));
    public static float SquadronCommandRangeMillimetres(bool boostedCommsActive)
        => boostedCommsActive ? TabletopMeasurements.LongRange : TabletopMeasurements.MediumRange;
    public static bool GrantsFlightControllersBlueDie(bool liveCommandingShipFound, bool commandingShipFlightControllersActive)
        => liveCommandingShipFound && commandingShipFlightControllersActive;
}

/// <summary>Port of SquadronUpgradeRules with explicit scene candidates. These kernels select
/// eligible support sources only; attack rerolls, die removal, and engagement enforcement are
/// committed by the relevant future gameplay handlers.</summary>
public static class SquadronUpgradeEffects
{
    public static int? BomberCommandCenterSource(bool attackerHasBomber, int attackerOwner,
        ReadOnlySpan<NearbyUpgradeSource> sources)
    {
        if (!attackerHasBomber) return null;
        foreach (var source in sources)
            if (!source.Destroyed && source.Owner == attackerOwner && IsActive(source, "bomber-command-center")
                && source.DistanceToSubjectMillimetres <= TabletopMeasurements.LongRange) return source.RuntimeIndex;
        return null;
    }
    public static int? JammingFieldSource(ReadOnlySpan<NearbyUpgradeSource> sources)
    {
        foreach (var source in sources)
            if (!source.Destroyed && IsActive(source, "jamming-field")
                && (source.DistanceToSubjectMillimetres <= TabletopMeasurements.CloseRange
                    || source.DistanceToOtherMillimetres <= TabletopMeasurements.CloseRange)) return source.RuntimeIndex;
        return null;
    }
    public static bool HeavyPreventsShipAttack(bool squadronHasHeavy, int squadronOwner, ReadOnlySpan<NearbyUpgradeSource> sources)
    {
        if (!squadronHasHeavy) return false;
        foreach (var source in sources)
            if (!source.Destroyed && source.Owner == squadronOwner && IsActive(source, "advanced-transponder-net")
                && source.DistanceToSubjectMillimetres <= TabletopMeasurements.CloseRange) return true;
        return false;
    }
    private static bool IsActive(NearbyUpgradeSource source, string cardId)
        => source.Card is { Active: true } && string.Equals(source.Card.Definition.id, cardId, StringComparison.OrdinalIgnoreCase);
}
