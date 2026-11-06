using Armada.Rules;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Effects;

/// <summary>Pure eligibility kernels from UpgradeCriticalRules. They do not resolve the named
/// critical effects; those require explicit player/chance choices in the gameplay scheduler.</summary>
public static class UpgradeCriticalEffects
{
    public static bool NeedsBlueCritical(string cardId) => cardId is "heavy-ion-emplacements"
        or "ion-cannon-batteries" or "ms-1-ion-cannons" or "nk-7-ion-cannons" or "overload-pulse" or "dodonnas-pride-title";
    public static bool NeedsBlackCritical(string cardId)
        => cardId is "assault-concussion-missiles" or "assault-proton-torpedoes" or "wide-area-barrage";
    public static bool NeedsBlueOrBlackCritical(string cardId) => cardId == "sunder-title";
    public static bool NeedsRedOrBlueCritical(string cardId) => cardId == "kyrsta-agate-officer";
    public static bool IsIgnitionCritical(string cardId)
        => cardId is "orbital-bombardment-particle-cannons" or "superheavy-composite-beam-turbolasers";
    public static bool HasRequiredIgnitionCritical(string cardId, AttackResolution attack, bool isIgnitionAttack)
        => !IsIgnitionCritical(cardId) || (isIgnitionAttack && AttackResolver.HasCriticalIcon(attack, AttackDieColor.Red));
    public static int CriticalEffectLimit(bool fireControlTeamActive, bool fireControlTeamExhausted)
        => fireControlTeamActive && !fireControlTeamExhausted ? 2 : 1;
    public static bool SunderCanDiscardSlot(string slot)
        => slot is "offensive-retro" or "defensive-retro" or "ordnance" or "ion-cannon" or "turbolaser";
    public static int CountCriticalIcons(AttackResolution attack, ReadOnlySpan<AttackDieColor> colors)
    {
        var count = 0;
        foreach (var die in attack.Dice)
        {
            if (die.Face is not (DieFace.Critical or DieFace.HitCritical)) continue;
            foreach (var color in colors)
                if (color == die.Color) { count++; break; }
        }
        return count;
    }
}

/// <summary>Per-owner round state. Store each player's value in BattleState registers so snapshots
/// preserve it; unlike the Unity source's static HashSet, this cannot leak between parallel games.</summary>
[Flags]
public enum FleetCommandMask : long
{
    None = 0,
    AllFightersFollowMe = 1,
    EntrapmentFormation = 2,
    IntensifyFirepower = 4,
    ShieldsToMaximum = 8,
    TakeEvasiveAction = 16
}

/// <summary>Functional port of FleetCommandRules. Paying a token/card cost and choosing the
/// start-of-round timing remain scheduler responsibilities; ResetRound is assignment to None.</summary>
public static class FleetCommandEffects
{
    public static FleetCommandMask MaskFor(string cardId) => cardId switch
    {
        "all-fighters-follow-me" => FleetCommandMask.AllFightersFollowMe,
        "entrapment-formation" => FleetCommandMask.EntrapmentFormation,
        "intensify-firepower" => FleetCommandMask.IntensifyFirepower,
        "shields-to-maximum" => FleetCommandMask.ShieldsToMaximum,
        "take-evasive-action" => FleetCommandMask.TakeEvasiveAction,
        _ => FleetCommandMask.None
    };
    public static bool IsSupported(string cardId) => MaskFor(cardId) != FleetCommandMask.None;
    public static bool Activate(int ownerId, FleetCommandMask current, string cardId, out FleetCommandMask updated)
    {
        updated = current;
        var bit = MaskFor(cardId);
        if (ownerId <= 0 || bit == FleetCommandMask.None || (current & bit) != 0) return false;
        updated |= bit;
        return true;
    }
    public static bool IsActive(int ownerId, FleetCommandMask current, string cardId)
        => ownerId > 0 && MaskFor(cardId) is var bit && bit != FleetCommandMask.None && (current & bit) != 0;
    public static bool IncreasesCommandedSquadronSpeed(bool commanded, FleetCommandMask commandingFleet)
        => commanded && (commandingFleet & FleetCommandMask.AllFightersFollowMe) != 0;
    public static bool AllowsDetermineCourseSpeedChange(FleetCommandMask fleet) => (fleet & FleetCommandMask.EntrapmentFormation) != 0;
    public static bool AllowsSingleHitChange(FleetCommandMask fleet) => (fleet & FleetCommandMask.IntensifyFirepower) != 0;
    public static bool AllowsPreRevealShieldRecovery(FleetCommandMask fleet) => (fleet & FleetCommandMask.ShieldsToMaximum) != 0;
    public static bool IncreasesFinalYaw(FleetCommandMask fleet) => (fleet & FleetCommandMask.TakeEvasiveAction) != 0;
    public static ShipCommand RequiredToken(string cardId) => cardId switch
    {
        "all-fighters-follow-me" => ShipCommand.Squadron,
        "intensify-firepower" => ShipCommand.ConcentrateFire,
        "shields-to-maximum" => ShipCommand.Engineering,
        _ => ShipCommand.Navigate
    };
}

/// <summary>Port of RaidRules functional behavior. Geometry2D.Distance supplies its independent
/// footprint distance calculation; this module only requires the precomputed distances.</summary>
public static class RaidEffects
{
    public static int DiscardLimitFor(int shipOwner, ReadOnlySpan<NearbyUpgradeSource> sources)
    {
        foreach (var source in sources)
            if (!source.Destroyed && source.Owner != shipOwner && source.Card is { Active: true }
                && string.Equals(source.Card.Definition.id, "captain-rex-officer", StringComparison.OrdinalIgnoreCase)
                && source.DistanceToSubjectMillimetres <= TabletopMeasurements.MediumRange) return 1;
        return int.MaxValue;
    }
    public static int BlackDiceFromRaidTokens(int raidTokenCount)
    {
        if (raidTokenCount < 0) throw new ArgumentOutOfRangeException(nameof(raidTokenCount));
        return Math.Min(2, raidTokenCount);
    }
}
