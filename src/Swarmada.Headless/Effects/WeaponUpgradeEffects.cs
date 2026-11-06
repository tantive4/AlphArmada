using Armada.Core;
using Armada.Rules;

namespace Swarmada.Headless.Effects;

/// <summary>Unrolled battery counts, before dice are rolled. Not a substitute for a live attack pool.</summary>
public readonly record struct DiceBattery
{
    public int Red { get; }
    public int Blue { get; }
    public int Black { get; }
    public int Total => checked(Red + Blue + Black);
    public DiceBattery(int red, int blue, int black)
    {
        if (red < 0 || blue < 0 || black < 0) throw new ArgumentOutOfRangeException(nameof(red), "Dice counts cannot be negative.");
        _ = checked(red + blue + black);
        Red = red; Blue = blue; Black = black;
    }
    public bool Has(AttackDieColor color) => color switch
    {
        AttackDieColor.Red => Red > 0, AttackDieColor.Blue => Blue > 0, AttackDieColor.Black => Black > 0, _ => false
    };
}

/// <summary>Port of Armada.Rules.IonCannonRules. Card ownership/activity is explicit.</summary>
public static class IonCannonEffects
{
    public static int UnspentAccuracyDamage(bool sw7IonBatteriesActive, int unspentAccuracies)
        => sw7IonBatteriesActive ? Math.Max(0, unspentAccuracies) : 0;
}

/// <summary>Pure/runtime-input port of SWArmada 0.2.7 TurbolaserRules. These kernels do not choose
/// player options, find Unity objects, or schedule attack timing windows.</summary>
public static class TurbolaserEffects
{
    /// <param name="targetEngagedByAttackerFriendlySquadron">Must include every surviving friendly squadron;
    /// engagement geometry and keyword exceptions are resolved by the caller.</param>
    public static bool ConvertsFlakBlueToRed(bool heavyFireZoneActive, bool targetIsSquadron,
        bool targetEngagedByAttackerFriendlySquadron)
        => heavyFireZoneActive && targetIsSquadron && !targetEngagedByAttackerFriendlySquadron;
    public static DiceBattery EffectiveFlakArmament(DiceBattery printed, bool heavyFireZoneActive,
        bool targetIsSquadron, bool targetEngagedByAttackerFriendlySquadron, bool defendingMine = false)
        => (defendingMine ? heavyFireZoneActive : ConvertsFlakBlueToRed(heavyFireZoneActive,
            targetIsSquadron, targetEngagedByAttackerFriendlySquadron))
            ? new(checked(printed.Red + printed.Blue), 0, printed.Black) : printed;
    public static bool LimitsBraceReduction(bool heavyTurbolaserTurretsActive, bool braceSpent, int defenseTokenTypesSpent)
        => heavyTurbolaserTurretsActive && braceSpent && defenseTokenTypesSpent > 1;
    public static int MaximumRedirectDamagePerZone(bool xi7TurbolasersActive) => xi7TurbolasersActive ? 1 : int.MaxValue;
    /// <summary>The caller supplies effective batteries for every zone adjacent to the focus zone.
    /// This preserves upgrades that change a printed battery before Swivel Mounts inspects it.</summary>
    public static IReadOnlyList<AttackDieColor> SwivelMountFocusColors(ReadOnlySpan<DiceBattery> adjacentEffectiveBatteries)
    {
        var red = false; var blue = false; var black = false;
        foreach (var battery in adjacentEffectiveBatteries)
        { red |= battery.Red > 0; blue |= battery.Blue > 0; black |= battery.Black > 0; }
        return Colors(red, blue, black);
    }
    public static bool IsSwivelMountAdjacentAttack(UpgradeCardRuntimeState? swivelMount,
        HullZone attackZone, bool auxiliaryHullZones)
        => swivelMount is { Active: true, Exhausted: true, FocusZone: not null }
            && HullZoneRules.Adjacent(attackZone, auxiliaryHullZones).Contains(swivelMount.FocusZone.Value);
    public static IReadOnlyList<AttackDieColor> SwivelMountFocusBonusColors(UpgradeCardRuntimeState? swivelMount,
        HullZone attackZone, ReadOnlySpan<DiceBattery> adjacentEffectiveBatteries)
        => swivelMount is { Active: true, Exhausted: true } && swivelMount.FocusZone == attackZone
            ? SwivelMountFocusColors(adjacentEffectiveBatteries) : Array.Empty<AttackDieColor>();
    /// <summary>Adds exactly the player-selected eligible die, preserving AttackResolver's seeded roll behavior.</summary>
    public static AttackResolution ApplySwivelMountFocusBonus(UpgradeCardRuntimeState? swivelMount,
        HullZone attackZone, ReadOnlySpan<DiceBattery> adjacentEffectiveBatteries, AttackDieColor selectedColor,
        AttackResolution attack, int seed, out AttackDieColor addedColor)
    {
        addedColor = SwivelMountFocusBonusColors(swivelMount, attackZone, adjacentEffectiveBatteries).Contains(selectedColor)
            ? selectedColor : AttackDieColor.None;
        return addedColor == AttackDieColor.None ? attack : AttackResolver.AddRolledDice(attack, addedColor, 1, seed);
    }
    public static IReadOnlyList<AttackDieColor> SwivelMountRemovalColors(DiceBattery battery)
        => Colors(battery.Red > 0, battery.Blue > 0, battery.Black > 0);
    public static bool TryRemoveSwivelMountDie(AttackDieColor color, DiceBattery battery, out DiceBattery remaining)
    {
        remaining = battery;
        if (!battery.Has(color)) return false;
        remaining = color switch
        {
            AttackDieColor.Red => new(battery.Red - 1, battery.Blue, battery.Black),
            AttackDieColor.Blue => new(battery.Red, battery.Blue - 1, battery.Black),
            AttackDieColor.Black => new(battery.Red, battery.Blue, battery.Black - 1),
            _ => battery
        };
        return true;
    }
    private static IReadOnlyList<AttackDieColor> Colors(bool red, bool blue, bool black)
    {
        var result = new List<AttackDieColor>(3);
        if (red) result.Add(AttackDieColor.Red);
        if (blue) result.Add(AttackDieColor.Blue);
        if (black) result.Add(AttackDieColor.Black);
        return result;
    }
}

/// <summary>Port of OffensiveRetrofitRules; distances are measured by the caller's hull-zone geometry.</summary>
public static class OffensiveRetrofitEffects
{
    public static int SizeRank(string? sizeClass) => sizeClass?.Trim().ToLowerInvariant() switch
    { "small" => 1, "medium" => 2, "large" => 3, "huge" => 4, _ => 0 };
    public static bool PreventsCollisionDamage(bool hardenedBulkheadsActive, string? protectedSizeClass, string? otherSizeClass)
        => hardenedBulkheadsActive && SizeRank(protectedSizeClass) > SizeRank(otherSizeClass);
    public static DiceBattery EffectiveAntiSquadronArmament(bool flakGunsActive, DiceBattery printed)
        => flakGunsActive ? new(0, 0, printed.Total) : printed;
    public static bool GrantsCounterOneAgainstSquadron(bool quadLaserTurretsActive, bool attackerIsSquadron,
        float hullZoneDistanceMillimetres)
        => quadLaserTurretsActive && attackerIsSquadron && hullZoneDistanceMillimetres <= 77;
}

/// <summary>Complete pure IgnitionDefenseRules behavior, without UI dependencies.</summary>
public static class IgnitionDefenseEffects
{
    public static bool AllowsSalvo(bool isIgnitionAttack) => !isIgnitionAttack;
    public static int ExtremeEvadeCancellationCount(bool foresightActive) => foresightActive ? 3 : 2;
}
