using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json;
using Armada.Core;

namespace Armada.Data;

/// <summary>
/// Immutable gameplay definition loaded from the original serialized fields. Unity art, colors,
/// and editor mutation methods are deliberately outside the simulation data model.
/// </summary>
public sealed class ShipDefinition
{
    public string ShipId { get; }
    public string DisplayName { get; }
    public ShipFaction Faction { get; }
    public string ShipType { get; }
    public IReadOnlyList<string> UpgradeSlots { get; }
    public string FactionLabel => ShipFactionCatalog.Label(Faction);
    public int Points { get; }
    public int Hull { get; }
    public int FrontShields { get; }
    public int LeftShields { get; }
    public int RightShields { get; }
    public int RearShields { get; }
    public bool HasAuxiliaryHullZones { get; }
    public int LeftAuxiliaryShields { get; }
    public int RightAuxiliaryShields { get; }
    public IReadOnlyList<HullZone> HullZones => HullZoneRules.Zones(HasAuxiliaryHullZones);
    public int Command { get; }
    public int Engineering { get; }
    public int Squadron { get; }
    public int MaximumSpeed { get; }
    public int BraceTokens { get; }
    public int RedirectTokens { get; }
    public int EvadeTokens { get; }
    public int FrontRedDice => frontBattery.RedDice;
    public int FrontBlueDice => frontBattery.BlueDice;
    public int FrontBlackDice => frontBattery.BlackDice;
    public BatteryArmament AntiSquadronArmament { get; }
    public IgnitionProfile? Ignition { get; }
    public bool HasIgnition => Ignition?.IsConfigured ?? false;
    public Vector2 GameplayFootprint { get; }
    public IReadOnlyList<ManeuverSpeedRow> ManeuverChart { get; }
    private readonly BatteryArmament frontBattery, leftBattery, rightBattery, rearBattery,
        leftAuxiliaryBattery, rightAuxiliaryBattery;

    public int ChartedMaximumSpeed
    {
        get
        {
            for (int speed = ManeuverChart.Count; speed >= 1; speed--)
                if (ManeuverChart[speed - 1].IsAvailable) return speed;
            return MaximumSpeed;
        }
    }
    public bool HasManeuverChart => ManeuverChart.Any(row => row.IsAvailable);
    public bool HasHullZone(HullZone zone) => zone >= HullZone.Front && (int)zone < (HasAuxiliaryHullZones ? 6 : 4);

    private ShipDefinition(JsonElement fields)
    {
        ShipId = fields.GetProperty("shipId").GetString() ?? throw new InvalidDataException("Missing ship ID.");
        DisplayName = fields.GetProperty("displayName").GetString() ?? ShipId;
        Faction = (ShipFaction)fields.GetProperty("faction").GetInt32();
        ShipType = fields.GetProperty("shipType").GetString() ?? string.Empty;
        UpgradeSlots = Array.AsReadOnly(fields.GetProperty("upgradeSlots").EnumerateArray().Select(x => x.GetString() ?? string.Empty).ToArray());
        Points = fields.GetProperty("points").GetInt32();
        Hull = fields.GetProperty("hull").GetInt32();
        FrontShields = fields.GetProperty("frontShields").GetInt32();
        LeftShields = fields.GetProperty("leftShields").GetInt32();
        RightShields = fields.GetProperty("rightShields").GetInt32();
        RearShields = fields.GetProperty("rearShields").GetInt32();
        HasAuxiliaryHullZones = fields.GetProperty("hasAuxiliaryHullZones").GetBoolean();
        LeftAuxiliaryShields = fields.GetProperty("leftAuxiliaryShields").GetInt32();
        RightAuxiliaryShields = fields.GetProperty("rightAuxiliaryShields").GetInt32();
        Command = fields.GetProperty("command").GetInt32();
        Engineering = fields.GetProperty("engineering").GetInt32();
        Squadron = fields.GetProperty("squadron").GetInt32();
        MaximumSpeed = fields.GetProperty("maximumSpeed").GetInt32();
        BraceTokens = fields.GetProperty("braceTokens").GetInt32();
        RedirectTokens = fields.GetProperty("redirectTokens").GetInt32();
        EvadeTokens = fields.GetProperty("evadeTokens").GetInt32();
        ManeuverChart = Array.AsReadOnly(fields.GetProperty("maneuverChart").EnumerateArray().Select(row =>
            new ManeuverSpeedRow(row.GetProperty("jointYaw").EnumerateArray().Select(x => x.GetInt32()).ToArray())).ToArray());
        frontBattery = BatteryArmament.Read(fields.GetProperty("frontBattery"));
        leftBattery = BatteryArmament.Read(fields.GetProperty("leftBattery"));
        rightBattery = BatteryArmament.Read(fields.GetProperty("rightBattery"));
        rearBattery = BatteryArmament.Read(fields.GetProperty("rearBattery"));
        leftAuxiliaryBattery = BatteryArmament.Read(fields.GetProperty("leftAuxiliaryBattery"));
        rightAuxiliaryBattery = BatteryArmament.Read(fields.GetProperty("rightAuxiliaryBattery"));
        AntiSquadronArmament = BatteryArmament.Read(fields.GetProperty("antiSquadronArmament"));
        if (fields.TryGetProperty("ignitionProfile", out var ignition) && ignition.ValueKind == JsonValueKind.Object)
            Ignition = IgnitionProfile.Read(ignition);
        GameplayFootprint = IgnitionProfile.ReadVector(fields.GetProperty("gameplayFootprint"));
    }

    public static ShipDefinition FromDefinitionFields(JsonElement fields) => new(fields);
    public ManeuverSpeedRow GetManeuverRow(int speed) => speed < 1 || speed > ManeuverChart.Count ? default : ManeuverChart[speed - 1];
    public BatteryArmament GetBattery(HullZone zone) => zone switch
    {
        HullZone.Front => frontBattery,
        HullZone.Left => leftBattery,
        HullZone.Right => rightBattery,
        HullZone.Rear => rearBattery,
        HullZone.LeftAuxiliary => HasAuxiliaryHullZones ? leftAuxiliaryBattery : default,
        HullZone.RightAuxiliary => HasAuxiliaryHullZones ? rightAuxiliaryBattery : default,
        _ => default
    };
}
