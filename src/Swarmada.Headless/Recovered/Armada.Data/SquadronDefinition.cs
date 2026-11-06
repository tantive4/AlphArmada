using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json;

namespace Armada.Data;

/// <summary>Immutable squadron rules data. Single-model classification retains the recovered algorithm.</summary>
public sealed class SquadronDefinition
{
    private static readonly string[] FormationWords =
    {
        "squad", "squads", "squadron", "squadrons", "wing", "wings", "flight", "flights", "fighters", "bombers",
        "interceptors", "prototypes", "patrol", "patrols", "group", "groups"
    };
    private readonly bool singleCraft;
    public string SquadronId { get; }
    public string DisplayName { get; }
    public ShipFaction Faction { get; }
    public string FactionLabel => ShipFactionCatalog.Label(Faction);
    public int Points { get; }
    public int Hull { get; }
    public int Speed { get; }
    public BatteryArmament AntiSquadronArmament { get; }
    public BatteryArmament AntiShipArmament { get; }
    public int AntiSquadronBlueDice => AntiSquadronArmament.BlueDice;
    public int AntiShipRedDice => AntiShipArmament.RedDice;
    public int AntiShipBlueDice => AntiShipArmament.BlueDice;
    public IReadOnlyList<string> Keywords { get; }
    public string AbilityText { get; }
    public bool IsUnique { get; }
    public bool IsSingleCraft => UsesSingleModel(DisplayName, Faction, IsUnique, singleCraft);
    public int BraceTokens { get; }
    public int ScatterTokens { get; }
    public Vector2 GameplayFootprint { get; }

    private SquadronDefinition(JsonElement fields)
    {
        SquadronId = fields.GetProperty("squadronId").GetString() ?? throw new InvalidDataException("Missing squadron ID.");
        DisplayName = fields.GetProperty("displayName").GetString() ?? SquadronId;
        Faction = (ShipFaction)fields.GetProperty("faction").GetInt32();
        Points = fields.GetProperty("points").GetInt32();
        Hull = fields.GetProperty("hull").GetInt32();
        Speed = fields.GetProperty("speed").GetInt32();
        AntiSquadronArmament = BatteryArmament.Read(fields.GetProperty("antiSquadronArmament"));
        AntiShipArmament = BatteryArmament.Read(fields.GetProperty("antiShipArmament"));
        Keywords = Array.AsReadOnly(fields.GetProperty("keywords").EnumerateArray().Select(x => x.GetString() ?? string.Empty).ToArray());
        AbilityText = fields.GetProperty("abilityText").GetString() ?? string.Empty;
        IsUnique = fields.GetProperty("unique").GetBoolean();
        singleCraft = fields.GetProperty("singleCraft").GetBoolean();
        BraceTokens = fields.GetProperty("braceTokens").GetInt32();
        ScatterTokens = fields.GetProperty("scatterTokens").GetInt32();
        GameplayFootprint = IgnitionProfile.ReadVector(fields.GetProperty("gameplayFootprint"));
    }

    public static SquadronDefinition FromDefinitionFields(JsonElement fields) => new(fields);
    public static bool UsesSingleModel(string title) => !ContainsFormationWord(title);
    public static bool UsesSingleModel(string title, ShipFaction faction, bool isUnique, bool sourceSingleCraft = false)
    {
        bool factionAce = isUnique && (faction == ShipFaction.Republic || faction == ShipFaction.Separatist);
        if (factionAce && ContainsFormationWord(title)) return false;
        if (!(sourceSingleCraft || factionAce)) return UsesSingleModel(title);
        return true;
    }
    private static bool ContainsFormationWord(string title) => FormationWords.Any(word => ContainsStandaloneWord(title, word));
    private static bool ContainsStandaloneWord(string value, string word)
    {
        if (string.IsNullOrWhiteSpace(value) || string.IsNullOrWhiteSpace(word)) return false;
        int start = 0;
        while (start < value.Length)
        {
            int index = value.IndexOf(word, start, StringComparison.OrdinalIgnoreCase);
            if (index < 0) return false;
            bool previousIsWord = index > 0 && char.IsLetterOrDigit(value[index - 1]);
            int end = index + word.Length;
            bool nextIsWord = end < value.Length && char.IsLetterOrDigit(value[end]);
            if (!previousIsWord && !nextIsWord) return true;
            start = end;
        }
        return false;
    }
}
