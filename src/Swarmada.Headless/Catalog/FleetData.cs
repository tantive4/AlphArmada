using System.Collections.ObjectModel;
using System.Text.Json;
using Armada.Data;
using Armada.Fleet;
using Armada.Session;

namespace Swarmada.Headless.Catalog;

/// <summary>Original content-layer fields that are separate from the ship's serialized definition.</summary>
public sealed record ShipCatalogEntry(string Id, ShipDefinition Definition, ShipFleetMetadata Metadata,
    int ContainTokens, int SalvoTokens, int ScatterTokens, string SourceObject);

public sealed record SquadronCatalogEntry(string Id, SquadronDefinition Definition, SquadronFleetMetadata Metadata,
    int BraceTokens, int ScatterTokens, int EvadeTokens, string SourceObject);

public sealed record ObjectiveCatalogEntry(string Id, string DisplayName, ObjectiveCategory Category,
    string Setup, string SpecialRule, string EndOfRound, string EndOfGame,
    bool UsesObjectiveTokens, int VictoryTokenPoints);

/// <summary>
/// Loads the asset-derived catalogs explicitly. Definitions are immutable; no Unity runtime,
/// rendering resources, singleton asset lookup, or reflection is needed to use them.
/// </summary>
public sealed class FleetData : IMatchContentResolver
{
    public IReadOnlyDictionary<string, ShipDefinition> Ships { get; }
    public IReadOnlyDictionary<string, SquadronDefinition> Squadrons { get; }
    public IReadOnlyList<ShipCatalogEntry> ShipEntries { get; }
    public IReadOnlyList<SquadronCatalogEntry> SquadronEntries { get; }
    public IReadOnlyList<ObjectiveCatalogEntry> Objectives { get; }

    private FleetData(ShipCatalogEntry[] ships, SquadronCatalogEntry[] squadrons, ObjectiveCatalogEntry[] objectives)
    {
        ShipEntries = Array.AsReadOnly(ships);
        SquadronEntries = Array.AsReadOnly(squadrons);
        Objectives = Array.AsReadOnly(objectives);
        var shipMap = new Dictionary<string, ShipDefinition>(StringComparer.OrdinalIgnoreCase);
        foreach (var entry in ships)
            if (!shipMap.TryAdd(entry.Definition.ShipId, entry.Definition))
                throw new InvalidDataException("Duplicate ship definition ID: " + entry.Definition.ShipId);
        var squadronMap = new Dictionary<string, SquadronDefinition>(StringComparer.OrdinalIgnoreCase);
        foreach (var entry in squadrons)
            if (!squadronMap.TryAdd(entry.Definition.SquadronId, entry.Definition))
                throw new InvalidDataException("Duplicate squadron definition ID: " + entry.Definition.SquadronId);
        // Catalog entry aliases can intentionally differ from the serialized definition's ID.
        foreach (var entry in squadrons)
            if (!string.Equals(entry.Id, entry.Definition.SquadronId, StringComparison.OrdinalIgnoreCase) &&
                !squadronMap.TryAdd(entry.Id, entry.Definition))
                throw new InvalidDataException("Conflicting squadron alias: " + entry.Id);
        Ships = new ReadOnlyDictionary<string, ShipDefinition>(shipMap);
        Squadrons = new ReadOnlyDictionary<string, SquadronDefinition>(squadronMap);
    }

    public static FleetData LoadDirectory(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        return Load(Path.Combine(directory, "ships.json"), Path.Combine(directory, "squadrons.json"),
            Path.Combine(directory, "objectives.json"));
    }

    public static FleetData Load(string shipsPath, string squadronsPath, string objectivesPath)
    {
        using var shipsJson = JsonDocument.Parse(File.ReadAllBytes(shipsPath), new JsonDocumentOptions { MaxDepth = 64 });
        using var squadronsJson = JsonDocument.Parse(File.ReadAllBytes(squadronsPath), new JsonDocumentOptions { MaxDepth = 64 });
        using var objectivesJson = JsonDocument.Parse(File.ReadAllBytes(objectivesPath), new JsonDocumentOptions { MaxDepth = 64 });
        var ships = shipsJson.RootElement.EnumerateArray().Select(row =>
        {
            var definition = ShipDefinition.FromDefinitionFields(row.GetProperty("definitionFields"));
            var metadata = new ShipFleetMetadata(Text(row, "size"), Strings(row, "traits"), row.GetProperty("flotilla").GetBoolean());
            return new ShipCatalogEntry(Text(row, "id"), definition, metadata, Int(row, "containTokens"),
                Int(row, "salvoTokens"), Int(row, "scatterTokens"), Text(row, "definitionObject"));
        }).ToArray();
        var squadrons = squadronsJson.RootElement.EnumerateArray().Select(row =>
        {
            var definition = SquadronDefinition.FromDefinitionFields(row.GetProperty("definitionFields"));
            int brace = Int(row, "braceTokens"), scatter = Int(row, "scatterTokens"), evade = Int(row, "evadeTokens");
            var metadata = new SquadronFleetMetadata(Strings(row, "uniqueClasses"), brace + scatter + evade,
                hasCompleteDefenseTokenMetadata: true);
            return new SquadronCatalogEntry(Text(row, "id"), definition, metadata, brace, scatter, evade, Text(row, "definitionObject"));
        }).ToArray();
        var objectives = objectivesJson.RootElement.EnumerateArray().Select(row =>
            new ObjectiveCatalogEntry(Text(row, "id"), Text(row, "displayName"),
                (ObjectiveCategory)Int(row, "category"), Text(row, "setup"), Text(row, "specialRule"),
                Text(row, "endOfRound"), Text(row, "endOfGame"), row.GetProperty("usesObjectiveTokens").GetBoolean(),
                Int(row, "victoryTokenPoints"))).ToArray();
        return new FleetData(ships, squadrons, objectives);
    }

    /// <summary>
    /// Uses the recovered fleet legality/import catalog. Its GameplayCoverage classification refers
    /// to the source game's implementation, not completion of this headless battle transition port.
    /// Load UpgradeCatalog explicitly before calling this method.
    /// </summary>
    public ImmutableFleetContentCatalog BuildFleetCatalog()
    {
        var shipMetadata = ShipEntries.ToDictionary(x => x.Id, x => x.Metadata, StringComparer.OrdinalIgnoreCase);
        var squadronMetadata = SquadronEntries.ToDictionary(x => x.Definition.SquadronId, x => x.Metadata, StringComparer.OrdinalIgnoreCase);
        var aliases = SquadronEntries.Where(x => !string.Equals(x.Id, x.Definition.SquadronId, StringComparison.OrdinalIgnoreCase))
            .Select(x => new FleetContentAlias(FleetContentKind.Squadron, x.Id, x.Definition.SquadronId));
        var objectives = Objectives.Select(x => new ObjectiveContentDescriptor(x.Id, x.DisplayName, x.Category,
            OfficialContentDisposition.StrictOfficial, GameplayCoverage.DisplayOnly));
        return FleetContentAdapter.Build(ShipEntries.Select(x => x.Definition), SquadronEntries.Select(x => x.Definition),
            objectives, shipMetadata, squadronMetadata, aliases);
    }

    public bool TryResolveShip(string definitionId, out MatchUnitValidationRules rules)
    {
        if (Ships.TryGetValue(definitionId, out var definition))
        {
            rules = new MatchUnitValidationRules(definition.ChartedMaximumSpeed, definition.GameplayFootprint.X,
                definition.GameplayFootprint.Y, definition.HasAuxiliaryHullZones);
            return true;
        }
        rules = default;
        return false;
    }

    public bool TryResolveSquadron(string definitionId, out MatchUnitValidationRules rules)
    {
        if (Squadrons.TryGetValue(definitionId, out var definition))
        {
            rules = new MatchUnitValidationRules(0, definition.GameplayFootprint.X, definition.GameplayFootprint.Y);
            return true;
        }
        rules = default;
        return false;
    }

    private static string Text(JsonElement row, string field) => row.GetProperty(field).GetString() ?? string.Empty;
    private static int Int(JsonElement row, string field) => row.GetProperty(field).GetInt32();
    private static string[] Strings(JsonElement row, string field) => row.GetProperty(field).EnumerateArray().Select(x => x.GetString() ?? string.Empty).ToArray();
}
