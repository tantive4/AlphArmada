using System.Collections.ObjectModel;
using System.Diagnostics.CodeAnalysis;
using System.Text.Json;

namespace Armada.Data;

/// <summary>Explicitly loaded, process-wide official catalog. No Unity resource loading.</summary>
public static class UpgradeCatalog
{
    public const string OfficialRuleset = "official";
    public const int ExpectedOfficialCardCount = 292;
    private sealed record CatalogState(string SourceHash, ReadOnlyCollection<UpgradeCardDefinition> Cards,
        IReadOnlyDictionary<string, UpgradeCardDefinition> ById);
    private static readonly object Gate = new();
    private static CatalogState? current;

    private static CatalogState Current => Volatile.Read(ref current) ??
        throw new InvalidOperationException("Load the official upgrade catalog explicitly before using upgrade rules.");
    public static IReadOnlyList<UpgradeCardDefinition> Cards => Current.Cards;
    public static string Ruleset => OfficialRuleset;
    public static string SourceSha256 => Current.SourceHash;
    public static bool IsLoaded => Volatile.Read(ref current) is not null;

    public static void LoadJson(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        Load(stream);
    }

    public static void Load(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        var document = JsonSerializer.Deserialize<UpgradeCatalogDocument>(stream,
            new JsonSerializerOptions { IncludeFields = true, MaxDepth = 64 }) ??
            throw new InvalidDataException("Missing upgrade catalog document.");
        if (document.schemaVersion != 1 || document.ruleset != OfficialRuleset ||
            document.cards is null || document.cards.Length != ExpectedOfficialCardCount ||
            string.IsNullOrWhiteSpace(document.sourceSha256))
            throw new InvalidDataException("Expected official schema 1, source hash, and exactly 292 cards (fighter-group content removed).");
        var byId = new Dictionary<string, UpgradeCardDefinition>(StringComparer.OrdinalIgnoreCase);
        foreach (var card in document.cards)
        {
            if (card is null || string.IsNullOrWhiteSpace(card.id) || string.IsNullOrWhiteSpace(card.name) ||
                card.points < 0 || card.id.EndsWith("-fighter-group", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(card.slot, "leader", StringComparison.OrdinalIgnoreCase) || !byId.TryAdd(card.id, card))
                throw new InvalidDataException("Invalid or duplicate card ID in upgrade catalog.");
        }
        var replacement = new CatalogState(document.sourceSha256,
            Array.AsReadOnly(document.cards), new ReadOnlyDictionary<string, UpgradeCardDefinition>(byId));
        lock (Gate)
        {
            // A running search must never change the definitions behind shared immutable card references.
            if (current is not null && current.SourceHash != replacement.SourceHash)
                throw new InvalidOperationException("Cannot replace a loaded catalog with a different rules version.");
            current ??= replacement;
        }
    }

    public static bool TryGet(string? id, [NotNullWhen(true)] out UpgradeCardDefinition? definition) =>
        Current.ById.TryGetValue(id ?? string.Empty, out definition);
    public static UpgradeCardDefinition GetRequired(string id) => TryGet(id, out var card) ? card :
        throw new KeyNotFoundException($"Unknown upgrade ID '{id}'.");
    public static bool IsFactionLegal(UpgradeCardDefinition? card, ShipFaction faction) =>
        card?.factions is null || card.factions.Length == 0 || card.factions.Any(id =>
            string.IsNullOrEmpty(id) || string.Equals(id, ShipFactionCatalog.Id(faction), StringComparison.OrdinalIgnoreCase));
}
