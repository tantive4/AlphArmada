using Armada.Core;
using Armada.Rules;
namespace Swarmada.Headless.Engine;

/// <summary>Former source singleton rule tables, now owned by one branchable game.</summary>
public sealed class SourceRoundState
{
    public void CopyFrom(SourceRoundState other)
    {
        ArgumentNullException.ThrowIfNull(other); if (ReferenceEquals(this, other)) return;
        SnapshotCollections.CopyInto(other.TrenchCommands, TrenchCommands);
        SnapshotCollections.CopyInto(other.DravenCommands, DravenCommands);
        SnapshotCollections.CopyInto(other.ThrawnCommands, ThrawnCommands);
        SnapshotCollections.CopyInto(other.AdmiralTarkinDials, AdmiralTarkinDials);
        SnapshotCollections.CopyInto(other.PalpatineTokens, PalpatineTokens);
        foreach (var entry in PalpatineRemaining)
            if (!other.PalpatineRemaining.ContainsKey(entry.Key)) PalpatineRemaining.Remove(entry.Key);
        foreach (var entry in other.PalpatineRemaining)
        {
            if (!PalpatineRemaining.TryGetValue(entry.Key, out var tokens)) PalpatineRemaining.Add(entry.Key, tokens = new(entry.Value.Count));
            tokens.Clear(); tokens.EnsureCapacity(entry.Value.Count);
            foreach (var token in entry.Value) tokens.Add(token);
        }
    }

    public Dictionary<int, ShipCommand> TrenchCommands { get; set; } = new();
    public Dictionary<int, ShipCommand> DravenCommands { get; set; } = new();
    public Dictionary<int, ShipCommand> ThrawnCommands { get; set; } = new();
    public Dictionary<int, ShipCommand> AdmiralTarkinDials { get; set; } = new();
    public Dictionary<int, DefenseTokenType> PalpatineTokens { get; set; } = new();
    public Dictionary<int, HashSet<DefenseTokenType>> PalpatineRemaining { get; set; } = new();
    public SourceRoundState DeepClone() => new()
    {
        TrenchCommands = new(TrenchCommands), DravenCommands = new(DravenCommands), ThrawnCommands = new(ThrawnCommands),
        AdmiralTarkinDials = new(AdmiralTarkinDials), PalpatineTokens = new(PalpatineTokens),
        PalpatineRemaining = PalpatineRemaining.ToDictionary(x => x.Key, x => new HashSet<DefenseTokenType>(x.Value))
    };
}
