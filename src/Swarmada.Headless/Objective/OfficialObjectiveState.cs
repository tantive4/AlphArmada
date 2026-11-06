using System.Text.Json.Serialization;

namespace Swarmada.Headless.Objective;

public sealed record OfficialTimingSequence(int Player, Engine.GameContinuation[] Frames)
{
    public OfficialTimingSequence DeepClone() => new(Player, Frames.Select(f => f.DeepClone()).ToArray());
}
public sealed record OfficialObjectiveSnapshot(string Id, int[] EntityTokens, bool[] ObjectiveShips,
    int[] VictoryTokens, Dictionary<int, string> PieceKinds, Dictionary<int, int> PieceHull,
    int[] DestroyedPieces, int[] ReservedEntities, Dictionary<string, long> Flags,
    Dictionary<long, OfficialTimingSequence>? TimingWindows = null, Dictionary<int, int>? PieceEntities = null,
    Dictionary<int, int>? ExogorthHosts = null, Engine.InvalidSetupResult? InvalidSetup = null) : ObjectiveSnapshot;

/// <summary>One supplied official objective per match. All mutable data is branch-owned.</summary>
public sealed class OfficialObjectiveState : IObjectiveRuntime
{
    public string Id { get; }
    public int[] EntityTokens { get; }
    public bool[] ObjectiveShips { get; }
    public int[] VictoryTokens { get; } = new int[2];
    public Dictionary<int, string> PieceKinds { get; } = new();
    public Dictionary<int, int> PieceHull { get; } = new();
    public Dictionary<int, int> PieceEntities { get; } = new();
    public Dictionary<int, int> ExogorthHosts { get; } = new();
    public Engine.InvalidSetupResult? InvalidSetup { get; set; }
    public HashSet<int> DestroyedPieces { get; } = new();
    public List<int> ReservedEntities { get; } = new();
    public Dictionary<string, long> Flags { get; } = new(StringComparer.Ordinal);
    public Dictionary<long, OfficialTimingSequence> TimingWindows { get; } = new();
    public OfficialObjectiveState(string id, int entities)
    { Id = id; EntityTokens = new int[entities]; ObjectiveShips = new bool[entities]; }
    public ObjectiveSnapshot CaptureSnapshot() => new OfficialObjectiveSnapshot(Id, (int[])EntityTokens.Clone(),
        (bool[])ObjectiveShips.Clone(), (int[])VictoryTokens.Clone(), new(PieceKinds), new(PieceHull),
        DestroyedPieces.ToArray(), ReservedEntities.ToArray(), new(Flags, StringComparer.Ordinal),
        TimingWindows.ToDictionary(p => p.Key, p => p.Value.DeepClone()), new(PieceEntities), new(ExogorthHosts), InvalidSetup);
    public static OfficialObjectiveState FromSnapshot(OfficialObjectiveSnapshot snapshot)
    {
        if (!Engine.OfficialObjectives.SupportedIds.Contains(snapshot.Id, StringComparer.Ordinal) ||
            snapshot.EntityTokens.Length != snapshot.ObjectiveShips.Length || snapshot.VictoryTokens.Length != 2 ||
            snapshot.EntityTokens.Any(x => x < 0) || snapshot.VictoryTokens.Any(x => x < 0))
            throw new ArgumentException("Invalid official objective snapshot.");
        if (snapshot.InvalidSetup is { } invalid && (invalid.ObjectiveId != snapshot.Id || invalid.Player is not (1 or 2) ||
            invalid.Entity < 0 || invalid.Entity >= snapshot.EntityTokens.Length || !Enum.IsDefined(invalid.Code) || string.IsNullOrWhiteSpace(invalid.Message)))
            throw new ArgumentException("Invalid rejected-setup snapshot.");
        var result = new OfficialObjectiveState(snapshot.Id, snapshot.EntityTokens.Length);
        snapshot.EntityTokens.CopyTo(result.EntityTokens, 0); snapshot.ObjectiveShips.CopyTo(result.ObjectiveShips, 0);
        snapshot.VictoryTokens.CopyTo(result.VictoryTokens, 0);
        foreach (var item in snapshot.PieceKinds) result.PieceKinds.Add(item.Key, item.Value);
        foreach (var item in snapshot.PieceHull) result.PieceHull.Add(item.Key, item.Value);
        result.DestroyedPieces.UnionWith(snapshot.DestroyedPieces); result.ReservedEntities.AddRange(snapshot.ReservedEntities);
        foreach (var item in snapshot.Flags) result.Flags.Add(item.Key, item.Value);
        if (snapshot.TimingWindows is not null) foreach (var item in snapshot.TimingWindows) result.TimingWindows.Add(item.Key, item.Value.DeepClone());
        if (snapshot.PieceEntities is not null) foreach (var item in snapshot.PieceEntities) result.PieceEntities.Add(item.Key, item.Value);
        if (snapshot.ExogorthHosts is not null) foreach (var item in snapshot.ExogorthHosts) result.ExogorthHosts.Add(item.Key, item.Value);
        else foreach (var item in snapshot.Flags)
            if (item.Key.StartsWith("creature.host.", StringComparison.Ordinal) && int.TryParse(item.Key.AsSpan("creature.host.".Length), out int exogorth))
                result.ExogorthHosts[exogorth] = checked((int)item.Value);
        result.InvalidSetup = snapshot.InvalidSetup;
        return result;
    }
    public void RestoreSnapshot(ObjectiveSnapshot snapshot)
        => CopyFrom(snapshot is OfficialObjectiveSnapshot official ? FromSnapshot(official) : throw new ArgumentException("Objective snapshot type differs."));
    public IObjectiveRuntime DeepClone()
    { var result = new OfficialObjectiveState(Id, EntityTokens.Length); result.CopyFrom(this); return result; }
    public void CopyFrom(IObjectiveRuntime source)
    {
        if (ReferenceEquals(this, source)) return;
        if (source is not OfficialObjectiveState other || other.Id != Id || other.EntityTokens.Length != EntityTokens.Length)
            throw new ArgumentException("Official objective identity or entity layout differs.");
        other.EntityTokens.CopyTo(EntityTokens, 0); other.ObjectiveShips.CopyTo(ObjectiveShips, 0);
        other.VictoryTokens.CopyTo(VictoryTokens, 0);
        CopyDictionary(other.PieceKinds, PieceKinds); CopyDictionary(other.PieceHull, PieceHull); CopyDictionary(other.Flags, Flags);
        CopyDictionary(other.PieceEntities, PieceEntities);
        CopyDictionary(other.ExogorthHosts, ExogorthHosts); InvalidSetup = other.InvalidSetup;
        if (other.TimingWindows.Count == 0) TimingWindows.Clear();
        else
        {
            List<long>? stale = null;
            foreach (long key in TimingWindows.Keys)
                if (!other.TimingWindows.ContainsKey(key)) (stale ??= []).Add(key);
            if (stale is not null) foreach (long key in stale) TimingWindows.Remove(key);
            foreach (var item in other.TimingWindows)
            {
                if (!TimingWindows.TryGetValue(item.Key, out var target) || target.Player != item.Value.Player || target.Frames.Length != item.Value.Frames.Length ||
                    ReferenceEquals(target.Frames, item.Value.Frames))
                { TimingWindows[item.Key] = item.Value.DeepClone(); continue; }
                for (int i = 0; i < target.Frames.Length; i++)
                {
                    var frame = item.Value.Frames[i]; var old = target.Frames[i];
                    if (frame.Indices is null) { target.Frames[i] = frame; continue; }
                    if (old.Rule == frame.Rule && old.Source == frame.Source && old.Target == frame.Target && old.Stage == frame.Stage &&
                        old.Value == frame.Value && old.Text == frame.Text && old.Indices is { } indices && indices.Length == frame.Indices.Length &&
                        (indices.Length == 0 || !ReferenceEquals(indices, frame.Indices))) frame.Indices.CopyTo(indices, 0);
                    else target.Frames[i] = frame.DeepClone();
                }
            }
        }
        DestroyedPieces.Clear(); DestroyedPieces.UnionWith(other.DestroyedPieces);
        ReservedEntities.Clear(); ReservedEntities.AddRange(other.ReservedEntities);
    }
    private static void CopyDictionary<TKey, TValue>(Dictionary<TKey, TValue> source, Dictionary<TKey, TValue> target) where TKey : notnull
    { target.Clear(); foreach (var item in source) target.Add(item.Key, item.Value); }
}
