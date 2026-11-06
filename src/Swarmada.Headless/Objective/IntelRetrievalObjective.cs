using System.Numerics;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Objective;

public readonly record struct IntelTokenState(int Sequence, Vector2 Position, int? CarrierShipId = null, bool Delivered = false)
{
    public bool Available => CarrierShipId is null && !Delivered;
}
public readonly record struct IntelDownload(int ShipId, int TokenSequence, int StartedRound, bool ContestedThisRound);
public sealed record IntelRetrievalSnapshot(IntelTokenState[] Tokens, IntelDownload[] Downloads,
    KeyValuePair<int, int>[] DepartureLocks, int[] DestructionHandled, int[] DeliveryPending,
    KeyValuePair<int, OrientedRectangle>[] StartingAreas, int PlayerOneDelivered, int PlayerTwoDelivered,
    int Round, bool MatchStarted, ObjectiveOutcome? Outcome) : ObjectiveSnapshot;

/// <summary>Gameplay port of IntelRetrievalObjectiveController plus IntelTokenView's logical
/// capture/release/delivery state. UI animation delays are intentionally collapsed: released tokens
/// become available immediately instead of after 0.72 seconds. Call Observe after every atomic
/// pose/speed/reserve/destruction change to preserve transient contest history. The game handler
/// commits returned RequiredSpeed and schedules each explicit player choice.</summary>
public sealed class IntelRetrievalObjective : IObjectiveRuntime
{
    public const int IntelRequiredToWin = 2;
    private readonly SortedDictionary<int, IntelTokenState> tokens = new();
    private readonly int[] tokenSequences;
    private readonly Dictionary<int, IntelDownload> downloads = new();
    private readonly Dictionary<int, int> departureLocks = new();
    private readonly HashSet<int> destructionHandled = new();
    private readonly HashSet<int> deliveryPending = new();
    private readonly Dictionary<int, OrientedRectangle> startingAreas;
    public int PlayerOneDelivered { get; private set; }
    public int PlayerTwoDelivered { get; private set; }
    public int Round { get; private set; } = 1;
    public bool MatchStarted { get; private set; }
    public ObjectiveOutcome? Outcome { get; private set; }
    public IntelRetrievalObjective(IEnumerable<IntelTokenState> initialTokens, IReadOnlyDictionary<int, OrientedRectangle> startingAreas)
    {
        ArgumentNullException.ThrowIfNull(initialTokens);
        ArgumentNullException.ThrowIfNull(startingAreas);
        foreach (var token in initialTokens)
        {
            Geometry2D.RequireFinite(token.Position, nameof(initialTokens));
            if (token.Sequence < 0 || token.CarrierShipId < 0 || (token.Delivered && token.CarrierShipId is not null)
                || !tokens.TryAdd(token.Sequence, token)) throw new ArgumentException("Invalid/duplicate Intel token.", nameof(initialTokens));
        }
        this.startingAreas = new(startingAreas);
        tokenSequences = tokens.Keys.ToArray();
    }
    public bool IsDownloading(int shipId) => downloads.ContainsKey(shipId);
    public bool IsDeliveryPending(int shipId) => deliveryPending.Contains(shipId);
    public IntelTokenState Token(int sequence) => tokens[sequence];
    public IReadOnlyList<IntelTokenState> Tokens => tokens.Values.ToArray();
    public void BeginMatch(int round = 1) { Round = Math.Max(1, round); MatchStarted = true; Outcome = null; }
    public void BeginRound(int round)
    {
        Round = Math.Max(1, round);
        foreach (var entry in downloads.ToArray()) downloads[entry.Key] = entry.Value with { ContestedThisRound = false };
    }
    public static bool IsRetrieverEligible(string? sizeClass)
        => (sizeClass ?? string.Empty).Trim().ToLowerInvariant() is "" or "small" or "medium" or "flotilla";
    private bool CanBeginDownload(ObjectiveShip ship) => MatchStarted && !ship.Destroyed && !ship.Reserved
        && IsRetrieverEligible(ship.SizeClass) && !downloads.ContainsKey(ship.Id) && !tokens.Values.Any(t => t.CarrierShipId == ship.Id);
    public int? GetDownloadTokenAtPose(ObjectiveShip ship, float requiredOverlap = ObjectiveGeometry.RequiredBaseOverlap, int? requiredToken = null)
    {
        if (!CanBeginDownload(ship)) return null;
        foreach (var token in tokens.Values)
            if (token.Available && (requiredToken is null || requiredToken == token.Sequence)
                && (!departureLocks.TryGetValue(ship.Id, out var locked) || locked != token.Sequence)
                && !downloads.Values.Any(d => d.TokenSequence == token.Sequence)
                && ObjectiveGeometry.MeetsOverlap(ObjectiveGeometry.IntelZoneOverlap(ship.Base, token.Position), requiredOverlap)) return token.Sequence;
        return null;
    }
    /// <summary>Call GetDownloadTokenAtPose to enumerate legal candidates before committing. As in
    /// the source BeginDownload, this low-level commit does not repeat the preview geometry test.</summary>
    public ObjectiveChoiceResult BeginDownload(ObjectiveShip ship, int sequence, int round, ReadOnlySpan<ObjectiveShip> ships)
    {
        if (!CanBeginDownload(ship) || !tokens.TryGetValue(sequence, out var token) || !token.Available
            || downloads.Values.Any(d => d.TokenSequence == sequence)) return new(false);
        departureLocks.Remove(ship.Id);
        downloads.Add(ship.Id, new(ship.Id, sequence, Math.Max(1, round), IsContested(ship.Owner, token.Position, ships)));
        return new(true, 0);
    }
    public ObjectiveChoiceResult RestartDownload(ObjectiveShip ship, int sequence, int round, ReadOnlySpan<ObjectiveShip> ships)
        => !tokens.TryGetValue(sequence, out var token) || !ObjectiveGeometry.MeetsOverlap(ObjectiveGeometry.IntelZoneOverlap(ship.Base, token.Position))
            ? new(false) : BeginDownload(ship, sequence, round, ships);
    public int? StopDownloadForSpeedChange(int shipId)
    {
        if (!downloads.Remove(shipId, out var state)) return null;
        return state.TokenSequence;
    }
    public bool ContinuePastIntelZone(int shipId, int sequence)
    {
        if (!tokens.ContainsKey(sequence)) return false;
        departureLocks[shipId] = sequence;
        return true;
    }
    public bool ConfirmPlotCourseAfterDownload(int shipId, int sequence) => ContinuePastIntelZone(shipId, sequence);
    public void TrackDepartureAtPose(ObjectiveShip ship)
    {
        if (departureLocks.TryGetValue(ship.Id, out var sequence)
            && (!tokens.TryGetValue(sequence, out var token) || !token.Available
                || !ObjectiveGeometry.MeetsOverlap(ObjectiveGeometry.IntelZoneOverlap(ship.Base, token.Position)))) departureLocks.Remove(ship.Id);
    }
    /// <summary>The source always returns false here: delivery is pending, never committed mid-round.</summary>
    public bool ResolveAfterManeuver(ObjectiveShip ship, int round)
    {
        if (ship.Destroyed) return false;
        Round = Math.Max(1, round);
        if (!tokens.Values.Any(t => t.CarrierShipId == ship.Id) || !IsFullyInsideStartingArea(ship)) deliveryPending.Remove(ship.Id);
        else deliveryPending.Add(ship.Id);
        return false;
    }
    public bool IsFullyInsideStartingArea(ObjectiveShip ship) => startingAreas.TryGetValue(ship.Owner, out var zone)
        && ObjectiveGeometry.MeetsOverlap(ObjectiveGeometry.RectangleOverlap(ship.Base, zone), ObjectiveGeometry.FullBaseContainment);
    public void NotifyShipDestroyed(ObjectiveShip ship)
    {
        if (!destructionHandled.Add(ship.Id)) return;
        departureLocks.Remove(ship.Id); deliveryPending.Remove(ship.Id); downloads.Remove(ship.Id);
        foreach (var token in tokens.Values.ToArray())
            if (token.CarrierShipId == ship.Id) { tokens[token.Sequence] = token with { CarrierShipId = null, Position = ship.Base.Center }; break; }
    }
    public void Observe(ReadOnlySpan<ObjectiveShip> ships)
    {
        if (!MatchStarted || Outcome is not null) return;
        foreach (var ship in ships)
        {
            if (ship.Destroyed) NotifyShipDestroyed(ship);
            else TrackDepartureAtPose(ship);
        }
        ValidateDownloads(ships);
        foreach (var entry in downloads.ToArray())
        {
            var ship = ObjectiveGeometry.Find(ships, entry.Key);
            if (!entry.Value.ContestedThisRound && ship is not null
                && IsContested(ship.Value.Owner, tokens[entry.Value.TokenSequence].Position, ships))
                downloads[entry.Key] = entry.Value with { ContestedThisRound = true };
        }
    }
    public ObjectiveOutcome? ResolveEndOfRound(int round, ReadOnlySpan<ObjectiveShip> ships)
    {
        if (Outcome is not null) return Outcome;
        Round = Math.Max(1, round);
        // Delivery comes before completing downloads, so a token obtained this end-of-round
        // cannot also be delivered at that same end-of-round, matching the source ordering.
        foreach (var token in tokens.Values.ToArray())
        {
            if (token.CarrierShipId is not { } id || ObjectiveGeometry.Find(ships, id) is not { Destroyed: false } ship) continue;
            deliveryPending.Remove(id);
            if (!IsFullyInsideStartingArea(ship)) continue;
            tokens[token.Sequence] = token with { CarrierShipId = null, Delivered = true };
            if (ship.Owner == 1) PlayerOneDelivered++; else if (ship.Owner == 2) PlayerTwoDelivered++;
            if ((ship.Owner == 1 ? PlayerOneDelivered : PlayerTwoDelivered) >= IntelRequiredToWin)
            { Outcome = new(ship.Owner, "delivered two Intel tokens"); return Outcome; }
        }
        ValidateDownloads(ships);
        foreach (var download in downloads.Values.ToArray())
        {
            if (download.StartedRound >= Round || download.ContestedThisRound) continue;
            var token = tokens[download.TokenSequence];
            var ship = ObjectiveGeometry.Find(ships, download.ShipId);
            downloads.Remove(download.ShipId);
            if (token.Available && ship is { Destroyed: false }) tokens[token.Sequence] = token with { CarrierShipId = download.ShipId };
        }
        return Outcome;
    }
    private void ValidateDownloads(ReadOnlySpan<ObjectiveShip> ships)
    {
        foreach (var download in downloads.Values.ToArray())
        {
            var ship = ObjectiveGeometry.Find(ships, download.ShipId);
            if (ship is null || ship.Value.Destroyed || ship.Value.Speed != 0
                || !tokens.TryGetValue(download.TokenSequence, out var token) || !token.Available
                || !ObjectiveGeometry.MeetsOverlap(ObjectiveGeometry.IntelZoneOverlap(ship.Value.Base, token.Position))) downloads.Remove(download.ShipId);
        }
    }
    private static bool IsContested(int downloadingOwner, Vector2 center, ReadOnlySpan<ObjectiveShip> ships)
    {
        foreach (var ship in ships)
            if (!ship.Destroyed && !ship.Reserved && ship.Owner != downloadingOwner
                && ObjectiveGeometry.MeetsOverlap(ObjectiveGeometry.IntelZoneOverlap(ship.Base, center))) return true;
        return false;
    }
    public IntelRetrievalSnapshot CaptureSnapshot() => new(tokens.Values.ToArray(), downloads.Values.ToArray(), departureLocks.ToArray(),
        destructionHandled.ToArray(), deliveryPending.ToArray(), startingAreas.ToArray(), PlayerOneDelivered, PlayerTwoDelivered, Round, MatchStarted, Outcome);
    ObjectiveSnapshot IObjectiveRuntime.CaptureSnapshot() => CaptureSnapshot();
    public IObjectiveRuntime DeepClone() => FromSnapshot(CaptureSnapshot());
    public void CopyFrom(IObjectiveRuntime source)
    {
        if (source is not IntelRetrievalObjective other || tokens.Count != other.tokens.Count || startingAreas.Count != other.startingAreas.Count)
            throw new ArgumentException("Intel objective configuration differs.", nameof(source));
        if (ReferenceEquals(this, other)) return;
        foreach (var entry in startingAreas)
            if (!other.startingAreas.TryGetValue(entry.Key, out var area) || entry.Value != area)
                throw new ArgumentException("Intel starting areas differ.", nameof(source));
        for (int i = 0; i < tokenSequences.Length; i++)
            if (tokenSequences[i] != other.tokenSequences[i]) throw new ArgumentException("Intel token identities differ.", nameof(source));
        for (int i = 0; i < tokenSequences.Length; i++) tokens[tokenSequences[i]] = other.tokens[tokenSequences[i]];
        downloads.Clear(); foreach (var entry in other.downloads) downloads.Add(entry.Key, entry.Value);
        departureLocks.Clear(); foreach (var entry in other.departureLocks) departureLocks.Add(entry.Key, entry.Value);
        destructionHandled.Clear(); foreach (int id in other.destructionHandled) destructionHandled.Add(id);
        deliveryPending.Clear(); foreach (int id in other.deliveryPending) deliveryPending.Add(id);
        PlayerOneDelivered = other.PlayerOneDelivered; PlayerTwoDelivered = other.PlayerTwoDelivered;
        Round = other.Round; MatchStarted = other.MatchStarted; Outcome = other.Outcome;
    }
    public void RestoreSnapshot(ObjectiveSnapshot snapshot)
    {
        if (snapshot is not IntelRetrievalSnapshot intel) throw new ArgumentException("Intel objective snapshot type mismatch.", nameof(snapshot));
        var restored = FromSnapshot(intel);
        if (!tokens.Keys.SequenceEqual(restored.tokens.Keys)
            || !startingAreas.OrderBy(p => p.Key).SequenceEqual(restored.startingAreas.OrderBy(p => p.Key)))
            throw new ArgumentException("Intel objective snapshot config mismatch.", nameof(snapshot));
        tokens.Clear(); downloads.Clear(); departureLocks.Clear(); destructionHandled.Clear(); deliveryPending.Clear();
        foreach (var entry in restored.tokens) tokens.Add(entry.Key, entry.Value);
        foreach (var entry in restored.downloads) downloads.Add(entry.Key, entry.Value);
        foreach (var entry in restored.departureLocks) departureLocks.Add(entry.Key, entry.Value);
        foreach (var id in restored.destructionHandled) destructionHandled.Add(id);
        foreach (var id in restored.deliveryPending) deliveryPending.Add(id);
        Round = restored.Round; MatchStarted = restored.MatchStarted; Outcome = restored.Outcome;
        PlayerOneDelivered = restored.PlayerOneDelivered; PlayerTwoDelivered = restored.PlayerTwoDelivered;
    }
    public static IntelRetrievalObjective FromSnapshot(IntelRetrievalSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        if (snapshot.Round < 1 || snapshot.PlayerOneDelivered < 0 || snapshot.PlayerTwoDelivered < 0) throw new ArgumentException("Invalid Intel score/round.", nameof(snapshot));
        var result = new IntelRetrievalObjective(snapshot.Tokens, snapshot.StartingAreas.ToDictionary())
        {
            PlayerOneDelivered = snapshot.PlayerOneDelivered, PlayerTwoDelivered = snapshot.PlayerTwoDelivered,
            Round = snapshot.Round, MatchStarted = snapshot.MatchStarted, Outcome = snapshot.Outcome
        };
        foreach (var download in snapshot.Downloads)
            if (download.ShipId < 0 || download.StartedRound < 1 || !result.tokens.ContainsKey(download.TokenSequence)
                || !result.downloads.TryAdd(download.ShipId, download)) throw new ArgumentException("Invalid Intel download.", nameof(snapshot));
        foreach (var entry in snapshot.DepartureLocks)
            if (!result.tokens.ContainsKey(entry.Value) || !result.departureLocks.TryAdd(entry.Key, entry.Value)) throw new ArgumentException("Invalid departure lock.", nameof(snapshot));
        foreach (var id in snapshot.DestructionHandled) result.destructionHandled.Add(id);
        foreach (var id in snapshot.DeliveryPending) result.deliveryPending.Add(id);
        return result;
    }
}
