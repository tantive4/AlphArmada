using System.Numerics;

namespace Swarmada.Headless.Objective;

public readonly record struct StationCapture(int ShipId, int StartedRound, bool ContestedThisRound);
public sealed record SeizeStationSnapshot(Vector2 ZoneCenter, int Round, bool MatchStarted,
    StationCapture[] Captures, KeyValuePair<int, int>[] CapturePoints, int[] DepartureLocks, ObjectiveOutcome? Outcome) : ObjectiveSnapshot;

/// <summary>Gameplay port of SeizeTheStationObjectiveController. Call Observe after every atomic
/// ship change to preserve the source Update loop's "contested at any time this round" history.
/// A friendly capital ship also contests capture. Stopping/restarting/continuing are explicit choices.</summary>
public sealed class SeizeStationObjective : IObjectiveRuntime
{
    public const int CapturePointsRequiredToWin = 2;
    private readonly Dictionary<int, StationCapture> captures = new();
    private readonly Dictionary<int, int> points = new();
    private readonly HashSet<int> departureLocks = new();
    public Vector2 ZoneCenter { get; }
    public int Round { get; private set; } = 1;
    public bool MatchStarted { get; private set; }
    public ObjectiveOutcome? Outcome { get; private set; }
    public SeizeStationObjective(Vector2 zoneCenter = default) => ZoneCenter = zoneCenter;
    public bool IsCapturing(int shipId) => captures.ContainsKey(shipId);
    public int CapturePoints(int shipId) => points.GetValueOrDefault(shipId);
    public bool IsDepartureLocked(int shipId) => departureLocks.Contains(shipId);
    public void BeginMatch(int round = 1) { Round = Math.Max(1, round); MatchStarted = true; Outcome = null; }
    public void BeginRound(int round)
    {
        Round = Math.Max(1, round);
        foreach (var entry in captures.ToArray()) captures[entry.Key] = entry.Value with { ContestedThisRound = false };
    }
    public float CaptureZoneOverlapAt(ObjectiveShip ship) => ObjectiveGeometry.IntelZoneOverlap(ship.Base, ZoneCenter);
    public bool CanBeginCaptureAtPose(ObjectiveShip ship, float requiredOverlap = ObjectiveGeometry.RequiredBaseOverlap)
        => MatchStarted && !ship.Destroyed && !ship.Reserved && !captures.ContainsKey(ship.Id)
            && !departureLocks.Contains(ship.Id) && ObjectiveGeometry.MeetsOverlap(CaptureZoneOverlapAt(ship), requiredOverlap);
    public ObjectiveChoiceResult BeginCapture(ObjectiveShip ship, int round, ReadOnlySpan<ObjectiveShip> ships)
    {
        if (!CanBeginCaptureAtPose(ship)) return new(false);
        departureLocks.Remove(ship.Id);
        captures[ship.Id] = new(ship.Id, Math.Max(1, round), IsContested(ship.Id, ships));
        points.TryAdd(ship.Id, 0);
        return new(true, 0);
    }
    public ObjectiveChoiceResult RestartCapture(ObjectiveShip ship, int round, ReadOnlySpan<ObjectiveShip> ships)
        => !ObjectiveGeometry.MeetsOverlap(CaptureZoneOverlapAt(ship)) ? new(false) : BeginCapture(ship, round, ships);
    public bool StopCaptureForSpeedChange(int shipId) => captures.Remove(shipId);
    public ObjectiveChoiceResult RestartCaptureAfterSpeedChange(ObjectiveShip ship, int round, ReadOnlySpan<ObjectiveShip> ships)
    {
        departureLocks.Remove(ship.Id);
        var result = BeginCapture(ship with { Speed = 0 }, round, ships);
        // Source sets speed to zero even if the subsequent capture eligibility check fails.
        return result with { RequiredSpeed = 0 };
    }
    public bool ContinuePastCaptureZone(int shipId) { departureLocks.Add(shipId); return true; }
    public bool ConfirmPlotCourseAfterCapture(int shipId) => ContinuePastCaptureZone(shipId);
    public void TrackDepartureAtPose(ObjectiveShip ship)
    {
        if (departureLocks.Contains(ship.Id) && !ObjectiveGeometry.MeetsOverlap(CaptureZoneOverlapAt(ship))) RemoveAllProgress(ship.Id);
    }
    public void NotifyShipDestroyed(int shipId) => RemoveAllProgress(shipId);
    public void Observe(ReadOnlySpan<ObjectiveShip> ships)
    {
        if (!MatchStarted || Outcome is not null) return;
        ValidateCaptures(ships);
        foreach (var entry in captures.ToArray())
            if (!entry.Value.ContestedThisRound && IsContested(entry.Key, ships))
                captures[entry.Key] = entry.Value with { ContestedThisRound = true };
    }
    public ObjectiveOutcome? ResolveEndOfRound(int round, ReadOnlySpan<ObjectiveShip> ships)
    {
        if (Outcome is not null) return Outcome;
        Round = Math.Max(1, round);
        ValidateCaptures(ships);
        foreach (var capture in captures.Values.ToArray())
        {
            var ship = ObjectiveGeometry.Find(ships, capture.ShipId);
            if (ship is null || !captures.ContainsKey(capture.ShipId) || capture.StartedRound >= Round
                || capture.ContestedThisRound || IsContested(capture.ShipId, ships)) continue;
            var score = points.GetValueOrDefault(capture.ShipId) + 1;
            points[capture.ShipId] = score;
            captures[capture.ShipId] = capture with { StartedRound = Round };
            if (score >= CapturePointsRequiredToWin)
            {
                Outcome = new(ship.Value.Owner, "secured two station capture points");
                break;
            }
        }
        return Outcome;
    }
    private void ValidateCaptures(ReadOnlySpan<ObjectiveShip> ships)
    {
        foreach (var capture in captures.Values.ToArray())
        {
            var ship = ObjectiveGeometry.Find(ships, capture.ShipId);
            if (ship is null || ship.Value.Destroyed || !ObjectiveGeometry.MeetsOverlap(CaptureZoneOverlapAt(ship.Value)))
                RemoveAllProgress(capture.ShipId);
            else if (ship.Value.Speed != 0) captures.Remove(capture.ShipId);
        }
        foreach (var id in points.Keys.ToArray())
        {
            var ship = ObjectiveGeometry.Find(ships, id);
            if (ship is null || ship.Value.Destroyed || !ObjectiveGeometry.MeetsOverlap(CaptureZoneOverlapAt(ship.Value))) RemoveAllProgress(id);
        }
    }
    private bool IsContested(int shipId, ReadOnlySpan<ObjectiveShip> ships)
    {
        foreach (var ship in ships)
            if (ship.Id != shipId && !ship.Destroyed && !ship.Reserved
                && ObjectiveGeometry.MeetsOverlap(CaptureZoneOverlapAt(ship))) return true;
        return false;
    }
    private void RemoveAllProgress(int shipId) { captures.Remove(shipId); points.Remove(shipId); departureLocks.Remove(shipId); }
    public int BestScoreForPlayer(int owner, ReadOnlySpan<ObjectiveShip> ships)
    {
        var best = 0;
        foreach (var entry in points)
            if (ObjectiveGeometry.Find(ships, entry.Key) is { } ship && ship.Owner == owner) best = Math.Max(best, entry.Value);
        return best;
    }
    public SeizeStationSnapshot CaptureSnapshot() => new(ZoneCenter, Round, MatchStarted, captures.Values.ToArray(), points.ToArray(), departureLocks.ToArray(), Outcome);
    ObjectiveSnapshot IObjectiveRuntime.CaptureSnapshot() => CaptureSnapshot();
    public IObjectiveRuntime DeepClone() => FromSnapshot(CaptureSnapshot());
    public void CopyFrom(IObjectiveRuntime source)
    {
        if (source is not SeizeStationObjective other || other.ZoneCenter != ZoneCenter)
            throw new ArgumentException("Station objective configuration differs.", nameof(source));
        if (ReferenceEquals(this, other)) return;
        captures.Clear(); foreach (var entry in other.captures) captures.Add(entry.Key, entry.Value);
        points.Clear(); foreach (var entry in other.points) points.Add(entry.Key, entry.Value);
        departureLocks.Clear(); foreach (int id in other.departureLocks) departureLocks.Add(id);
        Round = other.Round; MatchStarted = other.MatchStarted; Outcome = other.Outcome;
    }
    public void RestoreSnapshot(ObjectiveSnapshot snapshot)
    {
        if (snapshot is not SeizeStationSnapshot station || station.ZoneCenter != ZoneCenter)
            throw new ArgumentException("Station objective snapshot config mismatch.", nameof(snapshot));
        var restored = FromSnapshot(station);
        captures.Clear(); points.Clear(); departureLocks.Clear();
        foreach (var entry in restored.captures) captures.Add(entry.Key, entry.Value);
        foreach (var entry in restored.points) points.Add(entry.Key, entry.Value);
        foreach (var id in restored.departureLocks) departureLocks.Add(id);
        Round = restored.Round; MatchStarted = restored.MatchStarted; Outcome = restored.Outcome;
    }
    public static SeizeStationObjective FromSnapshot(SeizeStationSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        if (snapshot.Round < 1) throw new ArgumentException("Invalid objective round.", nameof(snapshot));
        var result = new SeizeStationObjective(snapshot.ZoneCenter) { Round = snapshot.Round, MatchStarted = snapshot.MatchStarted, Outcome = snapshot.Outcome };
        foreach (var capture in snapshot.Captures)
        {
            if (capture.ShipId < 0 || capture.StartedRound < 1 || !result.captures.TryAdd(capture.ShipId, capture))
                throw new ArgumentException("Invalid/duplicate station capture.", nameof(snapshot));
        }
        foreach (var entry in snapshot.CapturePoints)
            if (entry.Key < 0 || entry.Value < 0 || !result.points.TryAdd(entry.Key, entry.Value)) throw new ArgumentException("Invalid station score.", nameof(snapshot));
        foreach (var id in snapshot.DepartureLocks)
            if (id < 0 || !result.departureLocks.Add(id)) throw new ArgumentException("Invalid departure lock.", nameof(snapshot));
        return result;
    }
}
