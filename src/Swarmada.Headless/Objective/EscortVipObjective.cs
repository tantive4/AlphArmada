using Armada.Rules;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Objective;

public sealed record EscortVipSnapshot(int VipShipId, int EscortOwner, int InterceptorOwner,
    OrientedRectangle ExtractionZone, int Round, bool MatchStarted, ObjectiveOutcome? Outcome) : ObjectiveSnapshot;

/// <summary>Gameplay port of EscortTheVipObjectiveController. Extraction is evaluated only at
/// end of round. Activate-last/one-attack/Navigate-only restrictions are exposed for a future
/// battlefield scheduler to enforce; CanActivate by itself is the source's liveness predicate.</summary>
public sealed class EscortVipObjective : IObjectiveRuntime
{
    public const int FinalRound = 10;
    public const int VipMaximumAttacks = 1;
    public int VipShipId { get; }
    public int EscortOwner { get; }
    public int InterceptorOwner { get; }
    public OrientedRectangle ExtractionZone { get; }
    public int Round { get; private set; } = 1;
    public bool MatchStarted { get; private set; }
    public ObjectiveOutcome? Outcome { get; private set; }
    public EscortVipObjective(int vipShipId, int escortOwner, int interceptorOwner, OrientedRectangle extractionZone)
    {
        if (vipShipId < 0 || escortOwner is not (1 or 2) || interceptorOwner is not (1 or 2) || escortOwner == interceptorOwner)
            throw new ArgumentException("Escort and interceptor must be opposing players with a valid VIP ID.");
        VipShipId = vipShipId; EscortOwner = escortOwner; InterceptorOwner = interceptorOwner; ExtractionZone = extractionZone;
    }
    public void BeginMatch(int round = 1) { Round = Math.Max(1, round); MatchStarted = true; Outcome = null; }
    public void BeginRound(int round) => Round = Math.Max(1, round);
    public static bool IsVipCommandAllowed(ShipCommand command) => command == ShipCommand.Navigate;
    public bool CanActivate(ObjectiveShip? vip) => MatchStarted && Outcome is null && vip is { Destroyed: false } && vip.Value.Id == VipShipId;
    public bool IsFullyInsideExtractionZone(ObjectiveShip ship)
        => PlayAreaGeometry.Contains(ExtractionZone, ship.Base, 0.011425f);
    public void Observe(ReadOnlySpan<ObjectiveShip> ships)
    {
        if (!MatchStarted || Outcome is not null) return;
        if (ObjectiveGeometry.Find(ships, VipShipId) is not { Destroyed: false })
            Outcome = new(InterceptorOwner, "destroyed the VIP Transport");
    }
    public void NotifyShipDestroyed(ObjectiveShip ship, ReadOnlySpan<ObjectiveShip> survivingState)
    {
        if (Outcome is not null) return;
        if (ship.Id == VipShipId) Outcome = new(InterceptorOwner, "destroyed the VIP Transport");
        else if (ship.Owner == InterceptorOwner && !HasInterceptorCapitalShip(survivingState))
            Outcome = new(EscortOwner, "eliminated every Interceptor capital ship");
    }
    public ObjectiveOutcome? ResolveEndOfRound(int round, ReadOnlySpan<ObjectiveShip> ships)
    {
        if (Outcome is not null) return Outcome;
        Round = Math.Max(1, round);
        var vip = ObjectiveGeometry.Find(ships, VipShipId);
        if (vip is not { Destroyed: false }) Outcome = new(InterceptorOwner, "destroyed the VIP Transport");
        else if (!HasInterceptorCapitalShip(ships)) Outcome = new(EscortOwner, "eliminated every Interceptor capital ship");
        else if (IsFullyInsideExtractionZone(vip.Value)) Outcome = new(EscortOwner, "extracted the VIP Transport");
        else if (Round >= FinalRound) Outcome = new(InterceptorOwner, "prevented extraction through the end of Round 10");
        return Outcome;
    }
    private bool HasInterceptorCapitalShip(ReadOnlySpan<ObjectiveShip> ships)
    {
        // The source includes reserve ships here; retain that behavior rather than silently
        // reusing the Station/Intel reserve-exclusion rule.
        foreach (var ship in ships) if (ship.Id != VipShipId && ship.Owner == InterceptorOwner && !ship.Destroyed) return true;
        return false;
    }
    public EscortVipSnapshot CaptureSnapshot() => new(VipShipId, EscortOwner, InterceptorOwner, ExtractionZone, Round, MatchStarted, Outcome);
    ObjectiveSnapshot IObjectiveRuntime.CaptureSnapshot() => CaptureSnapshot();
    public IObjectiveRuntime DeepClone() => FromSnapshot(CaptureSnapshot());
    public void CopyFrom(IObjectiveRuntime source)
    {
        if (source is not EscortVipObjective other || other.VipShipId != VipShipId || other.EscortOwner != EscortOwner
            || other.InterceptorOwner != InterceptorOwner || other.ExtractionZone != ExtractionZone)
            throw new ArgumentException("VIP objective configuration differs.", nameof(source));
        Round = other.Round; MatchStarted = other.MatchStarted; Outcome = other.Outcome;
    }
    public void RestoreSnapshot(ObjectiveSnapshot snapshot)
    {
        if (snapshot is not EscortVipSnapshot escort || escort.VipShipId != VipShipId || escort.EscortOwner != EscortOwner
            || escort.InterceptorOwner != InterceptorOwner || escort.ExtractionZone != ExtractionZone)
            throw new ArgumentException("VIP objective snapshot config mismatch.", nameof(snapshot));
        var restored = FromSnapshot(escort);
        Round = restored.Round; MatchStarted = restored.MatchStarted; Outcome = restored.Outcome;
    }
    public static EscortVipObjective FromSnapshot(EscortVipSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        if (snapshot.Round < 1) throw new ArgumentException("Invalid objective round.", nameof(snapshot));
        return new(snapshot.VipShipId, snapshot.EscortOwner, snapshot.InterceptorOwner, snapshot.ExtractionZone)
            { Round = snapshot.Round, MatchStarted = snapshot.MatchStarted, Outcome = snapshot.Outcome };
    }
}
