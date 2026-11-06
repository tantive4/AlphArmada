using System;
using System.Collections.Generic;
using Armada.Core;
using Armada.Data;

namespace Armada.Rules;

/// <summary>Runtime card state; card definitions are shared, read-only catalog data.</summary>
public sealed record UpgradeCardSnapshot(
    string DefinitionId,
    string EffectiveDefinitionId,
    bool Exhausted,
    bool Discarded,
    int CommandTokens,
    ShipCommand? AssignedCommandToken,
    int UsesThisActivation,
    int UsesThisAttack,
    int LastUsedRound,
    HullZone? FocusZone,
    bool SetupInitialized,
    AttackDieColor StoredDieColor,
    DieFace StoredDieFace,
    KeyValuePair<ShipCommand, int>[] TypedCommandTokens,
    string[] StoredUnitIds,
    int[] ScheduledRounds);

public sealed partial class UpgradeCardRuntimeState
{
    public UpgradeCardSnapshot CaptureSnapshot() => new(Definition.id, EffectiveDefinitionId,
        Exhausted, Discarded, CommandTokens, AssignedCommandToken, UsesThisActivation, UsesThisAttack,
        LastUsedRound, FocusZone, SetupInitialized, StoredDieColor, StoredDieFace,
        SnapshotCollections.Capture(typedCommandTokens), storedUnitIds.ToArray(), scheduledRounds.ToArray());

    public void RestoreSnapshot(UpgradeCardSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        if (!string.Equals(Definition.id, snapshot.DefinitionId, StringComparison.Ordinal))
            throw new ArgumentException("An upgrade snapshot must have the same immutable definition identity.", nameof(snapshot));
        ArgumentNullException.ThrowIfNull(snapshot.EffectiveDefinitionId);
        ArgumentNullException.ThrowIfNull(snapshot.StoredUnitIds);
        ArgumentNullException.ThrowIfNull(snapshot.ScheduledRounds);
        if (snapshot.CommandTokens < 0 || snapshot.UsesThisActivation < 0 || snapshot.UsesThisAttack < 0 || snapshot.LastUsedRound < 0)
            throw new ArgumentException("Snapshot contains invalid upgrade counters.", nameof(snapshot));
        var tokens = SnapshotCollections.Restore(snapshot.TypedCommandTokens);
        foreach (var token in tokens)
            if (token.Value < 1)
                throw new ArgumentException("Snapshot typed command token counts must be positive.", nameof(snapshot));
        foreach (var unit in snapshot.StoredUnitIds)
            if (string.IsNullOrWhiteSpace(unit))
                throw new ArgumentException("Snapshot stored unit identities cannot be empty.", nameof(snapshot));
        foreach (int round in snapshot.ScheduledRounds)
            if (round < 1)
                throw new ArgumentException("Snapshot scheduled rounds must be positive.", nameof(snapshot));

        EffectiveDefinitionId = snapshot.EffectiveDefinitionId;
        Exhausted = snapshot.Exhausted;
        Discarded = snapshot.Discarded;
        CommandTokens = snapshot.CommandTokens;
        AssignedCommandToken = snapshot.AssignedCommandToken;
        UsesThisActivation = snapshot.UsesThisActivation;
        UsesThisAttack = snapshot.UsesThisAttack;
        LastUsedRound = snapshot.LastUsedRound;
        FocusZone = snapshot.FocusZone;
        SetupInitialized = snapshot.SetupInitialized;
        StoredDieColor = snapshot.StoredDieColor;
        StoredDieFace = snapshot.StoredDieFace;
        typedCommandTokens.Clear();
        foreach (var token in tokens)
            typedCommandTokens.Add(token.Key, token.Value);
        storedUnitIds.Clear();
        storedUnitIds.AddRange(snapshot.StoredUnitIds);
        scheduledRounds.Clear();
        scheduledRounds.AddRange(snapshot.ScheduledRounds);
    }

    public static UpgradeCardRuntimeState FromSnapshot(UpgradeCardSnapshot snapshot, UpgradeCardDefinition definition)
    {
        var result = new UpgradeCardRuntimeState(definition);
        result.RestoreSnapshot(snapshot);
        return result;
    }

    private UpgradeCardRuntimeState(UpgradeCardRuntimeState source)
    { Definition = source.Definition; EffectiveDefinitionId = source.EffectiveDefinitionId; CopyFrom(source); }

    public void CopyFrom(UpgradeCardRuntimeState source)
    {
        ArgumentNullException.ThrowIfNull(source); if (ReferenceEquals(this, source)) return;
        if (!string.Equals(Definition.id, source.Definition.id, StringComparison.Ordinal)) throw new ArgumentException("Upgrade identities differ.", nameof(source));
        EffectiveDefinitionId = source.EffectiveDefinitionId; Exhausted = source.Exhausted; Discarded = source.Discarded;
        CommandTokens = source.CommandTokens; AssignedCommandToken = source.AssignedCommandToken;
        UsesThisActivation = source.UsesThisActivation; UsesThisAttack = source.UsesThisAttack; LastUsedRound = source.LastUsedRound;
        FocusZone = source.FocusZone; SetupInitialized = source.SetupInitialized;
        StoredDieColor = source.StoredDieColor; StoredDieFace = source.StoredDieFace;
        SnapshotCollections.CopyInto(source.typedCommandTokens, typedCommandTokens);
        storedUnitIds.Clear(); storedUnitIds.AddRange(source.storedUnitIds);
        scheduledRounds.Clear(); scheduledRounds.AddRange(source.scheduledRounds);
    }

    public UpgradeCardRuntimeState DeepClone() => new(this);
}
