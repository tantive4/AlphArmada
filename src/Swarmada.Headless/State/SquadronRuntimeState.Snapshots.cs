using System;
using System.Collections.Generic;

namespace Armada.Core;

public sealed partial class SquadronRuntimeState
{
    public SquadronRuntimeSnapshot CaptureSnapshot()
    {
        EnsureTokenMaps();
        var tokens = new DefenseTokenTypeSnapshot[defenseTokenSlots.Count];
        int index = 0;
        foreach (var entry in defenseTokenSlots)
        {
            var slots = new DefenseTokenSlotSnapshot[entry.Value.Count];
            for (int i = 0; i < slots.Length; i++)
                slots[i] = new(entry.Value[i].Exhausted, entry.Value[i].Discarded);
            tokens[index++] = new(entry.Key, slots);
        }
        return new(CurrentHull, Activated, Commanded, CommandingShipRuntimeId,
            TemporaryGrit, TemporarySnipeValue, TemporaryAssault, TemporarySwarm,
            TemporarySpeedOverride, TemporaryCannotAttack, temporaryAttackOnly,
            movementPreventedForShipActivation, TemporaryYavarisBonusDie, TemporaryCloak,
            TemporaryRogue, TemporaryIgnoresEngagement, TemporaryCommanded, DestructionDeferred,
            SnapshotCollections.Capture(defenseTokenCounts), SnapshotCollections.Capture(spentDefenseTokens),
            SnapshotCollections.Capture(discardedDefenseTokens), tokens);
    }

    public void RestoreSnapshot(SquadronRuntimeSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        ArgumentNullException.ThrowIfNull(snapshot.DefenseTokenSlots);
        ArgumentNullException.ThrowIfNull(snapshot.CommandingShipRuntimeId);
        if (snapshot.CurrentHull < 0 || snapshot.TemporarySnipeValue < 0 || snapshot.TemporarySpeedOverride < -1)
            throw new ArgumentException("Snapshot contains invalid squadron hull or modifiers.", nameof(snapshot));
        var restoredCounts = SnapshotCollections.Restore(snapshot.DefenseTokenCounts);
        var restoredSpent = SnapshotCollections.Restore(snapshot.SpentDefenseTokens);
        var restoredDiscarded = SnapshotCollections.Restore(snapshot.DiscardedDefenseTokens);
        var restoredSlots = new Dictionary<DefenseTokenType, List<DefenseTokenSlotState>>(snapshot.DefenseTokenSlots.Length);
        foreach (var token in snapshot.DefenseTokenSlots)
        {
            ArgumentNullException.ThrowIfNull(token);
            ArgumentNullException.ThrowIfNull(token.Slots);
            var slots = new List<DefenseTokenSlotState>(token.Slots.Length);
            foreach (var slot in token.Slots)
                slots.Add(new() { Exhausted = slot.Exhausted, Discarded = slot.Discarded });
            if (!restoredSlots.TryAdd(token.Token, slots))
                throw new ArgumentException("Snapshot contains duplicate defense token types.", nameof(snapshot));
        }
        SnapshotCollections.ValidateDefenseTokens(restoredCounts, restoredSpent, restoredDiscarded, snapshot.DefenseTokenSlots);

        CurrentHull = snapshot.CurrentHull;
        Activated = snapshot.Activated;
        Commanded = snapshot.Commanded;
        CommandingShipRuntimeId = snapshot.CommandingShipRuntimeId;
        TemporaryGrit = snapshot.TemporaryGrit;
        TemporarySnipeValue = snapshot.TemporarySnipeValue;
        TemporaryAssault = snapshot.TemporaryAssault;
        TemporarySwarm = snapshot.TemporarySwarm;
        TemporarySpeedOverride = snapshot.TemporarySpeedOverride;
        TemporaryCannotAttack = snapshot.TemporaryCannotAttack;
        temporaryAttackOnly = snapshot.TemporaryAttackOnly;
        movementPreventedForShipActivation = snapshot.MovementPreventedForShipActivation;
        TemporaryYavarisBonusDie = snapshot.TemporaryYavarisBonusDie;
        TemporaryCloak = snapshot.TemporaryCloak;
        TemporaryRogue = snapshot.TemporaryRogue;
        TemporaryIgnoresEngagement = snapshot.TemporaryIgnoresEngagement;
        TemporaryCommanded = snapshot.TemporaryCommanded;
        DestructionDeferred = snapshot.DestructionDeferred;
        defenseTokenCounts = restoredCounts;
        spentDefenseTokens = restoredSpent;
        discardedDefenseTokens = restoredDiscarded;
        defenseTokenSlots = restoredSlots;
    }

    public static SquadronRuntimeState FromSnapshot(SquadronRuntimeSnapshot snapshot)
    {
        var result = new SquadronRuntimeState();
        result.RestoreSnapshot(snapshot);
        return result;
    }

    public void CopyFrom(SquadronRuntimeState source)
    {
        ArgumentNullException.ThrowIfNull(source); if (ReferenceEquals(this, source)) return;
        source.EnsureTokenMaps(); EnsureTokenMaps();
        CurrentHull = source.CurrentHull; Activated = source.Activated; Commanded = source.Commanded;
        CommandingShipRuntimeId = source.CommandingShipRuntimeId; TemporaryGrit = source.TemporaryGrit;
        TemporarySnipeValue = source.TemporarySnipeValue; TemporaryAssault = source.TemporaryAssault;
        TemporarySwarm = source.TemporarySwarm; TemporarySpeedOverride = source.TemporarySpeedOverride;
        TemporaryCannotAttack = source.TemporaryCannotAttack; temporaryAttackOnly = source.temporaryAttackOnly;
        movementPreventedForShipActivation = source.movementPreventedForShipActivation;
        TemporaryYavarisBonusDie = source.TemporaryYavarisBonusDie; TemporaryCloak = source.TemporaryCloak;
        TemporaryRogue = source.TemporaryRogue; TemporaryIgnoresEngagement = source.TemporaryIgnoresEngagement;
        TemporaryCommanded = source.TemporaryCommanded; DestructionDeferred = source.DestructionDeferred;
        SnapshotCollections.CopyInto(source.defenseTokenCounts, defenseTokenCounts);
        SnapshotCollections.CopyInto(source.spentDefenseTokens, spentDefenseTokens);
        SnapshotCollections.CopyInto(source.discardedDefenseTokens, discardedDefenseTokens);
        SnapshotCollections.CopyListsInto(source.defenseTokenSlots, defenseTokenSlots);
    }

    public SquadronRuntimeState DeepClone()
    {
        EnsureTokenMaps(); var clone = (SquadronRuntimeState)MemberwiseClone();
        clone.defenseTokenCounts = new(defenseTokenCounts); clone.spentDefenseTokens = new(spentDefenseTokens);
        clone.discardedDefenseTokens = new(discardedDefenseTokens);
        clone.defenseTokenSlots = new(defenseTokenSlots.Count);
        foreach (var entry in defenseTokenSlots) clone.defenseTokenSlots.Add(entry.Key, new(entry.Value));
        return clone;
    }
}
