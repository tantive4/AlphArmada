using System;
using System.Collections.Generic;

namespace Armada.Core;

public sealed partial class ShipRuntimeState
{
    public ShipRuntimeSnapshot CaptureSnapshot()
    {
        EnsureShieldMaps();
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
        var damage = new DamageCardSnapshot[damageCards.Count];
        for (int i = 0; i < damage.Length; i++)
            damage[i] = new(damageCards[i].FaceUp, damageCards[i].CriticalCard);
        return new(ShipId, MaximumHull, CurrentHull, CurrentSpeed,
            SnapshotCollections.Capture(shields), SnapshotCollections.Capture(maximumShields),
            SnapshotCollections.Capture(defenseTokenCounts), SnapshotCollections.Capture(spentDefenseTokens),
            SnapshotCollections.Capture(discardedDefenseTokens), tokens, damage);
    }

    /// <summary>Restores every runtime field without applying gameplay effects or drawing new cards.</summary>
    public void RestoreSnapshot(ShipRuntimeSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        if (!string.Equals(ShipId, snapshot.ShipId, StringComparison.Ordinal))
            throw new ArgumentException("A ship snapshot must have the same immutable ShipId.", nameof(snapshot));
        ArgumentNullException.ThrowIfNull(snapshot.DefenseTokenSlots);
        ArgumentNullException.ThrowIfNull(snapshot.DamageCards);
        if (snapshot.MaximumHull < 1 || snapshot.CurrentHull < 0 || snapshot.CurrentSpeed < 0)
            throw new ArgumentException("Snapshot contains invalid ship hull or speed.", nameof(snapshot));

        // Validate and copy before mutating this instance, so malformed input cannot half-restore it.
        var restoredShields = SnapshotCollections.Restore(snapshot.Shields);
        var restoredMaximumShields = SnapshotCollections.Restore(snapshot.MaximumShields);
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
        var restoredDamage = new List<DamageCardState>(snapshot.DamageCards.Length);
        foreach (var card in snapshot.DamageCards)
            restoredDamage.Add(new DamageCardState(card.FaceUp, card.CriticalCard));
        SnapshotCollections.ValidateDefenseTokens(restoredCounts, restoredSpent, restoredDiscarded, snapshot.DefenseTokenSlots);
        foreach (var shield in restoredShields)
            if (shield.Value < 0)
                throw new ArgumentException("Snapshot shields cannot be negative.", nameof(snapshot));
        foreach (var shield in restoredMaximumShields)
            if (shield.Value < 0)
                throw new ArgumentException("Snapshot maximum shields cannot be negative.", nameof(snapshot));

        MaximumHull = snapshot.MaximumHull;
        CurrentHull = snapshot.CurrentHull;
        CurrentSpeed = snapshot.CurrentSpeed;
        shields = restoredShields;
        maximumShields = restoredMaximumShields;
        defenseTokenCounts = restoredCounts;
        spentDefenseTokens = restoredSpent;
        discardedDefenseTokens = restoredDiscarded;
        defenseTokenSlots = restoredSlots;
        damageCards = restoredDamage;
    }

    public static ShipRuntimeState FromSnapshot(ShipRuntimeSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        var result = new ShipRuntimeState(snapshot.ShipId, 1, 0, 0, 0, 0, 0,
            braceTokens: 0, redirectTokens: 0, evadeTokens: 0);
        result.RestoreSnapshot(snapshot);
        return result;
    }

    /// <summary>Copies trusted branch state into reusable storage without allocating a snapshot DTO.</summary>
    public void CopyFrom(ShipRuntimeState source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (ReferenceEquals(this, source)) return;
        if (!string.Equals(ShipId, source.ShipId, StringComparison.Ordinal)) throw new ArgumentException("Ship identities differ.", nameof(source));
        source.EnsureShieldMaps(); source.EnsureTokenMaps(); EnsureShieldMaps(); EnsureTokenMaps();
        MaximumHull = source.MaximumHull; CurrentHull = source.CurrentHull; CurrentSpeed = source.CurrentSpeed;
        SnapshotCollections.CopyInto(source.shields, shields); SnapshotCollections.CopyInto(source.maximumShields, maximumShields);
        SnapshotCollections.CopyInto(source.defenseTokenCounts, defenseTokenCounts);
        SnapshotCollections.CopyInto(source.spentDefenseTokens, spentDefenseTokens);
        SnapshotCollections.CopyInto(source.discardedDefenseTokens, discardedDefenseTokens);
        SnapshotCollections.CopyListsInto(source.defenseTokenSlots, defenseTokenSlots);
        damageCards.Clear(); damageCards.AddRange(source.damageCards);
    }

    public ShipRuntimeState DeepClone()
    {
        EnsureShieldMaps(); EnsureTokenMaps();
        var clone = (ShipRuntimeState)MemberwiseClone();
        clone.shields = new(shields); clone.maximumShields = new(maximumShields);
        clone.defenseTokenCounts = new(defenseTokenCounts); clone.spentDefenseTokens = new(spentDefenseTokens);
        clone.discardedDefenseTokens = new(discardedDefenseTokens);
        clone.defenseTokenSlots = new(defenseTokenSlots.Count);
        foreach (var entry in defenseTokenSlots) clone.defenseTokenSlots.Add(entry.Key, new(entry.Value));
        clone.damageCards = new(damageCards); return clone;
    }
}
