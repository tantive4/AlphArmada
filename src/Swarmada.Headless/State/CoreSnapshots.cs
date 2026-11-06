using System;
using System.Collections.Generic;

namespace Armada.Core;

/// <summary>One physical token slot, including discarded slots and their last exhausted state.</summary>
public readonly record struct DefenseTokenSlotSnapshot(bool Exhausted, bool Discarded);

/// <summary>Ordered slots preserve instance identifiers used by pending defense choices.</summary>
public sealed record DefenseTokenTypeSnapshot(DefenseTokenType Token, DefenseTokenSlotSnapshot[] Slots);

public readonly record struct DamageCardSnapshot(bool FaceUp, CriticalDamageCard? CriticalCard);

/// <summary>
/// Explicit runtime state. Arrays belong to the snapshot, never to its captured object.
/// Restore copies arrays again so independently restored branches cannot affect one another.
/// </summary>
public sealed record ShipRuntimeSnapshot(
    string ShipId,
    int MaximumHull,
    int CurrentHull,
    int CurrentSpeed,
    KeyValuePair<HullZone, int>[] Shields,
    KeyValuePair<HullZone, int>[] MaximumShields,
    KeyValuePair<DefenseTokenType, int>[] DefenseTokenCounts,
    KeyValuePair<DefenseTokenType, int>[] SpentDefenseTokens,
    KeyValuePair<DefenseTokenType, int>[] DiscardedDefenseTokens,
    DefenseTokenTypeSnapshot[] DefenseTokenSlots,
    DamageCardSnapshot[] DamageCards);

/// <summary>
/// Both movement restriction flags are exported separately: TemporaryAttackOnly combines them
/// in the recovered API, but clearing either restriction can produce different future states.
/// </summary>
public sealed record SquadronRuntimeSnapshot(
    int CurrentHull,
    bool Activated,
    bool Commanded,
    string CommandingShipRuntimeId,
    bool TemporaryGrit,
    int TemporarySnipeValue,
    bool TemporaryAssault,
    bool TemporarySwarm,
    int TemporarySpeedOverride,
    bool TemporaryCannotAttack,
    bool TemporaryAttackOnly,
    bool MovementPreventedForShipActivation,
    bool TemporaryYavarisBonusDie,
    bool TemporaryCloak,
    bool TemporaryRogue,
    bool TemporaryIgnoresEngagement,
    bool TemporaryCommanded,
    bool DestructionDeferred,
    KeyValuePair<DefenseTokenType, int>[] DefenseTokenCounts,
    KeyValuePair<DefenseTokenType, int>[] SpentDefenseTokens,
    KeyValuePair<DefenseTokenType, int>[] DiscardedDefenseTokens,
    DefenseTokenTypeSnapshot[] DefenseTokenSlots);

/// <summary>DrawPile[0] is the next card drawn. Neither pile is reconstructed from a seed.</summary>
public sealed record DamageDeckSnapshot(CriticalDamageCard[] DrawPile, CriticalDamageCard[] DiscardPile);

internal static class SnapshotCollections
{
    internal static void CopyInto<TKey, TValue>(Dictionary<TKey, TValue> source, Dictionary<TKey, TValue> target)
        where TKey : notnull
    {
        target.Clear(); target.EnsureCapacity(source.Count);
        foreach (var entry in source) target.Add(entry.Key, entry.Value);
    }

    internal static void CopyListsInto<TKey, TValue>(Dictionary<TKey, List<TValue>> source, Dictionary<TKey, List<TValue>> target)
        where TKey : notnull
    {
        bool sameOrder = source.Count == target.Count;
        if (sameOrder)
        {
            using var current = target.GetEnumerator();
            foreach (var entry in source)
                if (!current.MoveNext() || !EqualityComparer<TKey>.Default.Equals(current.Current.Key, entry.Key)) { sameOrder = false; break; }
        }
        if (!sameOrder)
        {
            // Token types rarely change. Preserve their exact iteration order while reusing
            // existing slot buffers, and allocate this temporary only on a structural change.
            var buffers = new List<TValue>[source.Count]; int index = 0;
            foreach (var entry in source) buffers[index++] = target.TryGetValue(entry.Key, out var existing) ? existing : new(entry.Value.Count);
            target.Clear(); index = 0;
            foreach (var entry in source) target.Add(entry.Key, buffers[index++]);
        }
        foreach (var entry in source)
        {
            var list = target[entry.Key];
            list.Clear(); list.AddRange(entry.Value);
        }
    }

    internal static KeyValuePair<TKey, TValue>[] Capture<TKey, TValue>(Dictionary<TKey, TValue> source)
        where TKey : notnull
    {
        var result = new KeyValuePair<TKey, TValue>[source.Count];
        int index = 0;
        foreach (var entry in source)
            result[index++] = entry;
        return result;
    }

    internal static Dictionary<TKey, TValue> Restore<TKey, TValue>(KeyValuePair<TKey, TValue>[] source)
        where TKey : notnull
    {
        ArgumentNullException.ThrowIfNull(source);
        var result = new Dictionary<TKey, TValue>(source.Length);
        foreach (var entry in source)
            if (!result.TryAdd(entry.Key, entry.Value))
                throw new ArgumentException("Snapshot contains duplicate map keys.", nameof(source));
        return result;
    }

    internal static void ValidateDefenseTokens(
        Dictionary<DefenseTokenType, int> counts,
        Dictionary<DefenseTokenType, int> spent,
        Dictionary<DefenseTokenType, int> discarded,
        DefenseTokenTypeSnapshot[] tokens)
    {
        ValidateNonnegative(counts);
        ValidateNonnegative(spent);
        ValidateNonnegative(discarded);
        foreach (var token in tokens)
        {
            int actualSpent = 0, actualDiscarded = 0;
            foreach (var slot in token.Slots)
                if (slot.Discarded) actualDiscarded++;
                else if (slot.Exhausted) actualSpent++;
            if (Get(counts, token.Token) != token.Slots.Length ||
                Get(spent, token.Token) != actualSpent || Get(discarded, token.Token) != actualDiscarded)
                throw new ArgumentException("Snapshot defense token counts disagree with individual token slots.");
        }
        ValidateRepresented(counts, tokens);
        ValidateRepresented(spent, tokens);
        ValidateRepresented(discarded, tokens);
    }

    private static int Get(Dictionary<DefenseTokenType, int> values, DefenseTokenType token) =>
        values.TryGetValue(token, out int count) ? count : 0;

    private static void ValidateNonnegative(Dictionary<DefenseTokenType, int> values)
    {
        foreach (var entry in values)
            if (entry.Value < 0)
                throw new ArgumentException("Snapshot defense token counts cannot be negative.");
    }

    private static void ValidateRepresented(Dictionary<DefenseTokenType, int> values, DefenseTokenTypeSnapshot[] tokens)
    {
        foreach (var entry in values)
        {
            if (entry.Value == 0) continue;
            bool found = false;
            foreach (var token in tokens)
                if (token.Token == entry.Key) { found = true; break; }
            if (!found)
                throw new ArgumentException("Snapshot defense token count has no corresponding token slots.");
        }
    }
}
