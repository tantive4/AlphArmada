using System;
using System.Collections.Generic;
using Armada.Core;

namespace Armada.Rules;

public sealed record CommandStackSnapshot(
    int Capacity,
    int TokenCapacity,
    bool AllowsDuplicateTokens,
    ShipCommand[] Stack,
    KeyValuePair<ShipCommand, int>[] Tokens,
    ShipCommand[] RaidTokens);

public sealed partial class CommandStackState
{
    public static CommandStackState CreateWithoutCommands() => new(1) { Capacity = 0, TokenCapacity = 0 };
    public CommandStackSnapshot CaptureSnapshot() => new(Capacity, TokenCapacity, AllowsDuplicateTokens,
        commandStack.ToArray(), SnapshotCollections.Capture(commandTokens), raidTokens.ToArray());

    public void RestoreSnapshot(CommandStackSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        ArgumentNullException.ThrowIfNull(snapshot.Stack);
        ArgumentNullException.ThrowIfNull(snapshot.RaidTokens);
        if (snapshot.Capacity < 0 || snapshot.Capacity > 4 || snapshot.TokenCapacity < snapshot.Capacity ||
            snapshot.Capacity == 0 && (snapshot.TokenCapacity != 0 || snapshot.Stack.Length != 0 || snapshot.Tokens.Length != 0))
            throw new ArgumentException("Snapshot contains invalid command capacities.", nameof(snapshot));
        var tokens = SnapshotCollections.Restore(snapshot.Tokens);
        foreach (var token in tokens)
            if (token.Value < 1)
                throw new ArgumentException("Snapshot token counts must be positive.", nameof(snapshot));
        Capacity = snapshot.Capacity;
        TokenCapacity = snapshot.TokenCapacity;
        AllowsDuplicateTokens = snapshot.AllowsDuplicateTokens;
        commandStack.Clear();
        commandStack.AddRange(snapshot.Stack);
        commandTokens.Clear();
        foreach (var token in tokens)
            commandTokens.Add(token.Key, token.Value);
        raidTokens.Clear();
        raidTokens.AddRange(snapshot.RaidTokens);
    }

    public static CommandStackState FromSnapshot(CommandStackSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        var result = new CommandStackState(snapshot.Capacity);
        result.RestoreSnapshot(snapshot);
        return result;
    }

    public void CopyFrom(CommandStackState source)
    {
        ArgumentNullException.ThrowIfNull(source); if (ReferenceEquals(this, source)) return;
        Capacity = source.Capacity; TokenCapacity = source.TokenCapacity; AllowsDuplicateTokens = source.AllowsDuplicateTokens;
        commandStack.Clear(); commandStack.AddRange(source.commandStack);
        SnapshotCollections.CopyInto(source.commandTokens, commandTokens);
        raidTokens.Clear(); raidTokens.AddRange(source.raidTokens);
    }

    public CommandStackState DeepClone()
    {
        var clone = new CommandStackState(Capacity); clone.CopyFrom(this); return clone;
    }
}
