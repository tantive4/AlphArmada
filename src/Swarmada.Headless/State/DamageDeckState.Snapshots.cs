using System;

namespace Armada.Core;

public sealed partial class DamageDeckState
{
    // Skip construction and shuffling of a replacement deck when cloning a search branch.
    private DamageDeckState() { }

    public DamageDeckSnapshot CaptureSnapshot() => new(drawPile.ToArray(), discardPile.ToArray());

    public void RestoreSnapshot(DamageDeckSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        ArgumentNullException.ThrowIfNull(snapshot.DrawPile);
        ArgumentNullException.ThrowIfNull(snapshot.DiscardPile);
        drawPile.Clear();
        drawPile.AddRange(snapshot.DrawPile);
        discardPile.Clear();
        discardPile.AddRange(snapshot.DiscardPile);
    }

    public static DamageDeckState FromSnapshot(DamageDeckSnapshot snapshot)
    {
        var result = new DamageDeckState();
        result.RestoreSnapshot(snapshot);
        return result;
    }

    public void CopyFrom(DamageDeckState source)
    {
        ArgumentNullException.ThrowIfNull(source); if (ReferenceEquals(this, source)) return;
        drawPile.Clear(); drawPile.AddRange(source.drawPile);
        discardPile.Clear(); discardPile.AddRange(source.discardPile);
    }

    public DamageDeckState DeepClone()
    {
        var clone = new DamageDeckState(); clone.CopyFrom(this); return clone;
    }
}
