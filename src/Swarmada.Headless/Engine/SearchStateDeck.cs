namespace Armada.Core;

public sealed partial class DamageDeckState
{
    // Search may copy public discard contents without reading/allocating the secret draw pile.
    internal CriticalDamageCard[] CapturePublicDiscardPile() => discardPile.ToArray();
}
