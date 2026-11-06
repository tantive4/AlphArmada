namespace Armada.Core;

public sealed partial class ShipRuntimeState
{
    public IEnumerable<DefenseTokenInstance> GetDefenseTokenInstancesAll()
        => Enum.GetValues<DefenseTokenType>().SelectMany(GetDefenseTokenInstances);
    public bool TryReadyDefenseToken(DefenseTokenInstance token) => TrySetDefenseTokenExhausted(token, false);
    public bool TrySetDefenseTokenExhausted(DefenseTokenInstance token, bool exhausted)
    {
        var slots = GetDefenseTokenSlots(token.Token);
        if (token.Slot < 0 || token.Slot >= slots.Count || slots[token.Slot].Discarded || slots[token.Slot].Exhausted != token.Exhausted) return false;
        var slot = slots[token.Slot]; slot.Exhausted = exhausted; slots[token.Slot] = slot; SyncDefenseTokenCounts(token.Token); return true;
    }
}
public sealed partial class SquadronRuntimeState
{
    public bool TryReadyDefenseToken(DefenseTokenInstance token) => TrySetDefenseTokenExhausted(token, false);
    public bool TrySetDefenseTokenExhausted(DefenseTokenInstance token, bool exhausted)
    {
        var slots = GetDefenseTokenSlots(token.Token);
        if (token.Slot < 0 || token.Slot >= slots.Count || slots[token.Slot].Discarded || slots[token.Slot].Exhausted != token.Exhausted) return false;
        var slot = slots[token.Slot]; slot.Exhausted = exhausted; slots[token.Slot] = slot; SyncDefenseTokenCounts(token.Token); return true;
    }
}
