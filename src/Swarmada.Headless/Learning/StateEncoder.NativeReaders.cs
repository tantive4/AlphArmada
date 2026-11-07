using Armada.Core;
using Swarmada.Headless.Engine;

namespace Swarmada.Headless.Learning
{
    internal static class EncoderDefenseSlotWriter
    {
        internal static void Write(Span<float> row, Span<int> references, int entity, DefenseTokenType type,
            int slot, bool exhausted, bool discarded, AttackState? attack)
        {
            row.Clear(); row[(int)type] = 1; row[6] = 1; row[7] = !exhausted && !discarded ? 1 : 0;
            row[8] = exhausted ? 1 : 0; row[9] = discarded ? 1 : 0;
            if (attack?.Defender == entity)
            {
                foreach (var token in attack.LockedTokens) if (token.Token == type && token.Slot == slot) { row[10] = 1; break; }
                row[11] = attack.SpentTokenTypes.Contains(type) ? 1 : 0;
            }
            references[0] = entity; references[1] = (int)type; references[2] = slot;
        }
    }
}

namespace Armada.Core
{
    // These readers share the native runtime's physical slot ordering without allocating
    // recovery DTOs. They never create/remove/ready/spend a token or mutate game storage.
    public sealed partial class ShipRuntimeState
    {
        internal int EncoderDefenseSlotCount => defenseTokenSlots.Sum(entry => entry.Value.Count);
        internal int WriteEncoderDefenseSlots(Span<float> data, Span<int> references, int entity, AttackState? attack)
        {
            int count = 0;
            foreach (var entry in defenseTokenSlots) for (int slot = 0; slot < entry.Value.Count; slot++)
            {
                var value = entry.Value[slot];
                Swarmada.Headless.Learning.EncoderDefenseSlotWriter.Write(data.Slice(count * 16, 16), references.Slice(count * 3, 3), entity, entry.Key, slot, value.Exhausted, value.Discarded, attack); count++;
            }
            return count;
        }
    }
    public sealed partial class SquadronRuntimeState
    {
        internal int EncoderDefenseSlotCount => defenseTokenSlots.Sum(entry => entry.Value.Count);
        internal bool EncoderTemporaryAttackOnly => temporaryAttackOnly;
        internal bool EncoderMovementPrevented => movementPreventedForShipActivation;
        internal int WriteEncoderDefenseSlots(Span<float> data, Span<int> references, int entity, AttackState? attack)
        {
            int count = 0;
            foreach (var entry in defenseTokenSlots) for (int slot = 0; slot < entry.Value.Count; slot++)
            {
                var value = entry.Value[slot];
                Swarmada.Headless.Learning.EncoderDefenseSlotWriter.Write(data.Slice(count * 16, 16), references.Slice(count * 3, 3), entity, entry.Key, slot, value.Exhausted, value.Discarded, attack); count++;
            }
            return count;
        }
    }
}
