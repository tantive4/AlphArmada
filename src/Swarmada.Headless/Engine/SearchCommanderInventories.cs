using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Search;

internal static class CommanderInventoryRules
{
    internal static bool IsPrivate(string id) => id is "grand-admiral-thrawn-commander" or "admiral-trench-commander";
}

public sealed partial class NativeMctsEnvironment
{
    // Nondecreasing command indices give each unordered inventory one canonical
    // path: 20 possible three-dial stores, 35 possible four-dial stores. Each edge
    // has at most four choices and bypasses neural inference and policy targets.
    private sealed record PrivateCommanderInventory(int Upgrade, int Owner, int Total, int[] Selected);
    private PrivateCommanderInventory? pendingCommanderInventory;
    private readonly int[] privateCommanderCards;

    private void BeginCommanderInventory()
    {
        if (privateCommanderCards.Length == 0) return;
        var decision = GetSettled();
        if (decision.Kind != DecisionKind.Player || decision.Player == RootObserver) return;
        foreach (int index in privateCommanderCards)
        {
            var card = Position.Context.State.Upgrades[index];
            if (materializedCommands.HasMaterializedCommanderInventory(index)) continue;
            int host = checked((int)Position.Context.State.Registers[Position.Context.Topology.Registers.UpgradeOwner(index)] - 1);
            if (!Position.Context.IsShip(host) || Position.Context.State.Owners[host] != decision.Player) continue;
            if (card.CommandTokens == 0)
            {
                materializedCommands.MaterializeCommanderInventory(index);
                continue;
            }
            pendingCommanderInventory = new(index, decision.Player, card.CommandTokens, []);
            current = null; keyGeneration = -1; encoder.Invalidate(); return;
        }
    }

    private static ShipCommand[] InventoryCommands(PrivateCommanderInventory inventory)
        => Enum.GetValues<ShipCommand>().Where(command => inventory.Selected.Length == 0 ||
            (int)command >= inventory.Selected[^1]).ToArray();

    private void ApplyCommanderInventoryHypothesis(PrivateCommanderInventory inventory, ShipCommand command)
    {
        int[] selected = [.. inventory.Selected, (int)command];
        if (selected.Length < inventory.Total)
            pendingCommanderInventory = inventory with { Selected = selected };
        else
        {
            var card = Position.Context.State.Upgrades[inventory.Upgrade];
            card.ClearTypedCommandTokens();
            foreach (int value in selected) card.AddTypedCommandToken((ShipCommand)value);
            materializedCommands.MaterializeCommanderInventory(inventory.Upgrade);
            pendingCommanderInventory = null;
        }
        // Reset cached native legal options built from the neutral placeholder.
        Adapter.ResetPrefix(); Invalidate();
    }
}
