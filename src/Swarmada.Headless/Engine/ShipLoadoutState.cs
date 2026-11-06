using Armada.Data;
using Swarmada.Headless.Engine;

namespace Armada.Rules;

/// <summary>Live ownership view over snapshot-owned upgrade instances; never duplicates mutable card state.</summary>
public sealed class ShipLoadoutState
{
    private readonly GameContext game;
    private readonly int entity;
    public ShipLoadoutState(GameContext game, int entity) { this.game = game; this.entity = entity; }
    public IReadOnlyList<UpgradeCardRuntimeState> Upgrades => game.UpgradesFor(entity);
    public bool HasActive(string id) => Find(id)?.Active == true;
    public UpgradeCardRuntimeState? Find(string id)
    {
        for (int i = 0; i < game.Topology.Upgrades.Count; i++)
        {
            int index = game.Topology.Upgrades[i].UpgradeIndex;
            if (game.State.Registers[game.Topology.Registers.UpgradeOwner(index)] == entity + 1
                && string.Equals(game.State.Upgrades[index].Definition.id, id, StringComparison.OrdinalIgnoreCase))
                return game.State.Upgrades[index];
        }
        return null;
    }
    public bool TryTransferTo(ShipLoadoutState destination, UpgradeCardRuntimeState card)
    {
        if (destination.game != game || destination.entity == entity) return false;
        int index = Array.IndexOf(game.State.Upgrades, card);
        if (index < 0 || game.State.Registers[game.Topology.Registers.UpgradeOwner(index)] != entity + 1) return false;
        game.State.Registers[game.Topology.Registers.UpgradeOwner(index)] = destination.entity + 1; return true;
    }
    public void ReadyForStatusPhase() => ReadyForStatusPhase(null);
    public IReadOnlyList<string> ReadyForStatusPhase(CommandStackState? commands)
    {
        var messages = new List<string>();
        foreach (var card in Upgrades.Where(x => x.Active && x.Exhausted))
        {
            if (UpgradeReadyRules.RequiresPayment(card)) continue;
            card.Ready(); messages.Add(card.Definition.name + " readied.");
        }
        return messages;
    }
    public void ResetAttackUsage() { foreach (var card in Upgrades) card.ResetAttackUsage(); }
}
