using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

/// <summary>Cause-specific removal. Leaving the play area deals no damage and cannot be
/// postponed by Rieekan, Tel Trevura or Reserve Hangar Deck.</summary>
public static class PlayAreaDestruction
{
    public static bool WasRemoved(GameContext game, int entity)
        => game.Flow.Flags.ContainsKey(game.Topology.RemovedFromPlayAreaKeys[entity]);
    public static bool Contains(GameContext game, int entity)
        => game.IsShip(entity) ? PlayAreaGeometry.Contains(MovementModule.Board(game), game.ShipBase(entity))
            : PlayAreaGeometry.Contains(MovementModule.Board(game), game.SquadronBase(entity));
    public static bool DestroyIfOutside(GameContext game, int entity)
    {
        if (WasRemoved(game, entity)) return true;
        if (Contains(game, entity)) return false;
        game.Flow.Flags[game.Topology.RemovedFromPlayAreaKeys[entity]] = 1;
        game.Flow.Flags.Remove("rieekan.ship." + entity); game.Flow.Flags.Remove("rieekan.squadron." + entity);
        game.Flow.Flags.Remove("rieekan.pending." + entity);
        if (game.IsShip(entity)) game.State.Ships[entity].DestroyWithoutDamage();
        else game.State.Squadrons[game.SquadronIndex(entity)].DestroyWithoutDamage();
        game.State.Activated[entity] = true;
        return true;
    }
}
