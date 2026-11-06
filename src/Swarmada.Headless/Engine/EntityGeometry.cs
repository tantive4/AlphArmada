using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

/// <summary>Natural unit footprints, including physical Station Assault outlines.</summary>
public static class EntityGeometry
{
    public static float Distance(GameContext game, int first, int second)
    {
        bool firstStation = game.IsShip(first) && game.Topology.Ships[first].IsStation;
        bool secondStation = game.IsShip(second) && game.Topology.Ships[second].IsStation;
        if (firstStation)
        {
            var shape = CombatModule.StationShape(game, first);
            return secondStation ? Geometry2D.Distance(shape, CombatModule.StationShape(game, second))
                : game.IsShip(second) ? Geometry2D.Distance(shape, game.ShipBase(second)) : Geometry2D.Distance(shape, game.SquadronBase(second));
        }
        if (secondStation)
        {
            var shape = CombatModule.StationShape(game, second);
            return game.IsShip(first) ? Geometry2D.Distance(game.ShipBase(first), shape) : Geometry2D.Distance(game.SquadronBase(first), shape);
        }
        return game.IsShip(first) ? game.IsShip(second) ? Geometry2D.Distance(game.ShipBase(first), game.ShipBase(second))
            : Geometry2D.Distance(game.SquadronBase(second), game.ShipBase(first))
            : game.IsShip(second) ? Geometry2D.Distance(game.SquadronBase(first), game.ShipBase(second)) : Geometry2D.Distance(game.SquadronBase(first), game.SquadronBase(second));
    }
}
