using System.Numerics;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

public static partial class OfficialObjectives
{
    public static bool IsFlagship(GameContext game, int ship) => game.IsShip(ship) && game.Topology.Upgrades.Any(u =>
        game.State.Registers[game.Topology.Registers.UpgradeOwner(u.UpgradeIndex)] == ship + 1 && game.State.Upgrades[u.UpgradeIndex].Definition.slot == "commander");
    private static bool Placed(GameContext game, int entity) => game.State.Registers[game.Topology.Registers[$"entity.{entity}.placed"]] != 0;
    public static bool CanOfferDeployment(GameContext game, int entity)
    {
        if (State(game) is not { } state) return true;
        if (game.IsShip(entity) && game.Topology.Ships[entity].IsStation) return false;
        int owner = game.State.Owners[entity]; string id = state.Id;
        if (id is "superior-positions" or "solar-corona" && owner == SecondPlayer(game) &&
            Enumerable.Range(0, game.Topology.EntityCount).Any(e => game.State.Owners[e] == FirstPlayer(game) && !Placed(game, e) && !game.IsReserved(e))) return false;
        if (id == "fleet-ambush" && !game.IsShip(entity) && game.Topology.Ships.Any(s => s.Owner == owner && !s.IsScenarioUnit && !Placed(game, s.EntityIndex) && !game.IsReserved(s.EntityIndex))) return false;
        if (id == "surprise-attack" && owner == FirstPlayer(game) &&
            game.Topology.Ships.Any(s => s.Owner == owner && IsFlagship(game, s.EntityIndex) && !Placed(game, s.EntityIndex))) return IsFlagship(game, entity);
        if (!game.IsShip(entity) && game.Topology.Squadrons.Count(s => s.Owner == owner && !Placed(game, s.EntityIndex) && !game.IsReserved(s.EntityIndex)) == 1 &&
            game.Topology.Ships.Any(s => s.Owner == owner && !s.IsScenarioUnit && !Placed(game, s.EntityIndex) && !game.IsReserved(s.EntityIndex))) return false;
        return true;
    }
    public static OrientedRectangle AmbushZone(GameContext game)
    { var area = SetupArea(game); return new(area.Pose, Vector2.Max(new(.001f), area.Size - new Vector2(TabletopMeasurements.DistanceBand(5) * 2))); }
    /// <returns>null for normal deployment-zone rules; true for a valid objective override; false for invalid.</returns>
    public static bool? CheckShipDeployment(GameContext game, int entity, OrientedRectangle shape)
    {
        if (State(game) is not { } state) return null;
        int owner = game.State.Owners[entity];
        if (state.Id == "fleet-ambush" && owner == FirstPlayer(game) && state.Flags.GetValueOrDefault("deployment.ships." + owner) % 2 == 0)
            return ValidateAmbushFit(game, entity, shape);
        if (state.Id == "surprise-attack" && owner == FirstPlayer(game) && IsFlagship(game, entity) && !Placed(game, entity))
            return game.Topology.Ships[entity].CatalogEntry.Definition.HasAuxiliaryHullZones ? ValidateHugeSurpriseFit(game, entity, shape)
                : Obstacles(game).Any(i => ObstacleId(game, i) == "station" && Geometry2D.Overlaps(shape, ObstacleGeometry.Shape(game, i)));
        return null;
    }
    public static void RecordShipDeployment(GameContext game, int entity)
    { if (State(game) is { } state) { string key = "deployment.ships." + game.State.Owners[entity]; state.Flags[key] = state.Flags.GetValueOrDefault(key) + 1; } }
    public static void ValidateDeploymentSpeed(GameContext game, int entity, int speed)
    {
        if (Is(game, "surprise-attack") && game.State.Owners[entity] == FirstPlayer(game) && speed > (game.Topology.Ships[entity].CatalogEntry.Definition.MaximumSpeed + 1) / 2)
            throw new ArgumentException("Surprise Attack limits the first player's deployment speed to half maximum, rounded up.");
    }
}
