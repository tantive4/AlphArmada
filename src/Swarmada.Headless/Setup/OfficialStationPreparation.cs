using System.Text.Json;
using Armada.Data;
using Armada.Fleet;
using Armada.Session;
using Swarmada.Headless.Catalog;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Setup;

public static partial class MatchPreparation
{
    private static ShipCatalogEntry CreateUnarmedStation(MatchConfigDto config)
    {
        var given = config.map.placements.FirstOrDefault(p => p.contentId is "unarmed-station" or "space-station" or "nebula-field");
        var physicalSize = given is null ? ObstacleShapeCatalog.Get("station").Size : MapObstacleGeometry.Size(given);
        float width = physicalSize.X / TabletopMeasurements.MillimetresPerWorldUnit, length = physicalSize.Y / TabletopMeasurements.MillimetresPerWorldUnit;
        var zero = new { redDice = 0, blueDice = 0, blackDice = 0 };
        var definition = ShipDefinition.FromDefinitionFields(JsonSerializer.SerializeToElement(new
        {
            shipId = "objective-unarmed-station", displayName = "Unarmed Station", faction = 0, shipType = "station",
            points = 0, upgradeSlots = Array.Empty<string>(), hull = 10,
            frontShields = 0, leftShields = 0, rightShields = 0, rearShields = 0,
            hasAuxiliaryHullZones = false, leftAuxiliaryShields = 0, rightAuxiliaryShields = 0,
            command = 0, engineering = 0, squadron = 0, maximumSpeed = 0,
            braceTokens = 0, redirectTokens = 0, evadeTokens = 0, maneuverChart = Array.Empty<object>(),
            frontBattery = zero, leftBattery = zero, rightBattery = zero, rearBattery = zero,
            leftAuxiliaryBattery = zero, rightAuxiliaryBattery = zero, antiSquadronArmament = zero,
            gameplayFootprint = new { x = width, y = length }
        }));
        return new("objective-unarmed-station", definition, new ShipFleetMetadata("station", Array.Empty<string>()), 0, 0, 0, "Official Station Assault objective");
    }
}
