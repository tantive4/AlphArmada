using System.Numerics;
using Armada.Session;

namespace Swarmada.Headless.Geometry;

/// <summary>Map DTOs use recovered world units; official outlines use physical mm.
/// Custom sizing is explicit, so old envelope dimensions cannot silently stretch a token.</summary>
public static class MapObstacleGeometry
{
    public static PolygonDefinition Definition(MapPlacementDto placement)
        => ObstacleShapeCatalog.Get(string.IsNullOrWhiteSpace(placement.shapeId) ? placement.contentId : placement.shapeId);
    public static Vector2 Size(MapPlacementDto placement)
    {
        var definition = Definition(placement);
        return placement.useCustomSize || definition.Id == "custom-rectangle"
            ? new(TabletopMeasurements.FromWorldUnits(placement.widthWorldUnits), TabletopMeasurements.FromWorldUnits(placement.lengthWorldUnits))
            : definition.Size;
    }
    public static PolygonFootprint Shape(MapPlacementDto placement)
        => new(Definition(placement), new(new(TabletopMeasurements.FromWorldUnits(placement.positionX), TabletopMeasurements.FromWorldUnits(placement.positionZ)),
            placement.yawDegrees), Size(placement), placement.flipped);
}
