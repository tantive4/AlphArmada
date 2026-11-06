using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

public static class ObstacleGeometry
{
    public static PolygonFootprint NewShape(GameContext game, string kind, Pose pose, System.Numerics.Vector2 size, bool flipped = false)
        => new(ObstacleShapeCatalog.Get(OfficialObjectives.NextObstacleShapeId(game, kind)), pose, size, flipped);
    public static PolygonFootprint Shape(GameContext game, int index, Pose? pose = null, bool? flipped = null)
    {
        var piece = game.State.SpatialObjects[index];
        if (piece.Kind != 0) throw new ArgumentException("An obstacle spatial index is required.", nameof(index));
        var definition = ObstacleShapeCatalog.Get(!string.IsNullOrEmpty(piece.ShapeId) ? piece.ShapeId : OfficialObjectives.ObstacleId(game, index));
        return new(definition, pose ?? piece.Pose, piece.Size, flipped ?? piece.Flipped);
    }
}
