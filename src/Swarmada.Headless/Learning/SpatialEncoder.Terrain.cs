using System.Numerics;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Search;

namespace Swarmada.Headless.Learning;

public sealed partial class SpatialEncoder
{
    private Vector2[] cells = [];
    private ulong lastTerrainStamp;
    private bool terrainInitialized;
    public bool TerrainCacheHit { get; private set; }
    public long TerrainBuilds { get; private set; }

    private void InitializeTerrain()
    {
        if (!Options.IncludeTerrain) return;
        int height = Options.Height, width = Options.Width;
        cells = new Vector2[checked(height * width)]; var board = output.BoardSizeMillimetres;
        for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
            cells[y * width + x] = new(((x + .5f) / width - .5f) * board.X, ((y + .5f) / height - .5f) * board.Y);
    }

    /// <summary>Four neutral terrain channels: asteroid, debris, station, other.
    /// Cells sample exact concave outlines at their centers; a small/thin object
    /// with no sampled cell marks its nearest physical center cell. This map is
    /// an approximate contextual feature, never collision or action legality.
    /// Source poses, sizes, shape IDs and reflection remain exact in Pieces.
    /// No unit/objective-token raster is produced; the model scatters coordinates.</summary>
    private void EncodeTerrain(GameContext game, int observer)
    {
        if (!Options.IncludeTerrain) { TerrainCacheHit = true; return; }
        ulong stamp = SearchHash.Add(SearchHash.Initial, (ulong)observer);
        void Number(float value) => stamp = SearchHash.Add(stamp, BitConverter.SingleToUInt32Bits(value));
        for (int index = 0; index < game.State.SpatialObjects.Count; index++)
        {
            var piece = game.State.SpatialObjects[index]; if (piece.Kind != 0) continue;
            stamp = SearchHash.Add(stamp, (ulong)index); stamp = SearchHash.Add(stamp, piece.Active ? 1UL : 0);
            if (!piece.Active) continue;
            Number(piece.Pose.Position.X); Number(piece.Pose.Position.Y); Number(piece.Pose.YawDegrees);
            Number(piece.Size.X); Number(piece.Size.Y); stamp = SearchHash.Add(stamp, piece.Flipped ? 1UL : 0);
            stamp = SearchHash.Add(stamp, piece.ShapeId); stamp = SearchHash.Add(stamp, OfficialObjectives.ObstacleId(game, index));
        }
        TerrainCacheHit = terrainInitialized && lastTerrainStamp == stamp;
        if (TerrainCacheHit) return;
        Array.Clear(output.TerrainRaster);
        for (int index = 0; index < game.State.SpatialObjects.Count; index++)
        {
            var piece = game.State.SpatialObjects[index]; if (piece.Kind != 0 || !piece.Active) continue;
            string id = OfficialObjectives.ObstacleId(game, index);
            int channel = id.Contains("asteroid", StringComparison.Ordinal) ? 0 : id.Contains("debris", StringComparison.Ordinal) ? 1
                : ObstacleShapeCatalog.CanonicalId(id) == "station" ? 2 : 3;
            Raster(ObstacleGeometry.Shape(game, index), channel, observer);
        }
        lastTerrainStamp = stamp; terrainInitialized = true; TerrainBuilds++;
    }

    private void Raster(PolygonFootprint polygon, int channel, int observer)
    {
        int height = Options.Height, width = Options.Width, plane = height * width;
        var board = output.BoardSizeMillimetres;
        Span<Vector2> corners = stackalloc Vector2[4]; polygon.Bounds.CopyCorners(corners);
        Vector2 minimum = new(float.PositiveInfinity), maximum = new(float.NegativeInfinity);
        foreach (var corner in corners)
        {
            var point = observer == 1 ? corner : -corner;
            minimum = Vector2.Min(minimum, point); maximum = Vector2.Max(maximum, point);
        }
        int minX = (int)MathF.Floor((minimum.X / board.X + .5f) * width), maxX = (int)MathF.Ceiling((maximum.X / board.X + .5f) * width);
        int minY = (int)MathF.Floor((minimum.Y / board.Y + .5f) * height), maxY = (int)MathF.Ceiling((maximum.Y / board.Y + .5f) * height);
        if (maxX < 0 || maxY < 0 || minX >= width || minY >= height) return;
        minX = Math.Clamp(minX, 0, width - 1); maxX = Math.Clamp(maxX, 0, width - 1);
        minY = Math.Clamp(minY, 0, height - 1); maxY = Math.Clamp(maxY, 0, height - 1);
        bool any = false;
        for (int y = minY; y <= maxY; y++) for (int x = minX; x <= maxX; x++)
        {
            int cell = y * width + x; var point = observer == 1 ? cells[cell] : -cells[cell];
            if (polygon.Contains(point)) { output.TerrainRaster[channel * plane + cell] = 1; any = true; }
        }
        if (any) return;
        // Choose exact-grid-boundary ties before observer rotation, ensuring that
        // the two observers see rotations of the same physical sampled cell.
        int centerX = (int)MathF.Floor((polygon.Center.X / board.X + .5f) * width);
        int centerY = (int)MathF.Floor((polygon.Center.Y / board.Y + .5f) * height);
        if (centerX < 0 || centerY < 0 || centerX >= width || centerY >= height) return;
        if (observer == 2) { centerX = width - 1 - centerX; centerY = height - 1 - centerY; }
        output.TerrainRaster[channel * plane + centerY * width + centerX] = 1;
    }
}
