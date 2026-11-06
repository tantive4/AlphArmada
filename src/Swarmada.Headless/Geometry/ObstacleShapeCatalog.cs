using System.Numerics;
using System.Text.Json;
using System.Security.Cryptography;

namespace Swarmada.Headless.Geometry;

/// <summary>Physical obstacle outlines recovered from source meshes. Missing official
/// definitions are errors; no bounding-box replacement is used for rules geometry.</summary>
public static class ObstacleShapeCatalog
{
    private static readonly PolygonDefinition CustomRectangle = new("custom-rectangle", Vector2.One,
        [new(-.5f, -.5f), new(.5f, -.5f), new(.5f, .5f), new(-.5f, .5f)], "Explicit synthetic/custom rectangular geometry.");
    private sealed record ShapeData(float[] SizeMillimetres, float[][] Vertices, string Provenance, string SourceSha256);
    private static readonly Lazy<IReadOnlyDictionary<string, PolygonDefinition>> Definitions = new(Load);
    private static readonly Lazy<string> SourceHash = new(() =>
    {
        using var stream = typeof(ObstacleShapeCatalog).Assembly.GetManifestResourceStream("Swarmada.Headless.obstacle-shapes.json")
            ?? throw new InvalidOperationException("Recovered physical obstacle outlines are missing.");
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    });
    public static string SourceSha256 => SourceHash.Value;
    public static string CanonicalId(string id) => id switch
    {
        "asteroid-field-a" => "asteroid-1", "asteroid-field-b" => "asteroid-2", "asteroid-field-c" => "asteroid-3",
        "debris-field-a" => "debris-1", "debris-field-b" => "debris-2", "nebula-field" or "space-station" or "unarmed-station" => "station",
        "dust-field" => "dust-1", "purrgil" => "purrgil-1", "exogorth" => "exogorth-1", _ => id
    };
    public static PolygonDefinition Get(string id)
    {
        string canonical = CanonicalId(id);
        if (canonical == "custom-rectangle") return CustomRectangle;
        if (!Definitions.Value.TryGetValue(canonical, out var definition)) throw new ArgumentException("No physical outline is supplied for obstacle '" + id + "'.", nameof(id));
        return definition;
    }
    private static IReadOnlyDictionary<string, PolygonDefinition> Load()
    {
        using var stream = typeof(ObstacleShapeCatalog).Assembly.GetManifestResourceStream("Swarmada.Headless.obstacle-shapes.json")
            ?? throw new InvalidOperationException("Recovered physical obstacle outlines are missing.");
        var rows = JsonSerializer.Deserialize<Dictionary<string, ShapeData>>(stream, new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidOperationException("Recovered physical obstacle outlines are empty.");
        var result = new Dictionary<string, PolygonDefinition>(StringComparer.Ordinal);
        foreach (var pair in rows)
        {
            var row = pair.Value;
            var vertices = row.Vertices.Select(point => new Vector2(point[0], point[1])).ToArray();
            result.Add(pair.Key, new(pair.Key, new(row.SizeMillimetres[0], row.SizeMillimetres[1]), vertices, row.Provenance + " SHA256 " + row.SourceSha256));
        }
        return result;
    }
}
