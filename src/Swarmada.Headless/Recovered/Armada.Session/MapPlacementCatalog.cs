#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;

namespace Armada.Session;

public static class MapPlacementCatalog
{
	public const string NebulaContentId = "nebula-field";

	private const string LegacyStationContentId = "space-station";

	private static readonly MapPlacementTemplate[] Values = new MapPlacementTemplate[7]
	{
		new MapPlacementTemplate("asteroid-field-a", "Basalt Asteroid Cluster", MapPlacementKind.Obstacle, 5.4f, 4.1f, 0f, "Maps/Asteroids/T_Asteroid_Basalt_Albedo"),
		new MapPlacementTemplate("asteroid-field-b", "Iron Asteroid Cluster", MapPlacementKind.Obstacle, 5f, 3.6f, 0f, "Maps/Asteroids/T_Asteroid_Iron_Albedo"),
		new MapPlacementTemplate("asteroid-field-c", "Silicate Asteroid Cluster", MapPlacementKind.Obstacle, 4.7f, 4f, 0f, "Maps/Asteroids/T_Asteroid_Silicate_Albedo"),
		new MapPlacementTemplate("debris-field-a", "Debris Field A", MapPlacementKind.Obstacle, 5.8f, 3.4f),
		new MapPlacementTemplate("debris-field-b", "Debris Field B", MapPlacementKind.Obstacle, 5.1f, 3.7f),
		new MapPlacementTemplate("nebula-field", "Nebula Field", MapPlacementKind.Obstacle, 5.4f, 4.1f),
		new MapPlacementTemplate("objective-token", "Objective Token", MapPlacementKind.Objective, 1.25f, 1.25f, 6f)
	};

	private static readonly MapPlacementTemplate[] ObstacleValues = Values.Where((MapPlacementTemplate value) => value.Kind == MapPlacementKind.Obstacle).ToArray();

	public static IReadOnlyList<MapPlacementTemplate> All => Values;

	public static IReadOnlyList<MapPlacementTemplate> Obstacles => ObstacleValues;

	public static bool TryGet(string contentId, out MapPlacementTemplate template)
	{
		if (IsLegacyStation(contentId))
		{
			contentId = "nebula-field";
		}
		template = Values.FirstOrDefault((MapPlacementTemplate value) => string.Equals(value.ContentId, contentId, StringComparison.OrdinalIgnoreCase));
		return template != null;
	}

	public static bool IsNebula(string contentId)
	{
		if (!string.Equals(contentId, "nebula-field", StringComparison.OrdinalIgnoreCase))
		{
			return IsLegacyStation(contentId);
		}
		return true;
	}

	public static bool IsAsteroid(string contentId)
	{
		return contentId?.StartsWith("asteroid-field-", StringComparison.OrdinalIgnoreCase) ?? false;
	}

	public static bool IsDebris(string contentId)
	{
		return contentId?.StartsWith("debris-field-", StringComparison.OrdinalIgnoreCase) ?? false;
	}

	public static bool IsLegacyStation(string contentId)
	{
		return string.Equals(contentId, "space-station", StringComparison.OrdinalIgnoreCase);
	}
}
