#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Swarmada.Headless.Geometry;

namespace Armada.Session;

public static class MapLayoutUtility
{
	public static MapConfigDto CreateBlankStandard(string displayName = "Untitled Battlefield")
	{
		MapConfigDto mapConfigDto = DefaultMatchFactory.CreateStandardMap();
		mapConfigDto.mapId = Guid.NewGuid().ToString("N");
		mapConfigDto.displayName = (string.IsNullOrWhiteSpace(displayName) ? "Untitled Battlefield" : displayName.Trim());
		mapConfigDto.savedUtc = string.Empty;
		mapConfigDto.placements = Array.Empty<MapPlacementDto>();
		return mapConfigDto;
	}

	public static MapConfigDto Clone(MapConfigDto source)
	{
		if (source == null)
		{
			return null;
		}
		DeploymentZoneDto[] array = source.deploymentZones ?? Array.Empty<DeploymentZoneDto>();
		DeploymentZoneDto[] array2 = new DeploymentZoneDto[array.Length];
		for (int i = 0; i < array.Length; i++)
		{
			DeploymentZoneDto deploymentZoneDto = array[i];
			array2[i] = ((deploymentZoneDto == null) ? null : new DeploymentZoneDto
			{
				playerId = deploymentZoneDto.playerId,
				minimumX = deploymentZoneDto.minimumX,
				maximumX = deploymentZoneDto.maximumX,
				minimumZ = deploymentZoneDto.minimumZ,
				maximumZ = deploymentZoneDto.maximumZ
			});
		}
		MapPlacementDto[] array3 = source.placements ?? Array.Empty<MapPlacementDto>();
		MapPlacementDto[] array4 = new MapPlacementDto[array3.Length];
		for (int j = 0; j < array3.Length; j++)
		{
			MapPlacementDto mapPlacementDto = array3[j];
			bool flag = mapPlacementDto != null && MapPlacementCatalog.IsLegacyStation(mapPlacementDto.contentId);
			array4[j] = ((mapPlacementDto == null) ? null : new MapPlacementDto
			{
				placementId = mapPlacementDto.placementId,
				kind = mapPlacementDto.kind,
				contentId = (flag ? "nebula-field" : mapPlacementDto.contentId),
				displayName = (flag ? "Nebula Field" : mapPlacementDto.displayName),
				positionX = mapPlacementDto.positionX,
				positionZ = mapPlacementDto.positionZ,
				yawDegrees = mapPlacementDto.yawDegrees,
				flipped = mapPlacementDto.flipped,
				shapeId = mapPlacementDto.shapeId,
				useCustomSize = mapPlacementDto.useCustomSize,
				widthWorldUnits = mapPlacementDto.widthWorldUnits,
				lengthWorldUnits = mapPlacementDto.lengthWorldUnits,
				interactionRangeWorldUnits = mapPlacementDto.interactionRangeWorldUnits,
				ownerPlayerId = mapPlacementDto.ownerPlayerId,
				startingSpeed = mapPlacementDto.startingSpeed
			});
		}
		return new MapConfigDto
		{
			schemaVersion = source.schemaVersion,
			mapId = source.mapId,
			displayName = source.displayName,
			scenarioId = MapScenarioCatalog.Normalize(source.scenarioId),
			savedUtc = source.savedUtc,
			widthWorldUnits = source.widthWorldUnits,
			lengthWorldUnits = source.lengthWorldUnits,
			playPlaneY = source.playPlaneY,
			deploymentZones = array2,
			placements = array4
		};
	}

	public static MapPlacementDto AddPlacement(MapConfigDto map, string contentId, Vector2 requestedPosition)
	{
		if (map == null)
		{
			throw new ArgumentNullException("map");
		}
		if (!MapPlacementCatalog.TryGet(contentId, out var template))
		{
			throw new ArgumentException("Unknown map placement template.", "contentId");
		}
		MapPlacementDto mapPlacementDto = new MapPlacementDto
		{
			placementId = Guid.NewGuid().ToString("N"),
			kind = template.Kind,
			contentId = template.ContentId,
			displayName = template.DisplayName,
			widthWorldUnits = template.WidthWorldUnits,
			lengthWorldUnits = template.LengthWorldUnits,
			interactionRangeWorldUnits = template.InteractionRangeWorldUnits
		};
		if (mapPlacementDto.kind == MapPlacementKind.Obstacle)
		{
			var size = MapObstacleGeometry.Size(mapPlacementDto);
			mapPlacementDto.widthWorldUnits = size.X / TabletopMeasurements.MillimetresPerWorldUnit;
			mapPlacementDto.lengthWorldUnits = size.Y / TabletopMeasurements.MillimetresPerWorldUnit;
		}
		SetPosition(map, mapPlacementDto, requestedPosition);
		List<MapPlacementDto> list = (map.placements ?? Array.Empty<MapPlacementDto>()).Where((MapPlacementDto value) => value != null).ToList();
		list.Add(mapPlacementDto);
		map.placements = list.ToArray();
		return mapPlacementDto;
	}

	public static bool RemovePlacement(MapConfigDto map, string placementId)
	{
		if (map == null || string.IsNullOrWhiteSpace(placementId))
		{
			return false;
		}
		MapPlacementDto[] array = map.placements ?? Array.Empty<MapPlacementDto>();
		MapPlacementDto[] array2 = array.Where((MapPlacementDto value) => value != null && !string.Equals(value.placementId, placementId, StringComparison.Ordinal)).ToArray();
		if (array2.Length == array.Length)
		{
			return false;
		}
		map.placements = array2;
		return true;
	}

	public static bool RotatePlacement(MapConfigDto map, string placementId, float deltaDegrees)
	{
		MapPlacementDto mapPlacementDto = FindPlacement(map, placementId);
		if (mapPlacementDto == null)
		{
			return false;
		}
		mapPlacementDto.yawDegrees = NormalizeYaw(mapPlacementDto.yawDegrees + deltaDegrees);
		SetPosition(map, mapPlacementDto, new Vector2(mapPlacementDto.positionX, mapPlacementDto.positionZ));
		return true;
	}

	public static bool SetPlacementPosition(MapConfigDto map, string placementId, Vector2 requestedPosition)
	{
		MapPlacementDto mapPlacementDto = FindPlacement(map, placementId);
		if (mapPlacementDto == null)
		{
			return false;
		}
		SetPosition(map, mapPlacementDto, requestedPosition);
		return true;
	}

	public static MapPlacementDto FindPlacement(MapConfigDto map, string placementId)
	{
		if (map == null || string.IsNullOrWhiteSpace(placementId))
		{
			return null;
		}
		return (map.placements ?? Array.Empty<MapPlacementDto>()).FirstOrDefault((MapPlacementDto value) => value != null && string.Equals(value.placementId, placementId, StringComparison.Ordinal));
	}

	private static void SetPosition(MapConfigDto map, MapPlacementDto placement, Vector2 requestedPosition)
	{
		if (placement.kind == MapPlacementKind.Obstacle)
		{
			var shape = MapObstacleGeometry.Shape(placement).At(new Pose(Vector2.Zero, placement.yawDegrees));
			var minimum = new Vector2(float.PositiveInfinity); var maximum = new Vector2(float.NegativeInfinity);
			for (int i = 0; i < shape.Count; i++) { var point = shape.Vertex(i) / TabletopMeasurements.MillimetresPerWorldUnit; minimum = Vector2.Min(minimum, point); maximum = Vector2.Max(maximum, point); }
			var half = new Vector2(map.widthWorldUnits, map.lengthWorldUnits) / 2;
			var lower = -half - minimum; var upper = half - maximum;
			if (lower.X > upper.X || lower.Y > upper.Y) throw new ArgumentException("The physical obstacle cannot fit inside this map.");
			var position = Vector2.Clamp(requestedPosition, lower, upper);
			placement.positionX = position.X; placement.positionZ = position.Y; return;
		}
		float f = placement.yawDegrees * (MathF.PI / 180f);
		float num = MathF.Abs(MathF.Cos(f));
		float num2 = MathF.Abs(MathF.Sin(f));
		float num3 = num * placement.widthWorldUnits * 0.5f + num2 * placement.lengthWorldUnits * 0.5f;
		float num4 = num2 * placement.widthWorldUnits * 0.5f + num * placement.lengthWorldUnits * 0.5f;
		float num5 = MathF.Max(0f, map.widthWorldUnits * 0.5f - num3);
		float num6 = MathF.Max(0f, map.lengthWorldUnits * 0.5f - num4);
		placement.positionX = Math.Clamp(requestedPosition.X, 0f - num5, num5);
		placement.positionZ = Math.Clamp(requestedPosition.Y, 0f - num6, num6);
	}

	private static float NormalizeYaw(float value)
	{
		value %= 360f;
		if (!(value < 0f))
		{
			return value;
		}
		return value + 360f;
	}
}
