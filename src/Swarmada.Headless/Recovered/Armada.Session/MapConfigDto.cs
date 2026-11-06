#nullable disable
using System;

namespace Armada.Session;

[Serializable]
public sealed class MapConfigDto
{
	public int schemaVersion = 1;

	public string mapId = string.Empty;

	public string displayName = string.Empty;

	public string scenarioId = "default";

	public string savedUtc = string.Empty;

	public float widthWorldUnits;

	public float lengthWorldUnits;

	public float playPlaneY;

	public DeploymentZoneDto[] deploymentZones = Array.Empty<DeploymentZoneDto>();

	public MapPlacementDto[] placements = Array.Empty<MapPlacementDto>();
}
