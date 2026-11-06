#nullable disable
using System;

namespace Armada.Session;

[Serializable]
public sealed class MapPlacementDto
{
	public string placementId = string.Empty;

	public MapPlacementKind kind;

	public string contentId = string.Empty;

	public string displayName = string.Empty;

	public float positionX;

	public float positionZ;

	public float yawDegrees;

	public bool flipped;

	public string shapeId = string.Empty;

	public bool useCustomSize;

	public float widthWorldUnits;

	public float lengthWorldUnits;

	public float interactionRangeWorldUnits;

	public int ownerPlayerId;

	public int startingSpeed;
}
