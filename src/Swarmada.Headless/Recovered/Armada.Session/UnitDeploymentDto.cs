#nullable disable
using System;
using Armada.Core;

namespace Armada.Session;

[Serializable]
public sealed class UnitDeploymentDto
{
	public int schemaVersion = 1;

	public string runtimeUnitId = string.Empty;

	public string fleetEntryInstanceId = string.Empty;

	public int ownerPlayerId;

	public RuntimeUnitKind unitKind;

	public float positionX;

	public float positionY;

	public float positionZ;

	public float yawDegrees;

	public int startingSpeed;
}
