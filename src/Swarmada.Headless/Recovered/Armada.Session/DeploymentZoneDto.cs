#nullable disable
using System;

namespace Armada.Session;

[Serializable]
public sealed class DeploymentZoneDto
{
	public int playerId;

	public float minimumX;

	public float maximumX;

	public float minimumZ;

	public float maximumZ;
}
