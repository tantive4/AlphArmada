#nullable disable
using System;

namespace Armada.Session;

[Serializable]
public sealed class MatchConfigDto
{
	public int schemaVersion = 1;

	public string matchId = string.Empty;

	public string ruleset = "official";

	public string contentHash = string.Empty;

	public int firstPlayerId = 1;

	public string objectiveId = string.Empty;

	// Headless training variant. Explicitly opt in for the full relocation rules.
	public bool mapMovementEnabled = false;

	public int escortPlayerId = 2;

	public PlayerSlotDto[] players = Array.Empty<PlayerSlotDto>();

	public MapConfigDto map;

	public UnitDeploymentDto[] deployments = Array.Empty<UnitDeploymentDto>();

	public static MatchConfigDto Create()
	{
		return new MatchConfigDto
		{
			matchId = Guid.NewGuid().ToString("N")
		};
	}
}
