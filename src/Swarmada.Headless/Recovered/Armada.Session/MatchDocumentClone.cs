#nullable disable
using System;
using Armada.Fleet;

namespace Armada.Session;

public static class MatchDocumentClone
{
	public static MatchConfigDto Clone(MatchConfigDto source)
	{
		if (source == null)
		{
			return null;
		}
		PlayerSlotDto[] array = source.players ?? Array.Empty<PlayerSlotDto>();
		PlayerSlotDto[] array2 = new PlayerSlotDto[array.Length];
		for (int i = 0; i < array.Length; i++)
		{
			PlayerSlotDto playerSlotDto = array[i];
			array2[i] = ((playerSlotDto == null) ? null : new PlayerSlotDto
			{
				schemaVersion = playerSlotDto.schemaVersion,
				playerId = playerSlotDto.playerId,
				displayName = playerSlotDto.displayName,
				ready = playerSlotDto.ready,
				controlKind = playerSlotDto.controlKind,
				aiDifficulty = playerSlotDto.aiDifficulty,
				fleet = FleetDraft.Clone(playerSlotDto.fleet)
			});
		}
		UnitDeploymentDto[] array3 = source.deployments ?? Array.Empty<UnitDeploymentDto>();
		UnitDeploymentDto[] array4 = new UnitDeploymentDto[array3.Length];
		for (int j = 0; j < array3.Length; j++)
		{
			UnitDeploymentDto unitDeploymentDto = array3[j];
			array4[j] = ((unitDeploymentDto == null) ? null : new UnitDeploymentDto
			{
				schemaVersion = unitDeploymentDto.schemaVersion,
				runtimeUnitId = unitDeploymentDto.runtimeUnitId,
				fleetEntryInstanceId = unitDeploymentDto.fleetEntryInstanceId,
				ownerPlayerId = unitDeploymentDto.ownerPlayerId,
				unitKind = unitDeploymentDto.unitKind,
				positionX = unitDeploymentDto.positionX,
				positionY = unitDeploymentDto.positionY,
				positionZ = unitDeploymentDto.positionZ,
				yawDegrees = unitDeploymentDto.yawDegrees,
				startingSpeed = unitDeploymentDto.startingSpeed
			});
		}
		return new MatchConfigDto
		{
			schemaVersion = source.schemaVersion,
			matchId = source.matchId,
			ruleset = source.ruleset,
			contentHash = source.contentHash,
			firstPlayerId = source.firstPlayerId,
			objectiveId = source.objectiveId,
			mapMovementEnabled = source.mapMovementEnabled,
			escortPlayerId = source.escortPlayerId,
			players = array2,
			map = CloneMap(source.map),
			deployments = array4
		};
	}

	private static MapConfigDto CloneMap(MapConfigDto source)
	{
		return MapLayoutUtility.Clone(source);
	}
}
