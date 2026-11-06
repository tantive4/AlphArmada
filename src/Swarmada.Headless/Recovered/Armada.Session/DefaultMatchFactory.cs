#nullable disable
using System;
using System.Collections.Generic;
using Armada.Core;
using Armada.Fleet;

namespace Armada.Session;

public static class DefaultMatchFactory
{
	private const float ShipSpacing = 8.5f;

	private const float SquadronSpacing = 2.6f;

	public static MatchConfigDto CreateLocal(FleetListDto playerOneFleet, FleetListDto playerTwoFleet, int firstPlayerId = 1, float playPlaneY = 0f, MapConfigDto map = null, BattleStartingMode startingMode = BattleStartingMode.StandardFront, IMatchContentResolver content = null, string objectiveId = "targeting-beacons")
	{
		if (playerOneFleet == null)
		{
			throw new ArgumentNullException("playerOneFleet");
		}
		if (playerTwoFleet == null)
		{
			throw new ArgumentNullException("playerTwoFleet");
		}
		if (firstPlayerId != 1 && firstPlayerId != 2)
		{
			throw new ArgumentOutOfRangeException("firstPlayerId", firstPlayerId, "First player must be Player 1 or Player 2.");
		}
		MatchConfigDto matchConfigDto = MatchConfigDto.Create();
		matchConfigDto.ruleset = FleetRuleset.StandardPlay400.Id;
		matchConfigDto.firstPlayerId = firstPlayerId;
		matchConfigDto.objectiveId = objectiveId;
		matchConfigDto.escortPlayerId = 2;
		matchConfigDto.players = new PlayerSlotDto[2]
		{
			new PlayerSlotDto
			{
				playerId = 1,
				displayName = "Player 1",
				ready = true,
				controlKind = PlayerControlKind.Human,
				aiDifficulty = BattleAiDifficulty.Cadet,
				fleet = FleetDraft.Clone(playerOneFleet)
			},
			new PlayerSlotDto
			{
				playerId = 2,
				displayName = "Player 2",
				ready = true,
				controlKind = PlayerControlKind.Human,
				aiDifficulty = BattleAiDifficulty.Cadet,
				fleet = FleetDraft.Clone(playerTwoFleet)
			}
		};
		matchConfigDto.map = ((map == null) ? CreateStandardMap(playPlaneY) : MapLayoutUtility.Clone(map));
		matchConfigDto.map.playPlaneY = playPlaneY;
		startingMode = MapScenarioCatalog.ResolveStartingMode(matchConfigDto.map.scenarioId, startingMode);
		ConfigureDeploymentZones(matchConfigDto.map, startingMode);
		List<UnitDeploymentDto> list = new List<UnitDeploymentDto>();
		AddFleetDeployments(list, matchConfigDto.matchId, 1, matchConfigDto.players[0].fleet, matchConfigDto.map, startingMode, content);
		AddFleetDeployments(list, matchConfigDto.matchId, 2, matchConfigDto.players[1].fleet, matchConfigDto.map, startingMode, content);
		matchConfigDto.deployments = list.ToArray();
		SessionValidationResult sessionValidationResult = SessionValidation.ValidateMatch(matchConfigDto, content);
		if (!sessionValidationResult.IsValid)
		{
			throw new InvalidOperationException("Generated match failed structural validation: " + string.Join(" ", sessionValidationResult.Errors));
		}
		return matchConfigDto;
	}

	public static MapConfigDto CreateStandardMap(float playPlaneY = 0f)
	{
		MapConfigDto obj = new MapConfigDto
		{
			mapId = "standard-6x3",
			displayName = "Standard 6' x 3'",
			scenarioId = "default",
			widthWorldUnits = 80.035f,
			lengthWorldUnits = 40.0175f,
			playPlaneY = playPlaneY
		};
		ConfigureLongEdgeDeploymentZones(obj);
		return obj;
	}

	public static void ConfigureDeploymentZones(MapConfigDto map, BattleStartingMode startingMode)
	{
		startingMode = MapScenarioCatalog.ResolveStartingMode(map?.scenarioId, startingMode);
		if (startingMode == BattleStartingMode.StandardFront)
		{
			ConfigureLongEdgeDeploymentZones(map);
		}
		else
		{
			ConfigureShortEdgeDeploymentZones(map);
		}
	}

	public static BattleStartingMode GetStartingMode(MapConfigDto map)
	{
		if (MapScenarioCatalog.LocksLongApproach(map?.scenarioId))
		{
			return BattleStartingMode.LongApproach;
		}
		DeploymentZoneDto[] array = map?.deploymentZones;
		if (array == null || array.Length == 0)
		{
			return BattleStartingMode.StandardFront;
		}
		DeploymentZoneDto[] array2 = array;
		foreach (DeploymentZoneDto deploymentZoneDto in array2)
		{
			if (deploymentZoneDto != null)
			{
				float num = Math.Abs(deploymentZoneDto.maximumX - deploymentZoneDto.minimumX);
				if (!(Math.Abs(deploymentZoneDto.maximumZ - deploymentZoneDto.minimumZ) < num))
				{
					return BattleStartingMode.LongApproach;
				}
				return BattleStartingMode.StandardFront;
			}
		}
		return BattleStartingMode.StandardFront;
	}

	public static string StartingModeLabel(BattleStartingMode startingMode)
	{
		if (startingMode != BattleStartingMode.StandardFront)
		{
			return "LONG APPROACH";
		}
		return "STANDARD FRONT";
	}

	public static string StartingModeDescription(BattleStartingMode startingMode)
	{
		if (startingMode != BattleStartingMode.StandardFront)
		{
			return "SHORT-EDGE DEPLOYMENT";
		}
		return "LONG-EDGE DEPLOYMENT";
	}

	public static void ConfigureShortEdgeDeploymentZones(MapConfigDto map)
	{
		if (map == null)
		{
			throw new ArgumentNullException("map");
		}
		float num = map.widthWorldUnits * 0.5f;
		float num2 = map.lengthWorldUnits * 0.5f;
		float num3 = Math.Min(8.09628f, map.widthWorldUnits * 0.5f);
		map.deploymentZones = new DeploymentZoneDto[2]
		{
			new DeploymentZoneDto
			{
				playerId = 1,
				minimumX = 0f - num,
				maximumX = 0f - num + num3,
				minimumZ = 0f - num2,
				maximumZ = num2
			},
			new DeploymentZoneDto
			{
				playerId = 2,
				minimumX = num - num3,
				maximumX = num,
				minimumZ = 0f - num2,
				maximumZ = num2
			}
		};
	}

	public static void ConfigureLongEdgeDeploymentZones(MapConfigDto map)
	{
		if (map == null)
		{
			throw new ArgumentNullException("map");
		}
		float num = map.widthWorldUnits * 0.5f;
		float num2 = map.lengthWorldUnits * 0.5f;
		float num3 = Math.Min(8.09628f, map.lengthWorldUnits * 0.5f);
		float num4 = Math.Min(13.260394f, num);
		map.deploymentZones = new DeploymentZoneDto[2]
		{
			new DeploymentZoneDto
			{
				playerId = 1,
				minimumX = 0f - num + num4,
				maximumX = num - num4,
				minimumZ = 0f - num2,
				maximumZ = 0f - num2 + num3
			},
			new DeploymentZoneDto
			{
				playerId = 2,
				minimumX = 0f - num + num4,
				maximumX = num - num4,
				minimumZ = num2 - num3,
				maximumZ = num2
			}
		};
	}

	private static void AddFleetDeployments(ICollection<UnitDeploymentDto> destination, string matchId, int playerId, FleetListDto fleet, MapConfigDto map, BattleStartingMode startingMode, IMatchContentResolver content)
	{
		float num = map.widthWorldUnits * 0.5f;
		float num2 = map.lengthWorldUnits * 0.5f;
		bool flag = startingMode == BattleStartingMode.StandardFront;
		float num3 = Math.Min(8.09628f, flag ? (map.lengthWorldUnits * 0.5f) : (map.widthWorldUnits * 0.5f));
		float num4 = (flag ? 0f : ((playerId == 1) ? (0f - num + num3 * 0.5f) : (num - num3 * 0.5f)));
		float num5 = ((!flag) ? 0f : ((playerId == 1) ? (0f - num2 + num3 * 0.5f) : (num2 - num3 * 0.5f)));
		float num6 = (flag ? num4 : (num4 + ((playerId == 1) ? 3.369803f : (-3.369803f))));
		float num7 = (flag ? (num5 + ((playerId == 1) ? 3.369803f : (-3.369803f))) : num5);
		float yaw = ((!flag) ? ((playerId == 1) ? 90f : 270f) : ((playerId == 1) ? 0f : 180f));
		ShipBuildDto[] array = fleet.ships ?? Array.Empty<ShipBuildDto>();
		float num8 = 4.25f;
		ShipBuildDto[] array2 = array;
		foreach (ShipBuildDto shipBuildDto in array2)
		{
			if (shipBuildDto != null && content != null && content.TryResolveShip(shipBuildDto.shipDefinitionId, out var rules))
			{
				num8 = Math.Max(num8, rules.FootprintWidth * 0.5f);
			}
		}
		float spacing = BoundedSpacing(array.Length, 8.5f, flag ? num : num2, num8);
		for (int j = 0; j < array.Length; j++)
		{
			ShipBuildDto shipBuildDto2 = array[j];
			if (shipBuildDto2 != null)
			{
				float inset = ResolveShipEdgeInset(content, shipBuildDto2, num3);
				destination.Add(CreateDeployment(matchId, playerId, RuntimeUnitKind.Ship, shipBuildDto2.instanceId, flag ? CenteredAxis(j, array.Length, spacing) : EdgeCoordinate(playerId, num, inset), flag ? EdgeCoordinate(playerId, num2, inset) : CenteredAxis(j, array.Length, spacing), map.playPlaneY, yaw, 2));
			}
		}
		SquadronBuildDto[] array3 = fleet.squadrons ?? Array.Empty<SquadronBuildDto>();
		float spacing2 = BoundedSpacing(array3.Length, 2.6f, flag ? num : num2, 1.3f);
		for (int k = 0; k < array3.Length; k++)
		{
			SquadronBuildDto squadronBuildDto = array3[k];
			if (squadronBuildDto != null)
			{
				destination.Add(CreateDeployment(matchId, playerId, RuntimeUnitKind.Squadron, squadronBuildDto.instanceId, flag ? CenteredAxis(k, array3.Length, spacing2) : num6, flag ? num7 : CenteredAxis(k, array3.Length, spacing2), map.playPlaneY, yaw, 0));
			}
		}
	}

	private static float ResolveShipEdgeInset(IMatchContentResolver content, ShipBuildDto build, float zoneDepth)
	{
		float num = zoneDepth * 0.5f;
		if (content != null && build != null && content.TryResolveShip(build.shipDefinitionId, out var rules) && rules.FootprintLength > 0f)
		{
			num = Math.Max(num, rules.FootprintLength * 0.5f);
		}
		return num;
	}

	private static float EdgeCoordinate(int playerId, float halfExtent, float inset)
	{
		if (playerId != 1)
		{
			return halfExtent - inset;
		}
		return 0f - halfExtent + inset;
	}

	private static UnitDeploymentDto CreateDeployment(string matchId, int playerId, RuntimeUnitKind kind, string instanceId, float x, float z, float y, float yaw, int speed)
	{
		return new UnitDeploymentDto
		{
			runtimeUnitId = RuntimeUnitId.ForMatchEntry(matchId, playerId, kind, instanceId).Value,
			fleetEntryInstanceId = instanceId,
			ownerPlayerId = playerId,
			unitKind = kind,
			positionX = x,
			positionY = y,
			positionZ = z,
			yawDegrees = yaw,
			startingSpeed = speed
		};
	}

	private static float BoundedSpacing(int count, float spacing, float halfExtent, float margin)
	{
		if (count > 1)
		{
			return Math.Min(spacing, Math.Max(0f, 2f * (halfExtent - margin - 0.01f) / (float)(count - 1)));
		}
		return spacing;
	}

	private static float CenteredAxis(int index, int count, float spacing)
	{
		if (count <= 1)
		{
			return 0f;
		}
		return ((float)index - (float)(count - 1) * 0.5f) * spacing;
	}
}
