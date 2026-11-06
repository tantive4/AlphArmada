#nullable disable
using System;
using System.Collections.Generic;
using Armada.Core;
using Armada.Fleet;
using System.Numerics;
using Swarmada.Headless.Geometry;

namespace Armada.Session;

public static class SessionValidation
{
	private sealed class FleetEntrySets
	{
		private readonly Dictionary<string, string> ships = new Dictionary<string, string>(StringComparer.Ordinal);

		private readonly Dictionary<string, string> squadrons = new Dictionary<string, string>(StringComparer.Ordinal);

		public void Add(RuntimeUnitKind kind, string instanceId, string definitionId)
		{
			Dictionary<string, string> dictionary = ((kind == RuntimeUnitKind.Ship) ? ships : squadrons);
			if (!dictionary.ContainsKey(instanceId))
			{
				dictionary.Add(instanceId, definitionId ?? string.Empty);
			}
		}

		public bool Contains(RuntimeUnitKind kind, string instanceId)
		{
			return kind switch
			{
				RuntimeUnitKind.Squadron => squadrons.ContainsKey(instanceId),
				RuntimeUnitKind.Ship => ships.ContainsKey(instanceId),
				_ => false,
			};
		}

		public bool TryGetDefinitionId(RuntimeUnitKind kind, string instanceId, out string definitionId)
		{
			switch (kind)
			{
			case RuntimeUnitKind.Ship:
				return ships.TryGetValue(instanceId, out definitionId);
			case RuntimeUnitKind.Squadron:
				return squadrons.TryGetValue(instanceId, out definitionId);
			default:
				definitionId = string.Empty;
				return false;
			}
		}

		public IEnumerable<FleetEntryReference> Entries()
		{
			foreach (KeyValuePair<string, string> ship in ships)
			{
				yield return new FleetEntryReference(RuntimeUnitKind.Ship, ship.Key);
			}
			foreach (KeyValuePair<string, string> squadron in squadrons)
			{
				yield return new FleetEntryReference(RuntimeUnitKind.Squadron, squadron.Key);
			}
		}
	}

	private readonly struct FleetEntryReference
	{
		public RuntimeUnitKind Kind { get; }

		public string InstanceId { get; }

		public FleetEntryReference(RuntimeUnitKind kind, string instanceId)
		{
			Kind = kind;
			InstanceId = instanceId;
		}
	}

	private readonly struct DeploymentEntryKey : IEquatable<DeploymentEntryKey>
	{
		private readonly int ownerPlayerId;

		private readonly RuntimeUnitKind kind;

		private readonly string instanceId;

		public DeploymentEntryKey(int owner, RuntimeUnitKind unitKind, string fleetEntryInstanceId)
		{
			ownerPlayerId = owner;
			kind = unitKind;
			instanceId = fleetEntryInstanceId ?? string.Empty;
		}

		public bool Equals(DeploymentEntryKey other)
		{
			if (ownerPlayerId == other.ownerPlayerId && kind == other.kind)
			{
				return string.Equals(instanceId, other.instanceId, StringComparison.Ordinal);
			}
			return false;
		}

		public override bool Equals(object obj)
		{
			if (obj is DeploymentEntryKey other)
			{
				return Equals(other);
			}
			return false;
		}

		public override int GetHashCode()
		{
			return (int)(((uint)(ownerPlayerId * 397) ^ (uint)kind) * 397) ^ StringComparer.Ordinal.GetHashCode(instanceId);
		}
	}

	private const float PlayAreaTolerance = 0.0005f;

	public static SessionValidationResult ValidateFleet(FleetListDto fleet)
	{
		SessionValidationResult result = new SessionValidationResult();
		ValidateFleetInto(fleet, "fleet", result);
		return result;
	}

	public static SessionValidationResult ValidateMap(MapConfigDto map)
	{
		SessionValidationResult result = new SessionValidationResult();
		ValidateMapInto(map, "map", result);
		return result;
	}

	public static SessionValidationResult ValidateMatch(MatchConfigDto match)
	{
		return ValidateMatch(match, null);
	}

	public static SessionValidationResult ValidateMatch(MatchConfigDto match, IMatchContentResolver content)
	{
		SessionValidationResult sessionValidationResult = new SessionValidationResult();
		if (match == null)
		{
			sessionValidationResult.Add("Match document is missing.");
			return sessionValidationResult;
		}
		RequireSchema(match.schemaVersion, 1, "match", sessionValidationResult);
		bool flag = RequireCanonicalIdentifier(match.matchId, "match.matchId", sessionValidationResult);
		RequireText(match.ruleset, "match.ruleset", sessionValidationResult);
		PlayerSlotDto[] array = match.players ?? Array.Empty<PlayerSlotDto>();
		if (array.Length != 2)
		{
			sessionValidationResult.Add("match.players must contain exactly two local player slots.");
		}
		HashSet<int> hashSet = new HashSet<int>();
		Dictionary<int, FleetEntrySets> dictionary = new Dictionary<int, FleetEntrySets>();
		for (int i = 0; i < array.Length; i++)
		{
			PlayerSlotDto playerSlotDto = array[i];
			string text = $"match.players[{i}]";
			if (playerSlotDto == null)
			{
				sessionValidationResult.Add(text + " is missing.");
				continue;
			}
			RequireSchema(playerSlotDto.schemaVersion, 1, text, sessionValidationResult);
			if (playerSlotDto.playerId != 1 && playerSlotDto.playerId != 2)
			{
				sessionValidationResult.Add(text + ".playerId must be Player 1 or Player 2.");
			}
			else if (!hashSet.Add(playerSlotDto.playerId))
			{
				sessionValidationResult.Add(text + ".playerId duplicates another player slot.");
			}
			RequireText(playerSlotDto.displayName, text + ".displayName", sessionValidationResult);
			if (!Enum.IsDefined(typeof(PlayerControlKind), playerSlotDto.controlKind))
			{
				sessionValidationResult.Add(text + ".controlKind is not supported.");
			}
			if (!Enum.IsDefined(typeof(BattleAiDifficulty), playerSlotDto.aiDifficulty))
			{
				sessionValidationResult.Add(text + ".aiDifficulty is not supported.");
			}
			ValidateFleetInto(playerSlotDto.fleet, text + ".fleet", sessionValidationResult);
			if ((playerSlotDto.playerId == 1 || playerSlotDto.playerId == 2) && playerSlotDto.fleet != null && !dictionary.ContainsKey(playerSlotDto.playerId))
			{
				dictionary[playerSlotDto.playerId] = BuildFleetEntrySets(playerSlotDto.fleet);
			}
		}
		if (!hashSet.Contains(1) || !hashSet.Contains(2))
		{
			sessionValidationResult.Add("match.players must identify Player 1 and Player 2 exactly once.");
		}
		if (match.firstPlayerId != 1 && match.firstPlayerId != 2)
		{
			sessionValidationResult.Add("match.firstPlayerId must be Player 1 or Player 2.");
		}
		if (match.escortPlayerId != 1 && match.escortPlayerId != 2)
		{
			sessionValidationResult.Add("match.escortPlayerId must be Player 1 or Player 2.");
		}
		ValidateMapInto(match.map, "match.map", sessionValidationResult);
		HashSet<string> hashSet2 = new HashSet<string>(StringComparer.Ordinal);
		HashSet<DeploymentEntryKey> hashSet3 = new HashSet<DeploymentEntryKey>();
		UnitDeploymentDto[] array2 = match.deployments ?? Array.Empty<UnitDeploymentDto>();
		for (int j = 0; j < array2.Length; j++)
		{
			UnitDeploymentDto unitDeploymentDto = array2[j];
			string text2 = $"match.deployments[{j}]";
			if (unitDeploymentDto == null)
			{
				sessionValidationResult.Add(text2 + " is missing.");
				continue;
			}
			RequireSchema(unitDeploymentDto.schemaVersion, 1, text2, sessionValidationResult);
			RuntimeUnitId runtimeUnitId = default(RuntimeUnitId);
			if (!RuntimeUnitId.TryParse(unitDeploymentDto.runtimeUnitId, out runtimeUnitId))
			{
				sessionValidationResult.Add(text2 + ".runtimeUnitId is required.");
			}
			else
			{
				if (!string.Equals(unitDeploymentDto.runtimeUnitId, runtimeUnitId.Value, StringComparison.Ordinal))
				{
					sessionValidationResult.Add(text2 + ".runtimeUnitId must not contain leading or trailing whitespace.");
				}
				if (!hashSet2.Add(runtimeUnitId.Value))
				{
					sessionValidationResult.Add(text2 + ".runtimeUnitId duplicates another unit.");
				}
			}
			bool flag2 = RequireCanonicalIdentifier(unitDeploymentDto.fleetEntryInstanceId, text2 + ".fleetEntryInstanceId", sessionValidationResult);
			bool flag3 = (unitDeploymentDto.ownerPlayerId == 1 || unitDeploymentDto.ownerPlayerId == 2) && hashSet.Contains(unitDeploymentDto.ownerPlayerId);
			if (!flag3)
			{
				sessionValidationResult.Add(text2 + ".ownerPlayerId must identify Player 1 or Player 2 in this match.");
			}
			bool flag4 = unitDeploymentDto.unitKind == RuntimeUnitKind.Ship || unitDeploymentDto.unitKind == RuntimeUnitKind.Squadron;
			if (!flag4)
			{
				sessionValidationResult.Add(text2 + ".unitKind is not supported.");
			}
			if (!IsFinite(unitDeploymentDto.positionX))
			{
				sessionValidationResult.Add(text2 + ".positionX must be finite.");
			}
			if (!IsFinite(unitDeploymentDto.positionY))
			{
				sessionValidationResult.Add(text2 + ".positionY must be finite.");
			}
			else if (match.map != null && IsFinite(match.map.playPlaneY) && Math.Abs(unitDeploymentDto.positionY - match.map.playPlaneY) > 0.0005f)
			{
				sessionValidationResult.Add(text2 + ".positionY must equal match.map." + $"playPlaneY ({match.map.playPlaneY}).");
			}
			if (!IsFinite(unitDeploymentDto.positionZ))
			{
				sessionValidationResult.Add(text2 + ".positionZ must be finite.");
			}
			if (!IsFinite(unitDeploymentDto.yawDegrees))
			{
				sessionValidationResult.Add(text2 + ".yawDegrees must be finite.");
			}
			if (unitDeploymentDto.unitKind == RuntimeUnitKind.Ship && unitDeploymentDto.startingSpeed < 0)
			{
				sessionValidationResult.Add(text2 + ".startingSpeed cannot be negative.");
			}
			else if (unitDeploymentDto.unitKind == RuntimeUnitKind.Squadron && unitDeploymentDto.startingSpeed != 0)
			{
				sessionValidationResult.Add(text2 + ".startingSpeed must be 0 for a squadron.");
			}
			FleetEntrySets value = null;
			bool flag5 = flag3 && flag4 && flag2 && dictionary.TryGetValue(unitDeploymentDto.ownerPlayerId, out value) && value.Contains(unitDeploymentDto.unitKind, unitDeploymentDto.fleetEntryInstanceId);
			if (flag3 && flag4 && flag2 && !flag5)
			{
				sessionValidationResult.Add(text2 + ".fleetEntryInstanceId does not identify a matching unit in that player's fleet.");
			}
			if (flag5)
			{
				DeploymentEntryKey item = new DeploymentEntryKey(unitDeploymentDto.ownerPlayerId, unitDeploymentDto.unitKind, unitDeploymentDto.fleetEntryInstanceId);
				if (!hashSet3.Add(item))
				{
					sessionValidationResult.Add(text2 + " deploys the same fleet entry more than once.");
				}
			}
			if (flag && flag3 && flag4 && flag2 && runtimeUnitId.IsValid)
			{
				RuntimeUnitId runtimeUnitId2 = RuntimeUnitId.ForMatchEntry(match.matchId, unitDeploymentDto.ownerPlayerId, unitDeploymentDto.unitKind, unitDeploymentDto.fleetEntryInstanceId);
				if (runtimeUnitId != runtimeUnitId2)
				{
					sessionValidationResult.Add(text2 + ".runtimeUnitId must equal the deterministic ID '" + runtimeUnitId2.Value + "'.");
				}
			}
			if (content != null && flag5)
			{
				value.TryGetDefinitionId(unitDeploymentDto.unitKind, unitDeploymentDto.fleetEntryInstanceId, out var definitionId);
				if (!((unitDeploymentDto.unitKind == RuntimeUnitKind.Ship) ? content.TryResolveShip(definitionId, out var rules) : content.TryResolveSquadron(definitionId, out rules)))
				{
					sessionValidationResult.Add(text2 + " references unresolved " + unitDeploymentDto.unitKind.ToString().ToLowerInvariant() + " definition '" + definitionId + "'.");
				}
				else
				{
					ValidateResolvedDeployment(match.map, unitDeploymentDto, text2, rules, !string.Equals(match.ruleset, FleetRuleset.Sandbox.Id, StringComparison.Ordinal), sessionValidationResult);
				}
			}
		}
		foreach (KeyValuePair<int, FleetEntrySets> item3 in dictionary)
		{
			foreach (FleetEntryReference item4 in item3.Value.Entries())
			{
				DeploymentEntryKey item2 = new DeploymentEntryKey(item3.Key, item4.Kind, item4.InstanceId);
				if (!hashSet3.Contains(item2))
				{
					sessionValidationResult.Add("match.deployments is missing Player " + $"{item3.Key} " + item4.Kind.ToString().ToLowerInvariant() + " fleet entry '" + item4.InstanceId + "'.");
				}
			}
		}
		return sessionValidationResult;
	}

	private static void ValidateResolvedDeployment(MapConfigDto map, UnitDeploymentDto deployment, string path, MatchUnitValidationRules rules, bool enforceSetupRules, SessionValidationResult result)
	{
		if (rules.MaximumSpeed < 0)
		{
			result.Add(path + " resolved invalid negative maximum speed.");
		}
		else if (deployment.unitKind == RuntimeUnitKind.Ship && deployment.startingSpeed > rules.MaximumSpeed)
		{
			result.Add(path + $".startingSpeed {deployment.startingSpeed} " + "exceeds the referenced ship's chart maximum " + $"{rules.MaximumSpeed}.");
		}
		if (!IsFinite(rules.FootprintWidth) || !IsFinite(rules.FootprintLength) || rules.FootprintWidth <= 0f || rules.FootprintLength <= 0f)
		{
			result.Add(path + " resolved an invalid gameplay footprint.");
		}
		else if (MapDimensionsAreUsable(map) && IsFinite(deployment.positionX) && IsFinite(deployment.positionZ) && IsFinite(deployment.yawDegrees))
		{
			if (!ContainsOrientedFootprint(deployment.positionX, deployment.positionZ, deployment.yawDegrees, rules.FootprintWidth, rules.FootprintLength, map.widthWorldUnits, map.lengthWorldUnits))
			{
				result.Add(path + " gameplay footprint extends outside match.map '" + map.mapId + "'.");
			}
			else if (enforceSetupRules && deployment.unitKind == RuntimeUnitKind.Ship && rules.IsHugeShip)
			{
				ValidateHugeShipSetup(map, deployment, path, rules, result);
			}
		}
	}

	private static void ValidateHugeShipSetup(MapConfigDto map, UnitDeploymentDto deployment, string path, MatchUnitValidationRules rules, SessionValidationResult result)
	{
		DeploymentZoneDto[] zones = map?.deploymentZones ?? Array.Empty<DeploymentZoneDto>();
		if (!TryGetSetupBounds(zones, out var minimumX, out var maximumX, out var minimumZ, out var maximumZ))
		{
			result.Add(path + " cannot validate huge-ship setup without deployment zones.");
			return;
		}
		GetOrientedExtents(deployment.yawDegrees, rules.FootprintWidth, rules.FootprintLength, out var extentX, out var extentZ);
		if ((double)deployment.positionX - extentX < minimumX - 0.0005000000237487257 || (double)deployment.positionX + extentX > maximumX + 0.0005000000237487257 || (double)deployment.positionZ - extentZ < minimumZ - 0.0005000000237487257 || (double)deployment.positionZ + extentZ > maximumZ + 0.0005000000237487257)
		{
			result.Add(path + " huge-ship footprint must be fully inside the setup area.");
		}
		DeploymentZoneDto deploymentZoneDto = FindDeploymentZone(zones, deployment.ownerPlayerId);
		DeploymentZoneDto deploymentZoneDto2 = FindDeploymentZone(zones, (deployment.ownerPlayerId != 1) ? 1 : 2);
		if (deploymentZoneDto == null || deploymentZoneDto2 == null)
		{
			result.Add(path + " cannot resolve the huge ship's player edge.");
			return;
		}
		double num = (double)(deploymentZoneDto.minimumX + deploymentZoneDto.maximumX) * 0.5;
		double num2 = (double)(deploymentZoneDto.minimumZ + deploymentZoneDto.maximumZ) * 0.5;
		double num3 = (double)(deploymentZoneDto2.minimumX + deploymentZoneDto2.maximumX) * 0.5;
		double num4 = (double)(deploymentZoneDto2.minimumZ + deploymentZoneDto2.maximumZ) * 0.5;
		double value = num - num3;
		double value2 = num2 - num4;
		if (!((Math.Abs(value) > Math.Abs(value2)) ? (Math.Abs((double)deployment.positionX + (double)Math.Sign(value) * extentX - (double)((float)Math.Sign(value) * map.widthWorldUnits) * 0.5) <= 0.0005000000237487257) : (Math.Abs((double)deployment.positionZ + (double)Math.Sign(value2) * extentZ - (double)((float)Math.Sign(value2) * map.lengthWorldUnits) * 0.5) <= 0.0005000000237487257)))
		{
			result.Add(path + " huge-ship footprint must touch its player's play-area edge.");
		}
	}

	private static bool TryGetSetupBounds(DeploymentZoneDto[] zones, out double minimumX, out double maximumX, out double minimumZ, out double maximumZ)
	{
		minimumX = double.PositiveInfinity;
		maximumX = double.NegativeInfinity;
		minimumZ = double.PositiveInfinity;
		maximumZ = double.NegativeInfinity;
		bool flag = false;
		DeploymentZoneDto[] array = zones ?? Array.Empty<DeploymentZoneDto>();
		foreach (DeploymentZoneDto deploymentZoneDto in array)
		{
			if (deploymentZoneDto != null)
			{
				flag = true;
				minimumX = Math.Min(minimumX, deploymentZoneDto.minimumX);
				maximumX = Math.Max(maximumX, deploymentZoneDto.maximumX);
				minimumZ = Math.Min(minimumZ, deploymentZoneDto.minimumZ);
				maximumZ = Math.Max(maximumZ, deploymentZoneDto.maximumZ);
			}
		}
		if (flag && minimumX < maximumX)
		{
			return minimumZ < maximumZ;
		}
		return false;
	}

	private static DeploymentZoneDto FindDeploymentZone(DeploymentZoneDto[] zones, int playerId)
	{
		DeploymentZoneDto[] array = zones ?? Array.Empty<DeploymentZoneDto>();
		foreach (DeploymentZoneDto deploymentZoneDto in array)
		{
			if (deploymentZoneDto != null && deploymentZoneDto.playerId == playerId)
			{
				return deploymentZoneDto;
			}
		}
		return null;
	}

	private static void GetOrientedExtents(float yawDegrees, float footprintWidth, float footprintLength, out double extentX, out double extentZ)
	{
		double num = (double)yawDegrees * (Math.PI / 180.0);
		double num2 = Math.Abs(Math.Cos(num));
		double num3 = Math.Abs(Math.Sin(num));
		double num4 = (double)footprintWidth * 0.5;
		double num5 = (double)footprintLength * 0.5;
		extentX = num2 * num4 + num3 * num5;
		extentZ = num3 * num4 + num2 * num5;
	}

	private static bool ContainsOrientedFootprint(float positionX, float positionZ, float yawDegrees, float footprintWidth, float footprintLength, float mapWidth, float mapLength)
	{
		GetOrientedExtents(yawDegrees, footprintWidth, footprintLength, out var extentX, out var extentZ);
		if ((double)Math.Abs(positionX) + extentX <= (double)mapWidth * 0.5 + 0.0005000000237487257)
		{
			return (double)Math.Abs(positionZ) + extentZ <= (double)mapLength * 0.5 + 0.0005000000237487257;
		}
		return false;
	}

	private static bool MapDimensionsAreUsable(MapConfigDto map)
	{
		if (map != null && IsFinite(map.widthWorldUnits) && IsFinite(map.lengthWorldUnits) && map.widthWorldUnits > 0f)
		{
			return map.lengthWorldUnits > 0f;
		}
		return false;
	}

	private static void ValidateFleetInto(FleetListDto fleet, string path, SessionValidationResult result)
	{
		if (fleet == null)
		{
			result.Add(path + " document is missing.");
			return;
		}
		RequireSchema(fleet.schemaVersion, 1, path, result);
		RequireText(fleet.fleetId, path + ".fleetId", result);
		RequireText(fleet.displayName, path + ".displayName", result);
		RequireText(fleet.faction, path + ".faction", result);
		if (fleet.pointLimit <= 0)
		{
			result.Add(path + ".pointLimit must be greater than zero.");
		}
		HashSet<string> instanceIds = new HashSet<string>(StringComparer.Ordinal);
		ShipBuildDto[] array = fleet.ships ?? Array.Empty<ShipBuildDto>();
		for (int i = 0; i < array.Length; i++)
		{
			ShipBuildDto shipBuildDto = array[i];
			string text = $"{path}.ships[{i}]";
			if (shipBuildDto == null)
			{
				result.Add(text + " is missing.");
				continue;
			}
			RequireSchema(shipBuildDto.schemaVersion, 1, text, result);
			ValidateInstanceId(shipBuildDto.instanceId, text + ".instanceId", instanceIds, result);
			RequireText(shipBuildDto.shipDefinitionId, text + ".shipDefinitionId", result);
			string[] array2 = shipBuildDto.upgradeIds ?? Array.Empty<string>();
			for (int j = 0; j < array2.Length; j++)
			{
				RequireText(array2[j], $"{text}.upgradeIds[{j}]", result);
			}
		}
		SquadronBuildDto[] array3 = fleet.squadrons ?? Array.Empty<SquadronBuildDto>();
		for (int k = 0; k < array3.Length; k++)
		{
			SquadronBuildDto squadronBuildDto = array3[k];
			string text2 = $"{path}.squadrons[{k}]";
			if (squadronBuildDto == null)
			{
				result.Add(text2 + " is missing.");
				continue;
			}
			RequireSchema(squadronBuildDto.schemaVersion, 1, text2, result);
			ValidateInstanceId(squadronBuildDto.instanceId, text2 + ".instanceId", instanceIds, result);
			RequireText(squadronBuildDto.squadronDefinitionId, text2 + ".squadronDefinitionId", result);
		}
		string[] array4 = fleet.objectiveIds ?? Array.Empty<string>();
		for (int l = 0; l < array4.Length; l++)
		{
			RequireText(array4[l], $"{path}.objectiveIds[{l}]", result);
		}
	}

	private static void ValidateMapInto(MapConfigDto map, string path, SessionValidationResult result)
	{
		if (map == null)
		{
			result.Add(path + " document is missing.");
			return;
		}
		RequireSchema(map.schemaVersion, 1, path, result);
		RequireText(map.mapId, path + ".mapId", result);
		RequireText(map.displayName, path + ".displayName", result);
		if (!MapScenarioCatalog.IsSupported(map.scenarioId))
		{
			result.Add(path + ".scenarioId is not a supported game type.");
		}
		if (!IsFinite(map.widthWorldUnits) || map.widthWorldUnits <= 0f)
		{
			result.Add(path + ".widthWorldUnits must be positive.");
		}
		if (!IsFinite(map.lengthWorldUnits) || map.lengthWorldUnits <= 0f)
		{
			result.Add(path + ".lengthWorldUnits must be positive.");
		}
		if (!IsFinite(map.playPlaneY))
		{
			result.Add(path + ".playPlaneY must be finite.");
		}
		HashSet<int> hashSet = new HashSet<int>();
		DeploymentZoneDto[] array = map.deploymentZones ?? Array.Empty<DeploymentZoneDto>();
		for (int i = 0; i < array.Length; i++)
		{
			DeploymentZoneDto deploymentZoneDto = array[i];
			string text = $"{path}.deploymentZones[{i}]";
			if (deploymentZoneDto == null)
			{
				result.Add(text + " is missing.");
				continue;
			}
			if (deploymentZoneDto.playerId != 1 && deploymentZoneDto.playerId != 2)
			{
				result.Add(text + ".playerId must be Player 1 or Player 2.");
			}
			else if (!hashSet.Add(deploymentZoneDto.playerId))
			{
				result.Add(text + ".playerId duplicates another zone.");
			}
			if (!IsFinite(deploymentZoneDto.minimumX) || !IsFinite(deploymentZoneDto.maximumX) || deploymentZoneDto.minimumX >= deploymentZoneDto.maximumX)
			{
				result.Add(text + " must have minimumX below maximumX.");
			}
			if (!IsFinite(deploymentZoneDto.minimumZ) || !IsFinite(deploymentZoneDto.maximumZ) || deploymentZoneDto.minimumZ >= deploymentZoneDto.maximumZ)
			{
				result.Add(text + " must have minimumZ below maximumZ.");
			}
		}
		HashSet<string> hashSet2 = new HashSet<string>(StringComparer.Ordinal);
		MapPlacementDto[] array2 = map.placements ?? Array.Empty<MapPlacementDto>();
		for (int j = 0; j < array2.Length; j++)
		{
			MapPlacementDto mapPlacementDto = array2[j];
			string text2 = $"{path}.placements[{j}]";
			if (mapPlacementDto == null)
			{
				result.Add(text2 + " is missing.");
				continue;
			}
			if (RequireCanonicalIdentifier(mapPlacementDto.placementId, text2 + ".placementId", result) && !hashSet2.Add(mapPlacementDto.placementId))
			{
				result.Add(text2 + ".placementId duplicates another placement.");
			}
			RequireText(mapPlacementDto.contentId, text2 + ".contentId", result);
			RequireText(mapPlacementDto.displayName, text2 + ".displayName", result);
			if (!Enum.IsDefined(typeof(MapPlacementKind), mapPlacementDto.kind))
			{
				result.Add(text2 + ".kind is not supported.");
			}
			if (!IsFinite(mapPlacementDto.positionX) || !IsFinite(mapPlacementDto.positionZ) || !IsFinite(mapPlacementDto.yawDegrees))
			{
				result.Add(text2 + " position and rotation must be finite.");
			}
			bool requiresDimensions = mapPlacementDto.kind != MapPlacementKind.Obstacle || mapPlacementDto.useCustomSize || mapPlacementDto.shapeId == "custom-rectangle";
			if (requiresDimensions && (!IsFinite(mapPlacementDto.widthWorldUnits) || mapPlacementDto.widthWorldUnits <= 0f || !IsFinite(mapPlacementDto.lengthWorldUnits) || mapPlacementDto.lengthWorldUnits <= 0f))
			{
				result.Add(text2 + " footprint dimensions must be positive.");
				continue;
			}
			if (!IsFinite(mapPlacementDto.interactionRangeWorldUnits) || mapPlacementDto.interactionRangeWorldUnits < 0f)
			{
				result.Add(text2 + ".interactionRangeWorldUnits cannot be negative.");
			}
			else if (mapPlacementDto.kind == MapPlacementKind.Objective && mapPlacementDto.interactionRangeWorldUnits <= 0f)
			{
				result.Add(text2 + " objective range must be positive.");
			}
			if (MapDimensionsAreUsable(map) && IsFinite(mapPlacementDto.positionX) && IsFinite(mapPlacementDto.positionZ) && IsFinite(mapPlacementDto.yawDegrees))
			{
				try
				{
					bool contained = mapPlacementDto.kind == MapPlacementKind.Obstacle
						? PlayAreaGeometry.Contains(new OrientedRectangle(new Pose(Vector2.Zero), new Vector2(TabletopMeasurements.FromWorldUnits(map.widthWorldUnits), TabletopMeasurements.FromWorldUnits(map.lengthWorldUnits))), MapObstacleGeometry.Shape(mapPlacementDto))
						: ContainsOrientedFootprint(mapPlacementDto.positionX, mapPlacementDto.positionZ, mapPlacementDto.yawDegrees, mapPlacementDto.widthWorldUnits, mapPlacementDto.lengthWorldUnits, map.widthWorldUnits, map.lengthWorldUnits);
					if (!contained) result.Add(text2 + " footprint extends outside the play area.");
				}
				catch (ArgumentException error) { result.Add(text2 + " has invalid physical geometry: " + error.Message); }
			}
		}
	}

	private static void ValidateInstanceId(string value, string path, HashSet<string> instanceIds, SessionValidationResult result)
	{
		if (RequireCanonicalIdentifier(value, path, result) && !instanceIds.Add(value))
		{
			result.Add(path + " duplicates another fleet entry.");
		}
	}

	private static FleetEntrySets BuildFleetEntrySets(FleetListDto fleet)
	{
		FleetEntrySets fleetEntrySets = new FleetEntrySets();
		ShipBuildDto[] array = fleet.ships ?? Array.Empty<ShipBuildDto>();
		foreach (ShipBuildDto shipBuildDto in array)
		{
			if (shipBuildDto != null && IsCanonicalIdentifier(shipBuildDto.instanceId))
			{
				fleetEntrySets.Add(RuntimeUnitKind.Ship, shipBuildDto.instanceId, shipBuildDto.shipDefinitionId);
			}
		}
		SquadronBuildDto[] array2 = fleet.squadrons ?? Array.Empty<SquadronBuildDto>();
		foreach (SquadronBuildDto squadronBuildDto in array2)
		{
			if (squadronBuildDto != null && IsCanonicalIdentifier(squadronBuildDto.instanceId))
			{
				fleetEntrySets.Add(RuntimeUnitKind.Squadron, squadronBuildDto.instanceId, squadronBuildDto.squadronDefinitionId);
			}
		}
		return fleetEntrySets;
	}

	private static void RequireSchema(int actual, int expected, string path, SessionValidationResult result)
	{
		if (actual != expected)
		{
			result.Add($"{path}.schemaVersion is {actual}; expected {expected}.");
		}
	}

	private static bool RequireCanonicalIdentifier(string value, string path, SessionValidationResult result)
	{
		if (string.IsNullOrWhiteSpace(value))
		{
			result.Add(path + " is required.");
			return false;
		}
		if (!string.Equals(value, value.Trim(), StringComparison.Ordinal))
		{
			result.Add(path + " must not contain leading or trailing whitespace.");
			return false;
		}
		return true;
	}

	private static bool IsCanonicalIdentifier(string value)
	{
		if (!string.IsNullOrWhiteSpace(value))
		{
			return string.Equals(value, value.Trim(), StringComparison.Ordinal);
		}
		return false;
	}

	private static bool IsFinite(float value)
	{
		if (!float.IsNaN(value))
		{
			return !float.IsInfinity(value);
		}
		return false;
	}

	private static void RequireText(string value, string path, SessionValidationResult result)
	{
		if (string.IsNullOrWhiteSpace(value))
		{
			result.Add(path + " is required.");
		}
	}
}
