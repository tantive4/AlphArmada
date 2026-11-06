#nullable disable
using System;
using System.Collections.Generic;
using Armada.Session;

namespace Armada.Fleet;

public sealed class FleetDraft
{
	public FleetListDto Document { get; }

	public FleetDraft(FleetListDto document)
	{
		Document = Clone(document) ?? throw new ArgumentNullException("document");
		Normalize(Document);
	}

	public static FleetDraft Create(string displayName, string faction, int pointLimit = 400, string rulesMode = "official")
	{
		return new FleetDraft(FleetListDto.Create(displayName, faction, pointLimit, rulesMode));
	}

	public static bool HasRequiredIdentity(FleetListDto document)
	{
		if (document != null && !string.IsNullOrWhiteSpace(document.fleetId) && !string.IsNullOrWhiteSpace(document.displayName))
		{
			return !string.IsNullOrWhiteSpace(document.faction);
		}
		return false;
	}

	public static bool EnsureRequiredIdentity(FleetListDto document, string fallbackFaction, string fallbackDisplayName)
	{
		if (document == null)
		{
			return false;
		}
		if (string.IsNullOrWhiteSpace(document.fleetId))
		{
			document.fleetId = Guid.NewGuid().ToString("N");
		}
		else
		{
			document.fleetId = document.fleetId.Trim();
		}
		if (string.IsNullOrWhiteSpace(document.faction))
		{
			document.faction = fallbackFaction?.Trim() ?? string.Empty;
		}
		else
		{
			document.faction = document.faction.Trim();
		}
		if (string.IsNullOrWhiteSpace(document.displayName))
		{
			document.displayName = ((!string.IsNullOrWhiteSpace(fallbackDisplayName)) ? fallbackDisplayName.Trim() : (string.IsNullOrWhiteSpace(document.faction) ? string.Empty : ("New " + TitleCaseFaction(document.faction) + " Fleet")));
		}
		else
		{
			document.displayName = document.displayName.Trim();
		}
		Normalize(document);
		return HasRequiredIdentity(document);
	}

	public ShipBuildDto AddShip(string shipDefinitionId, IEnumerable<string> upgradeIds = null)
	{
		if (string.IsNullOrWhiteSpace(shipDefinitionId))
		{
			throw new ArgumentException("A ship definition ID is required.", "shipDefinitionId");
		}
		ShipBuildDto shipBuildDto = new ShipBuildDto
		{
			instanceId = NewInstanceId("ship"),
			shipDefinitionId = shipDefinitionId.Trim(),
			upgradeIds = CopyIds(upgradeIds)
		};
		Document.ships = Append(Document.ships, shipBuildDto);
		return shipBuildDto;
	}

	public SquadronBuildDto AddSquadron(string squadronDefinitionId)
	{
		if (string.IsNullOrWhiteSpace(squadronDefinitionId))
		{
			throw new ArgumentException("A squadron definition ID is required.", "squadronDefinitionId");
		}
		SquadronBuildDto squadronBuildDto = new SquadronBuildDto
		{
			instanceId = NewInstanceId("squadron"),
			squadronDefinitionId = squadronDefinitionId.Trim()
		};
		Document.squadrons = Append(Document.squadrons, squadronBuildDto);
		return squadronBuildDto;
	}

	public bool RemoveShip(string instanceId)
	{
		return RemoveByInstanceId(ref Document.ships, instanceId, (ShipBuildDto value) => value?.instanceId);
	}

	public bool RemoveSquadron(string instanceId)
	{
		return RemoveByInstanceId(ref Document.squadrons, instanceId, (SquadronBuildDto value) => value?.instanceId);
	}

	public bool SetSquadronLeader(string squadronInstanceId, string leaderUpgradeId)
	{
		if (string.IsNullOrWhiteSpace(squadronInstanceId))
		{
			return false;
		}
		SquadronBuildDto[] squadrons = Document.squadrons;
		foreach (SquadronBuildDto squadronBuildDto in squadrons)
		{
			if (squadronBuildDto != null && string.Equals(squadronBuildDto.instanceId, squadronInstanceId, StringComparison.Ordinal))
			{
				squadronBuildDto.leaderUpgradeId = leaderUpgradeId?.Trim() ?? string.Empty;
				return true;
			}
		}
		return false;
	}

	public bool ToggleObjective(string objectiveId)
	{
		Document.objectiveIds = Array.Empty<string>();
		return false;
	}

	public bool ToggleUpgrade(string shipInstanceId, string upgradeId)
	{
		ShipBuildDto shipBuildDto = FindShip(shipInstanceId);
		if (shipBuildDto == null || string.IsNullOrWhiteSpace(upgradeId))
		{
			return false;
		}
		List<string> list = new List<string>(shipBuildDto.upgradeIds ?? Array.Empty<string>());
		int num = list.FindIndex((string value) => string.Equals(value, upgradeId, StringComparison.OrdinalIgnoreCase));
		if (num >= 0)
		{
			list.RemoveAt(num);
		}
		else
		{
			list.Add(upgradeId.Trim());
		}
		shipBuildDto.upgradeIds = list.ToArray();
		return true;
	}

	public ShipBuildDto FindShip(string instanceId)
	{
		if (string.IsNullOrWhiteSpace(instanceId))
		{
			return null;
		}
		ShipBuildDto[] ships = Document.ships;
		foreach (ShipBuildDto shipBuildDto in ships)
		{
			if (shipBuildDto != null && string.Equals(shipBuildDto.instanceId, instanceId, StringComparison.Ordinal))
			{
				return shipBuildDto;
			}
		}
		return null;
	}

	public FleetListDto Snapshot()
	{
		return Clone(Document);
	}

	public static FleetListDto Clone(FleetListDto source)
	{
		if (source == null)
		{
			return null;
		}
		return new FleetListDto
		{
			schemaVersion = source.schemaVersion,
			fleetId = source.fleetId,
			displayName = source.displayName,
			faction = source.faction,
			pointLimit = source.pointLimit,
			rulesMode = source.rulesMode,
			ships = CloneShips(source.ships),
			squadrons = CloneSquadrons(source.squadrons),
			objectiveIds = Array.Empty<string>(),
			starterPresetId = source.starterPresetId,
			starterPresetRevision = source.starterPresetRevision,
			starterPresetFingerprint = source.starterPresetFingerprint,
			savedUtc = source.savedUtc
		};
	}

	public static FleetListDto CreateCopy(FleetListDto source)
	{
		FleetListDto fleetListDto = Clone(source) ?? throw new ArgumentNullException("source");
		fleetListDto.fleetId = Guid.NewGuid().ToString("N");
		fleetListDto.displayName = (string.IsNullOrWhiteSpace(source.displayName) ? "Fleet Copy" : (source.displayName.Trim() + " Copy"));
		fleetListDto.starterPresetId = string.Empty;
		fleetListDto.starterPresetRevision = 0;
		fleetListDto.starterPresetFingerprint = string.Empty;
		fleetListDto.savedUtc = string.Empty;
		ShipBuildDto[] ships = fleetListDto.ships;
		foreach (ShipBuildDto shipBuildDto in ships)
		{
			if (shipBuildDto != null)
			{
				shipBuildDto.instanceId = NewInstanceId("ship");
			}
		}
		SquadronBuildDto[] squadrons = fleetListDto.squadrons;
		foreach (SquadronBuildDto squadronBuildDto in squadrons)
		{
			if (squadronBuildDto != null)
			{
				squadronBuildDto.instanceId = NewInstanceId("squadron");
			}
		}
		return fleetListDto;
	}

	private static ShipBuildDto[] CloneShips(ShipBuildDto[] source)
	{
		if (source == null)
		{
			source = Array.Empty<ShipBuildDto>();
		}
		ShipBuildDto[] array = new ShipBuildDto[source.Length];
		for (int i = 0; i < source.Length; i++)
		{
			ShipBuildDto shipBuildDto = source[i];
			array[i] = ((shipBuildDto == null) ? null : new ShipBuildDto
			{
				schemaVersion = shipBuildDto.schemaVersion,
				instanceId = shipBuildDto.instanceId,
				shipDefinitionId = shipBuildDto.shipDefinitionId,
				upgradeIds = CopyIds(shipBuildDto.upgradeIds)
			});
		}
		return array;
	}

	private static SquadronBuildDto[] CloneSquadrons(SquadronBuildDto[] source)
	{
		if (source == null)
		{
			source = Array.Empty<SquadronBuildDto>();
		}
		SquadronBuildDto[] array = new SquadronBuildDto[source.Length];
		for (int i = 0; i < source.Length; i++)
		{
			SquadronBuildDto squadronBuildDto = source[i];
			array[i] = ((squadronBuildDto == null) ? null : new SquadronBuildDto
			{
				schemaVersion = squadronBuildDto.schemaVersion,
				instanceId = squadronBuildDto.instanceId,
				squadronDefinitionId = squadronBuildDto.squadronDefinitionId,
				leaderUpgradeId = squadronBuildDto.leaderUpgradeId
			});
		}
		return array;
	}

	private static void Normalize(FleetListDto document)
	{
		FleetListDto fleetListDto = document;
		if (fleetListDto.ships == null)
		{
			fleetListDto.ships = Array.Empty<ShipBuildDto>();
		}
		fleetListDto = document;
		if (fleetListDto.squadrons == null)
		{
			fleetListDto.squadrons = Array.Empty<SquadronBuildDto>();
		}
		document.rulesMode = FleetListDto.NormalizeRulesMode(document.rulesMode);
		document.objectiveIds = Array.Empty<string>();
		fleetListDto = document;
		if (fleetListDto.starterPresetId == null)
		{
			fleetListDto.starterPresetId = string.Empty;
		}
		fleetListDto = document;
		if (fleetListDto.starterPresetFingerprint == null)
		{
			fleetListDto.starterPresetFingerprint = string.Empty;
		}
		ShipBuildDto[] ships = document.ships;
		foreach (ShipBuildDto shipBuildDto in ships)
		{
			if (shipBuildDto != null)
			{
				ShipBuildDto shipBuildDto2 = shipBuildDto;
				if (shipBuildDto2.upgradeIds == null)
				{
					shipBuildDto2.upgradeIds = Array.Empty<string>();
				}
			}
		}
		SquadronBuildDto[] squadrons = document.squadrons;
		foreach (SquadronBuildDto squadronBuildDto in squadrons)
		{
			if (squadronBuildDto != null)
			{
				SquadronBuildDto squadronBuildDto2 = squadronBuildDto;
				if (squadronBuildDto2.leaderUpgradeId == null)
				{
					squadronBuildDto2.leaderUpgradeId = string.Empty;
				}
			}
		}
	}

	private static string[] CopyIds(IEnumerable<string> source)
	{
		if (source == null)
		{
			return Array.Empty<string>();
		}
		List<string> list = new List<string>();
		foreach (string item in source)
		{
			if (!string.IsNullOrWhiteSpace(item))
			{
				list.Add(item.Trim());
			}
		}
		return list.ToArray();
	}

	private static T[] Append<T>(T[] source, T value)
	{
		if (source == null)
		{
			source = Array.Empty<T>();
		}
		T[] array = new T[source.Length + 1];
		Array.Copy(source, array, source.Length);
		array[^1] = value;
		return array;
	}

	private static bool RemoveByInstanceId<T>(ref T[] source, string instanceId, Func<T, string> getId)
	{
		if (source == null || string.IsNullOrWhiteSpace(instanceId))
		{
			return false;
		}
		int num = Array.FindIndex(source, (T value) => string.Equals(getId(value), instanceId, StringComparison.Ordinal));
		if (num < 0)
		{
			return false;
		}
		T[] array = new T[source.Length - 1];
		if (num > 0)
		{
			Array.Copy(source, 0, array, 0, num);
		}
		if (num < source.Length - 1)
		{
			Array.Copy(source, num + 1, array, num, source.Length - num - 1);
		}
		source = array;
		return true;
	}

	private static string NewInstanceId(string prefix)
	{
		return prefix + "-" + Guid.NewGuid().ToString("N");
	}

	private static string TitleCaseFaction(string faction)
	{
		if (string.IsNullOrWhiteSpace(faction))
		{
			return string.Empty;
		}
		string text = faction.Trim();
		return char.ToUpperInvariant(text[0]) + text.Substring(1).ToLowerInvariant();
	}
}
