#nullable disable
using System;
using System.Collections.Generic;
using Armada.Session;

namespace Armada.Fleet;

public sealed class FleetLegalityValidator
{
	private readonly IFleetContentCatalog content;

	private readonly FleetRuleset ruleset;

	private readonly FleetValidationOptions options;

	public FleetLegalityValidator(IFleetContentCatalog content, FleetRuleset ruleset = null, FleetValidationOptions options = null)
	{
		this.content = content ?? throw new ArgumentNullException("content");
		this.ruleset = ruleset ?? FleetRuleset.Standard2025;
		this.options = options ?? FleetValidationOptions.BuilderDefault;
	}

	public FleetLegalityReport Validate(FleetListDto fleet)
	{
		if (ruleset.EnforcePointCapOnly)
		{
			return ValidateStandardPointCap(fleet);
		}
		FleetLegalityReportBuilder fleetLegalityReportBuilder = new FleetLegalityReportBuilder();
		if (fleet == null)
		{
			fleetLegalityReportBuilder.Error("fleet.missing", "fleet", "A fleet document is required.");
			return fleetLegalityReportBuilder.Build();
		}
		int num = ValidatePointLimit(fleet, fleetLegalityReportBuilder);
		string text = FleetContentText.NormalizeFaction(fleet.faction);
		if (text.Length == 0)
		{
			fleetLegalityReportBuilder.Error("fleet.faction.missing", "faction", "Choose a fleet faction.");
		}
		Dictionary<string, string> instanceIds = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
		Dictionary<string, string> uniqueKeys = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
		ValidateShips(fleet, text, instanceIds, uniqueKeys, fleetLegalityReportBuilder);
		ValidateSquadrons(fleet, text, instanceIds, uniqueKeys, fleetLegalityReportBuilder);
		ValidateObjectives(fleet, fleetLegalityReportBuilder);
		if (fleetLegalityReportBuilder.FlotillaCount > ruleset.MaximumFlotillas)
		{
			fleetLegalityReportBuilder.Error("ship.flotilla-limit.exceeded", "ships", "A fleet may contain at most " + $"{ruleset.MaximumFlotillas} flotillas; this fleet " + $"contains {fleetLegalityReportBuilder.FlotillaCount}.");
		}
		if (fleetLegalityReportBuilder.CommanderCount != ruleset.RequiredCommanders)
		{
			fleetLegalityReportBuilder.Error("upgrade.commander.count", "ships", "A fleet requires exactly " + $"{ruleset.RequiredCommanders} commander upgrade; " + $"this fleet contains {fleetLegalityReportBuilder.CommanderCount}.");
		}
		int num2 = ruleset.SquadronPointLimit(num);
		if (fleetLegalityReportBuilder.SquadronPoints > num2)
		{
			fleetLegalityReportBuilder.Error("squadron.points.exceeded", "squadrons", $"Squadrons cost {fleetLegalityReportBuilder.SquadronPoints} points; the " + $"{num}-point limit allows at most " + $"{num2}.");
		}
		int num3 = ruleset.DefenseTokenSquadronLimit(num);
		if (fleetLegalityReportBuilder.DefenseTokenSquadronCount > num3)
		{
			fleetLegalityReportBuilder.Error("squadron.defense-token-limit.exceeded", "squadrons", "The fleet contains " + $"{fleetLegalityReportBuilder.DefenseTokenSquadronCount} squadrons with " + "defense tokens; its agreed point limit permits " + $"{num3}.");
		}
		int num4 = fleetLegalityReportBuilder.ShipPoints + fleetLegalityReportBuilder.SquadronPoints + fleetLegalityReportBuilder.UpgradePoints;
		if (num4 > num)
		{
			fleetLegalityReportBuilder.Error("fleet.points.exceeded", "pointLimit", $"The fleet costs {num4} points and exceeds its " + $"{num}-point limit.");
		}
		return fleetLegalityReportBuilder.Build();
	}

	private FleetLegalityReport ValidateStandardPointCap(FleetListDto fleet)
	{
		FleetLegalityReportBuilder fleetLegalityReportBuilder = new FleetLegalityReportBuilder();
		if (fleet == null)
		{
			fleetLegalityReportBuilder.Error("fleet.missing", "fleet", "A fleet document is required.");
			return fleetLegalityReportBuilder.Build();
		}
		string fleetFaction = FleetContentText.NormalizeFaction(fleet.faction);
		Dictionary<string, string> uniqueKeys = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
		ShipBuildDto[] array = fleet.ships ?? Array.Empty<ShipBuildDto>();
		foreach (ShipBuildDto shipBuildDto in array)
		{
			if (shipBuildDto == null)
			{
				continue;
			}
			if (content.TryGetShip(shipBuildDto.shipDefinitionId, out var ship))
			{
				fleetLegalityReportBuilder.ShipPoints += ship.Points;
			}
			else
			{
				fleetLegalityReportBuilder.Error("ship.unknown", "ships", "Unknown ship definition '" + shipBuildDto.shipDefinitionId + "'.");
			}
			string[] array2 = shipBuildDto.upgradeIds ?? Array.Empty<string>();
			foreach (string text in array2)
			{
				if (content.TryGetUpgrade(text, out var upgrade))
				{
					fleetLegalityReportBuilder.UpgradePoints += upgrade.Points;
				}
				else
				{
					fleetLegalityReportBuilder.Error("upgrade.unknown", "ships.upgradeIds", "Unknown upgrade definition '" + text + "'.");
				}
			}
		}
		SquadronBuildDto[] array3 = fleet.squadrons ?? Array.Empty<SquadronBuildDto>();
		foreach (SquadronBuildDto squadronBuildDto in array3)
		{
			if (squadronBuildDto != null && content.TryGetSquadron(squadronBuildDto.squadronDefinitionId, out var squadron))
			{
				fleetLegalityReportBuilder.SquadronPoints += squadron.Points;
				ValidateSquadronLeader(squadronBuildDto, squadron, fleetFaction, "squadrons", uniqueKeys, fleetLegalityReportBuilder);
			}
			else if (squadronBuildDto != null)
			{
				fleetLegalityReportBuilder.Error("squadron.unknown", "squadrons", "Unknown squadron definition '" + squadronBuildDto.squadronDefinitionId + "'.");
			}
		}
		int num = fleetLegalityReportBuilder.ShipPoints + fleetLegalityReportBuilder.SquadronPoints + fleetLegalityReportBuilder.UpgradePoints;
		int num2 = ((ruleset.HonorFleetPointLimit && fleet.pointLimit > 0) ? fleet.pointLimit : ruleset.MaximumFleetPoints);
		if (num > num2)
		{
			fleetLegalityReportBuilder.Error("fleet.points.exceeded", "points", $"The fleet costs {num} points and exceeds the " + $"{num2}-point fleet limit.");
		}
		return fleetLegalityReportBuilder.Build();
	}

	private int ValidatePointLimit(FleetListDto fleet, FleetLegalityReportBuilder report)
	{
		if (fleet.pointLimit <= 0 || fleet.pointLimit > ruleset.MaximumFleetPoints)
		{
			report.Error("fleet.point-limit.invalid", "pointLimit", "The agreed fleet-point limit must be between 1 and " + $"{ruleset.MaximumFleetPoints}.");
		}
		if (fleet.pointLimit <= 0)
		{
			return ruleset.MaximumFleetPoints;
		}
		return Math.Min(fleet.pointLimit, ruleset.MaximumFleetPoints);
	}

	private void ValidateShips(FleetListDto fleet, string fleetFaction, Dictionary<string, string> instanceIds, Dictionary<string, string> uniqueKeys, FleetLegalityReportBuilder report)
	{
		ShipBuildDto[] array = fleet.ships ?? Array.Empty<ShipBuildDto>();
		for (int i = 0; i < array.Length; i++)
		{
			ShipBuildDto shipBuildDto = array[i];
			string text = $"ships[{i}]";
			if (shipBuildDto == null)
			{
				report.Error("ship.unknown", text, "The ship entry is null.");
				continue;
			}
			RegisterInstanceId(shipBuildDto.instanceId, text, instanceIds, report);
			if (!content.TryGetShip(shipBuildDto.shipDefinitionId, out var ship))
			{
				report.Error("ship.unknown", text + ".shipDefinitionId", "Unknown ship definition '" + shipBuildDto.shipDefinitionId + "'.");
				continue;
			}
			report.ShipPoints += ship.Points;
			ValidateFaction(fleetFaction, ship.Faction, text + ".shipDefinitionId", ship.DisplayName, report);
			if (ship.IsFlotilla == true)
			{
				report.FlotillaCount++;
			}
			else if (!ship.IsFlotilla.HasValue)
			{
				report.Deferred("ship.flotilla-metadata.deferred", text + ".shipDefinitionId", ship.DisplayName + " has no size/flotilla metadata; the flotilla cap cannot be fully checked.");
			}
			List<UpgradeContentDescriptor> list = ResolveUpgrades(shipBuildDto, ship, text, uniqueKeys, report);
			bool flag = false;
			int num = 0;
			int num2 = 0;
			HashSet<string> hashSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
			for (int j = 0; j < list.Count; j++)
			{
				UpgradeContentDescriptor upgradeContentDescriptor = list[j];
				if (upgradeContentDescriptor != null)
				{
					string path = $"{text}.upgradeIds[{j}]";
					if (!hashSet.Add(upgradeContentDescriptor.Id))
					{
						report.Error("upgrade.duplicate", path, upgradeContentDescriptor.DisplayName + " cannot be equipped more than once on the same ship.");
					}
					if (EqualsId(upgradeContentDescriptor.Slot, "commander"))
					{
						report.CommanderCount++;
						flag = true;
					}
					if (EqualsId(upgradeContentDescriptor.Slot, "title"))
					{
						num++;
					}
					if (upgradeContentDescriptor.IsModification)
					{
						num2++;
					}
				}
			}
			if (num > 1)
			{
				report.Error("upgrade.title.limit", text + ".upgradeIds", ship.DisplayName + " may equip at most one title.");
			}
			if (num2 > 1)
			{
				report.Error("upgrade.modification.limit", text + ".upgradeIds", ship.DisplayName + " may equip at most one Modification upgrade.");
			}
			for (int k = 0; k < list.Count; k++)
			{
				UpgradeContentDescriptor upgradeContentDescriptor2 = list[k];
				if (upgradeContentDescriptor2 != null)
				{
					ValidateUpgradeRestrictions(ship, upgradeContentDescriptor2, flag, $"{text}.upgradeIds[{k}]", report);
				}
			}
			if (flag)
			{
				if (ship.IsFlotilla == true)
				{
					report.Error("upgrade.commander.flotilla", text + ".upgradeIds", "A flotilla cannot be the fleet flagship.");
				}
				else if (!ship.IsFlotilla.HasValue)
				{
					report.Deferred("ship.flotilla-metadata.deferred", text + ".upgradeIds", "Commander-on-flotilla legality is deferred until this ship has size metadata.");
				}
			}
			foreach (FleetLegalityIssue issue in UpgradeSlotResolver.Resolve(ship, list, text).Issues)
			{
				report.Add(issue);
			}
		}
	}

	private List<UpgradeContentDescriptor> ResolveUpgrades(ShipBuildDto build, ShipContentDescriptor ship, string shipPath, Dictionary<string, string> uniqueKeys, FleetLegalityReportBuilder report)
	{
		string[] array = build.upgradeIds ?? Array.Empty<string>();
		List<UpgradeContentDescriptor> list = new List<UpgradeContentDescriptor>(array.Length);
		for (int i = 0; i < array.Length; i++)
		{
			string text = array[i];
			string path = $"{shipPath}.upgradeIds[{i}]";
			if (!content.TryGetUpgrade(text, out var upgrade))
			{
				report.Error("upgrade.unknown", path, "Unknown upgrade '" + text + "'.");
				list.Add(null);
				continue;
			}
			list.Add(upgrade);
			report.UpgradePoints += upgrade.Points;
			if (ruleset.RequireStrictOfficialContent && upgrade.OfficialDisposition != OfficialContentDisposition.StrictOfficial)
			{
				report.Error("upgrade.not-strict-official", path, upgrade.DisplayName + " is " + DescribeDisposition(upgrade.OfficialDisposition) + " and is unavailable in strict official fleets.");
			}
			if (!IsFactionLegal(upgrade.Factions, ship.Faction))
			{
				report.Error("upgrade.faction", path, upgrade.DisplayName + " is not legal for " + ship.Faction + " ships.");
			}
			if (upgrade.IsUnique)
			{
				RegisterUnique(upgrade.Id, upgrade.UniqueClasses, upgrade.DisplayName, path, uniqueKeys, report);
			}
			if (options.ReportGameplayCoverage && upgrade.GameplayCoverage != GameplayCoverage.Implemented)
			{
				report.Deferred("gameplay.coverage.deferred", path, upgrade.DisplayName + " is fleet-legal data, but its printed gameplay effect is not registered as fully implemented.");
			}
		}
		return list;
	}

	private void ValidateUpgradeRestrictions(ShipContentDescriptor ship, UpgradeContentDescriptor upgrade, bool isFlagship, string path, FleetLegalityReportBuilder report)
	{
		if (upgrade.BoundShipType.Length > 0)
		{
			if (ship.ShipType.Length == 0)
			{
				report.Deferred("metadata.deferred", path, upgrade.DisplayName + " is bound to " + upgrade.BoundShipType + ", but " + ship.DisplayName + " has no chassis metadata.");
			}
			else if (!EqualsId(upgrade.BoundShipType, ship.ShipType))
			{
				report.Error("upgrade.bound-ship", path, upgrade.DisplayName + " is restricted to " + upgrade.BoundShipType + ".");
			}
		}
		if (upgrade.RequiredSizes.Count > 0)
		{
			if (!ship.HasSizeMetadata)
			{
				report.Deferred("metadata.deferred", path, upgrade.DisplayName + "'s size restriction cannot be checked until this ship has size metadata.");
			}
			else if (!Contains(upgrade.RequiredSizes, ship.Size))
			{
				report.Error("upgrade.required-size", path, upgrade.DisplayName + " requires a " + string.Join(" or ", upgrade.RequiredSizes) + " ship.");
			}
		}
		if (upgrade.RequiredTraits.Count > 0)
		{
			if (!ship.HasTraitMetadata)
			{
				report.Deferred("metadata.deferred", path, upgrade.DisplayName + "'s trait restriction cannot be checked until this ship has trait metadata.");
			}
			else if (!AnyOverlap(upgrade.RequiredTraits, ship.Traits))
			{
				report.Error("upgrade.required-trait", path, upgrade.DisplayName + " requires one of these ship traits: " + string.Join(", ", upgrade.RequiredTraits) + ".");
			}
		}
		if (upgrade.FlagshipOnly && !isFlagship)
		{
			report.Error("upgrade.flagship-only", path, upgrade.DisplayName + " can only be equipped to the flagship carrying the commander.");
		}
		if (EqualsId(upgrade.Id, "minister-tua-officer") && (EqualsId(ship.Size, "medium") || EqualsId(ship.Size, "large")) && Contains(ship.UpgradeSlots, "defensive-retro"))
		{
			report.Error("upgrade.slot.disallowed", path, "Minister Tua cannot be equipped to a medium or large ship that already has a defensive-retro slot.");
		}
	}

	private void ValidateSquadrons(FleetListDto fleet, string fleetFaction, Dictionary<string, string> instanceIds, Dictionary<string, string> uniqueKeys, FleetLegalityReportBuilder report)
	{
		SquadronBuildDto[] array = fleet.squadrons ?? Array.Empty<SquadronBuildDto>();
		for (int i = 0; i < array.Length; i++)
		{
			SquadronBuildDto squadronBuildDto = array[i];
			string text = $"squadrons[{i}]";
			if (squadronBuildDto == null)
			{
				report.Error("squadron.unknown", text, "The squadron entry is null.");
				continue;
			}
			RegisterInstanceId(squadronBuildDto.instanceId, text, instanceIds, report);
			if (!content.TryGetSquadron(squadronBuildDto.squadronDefinitionId, out var squadron))
			{
				report.Error("squadron.unknown", text + ".squadronDefinitionId", "Unknown squadron definition '" + squadronBuildDto.squadronDefinitionId + "'.");
				continue;
			}
			report.SquadronPoints += squadron.Points;
			ValidateFaction(fleetFaction, squadron.Faction, text + ".squadronDefinitionId", squadron.DisplayName, report);
			if (squadron.IsUnique)
			{
				RegisterUnique(squadron.Id, squadron.UniqueClasses, squadron.DisplayName, text + ".squadronDefinitionId", uniqueKeys, report);
				if (!squadron.HasDefenseTokenMetadata)
				{
					report.Deferred("squadron.defense-token-metadata.deferred", text + ".squadronDefinitionId", squadron.DisplayName + " has incomplete defense token metadata; the defense-token squadron cap cannot be fully checked.");
				}
				else if (squadron.HasDefenseTokens)
				{
					report.DefenseTokenSquadronCount++;
				}
			}
			ValidateSquadronLeader(squadronBuildDto, squadron, fleetFaction, text, uniqueKeys, report);
			if (options.ReportGameplayCoverage && squadron.GameplayCoverage != GameplayCoverage.Implemented)
			{
				report.Deferred("gameplay.coverage.deferred", text + ".squadronDefinitionId", squadron.DisplayName + "'s complete printed ability is not registered as fully implemented.");
			}
		}
	}

	private void ValidateSquadronLeader(SquadronBuildDto build, SquadronContentDescriptor squadron, string fleetFaction, string path, Dictionary<string, string> uniqueKeys, FleetLegalityReportBuilder report)
    {
        if (!string.IsNullOrWhiteSpace(build.leaderUpgradeId))
            report.Error("upgrade.unknown", path + ".leaderUpgradeId", "Fighter-group cards have been removed from this engine.");
    }

	private static bool HasSquadronKeyword(SquadronContentDescriptor squadron, string keyword)
	{
		if (squadron?.Keywords == null)
		{
			return false;
		}
		foreach (string keyword2 in squadron.Keywords)
		{
			if (!string.IsNullOrWhiteSpace(keyword2) && keyword2.StartsWith(keyword, StringComparison.OrdinalIgnoreCase))
			{
				return true;
			}
		}
		return false;
	}

	private void ValidateObjectives(FleetListDto fleet, FleetLegalityReportBuilder report)
	{
		string[] array = fleet.objectiveIds ?? Array.Empty<string>();
		if (ruleset.RequireStandardObjectives && array.Length != 3)
		{
			report.Error("objective.count", "objectiveIds", "A standard fleet requires exactly one Assault, one Defense, and one Navigation objective.");
		}
		Dictionary<ObjectiveCategory, int> dictionary = new Dictionary<ObjectiveCategory, int>();
		for (int i = 0; i < array.Length; i++)
		{
			string path = $"objectiveIds[{i}]";
			if (!content.TryGetObjective(array[i], out var objective))
			{
				report.Error("objective.unknown", path, "Unknown objective '" + array[i] + "'.");
				continue;
			}
			dictionary.TryGetValue(objective.Category, out var value);
			dictionary[objective.Category] = value + 1;
			if (ruleset.RequireStrictOfficialContent && objective.OfficialDisposition != OfficialContentDisposition.StrictOfficial)
			{
				report.Error("objective.not-strict-official", path, objective.DisplayName + " is not available in strict official fleets.");
			}
			if (options.ReportGameplayCoverage && objective.GameplayCoverage != GameplayCoverage.Implemented)
			{
				report.Deferred("gameplay.coverage.deferred", path, objective.DisplayName + "'s setup and scoring rules are not registered as fully implemented.");
			}
		}
		if (ruleset.RequireStandardObjectives)
		{
			ValidateObjectiveCategory(ObjectiveCategory.Assault, dictionary, report);
			ValidateObjectiveCategory(ObjectiveCategory.Defense, dictionary, report);
			ValidateObjectiveCategory(ObjectiveCategory.Navigation, dictionary, report);
		}
	}

	private static void ValidateObjectiveCategory(ObjectiveCategory category, Dictionary<ObjectiveCategory, int> categoryCounts, FleetLegalityReportBuilder report)
	{
		categoryCounts.TryGetValue(category, out var value);
		if (value != 1)
		{
			report.Error("objective.category", "objectiveIds", $"A standard fleet requires exactly one {category} " + $"objective; this fleet contains {value}.");
		}
	}

	private static void RegisterInstanceId(string rawId, string path, Dictionary<string, string> instanceIds, FleetLegalityReportBuilder report)
	{
		string text = FleetContentText.Normalize(rawId);
		string value;
		if (text.Length == 0)
		{
			report.Error("fleet.instance-id.missing", path + ".instanceId", "Every fleet entry needs a stable instance ID.");
		}
		else if (instanceIds.TryGetValue(text, out value))
		{
			report.Error("fleet.instance-id.duplicate", path + ".instanceId", "Instance ID '" + text + "' is already used by " + value + ".");
		}
		else
		{
			instanceIds[text] = path;
		}
	}

	private static void RegisterUnique(string id, IReadOnlyList<string> uniqueClasses, string displayName, string path, Dictionary<string, string> uniqueKeys, FleetLegalityReportBuilder report)
	{
		List<string> list = new List<string> { "id:" + id };
		if (uniqueClasses != null)
		{
			foreach (string uniqueClass in uniqueClasses)
			{
				list.Add("class:" + uniqueClass);
			}
		}
		string value = null;
		foreach (string item in list)
		{
			if (uniqueKeys.TryGetValue(item, out value))
			{
				break;
			}
		}
		if (value != null)
		{
			report.Error("fleet.unique.conflict", path, displayName + " conflicts with unique content at " + value + ".");
		}
		foreach (string item2 in list)
		{
			if (!uniqueKeys.ContainsKey(item2))
			{
				uniqueKeys[item2] = path;
			}
		}
	}

	private static void ValidateFaction(string fleetFaction, string contentFaction, string path, string displayName, FleetLegalityReportBuilder report)
	{
		if (fleetFaction.Length != 0 && !string.Equals(fleetFaction, contentFaction, StringComparison.OrdinalIgnoreCase))
		{
			report.Error("fleet.faction.mixed", path, displayName + " belongs to " + contentFaction + ", not " + fleetFaction + ".");
		}
	}

	private static bool IsFactionLegal(IReadOnlyList<string> allowed, string faction)
	{
		if (allowed == null || allowed.Count == 0)
		{
			return true;
		}
		foreach (string item in allowed)
		{
			if (string.Equals(FleetContentText.NormalizeFaction(item), faction, StringComparison.OrdinalIgnoreCase))
			{
				return true;
			}
		}
		return false;
	}

	private static bool Contains(IReadOnlyList<string> values, string candidate)
	{
		if (values == null)
		{
			return false;
		}
		foreach (string value in values)
		{
			if (EqualsId(value, candidate))
			{
				return true;
			}
		}
		return false;
	}

	private static bool AnyOverlap(IReadOnlyList<string> left, IReadOnlyList<string> right)
	{
		if (left == null || right == null)
		{
			return false;
		}
		foreach (string item in left)
		{
			foreach (string item2 in right)
			{
				if (EqualsId(item, item2))
				{
					return true;
				}
			}
		}
		return false;
	}

	private static bool EqualsId(string left, string right)
	{
		return string.Equals(FleetContentText.Normalize(left), FleetContentText.Normalize(right), StringComparison.OrdinalIgnoreCase);
	}

	private static string DescribeDisposition(OfficialContentDisposition disposition)
	{
		if (disposition != OfficialContentDisposition.ProvenancePending)
		{
			return "excluded";
		}
		return "provenance-pending";
	}
}
