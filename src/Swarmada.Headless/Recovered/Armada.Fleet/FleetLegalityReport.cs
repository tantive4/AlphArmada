#nullable disable
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Armada.Fleet;

public sealed class FleetLegalityReport
{
	public IReadOnlyList<FleetLegalityIssue> Issues { get; }

	public int ShipPoints { get; }

	public int SquadronPoints { get; }

	public int UpgradePoints { get; }

	public int TotalPoints => ShipPoints + SquadronPoints + UpgradePoints;

	public int FlotillaCount { get; }

	public int CommanderCount { get; }

	public int DefenseTokenSquadronCount { get; }

	public bool IsLegal
	{
		get
		{
			foreach (FleetLegalityIssue issue in Issues)
			{
				if (issue.Severity == FleetLegalitySeverity.Error)
				{
					return false;
				}
			}
			return true;
		}
	}

	public bool HasDeferredChecks
	{
		get
		{
			foreach (FleetLegalityIssue issue in Issues)
			{
				if (issue.Severity == FleetLegalitySeverity.Deferred)
				{
					return true;
				}
			}
			return false;
		}
	}

	internal FleetLegalityReport(List<FleetLegalityIssue> issues, int shipPoints, int squadronPoints, int upgradePoints, int flotillaCount, int commanderCount, int defenseTokenSquadronCount)
	{
		Issues = new ReadOnlyCollection<FleetLegalityIssue>(new List<FleetLegalityIssue>(issues ?? new List<FleetLegalityIssue>()));
		ShipPoints = Math.Max(0, shipPoints);
		SquadronPoints = Math.Max(0, squadronPoints);
		UpgradePoints = Math.Max(0, upgradePoints);
		FlotillaCount = Math.Max(0, flotillaCount);
		CommanderCount = Math.Max(0, commanderCount);
		DefenseTokenSquadronCount = Math.Max(0, defenseTokenSquadronCount);
	}
}
