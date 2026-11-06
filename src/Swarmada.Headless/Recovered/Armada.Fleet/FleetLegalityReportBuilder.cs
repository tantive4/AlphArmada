#nullable disable
using System.Collections.Generic;

namespace Armada.Fleet;

internal sealed class FleetLegalityReportBuilder
{
	private readonly List<FleetLegalityIssue> issues = new List<FleetLegalityIssue>();

	public int ShipPoints;

	public int SquadronPoints;

	public int UpgradePoints;

	public int FlotillaCount;

	public int CommanderCount;

	public int DefenseTokenSquadronCount;

	public void Error(string code, string path, string message)
	{
		issues.Add(new FleetLegalityIssue(code, FleetLegalitySeverity.Error, path, message));
	}

	public void Warning(string code, string path, string message)
	{
		issues.Add(new FleetLegalityIssue(code, FleetLegalitySeverity.Warning, path, message));
	}

	public void Deferred(string code, string path, string message)
	{
		issues.Add(new FleetLegalityIssue(code, FleetLegalitySeverity.Deferred, path, message));
	}

	public void Add(FleetLegalityIssue issue)
	{
		if (issue != null)
		{
			issues.Add(issue);
		}
	}

	public FleetLegalityReport Build()
	{
		return new FleetLegalityReport(issues, ShipPoints, SquadronPoints, UpgradePoints, FlotillaCount, CommanderCount, DefenseTokenSquadronCount);
	}
}
