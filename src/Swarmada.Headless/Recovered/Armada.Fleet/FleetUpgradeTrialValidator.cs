#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using Armada.Session;

namespace Armada.Fleet;

internal static class FleetUpgradeTrialValidator
{
	public static bool CanAddUpgrade(FleetListDto fleet, string shipInstanceId, string upgradeId, FleetLegalityValidator validator, FleetLegalityReport baselineReport = null)
	{
		if (fleet == null || validator == null || string.IsNullOrWhiteSpace(shipInstanceId) || string.IsNullOrWhiteSpace(upgradeId))
		{
			return false;
		}
		FleetDraft fleetDraft = new FleetDraft(fleet);
		fleetDraft.Document.objectiveIds = fleet.objectiveIds?.ToArray() ?? Array.Empty<string>();
		ShipBuildDto shipBuildDto = fleetDraft.FindShip(shipInstanceId);
		if (shipBuildDto == null || ContainsId(shipBuildDto.upgradeIds, upgradeId) || !fleetDraft.ToggleUpgrade(shipInstanceId, upgradeId))
		{
			return false;
		}
		if (baselineReport == null)
		{
			baselineReport = validator.Validate(fleet);
		}
		FleetLegalityReport trial = validator.Validate(fleetDraft.Document);
		return IntroducesNoNewErrors(baselineReport, trial);
	}

	public static bool TryExplainAddUpgrade(FleetListDto fleet, string shipInstanceId, string upgradeId, FleetLegalityValidator validator, FleetLegalityReport baselineReport, out string reason)
	{
		reason = string.Empty;
		if (fleet == null || validator == null || string.IsNullOrWhiteSpace(shipInstanceId) || string.IsNullOrWhiteSpace(upgradeId))
		{
			reason = "Unavailable.";
			return false;
		}
		FleetDraft fleetDraft = new FleetDraft(fleet);
		fleetDraft.Document.objectiveIds = fleet.objectiveIds?.ToArray() ?? Array.Empty<string>();
		ShipBuildDto shipBuildDto = fleetDraft.FindShip(shipInstanceId);
		if (shipBuildDto == null)
		{
			reason = "No ship selected.";
			return false;
		}
		if (ContainsId(shipBuildDto.upgradeIds, upgradeId))
		{
			reason = "Already equipped.";
			return false;
		}
		if (!fleetDraft.ToggleUpgrade(shipInstanceId, upgradeId))
		{
			reason = "Unavailable.";
			return false;
		}
		if (baselineReport == null)
		{
			baselineReport = validator.Validate(fleet);
		}
		FleetLegalityReport trial = validator.Validate(fleetDraft.Document);
		if (IntroducesNoNewErrors(baselineReport, trial))
		{
			return true;
		}
		reason = FirstNewError(baselineReport, trial) ?? "Not legal on this ship.";
		return false;
	}

	private static string FirstNewError(FleetLegalityReport baseline, FleetLegalityReport trial)
	{
		Dictionary<string, int> dictionary = new Dictionary<string, int>(StringComparer.Ordinal);
		if (baseline != null)
		{
			foreach (FleetLegalityIssue issue in baseline.Issues)
			{
				if (issue.Severity == FleetLegalitySeverity.Error)
				{
					string key = Fingerprint(issue);
					dictionary.TryGetValue(key, out var value);
					dictionary[key] = value + 1;
				}
			}
		}
		if (trial == null)
		{
			return null;
		}
		foreach (FleetLegalityIssue issue2 in trial.Issues)
		{
			if (issue2.Severity == FleetLegalitySeverity.Error)
			{
				string key2 = Fingerprint(issue2);
				if (!dictionary.TryGetValue(key2, out var value2) || value2 <= 0)
				{
					return issue2.Message;
				}
				dictionary[key2] = value2 - 1;
			}
		}
		return null;
	}

	private static bool IntroducesNoNewErrors(FleetLegalityReport baseline, FleetLegalityReport trial)
	{
		Dictionary<string, int> dictionary = new Dictionary<string, int>(StringComparer.Ordinal);
		if (baseline != null)
		{
			foreach (FleetLegalityIssue issue in baseline.Issues)
			{
				if (issue.Severity == FleetLegalitySeverity.Error)
				{
					string key = Fingerprint(issue);
					dictionary.TryGetValue(key, out var value);
					dictionary[key] = value + 1;
				}
			}
		}
		if (trial == null)
		{
			return false;
		}
		foreach (FleetLegalityIssue issue2 in trial.Issues)
		{
			if (issue2.Severity == FleetLegalitySeverity.Error)
			{
				string key2 = Fingerprint(issue2);
				if (!dictionary.TryGetValue(key2, out var value2) || value2 <= 0)
				{
					return false;
				}
				dictionary[key2] = value2 - 1;
			}
		}
		return true;
	}

	private static string Fingerprint(FleetLegalityIssue issue)
	{
		return issue.Code + "\u001f" + issue.Path + "\u001f" + issue.Message;
	}

	private static bool ContainsId(IEnumerable<string> values, string candidate)
	{
		if (values == null)
		{
			return false;
		}
		foreach (string value in values)
		{
			if (string.Equals(value, candidate, StringComparison.OrdinalIgnoreCase))
			{
				return true;
			}
		}
		return false;
	}
}
