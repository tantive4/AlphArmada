#nullable disable
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Armada.Fleet;

public sealed class UpgradeSlotResolution
{
	public IReadOnlyDictionary<string, int> Capacity { get; }

	public IReadOnlyDictionary<string, int> Consumed { get; }

	public IReadOnlyList<FleetLegalityIssue> Issues { get; }

	public bool IsValid
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

	internal UpgradeSlotResolution(Dictionary<string, int> capacity, Dictionary<string, int> consumed, List<FleetLegalityIssue> issues)
	{
		Capacity = new ReadOnlyDictionary<string, int>(new Dictionary<string, int>(capacity, StringComparer.OrdinalIgnoreCase));
		Consumed = new ReadOnlyDictionary<string, int>(new Dictionary<string, int>(consumed, StringComparer.OrdinalIgnoreCase));
		Issues = new ReadOnlyCollection<FleetLegalityIssue>(new List<FleetLegalityIssue>(issues));
	}
}
