#nullable disable
using System;

namespace Armada.Session;

[Serializable]
public sealed class FleetListDto
{
	public const string OfficialRulesMode = "official";

	public const string CustomRulesMode = "custom";

	public int schemaVersion = 1;

	public string fleetId = string.Empty;

	public string displayName = string.Empty;

	public string faction = string.Empty;

	public int pointLimit = 400;

	public string rulesMode = string.Empty;

	public ShipBuildDto[] ships = Array.Empty<ShipBuildDto>();

	public SquadronBuildDto[] squadrons = Array.Empty<SquadronBuildDto>();

	public string[] objectiveIds = Array.Empty<string>();

	public string starterPresetId = string.Empty;

	public int starterPresetRevision;

	public string starterPresetFingerprint = string.Empty;

	public string savedUtc = string.Empty;

	public static FleetListDto Create(string displayName, string faction, int pointLimit = 400, string rulesMode = "official")
	{
		return new FleetListDto
		{
			fleetId = Guid.NewGuid().ToString("N"),
			displayName = (displayName ?? string.Empty),
			faction = (faction ?? string.Empty),
			pointLimit = pointLimit,
			rulesMode = NormalizeRulesMode(rulesMode)
		};
	}

	public static bool UsesOfficialRules(string value)
	{
		return string.Equals(value, "official", StringComparison.OrdinalIgnoreCase);
	}

	public static string NormalizeRulesMode(string value)
	{
		if (!UsesOfficialRules(value))
		{
			return "custom";
		}
		return "official";
	}
}
