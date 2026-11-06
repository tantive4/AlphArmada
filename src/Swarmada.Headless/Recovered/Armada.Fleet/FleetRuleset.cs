#nullable disable
using System;

namespace Armada.Fleet;

public sealed class FleetRuleset
{
	public static FleetRuleset StandardPlay400 { get; } = new FleetRuleset("standard-400", 400, 2, 1, requireStandardObjectives: false, requireStrictOfficialContent: false, enforcePointCapOnly: true);

	public static FleetRuleset OfficialStandard400 { get; } = new FleetRuleset("official-standard-400", 400, 2, 1, requireStandardObjectives: false, requireStrictOfficialContent: true);

	public static FleetRuleset Custom { get; } = new FleetRuleset("custom", 9999, 999, 0, requireStandardObjectives: false, requireStrictOfficialContent: false, enforcePointCapOnly: true, honorFleetPointLimit: true);

	public static FleetRuleset Standard2025 { get; } = new FleetRuleset("standard-2025", 400, 2, 1, requireStandardObjectives: true, requireStrictOfficialContent: true);

	public static FleetRuleset CustomUpgradeChecklist { get; } = new FleetRuleset("custom-upgrade-checklist", Custom.MaximumFleetPoints, 2, 1, requireStandardObjectives: true, requireStrictOfficialContent: true);

	public static FleetRuleset Sandbox { get; } = new FleetRuleset("fleet-sandbox", 9999, 999, 0, requireStandardObjectives: false, requireStrictOfficialContent: false, enforcePointCapOnly: true, honorFleetPointLimit: true);

	public string Id { get; }

	public int MaximumFleetPoints { get; }

	public int MaximumFlotillas { get; }

	public int RequiredCommanders { get; }

	public bool RequireStandardObjectives { get; }

	public bool RequireStrictOfficialContent { get; }

	public bool EnforcePointCapOnly { get; }

	public bool HonorFleetPointLimit { get; }

	public FleetRuleset(string id, int maximumFleetPoints, int maximumFlotillas, int requiredCommanders, bool requireStandardObjectives, bool requireStrictOfficialContent, bool enforcePointCapOnly = false, bool honorFleetPointLimit = false)
	{
		Id = FleetContentText.RequiredId(id, "id");
		MaximumFleetPoints = Math.Max(1, maximumFleetPoints);
		MaximumFlotillas = Math.Max(0, maximumFlotillas);
		RequiredCommanders = Math.Max(0, requiredCommanders);
		RequireStandardObjectives = requireStandardObjectives;
		RequireStrictOfficialContent = requireStrictOfficialContent;
		EnforcePointCapOnly = enforcePointCapOnly;
		HonorFleetPointLimit = honorFleetPointLimit;
	}

	public int SquadronPointLimit(int agreedFleetPoints)
	{
		return Math.Max(0, (agreedFleetPoints + 2) / 3);
	}

	public int DefenseTokenSquadronLimit(int agreedFleetPoints)
	{
		return Math.Max(0, agreedFleetPoints / 100);
	}
}
