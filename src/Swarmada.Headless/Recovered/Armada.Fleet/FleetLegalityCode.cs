#nullable disable
namespace Armada.Fleet;

public static class FleetLegalityCode
{
	public const string FleetMissing = "fleet.missing";

	public const string InvalidPointLimit = "fleet.point-limit.invalid";

	public const string PointLimitExceeded = "fleet.points.exceeded";

	public const string FactionMissing = "fleet.faction.missing";

	public const string MixedFaction = "fleet.faction.mixed";

	public const string DuplicateInstanceId = "fleet.instance-id.duplicate";

	public const string InstanceIdMissing = "fleet.instance-id.missing";

	public const string UnknownShip = "ship.unknown";

	public const string UnknownSquadron = "squadron.unknown";

	public const string UnknownUpgrade = "upgrade.unknown";

	public const string UnknownObjective = "objective.unknown";

	public const string SquadronPointsExceeded = "squadron.points.exceeded";

	public const string DefenseTokenSquadronLimitExceeded = "squadron.defense-token-limit.exceeded";

	public const string DefenseTokenMetadataDeferred = "squadron.defense-token-metadata.deferred";

	public const string FlotillaLimitExceeded = "ship.flotilla-limit.exceeded";

	public const string FlotillaMetadataDeferred = "ship.flotilla-metadata.deferred";

	public const string CommanderCount = "upgrade.commander.count";

	public const string CommanderOnFlotilla = "upgrade.commander.flotilla";

	public const string ObjectiveCount = "objective.count";

	public const string ObjectiveCategory = "objective.category";

	public const string ObjectiveNotStrictOfficial = "objective.not-strict-official";

	public const string UpgradeNotStrictOfficial = "upgrade.not-strict-official";

	public const string UpgradeFaction = "upgrade.faction";

	public const string UpgradeBoundShip = "upgrade.bound-ship";

	public const string UpgradeRequiredSize = "upgrade.required-size";

	public const string UpgradeRequiredTrait = "upgrade.required-trait";

	public const string UpgradeFlagshipOnly = "upgrade.flagship-only";

	public const string UpgradeDuplicate = "upgrade.duplicate";

	public const string UpgradeTitleLimit = "upgrade.title.limit";

	public const string UpgradeModificationLimit = "upgrade.modification.limit";

	public const string UpgradeSlotUnavailable = "upgrade.slot.unavailable";

	public const string UpgradeDisallowedSlot = "upgrade.slot.disallowed";

	public const string UpgradeSlotDisabled = "upgrade.slot.disabled";

	public const string MetadataDeferred = "metadata.deferred";

	public const string UniqueConflict = "fleet.unique.conflict";

	public const string GameplayCoverageDeferred = "gameplay.coverage.deferred";
}
