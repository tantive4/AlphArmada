#nullable disable
namespace Armada.Fleet;

public readonly struct OfficialUpgradePolicyAudit
{
	public int RawCount { get; }

	public int StrictCount { get; }

	public int PendingCount { get; }

	public int ExcludedCount { get; }

	public bool MatchesExpectedOfficialCatalog { get; }

	public OfficialUpgradePolicyAudit(int rawCount, int strictCount, int pendingCount, int excludedCount, bool matchesExpectedOfficialCatalog)
	{
		RawCount = rawCount;
		StrictCount = strictCount;
		PendingCount = pendingCount;
		ExcludedCount = excludedCount;
		MatchesExpectedOfficialCatalog = matchesExpectedOfficialCatalog;
	}
}
