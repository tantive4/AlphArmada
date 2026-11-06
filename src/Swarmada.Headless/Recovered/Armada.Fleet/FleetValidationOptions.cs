#nullable disable
namespace Armada.Fleet;

public sealed class FleetValidationOptions
{
	public static FleetValidationOptions BuilderDefault { get; } = new FleetValidationOptions(reportGameplayCoverage: true);

	public bool ReportGameplayCoverage { get; }

	public FleetValidationOptions(bool reportGameplayCoverage)
	{
		ReportGameplayCoverage = reportGameplayCoverage;
	}
}
