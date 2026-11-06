#nullable disable
namespace Armada.Fleet;

public sealed class FleetImportNotice
{
	public string Kind { get; }

	public string Name { get; }

	public string Reason { get; }

	public FleetImportNotice(string kind, string name, string reason)
	{
		Kind = (string.IsNullOrWhiteSpace(kind) ? "CARD" : kind.Trim().ToUpperInvariant());
		Name = name?.Trim() ?? string.Empty;
		Reason = reason?.Trim() ?? string.Empty;
	}
}
