#nullable disable
namespace Armada.Fleet;

public sealed class FleetLegalityIssue
{
	public string Code { get; }

	public FleetLegalitySeverity Severity { get; }

	public string Path { get; }

	public string Message { get; }

	public FleetLegalityIssue(string code, FleetLegalitySeverity severity, string path, string message)
	{
		Code = FleetContentText.RequiredId(code, "code");
		Severity = severity;
		Path = path?.Trim() ?? string.Empty;
		Message = message?.Trim() ?? string.Empty;
	}

	public override string ToString()
	{
		return $"[{Severity}] {Code} {Path}: {Message}";
	}
}
