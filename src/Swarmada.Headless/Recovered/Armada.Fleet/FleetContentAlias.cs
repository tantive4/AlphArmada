#nullable disable
namespace Armada.Fleet;

public sealed class FleetContentAlias
{
	public FleetContentKind Kind { get; }

	public string AliasId { get; }

	public string CanonicalId { get; }

	public FleetContentAlias(FleetContentKind kind, string aliasId, string canonicalId)
	{
		Kind = kind;
		AliasId = FleetContentText.RequiredId(aliasId, "aliasId");
		CanonicalId = FleetContentText.RequiredId(canonicalId, "canonicalId");
	}
}
