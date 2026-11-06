#nullable disable
using System.Collections.Generic;

namespace Armada.Fleet;

public sealed class FleetContentCatalogBuilder
{
	private readonly List<ShipContentDescriptor> ships = new List<ShipContentDescriptor>();

	private readonly List<SquadronContentDescriptor> squadrons = new List<SquadronContentDescriptor>();

	private readonly List<UpgradeContentDescriptor> upgrades = new List<UpgradeContentDescriptor>();

	private readonly List<ObjectiveContentDescriptor> objectives = new List<ObjectiveContentDescriptor>();

	private readonly List<FleetContentAlias> aliases = new List<FleetContentAlias>();

	public FleetContentCatalogBuilder Add(ShipContentDescriptor value)
	{
		ships.Add(value);
		return this;
	}

	public FleetContentCatalogBuilder Add(SquadronContentDescriptor value)
	{
		squadrons.Add(value);
		return this;
	}

	public FleetContentCatalogBuilder Add(UpgradeContentDescriptor value)
	{
		upgrades.Add(value);
		return this;
	}

	public FleetContentCatalogBuilder Add(ObjectiveContentDescriptor value)
	{
		objectives.Add(value);
		return this;
	}

	public FleetContentCatalogBuilder AddAlias(FleetContentKind kind, string aliasId, string canonicalId)
	{
		aliases.Add(new FleetContentAlias(kind, aliasId, canonicalId));
		return this;
	}

	public ImmutableFleetContentCatalog Build()
	{
		return new ImmutableFleetContentCatalog(ships, squadrons, upgrades, objectives, aliases);
	}
}
