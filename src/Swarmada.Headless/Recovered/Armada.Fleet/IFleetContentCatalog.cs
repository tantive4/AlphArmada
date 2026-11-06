#nullable disable
using System.Collections.Generic;

namespace Armada.Fleet;

public interface IFleetContentCatalog
{
	IReadOnlyCollection<ShipContentDescriptor> Ships { get; }

	IReadOnlyCollection<SquadronContentDescriptor> Squadrons { get; }

	IReadOnlyCollection<UpgradeContentDescriptor> Upgrades { get; }

	IReadOnlyCollection<ObjectiveContentDescriptor> Objectives { get; }

	IReadOnlyCollection<FleetContentAlias> Aliases { get; }

	bool TryGetShip(string id, out ShipContentDescriptor ship);

	bool TryGetSquadron(string id, out SquadronContentDescriptor squadron);

	bool TryGetUpgrade(string id, out UpgradeContentDescriptor upgrade);

	bool TryGetObjective(string id, out ObjectiveContentDescriptor objective);
}
