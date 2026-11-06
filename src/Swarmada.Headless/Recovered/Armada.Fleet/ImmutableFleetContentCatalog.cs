#nullable disable
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Armada.Fleet;

public sealed class ImmutableFleetContentCatalog : IFleetContentCatalog
{
	private readonly Dictionary<string, ShipContentDescriptor> ships;

	private readonly Dictionary<string, SquadronContentDescriptor> squadrons;

	private readonly Dictionary<string, UpgradeContentDescriptor> upgrades;

	private readonly Dictionary<string, ObjectiveContentDescriptor> objectives;

	public IReadOnlyCollection<ShipContentDescriptor> Ships { get; }

	public IReadOnlyCollection<SquadronContentDescriptor> Squadrons { get; }

	public IReadOnlyCollection<UpgradeContentDescriptor> Upgrades { get; }

	public IReadOnlyCollection<ObjectiveContentDescriptor> Objectives { get; }

	public IReadOnlyCollection<FleetContentAlias> Aliases { get; }

	public ImmutableFleetContentCatalog(IEnumerable<ShipContentDescriptor> ships = null, IEnumerable<SquadronContentDescriptor> squadrons = null, IEnumerable<UpgradeContentDescriptor> upgrades = null, IEnumerable<ObjectiveContentDescriptor> objectives = null, IEnumerable<FleetContentAlias> aliases = null)
	{
		this.ships = Index(ships, (ShipContentDescriptor value) => value.Id, "ship");
		this.squadrons = Index(squadrons, (SquadronContentDescriptor value) => value.Id, "squadron");
		this.upgrades = Index(upgrades, (UpgradeContentDescriptor value) => value.Id, "upgrade");
		this.objectives = Index(objectives, (ObjectiveContentDescriptor value) => value.Id, "objective");
		Ships = new ReadOnlyCollection<ShipContentDescriptor>(new List<ShipContentDescriptor>(this.ships.Values));
		Squadrons = new ReadOnlyCollection<SquadronContentDescriptor>(new List<SquadronContentDescriptor>(this.squadrons.Values));
		Upgrades = new ReadOnlyCollection<UpgradeContentDescriptor>(new List<UpgradeContentDescriptor>(this.upgrades.Values));
		Objectives = new ReadOnlyCollection<ObjectiveContentDescriptor>(new List<ObjectiveContentDescriptor>(this.objectives.Values));
		List<FleetContentAlias> list = ((aliases == null) ? new List<FleetContentAlias>() : new List<FleetContentAlias>(aliases));
		foreach (FleetContentAlias item in list)
		{
			ApplyAlias(item);
		}
		Aliases = new ReadOnlyCollection<FleetContentAlias>(list);
	}

	public bool TryGetShip(string id, out ShipContentDescriptor ship)
	{
		return ships.TryGetValue(FleetContentText.Normalize(id), out ship);
	}

	public bool TryGetSquadron(string id, out SquadronContentDescriptor squadron)
	{
		return squadrons.TryGetValue(FleetContentText.Normalize(id), out squadron);
	}

	public bool TryGetUpgrade(string id, out UpgradeContentDescriptor upgrade)
	{
		return upgrades.TryGetValue(FleetContentText.Normalize(id), out upgrade);
	}

	public bool TryGetObjective(string id, out ObjectiveContentDescriptor objective)
	{
		return objectives.TryGetValue(FleetContentText.Normalize(id), out objective);
	}

	private static Dictionary<string, T> Index<T>(IEnumerable<T> source, Func<T, string> id, string contentType) where T : class
	{
		Dictionary<string, T> dictionary = new Dictionary<string, T>(StringComparer.OrdinalIgnoreCase);
		if (source == null)
		{
			return dictionary;
		}
		foreach (T item in source)
		{
			if (item == null)
			{
				throw new ArgumentException("The " + contentType + " catalog contains a null entry.");
			}
			string text = FleetContentText.RequiredId(id(item), contentType + " id");
			if (!dictionary.TryAdd(text, item))
			{
				throw new ArgumentException("Duplicate " + contentType + " content id '" + text + "'.");
			}
		}
		return dictionary;
	}

	private void ApplyAlias(FleetContentAlias alias)
	{
		if (alias == null)
		{
			throw new ArgumentException("A catalog alias is null.");
		}
		switch (alias.Kind)
		{
		case FleetContentKind.Ship:
			ApplyAlias(ships, alias, "ship");
			break;
		case FleetContentKind.Squadron:
			ApplyAlias(squadrons, alias, "squadron");
			break;
		case FleetContentKind.Upgrade:
			ApplyAlias(upgrades, alias, "upgrade");
			break;
		case FleetContentKind.Objective:
			ApplyAlias(objectives, alias, "objective");
			break;
		default:
			throw new ArgumentException($"Unsupported alias kind '{alias.Kind}'.");
		}
	}

	private static void ApplyAlias<T>(Dictionary<string, T> index, FleetContentAlias alias, string contentType) where T : class
	{
		if (!index.TryGetValue(alias.CanonicalId, out var value))
		{
			throw new ArgumentException(contentType + " alias '" + alias.AliasId + "' points to unknown canonical ID '" + alias.CanonicalId + "'.");
		}
		if (index.TryGetValue(alias.AliasId, out var value2))
		{
			if (value2 != value)
			{
				throw new ArgumentException(contentType + " alias '" + alias.AliasId + "' conflicts with an existing content ID or alias.");
			}
		}
		else
		{
			index.Add(alias.AliasId, value);
		}
	}
}
