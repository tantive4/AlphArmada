#nullable disable
using System;
using System.Collections.Generic;
using Armada.Data;

namespace Armada.Fleet;

public static class FleetContentAdapter
{
	public static ShipContentDescriptor Describe(ShipDefinition definition, ShipFleetMetadata metadata = null)
	{
		if (definition == null)
		{
			throw new ArgumentNullException("definition");
		}
		return new ShipContentDescriptor(definition.ShipId, definition.DisplayName, FactionId(definition.Faction), definition.Points, definition.ShipType, definition.UpgradeSlots, metadata?.Size ?? string.Empty, metadata?.Traits, metadata != null, metadata?.IsFlotilla, metadata?.GameplayCoverage ?? GameplayCoverage.Unknown);
	}

	public static SquadronContentDescriptor Describe(SquadronDefinition definition, SquadronFleetMetadata metadata = null)
	{
		if (definition == null)
		{
			throw new ArgumentNullException("definition");
		}
		return new SquadronContentDescriptor(definition.SquadronId, definition.DisplayName, FactionId(definition.Faction), definition.Points, definition.IsUnique, definition.IsSingleCraft, metadata?.DefenseTokenCount ?? (definition.BraceTokens + definition.ScatterTokens), metadata?.UniqueClasses, metadata?.HasCompleteDefenseTokenMetadata ?? false, metadata?.GameplayCoverage ?? GameplayCoverage.Unknown, definition.Keywords);
	}

	public static ImmutableFleetContentCatalog Build(IEnumerable<ShipDefinition> ships, IEnumerable<SquadronDefinition> squadrons, IEnumerable<ObjectiveContentDescriptor> objectives, IReadOnlyDictionary<string, ShipFleetMetadata> shipMetadata = null, IReadOnlyDictionary<string, SquadronFleetMetadata> squadronMetadata = null, IEnumerable<FleetContentAlias> aliases = null)
	{
		FleetContentCatalogBuilder fleetContentCatalogBuilder = new FleetContentCatalogBuilder();
		if (ships != null)
		{
			foreach (ShipDefinition ship in ships)
			{
				if (!(ship == null))
				{
					fleetContentCatalogBuilder.Add(Describe(ship, TryGet(shipMetadata, ship.ShipId)));
				}
			}
		}
		if (squadrons != null)
		{
			foreach (SquadronDefinition squadron in squadrons)
			{
				if (!(squadron == null))
				{
					fleetContentCatalogBuilder.Add(Describe(squadron, TryGet(squadronMetadata, squadron.SquadronId)));
				}
			}
		}
		foreach (UpgradeContentDescriptor item in OfficialContentPolicy.DescribeAll(UpgradeCatalog.Cards))
		{
			fleetContentCatalogBuilder.Add(item);
		}
		if (objectives != null)
		{
			foreach (ObjectiveContentDescriptor objective in objectives)
			{
				if (objective != null)
				{
					fleetContentCatalogBuilder.Add(objective);
				}
			}
		}
		if (aliases != null)
		{
			foreach (FleetContentAlias alias in aliases)
			{
				if (alias != null)
				{
					fleetContentCatalogBuilder.AddAlias(alias.Kind, alias.AliasId, alias.CanonicalId);
				}
			}
		}
		return fleetContentCatalogBuilder.Build();
	}

	private static T TryGet<T>(IReadOnlyDictionary<string, T> values, string id) where T : class
	{
		if (values == null || string.IsNullOrWhiteSpace(id))
		{
			return null;
		}
		if (!values.TryGetValue(id, out var value))
		{
			return null;
		}
		return value;
	}

	private static string FactionId(ShipFaction faction)
	{
		return ShipFactionCatalog.Id(faction);
	}
}
