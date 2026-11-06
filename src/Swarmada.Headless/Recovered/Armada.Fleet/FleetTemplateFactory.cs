#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using Armada.Data;
using Armada.Session;

namespace Armada.Fleet;

public static class FleetTemplateFactory
{
	public const int StarterPresetRevision = 1;

	private const string StarterPresetPrefix = "official-showcase-400";

	public static FleetDraft CreateStarter(IFleetContentCatalog content, string faction, string displayName)
	{
		if (content == null)
		{
			throw new ArgumentNullException("content");
		}
		string text = NormalizeStarterFaction(faction);
		if (TryCreateShowcase(content, text, displayName, out var draft))
		{
			return draft;
		}
		FleetDraft fleetDraft = CreateCatalogFallback(content, text, displayName);
		ApplyStarterIdentity(fleetDraft.Document, text);
		return fleetDraft;
	}

	public static string GetStarterPresetId(string faction)
	{
		return "official-showcase-400-" + NormalizeStarterFaction(faction);
	}

	public static string GetStarterFleetId(string faction)
	{
		return "starter-showcase-" + NormalizeStarterFaction(faction);
	}

	private static bool TryCreateShowcase(IFleetContentCatalog content, string faction, string displayName, out FleetDraft draft)
	{
		draft = FleetDraft.Create(displayName, faction, FleetRuleset.OfficialStandard400.MaximumFleetPoints);
		if (faction switch
		{
			"rebel" => BuildRebelShowcase(content, draft) ? 1 : 0,
			"empire" => BuildEmpireShowcase(content, draft) ? 1 : 0,
			"republic" => BuildRepublicShowcase(content, draft) ? 1 : 0,
			"separatist" => BuildSeparatistShowcase(content, draft) ? 1 : 0,
			_ => 0,
		} == 0)
		{
			draft = null;
			return false;
		}
		if (!new FleetLegalityValidator(content, FleetRuleset.OfficialStandard400, new FleetValidationOptions(reportGameplayCoverage: false)).Validate(draft.Document).IsLegal)
		{
			draft = null;
			return false;
		}
		ApplyStarterIdentity(draft.Document, faction);
		return true;
	}

	private static bool BuildRebelShowcase(IFleetContentCatalog content, FleetDraft draft)
	{
		if (TryAddShip(content, draft, "mc30c-torpedo-frigate", "general-dodonna-commander", "ordnance-experts", "external-racks", "electronic-countermeasures") && TryAddShip(content, draft, "mc75-ordnance-cruiser", "ordnance-experts", "external-racks", "electronic-countermeasures", "aspiration-title") && TryAddShip(content, draft, "cr90-corvette-a", "turbolaser-reroute-circuits", "jainas-light-title") && TryAddShip(content, draft, "gr75-medium-transports", "comms-net"))
		{
			return TryAddSquadrons(content, draft, "luke-skywalker", "dutch-vander", "x-wing-squadron", "x-wing-squadron", "a-wing-squadron", "a-wing-squadron", "y-wing-squadron", "jan-ors-moldy-crow");
		}
		return false;
	}

	private static bool BuildEmpireShowcase(IFleetContentCatalog content, FleetDraft draft)
	{
		if (TryAddShip(content, draft, "imperial-ii-class-star-destroyer", "admiral-sloane-commander", "gunnery-team", "early-warning-system", "leading-shots", "xi7-turbolasers") && TryAddShip(content, draft, "quasar-fire-i-class-cruiser-carrier", "flight-controllers", "expanded-hangar-bay", "squall-title") && TryAddShip(content, draft, "gozanti-class-cruisers", "comms-net"))
		{
			return TryAddSquadrons(content, draft, "maarek-stele-tie-defender-squadron", "colonel-jendon-lambda-class-shuttle", "howlrunner", "tie-fighter-squadron", "tie-fighter-squadron", "tie-fighter-squadron", "tie-interceptor-squadron", "tie-interceptor-squadron", "tie-bomber-squadron", "tie-bomber-squadron", "black-squadron-tie-fighter-squadron");
		}
		return false;
	}

	private static bool BuildRepublicShowcase(IFleetContentCatalog content, FleetDraft draft)
	{
		if (TryAddShip(content, draft, "venator-ii-class-star-destroyer", "admiral-yularen-commander", "clone-commander-wolffe-officer", "clone-gunners", "spha-t", "thermal-shields", "heavy-turbolaser-turrets", "ordnance-pods", "resolute-title") && TryAddShip(content, draft, "acclamator-ii-class-assault-ship", "barriss-offee-officer", "gunnery-team", "early-warning-system", "swivel-mount-batteries", "assault-proton-torpedoes", "nevoota-bee-title") && TryAddShip(content, draft, "consular-class-charger-c70", "ahsoka-tano-officer-republic", "engine-techs", "expanded-hangar-bay", "radiant-vii-title", "bomber-command-center"))
		{
			return TryAddSquadrons(content, draft, "matchstick-btl-b-y-wing-squadron", "v-19-torrent-squadron", "v-19-torrent-squadron", "arc-170-starfighter-squadron", "btl-b-y-wing-squadron");
		}
		return false;
	}

	private static bool BuildSeparatistShowcase(IFleetContentCatalog content, FleetDraft draft)
	{
		if (TryAddShip(content, draft, "providence-class-dreadnought", "count-dooku-commander", "passel-argente-officer", "ruthless-strategists", "flag-bridge", "flechette-torpedoes", "overload-pulse", "heavy-turbolaser-turrets", "xi7-turbolasers", "invincible-title", "thermal-shields", "jedi-hostage") && TryAddShip(content, draft, "munificent-class-comms-frigate", "asajj-ventress-officer", "battle-droid-reserves", "b2-rocket-troopers", "swivel-mount-batteries", "ms-1-ion-cannons", "sa-nalaor-title") && TryAddShip(content, draft, "hardcell-class-transport", "t-series-tactical-droid-officer", "ti-99-officer", "nav-team", "munitions-resupply", "beast-of-burden-title"))
		{
			return TryAddSquadrons(content, draft, "belbullab-22-starfighter-squadron", "vulture-class-droid-fighter-squadron", "haor-chall-prototypes-vulture-class-droid-fighter-squadron", "dbs-404-hyena-class-droid-bomber-squadron", "droid-tri-fighter-squadron");
		}
		return false;
	}

	private static bool TryAddShip(IFleetContentCatalog content, FleetDraft draft, string shipId, params string[] upgradeIds)
	{
		if (!content.TryGetShip(shipId, out var ship) || !SameFaction(ship.Faction, draft.Document.faction))
		{
			return false;
		}
		foreach (string id in upgradeIds)
		{
			if (!content.TryGetUpgrade(id, out var upgrade) || upgrade.OfficialDisposition != OfficialContentDisposition.StrictOfficial)
			{
				return false;
			}
		}
		draft.AddShip(shipId, upgradeIds);
		return true;
	}

	private static bool TryAddSquadrons(IFleetContentCatalog content, FleetDraft draft, params string[] squadronIds)
	{
		string[] array = squadronIds;
		foreach (string id in array)
		{
			if (!content.TryGetSquadron(id, out var squadron) || !SameFaction(squadron.Faction, draft.Document.faction))
			{
				return false;
			}
		}
		array = squadronIds;
		foreach (string squadronDefinitionId in array)
		{
			draft.AddSquadron(squadronDefinitionId);
		}
		return true;
	}

	private static FleetDraft CreateCatalogFallback(IFleetContentCatalog content, string normalizedFaction, string displayName)
	{
		FleetDraft fleetDraft = FleetDraft.Create(displayName, normalizedFaction, FleetRuleset.OfficialStandard400.MaximumFleetPoints);
		ShipContentDescriptor[] array = (from value in content.Ships
			where SameFaction(value.Faction, normalizedFaction)
			orderby value.Points, value.DisplayName
			select value).ToArray();
		if (array.Length == 0)
		{
			return fleetDraft;
		}
		ShipContentDescriptor flagship = array.FirstOrDefault((ShipContentDescriptor value) => ContainsSlot(value.UpgradeSlots, "commander")) ?? array[0];
		fleetDraft.AddShip(flagship.Id);
		ShipContentDescriptor shipContentDescriptor = array.FirstOrDefault((ShipContentDescriptor value) => !string.Equals(value.Id, flagship.Id, StringComparison.OrdinalIgnoreCase) && value.IsFlotilla != true);
		if (shipContentDescriptor != null)
		{
			fleetDraft.AddShip(shipContentDescriptor.Id);
		}
		AddCommander(content, fleetDraft, flagship, normalizedFaction);
		AddSquadrons(content, fleetDraft, normalizedFaction, 4);
		AddOneLegalUpgradePerUnadornedShip(content, fleetDraft);
		return fleetDraft;
	}

	private static void ApplyStarterIdentity(FleetListDto fleet, string faction)
	{
		fleet.fleetId = GetStarterFleetId(faction);
		FleetStarterPresetReconciler.Stamp(fleet, GetStarterPresetId(faction), 1);
	}

	private static string NormalizeStarterFaction(string value)
	{
		string text = ShipFactionCatalog.NormalizeId(value);
		if (!ShipFactionCatalog.TryParse(text, out var _))
		{
			return "rebel";
		}
		return text;
	}

	private static void AddCommander(IFleetContentCatalog content, FleetDraft draft, ShipContentDescriptor flagship, string faction)
	{
		if (ContainsSlot(flagship.UpgradeSlots, "commander"))
		{
			UpgradeContentDescriptor upgradeContentDescriptor = (from value in content.Upgrades
				where value.OfficialDisposition == OfficialContentDisposition.StrictOfficial && string.Equals(value.Slot, "commander", StringComparison.OrdinalIgnoreCase) && FactionAllowed(value.Factions, faction)
				orderby value.Points, value.DisplayName
				select value).FirstOrDefault();
			if (upgradeContentDescriptor != null)
			{
				draft.ToggleUpgrade(draft.Document.ships[0].instanceId, upgradeContentDescriptor.Id);
			}
		}
	}

	private static void AddSquadrons(IFleetContentCatalog content, FleetDraft draft, string faction, int count)
	{
		SquadronContentDescriptor[] array = (from value in content.Squadrons
			where SameFaction(value.Faction, faction) && !value.IsUnique
			orderby value.Points, value.DisplayName
			select value).ToArray();
		if (array.Length == 0)
		{
			return;
		}
		int num = 0;
		int num2 = FleetRuleset.Standard2025.SquadronPointLimit(draft.Document.pointLimit);
		for (int num3 = 0; num3 < count; num3++)
		{
			SquadronContentDescriptor squadronContentDescriptor = array[num3 % array.Length];
			if (num + squadronContentDescriptor.Points <= num2)
			{
				draft.AddSquadron(squadronContentDescriptor.Id);
				num += squadronContentDescriptor.Points;
				continue;
			}
			break;
		}
	}

	private static void AddOneLegalUpgradePerUnadornedShip(IFleetContentCatalog content, FleetDraft draft)
	{
		FleetLegalityValidator fleetLegalityValidator = new FleetLegalityValidator(content);
		ShipBuildDto[] ships = draft.Document.ships;
		foreach (ShipBuildDto shipBuildDto in ships)
		{
			if (shipBuildDto == null || shipBuildDto.upgradeIds.Length != 0 || !content.TryGetShip(shipBuildDto.shipDefinitionId, out var _))
			{
				continue;
			}
			FleetLegalityReport baselineReport = fleetLegalityValidator.Validate(draft.Document);
			foreach (UpgradeContentDescriptor item in from value in content.Upgrades
				where value.OfficialDisposition == OfficialContentDisposition.StrictOfficial && !string.Equals(value.Slot, "commander", StringComparison.OrdinalIgnoreCase) && FactionAllowed(value.Factions, draft.Document.faction)
				orderby value.Points, value.DisplayName
				select value)
			{
				if (FleetUpgradeTrialValidator.CanAddUpgrade(draft.Document, shipBuildDto.instanceId, item.Id, fleetLegalityValidator, baselineReport) && draft.ToggleUpgrade(shipBuildDto.instanceId, item.Id))
				{
					break;
				}
			}
		}
	}

	private static bool ContainsSlot(IReadOnlyList<string> slots, string requested)
	{
		if (slots == null)
		{
			return false;
		}
		foreach (string slot in slots)
		{
			if (string.Equals(slot, requested, StringComparison.OrdinalIgnoreCase))
			{
				return true;
			}
		}
		return false;
	}

	private static bool FactionAllowed(IReadOnlyList<string> allowed, string faction)
	{
		if (allowed == null || allowed.Count == 0)
		{
			return true;
		}
		foreach (string item in allowed)
		{
			if (string.IsNullOrWhiteSpace(item) || SameFaction(item, faction))
			{
				return true;
			}
		}
		return false;
	}

	private static bool SameFaction(string left, string right)
	{
		return string.Equals(NormalizeFaction(left), NormalizeFaction(right), StringComparison.OrdinalIgnoreCase);
	}

	private static string NormalizeFaction(string value)
	{
		return ShipFactionCatalog.NormalizeId(value);
	}
}
