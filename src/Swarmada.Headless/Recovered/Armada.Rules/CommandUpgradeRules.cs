#nullable disable
using System;
using Swarmada.Headless.Engine;

namespace Armada.Rules;

public static class CommandUpgradeRules
{
	public const string BoostedComms = "boosted-comms";

	public const string ExpandedHangarBay = "expanded-hangar-bay";

	public const string FlightControllers = "flight-controllers";

	public const string RepairCrews = "repair-crews";

	public const string CommsNet = "comms-net";

	public const string BattleDroidReserves = "battle-droid-reserves";

	public const string ProjectionExperts = "projection-experts";

	public const string WatTambor = "wat-tambor-officer";

	public const string MedicalTeam = "medical-team";

	public const string LiraWessex = "lira-wessex-officer";

	public const string MunitionsResupply = "munitions-resupply";

	public const string PartsResupply = "parts-resupply";

	public const string ResupplyActionPrefix = "command.resupply.";

	public const string CloneNavigationOfficer = "clone-navigation-officer";

	public const string CommandantAresko = "commandant-aresko-officer";

	public const string CommanderVanto = "commander-vanto-officer";

	public const string LeiaOrganaOfficer = "leia-organa-officer";

	public const string DirectorIsard = "director-isard-officer";

	public static int SquadronValue(ShipView ship)
	{
		if (ship?.Definition == null)
		{
			return 0;
		}
		int num = ship.Definition.Squadron + ((ship.Loadout?.HasActive("expanded-hangar-bay") ?? false) ? 1 : 0);
		ShipLoadoutState loadout = ship.Loadout;
		return Math.Max(0, num - ((loadout != null && loadout.HasActive("spha-t")) ? 2 : 0));
	}

	public static float SquadronCommandRange(ShipView ship)
	{
		if ((object)ship == null || ship.Loadout?.HasActive("boosted-comms") != true)
		{
			return 8.227571f;
		}
		return 13.260394f;
	}

	public static bool GrantsFlightControllersBlueDie(SquadronView squadron)
	{
		ShipView shipView = TitleRules.FindCommandingShip(squadron);
		if ((object)shipView == null)
		{
			return false;
		}
		return shipView.Loadout?.HasActive("flight-controllers") == true;
	}
}
