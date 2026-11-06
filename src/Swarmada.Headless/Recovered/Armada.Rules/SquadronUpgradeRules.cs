#nullable disable
using Swarmada.Headless.Engine;
using System.Numerics;
using Mathf = Swarmada.Headless.Engine.EngineMath;

namespace Armada.Rules;

public static class SquadronUpgradeRules
{
	public const string BomberCommandCenter = "bomber-command-center";

	public const string PointDefenseReroute = "point-defense-reroute";

	public const string JammingField = "jamming-field";

	public const string AdvancedTransponderNet = "advanced-transponder-net";

	public const string ReserveHangarDeck = "reserve-hangar-deck";

	public static ShipView BomberCommandCenterSource(SquadronView attacker)
	{
		if (attacker == null || !SquadronAbilityEngine.HasBomber(attacker))
		{
			return null;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == attacker.OwnerId)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive("bomber-command-center") && SquadronAbilityEngine.FootprintDistance(attacker, shipView) <= 13.260394f)
				{
					return shipView;
				}
			}
		}
		return null;
	}

	public static ShipView JammingFieldSource(SquadronView attacker, SquadronView defender)
	{
		if (attacker == null || defender == null)
		{
			return null;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive("jamming-field") && (SquadronAbilityEngine.FootprintDistance(attacker, shipView) <= 5.426696f || SquadronAbilityEngine.FootprintDistance(defender, shipView) <= 5.426696f))
				{
					return shipView;
				}
			}
		}
		return null;
	}

	public static bool HeavyPreventsShipAttack(SquadronView heavySquadron)
	{
		if (heavySquadron == null || !SquadronAbilityEngine.HasKeyword(heavySquadron.Definition, "HEAVY"))
		{
			return false;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == heavySquadron.OwnerId)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive("advanced-transponder-net") && SquadronAbilityEngine.FootprintDistance(heavySquadron, shipView) <= 5.426696f)
				{
					return true;
				}
			}
		}
		return false;
	}
}
