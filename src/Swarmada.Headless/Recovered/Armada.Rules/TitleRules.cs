#nullable disable
using System;
using System.Collections.Generic;
using Armada.Core;
using Armada.Data;
using Swarmada.Headless.Engine;
using System.Numerics;
using Mathf = Swarmada.Headless.Engine.EngineMath;

namespace Armada.Rules;

public static class TitleRules
{
	public const string Annihilator = "annihilator-title";

	public const string Corrupter = "corrupter-title";

	public const string GallantHaven = "gallant-haven-title";

	public const string Liberty = "liberty-title";

	public const string Redemption = "redemption-title";

	public const string Stronghold = "stronghold-title";

	public const string Unity = "unity-title";

	public const string Vector = "vector-title";

	public const string EngineeringTeam = "engineering-team";

	public const string HomeOne = "home-one-title";

	public const string Insidious = "insidious-title";

	public const string LucidVoice = "lucid-voice-title";

	public const string Eclipse = "eclipse-title";

	public const string GarelsHonor = "garels-honor-title";

	public const string MonKarren = "mon-karren-title";

	public const string TideOfProgressXii = "tide-of-progress-xii-title";

	public const string MonCalamariExodusFleet = "mon-calamari-exodus-fleet-title";

	public static bool LimitsDefenseTokens(ShipView attacker, bool concentrateFireActive)
	{
		return (object)attacker != null && attacker.Loadout?.HasActive("mon-karren-title") == true && concentrateFireActive;
	}

	public static int CommandedSquadronSpeed(SquadronView squadron, int printedSpeed)
	{
		int num = Math.Max(0, printedSpeed);
		ShipView shipView = FindCommandingShip(squadron);
		if (shipView == null)
		{
			return num;
		}
		ShipLoadoutState loadout = shipView.Loadout;
		if (loadout != null && loadout.HasActive("corrupter-title") && SquadronAbilityEngine.HasKeyword(squadron.Definition, "BOMBER"))
		{
			num++;
		}
		ShipLoadoutState loadout2 = shipView.Loadout;
		if (loadout2 != null && loadout2.HasActive("vector-title") && !SquadronAbilityEngine.HasKeyword(squadron.Definition, "HEAVY"))
		{
			num = Math.Min(5, num + 1);
		}
		return num;
	}

	public static bool AllowsBlackDiceAtMediumAgainstRear(ShipView attacker, HullZone targetZone)
	{
		if ((object)attacker != null && attacker.Loadout?.HasActive("insidious-title") == true)
		{
			return targetZone == HullZone.Rear;
		}
		return false;
	}

	public static BatteryArmament EffectiveBattery(ShipView ship, HullZone zone, BatteryArmament printed)
	{
		if ((object)ship == null || ship.Loadout?.HasActive("lucid-voice-title") != true)
		{
			return printed;
		}
		switch (zone)
		{
		case HullZone.Rear:
			return new BatteryArmament(3, 0, 0);
		case HullZone.Left:
		case HullZone.Right:
			return new BatteryArmament(printed.RedDice, Math.Max(0, printed.BlueDice - 1), printed.BlackDice + 1);
		default:
			return printed;
		}
	}

	public static bool DealsFaceupCollisionCardToMover(ShipView stationary, ShipView mover)
	{
		if (stationary != null && mover != null && stationary.OwnerId != mover.OwnerId)
		{
			return stationary.Loadout?.HasActive("eclipse-title") ?? false;
		}
		return false;
	}

	public static bool DealsFaceupCollisionCardToRammedShip(ShipView mover, ShipView stationary)
	{
		if (mover != null && stationary != null && mover.OwnerId != stationary.OwnerId)
		{
			return mover.Loadout?.HasActive("garels-honor-title") ?? false;
		}
		return false;
	}

	public static PassiveEffectNotice CommandedSquadronSpeedNotice(ShipView source, SquadronView squadron)
	{
		if (source?.Loadout == null || squadron?.Definition == null)
		{
			return PassiveEffectNotice.None;
		}
		string text = string.Empty;
		if (source.Loadout.HasActive("corrupter-title") && SquadronAbilityEngine.HasKeyword(squadron.Definition, "BOMBER"))
		{
			text = "Corrupter";
		}
		else if (source.Loadout.HasActive("vector-title") && !SquadronAbilityEngine.HasKeyword(squadron.Definition, "HEAVY"))
		{
			text = "Vector";
		}
		if (string.IsNullOrEmpty(text))
		{
			return PassiveEffectNotice.None;
		}
		return PassiveEffectNotice.LogOnly("[TITLE] " + text + ": " + squadron.Definition.DisplayName + "'s speed is increased by 1 for this commanded activation.");
	}

	public static int AdditionalSquadronActivations(ShipView ship, CommandSpendMode mode)
	{
		if ((object)ship == null || ship.Loadout?.HasActive("liberty-title") != true || !CommandSpendRules.UsesToken(mode))
		{
			return 0;
		}
		return 1;
	}

	public static PassiveEffectNotice AdditionalEngineeringPoint(ShipView resolvingShip, ref int engineeringPoints)
	{
		List<string> list = new List<string>();
		if (FindFriendlyTitleAtDistance(resolvingShip, "redemption-title", 13.260394f) != null)
		{
			engineeringPoints++;
			list.Add("Redemption");
		}
		if ((object)resolvingShip != null && resolvingShip.Loadout?.HasActive("engineering-team") == true)
		{
			engineeringPoints++;
			list.Add("Engineering Team");
		}
		if (list.Count == 0)
		{
			return PassiveEffectNotice.None;
		}
		return PassiveEffectNotice.LogOnly("[UPGRADE] " + string.Join(" + ", list) + ": " + resolvingShip.Definition.DisplayName + " gains " + list.Count + " additional engineering point" + ((list.Count == 1) ? "." : "s."));
	}

	public static int FlakRerolls(ShipView attacker, SquadronView defender, IEnumerable<SquadronView> allSquadrons, out string cardName)
	{
		cardName = string.Empty;
		if ((object)attacker != null && attacker.Loadout?.HasActive("annihilator-title") == true)
		{
			cardName = "Annihilator";
			return 1;
		}
		if ((object)attacker == null || attacker.Loadout?.HasActive("unity-title") != true || defender == null || allSquadrons == null)
		{
			return 0;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !allSquadron.IsDestroyed && allSquadron.OwnerId == attacker.OwnerId && !SquadronAbilityEngine.HasKeyword(allSquadron.Definition, "HEAVY") && SquadronAbilityEngine.IsEngaged(allSquadron, defender))
			{
				cardName = "Unity";
				return 1;
			}
		}
		return 0;
	}

	public static ShipView StrongholdSource(SquadronView defender)
	{
		if (defender == null || !SquadronAbilityEngine.HasKeyword(defender.Definition, "SWARM"))
		{
			return null;
		}
		ShipView[] array = FindShips();
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == defender.OwnerId)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive("stronghold-title") && SquadronAbilityEngine.FootprintDistance(defender, shipView) <= 5.426696f)
				{
					return shipView;
				}
			}
		}
		return null;
	}

	public static PassiveEffectNotice ReduceSquadronAttackDamage(SquadronView defender, ref int damage)
	{
		if (defender == null || damage <= 1)
		{
			return PassiveEffectNotice.None;
		}
		ShipView shipView = FindFriendlyTitleAtDistance(defender, "gallant-haven-title", 3.369803f);
		if (shipView == null)
		{
			return PassiveEffectNotice.None;
		}
		damage = Math.Max(1, damage - 1);
		return PassiveEffectNotice.Battlefield(shipView, "[TITLE] Gallant Haven reduced attack damage to " + defender.Definition.DisplayName + " by 1.", "GALLANT HAVEN  -  DAMAGE REDUCED");
	}

	public static ShipView HomeOneSource(ShipView attacker)
	{
		if (attacker == null)
		{
			return null;
		}
		ShipView[] array = FindShips();
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !(shipView == attacker) && !shipView.IsDestroyed && shipView.OwnerId == attacker.OwnerId)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive("home-one-title") && RaidRules.FootprintDistance(attacker, shipView) <= 13.260394f)
				{
					return shipView;
				}
			}
		}
		return null;
	}

	public static ShipView FindCommandingShip(SquadronView squadron)
	{
		string text = squadron?.State?.CommandingShipRuntimeId ?? string.Empty;
		if (string.IsNullOrWhiteSpace(text))
		{
			return null;
		}
		ShipView[] array = FindShips();
		foreach (ShipView shipView in array)
		{
			if (shipView != null && string.Equals(shipView.RuntimeId.Value, text, StringComparison.Ordinal))
			{
				return shipView;
			}
		}
		return null;
	}

	private static ShipView FindFriendlyTitleAtDistance(ShipView target, string titleId, float maximumDistance)
	{
		if (target == null)
		{
			return null;
		}
		ShipView[] array = FindShips();
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == target.OwnerId)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive(titleId) && RaidRules.FootprintDistance(target, shipView) <= maximumDistance)
				{
					return shipView;
				}
			}
		}
		return null;
	}

	private static ShipView FindFriendlyTitleAtDistance(SquadronView target, string titleId, float maximumDistance)
	{
		if (target == null)
		{
			return null;
		}
		ShipView[] array = FindShips();
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == target.OwnerId)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive(titleId) && SquadronAbilityEngine.FootprintDistance(target, shipView) <= maximumDistance)
				{
					return shipView;
				}
			}
		}
		return null;
	}

	private static ShipView[] FindShips()
	{
		return GameScope.Current.SceneShipViews;
	}
}
