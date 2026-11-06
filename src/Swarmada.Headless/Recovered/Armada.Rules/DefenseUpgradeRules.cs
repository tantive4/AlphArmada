#nullable disable
using System;
using System.Collections.Generic;
using Armada.Core;
using Swarmada.Headless.Engine;
using System.Numerics;
using Mathf = Swarmada.Headless.Engine.EngineMath;

namespace Armada.Rules;

public static class DefenseUpgradeRules
{
	public const string BrightHope = "bright-hope-title";

	public const string ClusterBombs = "cluster-bombs";

	public const string EarlyWarningSystem = "early-warning-system";

	public const string ElectronicCountermeasures = "electronic-countermeasures";

	public const string ForemansLabor = "foremans-labor-title";

	public const string MajorDerlin = "major-derlin-officer";

	public const string RedundantShields = "redundant-shields";

	public const string ReinforcedBlastDoors = "reinforced-blast-doors";

	public const string AdvancedProjectors = "advanced-projectors";

	public const string Foresight = "foresight-title";

	public const string ThermalShields = "thermal-shields";

	public const string PointDefenseIonCannons = "point-defense-ion-cannons";

	public const string ReactiveGunnery = "reactive-gunnery";

	public const string TargetingScrambler = "targeting-scrambler";

	public const string ShipPhaseActionPrefix = "ship-phase.defensive-retrofit.";

	public const string ClusterBombsActionPrefix = "after-attack.cluster-bombs.";

	public static bool HasStartOfShipPhaseChoice(string cardId)
	{
		if (!(cardId == "early-warning-system"))
		{
			return cardId == "reinforced-blast-doors";
		}
		return true;
	}

	public static int MaximumRedirectZones(ShipView defender)
	{
		if ((object)defender != null && defender.Loadout?.HasActive("advanced-projectors") == true)
		{
			return 3;
		}
		if ((object)defender != null && defender.Loadout?.HasActive("foresight-title") == true)
		{
			return 2;
		}
		return 1;
	}

	public static bool AllowsNonAdjacentRedirect(ShipView defender)
	{
		if ((object)defender == null)
		{
			return false;
		}
		return defender.Loadout?.HasActive("advanced-projectors") == true;
	}

	public static bool CanUseThermalShields(ShipView defender, int gatheredDiceCount)
	{
		return CanUseThermalShields(defender, gatheredDiceCount, gatheredDiceCount);
	}

	public static bool CanUseThermalShields(ShipView defender, int initialGatheredDiceCount, int remainingDiceCount)
	{
		UpgradeCardRuntimeState upgradeCardRuntimeState = defender?.Loadout?.Find("thermal-shields");
		if (defender != null && !defender.IsDestroyed && defender.State != null && defender.State.CurrentSpeed > 0 && ThermalShieldsRemovalCount(initialGatheredDiceCount, remainingDiceCount) > 0 && upgradeCardRuntimeState != null && upgradeCardRuntimeState.Active && !upgradeCardRuntimeState.Exhausted && upgradeCardRuntimeState.UsesThisAttack == 0)
		{
			return defender.State.CanResolveDefenseToken(DefenseTokenType.Brace);
		}
		return false;
	}

	public static int ThermalShieldsRemovalCount(int initialGatheredDiceCount, int remainingDiceCount)
	{
		return Math.Min(Math.Max(0, remainingDiceCount), Math.Max(0, initialGatheredDiceCount) / 2);
	}

	public static ShipView TargetingScramblerSource(ShipView defender)
	{
		if (defender == null)
		{
			return null;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == defender.OwnerId)
			{
				UpgradeCardRuntimeState upgradeCardRuntimeState = shipView.Loadout?.Find("targeting-scrambler");
				if (upgradeCardRuntimeState != null && upgradeCardRuntimeState.Active && !upgradeCardRuntimeState.Exhausted && RaidRules.FootprintDistance(shipView, defender) <= 8.227571f)
				{
					return shipView;
				}
			}
		}
		return null;
	}

	public static IReadOnlyList<PassiveEffectNotice> ApplyBeforeDamageReductions(ShipView defender, HullZone defendingZone, int selectedReduction, ref DefenseResolution defense, int defenseTokensSpent = 0, Func<ShipView, HullZone, int, DamageReport> applyShipDamage = null)
	{
		List<PassiveEffectNotice> list = new List<PassiveEffectNotice>();
		int num = Math.Min(Math.Max(0, selectedReduction), defense.TotalDamage);
		if (num > 0)
		{
			defense = DefenseResolver.ReduceTotalDamage(defense, num);
		}
		if ((object)defender != null && defender.Loadout?.HasActive("bright-hope-title") == true && defendingZone != HullZone.Rear && defense.TotalDamage > 0)
		{
			defense = DefenseResolver.ReduceTotalDamage(defense, 1);
			list.Add(PassiveEffectNotice.Battlefield(defender, "[TITLE] Bright Hope reduced incoming damage by 1.", "BRIGHT HOPE  -  DAMAGE REDUCED"));
		}
		return list;
	}

	private static void ApplyAdiGalliaRedirect(ShipView defender, HullZone defendingZone, int defenseTokensSpent, ref DefenseResolution defense, ICollection<PassiveEffectNotice> notices)
	{
		if ((object)defender == null || defender.Loadout?.HasActive("adi-gallia-officer") != true || defenseTokensSpent <= 0 || defense.DamageToPrimaryZone <= 0)
		{
			return;
		}
		Dictionary<HullZone, int> dictionary = new Dictionary<HullZone, int>(defense.RedirectedDamageByZone);
		int num = defense.DamageToPrimaryZone;
		int num2 = 0;
		HullZone[] array = HullZoneRules.Adjacent(defendingZone, defender.Definition.HasAuxiliaryHullZones);
		foreach (HullZone hullZone in array)
		{
			if (num2 >= defenseTokensSpent || num <= 0)
			{
				break;
			}
			int value;
			int num3 = (dictionary.TryGetValue(hullZone, out value) ? value : 0);
			if (defender.State.GetShields(hullZone) > num3)
			{
				dictionary[hullZone] = num3 + 1;
				num--;
				num2++;
			}
		}
		if (num2 > 0)
		{
			defense = new DefenseResolution(defense.Choice, defense.OriginalDamage, num, dictionary, defense.RedirectZone);
			notices.Add(PassiveEffectNotice.Battlefield(defender, "[OFFICER] Adi Gallia moved " + num2 + " damage onto adjacent shields.", "ADI GALLIA  -  " + num2 + " DAMAGE REDIRECTED"));
		}
	}

	private static void ApplyTranquility(ShipView defender, HullZone defendingZone, int defenseTokensSpent, ICollection<PassiveEffectNotice> notices)
	{
		if ((object)defender == null || defender.Loadout?.HasActive("tranquility-title") != true || defenseTokensSpent >= 2 || defender.State.GetShields(defendingZone) >= 6)
		{
			return;
		}
		int i;
		for (i = 0; i < 2; i++)
		{
			if (defender.State.GetShields(defendingZone) >= 6)
			{
				break;
			}
			HullZone? hullZone = StrongestShieldZone(defender, defendingZone);
			if (!hullZone.HasValue || defender.State.TransferShield(hullZone.Value, defendingZone, increaseSideMaximum: false, 6) <= 0)
			{
				break;
			}
		}
		if (i > 0)
		{
			notices.Add(PassiveEffectNotice.Battlefield(defender, "[TITLE] Tranquility moved " + i + " shield" + ((i == 1) ? string.Empty : "s") + " to the defending hull zone.", "TRANQUILITY  -  +" + i + " SHIELD" + ((i == 1) ? string.Empty : "S")));
		}
	}

	private static void ApplyNearbyDamageReducers(ShipView defender, HullZone defendingZone, ref DefenseResolution defense, ICollection<PassiveEffectNotice> notices, Func<ShipView, HullZone, int, DamageReport> applyShipDamage)
	{
		if (defender == null || defense.TotalDamage <= 0)
		{
			return;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		ShipView[] array2 = array;
		foreach (ShipView shipView in array2)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == defender.OwnerId && defendingZone == HullZone.Front && shipView != defender && RaidRules.FootprintDistance(shipView, defender) <= 10.7221f)
			{
				UpgradeCardRuntimeState upgradeCardRuntimeState = shipView.Loadout?.Find("seventh-fleet-star-destroyer-title");
				if (upgradeCardRuntimeState != null && upgradeCardRuntimeState.Active && !upgradeCardRuntimeState.Exhausted && upgradeCardRuntimeState.Exhaust())
				{
					defense = DefenseResolver.ReduceTotalDamage(defense, 1);
					notices.Add(PassiveEffectNotice.Battlefield(defender, "[TITLE] Seventh Fleet Star Destroyer reduced front-hull attack damage by 1.", "SEVENTH FLEET  -  DAMAGE REDUCED"));
					break;
				}
			}
		}
		if (defense.TotalDamage <= 0)
		{
			return;
		}
		array2 = array;
		foreach (ShipView shipView2 in array2)
		{
			if (!(shipView2 == null) && !(shipView2 == defender) && !shipView2.IsDestroyed && shipView2.OwnerId == defender.OwnerId && !(RaidRules.FootprintDistance(shipView2, defender) > 8.227571f))
			{
				UpgradeCardRuntimeState upgradeCardRuntimeState2 = shipView2.Loadout?.Find("task-force-antilles-title");
				if (upgradeCardRuntimeState2 != null && upgradeCardRuntimeState2.Active && !upgradeCardRuntimeState2.Exhausted && upgradeCardRuntimeState2.Exhaust())
				{
					defense = DefenseResolver.ReduceTotalDamage(defense, 1);
					HullZone valueOrDefault = StrongestShieldZone(shipView2, null).GetValueOrDefault();
					DamageReport damageReport = applyShipDamage?.Invoke(shipView2, valueOrDefault, 1) ?? shipView2.State.ApplyDamage(valueOrDefault, 1);
					notices.Add(PassiveEffectNotice.Battlefield(defender, "[TITLE] Task Force Antilles transferred 1 damage to " + shipView2.Definition.DisplayName + " (" + damageReport.TotalApplied + " applied).", "TASK FORCE ANTILLES  -  DAMAGE TRANSFERRED"));
					break;
				}
			}
		}
		if (defense.TotalDamage <= 0)
		{
			return;
		}
		array2 = array;
		foreach (ShipView shipView3 in array2)
		{
			if (shipView3 == null || shipView3.IsDestroyed || shipView3.OwnerId != defender.OwnerId || RaidRules.FootprintDistance(shipView3, defender) > 5.426696f)
			{
				continue;
			}
			UpgradeCardRuntimeState upgradeCardRuntimeState3 = shipView3.Loadout?.Find("implacable-title");
			if (upgradeCardRuntimeState3 != null && upgradeCardRuntimeState3.Active && !upgradeCardRuntimeState3.Exhausted)
			{
				int num = Math.Min(2, Math.Min(defense.TotalDamage, shipView3.State.GetShields(HullZone.Front)));
				if (num > 0 && upgradeCardRuntimeState3.Exhaust())
				{
					shipView3.State.LoseShields(HullZone.Front, num);
					defense = DefenseResolver.ReduceTotalDamage(defense, num);
					notices.Add(PassiveEffectNotice.Battlefield(defender, "[TITLE] Implacable spent " + num + " front shield" + ((num == 1) ? string.Empty : "s") + " to reduce incoming damage.", "IMPLACABLE  -  " + num + " DAMAGE PREVENTED"));
					break;
				}
			}
		}
	}

	private static IEnumerable<HullZone> AdjacentZones(HullZone zone)
	{
		if (zone == HullZone.Front || zone == HullZone.Rear)
		{
			yield return HullZone.Left;
			yield return HullZone.Right;
		}
		else
		{
			yield return HullZone.Front;
			yield return HullZone.Rear;
		}
	}

	private static HullZone? StrongestShieldZone(ShipView ship, HullZone? excluded)
	{
		HullZone? result = null;
		int num = 0;
		foreach (HullZone value in Enum.GetValues(typeof(HullZone)))
		{
			if (excluded != value && ship.State.GetShields(value) > num)
			{
				num = ship.State.GetShields(value);
				result = value;
			}
		}
		return result;
	}
}
