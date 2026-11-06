#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using Armada.Core;
using Armada.Data;
using Swarmada.Headless.Engine;
using System.Numerics;
using Mathf = Swarmada.Headless.Engine.EngineMath;

namespace Armada.Rules;

public static class TurbolaserRules
{
	public const string HeavyTurbolaserTurrets = "heavy-turbolaser-turrets";

	public const string Xi7Turbolasers = "xi7-turbolasers";

	public const string HeavyFireZone = "heavy-fire-zone";

	public const string SwivelMountBatteries = "swivel-mount-batteries";

	public static bool ConvertsFlakBlueToRed(ShipView attacker, SquadronView defender)
	{
		if ((object)attacker == null || attacker.Loadout?.HasActive("heavy-fire-zone") != true || defender == null)
		{
			return false;
		}
		SquadronView[] array = GameScope.Current.SceneSquadronViews;
		foreach (SquadronView squadronView in array)
		{
			if (!(squadronView == null) && !squadronView.IsDestroyed && squadronView.OwnerId == attacker.OwnerId && SquadronAbilityEngine.IsEngaged(defender, squadronView))
			{
				return false;
			}
		}
		return true;
	}

	public static BatteryArmament EffectiveFlakArmament(ShipView attacker, SquadronView defender, BatteryArmament armament, bool defendingMine = false)
	{
		bool num;
		if (!defendingMine)
		{
			num = ConvertsFlakBlueToRed(attacker, defender);
		}
		else
		{
			if ((object)attacker == null)
			{
				goto IL_0040;
			}
			num = attacker.Loadout?.HasActive("heavy-fire-zone") == true;
		}
		if (num)
		{
			return new BatteryArmament(armament.RedDice + armament.BlueDice, 0, armament.BlackDice);
		}
		goto IL_0040;
		IL_0040:
		return armament;
	}

	public static bool LimitsBraceReduction(ShipView attacker, bool braceSpent, int defenseTokenTypesSpent)
	{
		if ((object)attacker != null && attacker.Loadout?.HasActive("heavy-turbolaser-turrets") == true && braceSpent)
		{
			return defenseTokenTypesSpent > 1;
		}
		return false;
	}

	public static int MaximumRedirectDamagePerZone(ShipView attacker)
	{
		if ((object)attacker == null || attacker.Loadout?.HasActive("xi7-turbolasers") != true)
		{
			return int.MaxValue;
		}
		return 1;
	}

	public static string ApplySwivelMountModifier(ShipView attacker, HullZone attackZone, RangeBand range, ref int red, ref int blue, ref int black)
	{
		if (!IsSwivelMountAdjacentAttack(attacker, attackZone))
		{
			return string.Empty;
		}
		return "adjacent die removal requires player choice";
	}

	public static IReadOnlyList<AttackDieColor> GetSwivelMountFocusColors(ShipView ship, HullZone focusZone)
	{
		List<AttackDieColor> list = new List<AttackDieColor>();
		if (ship?.Definition == null)
		{
			return list;
		}
		bool flag = false;
		bool flag2 = false;
		bool flag3 = false;
		HullZone[] array = HullZoneRules.Adjacent(focusZone, ship.Definition.HasAuxiliaryHullZones);
		foreach (HullZone zone in array)
		{
			BatteryArmament batteryArmament = UpgradeAttackDice.EffectiveBatteryArmament(ship, zone);
			flag |= batteryArmament.RedDice > 0;
			flag2 |= batteryArmament.BlueDice > 0;
			flag3 |= batteryArmament.BlackDice > 0;
		}
		if (flag)
		{
			list.Add(AttackDieColor.Red);
		}
		if (flag2)
		{
			list.Add(AttackDieColor.Blue);
		}
		if (flag3)
		{
			list.Add(AttackDieColor.Black);
		}
		return list;
	}

	public static IReadOnlyList<AttackDieColor> GetSwivelMountFocusBonusColors(ShipView attacker, HullZone attackZone)
	{
		UpgradeCardRuntimeState upgradeCardRuntimeState = attacker?.Loadout?.Find("swivel-mount-batteries");
		if (upgradeCardRuntimeState == null || !upgradeCardRuntimeState.Active || !upgradeCardRuntimeState.Exhausted || upgradeCardRuntimeState.FocusZone != attackZone)
		{
			return Array.Empty<AttackDieColor>();
		}
		return GetSwivelMountFocusColors(attacker, attackZone);
	}

	public static AttackResolution ApplySwivelMountFocusBonus(ShipView attacker, HullZone attackZone, AttackDieColor selectedColor, AttackResolution attack, int seed, out AttackDieColor addedColor)
	{
		addedColor = (GetSwivelMountFocusBonusColors(attacker, attackZone).Contains(selectedColor) ? selectedColor : AttackDieColor.None);
		if (addedColor != AttackDieColor.None)
		{
			return AttackResolver.AddRolledDice(attack, addedColor, 1, seed);
		}
		return attack;
	}

	public static bool IsSwivelMountAdjacentAttack(ShipView attacker, HullZone attackZone)
	{
		UpgradeCardRuntimeState upgradeCardRuntimeState = attacker?.Loadout?.Find("swivel-mount-batteries");
		if (upgradeCardRuntimeState != null && upgradeCardRuntimeState.Active && upgradeCardRuntimeState.Exhausted && upgradeCardRuntimeState.FocusZone.HasValue)
		{
			return HullZoneRules.Adjacent(attackZone, attacker.Definition.HasAuxiliaryHullZones).Contains(upgradeCardRuntimeState.FocusZone.Value);
		}
		return false;
	}

	public static IReadOnlyList<AttackDieColor> GetSwivelMountRemovalColors(int red, int blue, int black)
	{
		List<AttackDieColor> list = new List<AttackDieColor>();
		if (red > 0)
		{
			list.Add(AttackDieColor.Red);
		}
		if (blue > 0)
		{
			list.Add(AttackDieColor.Blue);
		}
		if (black > 0)
		{
			list.Add(AttackDieColor.Black);
		}
		return list;
	}

	public static bool TryRemoveSwivelMountDie(AttackDieColor color, ref int red, ref int blue, ref int black)
	{
		switch (color)
		{
		case AttackDieColor.Red:
			if (red > 0)
			{
				red--;
				return true;
			}
			break;
		case AttackDieColor.Blue:
			if (blue > 0)
			{
				blue--;
				return true;
			}
			break;
		case AttackDieColor.Black:
			if (black > 0)
			{
				black--;
				return true;
			}
			break;
		}
		return false;
	}
}
