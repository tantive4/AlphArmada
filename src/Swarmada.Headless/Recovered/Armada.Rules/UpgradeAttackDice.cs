#pragma warning disable CS0649 // Reserved optional source resolver slot.
#nullable disable
using System;
using System.Collections.Generic;
using Armada.Core;
using Armada.Data;
using Swarmada.Headless.Engine;
using System.Numerics;
using Mathf = Swarmada.Headless.Engine.EngineMath;

namespace Armada.Rules;

public static class UpgradeAttackDice
{
	private enum Scope
	{
		Ship,
		Fleet
	}

	private sealed class Entry
	{
		public string CardId;

		public string CardName;

		public Scope Scope;

		public int Red;

		public int Blue;

		public int Black;

		public bool OncePerRound;

		public Func<AttackDiceContext, int> RedResolver;

		public Func<AttackDiceContext, int> BlueResolver;

		public Func<AttackDiceContext, int> BlackResolver;

		public Func<AttackDiceContext, bool> Condition;
	}

	private static readonly Entry[] Table = new Entry[11]
	{
		new Entry
		{
			CardId = "slaved-turrets",
			CardName = "Slaved Turrets",
			Scope = Scope.Ship,
			Red = 1,
			Condition = (AttackDiceContext context) => context.DefenderShip != null
		},
		new Entry
		{
			CardId = "quad-battery-turrets",
			CardName = "Quad Battery Turrets",
			Scope = Scope.Ship,
			Blue = 1,
			Condition = (AttackDiceContext context) => context.DefenderShip != null && context.Attacker != null && context.DefenderShip.State.CurrentSpeed > context.Attacker.State.CurrentSpeed
		},
		new Entry
		{
			CardId = "tf-1726-commander",
			CardName = "TF-1726",
			Scope = Scope.Fleet,
			BlackResolver = (AttackDiceContext context) => RaidRules.BlackDiceFromRaidTokens(context.DefenderShip),
			Condition = (AttackDiceContext context) => context.DefenderShip != null
		},
		new Entry
		{
			CardId = "mar-tuuk-commander",
			CardName = "Mar Tuuk",
			Scope = Scope.Fleet,
			Red = 1,
			Condition = (AttackDiceContext context) => context.DefenderShip != null && context.DefenderShip.State.GetShields(context.TargetZone) > 0
		},
		new Entry
		{
			CardId = "rapid-reload",
			CardName = "Rapid Reload",
			Scope = Scope.Ship,
			Black = 1,
			Condition = (AttackDiceContext context) => context.AttackZone == HullZone.Left || context.AttackZone == HullZone.Right
		},
		new Entry
		{
			CardId = "enhanced-armament",
			CardName = "Enhanced Armament",
			Scope = Scope.Ship,
			Red = 1,
			Condition = (AttackDiceContext context) => context.AttackZone == HullZone.Left || context.AttackZone == HullZone.Right
		},
		new Entry
		{
			CardId = "spinal-armament",
			CardName = "Spinal Armament",
			Scope = Scope.Ship,
			Red = 1,
			Condition = (AttackDiceContext context) => context.AttackZone == HullZone.Front || context.AttackZone == HullZone.Rear
		},
		new Entry
		{
			CardId = "high-capacity-ion-turbines",
			CardName = "High-Capacity Ion Turbines",
			Scope = Scope.Ship,
			Blue = 1,
			Condition = (AttackDiceContext context) => context.AttackZone == HullZone.Left || context.AttackZone == HullZone.Right
		},
		new Entry
		{
			CardId = "expanded-launchers",
			CardName = "Expanded Launchers",
			Scope = Scope.Ship,
			Black = 2,
			Condition = (AttackDiceContext context) => context.AttackZone == HullZone.Front
		},
		new Entry
		{
			CardId = "paragon-title",
			CardName = "Paragon",
			Scope = Scope.Ship,
			Black = 1,
			Condition = (AttackDiceContext context) => context.DefenderShip != null && context.DefenderPreviouslyAttacked
		},
		new Entry
		{
			CardId = "devastator-title",
			CardName = "Devastator",
			Scope = Scope.Ship,
			BlueResolver = (AttackDiceContext context) => CountDiscardedDefenseTokens(context.Attacker),
			OncePerRound = true,
			Condition = (AttackDiceContext context) => context.DefenderShip != null && context.AttackZone == HullZone.Front
		}
	};

	public static AttackDiceBonus Evaluate(AttackDiceContext context)
	{
		return Evaluate(context, commitUsage: true);
	}

	public static AttackDiceBonus Preview(AttackDiceContext context)
	{
		return Evaluate(context, commitUsage: false);
	}

	public static BatteryArmament EffectiveBatteryArmament(ShipView ship, HullZone zone)
	{
		if (ship?.Definition == null)
		{
			return default(BatteryArmament);
		}
		BatteryArmament battery = ship.Definition.GetBattery(zone);
		BatteryArmament batteryArmament = TitleRules.EffectiveBattery(ship, zone, battery);
		bool num = zone == HullZone.Left || zone == HullZone.Right;
		bool flag = zone == HullZone.Front || zone == HullZone.Rear;
		int num2 = batteryArmament.RedDice;
		int num3 = batteryArmament.BlueDice;
		int num4 = batteryArmament.BlackDice;
		if (num)
		{
			ShipLoadoutState loadout = ship.Loadout;
			if (loadout != null && loadout.HasActive("enhanced-armament"))
			{
				num2++;
			}
		}
		if (flag)
		{
			ShipLoadoutState loadout2 = ship.Loadout;
			if (loadout2 != null && loadout2.HasActive("spinal-armament"))
			{
				num2++;
			}
		}
		if (num)
		{
			ShipLoadoutState loadout3 = ship.Loadout;
			if (loadout3 != null && loadout3.HasActive("high-capacity-ion-turbines"))
			{
				num3++;
			}
		}
		if (num)
		{
			ShipLoadoutState loadout4 = ship.Loadout;
			if (loadout4 != null && loadout4.HasActive("rapid-reload"))
			{
				num4++;
			}
		}
		if (zone == HullZone.Front)
		{
			ShipLoadoutState loadout5 = ship.Loadout;
			if (loadout5 != null && loadout5.HasActive("expanded-launchers"))
			{
				num4 += 2;
			}
		}
		return new BatteryArmament(num2, num3, num4);
	}

	public static BatteryArmament BatteryArmamentIncrease(ShipView ship, HullZone zone)
	{
		if (ship?.Definition == null)
		{
			return default(BatteryArmament);
		}
		BatteryArmament battery = ship.Definition.GetBattery(zone);
		BatteryArmament batteryArmament = TitleRules.EffectiveBattery(ship, zone, battery);
		BatteryArmament batteryArmament2 = EffectiveBatteryArmament(ship, zone);
		return new BatteryArmament(Math.Max(0, batteryArmament2.RedDice - batteryArmament.RedDice), Math.Max(0, batteryArmament2.BlueDice - batteryArmament.BlueDice), Math.Max(0, batteryArmament2.BlackDice - batteryArmament.BlackDice));
	}

	private static AttackDiceBonus Evaluate(AttackDiceContext context, bool commitUsage)
	{
		if (context.Attacker == null)
		{
			return new AttackDiceBonus(0, 0, 0, null);
		}
		int num = 0;
		int num2 = 0;
		int num3 = 0;
		List<string> list = null;
		Entry[] table = Table;
		foreach (Entry entry in table)
		{
			UpgradeCardRuntimeState upgradeCardRuntimeState = FindPresentCard(context.Attacker, entry);
			if (upgradeCardRuntimeState != null && (entry.Condition == null || entry.Condition(context)))
			{
				int num4 = entry.Red + (entry.RedResolver?.Invoke(context) ?? 0);
				int num5 = entry.Blue + (entry.BlueResolver?.Invoke(context) ?? 0);
				int num6 = entry.Black + (entry.BlackResolver?.Invoke(context) ?? 0);
				if ((num4 != 0 || num5 != 0 || num6 != 0) && (!entry.OncePerRound || (context.RoundNumber > 0 && upgradeCardRuntimeState.LastUsedRound != context.RoundNumber && (!commitUsage || upgradeCardRuntimeState.TryUseThisRound(context.RoundNumber)))))
				{
					num += num4;
					num2 += num5;
					num3 += num6;
					(list ?? (list = new List<string>())).Add(entry.CardName);
				}
			}
		}
		return new AttackDiceBonus(num, num2, num3, list);
	}

	private static UpgradeCardRuntimeState FindPresentCard(ShipView attacker, Entry entry)
	{
		if (entry.Scope == Scope.Ship)
		{
			UpgradeCardRuntimeState upgradeCardRuntimeState = attacker.Loadout?.Find(entry.CardId);
			if (upgradeCardRuntimeState == null || !upgradeCardRuntimeState.Active)
			{
				return null;
			}
			return upgradeCardRuntimeState;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == attacker.OwnerId)
			{
				UpgradeCardRuntimeState upgradeCardRuntimeState2 = shipView.Loadout?.Find(entry.CardId);
				if (upgradeCardRuntimeState2 != null && upgradeCardRuntimeState2.Active)
				{
					return upgradeCardRuntimeState2;
				}
			}
		}
		return null;
	}

	private static int CountDiscardedDefenseTokens(ShipView ship)
	{
		if (ship?.State == null)
		{
			return 0;
		}
		int num = 0;
		foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
		{
			num += ship.State.GetDiscardedDefenseTokenCount(value);
		}
		return num;
	}

	public static bool LimitsToOneAttackPerRound(ShipView ship)
	{
		if ((object)ship == null)
		{
			return false;
		}
		return ship.Loadout?.HasActive("slaved-turrets") == true;
	}
}
