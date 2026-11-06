#nullable disable
using System.Collections.Generic;
using Swarmada.Headless.Engine;
using System.Numerics;
using Mathf = Swarmada.Headless.Engine.EngineMath;

namespace Armada.Rules;

public static class CommanderRules
{
	public const string Madine = "general-madine-commander";

	public const string Cracken = "general-cracken-commander";

	public const string Motti = "admiral-motti-commander";

	public const string Romodi = "general-romodi-commander";

	public const string GarmBelIblis = "garm-bel-iblis-commander";

	public const string Tagge = "general-tagge-commander";

	public const string ShuMai = "shu-mai-officer";

	public const string AdmiralOzzel = "admiral-ozzel-commander";

	public const string AdmiralYularen = "admiral-yularen-commander";

	public const int AdmiralYularenRepairCost = 2;

	public const float AdmiralYularenRepairDistance = 8.09628f;

	public const string LeiaOrgana = "leia-organa-commander";

	public const string MonMothma = "mon-mothma-commander";

	public const string ObiWanKenobi = "obi-wan-kenobi-commander";

	public const string PloKoon = "plo-koon-commander";

	public const string GrandMoffTarkin = "grand-moff-tarkin-commander";

	public const string AdmiralAckbar = "admiral-ackbar-commander";

	public const string DarthVader = "darth-vader-commander";

	public const string Kraken = "kraken-commander";

	public const string AdmiralPiett = "admiral-piett-commander";

	public const string AdmiralScreed = "admiral-screed-commander";

	public const string AdmiralAckbarAction = "commander.admiral-ackbar.commit-broadsides";

	public static readonly string[] RoundScheduledCards = new string[3] { "garm-bel-iblis-commander", "general-tagge-commander", "shu-mai-officer" };

	public static readonly int[] DefaultScheduledRounds = new int[2] { 2, 4 };

	public static int AttackRerollsFromSchedule(ShipView attacker, int roundNumber)
	{
		if (attacker?.Loadout == null)
		{
			return 0;
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState = attacker.Loadout.Find("shu-mai-officer");
		if (upgradeCardRuntimeState == null || !upgradeCardRuntimeState.Active || !upgradeCardRuntimeState.IsScheduledFor(roundNumber))
		{
			return 0;
		}
		return 3;
	}

	public static PassiveEffectNotice ObstructionBecomesAdvantage(ShipView attacker)
	{
		if (attacker == null || !FleetHasCommander(attacker, "general-romodi-commander"))
		{
			return PassiveEffectNotice.None;
		}
		return PassiveEffectNotice.Battlefield(attacker, "[COMMANDER] General Romodi: the obstructed attack keeps its die and gains 1 red die.", "GENERAL ROMODI  -  OBSTRUCTION IGNORED,  +1 RED");
	}

	public static bool GrantsExtraManeuverYaw(ShipView ship)
	{
		return FleetHasCommander(ship, "general-madine-commander");
	}

	public static bool GrantsDetermineCourseSpeedChange(ShipView ship)
	{
		return FleetHasCommander(ship, "admiral-ozzel-commander");
	}

	public static CommandSpendMode UpgradeDialSpendMode(ShipView ship, CommandSpendMode mode)
	{
		return mode;
	}

	public static int AdditionalSquadronActivations(ShipView ship, CommandSpendMode mode)
	{
		if (!CommandSpendRules.UsesToken(mode) || !FleetHasCommander(ship, "admiral-yularen-commander"))
		{
			return 0;
		}
		return 1;
	}

	public static bool CanAdmiralYularenRepairSquadron(ShipView source, SquadronView target, int engineeringPoints)
	{
		if (source != null && !source.IsDestroyed && target != null && !target.IsDestroyed && target.State != null && target.Definition != null && engineeringPoints >= 2 && target.OwnerId == source.OwnerId && target.State.CurrentHull < target.Definition.Hull && FleetHasCommander(source, "admiral-yularen-commander"))
		{
			return SquadronAbilityEngine.FootprintDistance(source, target) <= 8.09628f;
		}
		return false;
	}

	public static ShipView FindCloseRangeEvadeCommander(ShipView ship)
	{
		if (ship == null || ship.OwnerId <= 0)
		{
			return null;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == ship.OwnerId)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive("mon-mothma-commander"))
				{
					return shipView;
				}
			}
		}
		return null;
	}

	public static bool EnhancesCloseRangeEvade(ShipView ship)
	{
		if (ship != null && ship.OwnerId > 0)
		{
			return FleetHasCommanderForOwner(ship.OwnerId, "mon-mothma-commander");
		}
		return false;
	}

	public static PassiveEffectNotice ReduceDamageAfterRedirect(ShipView defender, ref DefenseResolution defense, bool redirectSpent = false)
	{
		if (defender == null || (!redirectSpent && defense.Choice != DefenseChoice.Redirect) || defense.TotalDamage <= 0 || !FleetHasCommander(defender, "obi-wan-kenobi-commander"))
		{
			return PassiveEffectNotice.None;
		}
		defense = DefenseResolver.ReduceTotalDamage(defense, 1);
		return PassiveEffectNotice.Battlefield(defender, "[COMMANDER] Obi-Wan Kenobi reduced incoming damage by 1 after a readied Redirect token was spent.", "OBI-WAN KENOBI  -  DAMAGE REDUCED");
	}

	public static bool GrantsPloKoonAccuracy(SquadronView attacker, ShipView defender)
	{
		if (attacker == null || defender == null || !FleetHasCommanderForOwner(attacker.OwnerId, "plo-koon-commander"))
		{
			return false;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == attacker.OwnerId && RaidRules.FootprintDistance(shipView, defender) <= 8.227571f)
			{
				return true;
			}
		}
		return false;
	}

	public static bool GrantsPloKoonGrit(SquadronView squadron, IEnumerable<SquadronView> allSquadrons)
	{
		if (squadron == null || SquadronAbilityEngine.HasKeyword(squadron.Definition, "ADEPT") || !FleetHasCommanderForOwner(squadron.OwnerId, "plo-koon-commander") || allSquadrons == null)
		{
			return false;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !allSquadron.IsDestroyed && allSquadron.OwnerId == squadron.OwnerId && SquadronAbilityEngine.GetKeywordValue(allSquadron.Definition, "ADEPT") > 0 && SquadronAbilityEngine.FootprintDistance(squadron, allSquadron) <= 3.369803f)
			{
				return true;
			}
		}
		return false;
	}

	public static PassiveEffectNotice DefenderIsTreatedAsObstructed(ShipView defender)
	{
		if (defender == null || defender.IsDestroyed)
		{
			return PassiveEffectNotice.None;
		}
		if (defender.State == null || defender.State.CurrentSpeed < 3)
		{
			return PassiveEffectNotice.None;
		}
		if (!IsSmallOrMedium(defender))
		{
			return PassiveEffectNotice.None;
		}
		if (!FleetHasCommander(defender, "general-cracken-commander"))
		{
			return PassiveEffectNotice.None;
		}
		return PassiveEffectNotice.Battlefield(defender, "[COMMANDER] General Cracken treats this attack as obstructed; the attacker removes one die.", "GENERAL CRACKEN  -  ATTACK OBSTRUCTED");
	}

	public static int HullBonus(ShipView ship)
	{
		if (!FleetHasCommander(ship, "admiral-motti-commander"))
		{
			return 0;
		}
		return SizeOf(ship) switch
		{
			"small" => 1,
			"medium" => 2,
			"large" => 3,
			_ => 0,
		};
	}

	private static bool IsSmallOrMedium(ShipView ship)
	{
		string text = SizeOf(ship);
		if (!(text == "small"))
		{
			return text == "medium";
		}
		return true;
	}

	private static string SizeOf(ShipView ship)
	{
		return (ship?.SizeClass ?? string.Empty).Trim().ToLowerInvariant();
	}

	public static bool FleetHasCommander(ShipView member, string cardId)
	{
		return FindFleetUpgrade(member, cardId) != null;
	}

	public static UpgradeCardRuntimeState FindFleetUpgrade(ShipView member, string cardId)
	{
		if (member == null || string.IsNullOrEmpty(cardId))
		{
			return null;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == member.OwnerId)
			{
				UpgradeCardRuntimeState upgradeCardRuntimeState = shipView.Loadout?.Find(cardId);
				if (upgradeCardRuntimeState != null && upgradeCardRuntimeState.Active)
				{
					return upgradeCardRuntimeState;
				}
			}
		}
		return null;
	}

	private static bool FleetHasCommanderForOwner(int ownerId, string cardId)
	{
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == ownerId)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive(cardId))
				{
					return true;
				}
			}
		}
		return false;
	}
}
