#nullable disable
using System;
using System.Collections.Generic;
using Swarmada.Headless.Engine;

namespace Armada.Rules;

public static class FleetCommandRules
{
	public const string AllFightersFollowMe = "all-fighters-follow-me";

	public const string EntrapmentFormation = "entrapment-formation";

	public const string IntensifyFirepower = "intensify-firepower";

	public const string ShieldsToMaximum = "shields-to-maximum";

	public const string TakeEvasiveAction = "take-evasive-action";

	public const string StartActionPrefix = "fleet-command.start.";

	public const string SpeedActionPrefix = "fleet-command.speed.";

	public const string ShieldActionPrefix = "fleet-command.shield.";

	public const string IntensifyAction = "fleet-command.attack.intensify-firepower";

	private static ScopedEffectSet ActiveEffects => new ScopedEffectSet(GameScope.Current.Flow.Flags, "fleet-command.");

	public static void ResetRound()
	{
		ActiveEffects.Clear();
	}

	public static bool Activate(int ownerId, string cardId)
	{
		if (ownerId > 0 && IsSupported(cardId))
		{
			return ActiveEffects.Add(Key(ownerId, cardId));
		}
		return false;
	}

	public static bool IsActive(int ownerId, string cardId)
	{
		if (ownerId > 0 && !string.IsNullOrWhiteSpace(cardId))
		{
			return ActiveEffects.Contains(Key(ownerId, cardId));
		}
		return false;
	}

	public static bool IncreasesCommandedSquadronSpeed(SquadronView squadron)
	{
		if ((object)squadron != null && squadron.State?.Commanded == true)
		{
			return IsActive(TitleRules.FindCommandingShip(squadron)?.OwnerId ?? squadron.OwnerId, "all-fighters-follow-me");
		}
		return false;
	}

	public static bool AllowsDetermineCourseSpeedChange(ShipView ship)
	{
		if (ship != null)
		{
			return IsActive(ship.OwnerId, "entrapment-formation");
		}
		return false;
	}

	public static bool AllowsSingleHitChange(ShipView ship)
	{
		if (ship != null)
		{
			return IsActive(ship.OwnerId, "intensify-firepower");
		}
		return false;
	}

	public static bool AllowsPreRevealShieldRecovery(ShipView ship)
	{
		if (ship != null)
		{
			return IsActive(ship.OwnerId, "shields-to-maximum");
		}
		return false;
	}

	public static bool IncreasesFinalYaw(ShipView ship)
	{
		if (ship != null)
		{
			return IsActive(ship.OwnerId, "take-evasive-action");
		}
		return false;
	}

	public static bool IsSupported(string cardId)
	{
		switch (cardId)
		{
		default:
			return cardId == "take-evasive-action";
		case "all-fighters-follow-me":
		case "entrapment-formation":
		case "intensify-firepower":
		case "shields-to-maximum":
			return true;
		}
	}

	public static ShipCommand RequiredToken(string cardId)
	{
		return cardId switch
		{
			"all-fighters-follow-me" => ShipCommand.Squadron,
			"entrapment-formation" => ShipCommand.Navigate,
			"intensify-firepower" => ShipCommand.ConcentrateFire,
			"shields-to-maximum" => ShipCommand.Engineering,
			"take-evasive-action" => ShipCommand.Navigate,
			_ => ShipCommand.Navigate,
		};
	}

	private static string Key(int ownerId, string cardId)
	{
		return ownerId + ":" + cardId;
	}
}
