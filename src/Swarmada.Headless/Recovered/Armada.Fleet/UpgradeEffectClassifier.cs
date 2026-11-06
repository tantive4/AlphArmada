#nullable disable
using System;
using Armada.Data;

namespace Armada.Fleet;

public static class UpgradeEffectClassifier
{
	public static UpgradeEffectCategory Classify(UpgradeContentDescriptor upgrade)
	{
		if (upgrade == null)
		{
			throw new ArgumentNullException("upgrade");
		}
		UpgradeCatalog.TryGet(upgrade.Id, out var definition);
		string text = (upgrade.Slot ?? string.Empty).Trim().ToLowerInvariant();
		string value = string.Join(" ", upgrade.DisplayName, definition?.ability, definition?.startCommandType, definition?.exhaustType).ToLowerInvariant();
		switch (text)
		{
		case "commander":
		case "leader":
			return UpgradeEffectCategory.Command;
		case "fleet-command":
		case "fleet-support":
			return UpgradeEffectCategory.Comms;
		case "turbolaser":
		case "ion-cannon":
		case "ordnance":
		case "weapons-team":
		case "weapons-team-offensive-retro":
		case "super-weapon":
			return UpgradeEffectCategory.Weapons;
		case "defensive-retro":
			return UpgradeEffectCategory.Defense;
		default:
			if (HasAny(value, "navigate", "maneuver", "yaw", "speed dial", "change your speed", "engine tech"))
			{
				return UpgradeEffectCategory.Propulsion;
			}
			if (HasAny(value, "squadron", "fighter", "bomber", "rogue", "relay", "hangar", "activate up to"))
			{
				return UpgradeEffectCategory.Hangar;
			}
			if (HasAny(value, "command dial", "command token", "command value", "fleet command", "comms", "communications"))
			{
				return UpgradeEffectCategory.Comms;
			}
			if (HasAny(value, "defense token", "brace", "redirect", "evade", "scatter", "contain", "salvo", "shield"))
			{
				return UpgradeEffectCategory.Defense;
			}
			if (HasAny(value, "engineering", "repair", "damage card", "recover hull", "discard a facedown"))
			{
				return UpgradeEffectCategory.Engineering;
			}
			if (HasAny(value, "attack", "battery armament", "attack pool", "critical effect", "accuracy icon", "add a die", "reroll"))
			{
				return UpgradeEffectCategory.Weapons;
			}
			if (text == "support-team")
			{
				return UpgradeEffectCategory.Engineering;
			}
			if (text == "offensive-retro")
			{
				return UpgradeEffectCategory.Hangar;
			}
			return UpgradeEffectCategory.Systems;
		}
	}

	public static string LabelFor(UpgradeEffectCategory category)
	{
		return category switch
		{
			UpgradeEffectCategory.Command => "COMMAND",
			UpgradeEffectCategory.Comms => "COMMS",
			UpgradeEffectCategory.Weapons => "WEAPONS",
			UpgradeEffectCategory.Defense => "DEFENSE",
			UpgradeEffectCategory.Hangar => "HANGAR",
			UpgradeEffectCategory.Engineering => "ENGINEERING",
			UpgradeEffectCategory.Propulsion => "PROPULSION",
			_ => "SYSTEMS",
		};
	}

	public static int SortOrder(UpgradeEffectCategory category)
	{
		return (int)category;
	}

	private static bool HasAny(string value, params string[] terms)
	{
		foreach (string value2 in terms)
		{
			if (value.IndexOf(value2, StringComparison.Ordinal) >= 0)
			{
				return true;
			}
		}
		return false;
	}
}
