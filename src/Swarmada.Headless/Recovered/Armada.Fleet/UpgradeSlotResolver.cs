#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Fleet;

public static class UpgradeSlotResolver
{
	public const string WeaponsTeamOffensiveRetrofit = "weapons-team-offensive-retro";

	private static readonly HashSet<string> FullyDataDrivenSlotCards = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "invincible-title", "radiant-vii-title", "minister-tua-officer" };

	private static readonly Dictionary<string, string[]> CompositeRequirements = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase) {
	{
		"weapons-team-offensive-retro",
		new string[2] { "weapons-team", "offensive-retro" }
	} };

	public static UpgradeSlotResolution Resolve(ShipContentDescriptor ship, IReadOnlyList<UpgradeContentDescriptor> upgrades, string shipPath)
	{
		if (ship == null)
		{
			throw new ArgumentNullException("ship");
		}
		string text = shipPath ?? string.Empty;
		Dictionary<string, int> dictionary = CountBaseSlots(ship.UpgradeSlots);
		Dictionary<string, int> dictionary2 = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
		HashSet<string> hashSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
		List<FleetLegalityIssue> list = new List<FleetLegalityIssue>();
		IReadOnlyList<UpgradeContentDescriptor> readOnlyList = upgrades ?? Array.Empty<UpgradeContentDescriptor>();
		for (int i = 0; i < readOnlyList.Count; i++)
		{
			UpgradeContentDescriptor upgradeContentDescriptor = readOnlyList[i];
			if (upgradeContentDescriptor == null)
			{
				continue;
			}
			foreach (string disallowedSlot in upgradeContentDescriptor.DisallowedSlots)
			{
				if (dictionary.TryGetValue(disallowedSlot, out var value) && value > 0)
				{
					list.Add(Error("upgrade.slot.disallowed", UpgradePath(text, i), upgradeContentDescriptor.DisplayName + " cannot be equipped to a ship that already has a " + disallowedSlot + " slot."));
				}
			}
		}
		foreach (UpgradeContentDescriptor item in readOnlyList)
		{
			if (item == null)
			{
				continue;
			}
			if (FullyDataDrivenSlotCards.Contains(item.Id) && item.EnabledSlots.Count == 0 && item.DisabledSlots.Count == 0)
			{
				list.Add(Error("upgrade.slot.unavailable", text, item.DisplayName + " is missing its required slot-effect metadata."));
			}
			foreach (string enabledSlot in item.EnabledSlots)
			{
				AddSlot(dictionary, enabledSlot, 1);
			}
			foreach (string disabledSlot in item.DisabledSlots)
			{
				hashSet.Add(disabledSlot);
			}
		}
		foreach (string item2 in hashSet)
		{
			dictionary[item2] = 0;
		}
		for (int j = 0; j < readOnlyList.Count; j++)
		{
			UpgradeContentDescriptor upgradeContentDescriptor2 = readOnlyList[j];
			if (upgradeContentDescriptor2 == null)
			{
				continue;
			}
			IReadOnlyList<string> readOnlyList2 = RequirementsFor(upgradeContentDescriptor2.Slot);
			if (readOnlyList2.Count == 0)
			{
				list.Add(Error("upgrade.slot.unavailable", UpgradePath(text, j), upgradeContentDescriptor2.DisplayName + " has no supported upgrade slot metadata."));
				continue;
			}
			bool flag = true;
			foreach (string item3 in readOnlyList2)
			{
				if (hashSet.Contains(item3))
				{
					list.Add(Error("upgrade.slot.disabled", UpgradePath(text, j), upgradeContentDescriptor2.DisplayName + " requires the " + item3 + " slot, but another selected upgrade disables it."));
					flag = false;
					continue;
				}
				dictionary.TryGetValue(item3, out var value2);
				dictionary2.TryGetValue(item3, out var value3);
				if (value3 >= value2)
				{
					list.Add(Error("upgrade.slot.unavailable", UpgradePath(text, j), ship.DisplayName + " has no unused " + item3 + " slot for " + upgradeContentDescriptor2.DisplayName + "."));
					flag = false;
				}
			}
			if (!flag)
			{
				continue;
			}
			foreach (string item4 in readOnlyList2)
			{
				AddSlot(dictionary2, item4, 1);
			}
		}
		return new UpgradeSlotResolution(dictionary, dictionary2, list);
	}

	public static IReadOnlyList<string> RequirementsFor(string slot)
	{
		string text = FleetContentText.Normalize(slot);
		if (text.Length == 0)
		{
			return Array.Empty<string>();
		}
		if (CompositeRequirements.TryGetValue(text, out var value))
		{
			return (string[])value.Clone();
		}
		return new string[1] { text };
	}

	private static Dictionary<string, int> CountBaseSlots(IReadOnlyList<string> slots)
	{
		Dictionary<string, int> dictionary = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
		if (slots == null)
		{
			return dictionary;
		}
		foreach (string slot in slots)
		{
			string text = FleetContentText.Normalize(slot);
			if (text.Length != 0 && !CompositeRequirements.ContainsKey(text))
			{
				AddSlot(dictionary, text, 1);
			}
		}
		return dictionary;
	}

	private static void AddSlot(Dictionary<string, int> slots, string rawSlot, int amount)
	{
		string text = FleetContentText.Normalize(rawSlot);
		if (text.Length != 0 && amount != 0)
		{
			slots.TryGetValue(text, out var value);
			slots[text] = Math.Max(0, value + amount);
		}
	}

	private static string UpgradePath(string shipPath, int index)
	{
		return $"{shipPath}.upgradeIds[{index}]";
	}

	private static FleetLegalityIssue Error(string code, string path, string message)
	{
		return new FleetLegalityIssue(code, FleetLegalitySeverity.Error, path, message);
	}
}
