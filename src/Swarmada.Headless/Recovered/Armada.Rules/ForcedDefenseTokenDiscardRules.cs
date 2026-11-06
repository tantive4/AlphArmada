#nullable disable
using System;
using System.Collections.Generic;
using Armada.Core;

namespace Armada.Rules;

public static class ForcedDefenseTokenDiscardRules
{
	public const int ChoiceButtonCount = 4;

	public const int PagedTokenButtonCount = 3;

	public static string BuildChoiceId(string prefix, DefenseTokenInstance token)
	{
		return (prefix ?? string.Empty) + (int)token.Token + "." + token.Slot + "." + (token.Exhausted ? "1" : "0");
	}

	public static bool TryParseChoiceId(string choiceId, string prefix, out DefenseTokenInstance token)
	{
		token = default(DefenseTokenInstance);
		if (string.IsNullOrWhiteSpace(choiceId) || string.IsNullOrWhiteSpace(prefix) || !choiceId.StartsWith(prefix, StringComparison.Ordinal))
		{
			return false;
		}
		string[] array = choiceId.Substring(prefix.Length).Split('.');
		if (array.Length != 3 || !int.TryParse(array[0], out var result) || !Enum.IsDefined(typeof(DefenseTokenType), result) || !int.TryParse(array[1], out var result2) || result2 < 0 || (array[2] != "0" && array[2] != "1"))
		{
			return false;
		}
		token = new DefenseTokenInstance((DefenseTokenType)result, result2, array[2] == "1");
		return true;
	}

	public static IReadOnlyList<DefenseTokenInstance> GetAvailableInstances(ShipRuntimeState defender)
	{
		if (defender == null)
		{
			return Array.Empty<DefenseTokenInstance>();
		}
		List<DefenseTokenInstance> result = new List<DefenseTokenInstance>();
		foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
		{
			AddAvailableInstances(result, defender.GetDefenseTokenInstances(value));
		}
		return result;
	}

	public static IReadOnlyList<DefenseTokenInstance> GetAvailableInstances(SquadronRuntimeState defender)
	{
		if (defender == null)
		{
			return Array.Empty<DefenseTokenInstance>();
		}
		List<DefenseTokenInstance> result = new List<DefenseTokenInstance>();
		foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
		{
			AddAvailableInstances(result, defender.GetDefenseTokenInstances(value));
		}
		return result;
	}

	public static bool IsAvailable(ShipRuntimeState defender, DefenseTokenInstance token)
	{
		return defender?.IsDefenseTokenInstanceAvailable(token) ?? false;
	}

	public static bool IsAvailable(SquadronRuntimeState defender, DefenseTokenInstance token)
	{
		return defender?.IsDefenseTokenInstanceAvailable(token) ?? false;
	}

	public static bool TryDiscard(ShipRuntimeState defender, DefenseTokenInstance token)
	{
		if (IsAvailable(defender, token))
		{
			return defender.DiscardDefenseToken(token);
		}
		return false;
	}

	public static bool TryDiscard(SquadronRuntimeState defender, DefenseTokenInstance token)
	{
		if (IsAvailable(defender, token))
		{
			return defender.DiscardDefenseToken(token);
		}
		return false;
	}

	public static bool TryDiscard(ShipRuntimeState defender, DefenseTokenType token)
	{
		return defender?.DiscardDefenseToken(token) ?? false;
	}

	public static bool TryDiscard(SquadronRuntimeState defender, DefenseTokenType token)
	{
		return defender?.DiscardDefenseToken(token) ?? false;
	}

	public static bool UsesChoicePaging(int choiceCount)
	{
		return choiceCount > 4;
	}

	public static int GetChoicePageCount(int choiceCount)
	{
		if (!UsesChoicePaging(choiceCount))
		{
			return (choiceCount > 0) ? 1 : 0;
		}
		return (choiceCount + 3 - 1) / 3;
	}

	public static int GetVisibleChoiceCount(int choiceCount, int page)
	{
		if (!UsesChoicePaging(choiceCount))
		{
			return Math.Max(0, choiceCount);
		}
		int num = Math.Max(0, page) * 3;
		return Math.Clamp(choiceCount - num, 0, 3);
	}

	public static int ResolveVisibleChoiceIndex(int choiceCount, int page, int buttonIndex)
	{
		int visibleChoiceCount = GetVisibleChoiceCount(choiceCount, page);
		if (buttonIndex < 0 || buttonIndex >= visibleChoiceCount)
		{
			return -1;
		}
		if (!UsesChoicePaging(choiceCount))
		{
			return buttonIndex;
		}
		return Math.Max(0, page) * 3 + buttonIndex;
	}

	public static int GetNextChoicePage(int choiceCount, int page)
	{
		int choicePageCount = GetChoicePageCount(choiceCount);
		if (choicePageCount > 1)
		{
			return (Math.Max(0, page) + 1) % choicePageCount;
		}
		return 0;
	}

	private static void AddAvailableInstances(ICollection<DefenseTokenInstance> result, IReadOnlyList<DefenseTokenInstance> instances)
	{
		if (instances == null)
		{
			return;
		}
		foreach (DefenseTokenInstance instance in instances)
		{
			result.Add(instance);
		}
	}
}
