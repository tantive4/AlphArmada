#nullable disable
using System;
using System.Collections.Generic;
using Armada.Core;

namespace Armada.Rules;

public static class DefenseTokenInstanceChoiceRules
{
	public static IReadOnlyList<DefenseTokenInstance> GetInstances(ShipRuntimeState defender, DefenseTokenType token)
	{
		if (defender == null)
		{
			return Array.Empty<DefenseTokenInstance>();
		}
		List<DefenseTokenInstance> list = new List<DefenseTokenInstance>();
		foreach (DefenseTokenInstance availableInstance in ForcedDefenseTokenDiscardRules.GetAvailableInstances(defender))
		{
			if (availableInstance.Token == token)
			{
				list.Add(availableInstance);
			}
		}
		return list;
	}

	public static IReadOnlyList<DefenseTokenInstance> GetAccuracyTargets(ShipRuntimeState defender, DefenseTokenType token, ISet<DefenseTokenInstance> alreadyLocked)
	{
		List<DefenseTokenInstance> list = new List<DefenseTokenInstance>();
		foreach (DefenseTokenInstance instance in GetInstances(defender, token))
		{
			if (alreadyLocked == null || !alreadyLocked.Contains(instance))
			{
				list.Add(instance);
			}
		}
		return list;
	}

	public static IReadOnlyList<DefenseTokenInstance> GetSpendableInstances(ShipRuntimeState defender, DefenseTokenType token, ISet<DefenseTokenInstance> accuracyLocked)
	{
		List<DefenseTokenInstance> list = new List<DefenseTokenInstance>();
		foreach (DefenseTokenInstance instance in GetInstances(defender, token))
		{
			if ((accuracyLocked == null || !accuracyLocked.Contains(instance)) && (!instance.Exhausted || defender.CanSpendExhaustedDefenseTokens))
			{
				list.Add(instance);
			}
		}
		return list;
	}

	public static bool RequiresExplicitChoice(IReadOnlyList<DefenseTokenInstance> candidates, DefenseTokenInstance? distinguishedToken = null)
	{
		if (candidates == null || candidates.Count <= 1)
		{
			return false;
		}
		bool exhausted = candidates[0].Exhausted;
		for (int i = 1; i < candidates.Count; i++)
		{
			if (candidates[i].Exhausted != exhausted)
			{
				return true;
			}
		}
		if (!distinguishedToken.HasValue)
		{
			return false;
		}
		bool flag = false;
		bool flag2 = false;
		foreach (DefenseTokenInstance candidate in candidates)
		{
			if (candidate == distinguishedToken.Value)
			{
				flag = true;
			}
			else
			{
				flag2 = true;
			}
		}
		return flag && flag2;
	}
}
