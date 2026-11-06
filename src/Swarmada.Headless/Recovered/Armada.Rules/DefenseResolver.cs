#nullable disable
using System;
using System.Collections.Generic;
using Armada.Core;

namespace Armada.Rules;

public static class DefenseResolver
{
	public static bool CanTargetWithAccuracy(ShipRuntimeState defender, DefenseTokenType token)
	{
		return defender?.HasDefenseToken(token) ?? false;
	}

	public static bool CanUse(ShipRuntimeState defender, DefenseChoice choice, RangeBand range, bool evadeAtCloseRange = false, bool allowAtSpeedZero = false)
	{
		if (defender == null)
		{
			return false;
		}
		if (choice != DefenseChoice.TakeHit && defender.CurrentSpeed <= 0 && !allowAtSpeedZero)
		{
			return false;
		}
		return choice switch
		{
			DefenseChoice.Brace => defender.CanResolveDefenseToken(DefenseTokenType.Brace),
			DefenseChoice.Redirect => defender.CanResolveDefenseToken(DefenseTokenType.Redirect),
			DefenseChoice.Evade => defender.CanResolveDefenseToken(DefenseTokenType.Evade),
			_ => true,
		};
	}

	public static DefenseResolution Resolve(ShipRuntimeState defender, HullZone attackedZone, int incomingDamage, DefenseChoice choice, RangeBand range, bool evadeAtCloseRange = false, bool allowAtSpeedZero = false)
	{
		int num = Math.Max(0, incomingDamage);
		if (!CanUse(defender, choice, range, evadeAtCloseRange, allowAtSpeedZero))
		{
			choice = DefenseChoice.TakeHit;
		}
		bool discarded;
		switch (choice)
		{
		case DefenseChoice.Brace:
			defender.TryResolveDefenseToken(DefenseTokenType.Brace, out discarded);
			return new DefenseResolution(choice, num, (num + 1) / 2, 0, attackedZone);
		case DefenseChoice.Redirect:
		{
			defender.TryResolveDefenseToken(DefenseTokenType.Redirect, out discarded);
			HullZone hullZone = defender.FindBestRedirectZone(attackedZone);
			int num2 = Math.Min(num, defender.GetShields(hullZone));
			return new DefenseResolution(choice, num, num - num2, num2, hullZone);
		}
		case DefenseChoice.Evade:
			defender.TryResolveDefenseToken(DefenseTokenType.Evade, out discarded);
			return new DefenseResolution(choice, num, Math.Max(0, num - 1), 0, attackedZone);
		default:
			return new DefenseResolution(DefenseChoice.TakeHit, num, num, 0, attackedZone);
		}
	}

	public static DefenseResolution ReduceTotalDamage(DefenseResolution defense, int amount)
	{
		int num = Math.Max(0, amount);
		int damageToPrimaryZone = defense.DamageToPrimaryZone;
		int num2 = Math.Min(damageToPrimaryZone, num);
		damageToPrimaryZone -= num2;
		num -= num2;
		Dictionary<HullZone, int> dictionary = new Dictionary<HullZone, int>();
		foreach (KeyValuePair<HullZone, int> item in defense.RedirectedDamageByZone)
		{
			int value = item.Value;
			int num3 = Math.Min(value, num);
			value -= num3;
			num -= num3;
			if (value > 0)
			{
				dictionary[item.Key] = value;
			}
		}
		return new DefenseResolution(defense.Choice, defense.OriginalDamage, damageToPrimaryZone, dictionary, defense.RedirectZone);
	}

	public static DefenseResolution ResolveSelectedEffects(ShipRuntimeState defender, HullZone attackedZone, int incomingDamage, bool brace, bool redirect, bool evade, int maximumRedirectZones = 1, bool allowNonAdjacentRedirect = false, int maximumDamagePerRedirectZone = int.MaxValue)
	{
		int num = Math.Max(0, incomingDamage);
		if (brace)
		{
			num = (num + 1) / 2;
		}
		HullZone fallbackRedirectZone = attackedZone;
		Dictionary<HullZone, int> dictionary = new Dictionary<HullZone, int>();
		if (redirect && num > 0)
		{
			List<HullZone> list = RedirectCandidates(attackedZone, allowNonAdjacentRedirect, defender.HasAuxiliaryHullZones);
			list.Sort((HullZone left, HullZone right) => defender.GetShields(right).CompareTo(defender.GetShields(left)));
			int num2 = num;
			int num3 = Math.Max(1, maximumRedirectZones);
			int val = Math.Max(0, maximumDamagePerRedirectZone);
			for (int num4 = 0; num4 < list.Count; num4++)
			{
				if (dictionary.Count >= num3)
				{
					break;
				}
				if (num2 <= 0)
				{
					break;
				}
				HullZone hullZone = list[num4];
				int num5 = Math.Min(num2, Math.Min(defender.GetShields(hullZone), val));
				if (num5 > 0)
				{
					dictionary[hullZone] = num5;
					fallbackRedirectZone = hullZone;
					num2 -= num5;
				}
			}
		}
		return new DefenseResolution((!brace) ? (redirect ? DefenseChoice.Redirect : (evade ? DefenseChoice.Evade : DefenseChoice.TakeHit)) : DefenseChoice.Brace, incomingDamage, num - SumRedirects(dictionary), dictionary, fallbackRedirectZone);
	}

	public static DefenseResolution ResolveSelectedEffectsWithRedirects(ShipRuntimeState defender, HullZone attackedZone, int incomingDamage, bool brace, bool redirect, bool evade, IReadOnlyDictionary<HullZone, int> requestedRedirects, HullZone fallbackRedirectZone, int maximumRedirectZones = 1, bool allowNonAdjacentRedirect = false, int maximumDamagePerRedirectZone = int.MaxValue)
	{
		int num = Math.Max(0, incomingDamage);
		if (brace)
		{
			num = (num + 1) / 2;
		}
		Dictionary<HullZone, int> dictionary = new Dictionary<HullZone, int>();
		HullZone hullZone = attackedZone;
		if (redirect && requestedRedirects != null)
		{
			List<HullZone> list = RedirectCandidates(attackedZone, allowNonAdjacentRedirect, defender.HasAuxiliaryHullZones);
			int num2 = Math.Max(1, maximumRedirectZones);
			int val = Math.Max(0, maximumDamagePerRedirectZone);
			int num3 = num;
			int num4 = 0;
			foreach (KeyValuePair<HullZone, int> requestedRedirect in requestedRedirects)
			{
				if (num4 < num2 && list.Contains(requestedRedirect.Key))
				{
					num4++;
					if (hullZone == attackedZone)
					{
						hullZone = requestedRedirect.Key;
					}
					int num5 = Math.Min(num3, Math.Min(defender.GetShields(requestedRedirect.Key), Math.Min(Math.Max(0, requestedRedirect.Value), val)));
					if (num5 > 0)
					{
						dictionary[requestedRedirect.Key] = num5;
						num3 -= num5;
					}
				}
			}
			if (list.Contains(fallbackRedirectZone))
			{
				hullZone = fallbackRedirectZone;
			}
		}
		return new DefenseResolution((!brace) ? (redirect ? DefenseChoice.Redirect : (evade ? DefenseChoice.Evade : DefenseChoice.TakeHit)) : DefenseChoice.Brace, incomingDamage, num - SumRedirects(dictionary), dictionary, hullZone);
	}

	public static IReadOnlyList<HullZone> GetRedirectCandidates(HullZone attackedZone, bool includeNonAdjacent = false, bool hasAuxiliaryHullZones = false)
	{
		return RedirectCandidates(attackedZone, includeNonAdjacent, hasAuxiliaryHullZones);
	}

	private static List<HullZone> RedirectCandidates(HullZone attackedZone, bool includeNonAdjacent, bool hasAuxiliaryHullZones = false)
	{
		List<HullZone> list = new List<HullZone>(HullZoneRules.Adjacent(attackedZone, hasAuxiliaryHullZones));
		if (includeNonAdjacent)
		{
			foreach (HullZone item in HullZoneRules.Zones(hasAuxiliaryHullZones))
			{
				if (item != attackedZone && !list.Contains(item))
				{
					list.Add(item);
				}
			}
		}
		return list;
	}

	private static int SumRedirects(IReadOnlyDictionary<HullZone, int> redirects)
	{
		int num = 0;
		if (redirects == null)
		{
			return num;
		}
		foreach (KeyValuePair<HullZone, int> redirect in redirects)
		{
			num += Math.Max(0, redirect.Value);
		}
		return num;
	}
}
