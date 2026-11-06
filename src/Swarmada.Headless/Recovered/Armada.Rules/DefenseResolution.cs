#nullable disable
using System;
using System.Collections.Generic;
using Armada.Core;

namespace Armada.Rules;

public readonly struct DefenseResolution
{
	public DefenseChoice Choice { get; }

	public int OriginalDamage { get; }

	public int DamageToPrimaryZone { get; }

	public int RedirectedDamage { get; }

	public HullZone RedirectZone { get; }

	public IReadOnlyDictionary<HullZone, int> RedirectedDamageByZone { get; }

	public int TotalDamage => DamageToPrimaryZone + RedirectedDamage;

	public DefenseResolution(DefenseChoice choice, int originalDamage, int damageToPrimaryZone, int redirectedDamage, HullZone redirectZone)
	{
		Choice = choice;
		OriginalDamage = Math.Max(0, originalDamage);
		DamageToPrimaryZone = Math.Max(0, damageToPrimaryZone);
		RedirectedDamage = Math.Max(0, redirectedDamage);
		RedirectZone = redirectZone;
		RedirectedDamageByZone = ((RedirectedDamage > 0) ? new Dictionary<HullZone, int> { { redirectZone, RedirectedDamage } } : new Dictionary<HullZone, int>());
	}

	public DefenseResolution(DefenseChoice choice, int originalDamage, int damageToPrimaryZone, IReadOnlyDictionary<HullZone, int> redirectedDamageByZone, HullZone fallbackRedirectZone)
	{
		Choice = choice;
		OriginalDamage = Math.Max(0, originalDamage);
		DamageToPrimaryZone = Math.Max(0, damageToPrimaryZone);
		Dictionary<HullZone, int> dictionary = new Dictionary<HullZone, int>();
		int num = 0;
		HullZone hullZone = fallbackRedirectZone;
		bool flag = false;
		if (redirectedDamageByZone != null)
		{
			foreach (KeyValuePair<HullZone, int> item in redirectedDamageByZone)
			{
				int num2 = Math.Max(0, item.Value);
				if (num2 > 0)
				{
					dictionary[item.Key] = num2;
					num += num2;
					if (!flag)
					{
						hullZone = item.Key;
						flag = true;
					}
				}
			}
		}
		RedirectedDamageByZone = dictionary;
		RedirectedDamage = num;
		RedirectZone = hullZone;
	}
}
