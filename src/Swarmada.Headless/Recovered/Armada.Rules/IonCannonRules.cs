#nullable disable
using System;
using Swarmada.Headless.Engine;

namespace Armada.Rules;

public static class IonCannonRules
{
	public const string Sw7IonBatteries = "sw-7-ion-batteries";

	public static int UnspentAccuracyDamage(ShipView attacker, int unspentAccuracies)
	{
		if ((object)attacker == null || attacker.Loadout?.HasActive("sw-7-ion-batteries") != true)
		{
			return 0;
		}
		return Math.Max(0, unspentAccuracies);
	}
}
