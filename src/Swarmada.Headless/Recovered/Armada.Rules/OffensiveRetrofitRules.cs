#nullable disable
using Armada.Data;
using Swarmada.Headless.Engine;

namespace Armada.Rules;

public static class OffensiveRetrofitRules
{
	public const string HardenedBulkheads = "hardened-bulkheads";

	public const string FlakGuns = "flak-guns";

	public const string QuadLaserTurrets = "quad-laser-turrets";

	public static bool PreventsCollisionDamage(ShipView protectedShip, ShipView otherShip)
	{
		if ((object)protectedShip != null && protectedShip.Loadout?.HasActive("hardened-bulkheads") == true)
		{
			return SizeRank(protectedShip.SizeClass) > SizeRank(otherShip?.SizeClass);
		}
		return false;
	}

	public static BatteryArmament EffectiveAntiSquadronArmament(ShipView ship, BatteryArmament printed)
	{
		if ((object)ship == null || ship.Loadout?.HasActive("flak-guns") != true)
		{
			return printed;
		}
		return new BatteryArmament(0, 0, printed.TotalDice);
	}

	public static bool GrantsCounterOneAgainstSquadron(ShipView defender, SquadronView attacker)
	{
		if ((object)defender != null && defender.Loadout?.HasActive("quad-laser-turrets") == true && attacker != null)
		{
			return SquadronAbilityEngine.HullZoneDistance(attacker, defender, defender.DetermineFiringZone(attacker.transform.position)) <= 3.369803f;
		}
		return false;
	}

	private static int SizeRank(string sizeClass)
	{
		return sizeClass?.Trim().ToLowerInvariant() switch
		{
			"small" => 1,
			"medium" => 2,
			"large" => 3,
			"huge" => 4,
			_ => 0,
		};
	}
}
