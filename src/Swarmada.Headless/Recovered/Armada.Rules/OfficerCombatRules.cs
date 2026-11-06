#nullable disable
using Swarmada.Headless.Engine;
using System.Numerics;
using Swarmada.Headless.Geometry;
using Mathf = Swarmada.Headless.Engine.EngineMath;

namespace Armada.Rules;

public static class OfficerCombatRules
{
	public const string AdmiralMontferrat = "admiral-montferrat-officer";

	public const string CommanderGherant = "commander-gherant-officer";

	public const string DamageControlOfficer = "damage-control-officer";

	public const string ReevaDemesne = "reeva-demesne-officer";

	public const string LandoCalrissian = "lando-calrissian-officer";

	public const string CaptainBrunson = "captain-brunson-officer";

	public const string ExpertShieldTech = "expert-shield-tech-officer";

	public const string Admonition = "admonition-title";

	public const string SaNalaor = "sa-nalaor-title";

	public const string AdiGallia = "adi-gallia-officer";

	public static PassiveEffectNotice ObstructionNotice(ShipView defender)
	{
		if (!(defender == null) && defender.State != null && defender.State.CurrentSpeed >= 3)
		{
			ShipLoadoutState loadout = defender.Loadout;
			if (loadout != null && loadout.HasActive("admiral-montferrat-officer"))
			{
				return PassiveEffectNotice.Battlefield(defender, "[OFFICER] Admiral Montferrat obstructed the ship attack while moving at speed 3 or higher.", "ADMIRAL MONTFERRAT  -  ATTACK OBSTRUCTED");
			}
		}
		return PassiveEffectNotice.None;
	}

	public static PassiveEffectNotice DiscardMontferratAfterRam(ShipView ship)
	{
		UpgradeCardRuntimeState upgradeCardRuntimeState = ship?.Loadout?.Find("admiral-montferrat-officer");
		if (upgradeCardRuntimeState == null || !upgradeCardRuntimeState.Active || !upgradeCardRuntimeState.Discard())
		{
			return PassiveEffectNotice.None;
		}
		return PassiveEffectNotice.LogOnly("[OFFICER] Admiral Montferrat discarded after " + ship.Definition.DisplayName + " overlapped a ship.");
	}

	public static bool PreventsCriticalEffects(ShipView defender)
	{
		if ((object)defender == null)
		{
			return false;
		}
		return defender.Loadout?.HasActive("commander-gherant-officer") == true;
	}

	public static PassiveEffectNotice DiscardGherantAfterShipAttack(ShipView attacker)
	{
		UpgradeCardRuntimeState upgradeCardRuntimeState = attacker?.Loadout?.Find("commander-gherant-officer");
		if (upgradeCardRuntimeState == null || !upgradeCardRuntimeState.Active || !upgradeCardRuntimeState.Discard())
		{
			return PassiveEffectNotice.None;
		}
		return PassiveEffectNotice.LogOnly("[OFFICER] Commander Gherant discarded after " + attacker.Definition.DisplayName + " attacked a ship.");
	}

	public static bool ContainPreventsAllCriticalEffects(ShipView defender)
	{
		if ((object)defender == null)
		{
			return false;
		}
		return defender.Loadout?.HasActive("damage-control-officer") == true;
	}

	public static bool IsNearObstacle(ShipView ship)
	{
		if (ship == null)
		{
			return false;
		}
		var game = ship.Game;
		foreach (int obstacle in OfficialObjectives.Obstacles(game))
		{
			if (OfficialObjectives.DistanceToPiece(game, ship.Index, obstacle) <= TabletopMeasurements.DistanceBand(2))
			{
				return true;
			}
		}
		return false;
	}
}
