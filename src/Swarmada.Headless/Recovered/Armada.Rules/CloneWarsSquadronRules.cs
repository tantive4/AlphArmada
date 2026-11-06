#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using Armada.Core;
using Swarmada.Headless.Engine;

namespace Armada.Rules;

public static class CloneWarsSquadronRules
{
	public const string Ahsoka = "ahsoka-tano-delta-7-aethersprite-squadron";

	public const string AnakinYWing = "anakin-skywalker-btl-b-y-wing-squadron";

	public const string AnakinDelta = "anakin-skywalker-delta-7-aethersprite-squadron";

	public const string Axe = "axe-v-19-torrent-squadron";

	public const string Kickback = "kickback-v-19-torrent-squadron";

	public const string KitFisto = "kit-fisto-delta-7-aethersprite-squadron";

	public const string Luminara = "luminara-unduli-delta-7-aethersprite-squadron";

	public const string Matchstick = "matchstick-btl-b-y-wing-squadron";

	public const string OddBall = "odd-ball-arc-170-starfighter-squadron";

	public const string PloKoon = "plo-koon-delta-7-aethersprite-squadron";

	public const string Baktoid = "baktoid-prototypes-hyena-class-droid-bomber-squadron";

	public const string Dbs404 = "dbs-404-hyena-class-droid-bomber-squadron";

	public const string Dfs311 = "dfs-311-vulture-class-droid-fighter-squadron";

	public const string DisT81 = "dis-t81-droid-tri-fighter-squadron";

	public const string Grievous = "general-grievous-belbullab-22-starfighter-squadron";

	public const string HaorChall = "haor-chall-prototypes-vulture-class-droid-fighter-squadron";

	public const string Jango = "jango-fett-slave-i";

	public const string PhlacArphocc = "phlac-arphocc-prototypes-droid-tri-fighter-squadron";

	public const string AnakinIgnoreEngagementAction = "clone-wars.anakin-y-wing.ignore-engagement";

	public const string OddBallRerollAction = "clone-wars.odd-ball.reroll";

	public const string BaktoidAccuracyAction = "clone-wars.baktoid.set-accuracy";

	public const string Dbs404AddDieAction = "clone-wars.dbs-404.add-die";

	public const string Dfs311RerollAction = "clone-wars.dfs-311.force-reroll";

	public const string KitFistoAction = "clone-wars.kit-fisto.discard-token";

	public const string JangoFaceAction = "clone-wars.jango.change-face";

	public static bool Is(SquadronView squadron, string id)
	{
		if (squadron?.Definition != null)
		{
			return string.Equals(squadron.Definition.SquadronId, id, StringComparison.OrdinalIgnoreCase);
		}
		return false;
	}

	public static void AddActions(ICollection<AbilityAction> actions, SquadronView squadron, AbilityQueryContext context, IEnumerable<SquadronView> allSquadrons, ShipView defendingShip, SquadronView defendingSquadron)
	{
		if (actions == null || squadron == null || squadron.IsDestroyed)
		{
			return;
		}
		if (context.Timing == AbilityTiming.DuringActivation && Is(squadron, "anakin-skywalker-btl-b-y-wing-squadron") && !squadron.State.TemporaryIgnoresEngagement && HasAvailableDefenseToken(squadron) && CanBenefitFromIgnoringEngagement(squadron, context, allSquadrons))
		{
			actions.Add(CardAction("clone-wars.anakin-y-wing.ignore-engagement", squadron, "SPEND TOKEN: IGNORE ENGAGEMENT", "Spend one defense token. Anakin ignores engagement for the rest of this activation.", context.Timing));
		}
		if (context.Timing == AbilityTiming.ModifyAttackDice)
		{
			if (Is(squadron, "odd-ball-arc-170-starfighter-squadron") && context.MoveUsed)
			{
				actions.Add(CardAction("clone-wars.odd-ball.reroll", squadron, "ODD BALL: REROLL UP TO 2", "Choose up to two attack dice to reroll.", context.Timing));
			}
			if (defendingShip != null && Is(squadron, "baktoid-prototypes-hyena-class-droid-bomber-squadron") && HasOtherFriendlyBomberNearShip(squadron, defendingShip, allSquadrons))
			{
				actions.Add(CardAction("clone-wars.baktoid.set-accuracy", squadron, "BAKTOID: SET ACCURACY", "Change one eligible attack die to an accuracy result.", context.Timing));
			}
			if (Is(squadron, "dbs-404-hyena-class-droid-bomber-squadron") && squadron.State.CurrentHull > 0 && IsTouching(squadron, defendingShip, defendingSquadron))
			{
				actions.Add(CardAction("clone-wars.dbs-404.add-die", squadron, "DBS-404: SUFFER 1, ADD 1 DIE", "Suffer one damage, then add one die of a color already in the attack pool.", context.Timing));
			}
			if ((object)defendingSquadron != null && defendingSquadron.Definition?.IsUnique == true && Is(squadron, "jango-fett-slave-i") && JangoSacrifices(squadron, defendingSquadron, allSquadrons).Any())
			{
				actions.Add(CardAction("clone-wars.jango.change-face", squadron, "JANGO FETT: DAMAGE ALLY, SET DIE", "Damage another friendly squadron near the defender, then change one die to a face of your choice.", context.Timing));
			}
		}
	}

	private static bool CanBenefitFromIgnoringEngagement(SquadronView squadron, AbilityQueryContext context, IEnumerable<SquadronView> allSquadrons)
	{
		bool num = !context.MoveUsed && !squadron.State.TemporaryAttackOnly && SquadronAbilityEngine.IsPreventedFromMoving(squadron, allSquadrons);
		bool flag = !context.AttackUsed && !squadron.State.TemporaryCannotAttack && (!context.MoveUsed || SquadronAbilityEngine.CanMoveAndAttack(squadron)) && !SquadronAbilityEngine.CanAttackShip(squadron, allSquadrons);
		return num || flag;
	}

	public static bool AttacksCannotBeObstructed(SquadronView attacker)
	{
		return Is(attacker, "dis-t81-droid-tri-fighter-squadron");
	}

	public static bool LuminaraObstructs(SquadronView attacker, SquadronView defender, IEnumerable<SquadronView> allSquadrons)
	{
		if (attacker == null || defender?.Definition == null || defender.Definition.IsUnique || allSquadrons == null)
		{
			return false;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !allSquadron.IsDestroyed && allSquadron.OwnerId == defender.OwnerId && Is(allSquadron, "luminara-unduli-delta-7-aethersprite-squadron") && SquadronAbilityEngine.FootprintDistance(attacker, allSquadron) <= 3.369803f)
			{
				return true;
			}
		}
		return false;
	}

	public static int PloKoonCounterBonus(SquadronView defender, IEnumerable<SquadronView> allSquadrons)
	{
		if (defender == null || SquadronAbilityEngine.GetKeywordValue(defender.Definition, "COUNTER") > 0 || allSquadrons == null)
		{
			return 0;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !allSquadron.IsDestroyed && allSquadron.OwnerId == defender.OwnerId && Is(allSquadron, "plo-koon-delta-7-aethersprite-squadron") && SquadronAbilityEngine.FootprintDistance(defender, allSquadron) <= 3.369803f)
			{
				return 1;
			}
		}
		return 0;
	}

	public static int GrievousCriticalDamageBonus(SquadronView attacker, int criticalIcons, bool defenderHasReadyDefenseToken)
	{
		if (!Is(attacker, "general-grievous-belbullab-22-starfighter-squadron") || defenderHasReadyDefenseToken)
		{
			return 0;
		}
		return Math.Max(0, criticalIcons);
	}

	public static bool HasReadyDefenseToken(SquadronView squadron)
	{
		if (squadron == null)
		{
			return false;
		}
		foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
		{
			if (squadron.State.GetReadyDefenseTokenCount(value) > 0)
			{
				return true;
			}
		}
		return false;
	}

	public static bool HasReadyDefenseToken(ShipView ship)
	{
		if (ship == null)
		{
			return false;
		}
		foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
		{
			if (ship.State.GetReadyDefenseTokenCount(value) > 0)
			{
				return true;
			}
		}
		return false;
	}

	public static bool CanUseDfs311(SquadronView defender, bool attackWasObstructed)
	{
		if (attackWasObstructed)
		{
			return Is(defender, "dfs-311-vulture-class-droid-fighter-squadron");
		}
		return false;
	}

	public static IEnumerable<SquadronView> AxeProtectors(SquadronView defender, IEnumerable<SquadronView> allSquadrons)
	{
		if (defender == null || allSquadrons == null)
		{
			yield break;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !(allSquadron == defender) && !allSquadron.IsDestroyed && allSquadron.OwnerId == defender.OwnerId && Is(allSquadron, "axe-v-19-torrent-squadron") && HasAvailableDefenseToken(allSquadron) && !(SquadronAbilityEngine.FootprintDistance(allSquadron, defender) > 3.369803f))
			{
				yield return allSquadron;
			}
		}
	}

	public static IEnumerable<SquadronView> JangoSacrifices(SquadronView attacker, SquadronView defender, IEnumerable<SquadronView> allSquadrons)
	{
		if (attacker == null || defender == null || allSquadrons == null)
		{
			yield break;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !(allSquadron == attacker) && !allSquadron.IsDestroyed && allSquadron.OwnerId == attacker.OwnerId && !(SquadronAbilityEngine.FootprintDistance(allSquadron, defender) > 3.369803f))
			{
				yield return allSquadron;
			}
		}
	}

	private static bool HasOtherFriendlyBomberNearShip(SquadronView attacker, ShipView defender, IEnumerable<SquadronView> allSquadrons)
	{
		if (allSquadrons == null)
		{
			return false;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !(allSquadron == attacker) && !allSquadron.IsDestroyed && allSquadron.OwnerId == attacker.OwnerId && SquadronAbilityEngine.HasBomber(allSquadron) && SquadronAbilityEngine.FootprintDistance(allSquadron, defender) <= 3.369803f)
			{
				return true;
			}
		}
		return false;
	}

	private static bool IsTouching(SquadronView attacker, ShipView ship, SquadronView squadron)
	{
		if (ship != null)
		{
			return SquadronAbilityEngine.FootprintDistance(attacker, ship) <= 0.02f;
		}
		if (squadron != null)
		{
			return SquadronAbilityEngine.FootprintDistance(attacker, squadron) <= 0.02f;
		}
		return false;
	}

	private static bool HasAvailableDefenseToken(SquadronView squadron)
	{
		foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
		{
			if (squadron.State.GetAvailableDefenseTokenCount(value) > 0)
			{
				return true;
			}
		}
		return false;
	}

	private static AbilityAction CardAction(string actionId, SquadronView source, string label, string prompt, AbilityTiming timing)
	{
		return new AbilityAction(actionId, source.Definition.SquadronId, source.Definition.DisplayName, AbilitySourceKind.Squadron, label, prompt, RulesTextPresenter.ToVideoGameTerms(source.Definition.AbilityText), timing);
	}
}
