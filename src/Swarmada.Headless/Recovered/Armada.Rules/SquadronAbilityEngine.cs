#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using Armada.Core;
using Armada.Data;
using Swarmada.Headless.Engine;
using System.Numerics;
using Mathf = Swarmada.Headless.Engine.EngineMath;

namespace Armada.Rules;

public static class SquadronAbilityEngine
{
	public const string CommanderWoldar = "commander-woldar-officer";

	public const string TorynFarr = "toryn-farr-officer";

	public const string InstructorGoran = "instructor-goran-officer";

	public const string Ti99 = "ti-99-officer";

	public const float Ti99MaximumDistance = 8.09628f;

	public const string AgentKallus = "agent-kallus-officer";

	public const string GeneralDraven = "general-draven-officer";

	public const string HondoActivationAction = "squadron.hondo.toggle-or-reposition";

	public const string JendonAttackAction = "squadron.colonel-jendon.grant-attack";

	public const string BosskAccuracyAction = "squadron.bossk.add-accuracy";

	public const string MaarekCriticalAction = "squadron.maarek-stele.set-critical";

	public const string LockBraceAction = "attack.accuracy.lock-brace";

	public const string LockScatterAction = "attack.accuracy.lock-scatter";

	public const string LockEvadeAction = "attack.accuracy.lock-evade";

	public const string SpendBraceAction = "defense.squadron.spend-brace";

	public const string SpendScatterAction = "defense.squadron.spend-scatter";

	public const string SpendEvadeAction = "defense.squadron.spend-evade";

	public const string AdeptRerollAction = "keyword.adept.reroll";

	public const string SwarmRerollAction = "keyword.swarm.reroll";

	public const string DashRerollAction = "squadron.dash-rendar.reroll";

	public const string DodgeRerollAction = "keyword.dodge.force-reroll";

	public const string HeraSyndullaGhost = "hera-syndulla-ghost";

	public static IReadOnlyList<AbilityAction> GetActions(SquadronView squadron, AbilityQueryContext context, IEnumerable<SquadronView> allSquadrons, ShipView defendingShip = null, SquadronView defendingSquadron = null)
	{
		List<AbilityAction> list = new List<AbilityAction>();
		if (squadron == null || (squadron.IsDestroyed && !context.IsCounterAttack) || context.AttackUsed)
		{
			return list;
		}
		if (context.Timing == AbilityTiming.DuringActivation && string.Equals(squadron.Definition.SquadronId, "hondo-ohnaka-slave-i", StringComparison.OrdinalIgnoreCase) && HasSquadronAtDistance(squadron, allSquadrons, 3.369803f))
		{
			list.Add(new AbilityAction("squadron.hondo.toggle-or-reposition", squadron.Definition.SquadronId, squadron.Definition.DisplayName, AbilitySourceKind.Squadron, "HONDO'S FAVOR", "Use Hondo instead of attacking, then choose a squadron at distance 1.", RulesTextPresenter.ToVideoGameTerms(squadron.Definition.AbilityText), context.Timing));
		}
		if (context.Timing == AbilityTiming.DuringActivation && IsSquadron(squadron, "colonel-jendon-lambda-class-shuttle") && (!context.MoveUsed || CanMoveAndAttack(squadron)) && HasFriendlySquadronAtDistance(squadron, allSquadrons, 5.426696f))
		{
			list.Add(new AbilityAction("squadron.colonel-jendon.grant-attack", squadron.Definition.SquadronId, squadron.Definition.DisplayName, AbilitySourceKind.Squadron, "DIRECT ANOTHER ATTACK", "Use Colonel Jendon instead of attacking, then choose another friendly squadron at distance 1-2.", RulesTextPresenter.ToVideoGameTerms(squadron.Definition.AbilityText), context.Timing));
		}
		if (context.Timing == AbilityTiming.ModifyAttackDice && IsSquadron(squadron, "bossk-hounds-tooth") && squadron.State.CurrentHull <= 6)
		{
			list.Add(new AbilityAction("squadron.bossk.add-accuracy", squadron.Definition.SquadronId, squadron.Definition.DisplayName, AbilitySourceKind.Squadron, "TRIGGER BOSSK", "Add one guaranteed accuracy result to this attack.", RulesTextPresenter.ToVideoGameTerms(squadron.Definition.AbilityText), context.Timing));
		}
		if (context.Timing == AbilityTiming.ModifyAttackDice && IsSquadron(squadron, "maarek-stele-tie-defender-squadron"))
		{
			list.Add(new AbilityAction("squadron.maarek-stele.set-critical", squadron.Definition.SquadronId, squadron.Definition.DisplayName, AbilitySourceKind.Squadron, "AIM FOR A CRITICAL", "Change one rolled weapon die to its best critical result.", RulesTextPresenter.ToVideoGameTerms(squadron.Definition.AbilityText), context.Timing));
		}
		int effectiveAdept = GetEffectiveAdept(squadron, allSquadrons);
		if (context.Timing == AbilityTiming.ModifyAttackDice && effectiveAdept > 0)
		{
			list.Add(new AbilityAction("keyword.adept.reroll", "adept", $"Adept {effectiveAdept}", AbilitySourceKind.Keyword, $"USE ADEPT {effectiveAdept}", $"Choose up to {effectiveAdept} attack " + ((effectiveAdept == 1) ? "die." : "dice."), "While attacking, this squadron may reroll up to " + $"{effectiveAdept} attack " + ((effectiveAdept == 1) ? "die." : "dice."), context.Timing));
		}
		CloneWarsSquadronRules.AddActions(list, squadron, context, allSquadrons, defendingShip, defendingSquadron);
		return list;
	}

	public static int GetEffectiveAdept(SquadronView squadron, IEnumerable<SquadronView> allSquadrons)
	{
		if (squadron?.Definition == null)
		{
			return 0;
		}
		int keywordValue = GetKeywordValue(squadron.Definition, "ADEPT");
		if (!HasEscort(squadron) || allSquadrons == null)
		{
			return keywordValue;
		}
		if (!allSquadrons.Any((SquadronView source) => source != null && !source.IsDestroyed && source.OwnerId == squadron.OwnerId && IsSquadron(source, "hera-syndulla-x-wing-squadron") && FootprintDistance(source, squadron) <= 5.4704595f))
		{
			return keywordValue;
		}
		return Math.Max(keywordValue, 1);
	}

	public static IReadOnlyList<AbilityAction> GetAccuracyActions(SquadronView defender, int accuraciesRemaining, ISet<DefenseTokenInstance> lockedTokens)
	{
		List<AbilityAction> list = new List<AbilityAction>();
		if (defender == null || defender.IsDestroyed || accuraciesRemaining <= 0)
		{
			return list;
		}
		AddAccuracyAction(list, defender, DefenseTokenType.Brace, "attack.accuracy.lock-brace", "LOCK BRACE", lockedTokens);
		AddAccuracyAction(list, defender, DefenseTokenType.Scatter, "attack.accuracy.lock-scatter", "LOCK SCATTER", lockedTokens);
		AddAccuracyAction(list, defender, DefenseTokenType.Evade, "attack.accuracy.lock-evade", "LOCK EVADE", lockedTokens);
		return list;
	}

	public static IReadOnlyList<AbilityAction> GetDefenseActions(SquadronView defender, ISet<DefenseTokenInstance> lockedTokens, ISet<DefenseTokenType> spentThisAttack, bool dodgeUsed = false, SquadronView attacker = null, IEnumerable<SquadronView> allSquadrons = null)
	{
		List<AbilityAction> list = new List<AbilityAction>();
		if (defender == null || defender.IsDestroyed)
		{
			return list;
		}
		AddDefenseAction(list, defender, DefenseTokenType.Brace, "defense.squadron.spend-brace", "SPEND BRACE", "Halve the incoming hull damage, rounding up.", lockedTokens, spentThisAttack);
		AddDefenseAction(list, defender, DefenseTokenType.Scatter, "defense.squadron.spend-scatter", "SPEND SCATTER", "Cancel the attack and suffer no hull damage.", lockedTokens, spentThisAttack);
		AddDefenseAction(list, defender, DefenseTokenType.Evade, "defense.squadron.spend-evade", "SPEND EVADE", "Choose one attack die and force it to be rerolled.", lockedTokens, spentThisAttack);
		int keywordValue = GetKeywordValue(defender.Definition, "DODGE");
		int dodgeValue = GetDodgeValue(defender, attacker, allSquadrons);
		if (dodgeValue > 0 && !dodgeUsed)
		{
			int num = dodgeValue - keywordValue;
			list.Add(new AbilityAction("keyword.dodge.force-reroll", (num > 0) ? "screen" : "dodge", (num > 0) ? $"Screen / Dodge {dodgeValue}" : $"Dodge {dodgeValue}", AbilitySourceKind.Keyword, $"USE DODGE {dodgeValue}", $"Choose up to {dodgeValue} attack " + ((dodgeValue == 1) ? "die." : "dice."), (num > 0) ? ($"Screen grants Dodge {num} because the " + "attacker is engaged with " + $"{num} other friendly non-Screen " + ((num == 1) ? "squadron. " : "squadrons. ") + $"Reroll up to {dodgeValue} attack " + ((dodgeValue == 1) ? "die." : "dice.")) : ("When defending against a squadron, force up to " + $"{dodgeValue} attack " + ((dodgeValue == 1) ? "die" : "dice") + " to be rerolled."), AbilityTiming.Defense));
		}
		return list;
	}

	public static int GetDodgeValue(SquadronView defender, SquadronView attacker = null, IEnumerable<SquadronView> allSquadrons = null)
	{
		if (defender == null || defender.IsDestroyed || attacker == null || attacker.IsDestroyed || attacker.OwnerId == defender.OwnerId)
		{
			return 0;
		}
		int keywordValue = GetKeywordValue(defender.Definition, "DODGE");
		if (allSquadrons == null || !HasKeyword(defender.Definition, "SCREEN"))
		{
			return keywordValue;
		}
		int num = 0;
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !(allSquadron == defender) && !allSquadron.IsDestroyed && allSquadron.OwnerId == defender.OwnerId && !HasKeyword(allSquadron.Definition, "SCREEN") && IsEngaged(attacker, allSquadron))
			{
				num++;
				if (num >= 3)
				{
					break;
				}
			}
		}
		return keywordValue + num;
	}

	private static void AddAccuracyAction(ICollection<AbilityAction> actions, SquadronView defender, DefenseTokenType token, string abilityId, string label, ISet<DefenseTokenInstance> locked)
	{
		if (defender.State.GetDefenseTokenInstances(token).Any(delegate(DefenseTokenInstance instance)
		{
			ISet<DefenseTokenInstance> set = locked;
			return set == null || !set.Contains(instance);
		}))
		{
			actions.Add(new AbilityAction(abilityId, "accuracy-result", "Accuracy result", AbilitySourceKind.Keyword, label, $"Spend one accuracy result to lock one {token} token.", $"The locked {token} token cannot be spent against this " + "attack. A second copy remains usable unless it is locked separately.", AbilityTiming.ResolveAttackEffects));
		}
	}

	private static void AddDefenseAction(ICollection<AbilityAction> actions, SquadronView defender, DefenseTokenType token, string abilityId, string label, string rules, ISet<DefenseTokenInstance> locked, ISet<DefenseTokenType> spentThisAttack)
	{
		if ((spentThisAttack == null || !spentThisAttack.Contains(token)) && defender.State.GetDefenseTokenInstances(token).Any(delegate(DefenseTokenInstance instance)
		{
			ISet<DefenseTokenInstance> set = locked;
			return set == null || !set.Contains(instance);
		}))
		{
			actions.Add(new AbilityAction(abilityId, token.ToString().ToLowerInvariant(), $"{token} defense token", AbilitySourceKind.Keyword, label, rules, rules, AbilityTiming.Defense));
		}
	}

	private static bool HasSquadronAtDistance(SquadronView source, IEnumerable<SquadronView> allSquadrons, float maximumDistance)
	{
		if (source == null || allSquadrons == null)
		{
			return false;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !(allSquadron == source) && !allSquadron.IsDestroyed && FootprintDistance(source, allSquadron) <= maximumDistance)
			{
				return true;
			}
		}
		return false;
	}

	private static bool HasFriendlySquadronAtDistance(SquadronView source, IEnumerable<SquadronView> allSquadrons, float maximumDistance)
	{
		if (source == null || allSquadrons == null)
		{
			return false;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !(allSquadron == source) && !allSquadron.IsDestroyed && allSquadron.OwnerId == source.OwnerId && FootprintDistance(source, allSquadron) <= maximumDistance)
			{
				return true;
			}
		}
		return false;
	}

	public static bool HasKeyword(SquadronDefinition definition, string keyword)
	{
		if (definition == null || definition.Keywords == null || string.IsNullOrWhiteSpace(keyword))
		{
			return false;
		}
		string[] keywords = definition.Keywords.ToArray();
		for (int i = 0; i < keywords.Length; i++)
		{
			if (string.Equals(keywords[i], keyword, StringComparison.OrdinalIgnoreCase))
			{
				return true;
			}
		}
		return false;
	}

	public static bool CanMoveAndAttack(SquadronView squadron)
	{
		if (squadron != null)
		{
			if (!squadron.State.Commanded && !squadron.State.TemporaryCommanded)
			{
				return HasRogue(squadron);
			}
			return true;
		}
		return false;
	}

	public static bool HasRogue(SquadronView squadron)
	{
		if (squadron != null)
		{
			if (!squadron.State.TemporaryRogue)
			{
				return HasKeyword(squadron.Definition, "ROGUE");
			}
			return true;
		}
		return false;
	}

	public static int BuildAttackSeed(int round, int attackerOwner, int targetOwner, bool moved, string attackerRuntimeId, int attackOrdinal)
	{
		int num = StableRuntimeIdHash(attackerRuntimeId);
		return round * 17011 + attackerOwner * 193 + targetOwner * 997 + (moved ? 53 : 0) + num * 31 + Math.Max(0, attackOrdinal) * 104729;
	}

	public static int BuildFlakAttackSeed(int round, int attackerOwner, int targetOwner, string attackerRuntimeId, string targetRuntimeId, int attackOrdinal)
	{
		return BuildAttackSeed(round, attackerOwner, targetOwner, moved: false, attackerRuntimeId, attackOrdinal) + StableRuntimeIdHash(targetRuntimeId) * 524287;
	}

	private static int StableRuntimeIdHash(string runtimeId)
	{
		int num = -2128831035;
		string text = runtimeId ?? string.Empty;
		foreach (char c in text)
		{
			num ^= c;
			num *= 16777619;
		}
		return num;
	}

	// Official Tel Trevura rule, absent from the recovered implementation.
	public static bool HasEscort(SquadronView squadron) => squadron != null && (HasKeyword(squadron.Definition, "ESCORT")
		|| IsSquadron(squadron, "tel-trevura-jumpmaster-5000") && Enum.GetValues<DefenseTokenType>().Any(type => squadron.State.GetReadyDefenseTokenCount(type) > 0));

	public static bool HasBomber(SquadronView squadron)
	{
		if (squadron != null)
		{
			return HasKeyword(squadron.Definition, "BOMBER");
		}
		return false;
	}

	public static bool IsEngaged(SquadronView first, SquadronView second)
	{
		if (first == null || second == null || first == second || first.IsDestroyed || second.IsDestroyed || first.OwnerId == second.OwnerId)
		{
			return false;
		}
		return FootprintDistance(first, second) <= 3.369803f;
	}

	public static bool IsPreventedFromMoving(SquadronView squadron, IEnumerable<SquadronView> allSquadrons)
	{
		if (squadron == null || allSquadrons == null)
		{
			return false;
		}
		if (string.Equals(squadron.Definition.SquadronId, "tycho-celchu-a-wing-squadron", StringComparison.OrdinalIgnoreCase))
		{
			return false;
		}
		if (squadron.State.TemporaryIgnoresEngagement)
		{
			return false;
		}
		if (UnitAbilityEngine.IsVirtuallyEngagedByInstigator(squadron))
		{
			return true;
		}
		int num = 0;
		int num2 = 0;
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (IsEngaged(squadron, allSquadron))
			{
				num++;
				if (!HasKeyword(allSquadron.Definition, "HEAVY"))
				{
					num2++;
				}
			}
		}
		if (num2 == 0)
		{
			return false;
		}
		ShipView shipView = TitleRules.FindCommandingShip(squadron);
		if ((object)shipView != null && shipView.Loadout?.HasActive("admiral-chiraneau-officer") == true)
		{
			return false;
		}
		if (num == 1 && HasGrit(squadron, allSquadrons))
		{
			return false;
		}
		return true;
	}

	public static bool HasGrit(SquadronView squadron, IEnumerable<SquadronView> allSquadrons)
	{
		if (squadron?.State != null)
		{
			if (!squadron.State.TemporaryGrit && !HasKeyword(squadron.Definition, "GRIT") && !CommanderRules.GrantsPloKoonGrit(squadron, allSquadrons))
			{
				if (allSquadrons != null)
				{
					return HasFriendlyIntelProvider(squadron, allSquadrons);
				}
				return false;
			}
			return true;
		}
		return false;
	}

	private static bool HasFriendlyIntelProvider(SquadronView squadron, IEnumerable<SquadronView> allSquadrons)
	{
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !allSquadron.IsDestroyed && allSquadron.OwnerId == squadron.OwnerId && HasKeyword(allSquadron.Definition, "INTEL") && FootprintDistance(squadron, allSquadron) <= 3.369803f)
			{
				return true;
			}
		}
		return false;
	}

	public static bool CanAttackShip(SquadronView attacker, IEnumerable<SquadronView> allSquadrons)
	{
		if (attacker == null || attacker.IsDestroyed)
		{
			return false;
		}
		if (attacker.State.TemporaryIgnoresEngagement)
		{
			return true;
		}
		if (allSquadrons == null)
		{
			return true;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (IsEngaged(attacker, allSquadron) && (!HasKeyword(allSquadron.Definition, "HEAVY") || SquadronUpgradeRules.HeavyPreventsShipAttack(allSquadron)))
			{
				return false;
			}
		}
		return true;
	}

	public static bool CanAttackSquadron(SquadronView attacker, SquadronView target, IEnumerable<SquadronView> allSquadrons, float distance, out string reason)
	{
		reason = string.Empty;
		if (attacker == null || target == null || target.IsDestroyed || attacker.OwnerId == target.OwnerId)
		{
			reason = "TARGET IS NOT AN ENEMY SQUADRON";
			return false;
		}
		bool flag = IsSnipeAttack(attacker, distance);
		if (distance > GetSquadronAttackDistance(attacker))
		{
			reason = "OUT OF ATTACK RANGE";
			return false;
		}
		if (IsSquadron(target, "valen-rudor-tie-fighter-squadron") && IsEngagedWithAnotherSquadron(attacker, target, allSquadrons))
		{
			reason = "VALEN RUDOR CANNOT BE TARGETED WHILE YOU ARE ENGAGED WITH ANOTHER SQUADRON";
			return false;
		}
		if (flag || IgnoresEscortAndCounter(attacker))
		{
			return true;
		}
		if (allSquadrons == null || HasEscort(target))
		{
			return true;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && allSquadron.OwnerId == target.OwnerId && HasEscort(allSquadron) && IsEngaged(attacker, allSquadron))
			{
				reason = "AN ENGAGED ESCORT MUST BE TARGETED";
				return false;
			}
		}
		return true;
	}

	public static bool CanAttackSquadron(SquadronView attacker, SquadronView target, IEnumerable<SquadronView> allSquadrons, out string reason)
	{
		return CanAttackSquadron(attacker, target, allSquadrons, FootprintDistance(attacker, target), out reason);
	}

	private static bool IsEngagedWithAnotherSquadron(SquadronView attacker, SquadronView excludedTarget, IEnumerable<SquadronView> allSquadrons)
	{
		if (allSquadrons == null)
		{
			return false;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !(allSquadron == excludedTarget) && allSquadron.OwnerId != attacker.OwnerId && !allSquadron.IsDestroyed && IsEngaged(attacker, allSquadron))
			{
				return true;
			}
		}
		return false;
	}

	public static int GetKeywordValue(SquadronDefinition definition, string keywordName)
	{
		if (definition?.Keywords == null || string.IsNullOrWhiteSpace(keywordName))
		{
			return 0;
		}
		string[] keywords = definition.Keywords.ToArray();
		foreach (string text in keywords)
		{
			if (!string.IsNullOrWhiteSpace(text) && text.StartsWith(keywordName, StringComparison.OrdinalIgnoreCase))
			{
				string[] array = text.Split(new char[1] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
				if (array.Length == 1)
				{
					return 1;
				}
				if (int.TryParse(array[^1], out var result))
				{
					return Math.Max(0, result);
				}
			}
		}
		return 0;
	}

	public static bool CanRelaySquadronCommand(ShipView resolvingShip, SquadronView relaySource, SquadronView target, float commandRange, int alreadyRelayedBySource)
	{
		return CanRelaySquadronCommand(resolvingShip, resolvingShip, relaySource, target, commandRange, alreadyRelayedBySource);
	}

	public static bool CanRelaySquadronCommand(ShipView resolvingShip, ShipView rangeSource, SquadronView relaySource, SquadronView target, float commandRange, int alreadyRelayedBySource)
	{
		if (resolvingShip == null || rangeSource == null || relaySource == null || target == null || rangeSource.IsDestroyed || relaySource.IsDestroyed || target.IsDestroyed || rangeSource.OwnerId != resolvingShip.OwnerId || relaySource.OwnerId != resolvingShip.OwnerId || target.OwnerId != resolvingShip.OwnerId)
		{
			return false;
		}
		int keywordValue = GetKeywordValue(relaySource.Definition, "RELAY");
		if (alreadyRelayedBySource < keywordValue && FootprintDistance(relaySource, rangeSource) <= commandRange)
		{
			return FootprintDistance(target, relaySource) <= 8.09628f;
		}
		return false;
	}

	public static int GetSnipeValue(SquadronView attacker)
	{
		if (!(attacker == null))
		{
			return Math.Max(GetKeywordValue(attacker.Definition, "SNIPE"), attacker.State.TemporarySnipeValue);
		}
		return 0;
	}

	public static float GetSquadronAttackDistance(SquadronView attacker)
	{
		if (GetSnipeValue(attacker) <= 0)
		{
			return 3.369803f;
		}
		return 5.426696f;
	}

	public static bool IsSnipeAttack(SquadronView attacker, float distance)
	{
		if (GetSnipeValue(attacker) > 0 && distance > 3.369803f)
		{
			return distance <= 5.426696f;
		}
		return false;
	}

	public static float GetShipAttackDistance(SquadronView attacker, IEnumerable<SquadronView> allSquadrons)
	{
		return GetShipAttackDistanceAt(attacker, (attacker != null) ? attacker.transform.position : Vector3.Zero, allSquadrons);
	}

	public static float GetShipAttackDistanceAt(SquadronView attacker, Vector3 attackerPosition, IEnumerable<SquadronView> allSquadrons)
	{
		if (attacker == null || allSquadrons == null)
		{
			return 3.369803f;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !allSquadron.IsDestroyed && allSquadron.OwnerId == attacker.OwnerId && IsSquadron(allSquadron, "major-rhymer-tie-bomber-squadron") && (allSquadron == attacker || FootprintDistanceAt(attacker, attackerPosition, allSquadron) <= 3.369803f))
			{
				return 5.426696f;
			}
		}
		return 3.369803f;
	}

	public static int GetEffectiveSpeed(SquadronView squadron, IEnumerable<SquadronView> allSquadrons)
	{
		if (squadron?.Definition == null)
		{
			return 0;
		}
		if (squadron.State.TemporarySpeedOverride >= 0)
		{
			return squadron.State.TemporarySpeedOverride;
		}
		int num = TitleRules.CommandedSquadronSpeed(squadron, squadron.Definition.Speed);
		ShipView shipView = TitleRules.FindCommandingShip(squadron);
		if ((object)shipView != null && shipView.Loadout?.HasActive("admiral-chiraneau-officer") == true && allSquadrons != null)
		{
			foreach (SquadronView allSquadron in allSquadrons)
			{
				if (IsEngaged(squadron, allSquadron))
				{
					num = 2;
					break;
				}
			}
		}
		if (FleetCommandRules.IncreasesCommandedSquadronSpeed(squadron))
		{
			num = Math.Min(5, num + 1);
		}
		if (allSquadrons == null)
		{
			return num;
		}
		foreach (SquadronView allSquadron2 in allSquadrons)
		{
			if (!(allSquadron2 == null) && !allSquadron2.IsDestroyed && allSquadron2.OwnerId != squadron.OwnerId && IsSquadron(allSquadron2, "ketsu-onyo-shadow-caster") && FootprintDistance(squadron, allSquadron2) <= 3.369803f)
			{
				return Math.Max(1, num - 2);
			}
		}
		return num;
	}

	public static int GetAttackRerolls(SquadronView attacker, SquadronView target, IEnumerable<SquadronView> allSquadrons)
	{
		if (attacker == null || target == null)
		{
			return 0;
		}
		return GetDashRerollCount(attacker) + (HasSwarmRerollOpportunity(attacker, target, allSquadrons) ? 1 : 0);
	}

	public static bool HasSwarmRerollOpportunity(SquadronView attacker, SquadronView target, IEnumerable<SquadronView> allSquadrons)
	{
		if (attacker == null || target == null || allSquadrons == null || (!HasKeyword(attacker.Definition, "SWARM") && !attacker.State.TemporarySwarm))
		{
			return false;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !(allSquadron == attacker) && allSquadron.OwnerId == attacker.OwnerId && !allSquadron.IsDestroyed && IsEngaged(allSquadron, target))
			{
				return true;
			}
		}
		return false;
	}

	public static int GetDashRerollCount(SquadronView attacker)
	{
		if (!IsSquadron(attacker, "dash-rendar-outrider"))
		{
			return 0;
		}
		int num = 0;
		SquadronView[] array = GameScope.Current.SceneSquadronViews;
		foreach (SquadronView squadronView in array)
		{
			if (!(squadronView == null) && !squadronView.IsDestroyed && squadronView.OwnerId != attacker.OwnerId && FootprintDistance(attacker, squadronView) <= 3.369803f)
			{
				num++;
			}
		}
		ShipView[] array2 = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array2)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId != attacker.OwnerId && FootprintDistance(attacker, shipView) <= 3.369803f)
			{
				num++;
			}
		}
		return num;
	}

	public static int GetBonusBlueDice(SquadronView attacker, SquadronView target, IEnumerable<SquadronView> allSquadrons)
	{
		if (attacker == null || target == null)
		{
			return 0;
		}
		int num = 0;
		if (string.Equals(attacker.Definition.SquadronId, "wedge-antilles", StringComparison.OrdinalIgnoreCase) && target.State.Activated)
		{
			num += 2;
		}
		if (allSquadrons == null || !HasKeyword(attacker.Definition, "SWARM"))
		{
			return num;
		}
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !(allSquadron == attacker) && allSquadron.OwnerId == attacker.OwnerId && !allSquadron.IsDestroyed && string.Equals(allSquadron.Definition.SquadronId, "howlrunner", StringComparison.OrdinalIgnoreCase) && FootprintDistance(attacker, allSquadron) <= 3.369803f)
			{
				return num + 1;
			}
		}
		return num;
	}

	public static bool TreatsDefenderAsObstructed(SquadronView defender)
	{
		if (defender != null)
		{
			return string.Equals(defender.Definition.SquadronId, "ciena-ree", StringComparison.OrdinalIgnoreCase);
		}
		return false;
	}

	public static bool IgnoresShipShields(SquadronView attacker)
	{
		if (attacker != null)
		{
			return string.Equals(attacker.Definition.SquadronId, "luke-skywalker", StringComparison.OrdinalIgnoreCase);
		}
		return false;
	}

	public static void ResolveDutchVander(SquadronView attacker, SquadronView defender, int damageApplied)
	{
		if (!(attacker == null) && !(defender == null) && damageApplied > 0 && string.Equals(attacker.Definition.SquadronId, "dutch-vander", StringComparison.OrdinalIgnoreCase))
		{
			if (defender.State.Activated)
			{
				defender.ApplyDamage(1);
			}
			else
			{
				defender.State.MarkActivated(commanded: false);
			}
		}
	}

	public static int ResolveMaulerMithelAfterMove(SquadronView mover, IEnumerable<SquadronView> allSquadrons)
	{
		if (mover == null || allSquadrons == null || !string.Equals(mover.Definition.SquadronId, "mauler-mithel", StringComparison.OrdinalIgnoreCase))
		{
			return 0;
		}
		int num = 0;
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (IsEngaged(mover, allSquadron) && allSquadron.ApplyDamage(1) > 0)
			{
				num++;
			}
		}
		return num;
	}

	public static int GetCounterValue(SquadronView defender, SquadronView attacker = null, IEnumerable<SquadronView> allSquadrons = null)
	{
		if (IgnoresEscortAndCounter(attacker))
		{
			return 0;
		}
		if (defender?.Definition?.Keywords == null)
		{
			return 0;
		}
		int num = 0;
		var keywords = defender.Definition.Keywords;
		foreach (string text in keywords)
		{
			if (!string.IsNullOrWhiteSpace(text) && text.StartsWith("COUNTER", StringComparison.OrdinalIgnoreCase))
			{
				string[] array = text.Split(new char[1] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
				if (array.Length > 1 && int.TryParse(array[1], out var result))
				{
					num = Math.Max(num, Math.Max(0, result));
				}
			}
		}
		if (allSquadrons == null)
		{
			return num;
		}
		num += CloneWarsSquadronRules.PloKoonCounterBonus(defender, allSquadrons);
		foreach (SquadronView allSquadron in allSquadrons)
		{
			if (!(allSquadron == null) && !(allSquadron == defender) && !allSquadron.IsDestroyed && allSquadron.OwnerId == defender.OwnerId && IsSquadron(allSquadron, "dengar-punishing-one") && FootprintDistance(defender, allSquadron) <= 5.426696f)
			{
				num++;
				break;
			}
		}
		if (NearbyOfficerSource(defender, "instructor-goran-officer", 8.227571f) != null && !HasKeyword(defender.Definition, "HEAVY"))
		{
			num = Math.Max(1, num + 1);
		}
		if (NearbyOfficerSource(defender, "ti-99-officer", 8.09628f) != null && !defender.State.Activated)
		{
			num = ((num > 0) ? (num + 1) : 3);
		}
		return num;
	}

	public static ShipView CommanderWoldarSource(SquadronView attacker)
	{
		if (attacker == null || HasKeyword(attacker.Definition, "SWARM"))
		{
			return null;
		}
		return NearbyOfficerSource(attacker, "commander-woldar-officer", 8.09628f);
	}

	public static ShipView TorynFarrSource(SquadronView attacker)
	{
		return NearbyOfficerSource(attacker, "toryn-farr-officer", 8.09628f);
	}

	public static IReadOnlyList<string> FlakDieBonusSources(ShipView attacker, SquadronView defender, IEnumerable<SquadronView> allSquadrons = null)
	{
		List<string> list = new List<string>();
		if (attacker?.Loadout == null || defender?.Definition == null)
		{
			return list;
		}
		if (defender.Definition.IsUnique && attacker.Loadout.HasActive("agent-kallus-officer"))
		{
			list.Add("Agent Kallus");
		}
		if ((GetCounterValue(defender, null, allSquadrons) > 0 || HasKeyword(defender.Definition, "INTEL")) && attacker.Loadout.HasActive("general-draven-officer"))
		{
			list.Add("General Draven");
		}
		return list;
	}

	public static void ResolveTi99CounterActivation(SquadronView defender)
	{
		if (!(defender == null) && !defender.State.Activated && !(NearbyOfficerSource(defender, "ti-99-officer", 8.09628f) == null))
		{
			defender.State.MarkActivated(commanded: false);
		}
	}

	private static ShipView NearbyOfficerSource(SquadronView squadron, string cardId, float range)
	{
		if (squadron == null)
		{
			return null;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId == squadron.OwnerId)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive(cardId) && FootprintDistance(squadron, shipView) <= range)
				{
					return shipView;
				}
			}
		}
		return null;
	}

	public static bool CanCounterAttack(SquadronView source, SquadronView target, IEnumerable<SquadronView> allSquadrons)
	{
		if (source != null && target != null && !target.IsDestroyed && source.OwnerId != target.OwnerId)
		{
			if (IsSquadron(target, "valen-rudor-tie-fighter-squadron"))
			{
				return !IsEngagedWithAnotherSquadron(source, target, allSquadrons);
			}
			return true;
		}
		return false;
	}

	public static int DamageAgainstSquadron(SquadronView attacker, SquadronView defender, AttackResolution attack, bool counter = false)
	{
		bool flag = (counter && IsSquadron(attacker, "shara-bey-a-wing-squadron")) || (IsSquadron(attacker, "darth-vader-tie-defender-squadron") && (object)defender != null && defender.Definition?.IsUnique == true);
		return attack.Hits + (flag ? attack.Criticals : 0) + CloneWarsSquadronRules.GrievousCriticalDamageBonus(attacker, attack.Criticals, CloneWarsSquadronRules.HasReadyDefenseToken(defender));
	}

	public static int DamageFromCounter(SquadronView defender, AttackResolution counter)
	{
		return counter.Hits + (IsSquadron(defender, "shara-bey-a-wing-squadron") ? counter.Criticals : 0);
	}

	public static AttackResolution AddBosskAccuracy(AttackResolution attack)
	{
		List<AttackDie> dice = new List<AttackDie>(attack.Dice)
		{
			new AttackDie(AttackDieColor.Blue, DieFace.Accuracy)
		};
		return new AttackResolution(attack.Range, attack.DiceRolled + 1, attack.Hits, attack.Criticals, attack.Accuracies + 1, attack.Blanks, attack.IsLegal, attack.CriticalDamagePerIcon, dice);
	}

	public static AttackResolution SetBestDieToCritical(AttackResolution attack)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		List<AttackDie> list = new List<AttackDie>(attack.Dice);
		int num = -1;
		for (int i = 0; i < list.Count; i++)
		{
			if (list[i].Face == DieFace.Blank || list[i].Face == DieFace.Accuracy)
			{
				num = i;
				break;
			}
		}
		if (num < 0)
		{
			for (int j = 0; j < list.Count; j++)
			{
				if (list[j].Face == DieFace.Hit)
				{
					num = j;
					break;
				}
			}
		}
		if (num < 0)
		{
			num = 0;
		}
		AttackDie attackDie = list[num];
		list[num] = new AttackDie(attackDie.Color, (attackDie.Color == AttackDieColor.Black) ? DieFace.HitCritical : DieFace.Critical);
		return Recount(attack, list);
	}

	private static AttackResolution Recount(AttackResolution original, IReadOnlyList<AttackDie> dice)
	{
		int num = 0;
		int num2 = 0;
		int num3 = 0;
		int num4 = 0;
		using (IEnumerator<AttackDie> enumerator = dice.GetEnumerator())
		{
			while (enumerator.MoveNext())
			{
				switch (enumerator.Current.Face)
				{
				case DieFace.Hit:
					num++;
					break;
				case DieFace.DoubleHit:
					num += 2;
					break;
				case DieFace.Critical:
					num2++;
					break;
				case DieFace.HitCritical:
					num++;
					num2++;
					break;
				case DieFace.Accuracy:
					num3++;
					break;
				default:
					num4++;
					break;
				}
			}
		}
		return new AttackResolution(original.Range, dice.Count, num, num2, num3, num4, original.IsLegal, original.CriticalDamagePerIcon, dice);
	}

	public static bool IgnoresEscortAndCounter(SquadronView squadron)
	{
		return IsSquadron(squadron, "ig-88-ig-2000");
	}

	private static bool IsSquadron(SquadronView squadron, string id)
	{
		if (squadron?.Definition != null)
		{
			return string.Equals(squadron.Definition.SquadronId, id, StringComparison.OrdinalIgnoreCase);
		}
		return false;
	}

	public static AttackResolution ApplyCaptainJonus(ShipView attacker, ShipView defender, AttackResolution attack)
	{
		if (attacker == null || defender == null || attack.Accuracies > 0)
		{
			return attack;
		}
		SquadronView[] array = GameScope.Current.SceneSquadronViews;
		foreach (SquadronView squadronView in array)
		{
			if (squadronView == null || squadronView.IsDestroyed || squadronView.OwnerId != attacker.OwnerId || !string.Equals(squadronView.Definition.SquadronId, "captain-jonus", StringComparison.OrdinalIgnoreCase) || FootprintDistance(squadronView, defender) > 3.369803f)
			{
				continue;
			}
			int num = attack.Hits;
			int num2 = attack.Criticals;
			int num3 = attack.Blanks;
			if (num3 > 0)
			{
				num3--;
			}
			else if (num > 0)
			{
				num--;
			}
			else
			{
				if (num2 <= 0)
				{
					return attack;
				}
				num2--;
			}
			return new AttackResolution(attack.Range, attack.DiceRolled, num, num2, attack.Accuracies + 1, num3, attack.IsLegal, attack.CriticalDamagePerIcon);
		}
		return attack;
	}

	public static float FootprintDistance(SquadronView first, SquadronView second)
	{
		if (first == null || second == null)
		{
			return float.PositiveInfinity;
		}
		float footprintRadius = GetFootprintRadius(first);
		float footprintRadius2 = GetFootprintRadius(second);
		float magnitude = EngineVectors.ProjectOnPlane(first.transform.position - second.transform.position, Vector3.UnitY).Length();
		return Mathf.Max(0f, magnitude - footprintRadius - footprintRadius2);
	}

	public static float FootprintDistanceAt(SquadronView first, Vector3 firstPosition, SquadronView second)
	{
		if (first == null || second == null)
		{
			return float.PositiveInfinity;
		}
		float footprintRadius = GetFootprintRadius(first);
		float footprintRadius2 = GetFootprintRadius(second);
		float magnitude = EngineVectors.ProjectOnPlane(firstPosition - second.transform.position, Vector3.UnitY).Length();
		return Mathf.Max(0f, magnitude - footprintRadius - footprintRadius2);
	}

	public static float FootprintDistanceAt(SquadronView squadron, Vector3 squadronPosition, ShipView ship)
	{
		if (squadron == null || ship == null)
		{
			return float.PositiveInfinity;
		}
		float footprintRadius = GetFootprintRadius(squadron);
		if (ship.Game.Topology.Ships[ship.Index].IsStation)
			return Swarmada.Headless.Geometry.Geometry2D.Distance(
				new Swarmada.Headless.Geometry.Circle(EngineVectors.FromWorld(squadronPosition), Swarmada.Headless.Geometry.TabletopMeasurements.FromWorldUnits(footprintRadius)),
				CombatModule.StationShape(ship.Game, ship.Index)) / Swarmada.Headless.Geometry.TabletopMeasurements.MillimetresPerWorldUnit;
		GameplayFootprintView component = ship.GetComponent<GameplayFootprintView>();
		Vector3 vector = ship.transform.InverseTransformPoint(squadronPosition);
		float num = ((component != null) ? (component.SourceSize.X * 0.5f) : (ship.Definition.GameplayFootprint.X * 0.5f));
		float num2 = ((component != null) ? (component.SourceSize.Y * 0.5f) : (ship.Definition.GameplayFootprint.Y * 0.5f));
		float num3 = Mathf.Max(Mathf.Abs(vector.X) - num, 0f);
		float num4 = Mathf.Max(Mathf.Abs(vector.Z) - num2, 0f);
		return Mathf.Max(0f, Mathf.Sqrt(num3 * num3 + num4 * num4) - footprintRadius);
	}

	public static float FootprintDistance(SquadronView squadron, ShipView ship)
	{
		if (squadron == null || ship == null)
		{
			return float.PositiveInfinity;
		}
		return FootprintDistanceAt(squadron, squadron.transform.position, ship);
	}

	public static float FootprintDistance(ShipView ship, SquadronView squadron)
	{
		return FootprintDistance(squadron, ship);
	}

	public static float HullZoneDistance(SquadronView squadron, ShipView ship, HullZone zone)
	{
		if (squadron == null || ship == null || !ship.Definition.HasHullZone(zone))
		{
			return float.PositiveInfinity;
		}
		if (ship.Game.Topology.Ships[ship.Index].IsStation) return zone == HullZone.Front ? FootprintDistance(squadron, ship) : float.PositiveInfinity;
		Vector3[] hullZoneEdgesWorld = ship.GetHullZoneEdgesWorld(zone);
		float num = float.PositiveInfinity;
		for (int i = 0; i < hullZoneEdgesWorld.Length - 1; i++)
		{
			num = Mathf.Min(num, AttackRangeGeometry.MeasureSegmentToCircle(hullZoneEdgesWorld[i], hullZoneEdgesWorld[i + 1], squadron.transform.position, GetFootprintRadius(squadron)).Distance);
		}
		return num;
	}

	private static float GetFootprintRadius(SquadronView squadron)
	{
		GameplayFootprintView gameplayFootprintView = ((squadron != null) ? squadron.GetComponent<GameplayFootprintView>() : null);
		if (gameplayFootprintView != null)
		{
			return Mathf.Min(gameplayFootprintView.SourceSize.X, gameplayFootprintView.SourceSize.Y) * 0.5f;
		}
		if (!(squadron != null) || !(squadron.Definition != null))
		{
			return 0f;
		}
		return Mathf.Min(squadron.Definition.GameplayFootprint.X, squadron.Definition.GameplayFootprint.Y) * 0.5f;
	}
}
