using Armada.Core;
using Armada.Rules;

namespace Swarmada.Headless.Engine;

public sealed partial class SourceCombatEffects
{
    private static IEnumerable<AbilityAction> AdditionalSquadronActions(GameContext game, AttackState a)
    {
        var source = game.SquadronViews[game.SquadronIndex(a.Attacker)];
        bool sloane = !SquadronAbilityEngine.HasRogue(source)
            && game.SceneShipViews.Any(ship => ship.OwnerId == source.OwnerId && ship.Loadout.HasActive("admiral-sloane-commander"));
        if (sloane)
        {
            if (!a.EffectFlags.ContainsKey("sloane.accuracy-used") && Indices(a, die => die.Face == DieFace.Accuracy).Any())
                foreach (var type in Enum.GetValues<DefenseTokenType>())
                    if (AdditionalTokens(game, a.Defender, type).Any())
                        yield return Ability("remaining.squadron-dice.sloane.spend." + (int)type, "admiral-sloane-commander", "Spend an accuracy to spend enemy " + type);
            if (game.IsShip(a.Defender) && Indices(a, die => die.Face is DieFace.Critical or DieFace.HitCritical).Any())
                yield return Ability("remaining.squadron-dice.sloane.reroll-critical", "admiral-sloane-commander", "Reroll one critical die with Sloane");
        }
        if (!game.IsShip(a.Defender) && source.Definition.SquadronId == "ten-numb-b-wing-squadron"
            && Indices(a, die => die.Color == AttackDieColor.Blue && die.Face is DieFace.Critical or DieFace.HitCritical).Any())
            yield return Ability("squadron.ten-numb.spend-blue-critical", source.Definition.SquadronId, "Spend blue critical; damage nearby enemy squadrons");
        if (game.IsShip(a.Defender) && (source.State.TemporaryAssault || SquadronAbilityEngine.HasKeyword(source.Definition, "ASSAULT")) && !a.EffectFlags.ContainsKey("assault.used")
            && Indices(a, die => die.Face is DieFace.Hit or DieFace.DoubleHit or DieFace.HitCritical || source.State.TemporaryAssault && die.Face == DieFace.Accuracy).Any())
            foreach (var command in Enum.GetValues<ShipCommand>().Where(game.State.Commands[a.Defender].CanGainRaidToken))
                yield return Ability("remaining.squadron-dice.assault." + (int)command, "clone-commander-wolffe-officer", "Spend one hit or accuracy; gain " + command + " raid");
    }
    private static IEnumerable<DefenseTokenInstance> AdditionalTokens(GameContext game, int entity, DefenseTokenType type)
        => game.IsShip(entity) ? DefenseTokenInstanceChoiceRules.GetSpendableInstances(game.State.Ships[entity], type, null)
            : game.State.Squadrons[game.SquadronIndex(entity)].GetDefenseTokenInstances(type);
    private static bool TryBeginAdditionalCombatChoice(GameContext game, AttackState a, GameAction action)
    {
        string id = action.AbilityId; int owner = game.State.Owners[a.Attacker >= 0 ? a.Attacker : a.Defender];
        if (id == "remaining.attack-dice.sato.configure")
        { Begin(a, "additional.sato-source", id, owner, [], 0, 2); return true; }
        if (id.StartsWith("remaining.squadron-dice.sloane.spend.", StringComparison.Ordinal))
        { Begin(a, "additional.sloane-token", id, owner, [], 1, 1); return true; }
        if (id == "remaining.squadron-dice.sloane.reroll-critical")
        { Begin(a, "reroll", id, owner, Indices(a, die => die.Face is DieFace.Critical or DieFace.HitCritical), 1, 1); return true; }
        if (id == "squadron.ten-numb.spend-blue-critical")
        { Begin(a, "additional.ten-numb", id, owner, Indices(a, die => die.Color == AttackDieColor.Blue && die.Face is DieFace.Critical or DieFace.HitCritical), 1, 1); return true; }
        if (id.StartsWith("remaining.squadron-dice.assault.", StringComparison.Ordinal))
        { Begin(a, "additional.assault", id, owner, Indices(a, die => die.Face is DieFace.Hit or DieFace.DoubleHit or DieFace.HitCritical || game.State.Squadrons[game.SquadronIndex(a.Attacker)].TemporaryAssault && die.Face == DieFace.Accuracy), 1, 1); return true; }
        return false;
    }
    private static IEnumerable<GameActionOption> AdditionalCombatChoices(GameContext game, AttackState a)
    {
        if (a.EffectRule is "additional.ai-color" or "additional.b2-color")
        {
            foreach (int color in a.EffectEligibleIndices)
                yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, Amount: color, AbilityId: a.EffectRule == "additional.ai-color" ? "squadron.ai.color." + color : "b2.flak." + ((AttackDieColor)color).ToString().ToLowerInvariant()), "Add " + (AttackDieColor)color);
            if (a.EffectRule == "additional.b2-color") yield return new(new(GameActionKind.UseAbility, a.Attacker, AbilityId: "b2.flak.skip"), "Keep B2 Rocket Troopers ready");
            yield break;
        }
        if (a.EffectRule == "additional.sato-source")
        {
            foreach (var color in a.Dice.Select(die => die.Color).Distinct())
                yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, Amount: (int)color, AbilityId: "sato-choice.source." + (int)color), "Replace one " + color + " die");
            yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, AbilityId: "sato-choice.finish"), "Finish Sato replacements"); yield break;
        }
        if (a.EffectRule == "additional.sato-destination")
        {
            foreach (var color in new[] { AttackDieColor.Red, AttackDieColor.Blue, AttackDieColor.Black }.Where(color => (int)color != a.EffectValue))
                yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, Amount: (int)color, AbilityId: "sato-choice.destination." + (int)color), "Replace with " + color);
            yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, AbilityId: "sato-choice.back"), "Choose another source color"); yield break;
        }
        if (a.EffectRule == "additional.sloane-token")
        {
            foreach (var token in AdditionalTokens(game, a.Defender, (DefenseTokenType)SuffixInt(a.EffectActionId)))
                yield return new(new(GameActionKind.ChooseToken, a.Attacker, a.Defender, Token: token.Token, Index: token.Slot, AbilityId: "combat.effect.sloane-token"),
                    "Spend " + token.Token + " token " + token.Slot + (token.Exhausted ? " (exhausted)" : " (ready)"));
            yield break;
        }
        foreach (int index in a.EffectEligibleIndices)
            yield return new(new(GameActionKind.ChooseDie, a.Attacker, a.Defender, Index: index, AbilityId: "combat.effect.additional-die"),
                "Spend die " + index + " " + a.Dice[index].Color + " " + a.Dice[index].Face);
    }
    private static bool ApplyAdditionalCombatChoice(GameContext game, AttackState a, GameAction action)
    {
        string rule = a.EffectRule, id = a.EffectActionId;
        if (rule == "additional.b2-color")
        {
            ClearChoice(a); if (action.AbilityId == "b2.flak.skip") return true;
            if (!Card(game, a.Attacker, "b2-rocket-troopers").Exhaust()) return false;
            AppendDice(a, (AttackDieColor)action.Amount, 1); return true;
        }
        if (rule == "additional.ai-color")
        {
            AppendDice(a, (AttackDieColor)action.Amount, 1); a.EffectStage++;
            if (a.EffectStage >= a.EffectMaximum) ClearChoice(a); return true;
        }
        if (rule == "additional.sato-source")
        {
            if (action.AbilityId == "sato-choice.finish") { if (a.EffectStage > 0) Mark(a, id); ClearChoice(a); return true; }
            a.EffectValue = action.Amount; a.EffectRule = "additional.sato-destination"; return true;
        }
        if (rule == "additional.sato-destination")
        {
            if (action.AbilityId == "sato-choice.back") { a.EffectRule = "additional.sato-source"; return true; }
            int index = Array.FindIndex(a.Dice, die => (int)die.Color == a.EffectValue);
            a.Dice[index] = new((AttackDieColor)action.Amount, DieFace.Blank); a.EffectStage++;
            if (a.EffectStage >= 2) { Mark(a, id); ClearChoice(a); } else a.EffectRule = "additional.sato-source";
            return true;
        }
        if (rule == "additional.sloane-token")
        {
            a.EffectFlags["sloane.pending.type"] = (int)action.Token; a.EffectFlags["sloane.pending.slot"] = action.Index;
            Begin(a, "additional.sloane-spend", id, a.EffectOwner, Indices(a, die => die.Face == DieFace.Accuracy), 1, 1); return true;
        }
        if (rule == "additional.sloane-spend")
        {
            var token = AdditionalTokens(game, a.Defender, (DefenseTokenType)a.EffectFlags["sloane.pending.type"])
                .Single(token => token.Slot == a.EffectFlags["sloane.pending.slot"]);
            bool spent = game.IsShip(a.Defender) ? game.State.Ships[a.Defender].TryResolveDefenseToken(token, out _)
                : game.State.Squadrons[game.SquadronIndex(a.Defender)].TryResolveDefenseToken(token, out _);
            if (!spent) return false;
            a.EffectFlags["sloane.accuracy-used"] = 1; a.EffectFlags.Remove("sloane.pending.type"); a.EffectFlags.Remove("sloane.pending.slot");
        }
        if (rule == "additional.assault")
        {
            if (!game.State.Commands[a.Defender].TryGainRaidToken((ShipCommand)SuffixInt(id))) return false;
            a.EffectFlags["assault.used"] = 1;
        }
        MissingSquadronCombatEffects.RemoveDice(a, action.Index); Mark(a, id); ClearChoice(a);
        if (rule == "additional.ten-numb")
            foreach (var target in game.SceneSquadronViews.Where(target => target.Index != a.Defender
                && target.OwnerId != game.State.Owners[a.Attacker]
                && SquadronAbilityEngine.FootprintDistance(target, game.SquadronViews[game.SquadronIndex(a.Defender)]) <= 3.369803f)
                .OrderByDescending(target => target.RuntimeId.Value, StringComparer.Ordinal))
                CombatModule.QueueDamage(game, target.Index, 1, false, true, GamePhase.ModifyAttack, rule: "ten-numb.splash", source: a.Attacker);
        return true;
    }

    private static void BeginSquadronAi(GameContext game, AttackState a)
    {
        if (a.Counter || a.EffectFlags.ContainsKey("ai.gathered")) return;
        a.EffectFlags["ai.gathered"] = 1;
        var source = game.SquadronViews[game.SquadronIndex(a.Attacker)];
        if (!source.State.Commanded && !source.State.TemporaryCommanded
            && !SquadronModule.TreatsAsCommandedForAi(game, a.Attacker)) return;
        int count = SquadronAbilityEngine.GetKeywordValue(source.Definition, game.IsShip(a.Defender) ? "AI BATTERY" : "AI ANTISQUADRON");
        int[] colors = a.Dice.Select(die => (int)die.Color).Distinct().ToArray();
        if (count <= 0 || colors.Length == 0) return;
        if (MapAbilityEffects.HasInvisibleHandAiBonus(game, a.Attacker)) count++;
        Begin(a, "additional.ai-color", "keyword.ai.gather", source.OwnerId, colors, 1, count); a.EffectMaximum = count;
    }
}
