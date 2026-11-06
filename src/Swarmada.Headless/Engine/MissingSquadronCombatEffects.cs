using Armada.Core;
using Armada.Rules;

namespace Swarmada.Headless.Engine;

/// <summary>Official squadron rules absent from the recovered implementation. All choices
/// and physical die locks are carried by the existing snapshot-owned attack state.</summary>
public static class MissingSquadronCombatEffects
{
    public const string Vader = "darth-vader-tie-advanced-squadron", Jan = "jan-ors-moldy-crow",
        Kanan = "kanan-jarrus-hwk-290", Keyan = "keyan-farlander-b-wing-squadron",
        Lando = "lando-calrissian-millennium-falcon", Blount = "lieutenant-blount-z-95-headhunter-squadron",
        Malee = "malee-hurra-scurrg-h-6-bomber", Morna = "morna-kee-vt-49-decimator",
        Norra = "norra-wexley-y-wing-squadron", Nym = "nym-havoc",
        Soontir = "soontir-fel-tie-interceptor-squadron", Tel = "tel-trevura-jumpmaster-5000",
        Vult = "vult-skerris-tie-interceptor-squadron", Zertik = "zertik-strom-tie-advanced-squadron";
    internal static bool Is(GameContext game, int entity, string id) => entity >= game.State.Ships.Length
        && entity < game.Topology.EntityCount && game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition.SquadronId == id;
    public static bool IsDieModifiable(AttackState attack, int index) => !attack.EffectFlags.ContainsKey("lando.has-lock") || !attack.EffectFlags.ContainsKey("lando.lock." + index);
    internal static bool HasModifiableDice(AttackState attack)
    {
        if (!attack.EffectFlags.ContainsKey("lando.has-lock")) return attack.Dice.Length > 0;
        for (int index = 0; index < attack.Dice.Length; index++) if (IsDieModifiable(attack, index)) return true;
        return false;
    }
    internal static int ModifiableDiceCount(AttackState attack)
    {
        if (!attack.EffectFlags.ContainsKey("lando.has-lock")) return attack.Dice.Length;
        int count = 0;
        for (int index = 0; index < attack.Dice.Length; index++) if (IsDieModifiable(attack, index)) count++;
        return count;
    }
    internal static int[] ModifiableDice(AttackState attack) => Enumerable.Range(0, attack.Dice.Length).Where(index => IsDieModifiable(attack, index)).ToArray();
    internal static void RemoveDice(AttackState attack, params int[] removed)
    {
        if (removed.Length == 0) return;
        var locks = attack.EffectFlags.ContainsKey("lando.has-lock")
            ? attack.EffectFlags.Keys.Where(key => key.StartsWith("lando.lock.", StringComparison.Ordinal)).ToArray() : [];
        if (locks.Length > 0)
        {
            foreach (string key in locks) attack.EffectFlags.Remove(key);
            foreach (string key in locks)
            {
                int index = int.Parse(key["lando.lock.".Length..], System.Globalization.CultureInfo.InvariantCulture);
                if (!removed.Contains(index)) attack.EffectFlags["lando.lock." + (index - removed.Count(value => value < index))] = 1;
            }
        }
        attack.Dice = attack.Dice.Where((_, index) => !removed.Contains(index)).ToArray();
    }
    internal static IEnumerable<SquadronView> FriendlySources(GameContext game, int entity, string id, float maximum = 3.369803f)
        => game.SceneSquadronViews.Where(source => source.OwnerId == game.State.Owners[entity]
            && source.Definition.SquadronId == id && (game.IsShip(entity)
                ? SquadronAbilityEngine.FootprintDistance(source, game.ShipViews[entity])
                : SquadronAbilityEngine.FootprintDistance(source, game.SquadronViews[game.SquadronIndex(entity)])) <= maximum);
    public static bool DefersTelDestruction(GameContext game, int entity) => Is(game, entity, Tel)
        && game.Flow.Attacks.Any(attack => !attack.EnvironmentalDamage && attack.Defender == entity
            && attack.EffectFlags.ContainsKey("defender.discarded-token") && !attack.EffectFlags.ContainsKey("tel.resolved"));
    internal static bool HasCritical(AttackState a, AttackDieColor? color = null) => a.Dice.Any(die => (!color.HasValue || die.Color == color)
        && die.Face is DieFace.Critical or DieFace.HitCritical);
}

public sealed partial class SourceCombatEffects
{
    private static IEnumerable<AbilityAction> MissingSquadronActions(GameContext game, AttackState a, CombatWindow window)
    {
        bool hasDice = MissingSquadronCombatEffects.HasModifiableDice(a);
        if (window == CombatWindow.AttackEffects && hasDice)
        {
            if (!game.IsShip(a.Attacker))
            {
                if (MissingSquadronCombatEffects.Is(game, a.Attacker, MissingSquadronCombatEffects.Keyan)
                    && game.IsShip(a.Defender) && game.State.Ships[a.Defender].GetShields(a.ToZone) == 0)
                    yield return Ability("official.keyan.reroll", MissingSquadronCombatEffects.Keyan, "Keyan: reroll any attack dice");
                if (MissingSquadronCombatEffects.Is(game, a.Attacker, MissingSquadronCombatEffects.Lando)
                    && SquadronTokens(game, a.Attacker).Any())
                    yield return Ability("official.lando.attack", MissingSquadronCombatEffects.Lando, "Lando: discard a token and set one die permanently");
                if (MissingSquadronCombatEffects.Is(game, a.Attacker, MissingSquadronCombatEffects.Morna)
                    && SquadronTokens(game, a.Attacker).Any())
                    yield return Ability("official.morna.reroll", MissingSquadronCombatEffects.Morna, "Morna: spend a defense token to reroll dice");
                if (MissingSquadronCombatEffects.Is(game, a.Attacker, MissingSquadronCombatEffects.Zertik)
                    && ZertikRecipients(game, a).Any())
                    yield return Ability("official.zertik.reroll", MissingSquadronCombatEffects.Zertik, "Zertik: damage a friendly squadron and reroll dice");
                var source = game.SquadronViews[game.SquadronIndex(a.Attacker)];
                if (!game.IsShip(a.Defender) && (source.State.TemporarySwarm || SquadronAbilityEngine.HasKeyword(source.Definition, "SWARM"))
                    && MissingSquadronCombatEffects.FriendlySources(game, a.Attacker, MissingSquadronCombatEffects.Blount).Any(other => other.Index != a.Attacker))
                    yield return Ability("official.blount.reroll", MissingSquadronCombatEffects.Blount, "Blount: reroll one additional die");
                if (game.IsShip(a.Defender) && MissingSquadronCombatEffects.Is(game, a.Attacker, MissingSquadronCombatEffects.Kanan)
                    && Indices(a, die => die.Face is DieFace.Critical or DieFace.HitCritical).Any()
                    && Enum.GetValues<ShipCommand>().Any(game.State.Commands[a.Defender].CanGainRaidToken))
                    yield return Ability("official.kanan.raid", MissingSquadronCombatEffects.Kanan, "Kanan: spend a critical die to place raid");
            }
            else if (game.IsShip(a.Defender) && MissingSquadronCombatEffects.ModifiableDiceCount(a) >= 2 && EligibleMaleeSources(game, a).Any())
                yield return Ability("official.malee.change", MissingSquadronCombatEffects.Malee, "Malee: spend one die and change another to a critical face");
        }
        if (window == CombatWindow.DefenseEffects && !game.IsShip(a.Defender))
        {
            if (MissingSquadronCombatEffects.Is(game, a.Defender, MissingSquadronCombatEffects.Lando)
                && hasDice && SquadronTokens(game, a.Defender).Any(token => CombatModule.CanSpend(game, a, token)))
                yield return Ability("official.lando.defense", MissingSquadronCombatEffects.Lando, "Lando: spend a token to force chosen rerolls");
            if (!a.SpentTokenTypes.Contains(DefenseTokenType.Brace))
                foreach (var jan in MissingSquadronCombatEffects.FriendlySources(game, a.Defender, MissingSquadronCombatEffects.Jan, 5.4704595f)
                    .Where(source => source.Index != a.Defender && source.State.GetDefenseTokenInstances(DefenseTokenType.Brace)
                        .Any(token => CombatModule.CanSpend(game, a, token, ignoreAccuracy: true))))
                    yield return Ability("official.jan.borrow." + jan.Index, MissingSquadronCombatEffects.Jan, "Spend a Brace token belonging to Jan Ors");
        }
        if (window == CombatWindow.CriticalEffects && game.IsShip(a.Defender) && !game.Topology.Ships[a.Defender].IsStation && !a.EnvironmentalDamage
            && !game.IsShip(a.Attacker) && a.EffectFlags.GetValueOrDefault("critical.count") == 0 && !a.FirstHullCardFaceup
            && !OfficerCombatRules.PreventsCriticalEffects(game.ShipViews[a.Defender])
            && !(a.Contain && OfficerCombatRules.ContainPreventsAllCriticalEffects(game.ShipViews[a.Defender])))
        {
            if (MissingSquadronCombatEffects.Is(game, a.Attacker, MissingSquadronCombatEffects.Nym)
                && MissingSquadronCombatEffects.HasCritical(a, AttackDieColor.Blue) && SquadronOrShipTokens(game, a.Defender).Any())
                yield return Ability("official.nym.critical", MissingSquadronCombatEffects.Nym, "Nym: discard one physical enemy defense token");
            if (SquadronAbilityEngine.HasBomber(game.SquadronViews[game.SquadronIndex(a.Attacker)])
                && MissingSquadronCombatEffects.HasCritical(a) && game.State.Ships[a.Defender].GetShields(a.ToZone) > 0 && !a.IgnoreShields
                && MissingSquadronCombatEffects.FriendlySources(game, a.Attacker, MissingSquadronCombatEffects.Norra).Any())
                yield return Ability("official.norra.critical", MissingSquadronCombatEffects.Norra, "Norra: defending hull zone loses one shield");
        }
    }
    private static IEnumerable<DefenseTokenInstance> SquadronTokens(GameContext game, int entity)
        => Enum.GetValues<DefenseTokenType>().SelectMany(type => game.State.Squadrons[game.SquadronIndex(entity)].GetDefenseTokenInstances(type));
    private static IEnumerable<DefenseTokenInstance> SquadronOrShipTokens(GameContext game, int entity)
        => Enum.GetValues<DefenseTokenType>().SelectMany(type => game.IsShip(entity)
            ? game.State.Ships[entity].GetDefenseTokenInstances(type) : game.State.Squadrons[game.SquadronIndex(entity)].GetDefenseTokenInstances(type));
    private static IEnumerable<SquadronView> ZertikRecipients(GameContext game, AttackState a) => game.SceneSquadronViews.Where(source => source.Index != a.Attacker
        && source.OwnerId == game.State.Owners[a.Attacker] && SquadronAbilityEngine.FootprintDistance(source, game.SquadronViews[game.SquadronIndex(a.Attacker)]) <= 3.369803f);
    private static IEnumerable<SquadronView> EligibleMaleeSources(GameContext game, AttackState a) => game.SceneSquadronViews.Where(source => source.Definition.SquadronId == MissingSquadronCombatEffects.Malee
        && source.OwnerId == game.State.Owners[a.Attacker] && SquadronAbilityEngine.FootprintDistance(source, game.ShipViews[a.Defender]) <= 3.369803f
        && game.SceneSquadronViews.Count(enemy => SquadronAbilityEngine.IsEngaged(source, enemy)) <= 1
        && !game.Flow.Flags.ContainsKey(ActivationKey(game, a.Attacker, "malee." + source.Index)));

    private static bool TryMissingSquadronAction(GameContext game, AttackState a, GameAction action)
    {
        string id = action.AbilityId;
        if (!id.StartsWith("official.", StringComparison.Ordinal)) return false;
        int actor = a.Window == CombatWindow.DefenseEffects ? a.Defender : a.Attacker;
        int owner = game.State.Owners[actor];
        if (id is "official.keyan.reroll" or "official.blount.reroll")
        { Begin(a, "reroll", id, owner, MissingSquadronCombatEffects.ModifiableDice(a), 1, id == "official.blount.reroll" ? 1 : a.Dice.Length); return true; }
        if (id is "official.lando.attack" or "official.lando.defense" or "official.morna.reroll")
        { Begin(a, "official.token", id, owner, [], 1, 1); a.EffectValue = actor; return true; }
        if (id == "official.zertik.reroll") { Begin(a, "official.zertik", id, owner, [], 1, 1); return true; }
        if (id == "official.malee.change") { Begin(a, "official.malee-source", id, owner, [], 1, 1); return true; }
        if (id == "official.kanan.raid")
        { Begin(a, "official.kanan-die", id, owner, Indices(a, die => die.Face is DieFace.Critical or DieFace.HitCritical), 1, 1); return true; }
        if (id.StartsWith("official.jan.borrow.", StringComparison.Ordinal))
        { Begin(a, "official.token", id, owner, [], 1, 1); a.EffectValue = SuffixInt(id); return true; }
        if (id == "official.nym.critical") { Begin(a, "official.nym", id, owner, [], 1, 1); return true; }
        if (id == "official.norra.critical")
        { game.State.Ships[a.Defender].LoseShields(a.ToZone, 1); a.EffectFlags["critical.count"] = 1; Mark(a, id); return true; }
        return false;
    }

    private static IEnumerable<GameActionOption> MissingSquadronChoices(GameContext game, AttackState a)
    {
        string rule = a.EffectRule, id = a.EffectActionId;
        if (rule == "official.token")
            foreach (var token in SquadronTokens(game, a.EffectValue).Where(token => !id.StartsWith("official.jan.", StringComparison.Ordinal) || token.Token == DefenseTokenType.Brace)
                .Where(token => id.StartsWith("official.jan.", StringComparison.Ordinal) ? CombatModule.CanSpend(game, a, token, ignoreAccuracy: true)
                    : id != "official.lando.defense" || CombatModule.CanSpend(game, a, token)))
                yield return new(new(GameActionKind.ChooseToken, a.EffectValue, a.Defender, Token: token.Token, Index: token.Slot, AbilityId: "official.choose-token"), "Use " + token.Token + " slot " + token.Slot + (token.Exhausted ? " (exhausted)" : " (ready)"));
        else if (rule == "official.nym")
            foreach (var token in SquadronOrShipTokens(game, a.Defender))
                yield return new(new(GameActionKind.ChooseToken, a.Attacker, a.Defender, Token: token.Token, Index: token.Slot, AbilityId: "official.nym.discard"), "Discard enemy " + token.Token + " slot " + token.Slot);
        else if (rule == "official.zertik")
            foreach (var source in ZertikRecipients(game, a)) yield return new(new(GameActionKind.ChooseEntity, a.Attacker, source.Index, AbilityId: "official.zertik.target"), "Damage " + source.Definition.DisplayName);
        else if (rule == "official.malee-source")
            foreach (var source in EligibleMaleeSources(game, a)) yield return new(new(GameActionKind.ChooseEntity, a.Attacker, source.Index, AbilityId: "official.malee.source"), "Use Malee Hurra");
        else if (rule == "official.kanan-command")
            foreach (var command in Enum.GetValues<ShipCommand>().Where(game.State.Commands[a.Defender].CanGainRaidToken))
                yield return new(new(GameActionKind.ChooseCommand, a.Attacker, a.Defender, Command: command, AbilityId: "official.kanan.command"), "Place " + command + " raid");
        else if (rule == "official.lando-face")
            foreach (var face in Faces(a.Dice[a.EffectValue].Color)) yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, Amount: (int)face, AbilityId: "official.lando.face"), "Set protected " + face);
        else if (rule == "official.tel-heal")
            for (int amount = 1; amount <= Math.Min(3, game.Topology.Squadrons[game.SquadronIndex(a.Defender)].CatalogEntry.Definition.Hull - game.State.Squadrons[game.SquadronIndex(a.Defender)].CurrentHull); amount++)
                yield return new(new(GameActionKind.UseAbility, a.Defender, Amount: amount, AbilityId: "official.tel.heal"), "Recover " + amount + " hull");
        else
            foreach (int index in a.EffectEligibleIndices.Where(index => MissingSquadronCombatEffects.IsDieModifiable(a, index)))
                yield return new(new(GameActionKind.ChooseDie, a.Attacker, a.Defender, Index: index, AbilityId: "official.choose-die"), "Choose die " + index);
        if (rule is "official.tel-heal" or "official.zertik" or "official.malee-source")
            yield return new(new(GameActionKind.UseAbility, a.EffectOwner == game.State.Owners[a.Attacker] ? a.Attacker : a.Defender, AbilityId: "official.decline"), "Decline squadron effect");
    }

    private static bool ApplyMissingSquadronChoice(GameContext game, AttackState a, GameAction action)
    {
        string rule = a.EffectRule, id = a.EffectActionId; int owner = a.EffectOwner;
        if (action.AbilityId == "official.decline") { if (rule == "official.tel-heal") a.EffectFlags["tel.resolved"] = 1; ClearChoice(a); return true; }
        if (rule == "official.token")
        {
            int entity = a.EffectValue; var state = game.State.Squadrons[game.SquadronIndex(entity)];
            var token = state.GetDefenseTokenInstances(action.Token).Single(token => token.Slot == action.Index);
            if (id == "official.lando.attack")
            { if (!state.DiscardDefenseToken(token)) return false; Begin(a, "official.lando-die", id, owner, MissingSquadronCombatEffects.ModifiableDice(a), 1, 1); return true; }
            if (!state.TryResolveDefenseToken(token, out bool discarded)) return false;
            if (!discarded && entity == a.Defender && a.IntelOfficerTarget is { } intel
                && IntelOfficerTargetRules.MatchesSpend(intel, RuntimeUnitKind.Squadron, game.SquadronViews[game.SquadronIndex(entity)].RuntimeId, token))
                discarded = IntelOfficerTargetRules.TryDiscardAfterSpend(state, token);
            if (!discarded && RemainingUpgradeRules.PalpatinePunishesToken(game.State.Owners[entity], token.Token))
                discarded = state.DiscardDefenseToken(new DefenseTokenInstance(token.Token, token.Slot, true));
            if (entity == a.Defender)
            {
                a.SpentTokenTypes = [..a.SpentTokenTypes, token.Token]; a.EffectFlags["defender.spent-token"] = 1;
                if (discarded) a.EffectFlags["defender.discarded-token"] = 1;
            }
            if (id.StartsWith("official.jan.", StringComparison.Ordinal))
            {
                a.SpentTokenTypes = [..a.SpentTokenTypes, token.Token]; a.Brace = true;
                if (discarded) a.EffectFlags["exhausted-token-spends"] = a.EffectFlags.GetValueOrDefault("exhausted-token-spends") + 1;
                Mark(a, id); ClearChoice(a); return true;
            }
            if (discarded && entity == a.Defender) a.EffectFlags["exhausted-token-spends"] = a.EffectFlags.GetValueOrDefault("exhausted-token-spends") + 1;
            Begin(a, "reroll", id, owner, MissingSquadronCombatEffects.ModifiableDice(a), id == "official.morna.reroll" ? 0 : 1, a.Dice.Length); return true;
        }
        if (rule == "official.zertik")
        {
            game.State.Squadrons[game.SquadronIndex(action.Target)].ApplyDamage(1);
            EngineEffects.NotifyDamageApplied(game, a.Attacker, action.Target, game.Flow.Phase);
            Begin(a, "reroll", id, owner, MissingSquadronCombatEffects.ModifiableDice(a), 0, a.Dice.Length); return true;
        }
        if (rule == "official.malee-source")
        { a.EffectFlags["malee.selected"] = action.Target; Begin(a, "official.malee-spend", id, owner, MissingSquadronCombatEffects.ModifiableDice(a), 1, 1); return true; }
        if (rule == "official.malee-spend")
        {
            MissingSquadronCombatEffects.RemoveDice(a, action.Index);
            Begin(a, "change", id, owner, MissingSquadronCombatEffects.ModifiableDice(a), 1, 1);
            game.Flow.Flags[ActivationKey(game, a.Attacker, "malee." + a.EffectFlags["malee.selected"])] = 1; return true;
        }
        if (rule == "official.kanan-die")
        { a.EffectValue = action.Index; a.EffectRule = "official.kanan-command"; return true; }
        if (rule == "official.kanan-command")
        { if (!game.State.Commands[a.Defender].TryGainRaidToken(action.Command)) return false; MissingSquadronCombatEffects.RemoveDice(a, a.EffectValue); Mark(a, id); ClearChoice(a); return true; }
        if (rule == "official.lando-die") { a.EffectValue = action.Index; a.EffectRule = "official.lando-face"; return true; }
        if (rule == "official.lando-face")
        { a.Dice[a.EffectValue] = new(a.Dice[a.EffectValue].Color, (DieFace)action.Amount); a.EffectFlags["lando.lock." + a.EffectValue] = 1; a.EffectFlags["lando.has-lock"] = 1; Mark(a, id); ClearChoice(a); return true; }
        if (rule == "official.nym")
        { game.State.Ships[a.Defender].DiscardDefenseToken(game.State.Ships[a.Defender].GetDefenseTokenInstances(action.Token).Single(token => token.Slot == action.Index)); a.EffectFlags["critical.count"] = 1; Mark(a, id); ClearChoice(a); return true; }
        if (rule == "official.tel-heal")
        { game.State.Squadrons[game.SquadronIndex(a.Defender)].RecoverHullIncludingLethal(action.Amount, game.Topology.Squadrons[game.SquadronIndex(a.Defender)].CatalogEntry.Definition.Hull); a.EffectFlags["tel.resolved"] = 1; ClearChoice(a); return true; }
        throw new NotSupportedException("Unknown official squadron choice " + rule);
    }

    private static bool BeginMissingSquadronAfterAttack(GameContext game, AttackState a)
    {
        if (MissingSquadronCombatEffects.Is(game, a.Defender, MissingSquadronCombatEffects.Tel)
            && a.EffectFlags.ContainsKey("defender.discarded-token") && !a.EffectFlags.ContainsKey("tel.resolved"))
        {
            if (game.State.Squadrons[game.SquadronIndex(a.Defender)].CurrentHull < game.Topology.Squadrons[game.SquadronIndex(a.Defender)].CatalogEntry.Definition.Hull)
            { Begin(a, "official.tel-heal", "official.tel.heal", game.State.Owners[a.Defender], [], 1, 1); return true; }
            a.EffectFlags["tel.resolved"] = 1;
        }
        if (!a.Counter && !game.IsShip(a.Attacker) && !a.EffectFlags.ContainsKey("soontir.resolved"))
        {
            a.EffectFlags["soontir.resolved"] = 1;
            foreach (var soontir in game.SceneSquadronViews.Where(source => source.Definition.SquadronId == MissingSquadronCombatEffects.Soontir
                && source.Index != a.Defender && SquadronAbilityEngine.IsEngaged(source, game.SquadronViews[game.SquadronIndex(a.Attacker)])).ToArray())
                CombatModule.QueueDamage(game, a.Attacker, 1, false, true, GamePhase.ModifyAttack, rule: "soontir.after-attack", source: soontir.Index);
            if (!ReferenceEquals(game.Flow.Attacks[^1], a)) return true;
        }
        return false;
    }
}
