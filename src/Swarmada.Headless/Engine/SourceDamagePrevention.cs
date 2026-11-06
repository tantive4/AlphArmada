using Armada.Core;
using Armada.Rules;

namespace Swarmada.Headless.Engine;

public sealed partial class SourceCombatEffects
{
    private static int Incoming(AttackState a) => a.Damage + a.RedirectAllocations.Sum(entry => entry.Value);
    private static void ReduceIncoming(AttackState a, int amount)
    {
        int primary = Math.Min(a.Damage, amount); a.Damage -= primary; amount -= primary;
        var redirects = a.RedirectAllocations.ToArray();
        for (int index = 0; index < redirects.Length && amount > 0; index++)
        { int reduce = Math.Min(redirects[index].Value, amount); amount -= reduce; redirects[index] = new(redirects[index].Key, redirects[index].Value - reduce); }
        a.RedirectAllocations = redirects.Where(entry => entry.Value > 0).ToArray();
    }

    private static bool BeginDamagePrevention(GameContext game, AttackState a)
    {
        if (Incoming(a) <= 0 || game.IsDestroyed(a.Defender)) return false;
        string[] stages = game.IsShip(a.Defender)
            ? a.EnvironmentalDamage ? a.DamageRule is "upgrade-critical.wide-area-barrage" or "upgrade-critical.orbital-bombardment-particle-cannons" ? ["antilles"] : []
                : ["adi", "tranquility", "seventh", "antilles", "implacable"]
            : a.DamageRule is "biggs.transfer" or "cluster-bombs.explosion" ? [] : !a.EnvironmentalDamage && (!a.Counter || !game.IsShip(a.Attacker))
                && (!a.EffectFlags.ContainsKey("granted-attack") || a.EffectFlags.ContainsKey("granted-natural-attack"))
                ? ["implacable", "biggs"] : ["biggs"];
        foreach (string stage in stages)
        {
            if (a.EffectFlags.ContainsKey("damage-prevention.offered." + stage)) continue;
            a.EffectFlags["damage-prevention.offered." + stage] = 1;
            bool eligible = stage switch
            {
                "adi" => game.HasUpgrade(a.Defender, "adi-gallia-officer") && a.SpentTokenTypes.Length > 0 && a.Damage > 0 && AdiZones(game, a).Any(),
                "tranquility" => game.HasUpgrade(a.Defender, "tranquility-title") && a.SpentTokenTypes.Length < 2
                    && game.State.Ships[a.Defender].GetShields(a.ToZone) < 6 && TranquilityZones(game, a).Any(),
                "seventh" => a.ToZone == HullZone.Front && PreventionShips(game, a, stage).Any(),
                "antilles" or "implacable" => PreventionShips(game, a, stage).Any(),
                "biggs" => BiggsRecipients(game, a).Any(),
                _ => false
            };
            if (!eligible) continue;
            Begin(a, "damage-prevention." + stage, "damage-prevention." + stage, game.State.Owners[a.Defender], [], 1,
                stage == "adi" ? Math.Min(a.SpentTokenTypes.Length, a.Damage) : stage == "tranquility" ? 2 : 1);
            a.EffectValue = -1; return true;
        }
        return false;
    }

    private static IEnumerable<HullZone> AdiZones(GameContext game, AttackState a) => HullZoneRules.Adjacent(a.ToZone,
        game.State.Ships[a.Defender].HasAuxiliaryHullZones).Where(zone => game.State.Ships[a.Defender].GetShields(zone)
            > a.RedirectAllocations.Where(entry => entry.Key == zone).Sum(entry => entry.Value));
    private static IEnumerable<HullZone> TranquilityZones(GameContext game, AttackState a) => game.State.Ships[a.Defender].HullZones
        .Where(zone => zone != a.ToZone && game.State.Ships[a.Defender].GetShields(zone) > 0 && (a.EffectValue < 0 || a.EffectValue == (int)zone));
    private static IEnumerable<ShipView> PreventionShips(GameContext game, AttackState a, string stage)
    {
        string card = stage switch { "seventh" => "seventh-fleet-star-destroyer-title", "antilles" => "task-force-antilles-title", _ => "implacable-title" };
        if (stage == "antilles" && !game.HasUpgrade(a.Defender, card)) yield break;
        foreach (var source in game.SceneShipViews)
        {
            if (source.OwnerId != game.State.Owners[a.Defender] || stage != "implacable" && source.Index == a.Defender
                || source.Loadout.Find(card) is not { Active: true, Exhausted: false }) continue;
            float distance = game.IsShip(a.Defender) ? RaidRules.FootprintDistance(source, game.ShipViews[a.Defender])
                : SquadronAbilityEngine.FootprintDistance(game.SquadronViews[game.SquadronIndex(a.Defender)], source);
            float limit = stage switch { "seventh" => 10.7221f, "antilles" => 8.09628f, _ => 5.4704595f };
            if (distance > limit || stage == "implacable" && source.State.GetShields(HullZone.Front) == 0) continue;
            yield return source;
        }
    }
    private static IEnumerable<SquadronView> BiggsRecipients(GameContext game, AttackState a)
    {
        var target = game.SquadronViews[game.SquadronIndex(a.Defender)];
        if (!SquadronAbilityEngine.HasEscort(target)) yield break;
        var biggs = game.SceneSquadronViews.FirstOrDefault(squad => squad.OwnerId == target.OwnerId
            && squad.Definition.SquadronId == "biggs-darklighter-x-wing-squadron" && SquadronAbilityEngine.FootprintDistance(squad, target) <= 3.369803f);
        if (biggs is null) yield break;
        foreach (var candidate in game.SceneSquadronViews)
            if (candidate.Index != a.Defender && candidate.State.CurrentHull > 0 && candidate.OwnerId == biggs.OwnerId
                && SquadronAbilityEngine.HasEscort(candidate) && SquadronAbilityEngine.FootprintDistance(biggs, candidate) <= 3.369803f)
                yield return candidate;
    }
    private static IEnumerable<GameActionOption> DamagePreventionChoices(GameContext game, AttackState a)
    {
        string stage = a.EffectRule["damage-prevention.".Length..];
        if (stage is "adi" or "tranquility")
            foreach (var zone in stage == "adi" ? AdiZones(game, a) : TranquilityZones(game, a))
                yield return new(new(GameActionKind.ChooseZone, a.Defender, ToZone: zone, AbilityId: a.EffectActionId),
                    stage == "adi" ? "Move one damage to " + zone : "Move one shield from " + zone);
        else if (stage == "biggs")
            foreach (var recipient in BiggsRecipients(game, a))
                yield return new(new(GameActionKind.ChooseEntity, a.Defender, recipient.Index, AbilityId: a.EffectActionId), "Transfer one damage to " + recipient.Definition.DisplayName);
        else
            foreach (var source in PreventionShips(game, a, stage))
            {
                if (stage == "antilles")
                    foreach (var zone in source.State.HullZones)
                        yield return new(new(GameActionKind.ChooseEntity, a.Defender, source.Index, ToZone: zone, AbilityId: a.EffectActionId),
                            "Exhaust Task Force Antilles; transfer damage to " + source.Definition.DisplayName + " " + zone);
                else if (stage == "implacable")
                    for (int amount = 1; amount <= Math.Min(2, Math.Min(Incoming(a), source.State.GetShields(HullZone.Front))); amount++)
                        yield return new(new(GameActionKind.ChooseEntity, a.Defender, source.Index, Amount: amount, AbilityId: a.EffectActionId),
                            "Exhaust Implacable; spend " + amount + " front shields");
                else yield return new(new(GameActionKind.ChooseEntity, a.Defender, source.Index, AbilityId: a.EffectActionId), "Exhaust nearby Seventh Fleet");
            }
        yield return new(new(GameActionKind.UseAbility, a.Defender, AbilityId: "damage-prevention.skip"), "Finish " + stage);
    }
    private static bool ApplyDamagePrevention(GameContext game, AttackState a, GameAction action)
    {
        string stage = a.EffectRule["damage-prevention.".Length..];
        if (action.AbilityId == "damage-prevention.skip") { ClearChoice(a); return true; }
        if (stage == "adi")
        {
            a.Damage--; var values = a.RedirectAllocations.ToDictionary(entry => entry.Key, entry => entry.Value);
            values[action.ToZone] = values.GetValueOrDefault(action.ToZone) + 1; a.RedirectAllocations = values.ToArray(); a.EffectStage++;
            if (a.EffectStage >= a.EffectMaximum || a.Damage == 0 || !AdiZones(game, a).Any()) ClearChoice(a);
            return true;
        }
        if (stage == "tranquility")
        {
            game.State.Ships[a.Defender].TransferShield(action.ToZone, a.ToZone, false, 6); a.EffectValue = (int)action.ToZone; a.EffectStage++;
            if (a.EffectStage >= 2 || game.State.Ships[a.Defender].GetShields(a.ToZone) >= 6 || !TranquilityZones(game, a).Any()) ClearChoice(a);
            return true;
        }
        if (stage == "biggs")
        {
            ReduceIncoming(a, 1); ClearChoice(a);
            CombatModule.QueueDamage(game, action.Target, 1, false, true, GamePhase.DealDamage, rule: "biggs.transfer", source: a.Attacker); return true;
        }
        string card = stage switch { "seventh" => "seventh-fleet-star-destroyer-title", "antilles" => "task-force-antilles-title", _ => "implacable-title" };
        if (!Card(game, action.Target, card).Exhaust()) return false;
        int reduce = stage == "implacable" ? action.Amount : 1;
        if (stage == "implacable") game.State.Ships[action.Target].LoseShields(HullZone.Front, reduce);
        ReduceIncoming(a, reduce); ClearChoice(a);
        if (stage == "antilles") CombatModule.QueueDamage(game, action.Target, 1, false, false, GamePhase.DealDamage,
            action.ToZone, "antilles.transfer", source: a.Attacker);
        return true;
    }
}
