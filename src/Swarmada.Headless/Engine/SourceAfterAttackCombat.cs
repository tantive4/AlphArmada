using Armada.Core;
using Armada.Rules;

namespace Swarmada.Headless.Engine;

public sealed partial class SourceCombatEffects
{
    private static bool BeginFlechette(GameContext game, AttackState a)
    {
        if (a.MineSpatialIndex >= 0 || a.EffectFlags.ContainsKey("flechette.offered")) return false;
        a.EffectFlags["flechette.offered"] = 1;
        if (!game.IsShip(a.Attacker) || game.IsShip(a.Defender) || game.IsDestroyed(a.Defender)
            || game.State.Squadrons[game.SquadronIndex(a.Defender)].Activated || !game.HasUpgrade(a.Attacker, "flechette-torpedoes")) return false;
        int[] dice = Indices(a, die => die.Color == AttackDieColor.Black && die.Face is DieFace.Critical or DieFace.HitCritical).ToArray();
        if (dice.Length == 0) return false;
        Begin(a, "combat-extra.flechette", "flechette-torpedoes", game.State.Owners[a.Attacker], dice, 1, 1); return true;
    }
    private static bool BeginAfterAttackChoice(GameContext game, AttackState a)
    {
        foreach (string stage in new[] { "cluster", "quad-laser", "ruthless", "anakin-delta" })
        {
            if (a.EffectFlags.ContainsKey("after-attack.offered." + stage)) continue;
            a.EffectFlags["after-attack.offered." + stage] = 1;
            bool eligible = stage switch
            {
                "cluster" => !game.IsShip(a.Attacker) && game.IsShip(a.Defender) && game.HasUpgrade(a.Defender, "cluster-bombs"),
                "quad-laser" => !a.IsSalvoAttack && !a.Counter && game.IsShip(a.Defender) && !game.IsShip(a.Attacker)
                    && OffensiveRetrofitRules.GrantsCounterOneAgainstSquadron(game.ShipViews[a.Defender], game.SquadronViews[game.SquadronIndex(a.Attacker)]) && QuadLaserZones(game, a).Any(),
                "ruthless" => game.IsShip(a.Attacker) && !game.IsShip(a.Defender) && !game.IsDestroyed(a.Defender)
                    && game.HasUpgrade(a.Attacker, "ruthless-strategists") && RuthlessRecipients(game, a).Any(),
                "anakin-delta" => !game.IsShip(a.Attacker) && !game.IsShip(a.Defender) && !a.Counter && a.AppliedDamage > 0
                    && CloneWarsSquadronRules.Is(game.SquadronViews[game.SquadronIndex(a.Attacker)], CloneWarsSquadronRules.AnakinDelta)
                    && AnakinSplashTargets(game, a).Any(),
                _ => false
            };
            if (!eligible) continue;
            int owner = game.State.Owners[stage is "ruthless" or "anakin-delta" ? a.Attacker : a.Defender];
            Begin(a, stage == "quad-laser" ? "quad-laser-zone" : "combat-extra." + stage, stage, owner, [], 1, stage == "anakin-delta" ? 2 : 1); return true;
        }
        return false;
    }
    private static IEnumerable<HullZone> QuadLaserZones(GameContext game, AttackState a) => CombatModule.TargetZones(game, a.Defender)
        .Where(zone => OfficialAssaultObjectives.AllowsTarget(game, a.Defender, a.Attacker, zone, HullZone.Front)
            && SquadronAbilityEngine.HullZoneDistance(game.SquadronViews[game.SquadronIndex(a.Attacker)], game.ShipViews[a.Defender], zone) <= AttackResolver.SquadronAttackDistance
            && CombatModule.TryAttackGeometry(game, a.Defender, a.Attacker, zone, HullZone.Front, out _, out _, out _));
    private static IEnumerable<SquadronView> RuthlessRecipients(GameContext game, AttackState a) => game.SceneSquadronViews
        .Where(candidate => candidate.State.CurrentHull > 0 && candidate.OwnerId == game.State.Owners[a.Attacker]
            && SquadronAbilityEngine.IsEngaged(candidate, game.SquadronViews[game.SquadronIndex(a.Defender)]));
    private static IEnumerable<SquadronView> AnakinSplashTargets(GameContext game, AttackState a) => game.SceneSquadronViews
        .Where(candidate => candidate.Index != a.Defender && candidate.OwnerId != game.State.Owners[a.Attacker]
            && !a.EffectIndices.Contains(candidate.Index)
            && SquadronAbilityEngine.FootprintDistance(candidate, game.SquadronViews[game.SquadronIndex(a.Defender)]) <= 3.369803f);
    private static IEnumerable<GameActionOption> ExtraCombatChoices(GameContext game, AttackState a)
    {
        string stage = a.EffectRule["combat-extra.".Length..];
        if (stage == "flechette")
            foreach (int die in a.EffectEligibleIndices)
                yield return new(new(GameActionKind.ChooseDie, a.Attacker, a.Defender, Index: die, AbilityId: "flechette-torpedoes.die"), "Spend black critical die " + die + "; activate defender");
        else if (stage == "cluster")
            yield return new(new(GameActionKind.UseAbility, a.Defender, a.Attacker, AbilityId: "after-attack.cluster-bombs.use"), "Discard Cluster Bombs; roll four blue dice");
        else if (stage == "ruthless")
            foreach (var target in RuthlessRecipients(game, a))
                yield return new(new(GameActionKind.ChooseEntity, a.Attacker, target.Index, AbilityId: "ruthless-strategists.target"), "Damage " + target.Definition.DisplayName + " and defender");
        else if (stage == "anakin-delta")
            foreach (var target in AnakinSplashTargets(game, a))
                yield return new(new(GameActionKind.ChooseEntity, a.Attacker, target.Index, AbilityId: "clone-wars.anakin-delta.splash"), "Anakin splash damage to " + target.Definition.DisplayName);
        yield return new(new(GameActionKind.UseAbility, a.EffectOwner == game.State.Owners[a.Attacker] ? a.Attacker : a.Defender,
            AbilityId: "combat-extra.skip"), "Decline " + stage);
    }
    private static bool ApplyExtraCombatChoice(GameContext game, AttackState a, GameAction action)
    {
        string stage = a.EffectRule["combat-extra.".Length..];
        if (stage == "anakin-delta" && action.AbilityId != "combat-extra.skip")
        {
            a.EffectIndices = [..a.EffectIndices, action.Target]; a.EffectStage++;
            if (a.EffectStage >= 2 || !AnakinSplashTargets(game, a).Any()) ClearChoice(a);
            CombatModule.QueueDamage(game, action.Target, 1, false, true, GamePhase.ModifyAttack, rule: "anakin-delta.splash", source: a.Attacker); return true;
        }
        ClearChoice(a); if (action.AbilityId == "combat-extra.skip") return true;
        if (stage == "flechette")
        {
            MissingSquadronCombatEffects.RemoveDice(a, action.Index);
            game.State.Squadrons[game.SquadronIndex(a.Defender)].MarkActivated(false); return true;
        }
        if (stage == "cluster")
        {
            if (!Card(game, a.Defender, "cluster-bombs").Discard()) return false;
            CombatModule.QueueUnmodifiedDiceDamage(game, a.Defender, a.Attacker, 4, GamePhase.ModifyAttack, "cluster-bombs.explosion"); return true;
        }
        if (stage == "ruthless")
        {
            game.State.Squadrons[game.SquadronIndex(action.Target)].ApplyDamage(1);
            game.State.Squadrons[game.SquadronIndex(a.Defender)].ApplyDamage(1);
            EngineEffects.NotifyDamageApplied(game, a.Attacker, action.Target, GamePhase.ModifyAttack);
            EngineEffects.NotifyDamageApplied(game, a.Attacker, a.Defender, GamePhase.ModifyAttack); return true;
        }
        throw new NotSupportedException("Unknown source post-attack choice " + stage);
    }
}
