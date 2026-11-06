using Armada.Core;
using Armada.Rules;

namespace Swarmada.Headless.Engine;

public sealed partial class SourceCombatEffects
{
    private static bool BeginDeclarationChoice(GameContext game, AttackState a)
    {
        if (!game.IsShip(a.Defender)) return false;
        foreach (string stage in new[] { "barriss", "palpatine", "reeva", "hostage" })
        {
            if (a.EffectFlags.ContainsKey("declaration.offered." + stage)) continue;
            a.EffectFlags["declaration.offered." + stage] = 1;
            bool eligible = stage switch
            {
                "barriss" => game.HasUpgrade(a.Defender, "barriss-offee-officer") && AdditionalTokens(game, a.Defender, DefenseTokenType.Redirect).Any(),
                "palpatine" => a.Attacker >= 0 && game.HasUpgrade(a.Defender, "emperor-palpatine-officer")
                    && (game.IsShip(a.Attacker) || game.SquadronViews[game.SquadronIndex(a.Attacker)].Definition.IsUnique)
                    && DeclarationAttackerTokens(game, a).Any(),
                "reeva" => game.UpgradesFor(a.Defender).Any(card => card.Definition.id == "reeva-demesne-officer" && card.Active && !card.Exhausted)
                    && game.State.Ships[a.Defender].GetShields(a.ToZone) > 0 && ExhaustedTypes(game, a.Defender).Any(),
                "hostage" => game.IsShip(a.Attacker) && game.UpgradesFor(a.Defender).Any(card => card.Definition.id == "jedi-hostage" && card.Active && !card.Exhausted)
                    && Enum.GetValues<ShipCommand>().Any(game.State.Commands[a.Attacker].CanGainRaidToken),
                _ => false
            };
            if (!eligible) continue;
            int owner = game.State.Owners[stage is "palpatine" or "hostage" ? a.Attacker : a.Defender];
            Begin(a, "defense-timing." + stage, "defense-timing." + stage, owner, [], 1, 1); return true;
        }
        return false;
    }
    private static IEnumerable<DefenseTokenType> ExhaustedTypes(GameContext game, int entity) => Enum.GetValues<DefenseTokenType>()
        .Where(type => game.IsShip(entity) ? game.State.Ships[entity].GetAvailableDefenseTokenCount(type) > game.State.Ships[entity].GetReadyDefenseTokenCount(type)
            : game.State.Squadrons[game.SquadronIndex(entity)].GetAvailableDefenseTokenCount(type) > game.State.Squadrons[game.SquadronIndex(entity)].GetReadyDefenseTokenCount(type));
    private static IEnumerable<DefenseTokenInstance> DeclarationAttackerTokens(GameContext game, AttackState a) => Enum.GetValues<DefenseTokenType>()
        .SelectMany(type => AdditionalTokens(game, a.Attacker, type)).Where(token => game.IsShip(a.Attacker) || !token.Exhausted);
    private static bool BeginDefenseEndChoice(GameContext game, AttackState a)
    {
        if (a.EffectFlags.ContainsKey("barriss.redirect-spent") && !a.EffectFlags.ContainsKey("barriss.zone-chosen"))
        {
            Begin(a, "defense-timing.barriss-zone", "barriss-defense.zone", game.State.Owners[a.Defender], [], 1, 1); return true;
        }
        if (a.EffectFlags.ContainsKey("luminara.offered")) return false;
        a.EffectFlags["luminara.offered"] = 1;
        if (a.SpentTokenTypes.Length >= 2 || !game.IsShip(a.Defender) && !game.SquadronViews[game.SquadronIndex(a.Defender)].Definition.IsUnique
            || !game.SceneShipViews.Any(ship => ship.OwnerId == game.State.Owners[a.Defender] && ship.Loadout.HasActive("luminara-unduli-commander"))) return false;
        if (!LuminaraTargets(game, a).Any()) return false;
        Begin(a, "defense-timing.luminara", "luminara", game.State.Owners[a.Defender], [], 1, 1); return true;
    }
    private static IEnumerable<(int Entity, DefenseTokenType Token)> LuminaraTargets(GameContext game, AttackState a)
    {
        foreach (var type in ExhaustedTypes(game, a.Defender).Where(type => !a.SpentTokenTypes.Contains(type))) yield return (a.Defender, type);
        foreach (var other in game.SceneShipViews.Where(other => other.Index != a.Defender && other.OwnerId == game.State.Owners[a.Defender]))
        {
            float distance = game.IsShip(a.Defender) ? RaidRules.FootprintDistance(other, game.ShipViews[a.Defender])
                : SquadronAbilityEngine.FootprintDistance(game.SquadronViews[game.SquadronIndex(a.Defender)], other);
            if (distance <= 13.260394f) foreach (var type in ExhaustedTypes(game, other.Index)) yield return (other.Index, type);
        }
    }
    private static IEnumerable<GameActionOption> DefenseTimingChoices(GameContext game, AttackState a)
    {
        string stage = a.EffectRule["defense-timing.".Length..];
        if (stage == "hostage")
            foreach (var command in Enum.GetValues<ShipCommand>().Where(game.State.Commands[a.Attacker].CanGainRaidToken))
                yield return new(new(GameActionKind.ChooseCommand, a.Attacker, Command: command, AbilityId: "jedi-hostage.raid"), "Gain " + command + " raid to declare attack");
        else if (stage is "barriss" or "palpatine")
            foreach (var token in stage == "palpatine" ? DeclarationAttackerTokens(game, a) : AdditionalTokens(game, a.Defender, DefenseTokenType.Redirect))
                yield return new(new(GameActionKind.ChooseToken, stage == "palpatine" ? a.Attacker : a.Defender, Token: token.Token, Index: token.Slot, AbilityId: a.EffectActionId),
                    "Spend " + token.Token + " physical token " + token.Slot + (token.Exhausted ? " (exhausted)" : " (ready)"));
        else if (stage == "reeva")
            foreach (var type in ExhaustedTypes(game, a.Defender))
                yield return new(new(GameActionKind.ChooseToken, a.Defender, Token: type, AbilityId: a.EffectActionId), "Exhaust Reeva; ready " + type);
        else if (stage == "barriss-zone")
            foreach (var zone in game.State.Ships[a.Defender].HullZones.Where(zone => zone != a.ToZone))
                yield return new(new(GameActionKind.ChooseZone, a.Defender, ToZone: zone, AbilityId: a.EffectActionId), "Defend with " + zone);
        else if (stage == "luminara")
            foreach (var target in LuminaraTargets(game, a))
                yield return new(new(GameActionKind.ChooseToken, a.Defender, target.Entity, Token: target.Token, AbilityId: a.EffectActionId), "Luminara ready entity " + target.Entity + " " + target.Token);
        if (stage is "barriss" or "reeva" or "luminara")
            yield return new(new(GameActionKind.UseAbility, a.Defender, AbilityId: "defense-timing.skip"), "Decline " + stage);
    }
    private bool ApplyDefenseTimingChoice(GameContext game, AttackState a, GameAction action)
    {
        string stage = a.EffectRule["defense-timing.".Length..];
        if (action.AbilityId != "defense-timing.skip")
        {
            if (stage == "hostage") game.State.Commands[a.Attacker].TryGainRaidToken(action.Command);
            else if (stage == "barriss-zone") { a.ToZone = action.ToZone; a.EffectFlags["barriss.zone-chosen"] = 1; }
            else if (stage == "luminara")
            { if (game.IsShip(action.Target)) game.State.Ships[action.Target].TryReadyDefenseToken(action.Token); else game.State.Squadrons[game.SquadronIndex(action.Target)].TryReadyDefenseToken(action.Token); }
            else if (stage == "reeva") { Card(game, a.Defender, "reeva-demesne-officer").Exhaust(); game.State.Ships[a.Defender].TryReadyDefenseToken(action.Token); }
            else
            {
                int entity = stage == "palpatine" ? a.Attacker : a.Defender;
                var token = AdditionalTokens(game, entity, action.Token).Single(token => token.Slot == action.Index);
                bool discarded;
                bool spent = game.IsShip(entity) ? game.State.Ships[entity].TryResolveDefenseToken(token, out discarded)
                    : game.State.Squadrons[game.SquadronIndex(entity)].TryResolveDefenseToken(token, out discarded);
                if (!spent) return false;
                if (stage == "barriss") { a.EffectFlags["barriss.redirect-spent"] = 1; if (discarded) a.EffectFlags["exhausted-token-spends"] = a.EffectFlags.GetValueOrDefault("exhausted-token-spends") + 1; }
            }
        }
        ClearChoice(a);
        if (stage is "barriss-zone" or "luminara") { game.Flow.Phase = GamePhase.SpendDefense; return true; }
        EnterWindow(game, a, CombatWindow.BeforeRoll); return true;
    }
}
