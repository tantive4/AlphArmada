using Armada.Core;
using Armada.Data;
using Armada.Rules;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

/// <summary>Source action queries plus explicit costs, physical-die selections and chance
/// continuations. This class is stateless; every in-progress choice is part of AttackState.</summary>
public sealed partial class SourceCombatEffects : ICombatEffectHooks
{
    private static readonly HashSet<string> Supported = new(StringComparer.Ordinal)
    {
        "leading-shots", "caitken-and-shollan", "ordnance-experts", "admiral-screed-commander", "warlord-title",
        "h9-turbolasers", "home-one-title", "veteran-gunners", "sensor-team", "weapons-battery-techs",
        "linked-turbolaser-towers", "dual-turbolaser-turrets", "turbolaser-reroute-circuits", "external-racks",
        "defiance-title", "dominator-title", "patriot-fist-title", "swivel-mount-batteries", "admiral-ackbar-commander",
        "avenger-title", "thermal-shields", "electronic-countermeasures", "point-defense-ion-cannons",
        "targeting-scrambler", "lando-calrissian-officer", "captain-brunson-officer", "expert-shield-tech-officer",
        "admonition-title", "sa-nalaor-title", "reactive-gunnery", "major-derlin-officer", "foremans-labor-title",
        "rapid-reload", "high-capacity-ion-turbines", "slaved-turrets", "quad-battery-turrets", "tf-1726-commander",
        "mar-tuuk-commander", "paragon-title", "devastator-title", "salvation-title", "general-romodi-commander",
        "general-cracken-commander", "montferrat-officer", "lucid-voice-title",
        "governor-pryce-officer", "gunnery-chief-varnillian", "clone-captain-zak-officer",
        "instructor-goran-officer", "ti-99-officer",
        "darth-vader-commander", "darth-vader-officer", "kraken-commander", "director-krennic-officer", "task-force-organa-title",
        "toryn-farr-officer", "commander-woldar-officer", "bomber-command-center", "nevoota-bee-title",
        "agent-kallus-officer", "general-draven-officer", "intel-officer", "mon-karren-title",
        "heavy-ion-emplacements", "ion-cannon-batteries", "ms-1-ion-cannons", "nk-7-ion-cannons", "overload-pulse",
        "assault-concussion-missiles", "assault-proton-torpedoes", "xx-9-turbolasers", "fire-control-team",
        "dodonnas-pride-title", "sunder-title", "wide-area-barrage", "kyrsta-agate-officer",
        "orbital-bombardment-particle-cannons", "superheavy-composite-beam-turbolasers", "spha-t",
        "adi-gallia-officer", "tranquility-title", "seventh-fleet-star-destroyer-title", "task-force-antilles-title", "implacable-title",
        "commander-sato-commander", "admiral-sloane-commander", "shu-mai-officer", "clone-commander-wolffe-officer",
        "flight-controllers", "stronghold-title", "jamming-field", "flak-guns", "heavy-fire-zone", "insidious-title",
        "annihilator-title", "unity-title", "point-defense-reroute", "quad-turbolaser-cannons", "clone-gunners", "anakin-skywalker-commander", "dby-827-heavy-turbolaser",
        "emperor-palpatine-officer", "reeva-demesne-officer", "barriss-offee-officer", "jedi-hostage", "luminara-unduli-commander", "obi-wan-kenobi-commander",
        "flechette-torpedoes", "ruthless-strategists", "cluster-bombs"
    };
    private static readonly HashSet<string> CriticalCards = new(StringComparer.Ordinal)
    {
        "heavy-ion-emplacements", "ion-cannon-batteries", "ms-1-ion-cannons", "nk-7-ion-cannons", "overload-pulse",
        "assault-concussion-missiles", "assault-proton-torpedoes", "xx-9-turbolasers", "dodonnas-pride-title",
        "sunder-title", "wide-area-barrage", "kyrsta-agate-officer", "orbital-bombardment-particle-cannons", "superheavy-composite-beam-turbolasers"
    };

    public bool SupportsUpgrade(string id) => Supported.Contains(id);
    public bool HasPendingChoice(GameContext game, AttackState attack)
        => attack.EffectRule.Length > 0 && game.Flow.Phase != GamePhase.RollAttack;
    public int? DecisionPlayer(GameContext game, AttackState attack) => HasPendingChoice(game, attack) ? attack.EffectOwner : null;
    public bool AllowsStandardCritical(GameContext game, AttackState attack)
    {
        int limit = game.UpgradesFor(attack.Attacker).Any(card => card.Active && card.Definition.id == "fire-control-team" && !card.Exhausted) ? 2 : 1;
        return attack.EffectFlags.GetValueOrDefault("critical.count") < limit;
    }
    public bool TryFinishWindow(GameContext game, AttackState a, CombatWindow window)
    {
        if (a.Attacker < 0 && window is CombatWindow.AfterAttack or CombatWindow.AttackEffects) return false;
        if (window == CombatWindow.AfterAttack && BeginMissingSquadronAfterAttack(game, a)) return true;
        if (window == CombatWindow.BeforeDamage) return BeginDamagePrevention(game, a);
        if (window == CombatWindow.DefenseEffects) return BeginDefenseEndChoice(game, a);
        if (window == CombatWindow.AttackEffects) return BeginFlechette(game, a);
        if (window == CombatWindow.AfterAttack && BeginAfterAttackChoice(game, a)) return true;
        if (window == CombatWindow.AfterAttack && !a.EffectFlags.ContainsKey("after-attack.raid-offered"))
        {
            a.EffectFlags["after-attack.raid-offered"] = 1;
            if (RaidAbilityEffects.GetCombatActions(game, a, window).Any())
            { Begin(a, "after-attack-options", "after-attack-options", game.State.Owners[a.Attacker], [], 0, 0); return true; }
        }
        if (window != CombatWindow.BeforeRoll) return false;
        if (a.IsSalvoAttack) return false;
        if (game.IsShip(a.Attacker) && !game.IsShip(a.Defender) && !a.EffectFlags.ContainsKey("b2.flak.offered"))
        {
            a.EffectFlags["b2.flak.offered"] = 1;
            if (game.ShipViews[a.Attacker].Loadout.Find("b2-rocket-troopers") is { Active: true, Exhausted: false } && a.Dice.Length > 0)
            { Begin(a, "additional.b2-color", "b2.flak", game.State.Owners[a.Attacker], a.Dice.Select(die => (int)die.Color).Distinct(), 1, 1); return true; }
        }
        if (a.Attacker >= 0 && !game.IsShip(a.Attacker) && !a.Counter && !a.EffectFlags.ContainsKey("yavaris.applied"))
        {
            a.EffectFlags["yavaris.applied"] = 1;
            if (game.State.Squadrons[game.SquadronIndex(a.Attacker)].TemporaryYavarisBonusDie)
            {
                var color = a.Dice.Any(die => die.Color == AttackDieColor.Blue) ? AttackDieColor.Blue
                    : a.Dice.Any(die => die.Color == AttackDieColor.Black) ? AttackDieColor.Black
                    : a.Dice.Any(die => die.Color == AttackDieColor.Red) ? AttackDieColor.Red : AttackDieColor.None;
                if (color != AttackDieColor.None) AppendDice(a, color, 1);
            }
        }
        if (a.MineSpatialIndex < 0 && game.IsShip(a.Attacker) && !game.IsShip(a.Defender) && !a.EffectFlags.ContainsKey("flak-bonuses.offered"))
        {
            a.EffectFlags["flak-bonuses.offered"] = 1;
            int count = SquadronAbilityEngine.FlakDieBonusSources(game.ShipViews[a.Attacker], game.SquadronViews[game.SquadronIndex(a.Defender)], game.SceneSquadronViews).Count;
            if (count > 0)
            { Begin(a, "flak-color", "flak.add-die-color", game.State.Owners[a.Attacker], [], 1, count); return true; }
        }
        if (a.IsSalvoAttack || !game.IsShip(a.Defender) || a.EffectFlags.ContainsKey("thermal-window.offered")) return false;
        a.EffectFlags["thermal-window.offered"] = 1;
        if (!DefenseUpgradeRules.CanUseThermalShields(game.ShipViews[a.Defender], a.Dice.Length, a.Dice.Length)) return false;
        if (!game.State.Ships[a.Defender].GetDefenseTokenInstances(DefenseTokenType.Brace).Any(token => CombatModule.CanSpend(game, a, token))) return false;
        Begin(a, "thermal-offer", "thermal-shields.pre-roll", game.State.Owners[a.Defender], [], 1, 1); return true;
    }

    private static bool RequiresDefenseDie(string id) => id is "defense.upgrade.captain-brunson"
        or "defense.upgrade.point-defense-ion-cannons" or "defense.upgrade.lando-calrissian"
        || id.StartsWith("defense.upgrade.admonition.", StringComparison.Ordinal);

    public IEnumerable<GameActionOption> GetActions(GameContext game, AttackState a, CombatWindow window)
    {
        if (HasPendingChoice(game, a)) return ChoiceActions(game, a).ToArray();
        var result = new List<GameActionOption>();
        foreach (var source in QueryActions(game, a, window))
        {
            if (RequiresDefenseDie(source.AbilityId) && !MissingSquadronCombatEffects.HasModifiableDice(a)) continue;
            if (Used(a, source.AbilityId) && source.AbilityId != "defense.upgrade.targeting-scrambler") continue;
            if (source.AbilityId.StartsWith("defense.upgrade.electronic-countermeasures.", StringComparison.Ordinal)
                && (!ForcedDefenseTokenDiscardRules.TryParseChoiceId(source.AbilityId, "defense.upgrade.electronic-countermeasures.", out var forced)
                    || !CombatModule.CanSpend(game, a, forced, true))) continue;
            if (source.AbilityId.StartsWith("defense.upgrade.reactive-gunnery.", StringComparison.Ordinal)
                || source.AbilityId.StartsWith("defense.upgrade.sa-nalaor.", StringComparison.Ordinal) || source.AbilityId == "defense.upgrade.expert-shield-tech")
            {
                var type = source.AbilityId == "defense.upgrade.expert-shield-tech" ? DefenseTokenType.Redirect : (DefenseTokenType)SuffixInt(source.AbilityId);
                if (!game.State.Ships[a.Defender].GetDefenseTokenInstances(type).Any(token => CombatModule.CanSpend(game, a, token))) continue;
            }
            int actor = window == CombatWindow.DefenseEffects ? a.Defender : a.Attacker;
            result.Add(new(new(GameActionKind.UseAbility, actor, actor == a.Attacker ? a.Defender : a.Attacker,
                AbilityId: source.AbilityId), source.Label));
        }
        return result;
    }

    private static IEnumerable<AbilityAction> QueryActions(GameContext game, AttackState a, CombatWindow window)
    {
        foreach (var action in AssaultCombatActions(game, a, window)) yield return action;
        if (a.MineSpatialIndex >= 0)
        { foreach (var action in MineDiceActions(game, a, window)) yield return action; yield break; }
        if (a.Attacker >= 0) foreach (var action in MissingSquadronActions(game, a, window)) yield return action;
        var attack = Resolution(game, a);
        if (a.IsSalvoAttack && window == CombatWindow.BeforeRoll) yield break;
        if (a.IsSalvoAttack && window == CombatWindow.AttackEffects)
        {
            foreach (var action in UnitAbilityEngine.GetShipDiceActions(game.ShipViews[a.Attacker], attack, Used(a, "attack.dice.leading-shots"),
                a.EffectFlags.ContainsKey("caitken"), Used(a, "attack.dice.ordnance-experts"), true, Used(a, "attack.dice.warlord"), Used(a, "attack.dice.home-one"), !game.IsShip(a.Defender)))
                if (action.SourceId is "leading-shots" or "caitken-and-shollan" or "ordnance-experts" or "linked-turbolaser-towers"
                    or "h9-turbolasers" or "warlord-title" or "home-one-title" or "turbolaser-reroute-circuits") yield return action;
            if (game.HasUpgrade(a.Attacker, "dby-827-heavy-turbolaser") && a.Dice.Length > 0)
                yield return Ability("salvo.dby-827", "dby-827-heavy-turbolaser", "Change one Salvo die to a critical face");
            if (game.IsShip(a.Defender) && FleetCommandRules.AllowsSingleHitChange(game.ShipViews[a.Attacker]) && AttackResolver.HasDieWithoutSingleHit(attack))
                yield return Ability("fleet-command.attack.intensify-firepower", "intensify-firepower", "Change one die to a single hit");
            yield break;
        }
        if (window == CombatWindow.BeforeRoll && game.IsShip(a.Attacker))
        {
            var ship = game.ShipViews[a.Attacker];
            foreach (var action in UnitAbilityEngine.GetActions(ship, new AbilityQueryContext(AbilityTiming.BeforeAttack, a.Range))) yield return action;
            if (game.IsShip(a.Defender) && a.FromZone is HullZone.Left or HullZone.Right
                && CommanderRules.FleetHasCommander(ship, "admiral-ackbar-commander")
                && game.Flow.ShipActivations[a.Attacker].AttacksResolved == 0 && !game.Flow.Flags.ContainsKey(AckbarKey(game, a.Attacker)))
                yield return Ability("commander.admiral-ackbar.commit-broadsides", "admiral-ackbar-commander", "Commit to Ackbar broadsides");
            var avenger = ship.Loadout.Find("avenger-title");
            if (avenger is { Active: true, Exhausted: false }) yield return Ability("attack.before.avenger", "avenger-title", "Exhaust Avenger");
            if (game.IsShip(a.Defender) && RemainingUpgradeRules.FleetHas(ship, "commander-sato-commander") && a.Dice.Length > 0
                && RemainingUpgradeRules.HasFriendlySquadronNear(game.ShipViews[a.Defender], ship.OwnerId, 3.369803f))
                yield return Ability("remaining.attack-dice.sato.configure", "commander-sato-commander", "Choose up to two die-color replacements");
        }
        if (window == CombatWindow.AttackEffects)
        {
            if (a.Attacker < 0) yield break;
            if (!MissingSquadronCombatEffects.HasModifiableDice(a)) yield break;
            if (game.IsShip(a.Attacker))
            {
                var ship = game.ShipViews[a.Attacker];
                if (game.IsShip(a.Defender) && FleetCommandRules.AllowsSingleHitChange(ship) && AttackResolver.HasDieWithoutSingleHit(attack))
                    yield return Ability("fleet-command.attack.intensify-firepower", "intensify-firepower", "Change one die to a single hit");
                if (!game.IsShip(a.Defender))
                {
                    int rerolls = TitleRules.FlakRerolls(ship, game.SquadronViews[game.SquadronIndex(a.Defender)], game.SceneSquadronViews, out string source);
                    if (rerolls > 0 && a.Dice.Length > 0) yield return Ability("attack.dice.flak-title", source, "Reroll one anti-squadron die with " + source);
                    if (a.Range == RangeBand.Close && ship.Loadout.HasActive("point-defense-reroute") && a.Dice.Any(die => die.Face is DieFace.Critical or DieFace.HitCritical))
                        yield return Ability("attack.dice.point-defense-reroute", "point-defense-reroute", "Reroll critical results against squadron");
                }
                if (a.IsSalvoAttack && ship.Loadout.HasActive("dby-827-heavy-turbolaser") && a.Dice.Length > 0)
                    yield return Ability("salvo.dby-827", "dby-827-heavy-turbolaser", "Set one Salvo die to a critical face");
                if (CommanderRules.AttackRerollsFromSchedule(ship, game.State.Round) > 0 && a.Dice.Length > 0)
                    yield return Ability("attack.dice.shu-mai", "shu-mai-officer", "Reroll scheduled Shu Mai dice");
                foreach (var action in UnitAbilityEngine.GetShipDiceActions(ship, attack, Used(a, "attack.dice.leading-shots"),
                    a.EffectFlags.ContainsKey("caitken"), Used(a, "attack.dice.ordnance-experts"),
                    game.Flow.Flags.ContainsKey(ActivationKey(game, a.Attacker, "screed")), Used(a, "attack.dice.warlord"),
                    Used(a, "attack.dice.home-one"), !game.IsShip(a.Defender))) yield return action;
                if (game.IsShip(a.Defender) && !a.IsSalvoAttack)
                {
                    foreach (var action in UnitAbilityEngine.GetConditionalAttackDiceActions(ship, game.ShipViews[a.Defender], a.Range,
                        game.State.Activated[a.Defender], game.Flow.ShipActivations[a.Attacker].AttacksResolved == 0)) yield return action;
                    foreach (var action in AdvancedShipActions(game, a)) yield return action;
                }
                if (game.IsShip(a.Defender) && ship.Loadout.Find("governor-pryce-officer") is { Active: true, UsesThisActivation: 0 }
                    && ship.State.GetShields(a.FromZone) > 0)
                    yield return Ability("attack.dice.governor-pryce", "governor-pryce-officer", "Lose one shield and set a die face");
                if (ship.Loadout.Find("gunnery-chief-varnillian") is { Active: true, HasStoredDie: true, UsesThisAttack: 0 })
                    yield return Ability("attack.dice.gunnery-chief-varnillian", "gunnery-chief-varnillian", "Exchange stored Varnillian die");
                if (a.FromZone != HullZone.Front && !a.IsSalvoAttack && ship.Loadout.Find("clone-captain-zak-officer") is { Active: true, Exhausted: false })
                    foreach (var color in a.Dice.Select(die => die.Color).Distinct())
                        yield return Ability("attack.dice.clone-captain-zak." + (int)color, "clone-captain-zak-officer", "Add " + color + " die");
            }
            else
            {
                var source = game.SquadronViews[game.SquadronIndex(a.Attacker)];
                foreach (var action in AdditionalSquadronActions(game, a)) yield return action;
                foreach (var action in SquadronAbilityEngine.GetActions(source, new(AbilityTiming.ModifyAttackDice, a.Range,
                    moveUsed: !a.Counter && game.Flow.SquadronProgress is { Moved: true } progress && progress.Entity == a.Attacker,
                    isCounterAttack: a.Counter), game.SceneSquadronViews,
                    game.IsShip(a.Defender) ? game.ShipViews[a.Defender] : null,
                    game.IsShip(a.Defender) ? null : game.SquadronViews[game.SquadronIndex(a.Defender)])) yield return action;
                if (!game.IsShip(a.Defender) && SquadronAbilityEngine.HasSwarmRerollOpportunity(source,
                    game.SquadronViews[game.SquadronIndex(a.Defender)], game.SceneSquadronViews))
                    yield return Ability("keyword.swarm.reroll", "swarm", "Reroll one die with Swarm");
                if (SquadronAbilityEngine.GetDashRerollCount(source) > 0)
                    yield return Ability("squadron.dash-rendar.reroll", source.Definition.SquadronId, "Reroll a Dash Rendar die");
                if (!game.IsShip(a.Defender) && SquadronAbilityEngine.CommanderWoldarSource(source) is not null)
                    yield return Ability("squadron-upgrade-reroll.woldar", "commander-woldar-officer", "Woldar reroll one die");
                if (a.Dice.Any(die => die.Color == AttackDieColor.Blue) && SquadronAbilityEngine.TorynFarrSource(source) is not null)
                    yield return Ability("squadron-upgrade-reroll.toryn", "toryn-farr-officer", "Toryn Farr reroll one blue die");
                if (game.IsShip(a.Defender) && SquadronAbilityEngine.HasBomber(source) && SquadronUpgradeRules.BomberCommandCenterSource(source) is not null)
                    yield return Ability("squadron-upgrade-reroll.bomber-command-center", "bomber-command-center", "Bomber Command Center reroll one die");
                if (game.IsShip(a.Defender) && TitleRules.FindCommandingShip(source)?.Loadout.HasActive("nevoota-bee-title") == true
                    && (source.State.TemporarySwarm || SquadronAbilityEngine.HasKeyword(source.Definition, "SWARM")))
                    yield return Ability("squadron-upgrade-reroll.nevoota", "nevoota-bee-title", "Nevoota Bee reroll one die");
            }
        }
        if (window == CombatWindow.DefenseEffects)
        {
            if (game.IsShip(a.Defender))
            {
                var lockedTypes = a.LockedTokens.GroupBy(token => token.Token)
                    .Where(group => game.State.Ships[a.Defender].GetDefenseTokenInstances(group.Key).All(token => a.LockedTokens.Contains(token)))
                    .Select(group => group.Key).ToHashSet();
                foreach (var action in UnitAbilityEngine.GetShipDefenseActions(game.ShipViews[a.Defender], attack, a.ToZone,
                    lockedTypes, a.LockedTokens.ToHashSet(), a.SpentTokenTypes.ToHashSet(), !a.IsSalvoAttack && !a.Counter && !a.IgnitionAttack))
                    if (!action.AbilityId.StartsWith("defense.ship.", StringComparison.Ordinal)) yield return action;
            }
            else
            {
                if (!MissingSquadronCombatEffects.HasModifiableDice(a)) yield break;
                foreach (var action in SquadronAbilityEngine.GetDefenseActions(game.SquadronViews[game.SquadronIndex(a.Defender)],
                    a.LockedTokens.ToHashSet(), a.SpentTokenTypes.ToHashSet(), Used(a, "keyword.dodge.force-reroll"),
                    a.Attacker < 0 || game.IsShip(a.Attacker) ? null : game.SquadronViews[game.SquadronIndex(a.Attacker)], game.SceneSquadronViews))
                    if (action.AbilityId == "keyword.dodge.force-reroll") yield return action;
                var defender = game.SquadronViews[game.SquadronIndex(a.Defender)];
                if (CloneWarsSquadronRules.CanUseDfs311(defender, a.Obstructed))
                    yield return Ability("clone-wars.dfs-311.force-reroll", defender.Definition.SquadronId, "DFS-311 force one reroll");
                if (CloneWarsSquadronRules.Is(defender, CloneWarsSquadronRules.KitFisto)
                    && Enum.GetValues<DefenseTokenType>().Any(type => defender.State.GetAvailableDefenseTokenCount(type) > 0))
                    yield return Ability("clone-wars.kit-fisto.discard-token", defender.Definition.SquadronId, "Discard a token to reduce damage by three");
                foreach (var axe in CloneWarsSquadronRules.AxeProtectors(defender, game.SceneSquadronViews))
                    yield return Ability("clone-wars.axe." + axe.RuntimeId.Value, axe.Definition.SquadronId, "Axe spend a token to reduce damage");
            }
        }
        if (window == CombatWindow.CriticalEffects)
            foreach (var card in EligibleCriticals(game, a))
                yield return Ability("upgrade-critical." + card.Definition.id, card.Definition.id, card.Definition.name + " critical effect");
    }

    private static IEnumerable<AbilityAction> AdvancedShipActions(GameContext game, AttackState a)
    {
        var ship = game.ShipViews[a.Attacker]; var target = game.ShipViews[a.Defender];
        var activation = game.Flow.ShipActivations[a.Attacker];
        if (ship.Loadout.Find("clone-gunners") is { Active: true, UsesThisAttack: 0 }
            && (activation.ResolvedCommands & (1 << (int)ShipCommand.ConcentrateFire)) != 0)
            foreach (var other in game.SceneShipViews.Where(other => other.Index != a.Attacker && other.OwnerId == ship.OwnerId
                && other.Commands.HasToken(ShipCommand.ConcentrateFire) && RaidRules.FootprintDistance(other, target) <= 13.260394f))
                yield return Ability("remaining.attack-dice.clone-gunners." + other.RuntimeId.Value, "clone-gunners", "Discard nearby ship's CF token; add blue accuracy");
        var anakinSources = game.SceneShipViews.Where(other => other.Index != a.Attacker && other.OwnerId == ship.OwnerId
            && RaidRules.FootprintDistance(other, target) <= 13.260394f).ToArray();
        if (!a.EffectFlags.ContainsKey("anakin.used") && RemainingUpgradeRules.FleetHas(ship, "anakin-skywalker-commander") && anakinSources.Length >= 2)
            foreach (var other in anakinSources)
                foreach (var type in Enum.GetValues<DefenseTokenType>().Where(other.State.HasDefenseToken))
                    yield return Ability("remaining.attack-dice.anakin." + other.RuntimeId.Value + "." + (int)type, "anakin-skywalker-commander", "Spend nearby ship's " + type + "; reroll up to three dice");
        if (ship.Loadout.Find("director-krennic-officer") is { Active: true }
            && (activation.ResolvedCommands & (1 << (int)ShipCommand.ConcentrateFire)) != 0
            && a.Range is RangeBand.Medium or RangeBand.Long && a.Dice.Any(die => die.Color == AttackDieColor.Red))
            yield return Ability("attack.dice.director-krennic", "director-krennic-officer", "Krennic reroll red dice");
        if (CommanderRules.FleetHasCommander(ship, "darth-vader-commander"))
            foreach (var type in Enum.GetValues<DefenseTokenType>().Where(ship.State.HasDefenseToken))
                yield return Ability("attack.dice.vader-commander." + (int)type, "darth-vader-commander", "Vader spend " + type + " and reroll dice");
        if (ship.Loadout.HasActive("darth-vader-officer"))
            foreach (var card in ship.Loadout.Upgrades.Where(card => card.Active && card.Definition.id != "darth-vader-officer"
                && card.Definition.slot is "commander" or "officer"))
                yield return Ability("attack.dice.vader-officer." + card.Definition.id, "darth-vader-officer", "Discard " + card.Definition.name + " and reroll dice");
        if (!game.Flow.Flags.ContainsKey(ActivationKey(game, a.Attacker, "kraken")) && CommanderRules.FleetHasCommander(ship, "kraken-commander")
            && game.SceneShipViews.Any(other => other.Index != a.Attacker && other.OwnerId == ship.OwnerId && RaidRules.FootprintDistance(other, target) <= AttackResolver.MediumRange))
            yield return Ability("attack.dice.kraken", "kraken-commander", "Kraken set one icon");
        if (ship.Loadout.HasActive("task-force-organa-title"))
            foreach (var other in game.SceneShipViews.Where(other => other.Index != a.Attacker && other.OwnerId == ship.OwnerId
                && RaidRules.FootprintDistance(other, ship) <= 8.09628f && other.Loadout.Find("task-force-organa-title") is { Active: true, Exhausted: false }))
                yield return Ability("attack.dice.task-force-organa." + other.RuntimeId.Value, "task-force-organa-title", "Exhaust nearby Task Force Organa");
        if (a.Dice.Any(die => die.Color == AttackDieColor.Blue) && !a.EffectFlags.ContainsKey("toryn.ship.used"))
            foreach (var other in game.SceneShipViews.Where(other => other.Index != a.Attacker && other.OwnerId == ship.OwnerId
                && RaidRules.FootprintDistance(other, ship) <= 8.09628f && other.Loadout.HasActive("toryn-farr-officer")))
                yield return Ability("remaining.attack-dice.toryn-farr." + other.RuntimeId.Value, "toryn-farr-officer", "Toryn Farr reroll one blue die");
    }

    public void EnterWindow(GameContext game, AttackState a, CombatWindow window)
    {
        if (a.EnvironmentalDamage) return;
        if (window == CombatWindow.AttackEffects && BeginAssaultInitialRoll(game, a)) return;
        if (window == CombatWindow.BeforeRoll && !a.IsSalvoAttack && !a.Counter && BeginDeclarationChoice(game, a)) return;
        if (window == CombatWindow.BeforeRoll && a.Attacker >= 0 && !game.IsShip(a.Attacker)) BeginSquadronAi(game, a);
        if (window == CombatWindow.BeforeRoll && game.IsShip(a.Attacker))
        {
            if (!a.Counter && TurbolaserRules.IsSwivelMountAdjacentAttack(game.ShipViews[a.Attacker], a.FromZone)) a.MandatoryRemovals++;
        }
        if (window == CombatWindow.AttackEffects && game.IsShip(a.Attacker))
        {
            if (!a.EffectFlags.ContainsKey("intel-window.offered"))
            {
                a.EffectFlags["intel-window.offered"] = 1;
                if (game.ShipViews[a.Attacker].Loadout.Find("intel-officer") is { Active: true, Exhausted: false }
                    && IntelTargets(game, a).Any())
                { Begin(a, "intel-officer", "attack.dice.intel-officer.", game.State.Owners[a.Attacker], [], 1, 1); return; }
            }
            BeginPostRollChoices(game, a);
        }
        if (window == CombatWindow.BeforeDamage && a.EffectFlags.GetValueOrDefault("damage-reduction") > 0)
            ReduceIncoming(a, (int)a.EffectFlags["damage-reduction"]);
        if (window == CombatWindow.BeforeDamage && a.Redirect && game.IsShip(a.Defender)
            && CommanderRules.FleetHasCommander(game.ShipViews[a.Defender], "obi-wan-kenobi-commander")) ReduceIncoming(a, 1);
        if (window == CombatWindow.BeforeDamage && a.EffectFlags.GetValueOrDefault("critical.count") + (a.FirstHullCardFaceup ? 1 : 0) > 1)
            game.UpgradesFor(a.Attacker).FirstOrDefault(card => card.Active && card.Definition.id == "fire-control-team")?.Exhaust();
        if (window == CombatWindow.AfterAttack && game.IsShip(a.Attacker))
            OfficerCombatRules.DiscardGherantAfterShipAttack(game.ShipViews[a.Attacker]);
    }

    private static void BeginPostRollChoices(GameContext game, AttackState a)
    {
        if (!a.EffectFlags.ContainsKey("mar-tuuk.offered"))
        {
            a.EffectFlags["mar-tuuk.offered"] = 1;
            if (game.IsShip(a.Defender) && game.State.Ships[a.Defender].HullZones.All(zone => game.State.Ships[a.Defender].GetShields(zone) == 0)
                && RemainingUpgradeRules.FleetHas(game.ShipViews[a.Attacker], "mar-tuuk-commander") && a.Dice.Length > 0)
            { Begin(a, "remove", "mar-tuuk.cancel", game.State.Owners[a.Attacker], Enumerable.Range(0, a.Dice.Length), 1, 1); return; }
        }
        if (!a.EffectFlags.ContainsKey("swivel.focus.offered"))
        {
            a.EffectFlags["swivel.focus.offered"] = 1;
            if (!a.IsSalvoAttack && game.IsShip(a.Defender) && TurbolaserRules.GetSwivelMountFocusBonusColors(game.ShipViews[a.Attacker], a.FromZone).Count > 0)
            { Begin(a, "swivel-color", "swivel-mount.color", game.State.Owners[a.Attacker], [], 1, 1); return; }
        }
        if (!a.EffectFlags.ContainsKey("initial-roll.fixed-bonuses"))
        {
            a.EffectFlags["initial-roll.fixed-bonuses"] = 1;
            if (!a.IsSalvoAttack && game.IsShip(a.Defender))
            {
                a.Dice = SquadronAbilityEngine.ApplyCaptainJonus(game.ShipViews[a.Attacker], game.ShipViews[a.Defender], Resolution(game, a)).Dice.ToArray();
                if (game.HasUpgrade(a.Attacker, "quad-turbolaser-cannons") && a.Dice.Any(die => die.Color == AttackDieColor.Red && die.Face == DieFace.Accuracy))
                    a.Dice = [..a.Dice, new AttackDie(AttackDieColor.Red, DieFace.Accuracy)];
            }
        }
    }

    public void AfterDiceChance(GameContext game, AttackState attack)
    {
        if (game.Flow.Phase != GamePhase.RollAttack && attack.EffectFlags.Remove("objective.opening.resume"))
        { EnterWindow(game, attack, CombatWindow.AttackEffects); return; }
        if (attack.EffectFlags.Remove("swivel.focus.awaiting-chance")) BeginPostRollChoices(game, attack);
    }

    private static IEnumerable<IntelOfficerTokenTarget> IntelTargets(GameContext game, AttackState a)
    {
        var kind = game.IsShip(a.Defender) ? RuntimeUnitKind.Ship : RuntimeUnitKind.Squadron;
        var id = game.IsShip(a.Defender) ? game.ShipViews[a.Defender].RuntimeId : game.SquadronViews[game.SquadronIndex(a.Defender)].RuntimeId;
        foreach (var type in Enum.GetValues<DefenseTokenType>())
            foreach (var token in game.IsShip(a.Defender) ? game.State.Ships[a.Defender].GetDefenseTokenInstances(type)
                : game.State.Squadrons[game.SquadronIndex(a.Defender)].GetDefenseTokenInstances(type))
                yield return new(kind, id, token);
    }

    public bool TryApply(GameContext game, AttackState a, CombatWindow window, GameAction action)
    {
        if (HasPendingChoice(game, a)) return ApplyChoice(game, a, action);
        string id = action.AbilityId;
        // A mandatory die choice must remain possible before paying its cost.
        // Evade/Scatter or an earlier cancel can leave the attack pool empty.
        if (RequiresDefenseDie(id) && !MissingSquadronCombatEffects.HasModifiableDice(a)) return false;
        if (TryApplyAssaultCombat(game, a, action)) return true;
        if (id.StartsWith("mine-attack.ltt.", StringComparison.Ordinal))
        { if (!Card(game, a.Attacker, "linked-turbolaser-towers").Exhaust()) return false; return AddRolled(game, a, (AttackDieColor)SuffixInt(id), 2); }
        if (RaidAbilityEffects.TryCombatApply(game, a, window, action)) return true;
        if (TryMissingSquadronAction(game, a, action)) return true;
        if (TryBeginAdditionalCombatChoice(game, a, action)) return true;
        if (id.StartsWith("upgrade-critical.", StringComparison.Ordinal)) return BeginCritical(game, a, id["upgrade-critical.".Length..]);
        if (id == "upgrade.external-racks.add-black-dice")
        {
            int count = UnitAbilityEngine.UseExternalRacks(game.ShipViews[a.Attacker], a.Range);
            if (count <= 0) return false;
            AppendDice(a, AttackDieColor.Black, count); a.EffectFlags["used." + id] = 1; return true;
        }
        if (id == "commander.admiral-ackbar.commit-broadsides")
        { game.Flow.Flags[AckbarKey(game, a.Attacker)] = 1; AppendDice(a, AttackDieColor.Red, 2); Mark(a, id); return true; }
        if (id == "attack.before.avenger")
        { if (!Card(game, a.Attacker, "avenger-title").Exhaust()) return false; a.EffectFlags["avenger"] = 1; Mark(a, id); return true; }
        var all = MissingSquadronCombatEffects.ModifiableDice(a);
        int owner = game.State.Owners[window == CombatWindow.DefenseEffects ? a.Defender : a.Attacker];
        switch (id)
        {
            case "attack.dice.leading-shots": Begin(a, "spend-reroll", id, owner, Indices(a, die => die.Color == AttackDieColor.Blue), 1, 1); return true;
            case "attack.dice.admiral-screed":
            case "attack.dice.sensor-team": Begin(a, "spend-change", id, owner, all, 1, 1); return true;
            case "attack.dice.ordnance-experts": Begin(a, "reroll", id, owner, Indices(a, die => die.Color == AttackDieColor.Black), 1, 2); return true;
            case "attack.dice.veteran-gunners":
                if (!Card(game, a.Attacker, "veteran-gunners").Exhaust()) return false;
                Mark(a, id); Reroll(game, a, all); return true;
            case "attack.dice.warlord": Begin(a, "face", id, owner, Indices(a, die => die.Face == DieFace.Accuracy), 1, 1); return true;
            case "attack.dice.h9-turbolasers": Begin(a, "change", id, owner, Indices(a, die => die.Color != AttackDieColor.Black && die.Face is DieFace.Hit or DieFace.DoubleHit or DieFace.Critical or DieFace.HitCritical), 1, 1); return true;
            case "attack.dice.home-one": Begin(a, "change", id, owner, Indices(a, die => die.Color != AttackDieColor.Black), 1, 1); return true;
            case "attack.dice.linked-turbolaser-towers": Begin(a, "reroll", id, owner, Indices(a, die => die.Color == AttackDieColor.Red), 1, 1); return true;
            case "attack.dice.dual-turbolaser-turrets":
                if (!Card(game, a.Attacker, "dual-turbolaser-turrets").Exhaust()) return false;
                AppendDice(a, AttackDieColor.Red, 1); Mark(a, id);
                Begin(a, "remove", "dual-turbolaser-remove", owner, Enumerable.Range(0, a.Dice.Length), 1, 1);
                Reroll(game, a, [a.Dice.Length - 1]); return true;
            case "attack.dice.governor-pryce": Begin(a, "face", id, owner, all, 1, 1); return true;
            case "attack.dice.kraken": Begin(a, "face", id, owner, all, 1, 1); return true;
            case "attack.dice.director-krennic":
                int maximum = (game.Flow.ShipActivations[a.Attacker].ResolvedDialCommands & (1 << (int)ShipCommand.ConcentrateFire)) != 0 ? all.Length : 2;
                Begin(a, "reroll", id, owner, Indices(a, die => die.Color == AttackDieColor.Red), 1, maximum); return true;
            case "attack.dice.shu-mai":
                Begin(a, "reroll", id, owner, all, 1, CommanderRules.AttackRerollsFromSchedule(game.ShipViews[a.Attacker], game.State.Round)); return true;
            case "attack.dice.flak-title": Begin(a, "reroll", id, owner, all, 1, 1); return true;
            case "attack.dice.point-defense-reroute": Begin(a, "reroll", id, owner, Indices(a, die => die.Face is DieFace.Critical or DieFace.HitCritical), 1, all.Length); return true;
            case "salvo.dby-827": Begin(a, "change", id, owner, all, 1, 1); return true;
            case "fleet-command.attack.intensify-firepower": Begin(a, "change", id, owner, Indices(a, die => die.Face != DieFace.Hit), 1, 1); return true;
            case "attack.dice.gunnery-chief-varnillian": Begin(a, "exchange", id, owner, all, 1, 1); return true;
            case "attack.dice.patriot-fist":
                if (!Card(game, a.Attacker, "patriot-fist-title").TryUseThisActivation()) return false;
                Mark(a, id); return AddRolled(game, a, AttackDieColor.Blue, 2);
            case "squadron.bossk.add-accuracy": a.Dice = [..a.Dice, new AttackDie(AttackDieColor.Blue, DieFace.Accuracy)]; Mark(a, id); return true;
            case "squadron.maarek-stele.set-critical": Begin(a, "change", id, owner, all, 1, 1); return true;
            case "clone-wars.odd-ball.reroll": Begin(a, "reroll", id, owner, all, 1, 2); return true;
            case "clone-wars.baktoid.set-accuracy": Begin(a, "change", id, owner, Indices(a, die => die.Color != AttackDieColor.Black), 1, 1); return true;
            case "clone-wars.dfs-311.force-reroll": Begin(a, "reroll", id, owner, all, 1, 1); return true;
            case "clone-wars.dbs-404.add-die": Begin(a, "dbs-color", id, owner, [], 1, 1); return true;
            case "clone-wars.jango.change-face": Begin(a, "jango-ally", id, owner, [], 1, 1); return true;
            case "clone-wars.kit-fisto.discard-token":
                Begin(a, "squad-defense-token", id, owner, [], 1, 1); a.EffectValue = a.Defender; return true;
            case "keyword.adept.reroll":
                Begin(a, "reroll", id, owner, all, 1, SquadronAbilityEngine.GetEffectiveAdept(game.SquadronViews[game.SquadronIndex(a.Attacker)], game.SceneSquadronViews)); return true;
            case "keyword.swarm.reroll":
            case "squadron-upgrade-reroll.woldar":
            case "squadron-upgrade-reroll.bomber-command-center":
            case "squadron-upgrade-reroll.nevoota": Begin(a, "reroll", id, owner, all, 1, 1); return true;
            case "squadron-upgrade-reroll.toryn": Begin(a, "reroll", id, owner, Indices(a, die => die.Color == AttackDieColor.Blue), 1, 1); return true;
            case "squadron.dash-rendar.reroll":
                Begin(a, "reroll", id, owner, all, 1, SquadronAbilityEngine.GetDashRerollCount(game.SquadronViews[game.SquadronIndex(a.Attacker)])); return true;
            case "keyword.dodge.force-reroll":
                Begin(a, "reroll", id, owner, all, 1, SquadronAbilityEngine.GetDodgeValue(game.SquadronViews[game.SquadronIndex(a.Defender)],
                    game.IsShip(a.Attacker) ? null : game.SquadronViews[game.SquadronIndex(a.Attacker)], game.SceneSquadronViews)); return true;
            case "defense.upgrade.point-defense-ion-cannons": Begin(a, "reroll", id, owner, all, 1, 1); return true;
            case "defense.upgrade.targeting-scrambler": Begin(a, "reroll", id, owner, all, 0, 4); return true;
            case "defense.upgrade.lando-calrissian": Begin(a, "reroll", id, owner, all, 1, all.Length); return true;
            case "defense.upgrade.captain-brunson": Begin(a, "remove", id, owner, all, 1, 1); return true;
            case "defense.upgrade.major-derlin.reduce-damage":
            case "defense.upgrade.foremans-labor.reduce-damage":
                if (!Card(game, a.Defender, id.Contains("major-derlin", StringComparison.Ordinal) ? "major-derlin-officer" : "foremans-labor-title").Exhaust()) return false;
                a.EffectFlags["damage-reduction"] = a.EffectFlags.GetValueOrDefault("damage-reduction") + 1; Mark(a, id); return true;
        }
        if (id.StartsWith("attack.dice.caitken-and-shollan.", StringComparison.Ordinal))
        { Begin(a, "reroll", id, owner, Indices(a, die => die.Color == (AttackDieColor)SuffixInt(id)), 1, all.Length); return true; }
        if (id.StartsWith("attack.dice.weapons-battery-techs.", StringComparison.Ordinal))
        { Begin(a, "change", id, owner, Indices(a, die => die.Color == (AttackDieColor)SuffixInt(id) && die.Face == DieFace.Accuracy), 1, 1); return true; }
        if (id.StartsWith("attack.dice.vader-commander.", StringComparison.Ordinal))
        { Begin(a, "vader-token", id, owner, [], 1, 1); return true; }
        if (id.StartsWith("remaining.attack-dice.clone-gunners.", StringComparison.Ordinal))
        {
            int provider = game.Topology.EntityByRuntimeId[id["remaining.attack-dice.clone-gunners.".Length..]];
            if (!game.State.Commands[provider].TrySpendToken(ShipCommand.ConcentrateFire) || !Card(game, a.Attacker, "clone-gunners").TryUseThisAttack()) return false;
            a.Dice = [..a.Dice, new AttackDie(AttackDieColor.Blue, DieFace.Accuracy)]; Mark(a, id); return true;
        }
        if (id.StartsWith("remaining.attack-dice.anakin.", StringComparison.Ordinal))
        {
            int split = id.LastIndexOf('.'); int provider = game.Topology.EntityByRuntimeId[id["remaining.attack-dice.anakin.".Length..split]];
            Begin(a, "anakin-token", id, owner, [], 1, 1); a.EffectValue = provider; return true;
        }
        if (id.StartsWith("attack.dice.vader-officer.", StringComparison.Ordinal))
        {
            var card = Card(game, a.Attacker, id["attack.dice.vader-officer.".Length..]);
            if (!UnitAbilityEngine.DiscardUpgradeFromShip(game.ShipViews[a.Attacker], card, out _)) return false;
            Begin(a, "reroll", id, owner, all, 0, all.Length); return true;
        }
        if (id.StartsWith("attack.dice.task-force-organa.", StringComparison.Ordinal))
        {
            int entity = game.Topology.EntityByRuntimeId[id["attack.dice.task-force-organa.".Length..]];
            if (!Card(game, entity, "task-force-organa-title").Exhaust()) return false;
            Begin(a, "reroll", id, owner, all, 0, Math.Min(2, all.Length)); return true;
        }
        if (id.StartsWith("remaining.attack-dice.toryn-farr.", StringComparison.Ordinal))
        { Begin(a, "reroll", id, owner, Indices(a, die => die.Color == AttackDieColor.Blue), 1, 1); a.EffectFlags["toryn.ship.used"] = 1; return true; }
        if (id.StartsWith("clone-wars.axe.", StringComparison.Ordinal))
        {
            int provider = game.Topology.EntityByRuntimeId[id["clone-wars.axe.".Length..]];
            Begin(a, "squad-defense-token", id, owner, [], 1, 1); a.EffectValue = provider; return true;
        }
        if (id.StartsWith("attack.dice.turbolaser-reroute-circuits.", StringComparison.Ordinal))
        { Begin(a, "trc-token", id, owner, [], 1, 1); return true; }
        if (id.StartsWith("attack.dice.defiance.", StringComparison.Ordinal))
        { if (!Card(game, a.Attacker, "defiance-title").TryUseThisAttack()) return false; Mark(a, id); return AddRolled(game, a, (AttackDieColor)SuffixInt(id), 1); }
        if (id.StartsWith("attack.dice.clone-captain-zak.", StringComparison.Ordinal))
        { if (!Card(game, a.Attacker, "clone-captain-zak-officer").Exhaust()) return false; Mark(a, id); return AddRolled(game, a, (AttackDieColor)SuffixInt(id), 1); }
        if (id.StartsWith("attack.dice.dominator.", StringComparison.Ordinal))
        { Begin(a, "dominator-shield", id, owner, [], SuffixInt(id), SuffixInt(id)); return true; }
        if (id.StartsWith("defense.upgrade.electronic-countermeasures.", StringComparison.Ordinal))
        {
            if (!ForcedDefenseTokenDiscardRules.TryParseChoiceId(id, "defense.upgrade.electronic-countermeasures.", out var token)) return false;
            if (!Card(game, a.Defender, "electronic-countermeasures").Exhaust()) return false;
            CombatModule.SpendToken(game, a, new(GameActionKind.SpendDefenseToken, a.Defender, a.Attacker, Token: token.Token, Index: token.Slot)); Mark(a, id); return true;
        }
        if (id.StartsWith("defense.upgrade.admonition.", StringComparison.Ordinal))
        {
            if (!ForcedDefenseTokenDiscardRules.TryParseChoiceId(id, "defense.upgrade.admonition.", out var token)
                || !ForcedDefenseTokenDiscardRules.TryDiscard(game.State.Ships[a.Defender], token) || !Card(game, a.Defender, "admonition-title").TryUseThisAttack()) return false;
            Begin(a, "remove", id, owner, all, 1, 1); return true;
        }
        if (id.StartsWith("defense.upgrade.reactive-gunnery.", StringComparison.Ordinal)
            || id.StartsWith("defense.upgrade.sa-nalaor.", StringComparison.Ordinal) || id == "defense.upgrade.expert-shield-tech")
        { Begin(a, "alternate-token", id, owner, [], 1, 1); return true; }
        throw new NotSupportedException("Source combat ability has no complete live handler: " + id);
    }

    private static IEnumerable<GameActionOption> ChoiceActions(GameContext game, AttackState a)
    {
        if (a.EffectRule.StartsWith("objective.assault.", StringComparison.Ordinal))
        { foreach (var option in AssaultCombatChoices(game, a)) yield return option; yield break; }
        if (a.EffectRule.StartsWith("official.", StringComparison.Ordinal))
        { foreach (var option in MissingSquadronChoices(game, a)) yield return option; yield break; }
        if (a.EffectRule.StartsWith("combat-extra.", StringComparison.Ordinal))
        { foreach (var option in ExtraCombatChoices(game, a)) yield return option; yield break; }
        if (a.EffectRule.StartsWith("defense-timing.", StringComparison.Ordinal))
        { foreach (var option in DefenseTimingChoices(game, a)) yield return option; yield break; }
        if (a.EffectRule == "after-attack-options")
        {
            foreach (var option in RaidAbilityEffects.GetCombatActions(game, a, CombatWindow.AfterAttack)) yield return option;
            yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, AbilityId: "combat.after-attack.finish"), "Finish after-attack effects");
            yield break;
        }
        if (a.EffectRule.StartsWith("additional.", StringComparison.Ordinal))
        { foreach (var option in AdditionalCombatChoices(game, a)) yield return option; yield break; }
        if (a.EffectRule.StartsWith("damage-prevention.", StringComparison.Ordinal))
        { foreach (var option in DamagePreventionChoices(game, a)) yield return option; yield break; }
        if (a.EffectRule == "intel-officer")
        {
            foreach (var target in IntelTargets(game, a))
                yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender,
                    AbilityId: IntelOfficerTargetRules.BuildChoiceId("attack.dice.intel-officer.", target)),
                    "Mark " + target.Token.Token + " token " + target.Token.Slot + (target.Token.Exhausted ? " (exhausted)" : " (ready)"));
            yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, AbilityId: "attack.dice.intel-officer.skip"), "Decline Intel Officer");
            yield break;
        }
        if (a.EffectRule == "squad-defense-token")
        {
            int entity = a.EffectValue;
            foreach (var type in Enum.GetValues<DefenseTokenType>())
                foreach (var token in game.State.Squadrons[game.SquadronIndex(entity)].GetDefenseTokenInstances(type))
                    yield return new(new(GameActionKind.ChooseToken, entity, Token: type, Index: token.Slot,
                        AbilityId: "combat.effect.squad-defense-token"), "Use " + type + " token " + token.Slot);
            yield break;
        }
        if (a.EffectRule == "dbs-color")
        {
            foreach (var color in a.Dice.Select(die => die.Color).Distinct())
                yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, Amount: (int)color,
                    AbilityId: "combat.effect.dbs-color"), "Suffer one damage and add " + color);
            yield break;
        }
        if (a.EffectRule == "jango-ally")
        {
            foreach (var ally in CloneWarsSquadronRules.JangoSacrifices(game.SquadronViews[game.SquadronIndex(a.Attacker)],
                game.SquadronViews[game.SquadronIndex(a.Defender)], game.SceneSquadronViews))
                yield return new(new(GameActionKind.ChooseEntity, a.Attacker, ally.Index, AbilityId: "combat.effect.jango-ally"), "Damage " + ally.Definition.DisplayName);
            yield break;
        }
        if (a.EffectRule == "thermal-offer")
        {
            yield return new(new(GameActionKind.UseAbility, a.Defender, a.Attacker, AbilityId: "combat.effect.thermal-use"), "Use Thermal Shields");
            yield return new(new(GameActionKind.UseAbility, a.Defender, a.Attacker, AbilityId: "combat.effect.thermal-decline"), "Decline Thermal Shields");
            yield break;
        }
        if (a.EffectRule == "quad-laser-zone")
        {
            foreach (var zone in QuadLaserZones(game, a))
                    yield return new(new(GameActionKind.ChooseZone, a.Defender, a.Attacker, FromZone: zone,
                        AbilityId: "quad-laser-turrets.zone." + (int)zone), "Counter from " + zone);
            yield return new(new(GameActionKind.UseAbility, a.Defender, a.Attacker, AbilityId: "quad-laser-turrets.zone.skip"), "Decline Counter");
            yield break;
        }
        if (a.EffectRule == "thermal-token")
        {
            foreach (var token in game.State.Ships[a.Defender].GetDefenseTokenInstances(DefenseTokenType.Brace)
                .Where(token => CombatModule.CanSpend(game, a, token)))
                yield return new(new(GameActionKind.ChooseToken, a.Defender, Token: token.Token, Index: token.Slot,
                    AbilityId: "combat.effect.choose-token"), "Spend Brace token " + token.Slot);
            yield break;
        }
        if (a.EffectRule is "trc-token" or "alternate-token" or "vader-token" or "anakin-token")
        {
            var type = a.EffectRule == "trc-token" ? DefenseTokenType.Evade
                : a.EffectActionId == "defense.upgrade.expert-shield-tech" ? DefenseTokenType.Redirect : (DefenseTokenType)SuffixInt(a.EffectActionId);
            int entity = a.EffectRule == "anakin-token" ? a.EffectValue : a.EffectRule is "trc-token" or "vader-token" ? a.Attacker : a.Defender;
            foreach (var token in game.State.Ships[entity].GetDefenseTokenInstances(type))
                if ((entity != a.Defender || !a.LockedTokens.Any(x => x.Token == token.Token && x.Slot == token.Slot))
                    && (!token.Exhausted || game.State.Ships[entity].CanSpendExhaustedDefenseTokens)
                    && (entity != a.Defender || CombatModule.CanSpend(game, a, token)))
                    yield return new(new(GameActionKind.ChooseToken, entity, Token: token.Token, Index: token.Slot,
                        AbilityId: "combat.effect.choose-token"), "Spend " + token.Token + " token " + token.Slot);
            yield break;
        }
        if (a.EffectRule == "flak-color")
        {
            foreach (var color in new[] { AttackDieColor.Red, AttackDieColor.Blue, AttackDieColor.Black })
                yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, Amount: (int)color, AbilityId: "combat.effect.flak-color"), "Add " + color + " die");
            yield break;
        }
        if (a.EffectRule == "dominator-shield")
        {
            foreach (var zone in game.State.Ships[a.Attacker].HullZones.Where(zone => game.State.Ships[a.Attacker].GetShields(zone) > 0))
                yield return new(new(GameActionKind.ChooseZone, a.Attacker, ToZone: zone, AbilityId: "combat.effect.choose-zone"), "Spend one " + zone + " shield");
            yield break;
        }
        if (a.EffectRule == "swivel-color")
        {
            foreach (var color in TurbolaserRules.GetSwivelMountFocusBonusColors(game.ShipViews[a.Attacker], a.FromZone))
                yield return new(new(GameActionKind.UseAbility, a.Attacker, Amount: (int)color, AbilityId: "combat.effect.choose-color"), "Add " + color + " die");
            yield break;
        }
        if (a.EffectRule.StartsWith("critical-", StringComparison.Ordinal))
        {
            foreach (var option in CriticalChoices(game, a)) yield return option;
            yield break;
        }
        if (a.EffectRule == "face" && a.EffectStage == 1)
        {
            foreach (var face in Faces(a.Dice[a.EffectValue].Color).Where(face =>
                a.EffectActionId == "attack.dice.governor-pryce" ? face != DieFace.Accuracy :
                a.EffectActionId == "attack.dice.kraken" ? face is DieFace.Hit or DieFace.Critical or DieFace.Accuracy :
                a.EffectActionId == "attack.dice.warlord" ? face is DieFace.Hit or DieFace.DoubleHit : true))
                yield return new(new(GameActionKind.ChooseDie, a.Attacker, Index: a.EffectValue, Amount: (int)face,
                    AbilityId: "combat.effect.choose-face"), "Set " + face);
            yield break;
        }
        foreach (int index in a.EffectEligibleIndices.Where(index => !a.EffectIndices.Contains(index) && MissingSquadronCombatEffects.IsDieModifiable(a, index)))
            if (a.EffectIndices.Length < a.EffectMaximum)
                yield return new(new(GameActionKind.ChooseDie, a.Attacker, a.Defender, Index: index,
                    AbilityId: "combat.effect.choose-die"), "Select die " + index + ": " + a.Dice[index].Color + " " + a.Dice[index].Face);
        if (a.EffectIndices.Length >= a.EffectMinimum && (a.EffectMaximum > 1 || a.EffectRule == "reroll"))
            yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, AbilityId: "combat.effect.finish-dice"), "Finish die selection");
    }

    private bool ApplyChoice(GameContext game, AttackState a, GameAction action)
    {
        string rule = a.EffectRule, id = a.EffectActionId;
        if (rule.StartsWith("objective.assault.", StringComparison.Ordinal)) return ApplyAssaultCombatChoice(game, a, action);
        if (rule.StartsWith("official.", StringComparison.Ordinal)) return ApplyMissingSquadronChoice(game, a, action);
        if (rule.StartsWith("combat-extra.", StringComparison.Ordinal)) return ApplyExtraCombatChoice(game, a, action);
        if (rule.StartsWith("defense-timing.", StringComparison.Ordinal)) return ApplyDefenseTimingChoice(game, a, action);
        if (rule == "after-attack-options")
        {
            if (action.AbilityId == "combat.after-attack.finish") { ClearChoice(a); return true; }
            return RaidAbilityEffects.TryCombatApply(game, a, CombatWindow.AfterAttack, action);
        }
        if (rule.StartsWith("additional.", StringComparison.Ordinal)) return ApplyAdditionalCombatChoice(game, a, action);
        if (rule.StartsWith("damage-prevention.", StringComparison.Ordinal)) return ApplyDamagePrevention(game, a, action);
        if (rule == "intel-officer")
        {
            if (action.AbilityId != "attack.dice.intel-officer.skip")
            {
                if (!IntelOfficerTargetRules.TryParseChoiceId(action.AbilityId, "attack.dice.intel-officer.", out var target)
                    || !IntelTargets(game, a).Contains(target) || !Card(game, a.Attacker, "intel-officer").Exhaust()) return false;
                a.IntelOfficerTarget = target;
            }
            ClearChoice(a); BeginPostRollChoices(game, a); return true;
        }
        if (rule == "dbs-color")
        {
            game.State.Squadrons[game.SquadronIndex(a.Attacker)].ApplyDamage(1);
            Mark(a, id); ClearChoice(a); return AddRolled(game, a, (AttackDieColor)action.Amount, 1);
        }
        if (rule == "jango-ally")
        {
            game.State.Squadrons[game.SquadronIndex(action.Target)].ApplyDamage(1);
            Begin(a, "face", id, a.EffectOwner, Enumerable.Range(0, a.Dice.Length), 1, 1); return true;
        }
        if (rule == "squad-defense-token")
        {
            var state = game.State.Squadrons[game.SquadronIndex(a.EffectValue)];
            var token = state.GetDefenseTokenInstances(action.Token).Single(token => token.Slot == action.Index);
            if (id == "clone-wars.kit-fisto.discard-token")
            {
                if (!state.DiscardDefenseToken(token)) return false;
                a.SpentTokenTypes = [..a.SpentTokenTypes, token.Token];
                a.EffectFlags["prebrace.damage-reduction"] = a.EffectFlags.GetValueOrDefault("prebrace.damage-reduction") + 3;
            }
            else
            {
                if (!state.TryResolveDefenseToken(token, out _)) return false;
                a.EffectFlags["prebrace.damage-reduction"] = a.EffectFlags.GetValueOrDefault("prebrace.damage-reduction") + 1;
            }
            Mark(a, id); ClearChoice(a); return true;
        }
        if (rule == "flak-color")
        {
            AppendDice(a, (AttackDieColor)action.Amount, 1); a.EffectStage++;
            if (a.EffectStage >= a.EffectMaximum) ClearChoice(a);
            return true;
        }
        if (rule == "quad-laser-zone")
        {
            if (action.Kind == GameActionKind.ChooseZone) a.EffectFlags["response.qlt.zone"] = (int)action.FromZone;
            ClearChoice(a); return true;
        }
        if (rule == "thermal-offer")
        {
            if (action.AbilityId == "combat.effect.thermal-decline") ClearChoice(a);
            else Begin(a, "thermal-token", id, a.EffectOwner, [], 1, 1);
            return true;
        }
        if (rule == "thermal-token")
        {
            CombatModule.SpendToken(game, a, new(GameActionKind.SpendDefenseToken, a.Defender, a.Attacker, Token: action.Token, Index: action.Index));
            a.Brace = false;
            var card = Card(game, a.Defender, "thermal-shields"); card.Exhaust(); card.TryUseThisAttack();
            int remove = DefenseUpgradeRules.ThermalShieldsRemovalCount(a.Dice.Length, a.Dice.Length);
            Begin(a, "remove", "thermal-shields.pre-roll.remove", game.State.Owners[a.Defender], Enumerable.Range(0, a.Dice.Length), remove, remove);
            return true;
        }
        if (rule.StartsWith("critical-", StringComparison.Ordinal)) return ApplyCriticalChoice(game, a, action);
        if (rule == "swivel-color")
        { ClearChoice(a); a.EffectFlags["swivel.focus.awaiting-chance"] = 1; return AddRolled(game, a, (AttackDieColor)action.Amount, 1); }
        if (rule == "dominator-shield")
        {
            game.State.Ships[a.Attacker].LoseShields(action.ToZone, 1); a.EffectStage++;
            if (a.EffectStage == a.EffectMaximum)
            { int count = a.EffectMaximum; Card(game, a.Attacker, "dominator-title").TryUseThisAttack(); Mark(a, id); ClearChoice(a); return AddRolled(game, a, AttackDieColor.Blue, count); }
            return true;
        }
        if (rule is "trc-token" or "alternate-token" or "vader-token" or "anakin-token")
        {
            int entity = rule == "anakin-token" ? a.EffectValue : rule is "trc-token" or "vader-token" ? a.Attacker : a.Defender;
            var token = game.State.Ships[entity].GetDefenseTokenInstances(action.Token).Single(x => x.Slot == action.Index);
            if (rule == "anakin-token")
            {
                if (!game.State.Ships[entity].TryResolveDefenseToken(token, out _)) return false;
                a.EffectFlags["anakin.used"] = 1; Begin(a, "reroll", id, a.EffectOwner, Enumerable.Range(0, a.Dice.Length), 0, 3); return true;
            }
            if (rule == "vader-token")
            {
                if (!game.State.Ships[entity].TryResolveDefenseToken(token, out _)) return false;
                Begin(a, "reroll", id, a.EffectOwner, Enumerable.Range(0, a.Dice.Length), 0, a.Dice.Length); return true;
            }
            if (rule == "trc-token")
            {
                if (!game.State.Ships[entity].TryResolveDefenseToken(token, out _) || !Card(game, entity, "turbolaser-reroute-circuits").Exhaust()) return false;
                Begin(a, "change", id, a.EffectOwner, Indices(a, die => die.Color == AttackDieColor.Red), 1, 1); return true;
            }
            CombatModule.SpendToken(game, a, new(GameActionKind.SpendDefenseToken, entity, a.Attacker, Token: token.Token, Index: token.Slot));
            if (id.StartsWith("defense.upgrade.reactive-gunnery.", StringComparison.Ordinal))
            { ResetTokenEffect(a, token.Token); a.Salvo = true; Card(game, entity, "reactive-gunnery").Exhaust(); Card(game, entity, "reactive-gunnery").TryUseThisAttack(); }
            else if (id == "defense.upgrade.expert-shield-tech")
            { a.Redirect = false; a.EffectFlags["damage-reduction"] = a.EffectFlags.GetValueOrDefault("damage-reduction") + 1; Card(game, entity, "expert-shield-tech-officer").TryUseThisAttack(); }
            else
            {
                ResetTokenEffect(a, token.Token); a.Evade = true;
                a.SpentEvade = token; a.EvadeWasDiscarded = token.Exhausted;
                a.EvadeBaseCount = game.HasUpgrade(entity, "foresight-title") ? 2 : 1;
                a.EvadeCancels = a.Range == RangeBand.Long; game.Flow.Phase = GamePhase.EvadeDice;
                Card(game, entity, "sa-nalaor-title").TryUseThisAttack();
            }
            Mark(a, id); ClearChoice(a); return true;
        }
        if (rule == "face" && a.EffectStage == 1)
        {
            if (id == "attack.dice.governor-pryce")
            { Card(game, a.Attacker, "governor-pryce-officer").TryUseThisActivation(); game.State.Ships[a.Attacker].LoseShields(a.FromZone, 1); }
            if (id == "attack.dice.kraken") game.Flow.Flags[ActivationKey(game, a.Attacker, "kraken")] = 1;
            a.Dice[a.EffectValue] = new(a.Dice[a.EffectValue].Color, (DieFace)action.Amount); Mark(a, id); ClearChoice(a); return true;
        }
        if (action.Kind == GameActionKind.ChooseDie)
        {
            a.EffectIndices = [..a.EffectIndices, action.Index];
            if (a.EffectMaximum > 1) return true;
        }
        int[] indices = a.EffectIndices.ToArray();
        if (rule == "spend-reroll" && a.EffectStage == 0)
        {
            MissingSquadronCombatEffects.RemoveDice(a, indices[0]);
            a.EffectStage = 1; a.EffectRule = "reroll"; a.EffectIndices = [];
            a.EffectEligibleIndices = MissingSquadronCombatEffects.ModifiableDice(a); a.EffectMaximum = a.Dice.Length; return true;
        }
        if (rule == "spend-change" && a.EffectStage == 0)
        {
            MissingSquadronCombatEffects.RemoveDice(a, indices[0]);
            a.EffectStage = 1; a.EffectRule = "change"; a.EffectIndices = [];
            a.EffectEligibleIndices = Indices(a, die => id != "attack.dice.sensor-team" || die.Color != AttackDieColor.Black).ToArray(); return true;
        }
        if (rule == "face") { a.EffectValue = indices[0]; a.EffectStage = 1; a.EffectIndices = []; return true; }
        if (rule == "reroll")
        {
            CommitCosts(game, a, id); Mark(a, id); ClearChoice(a); Reroll(game, a, indices); return true;
        }
        if (rule == "remove")
        {
            CommitCosts(game, a, id); MissingSquadronCombatEffects.RemoveDice(a, indices); Mark(a, id); ClearChoice(a);
            if (id == "mar-tuuk.cancel") BeginPostRollChoices(game, a); return true;
        }
        if (rule == "exchange")
        {
            var card = Card(game, a.Attacker, "gunnery-chief-varnillian");
            var previous = a.Dice[indices[0]]; a.Dice[indices[0]] = new(card.StoredDieColor, card.StoredDieFace);
            card.StoreDie(previous.Color, previous.Face); card.TryUseThisAttack(); Mark(a, id); ClearChoice(a); return true;
        }
        if (rule == "change")
        {
            int index = indices[0]; var die = a.Dice[index];
            DieFace face = id switch
            {
                "attack.dice.warlord" => die.Color == AttackDieColor.Red ? DieFace.DoubleHit : DieFace.Hit,
                "attack.dice.h9-turbolasers" or "attack.dice.home-one" or "attack.dice.sensor-team" or "clone-wars.baktoid.set-accuracy" => DieFace.Accuracy,
                "attack.dice.turbolaser-reroute-circuits.double-hit" => DieFace.DoubleHit,
                "fleet-command.attack.intensify-firepower" => DieFace.Hit,
                _ => die.Color == AttackDieColor.Black ? DieFace.HitCritical : DieFace.Critical
            };
            CommitCosts(game, a, id); a.Dice[index] = new(die.Color, face); Mark(a, id); ClearChoice(a); return true;
        }
        throw new NotSupportedException("Pending combat effect requires a handler: " + rule + " / " + id);
    }

    private static void CommitCosts(GameContext game, AttackState a, string id)
    {
        string? exhaust = id switch
        {
            "attack.dice.sensor-team" => "sensor-team", "defense.upgrade.captain-brunson" => "captain-brunson-officer", _ => null
        };
        if (exhaust is not null && !Card(game, id.StartsWith("defense.", StringComparison.Ordinal) ? a.Defender : a.Attacker, exhaust).Exhaust())
            throw new InvalidOperationException("Combat effect exhaustion cost is unavailable.");
        if (id.StartsWith("attack.dice.caitken-and-shollan.", StringComparison.Ordinal))
        { Card(game, a.Attacker, "caitken-and-shollan").Exhaust(); a.EffectFlags["caitken"] = 1; }
        if (id == "defense.upgrade.targeting-scrambler")
        {
            var source = DefenseUpgradeRules.TargetingScramblerSource(game.ShipViews[a.Defender])
                ?? throw new InvalidOperationException("Targeting Scrambler source is unavailable.");
            Card(game, source.Index, "targeting-scrambler").Exhaust();
        }
        if (id == "defense.upgrade.lando-calrissian") Card(game, a.Defender, "lando-calrissian-officer").Discard();
        if (id == "attack.dice.admiral-screed") game.Flow.Flags[ActivationKey(game, a.Attacker, "screed")] = 1;
        foreach (string card in new[] { "ordnance-experts", "h9-turbolasers", "weapons-battery-techs", "linked-turbolaser-towers", "point-defense-ion-cannons" })
            if (id.Contains(card, StringComparison.Ordinal)) Card(game, card == "point-defense-ion-cannons" ? a.Defender : a.Attacker, card).TryUseThisAttack();
    }

    private static void Begin(AttackState a, string rule, string id, int owner, IEnumerable<int> eligible, int minimum, int maximum)
    {
        a.EffectRule = rule; a.EffectActionId = id; a.EffectOwner = owner; a.EffectStage = 0;
        a.EffectIndices = []; a.EffectEligibleIndices = (rule is "reroll" or "remove" or "face" or "change" or "exchange" or "spend-reroll" or "spend-change"
            ? eligible.Where(index => MissingSquadronCombatEffects.IsDieModifiable(a, index)) : eligible).ToArray();
        a.EffectMinimum = minimum; a.EffectMaximum = Math.Min(maximum, a.EffectEligibleIndices.Length > 0 ? a.EffectEligibleIndices.Length : maximum);
    }
    private static void ClearChoice(AttackState a)
    { a.EffectRule = ""; a.EffectActionId = ""; a.EffectIndices = []; a.EffectEligibleIndices = []; a.EffectStage = 0; }
    private static void Mark(AttackState a, string id) => a.EffectFlags["used." + id] = 1;
    private static bool Used(AttackState a, string id) => a.EffectFlags.ContainsKey("used." + id);
    private static UpgradeCardRuntimeState Card(GameContext game, int entity, string id) => game.UpgradesFor(entity)
        .FirstOrDefault(card => card.Active && card.Definition.id == id) ?? throw new InvalidOperationException("Unavailable combat card: " + id);
    private static IEnumerable<int> Indices(AttackState a, Func<AttackDie, bool> predicate) => Enumerable.Range(0, a.Dice.Length).Where(index => MissingSquadronCombatEffects.IsDieModifiable(a, index) && predicate(a.Dice[index]));
    private static void AppendDice(AttackState a, AttackDieColor color, int count)
    { a.Dice = [..a.Dice, ..Enumerable.Repeat(new AttackDie(color, DieFace.Blank), count)]; }
    private static bool AddRolled(GameContext game, AttackState a, AttackDieColor color, int count)
    {
        int start = a.Dice.Length; AppendDice(a, color, count);
        Reroll(game, a, Enumerable.Range(start, count).ToArray()); return true;
    }
    private static void Reroll(GameContext game, AttackState a, int[] indices)
    { a.PendingRollIndices = indices.Where(index => MissingSquadronCombatEffects.IsDieModifiable(a, index)).ToArray(); a.AfterRollPhase = game.Flow.Phase; game.Flow.Phase = GamePhase.RollAttack; }
    private static int SuffixInt(string value) => int.Parse(value[(value.LastIndexOf('.') + 1)..], System.Globalization.CultureInfo.InvariantCulture);
    private static AbilityAction Ability(string id, string source, string label) => new(id, source, source,
        AbilitySourceKind.Upgrade, label, "", "", AbilityTiming.ModifyAttackDice);
    private static string ActivationKey(GameContext game, int entity, string rule) => $"combat.activation.{game.State.Round}.{entity}.{rule}";
    internal static string AckbarKey(GameContext game, int entity) => ActivationKey(game, entity, "ackbar");
    private static IEnumerable<DieFace> Faces(AttackDieColor color) => color switch
    {
        AttackDieColor.Red => [DieFace.Blank, DieFace.Hit, DieFace.Critical, DieFace.Accuracy, DieFace.DoubleHit],
        AttackDieColor.Blue => [DieFace.Hit, DieFace.Critical, DieFace.Accuracy],
        _ => [DieFace.Blank, DieFace.Hit, DieFace.HitCritical]
    };
    private static void ResetTokenEffect(AttackState a, DefenseTokenType type)
    {
        if (type == DefenseTokenType.Brace) a.Brace = false;
        if (type == DefenseTokenType.Redirect) a.Redirect = false;
        if (type == DefenseTokenType.Evade) a.Evade = false;
        if (type == DefenseTokenType.Contain) a.Contain = false;
        if (type == DefenseTokenType.Scatter) a.Scatter = false;
        if (type == DefenseTokenType.Salvo) a.Salvo = false;
    }
    public static AttackResolution Resolution(GameContext game, AttackState a)
    {
        int hits = 0, crits = 0, accuracies = 0, blanks = 0;
        foreach (var die in a.Dice)
            switch (die.Face)
            { case DieFace.Hit: hits++; break; case DieFace.DoubleHit: hits += 2; break; case DieFace.Critical: crits++; break;
                case DieFace.HitCritical: hits++; crits++; break; case DieFace.Accuracy: accuracies++; break; default: blanks++; break; }
        return new(a.Range, a.Dice.Length, hits, crits, accuracies, blanks, true,
            game.IsShip(a.Attacker) ? UnitAbilityEngine.GetCriticalDamagePerIcon(game.ShipViews[a.Attacker], a.FromZone) : 1, a.Dice);
    }

    // Critical handlers follow the source controller's explicit card set and icon requirements.
    private static IEnumerable<UpgradeCardRuntimeState> EligibleCriticals(GameContext game, AttackState a)
    {
        if (a.Scatter || !game.IsShip(a.Attacker) || !game.IsShip(a.Defender) || game.Topology.Ships[a.Defender].IsStation || !game.State.Ships[a.Attacker].CanResolveCriticalEffects
            || OfficerCombatRules.PreventsCriticalEffects(game.ShipViews[a.Defender])
            || a.Contain && OfficerCombatRules.ContainPreventsAllCriticalEffects(game.ShipViews[a.Defender])) yield break;
        var attack = Resolution(game, a); if (!attack.CanResolveCriticalEffect) yield break;
        int limit = game.UpgradesFor(a.Attacker).Any(card => card.Active && card.Definition.id == "fire-control-team" && !card.Exhausted) ? 2 : 1;
        if (a.EffectFlags.GetValueOrDefault("critical.count") + (a.FirstHullCardFaceup ? 1 : 0) >= limit) yield break;
        foreach (var card in game.UpgradesFor(a.Attacker))
        {
            string id = card.Definition.id;
            if (!card.Active || !CriticalCards.Contains(id) || Used(a, "upgrade-critical." + id)) continue;
            if (!UpgradeCriticalRules.HasRequiredIgnitionCritical(id, attack, a.IgnitionAttack)) continue;
            if (UpgradeCriticalRules.NeedsBlueCritical(id) && !AttackResolver.HasCriticalIcon(attack, AttackDieColor.Blue)) continue;
            if (UpgradeCriticalRules.NeedsBlackCritical(id) && !AttackResolver.HasCriticalIcon(attack, AttackDieColor.Black)) continue;
            if (UpgradeCriticalRules.NeedsBlueOrBlackCritical(id) && !AttackResolver.HasCriticalIcon(attack, AttackDieColor.Blue) && !AttackResolver.HasCriticalIcon(attack, AttackDieColor.Black)) continue;
            if (UpgradeCriticalRules.NeedsRedOrBlueCritical(id) && !AttackResolver.HasCriticalIcon(attack, AttackDieColor.Red) && !AttackResolver.HasCriticalIcon(attack, AttackDieColor.Blue)) continue;
            if (id is "heavy-ion-emplacements" or "assault-concussion-missiles" or "assault-proton-torpedoes" or "nk-7-ion-cannons" or "kyrsta-agate-officer" && card.Exhausted) continue;
            if (id == "ms-1-ion-cannons" && !game.UpgradesFor(a.Defender).Any(target => target.Active && !target.Exhausted)) continue;
            if (id == "sunder-title" && !game.UpgradesFor(a.Defender).Any(target => target.Active && UpgradeCriticalRules.SunderCanDiscardSlot(target.Definition.slot))) continue;
            if (id == "nk-7-ion-cannons" && !Enum.GetValues<DefenseTokenType>().Any(game.State.Ships[a.Defender].HasDefenseToken)) continue;
            if (id == "kyrsta-agate-officer" && !Enum.GetValues<DefenseTokenType>().Any(type => game.State.Ships[a.Attacker].GetAvailableDefenseTokenCount(type) > game.State.Ships[a.Attacker].GetReadyDefenseTokenCount(type))) continue;
            yield return card;
        }
    }

    private static bool BeginCritical(GameContext game, AttackState a, string id)
    {
        if (!EligibleCriticals(game, a).Any(card => card.Definition.id == id)) return false;
        var card = Card(game, a.Attacker, id); var target = game.State.Ships[a.Defender];
        a.EffectFlags["critical.count"] = a.EffectFlags.GetValueOrDefault("critical.count") + 1;
        Mark(a, "upgrade-critical." + id);
        switch (id)
        {
            case "heavy-ion-emplacements":
                card.Exhaust(); target.LoseShields(a.ToZone, 1);
                foreach (var zone in HullZoneRules.Adjacent(a.ToZone, target.HasAuxiliaryHullZones)) target.LoseShields(zone, 1);
                return true;
            case "overload-pulse":
                foreach (var type in Enum.GetValues<DefenseTokenType>())
                    foreach (var token in target.GetDefenseTokenInstances(type).Where(token => !token.Exhausted)) target.SpendDefenseToken(token);
                return true;
            case "xx-9-turbolasers": a.EffectFlags["faceup.limit"] = 2; return true;
            case "assault-proton-torpedoes":
                card.Exhaust(); CombatModule.QueueDamage(game, a.Defender, 1, true, true, GamePhase.ResolveCritical, rule: "upgrade-critical." + id, source: a.Attacker); return true;
            case "dodonnas-pride-title":
                a.Damage = 0; CombatModule.QueueDamage(game, a.Defender, 1, true, true, GamePhase.ResolveCritical, rule: "upgrade-critical." + id, source: a.Attacker); return true;
            case "ion-cannon-batteries":
            case "ms-1-ion-cannons":
            case "nk-7-ion-cannons":
            case "sunder-title":
            case "kyrsta-agate-officer":
                Begin(a, "critical-" + id, "upgrade-critical." + id,
                    id == "nk-7-ion-cannons" ? game.State.Owners[a.Defender] : game.State.Owners[a.Attacker], [], 1, 1); return true;
            case "assault-concussion-missiles":
                card.Exhaust();
                foreach (var zone in HullZoneRules.Adjacent(a.ToZone, target.HasAuxiliaryHullZones).Reverse())
                    CombatModule.QueueDamage(game, a.Defender, 1, false, false, GamePhase.ResolveCritical, zone, "upgrade-critical." + id);
                return true;
            case "wide-area-barrage":
                Begin(a, "critical-wide-area-barrage", "upgrade-critical." + id, game.State.Owners[a.Attacker], [], 1, 1); return true;
            case "superheavy-composite-beam-turbolasers":
                Begin(a, "critical-" + id, "upgrade-critical." + id, game.State.Owners[a.Attacker], [], 1,
                    UpgradeCriticalRules.CountCriticalIcons(Resolution(game, a), AttackDieColor.Red, AttackDieColor.Blue)); return true;
            case "orbital-bombardment-particle-cannons":
                Begin(a, "critical-" + id, "upgrade-critical." + id, game.State.Owners[a.Attacker],
                    OrbitalShips(game, a), 1, int.MaxValue);
                if (a.EffectEligibleIndices.Length == 0) { ClearChoice(a); QueueOrbitalSplash(game, a, [], []); }
                return true;
            default: throw new NotSupportedException("Critical source effect needs implementation: " + id);
        }
    }
    private static IEnumerable<GameActionOption> CriticalChoices(GameContext game, AttackState a)
    {
        string id = a.EffectRule["critical-".Length..];
        switch (id)
        {
            case "superheavy-composite-beam-turbolasers":
            case "orbital-bombardment-particle-cannons":
                int defender = id == "superheavy-composite-beam-turbolasers" ? a.Defender : a.EffectEligibleIndices[a.EffectStage];
                foreach (var zone in CombatModule.TargetZones(game, defender))
                    yield return new(new(GameActionKind.ChooseZone, a.Attacker, defender, ToZone: zone, AbilityId: "combat.critical.ignition-zone"),
                        "Damage entity " + defender + " " + zone);
                break;
            case "ion-cannon-batteries":
                foreach (var command in Enum.GetValues<ShipCommand>().Where(command => game.State.Commands[a.Defender].HasToken(command)))
                    yield return new(new(GameActionKind.ChooseCommand, a.Attacker, a.Defender, Command: command, AbilityId: "combat.critical.command"), "Discard " + command + " token");
                if (game.State.Commands[a.Defender].TokenCount == 0)
                    yield return new(new(GameActionKind.UseAbility, a.Attacker, a.Defender, AbilityId: "combat.critical.shield"), "Remove defending shield");
                break;
            case "ms-1-ion-cannons":
            case "sunder-title":
                foreach (var upgrade in game.Topology.Upgrades)
                {
                    if (game.State.Registers[game.Topology.Registers.UpgradeOwner(upgrade.UpgradeIndex)] != a.Defender + 1) continue;
                    var card = game.State.Upgrades[upgrade.UpgradeIndex];
                    if (card.Active && (id == "sunder-title" ? UpgradeCriticalRules.SunderCanDiscardSlot(card.Definition.slot) : !card.Exhausted))
                        yield return new(new(GameActionKind.ChooseEntity, a.Attacker, a.Defender, Index: upgrade.UpgradeIndex,
                            AbilityId: "combat.critical.upgrade"), (id == "sunder-title" ? "Discard " : "Exhaust ") + card.Definition.name);
                }
                break;
            case "nk-7-ion-cannons":
                foreach (var type in Enum.GetValues<DefenseTokenType>())
                    foreach (var token in game.State.Ships[a.Defender].GetDefenseTokenInstances(type))
                        yield return new(new(GameActionKind.ChooseToken, a.Attacker, a.Defender, Token: token.Token, Index: token.Slot,
                            AbilityId: "combat.critical.token"), "Discard " + token.Token + " token " + token.Slot);
                break;
            case "kyrsta-agate-officer":
                foreach (var type in Enum.GetValues<DefenseTokenType>().Where(type => game.State.Ships[a.Attacker].GetAvailableDefenseTokenCount(type) > game.State.Ships[a.Attacker].GetReadyDefenseTokenCount(type)))
                    yield return new(new(GameActionKind.ChooseToken, a.Attacker, Token: type, AbilityId: "combat.critical.ready"), "Ready " + type);
                break;
            case "wide-area-barrage":
                for (int entity = 0; entity < game.Topology.EntityCount; entity++)
                {
                    if (entity == a.Defender || !game.IsOnBoard(entity)) continue;
                    float distance = EntityGeometry.Distance(game, entity, a.Defender);
                    if (distance > TabletopMeasurements.CloseRange) continue;
                    foreach (var zone in CombatModule.TargetZones(game, entity))
                        yield return new(new(GameActionKind.ChooseEntity, a.Attacker, entity, ToZone: zone, AbilityId: "combat.critical.barrage"), "Barrage entity " + entity + " " + zone);
                }
                break;
        }
    }
    private static bool ApplyCriticalChoice(GameContext game, AttackState a, GameAction action)
    {
        string id = a.EffectRule["critical-".Length..];
        switch (id)
        {
            case "superheavy-composite-beam-turbolasers":
                a.EffectIndices = [..a.EffectIndices, (int)action.ToZone]; a.EffectStage++;
                if (a.EffectStage < a.EffectMaximum) return true;
                var beamZones = a.EffectIndices.ToArray(); ClearChoice(a);
                foreach (int zone in beamZones.Reverse()) CombatModule.QueueDamage(game, a.Defender, 1, false, false,
                    GamePhase.ResolveCritical, (HullZone)zone, "upgrade-critical." + id, source: a.Attacker);
                return true;
            case "orbital-bombardment-particle-cannons":
                a.EffectIndices = [..a.EffectIndices, (int)action.ToZone]; a.EffectStage++;
                if (a.EffectStage < a.EffectEligibleIndices.Length) return true;
                var splashShips = a.EffectEligibleIndices.ToArray(); var splashZones = a.EffectIndices.ToArray(); ClearChoice(a);
                QueueOrbitalSplash(game, a, splashShips, splashZones); return true;
            case "ion-cannon-batteries":
                if (action.Kind == GameActionKind.ChooseCommand) game.State.Commands[a.Defender].TrySpendToken(action.Command);
                else game.State.Ships[a.Defender].LoseShields(a.ToZone, 1);
                break;
            case "ms-1-ion-cannons": game.State.Upgrades[action.Index].Exhaust(); break;
            case "sunder-title":
                Card(game, a.Attacker, id).Discard();
                UnitAbilityEngine.DiscardUpgradeFromShip(game.ShipViews[a.Defender], game.State.Upgrades[action.Index], out _); break;
            case "nk-7-ion-cannons":
                Card(game, a.Attacker, id).Exhaust();
                game.State.Ships[a.Defender].DiscardDefenseToken(game.State.Ships[a.Defender].GetDefenseTokenInstances(action.Token).Single(token => token.Slot == action.Index)); break;
            case "kyrsta-agate-officer": Card(game, a.Attacker, id).Exhaust(); game.State.Ships[a.Attacker].TryReadyDefenseToken(action.Token); break;
            case "wide-area-barrage":
                int damage = (a.Dice.Count(die => die.Color == AttackDieColor.Black && die.Face is DieFace.Hit or DieFace.HitCritical) + 1) / 2;
                ClearChoice(a); CombatModule.QueueDamage(game, action.Target, damage, false, false, GamePhase.ResolveCritical, action.ToZone, "upgrade-critical.wide-area-barrage"); return true;
        }
        ClearChoice(a); return true;
    }

    private static IEnumerable<int> OrbitalShips(GameContext game, AttackState a) => game.SceneShipViews
        .Where(ship => ship.Index != a.Defender && RaidRules.FootprintDistance(ship, game.ShipViews[a.Defender]) <= 3.369803f)
        .OrderBy(ship => ship.RuntimeId.Value, StringComparer.Ordinal).Select(ship => ship.Index);
    private static void QueueOrbitalSplash(GameContext game, AttackState a, int[] ships, int[] zones)
    {
        // Push in reverse so resolution follows the source's stable ship-then-squadron ordering.
        foreach (var squad in game.SceneSquadronViews.Where(squad => SquadronAbilityEngine.FootprintDistance(squad, game.ShipViews[a.Defender]) <= 3.369803f)
            .OrderByDescending(squad => squad.RuntimeId.Value, StringComparer.Ordinal))
            CombatModule.QueueDamage(game, squad.Index, 1, false, false, GamePhase.ResolveCritical,
                rule: "upgrade-critical.orbital-bombardment-particle-cannons", source: a.Attacker);
        for (int index = ships.Length - 1; index >= 0; index--)
            CombatModule.QueueDamage(game, ships[index], 2, false, false, GamePhase.ResolveCritical, (HullZone)zones[index],
                "upgrade-critical.orbital-bombardment-particle-cannons", source: a.Attacker);
    }
}
