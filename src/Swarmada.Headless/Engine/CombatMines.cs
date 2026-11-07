using System.Numerics;
using Armada.Core;
using Armada.Data;
using Armada.Rules;
using Swarmada.Headless.Effects;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

public sealed partial class CombatModule
{
    public static bool TryMineAttackGeometry(GameContext game, int attacker, int index, HullZone from,
        out RangeBand range, out bool obstructed, out DiceBattery battery)
    {
        range = RangeBand.OutOfRange; obstructed = false; battery = default;
        if (!game.IsShip(attacker) || index < 0 || index >= game.State.SpatialObjects.Count) return false;
        var mine = game.State.SpatialObjects[index];
        if (!mine.Active || mine.Kind != SetupModule.ProximityMineKind) return false;
        if (CanReuseZone(game, attacker, from) && game.Flow.Flags.ContainsKey(MineTargetKey(game, attacker, index, from))) return false;
        using var scope = new GameScope(game);
        var target = new Circle(mine.Pose.Position, TabletopMeasurements.FromWorldUnits(.26f));
        if (Geometry2D.Distance(target, game.ShipBase(attacker)) <= 0) return false;
        var zone = Zone(game, attacker, from);
        if (!zone.ContainsInArc(target.Center) && !ArcIntersectsCircle(zone, target)) return false;
        range = HullGeometry.ClassifyRange(HullGeometry.MeasureToSquadron(zone, target).Distance);
        if (!game.State.Ships[attacker].CanAttackAtRange(range)) return false;
        var printed = TurbolaserRules.EffectiveFlakArmament(game.ShipViews[attacker], null,
            OffensiveRetrofitRules.EffectiveAntiSquadronArmament(game.ShipViews[attacker], game.ShipViews[attacker].Definition.AntiSquadronArmament), true);
        if (game.Flow.Flags.TryGetValue("combat.granted.black", out long black)) printed = new(0, 0, (int)black);
        battery = new(range != RangeBand.OutOfRange ? printed.RedDice : 0,
            range is RangeBand.Close or RangeBand.Medium ? printed.BlueDice : 0, range == RangeBand.Close ? printed.BlackDice : 0);
        if (battery.Total == 0 || game.Flow.Flags.TryGetValue("combat.granted.maximum-range", out long maximum) && (int)range > maximum) return false;
        var blockers = new List<LineOfSightObstacle>();
        foreach (var ship in game.SceneShipViews.Where(ship => ship.Index != attacker && !Station(game, ship.Index))) blockers.Add(new(game.ShipBase(ship.Index)));
        foreach (int obstacle in OfficialObjectives.Obstacles(game))
        {
            if (OfficialObjectives.Is(game, "contested-outpost") && OfficialObjectives.ObstacleId(game, obstacle) == "station") continue;
            blockers.Add(new(ObstacleGeometry.Shape(game, obstacle)));
        }
        obstructed = LineOfSightGeometry.Evaluate(new(zone.TargetingPoint, target.Center), blockers.ToArray()) == LineOfSightResult.Obstructed;
        return true;
    }
    private void BeginMineAttack(GameContext game, GameAction action)
    {
        ValidateUpgrades(game, action.Source, -1);
        if (!TryMineAttackGeometry(game, action.Source, action.Index, action.FromZone, out var range, out bool obstructed, out var battery))
            throw new ArgumentException("Mine is no longer a legal target.", nameof(action));
        var attack = new AttackState { Attacker = action.Source, Defender = -1, MineSpatialIndex = action.Index,
            FromZone = action.FromZone, Range = range, DiceRange = range, Obstructed = obstructed,
            RedDice = battery.Red, BlueDice = battery.Blue, BlackDice = battery.Black, Dice = Unrolled(battery),
            MandatoryRemovals = (obstructed ? 1 : 0) + (game.State.Ships[action.Source].MustRemoveDieWhenAttackingSquadron ? 1 : 0),
            ResumePhase = game.Flow.Flags.ContainsKey($"combat.demolisher.after-maneuver.{game.State.Round}.{action.Source}") ? GamePhase.AbilityChoice : GamePhase.SelectAttack };
        if (game.Flow.Flags.ContainsKey("combat.granted.active"))
        {
            attack.EffectFlags["granted-attack"] = 1; attack.ResumePhase = (GamePhase)game.Flow.Flags["combat.granted.return"];
            if (game.Flow.Flags.TryGetValue("combat.granted.upgrade-index", out long upgradeIndex))
            {
                var card = game.State.Upgrades[(int)upgradeIndex];
                if (!(game.Flow.Flags.ContainsKey("combat.granted.exhaust") ? card.Exhaust() : card.TryUseThisActivation()))
                    throw new InvalidOperationException("Mine attack's granting upgrade is unavailable.");
                game.Flow.Flags["combat.granted.paid"] = 1;
            }
        }
        foreach (var card in game.UpgradesFor(action.Source)) card.ResetAttackUsage();
        game.Flow.Flags[MineTargetKey(game, action.Source, action.Index, action.FromZone)] = 1;
        game.Flow.Attacks.Add(attack); Enter(game, attack, GamePhase.ModifyAttack, CombatWindow.BeforeRoll);
        if (attack.MandatoryRemovals >= attack.Dice.Length) CompleteMineAttack(game, attack);
    }
    private static string MineTargetKey(GameContext game, int ship, int index, HullZone zone)
        => $"combat.mine.{game.State.Round}.{ship}.{(int)zone}.{index}";
    private static void CompleteMineAttack(GameContext game, AttackState attack)
    {
        var mine = game.State.SpatialObjects[attack.MineSpatialIndex];
        if (attack.Rolled && mine.Active && attack.Dice.Any(die => die.Face is DieFace.Hit or DieFace.DoubleHit or DieFace.HitCritical))
        {
            game.State.SpatialObjects[attack.MineSpatialIndex] = mine with { Active = false };
            TokenObstacleAttachments.ForgetToken(game, attack.MineSpatialIndex);
        }
        game.Flow.Attacks.RemoveAt(game.Flow.Attacks.Count - 1);
        if (attack.EffectFlags.ContainsKey("granted-attack"))
        { game.Flow.Flags["combat.granted.completed"] = 1; ClearGrantedFlags(game); }
        else
        { var activation = game.Flow.ShipActivations[attack.Attacker]; activation.AttacksResolved++; activation.UsedHullZones |= 1 << (int)attack.FromZone; }
        game.Flow.Phase = attack.ResumePhase;
    }
}

public sealed partial class SourceCombatEffects
{
    private static IEnumerable<AbilityAction> MineDiceActions(GameContext game, AttackState attack, CombatWindow window)
    {
        var ship = game.ShipViews[attack.Attacker];
        if (window != CombatWindow.AttackEffects) yield break;
        foreach (var action in UnitAbilityEngine.GetShipDiceActions(ship, Resolution(game, attack), Used(attack, "attack.dice.leading-shots"),
            attack.EffectFlags.ContainsKey("caitken"), Used(attack, "attack.dice.ordnance-experts"),
            game.Flow.Flags.ContainsKey(ActivationKey(game, attack.Attacker, "screed")), Used(attack, "attack.dice.warlord"), Used(attack, "attack.dice.home-one"), true)) yield return action;
        if (ship.Loadout.Find("linked-turbolaser-towers") is { Active: true, Exhausted: false } && attack.Dice.Length > 0)
            foreach (var color in new[] { AttackDieColor.Red, AttackDieColor.Blue, AttackDieColor.Black })
                yield return Ability("mine-attack.ltt." + (int)color, "linked-turbolaser-towers", "Exhaust LTT; add two " + color + " dice");
        if (attack.FromZone != HullZone.Front && ship.Loadout.Find("clone-captain-zak-officer") is { Active: true, Exhausted: false })
            foreach (var color in attack.Dice.Select(die => die.Color).Distinct())
                yield return Ability("attack.dice.clone-captain-zak." + (int)color, "clone-captain-zak-officer", "Exhaust Zak; add " + color);
        foreach (var other in game.SceneShipViews.Where(other => other.Index != ship.Index && other.OwnerId == ship.OwnerId && RaidRules.FootprintDistance(other, ship) <= 8.09628f))
        {
            if (ship.Loadout.HasActive("task-force-organa-title") && other.Loadout.Find("task-force-organa-title") is { Active: true, Exhausted: false })
                yield return Ability("attack.dice.task-force-organa." + other.RuntimeId.Value, "task-force-organa-title", "Exhaust nearby Organa; reroll two dice");
            if (attack.Dice.Any(die => die.Color == AttackDieColor.Blue) && other.Loadout.HasActive("toryn-farr-officer") && !attack.EffectFlags.ContainsKey("toryn.ship.used"))
                yield return Ability("remaining.attack-dice.toryn-farr." + other.RuntimeId.Value, "toryn-farr-officer", "Toryn reroll one blue die");
        }
        foreach (var action in KrennicActions(game, attack)) yield return action;
        if (ship.Loadout.HasActive("point-defense-reroute") && attack.Range == RangeBand.Close && attack.Dice.Any(die => die.Face is DieFace.Critical or DieFace.HitCritical))
            yield return Ability("attack.dice.point-defense-reroute", "point-defense-reroute", "Reroll critical flak dice");
        if (TitleRules.FlakRerolls(ship, null, null, out string source) > 0)
            yield return Ability("attack.dice.flak-title", source, "Reroll one anti-squadron die");
    }
}
