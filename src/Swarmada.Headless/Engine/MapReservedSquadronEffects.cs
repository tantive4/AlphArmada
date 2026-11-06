using System.Numerics;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public static partial class MapAbilityEffects
{
    /// <summary>Controller-owned reserve release. Rapid Launch Bays discards before placing every
    /// stored squadron; Invisible Hand allows any chosen subset. Positions remain continuous.</summary>
    public static bool BeginReservedSquadronRelease(GameContext game, int ship, string cardId, GamePhase resumePhase)
    {
        if (cardId is not ("rapid-launch-bays" or "invisible-hand-title")) throw new ArgumentException("Unsupported squadron reserve card.", nameof(cardId));
        if (!AvailableShip(game, ship)) return false;
        int upgrade = FindReserveUpgrade(game, ship, cardId);
        if (upgrade < 0 || !ReservedSquadrons(game, ship, upgrade).Any()) return false;
        EngineEffects.PushChoice(game, new("map-choice.reserve-release", ship, upgrade, (int)resumePhase, Text: cardId), []); return true;
    }
    private static GameDecision? GetReservedReleaseDecision(GameContext game, GameContinuation frame)
    {
        if (frame.Rule != "map-choice.reserve-release") return null;
        bool rapid = frame.Text == "rapid-launch-bays"; var options = new List<GameActionOption>();
        if (frame.Value == 0)
        {
            if (rapid) options.Add(new(new(GameActionKind.UseAbility, frame.Source, AbilityId: "map.reserve-release.use"), "Discard Rapid Launch Bays and deploy every stored squadron"));
            else foreach (int squad in ReservedSquadrons(game, frame.Source, frame.Target))
                options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, squad), "Choose stored squadron to deploy"));
            options.Add(new(new(GameActionKind.Pass, frame.Source), "Keep remaining squadrons stored"));
        }
        else
        {
            var remaining = frame.Indices ?? [];
            if (remaining.Length == 0) return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), "Finish stored squadron deployment")]);
            foreach (int squad in remaining)
                options.Add(new(new(GameActionKind.PlaceReservedUnit, frame.Source, squad, AbilityId: "map.reserve-release.place"),
                    rapid ? "Choose a remaining stored squadron to place within distance two" : "Place stored squadron within distance three", AcceptsPosition: true));
            if (!rapid) options.Add(new(new(GameActionKind.Pass, frame.Source), "Choose another stored squadron"));
        }
        return new(DecisionKind.Player, game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
    }
    private static bool ApplyReservedRelease(GameContext game, GameContinuation frame, GameAction action)
    {
        if (frame.Rule != "map-choice.reserve-release") return false;
        bool rapid = frame.Text == "rapid-launch-bays";
        if (action.Kind == GameActionKind.Pass)
        {
            if (!rapid && frame.Value == 1) { game.Flow.Continuations[^1] = frame with { Value = 0, Indices = [] }; return true; }
            ReturnReservedRelease(game, frame); return true;
        }
        if (frame.Value == 0)
        {
            if (rapid)
            {
                int[] required = ReservedSquadrons(game, frame.Source, frame.Target).ToArray();
                if (required.Length == 0 || !game.State.Upgrades[frame.Target].Discard()) throw new ArgumentException("Rapid Launch Bays release unavailable.");
                game.Flow.Continuations[^1] = frame with { Value = 1, Indices = required };
            }
            else game.Flow.Continuations[^1] = frame with { Value = 1, Indices = [action.Target] };
            return true;
        }
        int entity = action.Target;
        if (!(frame.Indices ?? []).Contains(entity)) throw new ArgumentException("Choose a remaining stored squadron.");
        var position = action.Position ?? throw new ArgumentException("Stored squadron placement required.");
        if (!IsLegalReservedSquadronPlacement(game, frame.Source, entity, frame.Target, position,
            rapid ? TabletopMeasurements.DistanceBand(2) : TabletopMeasurements.DistanceBand(3))) throw new ArgumentException("Illegal stored squadron placement.");
        game.State.Poses[entity] = new(position, game.State.Poses[entity].YawDegrees);
        game.State.Registers[game.Topology.Registers.EntityReserved(entity)] = 0;
        game.State.Registers[game.Topology.Registers[$"entity.{entity}.placed"]] = 1;
        game.State.Squadrons[game.SquadronIndex(entity)].PreventMovementForShipActivation();
        game.State.Upgrades[frame.Target].RemoveStoredUnit(game.Topology.Squadrons[game.SquadronIndex(entity)].RuntimeId);
        if (!rapid) game.Flow.Flags[$"movement.invisible-hand.{game.State.Round}.{frame.Source}.{entity}"] = 1;
        RoundModule.ObserveObjectives(game);
        if (rapid)
        {
            int[] remaining = (frame.Indices ?? []).Where(squad => squad != entity).ToArray();
            if (remaining.Length == 0) ReturnReservedRelease(game, frame);
            else game.Flow.Continuations[^1] = frame with { Indices = remaining };
        }
        else game.Flow.Continuations[^1] = frame with { Value = 0, Indices = [] };
        return true;
    }
    public static bool IsLegalReservedSquadronPlacement(GameContext game, int ship, int squadron, int upgradeIndex, Vector2 position, float range)
    {
        if (!game.IsReserved(squadron) || game.State.Registers[game.Topology.Registers[$"entity.{squadron}.reserveUpgradePlusOne"]] != upgradeIndex + 1
            || game.State.Registers[game.Topology.Registers[$"entity.{squadron}.reserveSourcePlusOne"]] != ship + 1) return false;
        var destination = new Circle(position, game.SquadronBase(squadron).Radius);
        return Geometry2D.Distance(destination, game.ShipBase(ship)) <= range + .011425f && SquadronModule.IsLegalPlacement(game, squadron, position);
    }
    public static bool HasInvisibleHandAiBonus(GameContext game, int squadron) => game.IsShip(game.Flow.ActiveShip)
        && game.Flow.Flags.ContainsKey($"movement.invisible-hand.{game.State.Round}.{game.Flow.ActiveShip}.{squadron}");
    private static int FindReserveUpgrade(GameContext game, int ship, string cardId)
    {
        for (int i = 0; i < game.State.Upgrades.Length; i++)
            if (game.State.Upgrades[i].Active && game.State.Upgrades[i].Definition.id == cardId && game.State.Registers[game.Topology.Registers.UpgradeOwner(i)] == ship + 1) return i;
        return -1;
    }
    private static IEnumerable<int> ReservedSquadrons(GameContext game, int ship, int upgradeIndex)
        => game.Topology.Squadrons.Where(s => game.IsReserved(s.EntityIndex) && !game.IsDestroyed(s.EntityIndex)
            && game.State.Registers[game.Topology.Registers[$"entity.{s.EntityIndex}.reserveUpgradePlusOne"]] == upgradeIndex + 1
            && game.State.Registers[game.Topology.Registers[$"entity.{s.EntityIndex}.reserveSourcePlusOne"]] == ship + 1).Select(s => s.EntityIndex);
    private static void ReturnReservedRelease(GameContext game, GameContinuation frame)
    { EngineEffects.ReturnFromChoice(game); game.Flow.Phase = (GamePhase)frame.Stage; }
}
