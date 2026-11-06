using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public sealed record DeathTrigger(int Entity, int Source, int Effects);

/// <summary>One destruction observer for attacks, faceup cards, mines, rams and direct
/// squadron abilities. Detection never interrupts the mutating rule midway through its
/// bookkeeping; the session opens queued choices at the next decision boundary.</summary>
public static class DamageEvents
{
    private const int Grievous = 1, Reserve = 2, HotLanding = 4, Rieekan = 8;
    public static bool SupportsUpgrade(string id) => id is "general-rieekan-commander" or "general-grievous-commander" or "reserve-hangar-deck" or "hot-landing";
    public static bool HasPending(GameContext game) => game.Flow.DeathQueue.Count > 0 || game.Flow.Continuations.Any(frame => Handles(frame.Rule));
    public static void Observe(GameContext game, int source = -1)
    {
        using var scope = new GameScope(game);
        if (game.Flow.DeathRecorded.Length != game.Topology.EntityCount) game.Flow.DeathRecorded = new bool[game.Topology.EntityCount];
        for (int entity = 0; entity < game.Topology.EntityCount; entity++)
        {
            bool lethal = game.IsShip(entity) ? game.State.Ships[entity].Destroyed : game.State.Squadrons[game.SquadronIndex(entity)].LethallyDamaged;
            if (!lethal) { game.Flow.DeathRecorded[entity] = false; continue; }
            bool outside = PlayAreaDestruction.WasRemoved(game, entity);
            if (!outside && MissingSquadronCombatEffects.DefersTelDestruction(game, entity)) continue;
            if (game.Flow.DeathRecorded[entity] || game.IsReserved(entity)) continue;
            game.Flow.DeathRecorded[entity] = true;
            int owner = game.State.Owners[entity]; int effects = 0;
            bool grievous = FleetCard(game, owner, "general-grievous-commander", entity);
            bool unique = game.IsShip(entity) || game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition.IsUnique;
            if (!outside && RieekanWindowOpen(game) && unique && FleetCard(game, owner, "general-rieekan-commander", entity)
                && game.Flow.Flags.GetValueOrDefault("rieekan.used." + owner) != game.State.Round)
            {
                // Keep the unit available until its owner decides whether to use the
                // replacement. The round limit is spent only by accepting the effect.
                game.Flow.Flags["rieekan.pending." + entity] = 1; effects |= Rieekan;
                if (game.IsShip(entity)) game.Flow.Flags["rieekan.ship." + entity] = 1;
                else
                {
                    game.State.Squadrons[game.SquadronIndex(entity)].DeferDestruction();
                    game.Flow.Flags["rieekan.squadron." + entity] = 1;
                }
            }
            if (grievous && GrievousOptions(game, entity).Any()) effects |= Grievous;
            if (!game.IsShip(entity))
            {
                var squadron = game.SquadronViews[game.SquadronIndex(entity)];
                if (!outside && !unique && SquadronAbilityEngine.HasKeyword(squadron.Definition, "SWARM") && ReserveCarriers(game, entity).Any())
                { squadron.State.DeferDestruction(); effects |= Reserve; }
                if (SquadronAbilityEngine.GetKeywordValue(squadron.Definition, "ADEPT") > 0 && FleetCard(game, owner, "hot-landing", entity)
                    && HotLandingOptions(game, entity).Any()) effects |= HotLanding;
            }
            if (effects != 0) game.Flow.DeathQueue.Add(new(entity, source, effects));
        }
    }
    public static bool TryBeginNext(GameContext game)
    {
        if (game.Flow.DeathQueue.Count == 0 || game.Flow.Continuations.LastOrDefault() is { } current && Handles(current.Rule)) return false;
        var trigger = game.Flow.DeathQueue[0]; game.Flow.DeathQueue.RemoveAt(0);
        var frame = new GameContinuation("death.sequence", trigger.Entity, trigger.Source, (int)game.Flow.Phase, trigger.Effects);
        EngineEffects.PushChoice(game, frame, []); return true;
    }
    public static bool Handles(string rule) => rule.StartsWith("death.", StringComparison.Ordinal);
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = Options(game, frame).ToArray();
        bool forced = options.Length == 1 && (options[0].Template.Kind == GameActionKind.Pass || frame.Rule == "death.sequence");
        return new(forced ? DecisionKind.Forced : DecisionKind.Player, forced ? 0 : game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
    }
    private static IEnumerable<GameActionOption> Options(GameContext game, GameContinuation frame)
    {
        if (frame.Rule == "death.sequence")
        {
            if (frame.Value == 0) { yield return Finish(frame); yield break; }
            if ((frame.Value & Rieekan) != 0)
            { yield return new(new(GameActionKind.UseAbility, frame.Source, AbilityId: "death.rieekan"), "Decide whether to resolve General Rieekan"); yield break; }
            if ((frame.Value & Grievous) != 0) yield return new(new(GameActionKind.UseAbility, frame.Source, AbilityId: "death.grievous"), "Resolve General Grievous");
            if ((frame.Value & Reserve) != 0) yield return new(new(GameActionKind.UseAbility, frame.Source, AbilityId: "death.reserve"), "Resolve Reserve Hangar Deck");
            if ((frame.Value & (Grievous | Reserve)) == 0 && (frame.Value & HotLanding) != 0)
                yield return new(new(GameActionKind.UseAbility, frame.Source, AbilityId: "death.hot-landing"), "Resolve Hot Landing raid");
            yield break;
        }
        if (frame.Rule == "death.rieekan")
        {
            if (RieekanWindowOpen(game) && !PlayAreaDestruction.WasRemoved(game, frame.Source) && game.Flow.Flags.ContainsKey("rieekan.pending." + frame.Source) &&
                game.Flow.Flags.GetValueOrDefault("rieekan.used." + game.State.Owners[frame.Source]) != game.State.Round)
                yield return new(new(GameActionKind.UseAbility, frame.Source, AbilityId: "death.rieekan.defer"), "Use General Rieekan until the end of Status");
        }
        else if (frame.Rule == "death.grievous") foreach (var option in GrievousOptions(game, frame.Source)) yield return option;
        else if (frame.Rule == "death.reserve")
            foreach (int carrier in ReserveCarriers(game, frame.Source))
                yield return new(new(GameActionKind.PlaceReservedUnit, frame.Source, carrier, AbilityId: "death.reserve.place"), "Return squadron unengaged at distance 1 of carrier", AcceptsPosition: true);
        else if (frame.Rule == "death.hot-landing") foreach (var option in HotLandingOptions(game, frame.Source)) yield return option;
        else throw new NotSupportedException("Unknown destruction rule " + frame.Rule);
        yield return Finish(frame);
    }
    private static IEnumerable<GameActionOption> GrievousOptions(GameContext game, int destroyed)
    {
        for (int target = 0; target < game.Topology.EntityCount; target++)
        {
            if (target == destroyed || game.State.Owners[target] != game.State.Owners[destroyed] || game.IsDestroyed(target) || game.IsReserved(target)
                || !game.IsShip(target) && !game.Topology.Squadrons[game.SquadronIndex(target)].CatalogEntry.Definition.IsUnique || Distance(game, destroyed, target) > TabletopMeasurements.LongRange) continue;
            foreach (var token in Enum.GetValues<DefenseTokenType>())
            {
                int ready = game.IsShip(target) ? game.State.Ships[target].GetReadyDefenseTokenCount(token) : game.State.Squadrons[game.SquadronIndex(target)].GetReadyDefenseTokenCount(token);
                int available = game.IsShip(target) ? game.State.Ships[target].GetAvailableDefenseTokenCount(token) : game.State.Squadrons[game.SquadronIndex(target)].GetAvailableDefenseTokenCount(token);
                int discarded = game.IsShip(target) ? game.State.Ships[target].GetDiscardedDefenseTokenCount(token) : game.State.Squadrons[game.SquadronIndex(target)].GetDiscardedDefenseTokenCount(token);
                if (available > ready) yield return new(new(GameActionKind.ChooseToken, destroyed, target, Token: token, AbilityId: "death.grievous.ready"), "Ready friendly " + token);
                if (discarded > 0) yield return new(new(GameActionKind.ChooseToken, destroyed, target, Token: token, AbilityId: "death.grievous.recover"), "Recover friendly " + token + " exhausted");
            }
        }
    }
    private static IEnumerable<GameActionOption> HotLandingOptions(GameContext game, int destroyed)
    {
        foreach (var ship in game.SceneShipViews.Where(ship => ship.OwnerId != game.State.Owners[destroyed] && Distance(game, destroyed, ship.Index) <= TabletopMeasurements.CloseRange))
            foreach (var command in Enum.GetValues<ShipCommand>().Where(ship.Commands.CanGainRaidToken))
                yield return new(new(GameActionKind.ChooseCommand, destroyed, ship.Index, Command: command, AbilityId: "death.hot-landing.raid"), "Place " + command + " raid token");
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!Options(game, frame).Any(option => GameSession.Matches(option, action))) throw new ArgumentException("Illegal destruction choice.");
        if (frame.Rule == "death.sequence")
        {
            if (action.Kind == GameActionKind.Pass) { Pop(game, (GamePhase)frame.Stage); return; }
            int bit = action.AbilityId switch { "death.rieekan" => Rieekan, "death.grievous" => Grievous, "death.reserve" => Reserve, "death.hot-landing" => HotLanding, _ => throw new ArgumentException("Unknown destruction effect.") };
            game.Flow.Continuations[^1] = frame with { Value = frame.Value & ~bit };
            EngineEffects.PushChoice(game, new(action.AbilityId, frame.Source, frame.Target), []); return;
        }
        if (action.Kind == GameActionKind.Pass)
        {
            if (frame.Rule == "death.rieekan")
            {
                game.Flow.Flags.Remove("rieekan.pending." + frame.Source);
                if (game.IsShip(frame.Source)) game.Flow.Flags.Remove("rieekan.ship." + frame.Source);
                else
                {
                    game.Flow.Flags.Remove("rieekan.squadron." + frame.Source);
                    game.State.Squadrons[game.SquadronIndex(frame.Source)].FinalizeDeferredDestruction();
                }
            }
            if (frame.Rule == "death.reserve") game.State.Squadrons[game.SquadronIndex(frame.Source)].FinalizeDeferredDestruction();
            Pop(game, GamePhase.AbilityChoice); return;
        }
        if (frame.Rule == "death.rieekan")
        {
            game.Flow.Flags.Remove("rieekan.pending." + frame.Source);
            game.Flow.Flags["rieekan.used." + game.State.Owners[frame.Source]] = game.State.Round;
        }
        else if (frame.Rule == "death.grievous")
        {
            bool recover = action.AbilityId.EndsWith("recover", StringComparison.Ordinal);
            if (game.IsShip(action.Target))
            {
                var state = game.State.Ships[action.Target];
                if (recover) { if (!state.TryRecoverDiscardedDefenseToken(action.Token) || !state.SpendDefenseToken(action.Token)) throw new InvalidOperationException("Cannot recover token."); }
                else if (!state.TryReadyDefenseToken(action.Token)) throw new InvalidOperationException("Cannot ready token.");
            }
            else
            {
                var state = game.State.Squadrons[game.SquadronIndex(action.Target)];
                if (recover) { if (!state.TryRecoverDiscardedDefenseToken(action.Token) || !state.SpendDefenseToken(action.Token)) throw new InvalidOperationException("Cannot recover token."); }
                else if (!state.TryReadyDefenseToken(action.Token)) throw new InvalidOperationException("Cannot ready token.");
            }
        }
        else if (frame.Rule == "death.reserve")
        {
            if (action.Position is not { } position || !LegalReservePosition(game, frame.Source, action.Target, position)) throw new ArgumentException("Illegal Reserve Hangar return position.");
            var card = game.ShipViews[action.Target].Loadout.Find("reserve-hangar-deck")!; var state = game.State.Squadrons[game.SquadronIndex(frame.Source)];
            if (!card.Discard() || !state.FinalizeDeferredDestruction() || !state.Revive(2)) throw new InvalidOperationException("Prevalidated squadron revival failed.");
            game.State.Poses[frame.Source] = new(position, game.State.Poses[frame.Source].YawDegrees); game.State.Activated[frame.Source] = true; game.Flow.DeathRecorded[frame.Source] = false;
        }
        else game.State.Commands[action.Target].GainRaidToken(action.Command);
        Pop(game, GamePhase.AbilityChoice);
    }
    private static IEnumerable<int> ReserveCarriers(GameContext game, int destroyed) => game.SceneShipViews.Where(ship => ship.OwnerId == game.State.Owners[destroyed]
        && ship.Loadout.HasActive("reserve-hangar-deck") && Distance(game, destroyed, ship.Index) <= TabletopMeasurements.LongRange).Select(ship => ship.Index);
    private static bool LegalReservePosition(GameContext game, int entity, int carrier, Vector2 position)
    {
        if (!float.IsFinite(position.X) || !float.IsFinite(position.Y) || !ReserveCarriers(game, entity).Contains(carrier)) return false;
        var circle = new Circle(position, game.SquadronBase(entity).Radius);
        if (Geometry2D.Distance(circle, game.ShipBase(carrier)) > TabletopMeasurements.DistanceOne
            || !PlayAreaGeometry.Contains(new(new(Vector2.Zero), game.Topology.BoardSizeMillimetres), circle)) return false;
        foreach (var ship in game.SceneShipViews) if (!game.Topology.Ships[ship.Index].IsStation && Geometry2D.Overlaps(circle, game.ShipBase(ship.Index))) return false;
        foreach (var squad in game.SceneSquadronViews.Where(squad => squad.Index != entity))
            if (Geometry2D.Overlaps(circle, game.SquadronBase(squad.Index)) || squad.OwnerId != game.State.Owners[entity]
                && Geometry2D.Distance(circle, game.SquadronBase(squad.Index)) <= TabletopMeasurements.DistanceOne) return false;
        for (int index = 0; index < game.State.SpatialObjects.Count; index++)
        {
            var obstacle = game.State.SpatialObjects[index]; if (!obstacle.Active) continue;
            if (obstacle.Kind == 0 ? Geometry2D.Overlaps(circle, ObstacleGeometry.Shape(game, index))
                : obstacle.Kind == SetupModule.ProximityMineKind && Geometry2D.Overlaps(circle, OfficialObjectives.PieceCircle(game, index))) return false;
        }
        return true;
    }
    private static float Distance(GameContext game, int first, int second) => EntityGeometry.Distance(game, first, second);
    private static bool FleetCard(GameContext game, int owner, string id, int dying) => game.Topology.Ships.Any(ship => ship.Owner == owner
        && !game.IsReserved(ship.EntityIndex) && (ship.EntityIndex == dying || !game.IsDestroyed(ship.EntityIndex)) && game.HasUpgrade(ship.EntityIndex, id));
    // The user ruling closes Rieekan after Status, including a previously unused
    // once-per-round effect: end-of-round objective damage cannot begin a new deferral.
    private static bool RieekanWindowOpen(GameContext game)
        => game.Flow.Flags.GetValueOrDefault("status.finalized") != game.State.Round;
    private static GameActionOption Finish(GameContinuation frame) => new(new(GameActionKind.Pass, frame.Source, AbilityId: "death.finish"), "Finish destruction effect");
    private static void Pop(GameContext game, GamePhase phase)
    { game.Flow.Continuations.RemoveAt(game.Flow.Continuations.Count - 1); game.Flow.PendingOptions = []; game.Flow.Phase = phase; }
}
