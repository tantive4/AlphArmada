using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Armada.Session;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Official squadron card timings absent from the recovered engine. All pending
/// choices and parent activations live in branch-owned flow data, never in static fields.</summary>
public static class MissingSquadronLifecycleEffects
{
    public const string Boba = "boba-fett-slave-i", Gar = "gar-saxon-mandalorian-gauntlet-fighter",
        Han = "han-solo-millennium-falcon", Ig88B = "ig-88b-ig-2000b", Moralo = "moralo-eval-yv-666",
        Morna = "morna-kee-vt-49-decimator", Whisper = "whisper-tie-phantom-squadron";
    public static bool Handles(string rule) => rule.StartsWith("official.squad.", StringComparison.Ordinal);

    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        if (timing is not ("start-ship-phase" or "start-squadron-phase" or "end-squadron-phase")) yield break;
        foreach (var squad in game.Topology.Squadrons)
        {
            int entity = squad.EntityIndex;
            if (!Live(game, entity)) continue;
            string id = squad.CatalogEntry.Definition.SquadronId;
            if (timing == "start-ship-phase" && id == Han && SquadronModule.CanActivate(game, entity, game.State.Owners[entity], false))
                yield return new(new(GameActionKind.UseAbility, entity, AbilityId: "official.han.activate"), "Han Solo: activate as if by a Squadron command");
            if (timing == "start-squadron-phase" && id == Ig88B && SquadronModule.CanActivate(game, entity, game.State.Owners[entity], false))
                yield return new(new(GameActionKind.UseAbility, entity, AbilityId: "official.ig88b.begin"), "IG-88B: toggle activated and attack enemy squadrons");
            if (timing == "end-squadron-phase" && (game.State.Squadrons[squad.SquadronIndex].TemporaryCloak
                || SquadronAbilityEngine.HasKeyword(squad.CatalogEntry.Definition, "CLOAK")))
                yield return new(new(GameActionKind.UseAbility, entity, AbilityId: "official.cloak.move"), "Cloak: move up to distance 1, even while engaged");
        }
    }
    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        switch (action.AbilityId)
        {
            case "official.han.activate":
                SquadronModule.BeginStandaloneActivation(game, action.Source, GamePhase.AbilityChoice); return true;
            case "official.ig88b.begin":
                game.State.Squadrons[game.SquadronIndex(action.Source)].MarkActivated(false); game.State.Activated[action.Source] = true;
                EngineEffects.PushChoice(game, new("official.squad.ig88b", action.Source, Stage: (int)GamePhase.AbilityChoice,
                    Indices: [game.Flow.ActiveShip, game.Flow.ActiveSquadron, game.Flow.SquadronCanMove ? 1 : 0, game.Flow.SquadronCanAttack ? 1 : 0, game.Flow.SquadronCommanded ? 1 : 0]), []);
                return true;
            case "official.cloak.move":
                BeginMove(game, action.Source, GamePhase.AbilityChoice, "cloak"); return true;
        }
        return false;
    }

    public static bool BeginActivationTiming(GameContext game, int entity, GamePhase returnPhase)
    {
        if (!Live(game, entity)) return false;
        using var scope = new GameScope(game);
        var definition = game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition;
        int[] gars = definition.Keywords.Any(k => k.Equals("INTEL", StringComparison.OrdinalIgnoreCase) || k.StartsWith("RELAY ", StringComparison.OrdinalIgnoreCase) || k.Equals("RELAY", StringComparison.OrdinalIgnoreCase))
            ? game.Topology.Squadrons.Where(s => Is(game, s.EntityIndex, Gar) && game.State.Owners[s.EntityIndex] != game.State.Owners[entity]
                && Distance(game, s.EntityIndex, entity) <= TabletopMeasurements.DistanceOne).Select(s => s.EntityIndex).ToArray() : [];
        if (gars.Length == 0 && !Is(game, entity, Boba) && !(Is(game, entity, Morna) && DiscardedTokens(game, entity).Any())) return false;
        if (game.Flow.SquadronProgress is { } progress)
        {
            progress.ActivationCommitted = true;
            if (progress.Commanded) CommandModule.CommitSquadronSelection(game, entity);
        }
        EngineEffects.PushChoice(game, new("official.squad.activation", entity, Stage: (int)returnPhase, Text: "gar", Indices: gars), []); return true;
    }

    /// <summary>Called after every response/counter to the defended attack has resolved.</summary>
    public static bool BeginWhisperAfterAttack(GameContext game, int defender, GamePhase resumePhase)
    {
        if (!Is(game, defender, Whisper)) return false;
        BeginMove(game, defender, resumePhase, "whisper"); return true;
    }
    private static void BeginMove(GameContext game, int entity, GamePhase returnPhase, string reason)
        => EngineEffects.PushChoice(game, new("official.squad.move", entity, Stage: (int)returnPhase, Text: reason), []);

    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = new List<GameActionOption>();
        if (frame.Rule == "official.squad.activation")
        {
            if (!Live(game, frame.Source)) return Forced("Finish destroyed squadron's activation triggers");
            if (frame.Text == "gar") return Forced("Resolve activation damage from Gar Saxon");
            if (frame.Text == "boba")
            {
                if (Is(game, frame.Source, Boba))
                    foreach (int target in Enumerable.Range(0, game.Topology.EntityCount).Where(e => e != frame.Source && Live(game, e)
                        && game.State.Owners[e] != game.State.Owners[frame.Source] && Distance(game, frame.Source, e) <= TabletopMeasurements.DistanceOne))
                        options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, target), "Boba Fett: chosen enemy suffers 1 damage"));
                return options.Count > 0 ? Player(game, frame.Source, options) : Forced("No eligible Boba Fett target");
            }
            if (Is(game, frame.Source, Morna))
                foreach (var token in DiscardedTokens(game, frame.Source)) options.Add(new(new(GameActionKind.ChooseToken, frame.Source, Token: token), "Morna Kee: recover discarded " + token));
        }
        else if (frame.Rule == "official.squad.move")
        {
            if (Live(game, frame.Source)) options.Add(new(new(GameActionKind.MoveSquadron, frame.Source), "Choose destination up to distance 1", AcceptsPosition: true));
        }
        else if (frame.Rule is "official.squad.move-completed" or "official.squad.move-before") return Forced("Finish bonus squadron movement");
        else if (frame.Rule == "official.squad.ig88b")
        {
            if (Live(game, frame.Source))
                foreach (int target in IgTargets(game, frame)) options.Add(new(new(GameActionKind.Attack, frame.Source, target), "IG-88B: obstructed attack against this squadron"));
        }
        else if (frame.Rule == "official.squad.moralo")
        {
            if (Is(game, frame.Source, Moralo)) options.Add(new(new(GameActionKind.ActivateSquadron, frame.Source), "Moralo Eval: move up to distance 1 and attack"));
        }
        else throw new NotSupportedException(frame.Rule);
        options.Add(new(new(GameActionKind.Pass, frame.Source), "Skip or finish"));
        return options.Count == 1 ? Forced(options[0].Label, options[0].Template) : Player(game, frame.Source, options);
    }

    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!GetDecision(game, frame).Actions.Any(o => GameSession.Matches(o, action))) throw new ArgumentException("Illegal squadron ability choice.");
        if (frame.Rule == "official.squad.activation")
        {
            if (!Live(game, frame.Source)) { Return(game, frame); return; }
            if (frame.Text == "gar")
            {
                var gars = frame.Indices ?? [];
                if (frame.Value < gars.Length)
                {
                    int gar = gars[(int)frame.Value];
                    game.Flow.Continuations[^1] = frame with { Value = frame.Value + 1 };
                    CombatModule.QueueDamage(game, frame.Source, 1, false, true, GamePhase.AbilityChoice, rule: "official.gar-saxon", source: gar); return;
                }
                game.Flow.Continuations[^1] = frame with { Text = "boba", Value = 0 }; return;
            }
            if (frame.Text == "boba")
            {
                game.Flow.Continuations[^1] = frame with { Text = "morna" };
                if (action.Kind == GameActionKind.ChooseEntity)
                    CombatModule.QueueDamage(game, action.Target, 1, false, false, GamePhase.AbilityChoice, rule: "official.boba-fett", source: frame.Source);
                return;
            }
            if (action.Kind == GameActionKind.ChooseToken && !game.State.Squadrons[game.SquadronIndex(frame.Source)].TryRecoverDiscardedDefenseToken(action.Token))
                throw new ArgumentException("Discarded defense token unavailable.");
            Return(game, frame); return;
        }
        if (frame.Rule == "official.squad.move")
        {
            if (action.Kind == GameActionKind.Pass) { Return(game, frame); return; }
            var position = action.Position ?? throw new ArgumentException("Destination required.");
            if (Vector2.Distance(game.State.Poses[frame.Source].Position, position) > TabletopMeasurements.DistanceOne + Geometry2D.DefaultTolerance
                || !SquadronModule.IsLegalMoveDestination(game, frame.Source, position)) throw new ArgumentException("Illegal bonus movement destination.");
            var pending = frame with { Rule = "official.squad.move-before", Indices = [BitConverter.SingleToInt32Bits(position.X), BitConverter.SingleToInt32Bits(position.Y)] };
            game.Flow.Continuations[^1] = pending;
            if (!EngineEffects.BeginWindow(game, "before-squadron-move", frame.Source, resumePhase: GamePhase.AbilityChoice)) CommitBonusMove(game, pending);
            return;
        }
        if (frame.Rule == "official.squad.move-before") { CommitBonusMove(game, frame); return; }
        if (frame.Rule == "official.squad.move-completed")
        {
            if (frame.Value == 0) CompleteBonusMove(game, frame); else Return(game, frame); return;
        }
        if (frame.Rule == "official.squad.ig88b")
        {
            if (action.Kind == GameActionKind.Pass)
            {
                var saved = frame.Indices!;
                game.Flow.ActiveShip = saved[0]; game.Flow.ActiveSquadron = saved[1]; game.Flow.SquadronCanMove = saved[2] != 0;
                game.Flow.SquadronCanAttack = saved[3] != 0; game.Flow.SquadronCommanded = saved[4] != 0;
                Return(game, frame); return;
            }
            game.Flow.Continuations[^1] = frame with { Indices = [..frame.Indices!, action.Target] };
            game.Flow.Flags["combat.granted.obstructed"] = 1;
            game.Flow.Flags["combat.granted.ignore-engagement"] = 1;
            CombatModule.QueueGrantedAttack(game, frame.Source, action.Target, returnPhase: GamePhase.AbilityChoice); return;
        }
        if (frame.Rule == "official.squad.moralo")
        {
            Return(game, frame);
            if (action.Kind == GameActionKind.ActivateSquadron)
                SquadronModule.BeginStandaloneActivation(game, frame.Source, (GamePhase)frame.Stage, true, TabletopMeasurements.DistanceOne);
            return;
        }
        throw new NotSupportedException(frame.Rule);
    }
    private static void CommitBonusMove(GameContext game, GameContinuation frame)
    {
        var position = new Vector2(BitConverter.Int32BitsToSingle(frame.Indices![0]), BitConverter.Int32BitsToSingle(frame.Indices[1]));
        if (!Live(game, frame.Source)) { Return(game, frame); return; }
        if (!SquadronModule.IsLegalMoveDestination(game, frame.Source, position))
        { game.Flow.Continuations[^1] = frame with { Rule = "official.squad.move", Indices = null }; return; }
        game.State.Poses[frame.Source] = new(position, game.State.Poses[frame.Source].YawDegrees);
        if (PlayAreaDestruction.DestroyIfOutside(game, frame.Source)) { Return(game, frame); return; }
        var completed = frame with { Rule = "official.squad.move-completed", Value = 0 };
        game.Flow.Continuations[^1] = completed;
        RoundModule.ObserveObjectives(game);
        if (!OfficialCreatureObjectives.BeginSquadronMoveConsequences(game, frame.Source, GamePhase.AbilityChoice)) CompleteBonusMove(game, completed);
    }
    private static void CompleteBonusMove(GameContext game, GameContinuation frame)
    {
        if (!Live(game, frame.Source)) { Return(game, frame); return; }
        game.Flow.Continuations[^1] = frame with { Value = 1 };
        if (!EngineEffects.BeginWindow(game, "after-squadron-move", frame.Source, resumePhase: GamePhase.AbilityChoice)) Return(game, frame);
    }
    /// <summary>Complete an actual bonus move through obstacle effects and the same
    /// after-movement timing window used by ordinary activation movement.</summary>
    public static void BeginBonusMoveConsequences(GameContext game, int squadron, GamePhase returnPhase)
    {
        if (!Live(game, squadron)) return;
        var completed = new GameContinuation("official.squad.move-completed", squadron, Stage: (int)returnPhase);
        EngineEffects.PushChoice(game, completed, []);
        if (!OfficialCreatureObjectives.BeginSquadronMoveConsequences(game, squadron, GamePhase.AbilityChoice)) CompleteBonusMove(game, completed);
    }
    private static IEnumerable<int> IgTargets(GameContext game, GameContinuation frame)
    {
        using var scope = new GameScope(game);
        int source = frame.Source;
        foreach (var squad in game.Topology.Squadrons)
            if (Live(game, squad.EntityIndex) && game.State.Owners[squad.EntityIndex] != game.State.Owners[source]
                && !(frame.Indices ?? []).Skip(5).Contains(squad.EntityIndex)
                && Distance(game, source, squad.EntityIndex) <= TabletopMeasurements.DistanceOne
                && SquadronAbilityEngine.CanAttackSquadron(game.SquadronViews[game.SquadronIndex(source)], game.SquadronViews[squad.SquadronIndex], game.SceneSquadronViews, out _))
                yield return squad.EntityIndex;
    }
    private static IEnumerable<DefenseTokenType> DiscardedTokens(GameContext game, int entity)
        => Enum.GetValues<DefenseTokenType>().Where(t => game.State.Squadrons[game.SquadronIndex(entity)].GetDiscardedDefenseTokenCount(t) > 0);
    private static bool Live(GameContext game, int entity) => entity >= 0 && entity < game.Topology.EntityCount && !game.IsDestroyed(entity) && !game.IsReserved(entity);
    private static bool Is(GameContext game, int entity, string id) => Live(game, entity) && !game.IsShip(entity)
        && game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition.SquadronId == id;
    private static float Distance(GameContext game, int squadron, int other) => EntityGeometry.Distance(game, squadron, other);
    private static GameDecision Forced(string label, GameAction? action = null) => new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(action ?? new(GameActionKind.Pass), label)]);
    private static GameDecision Player(GameContext game, int entity, IReadOnlyList<GameActionOption> options) => new(DecisionKind.Player, game.State.Owners[entity], GamePhase.AbilityChoice, options);
    private static void Return(GameContext game, GameContinuation frame)
    { EngineEffects.ReturnFromChoice(game); game.Flow.Phase = (GamePhase)frame.Stage; }

    /// <summary>Objective implementations with token stores outside map/Intel storage may call
    /// this after moving/removing a token. Map/Intel tokens are observed automatically; do not
    /// also notify them explicitly. Eligibility is captured at the token's former position.</summary>
    public static void NotifyObjectiveTokenMovedOrRemoved(GameContext game, Vector2 formerPosition, bool wasOnObstacle, float tokenRadius = 0)
    {
        if (!float.IsFinite(tokenRadius) || tokenRadius < 0) throw new ArgumentOutOfRangeException(nameof(tokenRadius));
        if (wasOnObstacle) return;
        for (int priority = 0; priority < 2; priority++)
        {
            int owner = priority == 0 ? game.Topology.FirstPlayer : 3 - game.Topology.FirstPlayer;
            foreach (var squad in game.Topology.Squadrons)
                if (game.State.Owners[squad.EntityIndex] == owner && Is(game, squad.EntityIndex, Moralo)
                    && Math.Max(0, Vector2.Distance(game.State.Poses[squad.EntityIndex].Position, formerPosition)
                        - game.SquadronBase(squad.EntityIndex).Radius - tokenRadius) <= TabletopMeasurements.DistanceBand(2) + Geometry2D.DefaultTolerance)
                    game.Flow.PendingMoraloTriggers.Add(new(squad.EntityIndex, formerPosition));
        }
    }
    /// <summary>Watches both generic map objective pieces and SWArmada Intel tokens. The
    /// first observation establishes a baseline; exact observations are snapshot data.</summary>
    public static void ObserveObjectiveTokens(GameContext game)
    {
        if (!game.Topology.HasMoraloEval) return;
        if (game.Flow.Phase == GamePhase.Deployment)
        { game.Flow.ObservedObjectiveTokens.Clear(); return; }
        for (int i = 0; i < game.State.SpatialObjects.Count; i++)
        {
            var token = game.State.SpatialObjects[i];
            if (token.Kind == (int)MapPlacementKind.Objective)
                ObserveToken(game, 0, i, token.Pose.Position, token.Active, Math.Max(token.Size.X, token.Size.Y) * 0.5f);
        }
        if (game.State.Objective is IntelRetrievalObjective intel)
            foreach (var token in intel.Tokens) ObserveToken(game, 1, token.Sequence, token.Position, token.Available, 0);
    }
    private static void ObserveToken(GameContext game, int kind, int index, Vector2 position, bool active, float radius)
    {
        var observations = game.Flow.ObservedObjectiveTokens;
        for (int i = 0; i < observations.Count; i++)
        {
            var old = observations[i];
            if (old.Kind != kind || old.Index != index) continue;
            if (old.Active == active && old.Position == position)
            {
                bool onObstacle = TokenOnObstacle(game, position, radius);
                if (old.OnObstacle != onObstacle || old.Radius != radius) observations[i] = old with { OnObstacle = onObstacle, Radius = radius };
                return;
            }
            if (old.Active && (!active || old.Position != position)) NotifyObjectiveTokenMovedOrRemoved(game, old.Position, old.OnObstacle, old.Radius);
            observations[i] = new(kind, index, position, active, TokenOnObstacle(game, position, radius), radius); return;
        }
        observations.Add(new(kind, index, position, active, TokenOnObstacle(game, position, radius), radius));
    }
    public static bool TryBeginNext(GameContext game)
    {
        if (game.Flow.PendingMoraloTriggers.Count == 0 || game.State.Outcome is not null) return false;
        // Do not interrupt an unfinished physical damage assignment or nested choice.
        if (game.Flow.Phase is GamePhase.RollAttack or GamePhase.DealDamage or GamePhase.ChooseDamageCard or GamePhase.ResolveCritical
            || game.Flow.Phase == GamePhase.AbilityChoice && game.Flow.Continuations.LastOrDefault()?.Rule != "effects.window") return false;
        while (game.Flow.PendingMoraloTriggers.Count > 0)
        {
            var trigger = game.Flow.PendingMoraloTriggers[0]; game.Flow.PendingMoraloTriggers.RemoveAt(0);
            if (!Is(game, trigger.Squadron, Moralo)) continue;
            EngineEffects.PushChoice(game, new("official.squad.moralo", trigger.Squadron, Stage: (int)game.Flow.Phase), []); return true;
        }
        return false;
    }
    private static bool TokenOnObstacle(GameContext game, Vector2 token, float radius)
    {
        for (int i = 0; i < game.State.SpatialObjects.Count; i++)
        {
            var obstacle = game.State.SpatialObjects[i];
            if (obstacle.Active && obstacle.Kind == (int)MapPlacementKind.Obstacle)
            {
                var shape = ObstacleGeometry.Shape(game, i);
                if (radius > 0 ? Geometry2D.Overlaps(new Circle(token, radius), shape) : shape.Contains(token)) return true;
            }
        }
        return false;
    }
}
