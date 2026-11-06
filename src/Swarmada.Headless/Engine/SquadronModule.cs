using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;
using PlayAreaGeometry = Swarmada.Headless.Geometry.PlayAreaGeometry;

namespace Swarmada.Headless.Engine;

public enum SquadronBatchKind { Normal, HyperwaveSignalBoost }

/// <summary>Source squadron activation and continuous destination rules. Movement positions are
/// geometric player inputs; the engine imposes neither a grid nor a finite candidate sampler.</summary>
public sealed class SquadronModule : IGameplayModule
{
    public bool Handles(GamePhase phase) => phase is GamePhase.SquadronPhase or GamePhase.SquadronActivation or GamePhase.SquadronMove;
    public static bool CanActivate(GameContext game, int entity, int owner, bool commanded)
    {
        if (entity < game.State.Ships.Length || entity >= game.Topology.EntityCount || game.State.Owners[entity] != owner
            || game.IsDestroyed(entity) || game.IsReserved(entity)) return false;
        var state = game.State.Squadrons[game.SquadronIndex(entity)];
        return !state.Activated && !game.State.Activated[entity];
    }
    public static void BeginActivation(GameContext game, int entity, bool commanded)
    {
        if (!CanActivate(game, entity, game.State.Owners[entity], commanded)) throw new ArgumentException("Squadron cannot activate.", nameof(entity));
        var state = game.State.Squadrons[game.SquadronIndex(entity)];
        var progress = new SquadronActivationProgress { Entity = entity, Commanded = commanded, BeforeSelection = state.CaptureSnapshot() };
        if (commanded)
        {
            if (!game.IsShip(game.Flow.ActiveShip)) throw new InvalidOperationException("Commanded squadron requires an active commanding ship.");
            state.MarkCommandedBy(game.Topology.Ships[game.Flow.ActiveShip].RuntimeId);
            MovementAbilityEffects.ApplyCommandedSquadronGrants(game, game.Flow.ActiveShip, entity);
            var wolffeKey = $"wolffe.{game.Flow.ActiveShip}.{entity}";
            bool hadWolffeGrant = game.Flow.Flags.ContainsKey(wolffeKey);
            ActivationAbilityEffects.ApplyCommandedKeywords(game, entity);
            progress.WolffeGrantAdded = !hadWolffeGrant && game.Flow.Flags.ContainsKey(wolffeKey);
        }
        // Hyperwave treats AI squadrons as commanded only while attacking. It does
        // not grant a commanded activation or permission to both move and attack.
        using var scope = new GameScope(game);
        progress.CanMoveAndAttack = SquadronAbilityEngine.CanMoveAndAttack(View(game, entity));
        game.Flow.ActiveSquadron = entity; game.Flow.SquadronCommanded = commanded; game.Flow.SquadronProgress = progress;
        game.Flow.SquadronCanMove = !state.TemporaryAttackOnly && !IsPreventedFromMoving(game, entity);
        game.Flow.SquadronCanAttack = !state.TemporaryCannotAttack;
        game.Flow.Phase = GamePhase.SquadronActivation;
    }
    public static void BeginStandaloneActivation(GameContext game, int entity, GamePhase returnPhase, bool preserveActivation = false, float maximumMoveDistance = -1)
    {
        if (entity < game.State.Ships.Length || entity >= game.Topology.EntityCount || game.IsDestroyed(entity) || game.IsReserved(entity)
            || !preserveActivation && !CanActivate(game, entity, game.State.Owners[entity], false)) throw new ArgumentException("Squadron cannot activate.");
        var state = game.State.Squadrons[game.SquadronIndex(entity)];
        game.Flow.StandaloneSquadronActivations.Add(new()
        {
            ActiveShip = game.Flow.ActiveShip, ActiveSquadron = game.Flow.ActiveSquadron, CanMove = game.Flow.SquadronCanMove,
            CanAttack = game.Flow.SquadronCanAttack, Commanded = game.Flow.SquadronCommanded, Progress = game.Flow.SquadronProgress?.DeepClone(),
            ReturnPhase = returnPhase
        });
        game.Flow.ActiveShip = -1; game.Flow.ActiveSquadron = entity; game.Flow.SquadronCommanded = true;
        game.Flow.SquadronCanMove = !state.TemporaryAttackOnly; game.Flow.SquadronCanAttack = !state.TemporaryCannotAttack;
        game.Flow.SquadronProgress = new() { Entity = entity, CanMoveAndAttack = true, Standalone = true,
            PreserveActivation = preserveActivation, MaximumMoveDistance = maximumMoveDistance, ActivationCommitted = true };
        game.Flow.Phase = GamePhase.SquadronActivation;
    }
    public GameDecision GetDecision(GameContext game)
    {
        if (game.Flow.Phase == GamePhase.SquadronPhase)
        {
            if (game.Flow.Flags.GetValueOrDefault("round.squadron-phase-end") == game.State.Round) return Forced(game, "Resolve end of Squadron Phase");
            if (game.Flow.Flags.TryGetValue("round.squadron-phase-start", out var startStep) && startStep is 0 or 1 or 2) return Forced(game, "Resolve start of Squadron Phase");
            var choices = ReadySquadrons(game, game.Flow.SquadronPhasePlayer)
                .Select(entity => new GameActionOption(new(GameActionKind.ActivateSquadron, entity), "Activate " + DefinitionId(game, entity))).ToList();
            choices.AddRange(MovementAbilityEffects.GetSquadronPhaseActions(game));
            return choices.Count == 0 || game.Flow.SquadronPhaseRemaining <= 0 ? Forced(game, "Advance squadron batch")
                : new(DecisionKind.Player, game.Flow.SquadronPhasePlayer, GamePhase.SquadronPhase, choices);
        }
        var progress = game.Flow.SquadronProgress;
        if (progress is null) throw new InvalidOperationException("Squadron activation has no progress state.");
        if (progress.Stage is 0 or 2 or 3 or 4 or 5 or 6 || game.IsDestroyed(progress.Entity)) return Forced(game, "Resolve squadron timing");
        int entity = progress.Entity; var state = game.State.Squadrons[game.SquadronIndex(entity)];
        if (game.Flow.Phase == GamePhase.SquadronMove)
            return Player(game, [new(new(GameActionKind.MoveSquadron, entity), "Choose destination within movement ruler", AcceptsPosition: true),
                new(new(GameActionKind.Pass, entity, AbilityId: "squadron.cancel-move"), "Cancel movement")]);
        var options = new List<GameActionOption>();
        if (CanMoveNow(game, progress)) options.Add(new(new(GameActionKind.MoveSquadron, entity), "Move squadron", AcceptsPosition: true));
        if (CanAttackNow(game, progress)) options.Add(new(new(GameActionKind.Attack, entity), "Choose an attack"));
        options.AddRange(MovementAbilityEffects.GetActivationActions(game, entity));
        if (!progress.Moved && !progress.Attacked)
        {
            var commander = CommandingShip(game, entity);
            if (commander >= 0 && game.HasUpgrade(commander, "independence-title") && state.TemporarySpeedOverride < 0)
                options.Add(new(new(GameActionKind.UseAbility, entity, AbilityId: "remaining.squadron-activation.independence"), "Speed 4 and give up attack"));
            if (commander >= 0 && game.HasUpgrade(commander, "yavaris-title") && !state.TemporaryAttackOnly)
                options.Add(new(new(GameActionKind.UseAbility, entity, AbilityId: "remaining.squadron-activation.yavaris"), "Give up movement and gain Yavaris die"));
            if (!progress.ActivationCommitted) options.Add(new(new(GameActionKind.Pass, entity, AbilityId: "squadron.deselect"), "Choose another squadron"));
        }
        options.Add(new(new(GameActionKind.EndActivation, entity), progress.Moved || progress.Attacked ? "End squadron activation" : "Skip squadron activation"));
        return Player(game, options);
    }
    public void Apply(GameContext game, GameAction action)
    {
        if (game.Flow.Phase == GamePhase.SquadronPhase)
        {
            if (game.Flow.Flags.GetValueOrDefault("round.squadron-phase-end") == game.State.Round)
            {
                game.Flow.Flags.Remove("round.squadron-phase-end"); game.Flow.Phase = GamePhase.Status;
                EngineEffects.BeginWindow(game, "end-squadron-phase", -1, resumePhase: GamePhase.Status); return;
            }
            if (game.Flow.Flags.TryGetValue("round.squadron-phase-start", out var startStep) && startStep == 0)
            {
                game.Flow.Flags["round.squadron-phase-start"] = 1;
                EngineEffects.BeginWindow(game, "before-start-squadron-phase", -1, resumePhase: GamePhase.SquadronPhase); return;
            }
            if (game.Flow.Flags.GetValueOrDefault("round.squadron-phase-start") == 1)
            {
                game.Flow.Flags["round.squadron-phase-start"] = 2;
                EngineEffects.BeginWindow(game, "start-squadron-phase", -1, resumePhase: GamePhase.SquadronPhase); return;
            }
            if (game.Flow.Flags.GetValueOrDefault("round.squadron-phase-start") == 2)
            {
                game.Flow.Flags["round.squadron-phase-start"] = 3;
                EngineEffects.BeginWindow(game, "after-start-squadron-phase", -1, resumePhase: GamePhase.SquadronPhase); return;
            }
            if (game.Flow.SquadronPhaseRemaining <= 0 || !ReadySquadrons(game, game.Flow.SquadronPhasePlayer).Any())
            { AdvanceBatch(game); return; }
            if (action.Kind == GameActionKind.UseAbility)
            {
                if (!GetDecision(game).Actions.Any(o => GameSession.Matches(o, action)) || !MovementAbilityEffects.ApplySquadronPhaseAction(game, action))
                    throw new ArgumentException("Illegal squadron batch ability.");
                return;
            }
            if (action.Kind != GameActionKind.ActivateSquadron || !CanActivate(game, action.Source, game.Flow.SquadronPhasePlayer, false))
                throw new ArgumentException("Select an unactivated squadron owned by this player.", nameof(action));
            if (game.Flow.SquadronBatchKind != SquadronBatchKind.Normal && !game.Flow.SquadronBatchSelections.Contains(action.Source)) throw new ArgumentException("Squadron is not in the chosen batch.");
            BeginActivation(game, action.Source, false); return;
        }
        var progress = game.Flow.SquadronProgress ?? throw new InvalidOperationException("No active squadron progress.");
        int entity = progress.Entity;
        if (game.IsDestroyed(entity)) { FinishActivation(game); return; }
        if (progress.Stage == 0)
        {
            progress.Stage = 6;
            if (MissingSquadronLifecycleEffects.BeginActivationTiming(game, entity, GamePhase.SquadronActivation)) return;
        }
        if (progress.Stage == 6)
        {
            progress.Stage = 1;
            if (EngineEffects.BeginWindow(game, "squadron-activation", entity, resumePhase: GamePhase.SquadronActivation)) return;
        }
        if (progress.Stage == 2)
        {
            progress.Stage = 1;
            if (EngineEffects.BeginWindow(game, "after-squadron-move", entity, resumePhase: GamePhase.SquadronActivation)) return;
            return;
        }
        if (progress.Stage == 3) { FinishActivation(game); return; }
        if (progress.Stage == 5)
        {
            progress.Stage = 1;
            EngineEffects.BeginWindow(game, "after-squadron-attack", entity, resumePhase: GamePhase.SquadronActivation);
            return;
        }
        if (progress.Stage == 4)
        {
            progress.Stage = 1;
            if (progress.PendingMove is not { } pendingPosition || !IsLegalMove(game, entity, pendingPosition))
            { progress.PendingMove = null; game.Flow.Phase = GamePhase.SquadronActivation; return; }
            CommitMove(game, progress, pendingPosition); return;
        }
        if (action.Kind == GameActionKind.Pass && action.AbilityId.Length == 0) return;
        if (action.Kind == GameActionKind.Pass && action.AbilityId == "squadron.cancel-move")
        { game.Flow.Phase = GamePhase.SquadronActivation; return; }
        if (action.Kind == GameActionKind.Pass && action.AbilityId == "squadron.deselect")
        {
            if (progress.Moved || progress.Attacked || progress.ActivationCommitted) throw new ArgumentException("Cannot deselect a committed squadron.", nameof(action));
            if (progress.BeforeSelection is not null) game.State.Squadrons[game.SquadronIndex(entity)].RestoreSnapshot(progress.BeforeSelection);
            if (progress.Commanded) CommandModule.CancelSquadronSelection(game, entity);
            if (progress.WolffeGrantAdded) game.Flow.Flags.Remove($"wolffe.{game.Flow.ActiveShip}.{entity}");
            game.Flow.ActiveSquadron = -1; game.Flow.SquadronProgress = null;
            game.Flow.Phase = progress.Commanded ? GamePhase.SquadronCommand : GamePhase.SquadronPhase; return;
        }
        if (action.Kind == GameActionKind.UseAbility)
        {
            if (!GetDecision(game).Actions.Any(o => o.Template.AbilityId == action.AbilityId)) throw new ArgumentException("Illegal activation ability.", nameof(action));
            var state = game.State.Squadrons[game.SquadronIndex(entity)];
            if (action.AbilityId == "remaining.squadron-activation.independence")
            { state.SetTemporarySpeedOverride(4, cannotAttack: true); game.Flow.SquadronCanAttack = false; }
            else if (action.AbilityId == "remaining.squadron-activation.yavaris")
            { state.ChooseTemporaryAttackOnly(yavarisBonusDie: true); game.Flow.SquadronCanMove = false; }
            else if (!MovementAbilityEffects.ApplyActivationAction(game, action)) throw new ArgumentException("Unsupported activation ability.", nameof(action));
            return;
        }
        if (action.Kind == GameActionKind.MoveSquadron)
        {
            if (!CanMoveNow(game, progress)) throw new ArgumentException("Squadron cannot move now.", nameof(action));
            if (action.Position is not { } destination) { game.Flow.Phase = GamePhase.SquadronMove; return; }
            if (!IsLegalMove(game, entity, destination)) throw new ArgumentException("Squadron destination is not legal.", nameof(action));
            // Preserve the complete desired move while a before-move reaction window is open.
            progress.PendingMove = destination;
            if (EngineEffects.BeginWindow(game, "before-squadron-move", entity, resumePhase: GamePhase.SquadronMove))
            { progress.Stage = 4; return; }
            CommitMove(game, progress, destination); return;
        }
        if (action.Kind == GameActionKind.Attack)
        {
            if (!CanAttackNow(game, progress)) throw new ArgumentException("Squadron cannot attack now.", nameof(action));
            game.Flow.Phase = GamePhase.SelectAttack; return;
        }
        if (action.Kind == GameActionKind.EndActivation)
        {
            if (!progress.PreserveActivation) game.State.Squadrons[game.SquadronIndex(entity)].MarkActivated(progress.Commanded || progress.Standalone);
            progress.Stage = 3;
            if (EngineEffects.BeginWindow(game, "after-squadron-activation", entity, resumePhase: GamePhase.SquadronActivation)) return;
            FinishActivation(game); return;
        }
        throw new ArgumentException("Illegal squadron action.", nameof(action));
    }
    private static void CommitMove(GameContext game, SquadronActivationProgress progress, Vector2 destination)
    {
        if (progress.Commanded) CommandModule.CommitSquadronSelection(game, progress.Entity);
        game.State.Poses[progress.Entity] = new(destination, game.State.Poses[progress.Entity].YawDegrees);
        PlayAreaDestruction.DestroyIfOutside(game, progress.Entity);
        progress.Moved = true; progress.Stage = 2; game.Flow.SquadronCanMove = false;
        if (!progress.CanMoveAndAttack) game.Flow.SquadronCanAttack = false;
        progress.PendingMove = null; game.Flow.Phase = GamePhase.SquadronActivation;
        RoundModule.ObserveObjectives(game);
        RoundModule.CheckTerminal(game);
        OfficialCreatureObjectives.BeginSquadronMoveConsequences(game, progress.Entity, GamePhase.SquadronActivation);
    }
    public static void NotifyAttackCompleted(GameContext game)
    {
        if (game.Flow.SquadronProgress is not { } progress) return;
        if (progress.Commanded) CommandModule.CommitSquadronSelection(game, progress.Entity);
        progress.Attacked = true; game.Flow.SquadronCanAttack = false;
        if (!progress.CanMoveAndAttack) game.Flow.SquadronCanMove = false;
        progress.Stage = 5;
        game.Flow.Phase = GamePhase.SquadronActivation;
    }
    private static bool CanMoveNow(GameContext game, SquadronActivationProgress progress)
        => game.Flow.SquadronCanMove && !progress.Moved && (progress.CanMoveAndAttack || !progress.Attacked)
            && !game.State.Squadrons[game.SquadronIndex(progress.Entity)].TemporaryAttackOnly && !IsPreventedFromMoving(game, progress.Entity);
    private static bool CanAttackNow(GameContext game, SquadronActivationProgress progress)
        => game.Flow.SquadronCanAttack && !progress.Attacked && (progress.CanMoveAndAttack || !progress.Moved)
            && !game.State.Squadrons[game.SquadronIndex(progress.Entity)].TemporaryCannotAttack;
    public static bool IsPreventedFromMoving(GameContext game, int entity)
    {
        using var scope = new GameScope(game);
        return SquadronAbilityEngine.IsPreventedFromMoving(View(game, entity), ActiveViews(game));
    }
    public static int EffectiveSpeed(GameContext game, int entity)
    {
        using var scope = new GameScope(game);
        return SquadronAbilityEngine.GetEffectiveSpeed(View(game, entity), ActiveViews(game));
    }
    /// <summary>Prepare exact movement predicates for many destinations in one unchanged
    /// state. The query captures speed, range restrictions and surviving blockers; discard
    /// it after any game mutation. It does not choose, rank or quantize destinations.</summary>
    public static SquadronMoveLegalityQuery PrepareMoveLegality(GameContext game, int entity)
        => new(game, entity);
    public static bool IsLegalMove(GameContext game, int entity, Vector2 destination)
        => (game.Flow.SquadronProgress is { MaximumMoveDistance: >= 0 } progress && progress.Entity == entity
                ? Vector2.Distance(game.State.Poses[entity].Position, destination) <= progress.MaximumMoveDistance + 0.001f
                : SquadronGeometry.CanReach(game.SquadronBase(entity), destination, Math.Clamp(EffectiveSpeed(game, entity), 0, 5)))
            && IsLegalMoveDestination(game, entity, destination);
    public static bool IsLegalPlacement(GameContext game, int entity, Vector2 destination)
    {
        if (!float.IsFinite(destination.X) || !float.IsFinite(destination.Y)) return false;
        return PlayAreaGeometry.Contains(MovementModule.Board(game), new Circle(destination, game.SquadronBase(entity).Radius))
            && IsLegalMoveDestination(game, entity, destination);
    }
    /// <summary>A squadron may move out of the play area, where it is then destroyed.
    /// Placement by deployment, reserves or overlap still requires IsLegalPlacement.</summary>
    public static bool IsLegalMoveDestination(GameContext game, int entity, Vector2 destination)
    {
        if (!float.IsFinite(destination.X) || !float.IsFinite(destination.Y)) return false;
        var circle = new Circle(destination, game.SquadronBase(entity).Radius);
        for (int ship = 0; ship < game.State.Ships.Length; ship++)
            if (!game.Topology.Ships[ship].IsStation && !game.IsDestroyed(ship) && !game.IsReserved(ship) && Geometry2D.Overlaps(circle, game.ShipBase(ship))) return false;
        for (int index = 0; index < game.State.Squadrons.Length; index++)
        {
            int other = game.State.Ships.Length + index;
            if (other != entity && !game.IsDestroyed(other) && !game.IsReserved(other) && Geometry2D.Overlaps(circle, game.SquadronBase(other))) return false;
        }
        return true;
    }
    private static void FinishActivation(GameContext game)
    {
        var progress = game.Flow.SquadronProgress ?? throw new InvalidOperationException("No squadron activation to finish.");
        if (progress.Commanded) CommandModule.CommitSquadronSelection(game, progress.Entity);
        int entity = progress.Entity; var state = game.State.Squadrons[game.SquadronIndex(entity)];
        if (!progress.PreserveActivation) { state.MarkActivated(progress.Commanded || progress.Standalone); game.State.Activated[entity] = true; }
        // Source controller trigger, not a generic ability query: each nearby surviving hostile
        // Phlac-Arphocc deals one damage after the activation ends.
        foreach (var other in game.Topology.Squadrons)
        {
            if (state.Destroyed) break;
            if (other.Owner != game.State.Owners[entity] && !game.IsDestroyed(other.EntityIndex) && !game.IsReserved(other.EntityIndex)
                && other.CatalogEntry.Definition.SquadronId == "phlac-arphocc-prototypes-droid-tri-fighter-squadron"
                && Geometry2D.Distance(game.SquadronBase(entity), game.SquadronBase(other.EntityIndex)) <= TabletopMeasurements.DistanceOne) state.ApplyDamage(1);
        }
        if (!progress.PreserveActivation) state.ClearActivationModifiers();
        game.Flow.ActiveSquadron = -1; game.Flow.SquadronProgress = null;
        RoundModule.ObserveObjectives(game); RoundModule.CheckTerminal(game);
        if (game.State.Outcome is not null) return;
        if (progress.Standalone)
        {
            var parent = game.Flow.StandaloneSquadronActivations[^1];
            game.Flow.StandaloneSquadronActivations.RemoveAt(game.Flow.StandaloneSquadronActivations.Count - 1);
            game.Flow.ActiveShip = parent.ActiveShip; game.Flow.ActiveSquadron = parent.ActiveSquadron;
            game.Flow.SquadronCanMove = parent.CanMove; game.Flow.SquadronCanAttack = parent.CanAttack;
            game.Flow.SquadronCommanded = parent.Commanded; game.Flow.SquadronProgress = parent.Progress;
            game.Flow.Phase = parent.ReturnPhase; return;
        }
        if (progress.Commanded)
        {
            if (MovementAbilityEffects.FinishFennRauBonusActivation(game)) return;
            var activation = game.Flow.ShipActivations[game.Flow.ActiveShip];
            activation.CommandedSquadronsRemaining = Math.Max(0, activation.CommandedSquadronsRemaining - 1);
            game.Flow.Phase = activation.CommandedSquadronsRemaining > 0 ? GamePhase.SquadronCommand : GamePhase.SelectAttack;
            if (MovementAbilityEffects.BeginFennRauOffers(game, entity, game.Flow.Phase)) return;
            if (activation.CommandedSquadronsRemaining == 0) ActivationAbilityEffects.CompleteCommand(game);
        }
        else
        {
            game.Flow.SquadronPhaseRemaining = Math.Max(0, game.Flow.SquadronPhaseRemaining - 1);
            game.Flow.Phase = GamePhase.SquadronPhase;
            if (game.Flow.SquadronPhaseRemaining == 0 || !ReadySquadrons(game, game.Flow.SquadronPhasePlayer).Any()) AdvanceBatch(game);
        }
    }
    private static void AdvanceBatch(GameContext game)
    {
        game.Flow.SquadronBatchKind = SquadronBatchKind.Normal; game.Flow.SquadronBatchSource = -1; game.Flow.SquadronBatchSelections = [];
        int other = 3 - game.Flow.SquadronPhasePlayer;
        if (!ReadySquadrons(game, other).Any()) other = game.Flow.SquadronPhasePlayer;
        if (!ReadySquadrons(game, other).Any())
        {
            game.Flow.Flags["round.squadron-phase-end"] = game.State.Round; game.Flow.Phase = GamePhase.SquadronPhase;
            EngineEffects.BeginWindow(game, "before-end-squadron-phase", -1, resumePhase: GamePhase.SquadronPhase); return;
        }
        game.Flow.SquadronPhasePlayer = other; game.Flow.SquadronPhaseRemaining = 2; game.Flow.Phase = GamePhase.SquadronPhase;
    }
    private static IEnumerable<int> ReadySquadrons(GameContext game, int owner) => game.Topology.Squadrons
        .Where(s => CanActivate(game, s.EntityIndex, owner, false) && (game.Flow.SquadronBatchKind == SquadronBatchKind.Normal
            || owner != game.Flow.SquadronPhasePlayer || game.Flow.SquadronBatchSelections.Contains(s.EntityIndex))).Select(s => s.EntityIndex);
    public static bool TreatsAsCommandedForAi(GameContext game, int squadron) => game.Flow.SquadronBatchKind == SquadronBatchKind.HyperwaveSignalBoost
        && game.Flow.SquadronBatchSelections.Contains(squadron)
        && HasAiKeyword(game, squadron);
    private static bool HasAiKeyword(GameContext game, int squadron)
    {
        var definition = game.Topology.Squadrons[game.SquadronIndex(squadron)].CatalogEntry.Definition;
        return SquadronAbilityEngine.GetKeywordValue(definition, "AI ANTISQUADRON") > 0 || SquadronAbilityEngine.GetKeywordValue(definition, "AI BATTERY") > 0;
    }
    private static SquadronView View(GameContext game, int entity) => game.SquadronViews[game.SquadronIndex(entity)];
    private static SquadronView[] ActiveViews(GameContext game) => game.SquadronViews.Where(s => !game.IsReserved(s.Index)).ToArray();
    private static string DefinitionId(GameContext game, int entity) => game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition.DisplayName;
    private static int CommandingShip(GameContext game, int entity)
    {
        var id = game.State.Squadrons[game.SquadronIndex(entity)].CommandingShipRuntimeId;
        return id.Length > 0 && game.Topology.EntityByRuntimeId.TryGetValue(id, out var ship) && game.IsShip(ship) && !game.IsDestroyed(ship) ? ship : -1;
    }
    private static GameDecision Forced(GameContext game, string label) => new(DecisionKind.Forced, 0, game.Flow.Phase, [new(new(GameActionKind.Pass), label)]);
    private static GameDecision Player(GameContext game, IReadOnlyList<GameActionOption> options) => new(DecisionKind.Player,
        game.State.Owners[game.Flow.ActiveSquadron], game.Flow.Phase, options);
}
