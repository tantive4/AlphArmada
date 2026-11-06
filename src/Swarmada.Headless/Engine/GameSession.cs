using Swarmada.Headless.Runtime;
using Swarmada.Headless.Setup;

namespace Swarmada.Headless.Engine;

/// <summary>Synchronous game driver. All continuations and chance state belong to Context;
/// modules contain no per-match mutable state. GetDecision settles forced rules but leaves
/// player and chance decisions visible to a search caller.</summary>
public sealed class GameSession
{
    private static long nextInstanceIdentity;
    internal long InstanceIdentity { get; } = Interlocked.Increment(ref nextInstanceIdentity);
    internal long PositionGeneration { get; private set; }
    private readonly IGameplayModule[] modules;
    private static readonly IGameplayModule[] DefaultModules = [new SetupModule(), new CommandModule(), new RoundModule(),
        new CombatModule(), new MovementModule(compactDecisions: true), new SquadronModule(), new AbilityModule()];
    public GameContext Context { get; }
    public InvalidSetupResult? InvalidSetup => OfficialObjectives.State(Context)?.InvalidSetup;
    public GameSession(PreparedMatch prepared) : this(new GameContext(prepared)) { }
    public GameSession(GameContext context)
    {
        Context = context;
        modules = DefaultModules;
    }
    public GameSession DeepClone() => new(Context.DeepClone());
    public void CopyFrom(GameSession source)
    { ArgumentNullException.ThrowIfNull(source); PositionGeneration++; Context.CopyFrom(source.Context); }
    public GameSnapshot CaptureSnapshot() => Context.CaptureSnapshot();
    public void RestoreSnapshot(GameSnapshot snapshot)
    { PositionGeneration++; Context.RestoreSnapshot(snapshot); }
    private IGameplayModule Module()
    {
        for (int i = 0; i < modules.Length; i++) if (modules[i].Handles(Context.Flow.Phase)) return modules[i];
        throw new NotSupportedException("No gameplay module for " + Context.Flow.Phase);
    }
    public GameDecision GetDecision()
    {
        using var scope = new GameScope(Context);
        for (int step = 0; step < 10000; step++)
        {
            if (InvalidSetup is { } invalid)
                return new(DecisionKind.InvalidSetup, invalid.Player, Context.Flow.Phase, [], InvalidSetup: invalid);
            if (Context.State.Outcome is { } outcome)
                return new(DecisionKind.Terminal, 0, GamePhase.Terminal, [], outcome);
            DamageEvents.Observe(Context);
            OfficialObjectives.Observe(Context);
            if (DamageEvents.TryBeginNext(Context)) { PositionGeneration++; continue; }
            if (TokenAssignments.TryBeginNext(Context)) { PositionGeneration++; continue; }
            MissingSquadronLifecycleEffects.ObserveObjectiveTokens(Context);
            if (MissingSquadronLifecycleEffects.TryBeginNext(Context)) { PositionGeneration++; continue; }
            var phase = Context.Flow.Phase;
            var module = Module();
            var decision = module.GetDecision(Context);
            if (Context.Flow.Phase != phase) { PositionGeneration++; continue; }
            if (decision.Kind != DecisionKind.Forced)
                return decision.Actions.Any(option => option.AcceptsCourse) ? decision with
                { CourseConstraints = MovementModule.GetCourseConstraints(Context, Context.Flow.ActiveShip) } : decision;
            if (decision.Actions.Count != 1) throw new InvalidOperationException("Forced rules require exactly one action.");
            PositionGeneration++;
            if (module is IVerifiedDecisionModule verified) verified.ApplyVerified(Context, decision.Actions[0].Template);
            else module.Apply(Context, decision.Actions[0].Template);
            Context.State.Revision++;
        }
        throw new InvalidOperationException("Forced gameplay failed to reach a decision after 10000 transitions.");
    }
    public void Apply(GameAction action)
    {
        ArgumentNullException.ThrowIfNull(action);
        PositionGeneration++;
        if (!float.IsFinite(action.YawDegrees) || action.Position is { } position && (!float.IsFinite(position.X) || !float.IsFinite(position.Y)))
            throw new ArgumentException("Gameplay positions and angles must be finite.", nameof(action));
        using var scope = new GameScope(Context);
        var decision = GetDecision();
        if (decision.InvalidSetup is { } invalid) throw new InvalidSetupException(invalid);
        if (decision.Kind == DecisionKind.Terminal) throw new InvalidOperationException("The match is complete.");
        if (decision.Kind != DecisionKind.Chance)
        {
            bool legal = false;
            for (int i = 0; i < decision.Actions.Count; i++)
                if (Matches(decision.Actions[i], action)) { legal = true; break; }
            if (!legal) throw new ArgumentException("The action is not one of the current legal choices.", nameof(action));
        }
        var module = Module();
        // Explicit MCTS chance outcomes consume the same draw schedule as sampled ones.
        // Hold the updated RNG locally so an invalid supplied outcome leaves it unchanged.
        var random = Context.State.Random;
        if (decision.Kind == DecisionKind.Chance && module is IChanceGameplayModule chance)
            chance.AdvanceChanceRandom(Context, ref random);
        if (module is IVerifiedDecisionModule verified) verified.ApplyVerified(Context, action);
        else module.Apply(Context, action);
        if (decision.Kind == DecisionKind.Chance) Context.State.Random = random;
        Context.State.Revision++;
    }
    public GameAction SampleChance()
    {
        PositionGeneration++;
        using var scope = new GameScope(Context);
        var decision = GetDecision();
        if (decision.Kind != DecisionKind.Chance || Module() is not IChanceGameplayModule chance)
            throw new InvalidOperationException("The current decision is not a chance node.");
        var action = chance.SampleChance(Context, ref Context.State.Random);
        chance.Apply(Context, action); Context.State.Revision++; return action;
    }
    /// <summary>Sample with a caller-owned search stream, so repeated MCTS visits to
    /// an identical cloned position need not receive the same seeded outcome.
    /// The position still consumes its normal RNG draw schedule for exact path replay.</summary>
    public GameAction SampleChance(ref DeterministicRandom searchRandom)
    {
        PositionGeneration++;
        using var scope = new GameScope(Context);
        var decision = GetDecision();
        if (decision.Kind != DecisionKind.Chance || Module() is not IChanceGameplayModule chance)
            throw new InvalidOperationException("The current decision is not a chance node.");
        var action = chance.SampleChance(Context, ref searchRandom);
        var positionRandom = Context.State.Random; chance.AdvanceChanceRandom(Context, ref positionRandom);
        chance.Apply(Context, action); Context.State.Random = positionRandom; Context.State.Revision++; return action;
    }
    internal static bool Matches(GameActionOption option, GameAction action)
    {
        var template = option.Template;
        if (ReferenceEquals(template, action)) return true;
        return template.Kind == action.Kind && template.Source == action.Source && template.Target == action.Target
            && template.FromZone == action.FromZone && template.ToZone == action.ToZone
            && template.Command == action.Command && template.SpendMode == action.SpendMode && template.Token == action.Token
            && (option.AcceptsCourse || template.Index == action.Index) && template.Amount == action.Amount
            && (option.AcceptsPosition || template.Position == action.Position)
            && (option.AcceptsYaw || option.AcceptsCourse || template.YawDegrees == action.YawDegrees)
            && (option.AcceptsFlip || template.Flipped == action.Flipped)
            && (option.AcceptsPosition && action.Kind is GameActionKind.Deploy or GameActionKind.PlaceReservedUnit || template.Speed == action.Speed)
            && template.AbilityId == action.AbilityId
            && (option.AcceptsCourse || Equal(template.Course, action.Course))
            && Equal(template.DiceIndices, action.DiceIndices) && Equal(template.Dice, action.Dice);
    }
    private static bool Equal<T>(T[]? first, T[]? second) => first is null ? second is null : second is not null && first.SequenceEqual(second);
}
