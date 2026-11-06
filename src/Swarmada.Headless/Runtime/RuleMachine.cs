namespace Swarmada.Headless.Runtime;

/// <summary>Synchronous stepping infrastructure. A full game additionally requires all gameplay handlers.</summary>
public sealed class RuleMachine
{
    private readonly RuleRegistry registry;
    private readonly int forcedStepLimit;
    public BattleState State { get; }
    public RuleMachine(BattleState state, RuleRegistry registry, int forcedStepLimit = 10000)
    {
        ArgumentNullException.ThrowIfNull(state); ArgumentNullException.ThrowIfNull(registry);
        if (forcedStepLimit < 1) throw new ArgumentOutOfRangeException(nameof(forcedStepLimit));
        State = state; this.registry = registry; this.forcedStepLimit = forcedStepLimit;
    }
    public DecisionView GetDecision()
    {
        AdvanceForced();
        if (State.Outcome is not null)
            return new(DecisionKind.Terminal, 0, 0, State.Revision, Array.Empty<EngineAction>(), State.Outcome);
        var frame = State.CurrentFrame;
        var handler = registry.GetRequired(frame.RuleId);
        var kind = handler.GetKind(State, frame);
        if (kind is not (DecisionKind.Player or DecisionKind.Chance))
            throw new InvalidOperationException("Rule handler returned an invalid decision kind.");
        if (kind == DecisionKind.Player && frame.Owner is not (1 or 2))
            throw new InvalidOperationException("A player decision requires owner 1 or 2.");
        var actions = new List<EngineAction>(); handler.GetLegalActions(State, frame, actions);
        if (kind == DecisionKind.Player && actions.Count == 0)
            throw new InvalidOperationException("A player decision has no legal actions; this is a rules error.");
        if (actions.Distinct().Count() != actions.Count)
            throw new InvalidOperationException("A rule emitted duplicate numeric actions.");
        return new(kind, kind == DecisionKind.Player ? frame.Owner : 0, frame.RuleId, State.Revision,
            actions.AsReadOnly(), null);
    }
    public void Step(EngineAction action, int player, long expectedRevision)
    {
        var decision = GetDecision();
        if (decision.Kind == DecisionKind.Terminal) throw new InvalidOperationException("The match is terminal.");
        if (expectedRevision != State.Revision) throw new InvalidOperationException("Stale action revision.");
        if (player != decision.Player) throw new InvalidOperationException("Action submitted by the wrong decision owner.");
        var frame = State.CurrentFrame; var handler = registry.GetRequired(frame.RuleId);
        bool legal = decision.Kind == DecisionKind.Player ? decision.Actions.Contains(action) :
            handler.IsLegalChanceOutcome(State, frame, action);
        if (!legal) throw new ArgumentException("Action is not legal at the current decision.", nameof(action));
        handler.Apply(State, frame, action); State.Revision = checked(State.Revision + 1);
        AdvanceForced();
    }
    public EngineAction SampleChanceAction()
    {
        var decision = GetDecision();
        if (decision.Kind != DecisionKind.Chance) throw new InvalidOperationException("The state is not a chance node.");
        var frame = State.CurrentFrame; var handler = registry.GetRequired(frame.RuleId);
        // Do not commit the PRNG change if a defective sampler returns an illegal outcome.
        var random = State.Random; var action = handler.SampleChance(State, frame, ref random);
        if (!handler.IsLegalChanceOutcome(State, frame, action))
            throw new InvalidOperationException("Chance sampler returned an illegal outcome.");
        State.Random = random; return action;
    }
    public RuleMachine DeepClone() => new(State.DeepClone(), registry, forcedStepLimit);
    private void AdvanceForced()
    {
        for (int step = 0; State.Outcome is null; step++)
        {
            var frame = State.CurrentFrame; var handler = registry.GetRequired(frame.RuleId);
            if (handler.GetKind(State, frame) != DecisionKind.Forced) return;
            if (step >= forcedStepLimit)
                throw new InvalidOperationException("Forced-transition limit exceeded; possible rules loop.");
            handler.Apply(State, frame, default); State.Revision = checked(State.Revision + 1);
        }
    }
}
