using Swarmada.Headless.Runtime;
using Swarmada.Headless.Setup;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

/// <summary>A fully specified action and its caller-owned, stable policy index.
/// Indices are never compressed after illegal candidates are masked.</summary>
public sealed record DiscreteActionCandidate(int Id, GameAction Action);

/// <summary>The supplied action, copied independently of the retained catalog.</summary>
public sealed record DiscreteActionStatus(int Id, GameAction Action, bool Legal, string? Reason = null);

public sealed record DiscreteDecision(DecisionKind Kind, int Player, GamePhase Phase,
    long Revision, long DecisionToken, IReadOnlyList<DiscreteActionStatus> Actions,
    TerminalOutcome? Outcome = null, InvalidSetupResult? InvalidSetup = null);

/// <summary>Validates a caller's finite action domain without generating, ranking or
/// substituting actions. One reusable scratch session isolates all trial mutations.
/// This object is a per-search-worker workspace and is not thread safe. The caller
/// must submit a new catalog after changing the live session; direct Context edits
/// between GetDecision and Apply are outside this API's mutation contract.</summary>
public sealed class DiscreteActionAdapter
{
    private static long nextDecisionToken;
    private GameSession? scratch;
    private DiscreteActionCandidate[] catalog = [];
    private bool[] legal = [];
    private long currentToken, sourceSession, sourceGeneration, sourceRevision;
    private string sourceSchema = string.Empty;
    private bool playerDecision;

    /// <summary>Return one result per candidate in its original order. Player legality
    /// is checked by executing each exact candidate in the scratch position. Chance,
    /// terminal and invalid-setup decisions have empty masks and are never optimized
    /// or sampled here. An empty player mask remains empty; it does not invent a pass.</summary>
    public DiscreteDecision GetDecision(GameSession game, IReadOnlyList<DiscreteActionCandidate> candidates)
    {
        ArgumentNullException.ThrowIfNull(game);
        ArgumentNullException.ThrowIfNull(candidates);
        // Fail malformed catalogs before settling the live position or invalidating
        // a previously valid selection. Copy mutable payloads at the input boundary.
        var nextCatalog = new DiscreteActionCandidate[candidates.Count];
        var ids = new HashSet<int>();
        for (int i = 0; i < candidates.Count; i++)
        {
            var candidate = candidates[i] ?? throw new ArgumentException("Candidates cannot contain null entries.", nameof(candidates));
            if (candidate.Id < 0 || !ids.Add(candidate.Id))
                throw new ArgumentException("Discrete action IDs must be nonnegative and unique.", nameof(candidates));
            ArgumentNullException.ThrowIfNull(candidate.Action);
            nextCatalog[i] = candidate with { Action = candidate.Action.DeepClone() };
        }

        var decision = game.GetDecision();
        currentToken = Interlocked.Increment(ref nextDecisionToken);
        sourceSession = game.InstanceIdentity;
        sourceGeneration = game.PositionGeneration;
        sourceRevision = game.Context.State.Revision;
        sourceSchema = game.Context.State.SchemaId;
        playerDecision = decision.Kind == DecisionKind.Player;
        catalog = playerDecision ? nextCatalog : [];
        legal = new bool[catalog.Length];
        var statuses = new DiscreteActionStatus[catalog.Length];
        if (playerDecision)
        {
            if (scratch is null || scratch.Context.State.SchemaId != sourceSchema)
                scratch = game.DeepClone();
            for (int i = 0; i < catalog.Length; i++)
            {
                var candidate = catalog[i];
                string? reason = CandidateRejection(game.Context, decision, candidate.Action);
                if (reason is null)
                {
                    scratch.CopyFrom(game);
                    try
                    {
                        // Apply exactly the candidate. Do not settle its resulting
                        // forced continuation: legality concerns this action alone.
                        scratch.Apply(candidate.Action.DeepClone());
                        legal[i] = true;
                    }
                    catch (ArgumentException error) { reason = error.Message; }
                    catch (InvalidSetupException error)
                    {
                        // Proven setup impossibility is a restart signal, not a
                        // candidate that MCTS should simply mask and keep playing.
                        // The proof ran only in scratch: the live parent is pure.
                        playerDecision = false; catalog = []; legal = [];
                        return new(DecisionKind.InvalidSetup, error.Result.Player, decision.Phase,
                            sourceRevision, currentToken, Array.Empty<DiscreteActionStatus>(), InvalidSetup: error.Result);
                    }
                }
                statuses[i] = new(candidate.Id, candidate.Action.DeepClone(), legal[i], reason);
            }
        }
        return new(decision.Kind, decision.Player, decision.Phase, sourceRevision, currentToken,
            Array.AsReadOnly(statuses), decision.Outcome, decision.InvalidSetup);
    }

    /// <summary>Apply precisely the selected legal ID from the most recent catalog.
    /// Neither a stale selection nor an illegal/unknown ID changes the live game.</summary>
    public void Apply(GameSession game, int id, long decisionToken)
    {
        ArgumentNullException.ThrowIfNull(game);
        if (!playerDecision || decisionToken != currentToken || game.InstanceIdentity != sourceSession ||
            game.PositionGeneration != sourceGeneration || game.Context.State.Revision != sourceRevision ||
            game.Context.State.SchemaId != sourceSchema)
            throw new InvalidOperationException("The discrete decision is stale or is not a player decision; request a new candidate mask.");
        int selected = -1;
        for (int i = 0; i < catalog.Length; i++) if (catalog[i].Id == id) { selected = i; break; }
        if (selected < 0) throw new ArgumentException("The selected ID is not in the supplied discrete catalog.", nameof(id));
        if (!legal[selected]) throw new ArgumentException("The selected discrete candidate is illegal.", nameof(id));
        // Invalidate before applying, including any exceptional partial rule failure.
        // DeepClone prevents modules retaining caller-visible mutable array payloads.
        playerDecision = false;
        game.Apply(catalog[selected].Action.DeepClone());
    }

    /// <summary>Discard a selection before directly mutating Context outside the
    /// session API. The reusable scratch storage is retained for the next mask.</summary>
    public void Invalidate()
    {
        playerDecision = false; currentToken = 0;
        catalog = []; legal = [];
    }

    private static string? CandidateRejection(GameContext game, GameDecision decision, GameAction action)
    {
        if (!float.IsFinite(action.YawDegrees) || action.Position is { } position &&
            (!float.IsFinite(position.X) || !float.IsFinite(position.Y)))
            return "Gameplay positions and angles must be finite.";
        foreach (var option in decision.Actions)
        {
            if (!GameSession.Matches(option, action)) continue;
            // Squadron activation exposes a genuine phase-control action: choose
            // movement now, then choose its destination at SquadronMove. It is a
            // complete discrete action even without a spatial payload. The same
            // omission at SquadronMove would only re-enter itself and is rejected.
            bool chooseSquadronMove = decision.Phase == GamePhase.SquadronActivation && action.Kind == GameActionKind.MoveSquadron;
            if (option.AcceptsPosition && action.Position is null && !chooseSquadronMove)
                return "A discrete spatial candidate must specify its position.";
            if (option.AcceptsCourse && action.Course is null)
                return "A discrete maneuver candidate must specify its complete course.";
            // A compact course offer accepts arbitrary click arrays. Reject invalid
            // canonical moving courses with the same pure evaluator used by Apply,
            // avoiding a copied trial position and an exception for each bad grid
            // entry. Concrete standstill/forced-course offers keep their own matching
            // rules and are deliberately outside this optimization.
            if (option.AcceptsCourse && decision.Phase == GamePhase.SelectCourse &&
                action.Kind == GameActionKind.SetCourse && action.AbilityId == "movement.course" &&
                game.Flow.Movement is null && action.Source == game.Flow.ActiveShip &&
                game.IsShip(action.Source) && game.State.Ships[action.Source].CurrentSpeed > 0 &&
                (action.YawDegrees != 0 || !MovementModule.TryEvaluateLegalCourse(game, action.Source,
                    action.Course!, (GuideSide)action.Index, out _)))
                return "Illegal ship maneuver.";
            return null;
        }
        return "The action is not one of the current legal choices.";
    }
}
