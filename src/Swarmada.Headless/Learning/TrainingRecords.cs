using Swarmada.Headless.Engine;
using Swarmada.Headless.Search;

namespace Swarmada.Headless.Learning;

public sealed record SparsePolicyVisits(int[] CandidateIndices, int[] Visits)
{
    public int TotalVisits => Visits.Sum();
    public long ByteSize => 4L * (CandidateIndices.Length + Visits.Length);
}

/// <summary>One conditional decision, with a policy target conditional on its
/// prefix. Prefixes can share the same owned State/Spatial objects. Hidden UCT
/// hypotheses are not records and are never neural policy targets.</summary>
public sealed record TrainingDecisionRecord(int Observer, string[] Prefix,
    EncodedSearchInput Input, SparsePolicyVisits Policy);

public sealed record TerminalTrainingTargets(int Winner, float[] RawPoints,
    float[] FinalShipHullFractions, float[] FinalSquadronHullFractions, int FinalRound)
{
    public float[] RawPointsFor(int observer)
    {
        if (observer is not (1 or 2)) throw new ArgumentOutOfRangeException(nameof(observer));
        return [RawPoints[observer - 1], RawPoints[2 - observer]];
    }
    public float ValueFor(int observer)
    {
        if (observer is not (1 or 2)) throw new ArgumentOutOfRangeException(nameof(observer));
        return Winner == 0 ? 0 : Winner == observer ? 1 : -1;
    }
    public static TerminalTrainingTargets FromTerminal(GameSession game)
    {
        var outcome = game.Context.State.Outcome ?? throw new InvalidOperationException("Auxiliary labels require a genuine terminal outcome.");
        return new(outcome.Winner, [(float)outcome.PlayerOneScore, (float)outcome.PlayerTwoScore],
            game.Context.State.Ships.Select(ship => ship.CurrentHull / (float)Math.Max(1, ship.MaximumHull)).ToArray(),
            game.Context.State.Squadrons.Select((squad, index) => squad.CurrentHull / (float)Math.Max(1, game.Context.Topology.Squadrons[index].CatalogEntry.Definition.Hull)).ToArray(),
            game.Context.State.Round);
    }
}

public static class TrainingRecordCapture
{
    /// <summary>Call on the settled actual root before applying the selected
    /// native action. Only the adapter's intent is buffered; no game action/RNG is
    /// applied. This is export outside the MCTS traversal hot path.</summary>
    public static TrainingDecisionRecord[] Capture(GameSession actual, IReadOnlyList<SearchKnowledge> knowledge,
        MctsResult result, ProductionSearchEncoder encoder, ActionCatalogProfile? profile = null)
    {
        var descriptors = result.SelectedDecisions ?? throw new ArgumentException("Search did not export conditional policy statistics.");
        var adapter = new DecisionAdapter(actual, profile); var records = new List<TrainingDecisionRecord>();
        foreach (var selected in descriptors)
        {
            if (selected.HiddenHypothesis) continue;
            adapter.RestorePrefix(selected.Prefix);
            var decision = adapter.Current with { Candidates = adapter.DescribeChoices() };
            if (decision.Actor != selected.Actor || !decision.Choices.Select(choice => choice.Id).SequenceEqual(selected.Actions.Select(action => action.Id)))
                throw new ArgumentException("Search policy target does not match this actual root and conditional prefix.");
            // Singleton choices can be reconstructed and do not train a policy.
            if (selected.Actions.Length < 2) continue;
            var observation = encoder.Encode(actual, knowledge[decision.Actor - 1], decision, adapter.Prefix);
            var nonzero = selected.Actions.Select((action, index) => (action, index)).Where(pair => pair.action.Visits > 0).ToArray();
            if (nonzero.Length == 0) throw new InvalidOperationException("A played policy decision has no search visits.");
            records.Add(new(decision.Actor, selected.Prefix.ToArray(), observation.Encoded!,
                new(nonzero.Select(pair => pair.index).ToArray(), nonzero.Select(pair => pair.action.Visits).ToArray())));
        }
        return records.ToArray();
    }
}
