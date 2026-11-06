namespace Swarmada.Headless.Engine;

/// <summary>Reusable in-process branch evaluator. A search can keep its tree as actions and
/// statistics, reset this one position from a root and encode/evaluate each requested leaf.
/// It does not construct a game or export JSON for every node. Use one workspace per worker.</summary>
public sealed class BranchWorkspace
{
    public GameSession Position { get; }
    public BranchWorkspace(GameSession template) { ArgumentNullException.ThrowIfNull(template); Position = template.DeepClone(); }

    /// <summary>The callback borrows the position until it returns. Copy any observation data
    /// that must survive the next visit into caller-owned storage. Chance outcomes may appear
    /// explicitly in path; this API does not choose actions, outcomes or policy discretization.</summary>
    public T Visit<T>(GameSession root, IReadOnlyList<GameAction> path, Func<GameContext, GameDecision, T> evaluate)
    {
        ArgumentNullException.ThrowIfNull(root); ArgumentNullException.ThrowIfNull(path); ArgumentNullException.ThrowIfNull(evaluate);
        if (ReferenceEquals(root, Position)) throw new ArgumentException("The workspace position cannot also be its root.", nameof(root));
        Position.CopyFrom(root);
        for (int i = 0; i < path.Count; i++) Position.Apply(path[i]);
        return evaluate(Position.Context, Position.GetDecision());
    }
    public T[] VisitBatch<T>(GameSession root, IReadOnlyList<IReadOnlyList<GameAction>> paths, Func<GameContext, GameDecision, T> evaluate)
    {
        ArgumentNullException.ThrowIfNull(paths); var result = new T[paths.Count];
        for (int index = 0; index < paths.Count; index++) result[index] = Visit(root, paths[index], evaluate);
        return result;
    }
}
