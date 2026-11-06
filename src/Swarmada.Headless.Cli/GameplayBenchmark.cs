using System.Diagnostics;
using Swarmada.Headless.Engine;

internal static class GameplayBenchmark
{
    public static object Run(GameSession game, IReadOnlyList<GameAction> path, int iterations)
    {
        if (iterations is < 1 or > 1_000_000) throw new ArgumentOutOfRangeException(nameof(iterations));
        // Keep the caller's live handle unchanged, including any unsettled forced rules.
        var root = game.DeepClone(); root.GetDecision();
        var workspace = new BranchWorkspace(root);
        static long Evaluate(GameContext context, GameDecision decision) =>
            context.State.Round + (int)decision.Phase + decision.Actions.Count + context.State.Ships[0].CurrentHull;
        for (int i = 0; i < 100; i++) workspace.Visit(root, path, Evaluate);
        long checksum = 0;
        var replay = Measure(iterations, () => checksum += workspace.Visit(root, path, Evaluate));
        var copy = Measure(iterations, () => workspace.Position.CopyFrom(root));
        int cloneIterations = Math.Min(iterations, 10_000);
        var clone = Measure(cloneIterations, () => checksum += root.DeepClone().Context.State.Round);
        return new
        {
            scope = "In-process full-state copy, clone and validated path replay through gameplay decisions; excludes JSON, encoding, MCTS policy and neural inference.",
            ships = root.Context.State.Ships.Length, squadrons = root.Context.State.Squadrons.Length,
            upgrades = root.Context.State.Upgrades.Length, rootPhase = root.Context.Flow.Phase,
            pathLength = path.Count, replay, reusableCopy = copy, independentClone = clone, checksum
        };
    }

    private static object Measure(int iterations, Action operation)
    {
        // Caller owns one bounded scratch position; the operation does not retain a node history.
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long retainedBefore = GC.GetTotalMemory(false);
        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
        var stopwatch = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++) operation();
        stopwatch.Stop();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
        long retainedAfter = GC.GetTotalMemory(true);
        return new { iterations, seconds = stopwatch.Elapsed.TotalSeconds,
            operationsPerSecond = iterations / stopwatch.Elapsed.TotalSeconds,
            allocatedBytesPerOperation = allocated / (double)iterations,
            retainedBytesDeltaAfterCollection = retainedAfter - retainedBefore };
    }
}
