using System.Diagnostics;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Search;

/// <summary>Lazy, single-owner PUCT tree with outcome variants. Parallelism is
/// across environments; no two threads mutate one tree/workspace. Argument nodes
/// share the simulation budget and do not advance the native-action horizon.</summary>
public sealed class MctsSearch
{
    private sealed class Node(SearchObservation observation, SearchTraversalKey key, object? descriptor)
    {
        public readonly SearchTraversalKey Key = key;
        public readonly object? Descriptor = descriptor;
        public readonly string[] Actions = observation.LegalActionIds;
        public readonly int Actor = observation.Actor;
        public readonly string Head = observation.Head;
        public readonly bool UseUct = observation.UseUct;
        public float[] Priors = [];
        public readonly int[] Visits = new int[observation.LegalActionIds.Length];
        public readonly double[] Values = new double[observation.LegalActionIds.Length];
        public readonly Dictionary<int, Dictionary<SearchTraversalKey, Node>> Children = [];
        public int TotalVisits;
        public bool Expanded;
        public float? PlayerOneEvaluation;
    }
    private readonly record struct PathEdge(Node Parent, int Index);
    private IMctsEnvironment environment;
    private readonly InferenceBatcher inference;
    private readonly MctsOptions options;
    private readonly ulong seed;
    private readonly SearchWorkScheduler? scheduler;
    private Dictionary<SearchTraversalKey, Node> roots = [];
    private Dictionary<SearchTraversalKey, Node>? selectedSuccessors;
    private ulong simulationSerial;
    private bool preserveRoots;
    private long nodeCount, slots, requests, decisionEdges, nativeActions, chanceSamples, limitEvaluations, uctNodes;
    private int maximumNativeDepth, maximumPrefixDepth, running;

    public MctsSearch(IMctsEnvironment environment, InferenceBatcher inference,
        MctsOptions? options = null, ulong seed = 1, SearchWorkScheduler? scheduler = null)
    {
        this.environment = environment ?? throw new ArgumentNullException(nameof(environment));
        this.inference = inference ?? throw new ArgumentNullException(nameof(inference));
        this.options = options ?? new(); this.seed = seed; this.scheduler = scheduler;
        if (this.options.Simulations < 1 || this.options.Exploration < 0 || !double.IsFinite(this.options.Exploration)
            || this.options.MaximumNativeActions < 1 || this.options.MaximumDecisionEdges < 1)
            throw new ArgumentOutOfRangeException(nameof(options));
    }

    /// <summary>Promote a matching descendant of the last played action. Reuse is
    /// observer/rules/catalog compatible and requires a complete settled-state key.
    /// This does not restore node snapshots: every simulation still rebuilds from
    /// the new actual root. Call only while this tree has no outstanding search.</summary>
    public bool Rebase(IMctsEnvironment nextEnvironment)
    {
        ArgumentNullException.ThrowIfNull(nextEnvironment);
        if (Volatile.Read(ref running) != 0) throw new InvalidOperationException("Cannot rebase a running tree.");
        var old = environment as ICachedMctsEnvironment; var next = nextEnvironment as ICachedMctsEnvironment;
        environment = nextEnvironment; preserveRoots = false;
        if (old is { SupportsCachedTraversal: true } && next is { SupportsCachedTraversal: true }
            && old.CompatibilityToken == next.CompatibilityToken && selectedSuccessors is not null)
        {
            next.Reset(SearchHash.Add(seed, 1UL)); var state = next.Inspect();
            if (state.Kind == DecisionKind.Player)
            {
                // Actual forced actions/chance may have advanced beyond the first
                // successor. Bound the lookup; a miss simply starts a fresh tree.
                var pending = new Queue<Node>(selectedSuccessors.Values); int checkedNodes = 0;
                while (pending.TryDequeue(out var candidate) && checkedNodes++ < 4096)
                {
                    if (candidate.Key == state.Key)
                    {
                        roots = new() { [state.Key] = candidate }; selectedSuccessors = null;
                        preserveRoots = true; return true;
                    }
                    foreach (var outcomes in candidate.Children.Values)
                        foreach (var child in outcomes.Values) pending.Enqueue(child);
                }
            }
        }
        roots.Clear(); selectedSuccessors = null; return false;
    }

    public async Task<MctsResult> RunAsync(CancellationToken cancellationToken = default)
    {
        if (Interlocked.Exchange(ref running, 1) != 0) throw new InvalidOperationException("One search tree has one owner.");
        IDisposable? cpuLease = null;
        try
        {
            if (scheduler is not null) cpuLease = await scheduler.EnterAsync(cancellationToken).ConfigureAwait(false);
            if (!preserveRoots) roots.Clear(); preserveRoots = false;
            nodeCount = slots = requests = decisionEdges = nativeActions = chanceSamples = limitEvaluations = uctNodes = 0;
            maximumNativeDepth = maximumPrefixDepth = 0;
            var timer = Stopwatch.StartNew();
            var cached = environment is ICachedMctsEnvironment { SupportsCachedTraversal: true } fast ? fast : null;
            environment.Reset(SearchHash.Add(seed, 1UL));
            int inherited = 0;
            if (cached is not null && roots.TryGetValue(cached.Inspect().Key, out var existingRoot)) inherited = existingRoot.TotalVisits;
            int newSimulations = Math.Max(0, options.Simulations - inherited); long cachedTraversals = 0;
            for (int simulation = 0; simulation < newSimulations; simulation++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ulong simulationSeed = SearchHash.Add(seed, ++simulationSerial);
                environment.Reset(simulationSeed);
                var chanceRandom = new DeterministicRandom(SearchHash.Add(simulationSeed, "chance"));
                var path = new List<PathEdge>(); var variants = roots;
                int completed = 0, prefixes = 0; float playerOneValue;
                while (true)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var state = cached?.Inspect();
                    var decision = state is null ? environment.Current : null;
                    DecisionKind kind = state?.Kind ?? decision!.Kind;
                    if (kind == DecisionKind.InvalidSetup) throw new InvalidOperationException("Invalid setup cannot become a search reward.");
                    if (kind == DecisionKind.Terminal)
                    {
                        var outcome = state?.Outcome ?? decision?.Outcome ?? throw new InvalidOperationException("Terminal decision has no outcome.");
                        playerOneValue = outcome.Winner == 1 ? 1 : outcome.Winner == 2 ? -1 : 0; break;
                    }
                    if (kind == DecisionKind.Chance)
                    { environment.SampleChance(ref chanceRandom); chanceSamples++; continue; }
                    SearchObservation? observation = null;
                    SearchTraversalKey key;
                    if (cached is null) { observation = environment.Encode(); key = new(observation.InformationKey, 0); }
                    else key = state!.Key;
                    if (!variants.TryGetValue(key, out var node))
                    {
                        decision ??= environment.Current;
                        if (kind != DecisionKind.Player || decision.Choices.Count == 0)
                            throw new InvalidOperationException("Settled player search decision has no legal choices.");
                        observation ??= environment.Encode();
                        if (observation.Actor != decision.Actor || decision.Actor is not (1 or 2))
                            throw new InvalidOperationException("Observation and decision actor disagree.");
                        node = new(observation, key, cached?.CaptureDecision()); variants.Add(key, node);
                        nodeCount++; slots += observation.LegalActionIds.Length;
                    }
                    else if (cached is not null)
                    { cached.BindDecision(node.Descriptor!); cachedTraversals++; }
                    else if (node.Actor != observation!.Actor || !node.Actions.SequenceEqual(observation.LegalActionIds))
                        throw new InvalidOperationException("Observation discriminator collision or inconsistent legal catalog.");
                    if (!node.Expanded)
                    {
                        SearchPrediction? prediction = null;
                        if (node.UseUct || node.Actions.Length == 1)
                        {
                            node.Priors = Enumerable.Repeat(1f / node.Actions.Length, node.Actions.Length).ToArray();
                            if (node.UseUct) uctNodes++;
                        }
                        else
                        {
                            observation ??= environment.Encode();
                            cpuLease?.Dispose(); cpuLease = null;
                            prediction = await inference.PredictAsync(observation, cancellationToken).ConfigureAwait(false); requests++;
                            if (scheduler is not null) cpuLease = await scheduler.EnterAsync(cancellationToken).ConfigureAwait(false);
                            node.Priors = Normalize(prediction.Priors);
                            node.PlayerOneEvaluation = prediction.ObserverValue * (node.Actor == 1 ? 1 : -1);
                        }
                        node.Expanded = true;
                        // A newly exposed argument prefix is not a completed simulation.
                        if (prediction is not null && completed > 0 && !environment.InBufferedPrefix)
                        { playerOneValue = node.PlayerOneEvaluation!.Value; break; }
                    }
                    if (completed >= options.MaximumNativeActions && !environment.InBufferedPrefix && !node.UseUct && node.Actions.Length > 1)
                    {
                        playerOneValue = node.PlayerOneEvaluation ?? throw new InvalidOperationException("Expanded model node has no cached value.");
                        limitEvaluations++; break;
                    }
                    if (path.Count >= options.MaximumDecisionEdges)
                        throw new InvalidOperationException("Decision-edge limit reached; check cyclic prefixes or increase the explicit bound.");
                    int selected = Select(node); path.Add(new(node, selected)); decisionEdges++;
                    bool committed = cached is null ? environment.Select(node.Actions[selected]) : cached.Select(selected);
                    if (committed) { completed++; nativeActions++; prefixes = 0; }
                    else { prefixes++; maximumPrefixDepth = Math.Max(maximumPrefixDepth, prefixes); }
                    maximumNativeDepth = Math.Max(maximumNativeDepth, completed);
                    if (!node.Children.TryGetValue(selected, out variants))
                    { variants = []; node.Children.Add(selected, variants); }
                }
                foreach (var edge in path)
                { edge.Parent.TotalVisits++; edge.Parent.Visits[edge.Index]++; edge.Parent.Values[edge.Index] += playerOneValue; }
            }
            environment.Reset(SearchHash.Add(seed, 1UL));
            var rootKey = cached is null ? new SearchTraversalKey(environment.Encode().InformationKey, 0) : cached.Inspect().Key;
            if (!roots.TryGetValue(rootKey, out var root))
                throw new InvalidOperationException("Search root changed its visible decision during recovery.");
            var selectedPath = new List<string>(); var selectedDecisions = new List<SearchDecisionStatistics>(); var choiceNode = root;
            while (true)
            {
                if (cached is not null) cached.BindDecision(choiceNode.Descriptor!);
                selectedDecisions.Add(new(choiceNode.Actor, choiceNode.Head, selectedPath.ToArray(),
                    Statistics(choiceNode), choiceNode.UseUct));
                int index = MostVisited(choiceNode); string id = choiceNode.Actions[index]; selectedPath.Add(id);
                if (cached is null ? environment.Select(id) : cached.Select(index))
                { choiceNode.Children.TryGetValue(index, out selectedSuccessors); break; }
                var nextKey = cached is null ? new SearchTraversalKey(environment.Encode().InformationKey, 0) : cached.Inspect().Key;
                if (!choiceNode.Children.TryGetValue(index, out var children) || !children.TryGetValue(nextKey, out var child))
                    throw new InvalidOperationException("Selected search prefix has no completed explored continuation.");
                choiceNode = child;
            }
            timer.Stop();
            return new(options.Simulations, selectedPath.ToArray(), root.Actions.Select((id, index) =>
                new SearchActionStatistics(id, root.Priors[index], root.Visits[index], root.Values[index])).ToArray(),
                nodeCount, slots, requests, decisionEdges, nativeActions, chanceSamples,
                maximumNativeDepth, maximumPrefixDepth, limitEvaluations, timer.Elapsed.TotalMilliseconds, uctNodes,
                inherited, newSimulations, cachedTraversals, selectedDecisions.ToArray());
        }
        finally { cpuLease?.Dispose(); Volatile.Write(ref running, 0); }
    }

    private static SearchActionStatistics[] Statistics(Node node) => node.Actions.Select((id, index) =>
        new SearchActionStatistics(id, node.Priors[index], node.Visits[index], node.Values[index])).ToArray();

    private int Select(Node node)
    {
        double best = double.NegativeInfinity; int selected = 0;
        double factor = options.Exploration * Math.Sqrt(node.TotalVisits + 1);
        for (int index = 0; index < node.Actions.Length; index++)
        {
            if (node.UseUct && node.Visits[index] == 0) return index;
            double q = node.Visits[index] == 0 ? 0 : node.Values[index] / node.Visits[index] * (node.Actor == 1 ? 1 : -1);
            double score = node.UseUct
                ? q + options.Exploration * Math.Sqrt(Math.Log(node.TotalVisits + 1.0) / node.Visits[index])
                : q + factor * node.Priors[index] / (1 + node.Visits[index]);
            if (score > best) { best = score; selected = index; }
        }
        return selected;
    }
    private static int MostVisited(Node node)
    {
        int selected = 0;
        for (int index = 1; index < node.Actions.Length; index++)
            if (node.Visits[index] > node.Visits[selected] || node.Visits[index] == node.Visits[selected]
                && node.Priors[index] > node.Priors[selected]) selected = index;
        return selected;
    }
    private static float[] Normalize(float[] priors)
    {
        var result = new float[priors.Length]; double total = priors.Sum(value => (double)value);
        for (int index = 0; index < result.Length; index++) result[index] = (float)(priors[index] / total);
        return result;
    }
}
