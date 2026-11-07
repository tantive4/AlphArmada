using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Armada.Rules;
using Armada.Data;
using Swarmada.Headless.Catalog;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Learning;
using Swarmada.Headless.Runtime;
using Swarmada.Headless.Search;

namespace Swarmada.Headless.Worker;

public sealed record TensorMockFile(string Format, string ModelVersion, ulong Seed);
public sealed class GameMetrics
{
    public string GameId { get; set; } = "";
    public long Index { get; set; }
    public ulong Seed { get; set; }
    public string Status { get; set; } = "setup";
    public string Phase { get; set; } = "setup";
    public string? Error { get; set; }
    public int Round { get; set; }
    public int ActualNativeActions { get; set; }
    public int Searches { get; set; }
    public int ForcedNativeActions { get; set; }
    public int ChanceActions { get; set; }
    public int VisibleReveals { get; set; }
    public int SearchedPrefixEdges { get; set; }
    public int ForcedPrefixEdges { get; set; }
    public long Simulations { get; set; }
    public long InheritedVisits { get; set; }
    public long Nodes { get; set; }
    public long ActionSlots { get; set; }
    public long ModelRequests { get; set; }
    public long UctNodes { get; set; }
    public long TraversalNativeActions { get; set; }
    public long TraversalDecisionEdges { get; set; }
    public long CachedTraversals { get; set; }
    public int ReusedRoots { get; set; }
    public double SearchMilliseconds { get; set; }
    public double SetupMilliseconds { get; set; }
    public double ExportMilliseconds { get; set; }
    public double ElapsedMilliseconds { get; set; }
    public int TrainingRecords { get; set; }
    public long RawTensorBytes { get; set; }
    public long CompressedTensorBytes { get; set; }
    public long SparsePolicyBytes { get; set; }
    public TerminalOutcome? Outcome { get; set; }
    public bool? ExactReplay { get; set; }
    public string? FinalStateSha256 { get; set; }
    public PublishedShard? Shard { get; set; }
    public SortedDictionary<string, int> Phases { get; } = new(StringComparer.Ordinal);
    public SortedDictionary<int, int> RootCandidateCounts { get; } = [];
    public SortedDictionary<int, int> PlayedPrefixLengths { get; } = [];
    public SortedDictionary<string, int> PolicyHeads { get; } = new(StringComparer.Ordinal);
    public string[] RecentActions { get; set; } = [];
}

internal sealed class LiveGame
{
    public readonly GameMetrics Metrics;
    public LiveGame(GameMetrics metrics) => Metrics = metrics;
    public NativeMctsEnvironment? SearchEnvironment;
    public long SearchStarted;
    public object Snapshot()
    {
        var metrics = Metrics;
        var environment = Volatile.Read(ref SearchEnvironment);
        return new { metrics.GameId, metrics.Index, metrics.Status, metrics.Phase, metrics.Round, metrics.Searches,
            metrics.ActualNativeActions, metrics.Simulations, metrics.TrainingRecords,
            currentSearchMilliseconds = SearchStarted == 0 ? 0 : Stopwatch.GetElapsedTime(SearchStarted).TotalMilliseconds,
            currentTraversal = environment?.Timings };
    }
}

public sealed class WorkerRunner
{
    private readonly WorkerOptions options;
    private readonly WorkerLog log;
    private readonly WorkerLifetime lifetime;
    private readonly string identity, executionId = Guid.NewGuid().ToString("N")[..12];
    private readonly ConcurrentDictionary<string, LiveGame> active = new(StringComparer.Ordinal);
    private readonly ConcurrentBag<GameMetrics> results = [];
    private long nextGame = -1;
    private int consecutiveFailures;
    public WorkerRunner(WorkerOptions options, WorkerLog log, WorkerLifetime lifetime)
    {
        this.options = options; this.log = log; this.lifetime = lifetime;
        identity = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(options.RunId + "|" + options.WorkerId + "|" + options.CellId)))[..20];
    }
    public async Task<int> RunAsync()
    {
        var runTimer = Stopwatch.StartNew(); var starting = ProcessSample.Capture();
        Directory.CreateDirectory(options.Output); Directory.CreateDirectory(Path.Combine(options.Output, "locks"));
        using var workerLock = new FileStream(Path.Combine(options.Output, "locks", identity + ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        string staging = Path.Combine(options.Output, ".staging", identity);
        if (Directory.Exists(staging)) foreach (string abandoned in Directory.EnumerateDirectories(staging)) GameRecordSpool.CleanupOwnedDirectory(staging, abandoned);
        var budget = new OutputBudget(options.Output, options.MaxOutputBytes);
        UpgradeCatalog.LoadJson(Path.Combine(options.Data, "official-upgrades.json")); var data = FleetData.LoadDirectory(options.Data);
        var encoder = new ProductionSearchEncoder(data);
        var modelFile = options.Model is null ? new TensorMockFile("swarmada.tensor-random-model.v1", "tensor-random-seed-" + options.Seed, options.Seed) :
            JsonSerializer.Deserialize<TensorMockFile>(File.ReadAllText(options.Model), WorkerOptions.Json) ?? throw new ArgumentException("Model configuration is empty.");
        if (modelFile.Format != "swarmada.tensor-random-model.v1" || string.IsNullOrWhiteSpace(modelFile.ModelVersion))
            throw new ArgumentException("The worker currently supports only an explicitly versioned TensorRandom mock model file.");
        string code = GameRecordSpool.HashFile(typeof(GameSession).Assembly.Location) + ":" + GameRecordSpool.HashFile(typeof(WorkerRunner).Assembly.Location);
        var versions = new WorkerVersions(encoder.SchemaVersion, encoder.Schema.Fingerprint, code, modelFile.ModelVersion,
            EncoderSymbols.Fingerprint, ActionCatalogProfile.Version, "static-map-v1", options.SourceRevision);
        byte[] rawManifest = JsonSerializer.SerializeToUtf8Bytes(EncoderManifest.Describe(encoder.Schema), WorkerOptions.Json);
        using var compressedManifest = new MemoryStream();
        using (var gzip = new GZipStream(compressedManifest, CompressionLevel.SmallestSize, leaveOpen: true)) gzip.Write(rawManifest);
        byte[] manifestBytes = compressedManifest.ToArray();
        log.Write("started", new { executionId, options, versions, encoderManifestRawBytes = rawManifest.Length, encoderManifestGzipBytes = manifestBytes.Length,
            model = new { modelFile.Format, modelFile.ModelVersion, modelFile.Seed,
                fileBytes = options.Model is null ? 0 : new FileInfo(options.Model).Length,
                sha256 = options.Model is null ? null : GameRecordSpool.HashFile(options.Model), inference = "CPU tensor preparation and random outputs; no trained neural network or GPU inference" },
            runtime = new { framework = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
                os = System.Runtime.InteropServices.RuntimeInformation.OSDescription, processors = Environment.ProcessorCount }, starting });
        var factory = new ProductionGameSetup(options, data);
        if (options.ValidateOnly)
        {
            var prepared = factory.Create("validate-" + identity, options.Seed, lifetime.CancelGames);
            log.Write("validated", new { prepared.Game.Context.Topology.ObjectiveId, ships = prepared.Game.Context.State.Ships.Length,
                squadrons = prepared.Game.Context.State.Squadrons.Length, prepared.FleetPoints, prepared.OfficialFleetValidation,
                prepared.SetupActions, prepared.SetupAttempt, prepared.MapAudit, versions }); return 0;
        }
        var model = new TensorRandomSearchModel(modelFile.Seed);
        await using var inference = new InferenceBatcher(model, options.BatchSize, TimeSpan.FromMilliseconds(options.BatchDelayMs), modelFile.ModelVersion);
        var scheduler = new SearchWorkScheduler(options.CpuWorkers);
        var monitor = MonitorAsync(inference, scheduler, budget);
        try
        {
            await Task.WhenAll(Enumerable.Range(0, options.Concurrency).Select(_ => Task.Run(async () =>
            {
                while (!lifetime.StopNew.IsCancellationRequested)
                {
                    long index = Interlocked.Increment(ref nextGame);
                    if (options.Games > 0 && index >= options.Games) return;
                    ulong seed = SearchHash.Add(SearchHash.Add(options.Seed, options.WorkerId), (ulong)index);
                    string id = "g-" + identity + "-" + index.ToString("D8", System.Globalization.CultureInfo.InvariantCulture) + "-" + Guid.NewGuid().ToString("N")[..8];
                    var result = await PlayGame(data, factory, inference, scheduler, budget, id, index, seed, versions, manifestBytes);
                    results.Add(result);
                    if (result.Status == "terminal") Interlocked.Exchange(ref consecutiveFailures, 0);
                    else if (result.Status is not ("canceled" or "time-limit") && Interlocked.Increment(ref consecutiveFailures) >= options.MaximumConsecutiveFailures)
                        lifetime.RequestStop("consecutive-failure-limit", immediate: true);
                }
            }))).ConfigureAwait(false);
        }
        finally { lifetime.EndMonitoring(); await monitor.ConfigureAwait(false); }
        var ending = ProcessSample.Capture(); var completed = results.OrderBy(game => game.Index).ToArray();
        object summary = new
        {
            format = "swarmada.worker-run.v1", executionId, options.RunId, options.WorkerId, options.CellId, versions, configuration = options,
            stopReason = lifetime.StopReason, elapsedMilliseconds = runTimer.Elapsed.TotalMilliseconds,
            cpuSeconds = ending.CpuSeconds - starting.CpuSeconds, startingMemory = starting, endingMemory = ending,
            completedGames = completed.Count(game => game.Status == "terminal"), abortedGames = completed.Count(game => game.Status is "time-limit" or "canceled"),
            errorGames = completed.Count(game => game.Status is not ("terminal" or "time-limit" or "canceled")),
            searches = completed.Sum(game => game.Searches), actualNativeActions = completed.Sum(game => game.ActualNativeActions),
            searchedPrefixEdges = completed.Sum(game => game.SearchedPrefixEdges), forcedPrefixEdges = completed.Sum(game => game.ForcedPrefixEdges),
            simulations = completed.Sum(game => game.Simulations), uctNodes = completed.Sum(game => game.UctNodes),
            modelRequests = completed.Sum(game => game.ModelRequests), reusedRoots = completed.Sum(game => game.ReusedRoots),
            rawTensorBytes = completed.Where(game => game.Shard is not null).Sum(game => game.RawTensorBytes),
            compressedTensorBytes = completed.Where(game => game.Shard is not null).Sum(game => game.CompressedTensorBytes),
            targetsBytes = completed.Sum(game => game.Shard?.TargetsBytes ?? 0), uploadBytes = completed.Sum(game => game.Shard?.UploadBytes ?? 0),
            inference = inference.Statistics, model = new { model.Samples, model.InputTensorBytes, model.MaximumPaddedBatchBytes },
            cpuScheduling = new { scheduler.MaximumCpuWorkers, scheduler.MaximumObservedWorkers, scheduler.Entries },
            outputBudgetBytes = budget.UsedBytes, games = completed,
            scope = "Natural native games and complete discrete MCTS; real production encoding with untrained CPU TensorRandom policy/value. Initial map/deployment/setup choices precede MCTS. Limits abort without terminal training labels. Replay overhead is included when enabled."
        };
        Directory.CreateDirectory(Path.Combine(options.Output, "summaries"));
        string summaryPath = Path.Combine(options.Output, "summaries", options.RunId + "." + options.WorkerId + "." + options.CellId + "." + executionId + ".json");
        WriteAtomicJson(summaryPath, summary); log.Write("finished", new { summaryPath, summary }); log.FlushDurably();
        return completed.Any(game => game.Status is not ("terminal" or "time-limit" or "canceled")) ? 2 : 0;
    }

    private async Task<GameMetrics> PlayGame(FleetData data, ProductionGameSetup factory, InferenceBatcher inference, SearchWorkScheduler scheduler,
        OutputBudget budget, string id, long index, ulong seed, WorkerVersions versions, byte[] encoderManifest)
    {
        var metrics = new GameMetrics { GameId = id, Index = index, Seed = seed }; var live = new LiveGame(metrics); active[id] = live;
        var timer = Stopwatch.StartNew(); var trace = new Queue<string>();
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(lifetime.CancelGames);
        cancellation.CancelAfter(TimeSpan.FromSeconds(options.MaxGameSeconds)); var token = cancellation.Token;
        GameRecordSpool? spool = null;
        PreparedWorkerGame? preparedForDebug = null; GameDecision? rootDecisionForDebug = null;
        log.Write("game_started", new { gameId = id, index, seed, options.Preset, options.Simulations });
        try
        {
            PreparedWorkerGame prepared;
            using (await scheduler.EnterAsync(token)) prepared = factory.Create(id, seed, token);
            preparedForDebug = prepared;
            metrics.SetupMilliseconds = timer.Elapsed.TotalMilliseconds;
            var game = prepared.Game; var before = game.DeepClone(); var replay = options.Replay ? game.DeepClone() : null;
            SearchKnowledge[] knowledge = [SearchKnowledge.For(game, 1), SearchKnowledge.For(game, 2)];
            var exportEncoder = new ProductionSearchEncoder(data); MctsSearch? retained = null;
            spool = new(options, budget, identity, id);
            if (options.Replay) spool.WriteJsonPayload("initial-state.json", game.CaptureSnapshot());
            log.Write("game_setup", new { gameId = id, prepared.SetupActions, prepared.SetupAttempt, prepared.DeploymentPolicy, prepared.SetupChoicePolicy,
                objective = game.Context.Topology.ObjectiveId, ships = game.Context.State.Ships.Length, squadrons = game.Context.State.Squadrons.Length,
                upgrades = game.Context.State.Upgrades.Length, mapSelections = prepared.MapAudit.Selections.Count,
                prepared.FleetPoints, prepared.OfficialFleetValidation,
                milliseconds = metrics.SetupMilliseconds });
            for (int step = 0; step < options.MaxSteps; step++)
            {
                token.ThrowIfCancellationRequested(); GameAction? action = null; MctsSearch? pending = null; GameDecision decision;
                using (await scheduler.EnterAsync(token))
                {
                    decision = game.GetDecision(); metrics.Round = game.Context.State.Round; metrics.Phase = decision.Phase.ToString();
                    rootDecisionForDebug = decision;
                    metrics.Phases[metrics.Phase] = metrics.Phases.GetValueOrDefault(metrics.Phase) + 1;
                    if (decision.Kind == DecisionKind.InvalidSetup) throw new InvalidSetupException(decision.InvalidSetup!);
                    if (decision.Kind == DecisionKind.Terminal)
                    {
                        if (replay is not null)
                        {
                            replay.GetDecision(); string actualHash = SnapshotHash(game), replayHash = SnapshotHash(replay);
                            metrics.ExactReplay = actualHash == replayHash; metrics.FinalStateSha256 = actualHash;
                            if (!metrics.ExactReplay.Value) throw new InvalidDataException("The native action trace did not replay to the exact terminal state.");
                        }
                        metrics.Status = "terminal"; metrics.Outcome = decision.Outcome;
                        metrics.ElapsedMilliseconds = timer.Elapsed.TotalMilliseconds;
                        var report = new { format = "swarmada.worker-game.v1", executionId, options.RunId, options.WorkerId, options.CellId,
                            gameId = id, versions, configuration = options, metrics, match = prepared.Match,
                            encoderManifest = new { path = "encoder-manifest.json.gz", codec = "gzip" },
                            setup = new { prepared.DeploymentPolicy, prepared.SetupChoicePolicy, prepared.SetupActions, prepared.SetupAttempt, prepared.MapAudit,
                                prepared.FleetPoints, prepared.OfficialFleetValidation },
                            finalState = new { round = game.Context.State.Round, outcome = decision.Outcome },
                            terminalTargets = TerminalTrainingTargets.FromTerminal(game) };
                        metrics.Shard = spool.Publish(game, report, versions, encoderManifest);
                        log.Write("game_terminal", new { gameId = id, metrics }); return metrics;
                    }
                    if (decision.Kind == DecisionKind.Chance)
                    {
                        before.CopyFrom(game); action = game.SampleChance(); game.GetDecision();
                        foreach (var observer in knowledge) observer.ObserveTransition(before, action, game);
                        metrics.ChanceActions++;
                    }
                    else if (ForcedActionAdvancer.TryApply(game, knowledge, out action, out var path, beforeWorkspace: before, settledDecision: decision))
                    {
                        metrics.ForcedNativeActions++; metrics.ForcedPrefixEdges += path.Length;
                        if (action!.Kind == GameActionKind.RevealCommand) metrics.VisibleReveals++;
                    }
                    else
                    {
                        var environment = new NativeMctsEnvironment(game, knowledge[decision.Player - 1], SearchInformationMode.DeferredUctReveal,
                            encoder: new ProductionSearchEncoder(data), observerKnowledge: knowledge);
                        live.SearchEnvironment = environment;
                        if (!options.SubtreeReuse || retained is null)
                            retained = new(environment, inference, new(options.Simulations, MaximumNativeActions: options.MaximumNativeActions,
                                MaximumDecisionEdges: options.MaximumDecisionEdges), SearchHash.Add(seed, (ulong)step), scheduler);
                        else retained.Rebase(environment);
                        pending = retained;
                    }
                }
                if (pending is not null)
                {
                    metrics.Status = "search"; live.SearchStarted = Stopwatch.GetTimestamp();
                    var result = await pending.RunAsync(token).ConfigureAwait(false); live.SearchStarted = 0;
                    using (await scheduler.EnterAsync(token))
                    {
                        long exported = Stopwatch.GetTimestamp();
                        foreach (var record in TrainingRecordCapture.Capture(game, knowledge, result, exportEncoder))
                            spool.Append(record, metrics.Searches, versions.SchemaVersion);
                        metrics.ExportMilliseconds += Stopwatch.GetElapsedTime(exported).TotalMilliseconds;
                        action = NativeMctsEnvironment.ApplySelectedPath(game, result.SelectedPath, knowledge);
                        metrics.Searches++; metrics.Simulations += result.NewSimulations; metrics.InheritedVisits += result.InheritedVisits;
                        metrics.Nodes += result.Nodes; metrics.ActionSlots += result.ActionSlots; metrics.ModelRequests += result.ModelRequests; metrics.UctNodes += result.UctNodes;
                        metrics.TraversalNativeActions += result.NativeActions; metrics.TraversalDecisionEdges += result.DecisionEdges; metrics.CachedTraversals += result.CachedTraversals;
                        if (result.InheritedVisits > 0) metrics.ReusedRoots++;
                        metrics.SearchMilliseconds += result.ElapsedMilliseconds; metrics.SearchedPrefixEdges += result.SelectedPath.Length;
                        metrics.RootCandidateCounts[result.RootActions.Length] = metrics.RootCandidateCounts.GetValueOrDefault(result.RootActions.Length) + 1;
                        metrics.PlayedPrefixLengths[result.SelectedPath.Length] = metrics.PlayedPrefixLengths.GetValueOrDefault(result.SelectedPath.Length) + 1;
                        foreach (var selected in result.SelectedDecisions ?? [])
                            if (!selected.HiddenHypothesis && selected.Actions.Length > 1) metrics.PolicyHeads[selected.Head] = metrics.PolicyHeads.GetValueOrDefault(selected.Head) + 1;
                        metrics.TrainingRecords = spool.Records; metrics.RawTensorBytes = spool.RawTensorBytes;
                        metrics.CompressedTensorBytes = spool.CompressedTensorBytes; metrics.SparsePolicyBytes = spool.SparsePolicyBytes;
                        log.Write("search_completed", new { gameId = id, step, phase = metrics.Phase, round = metrics.Round,
                            result.Simulations, result.NewSimulations, result.InheritedVisits, result.Nodes, result.ActionSlots, result.ModelRequests,
                            result.UctNodes, result.NativeActions, result.DecisionEdges, result.CachedTraversals, result.ElapsedMilliseconds,
                            rootCandidates = result.RootActions.Length, prefixLength = result.SelectedPath.Length,
                            trainingRecords = spool.Records, rawTensorBytes = spool.RawTensorBytes, compressedTensorBytes = spool.CompressedTensorBytes,
                            process = ProcessSample.Capture(), inference = inference.Statistics });
                    }
                    live.SearchEnvironment = null;
                }
                using (await scheduler.EnterAsync(token))
                {
                    if (action is null) throw new InvalidOperationException("No actual native action was committed.");
                    metrics.ActualNativeActions++; metrics.Status = "playing"; spool.RecordAction(action);
                    trace.Enqueue(step + ":" + decision.Player + ":" + decision.Phase + ":" + action.Kind + ":" + action.AbilityId);
                    while (trace.Count > 16) trace.Dequeue(); metrics.RecentActions = trace.ToArray();
                    if (replay is not null) { replay.Apply(action); replay.GetDecision(); }
                }
            }
            throw new WorkerLimitException("step-limit", "Game action cap reached without a terminal outcome; training data discarded.");
        }
        catch (OperationCanceledException)
        { metrics.Status = lifetime.CancelGames.IsCancellationRequested ? "canceled" : "time-limit"; metrics.Error = "Stopped without terminal labels; partial records discarded."; }
        catch (WorkerLimitException error)
        { metrics.Status = error.Status; metrics.Error = error.Message; if (error.Status == "output-limit") lifetime.RequestStop(error.Status, immediate: true); }
        catch (OutOfMemoryException)
        { metrics.Status = "out-of-memory"; metrics.Error = "Managed allocation failed; no READY data published."; lifetime.RequestStop(metrics.Status, immediate: true); }
        catch (Exception error)
        {
            metrics.Status = "error"; metrics.Error = error.ToString()[..Math.Min(error.ToString().Length, 16000)];
            CaptureFailureContext(id, metrics, versions, preparedForDebug, rootDecisionForDebug, live.SearchEnvironment);
        }
        finally
        {
            live.SearchEnvironment = null; live.SearchStarted = 0; metrics.ElapsedMilliseconds = timer.Elapsed.TotalMilliseconds;
            try { spool?.Dispose(); } catch (Exception error) { log.Write("cleanup_error", new { gameId = id, error = error.Message }); }
            active.TryRemove(id, out _);
        }
        Directory.CreateDirectory(Path.Combine(options.Output, "failures"));
        WriteAtomicJson(Path.Combine(options.Output, "failures", id + ".json"), new { gameId = id, metrics, trainingDataPublished = false });
        log.Write("game_aborted", new { gameId = id, metrics, trainingDataPublished = false }); return metrics;
    }

    private async Task MonitorAsync(InferenceBatcher inference, SearchWorkScheduler scheduler, OutputBudget budget)
    {
        using var timer = new PeriodicTimer(TimeSpan.FromMilliseconds(500)); long lastLog = Stopwatch.GetTimestamp();
        try
        {
            while (await timer.WaitForNextTickAsync(lifetime.Monitoring).ConfigureAwait(false))
            {
                var sample = ProcessSample.Capture();
                if (options.MemoryLimitMb > 0 && sample.RssBytes > options.MemoryLimitMb * 1024 * 1024)
                    lifetime.RequestStop("memory-limit", immediate: true);
                if (options.StopFile is not null && File.Exists(options.StopFile)) lifetime.RequestStop("stop-file");
                if (Stopwatch.GetElapsedTime(lastLog).TotalSeconds < options.HeartbeatSeconds) continue;
                lastLog = Stopwatch.GetTimestamp();
                log.Write("heartbeat", new { process = sample, activeGames = active.Values.Select(game => game.Snapshot()).ToArray(),
                    completedGames = results.Count(game => game.Status == "terminal"), inference = inference.Statistics,
                    scheduler.ActiveWorkers, scheduler.MaximumObservedWorkers, outputBytes = budget.UsedBytes });
                log.FlushDurably();
            }
        }
        catch (OperationCanceledException) { }
    }
    private static string SnapshotHash(GameSession game)
        => Convert.ToHexStringLower(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(game.CaptureSnapshot(), WorkerOptions.Json)));
    private void CaptureFailureContext(string id, GameMetrics metrics, WorkerVersions versions, PreparedWorkerGame? prepared,
        GameDecision? rootDecision, NativeMctsEnvironment? environment)
    {
        string directory = Path.Combine(options.Output, "failures");
        string path = Path.Combine(directory, id + ".context.json"), temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            Directory.CreateDirectory(directory);
            var debug = new { format = "swarmada.worker-failure-context.v1", gameId = id, options.RunId, options.WorkerId,
                options.CellId, versions, metrics.Seed, metrics.Round, metrics.Phase,
                trainingDataPublished = false,
                scope = "Mechanical debugging only. The simulated snapshot omits search-wrapper materialized hypotheses and pending UCT state; it is not a complete MCTS recovery checkpoint.",
                match = prepared?.Match, rootDecision,
                actualRootSnapshot = prepared?.Game.CaptureSnapshot(),
                simulatedDebugSnapshot = environment?.Position.CaptureSnapshot(),
                searchCompatibilityToken = environment?.CompatibilityToken, rootObserver = environment?.RootObserver,
                searchTimings = environment?.Timings };
            using (var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
            {
                using var bounded = new BoundedDiagnosticStream(file, 2 * 1024 * 1024);
                JsonSerializer.Serialize(bounded, debug, WorkerOptions.Json); bounded.Flush(); file.Flush(true);
            }
            File.Move(temporary, path);
            log.Write("failure_context_saved", new { gameId = id, path, bytes = new FileInfo(path).Length,
                trainingDataPublished = false, completeSearchRecovery = false });
        }
        catch (Exception error)
        {
            try { if (File.Exists(temporary)) File.Delete(temporary); } catch { }
            log.Write("failure_context_error", new { gameId = id, error = error.Message, trainingDataPublished = false });
        }
    }
    private sealed class BoundedDiagnosticStream(Stream inner, long maximumBytes) : Stream
    {
        private long written;
        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => written;
        public override long Position { get => written; set => throw new NotSupportedException(); }
        public override void Flush() => inner.Flush();
        public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
        public override void Write(ReadOnlySpan<byte> buffer)
        {
            if (buffer.Length > maximumBytes - written) throw new InvalidDataException("Failure diagnostic exceeds its 2 MiB limit.");
            inner.Write(buffer); written += buffer.Length;
        }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
    }
    private static void WriteAtomicJson(string path, object value)
    {
        string temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
        { JsonSerializer.Serialize(stream, value, WorkerOptions.Json); stream.Flush(true); }
        File.Move(temporary, path, overwrite: true);
    }
}
