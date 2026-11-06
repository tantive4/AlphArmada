using System.Diagnostics;
using System.Threading.Channels;

namespace Swarmada.Headless.Search;

/// <summary>Random, untrained outputs derived only from the supplied observation.
/// Request/batch scheduling cannot change an observation's mock prediction.</summary>
public sealed class RandomSearchModel(ulong seed = 1) : IBatchedSearchModel
{
    public ValueTask<IReadOnlyList<SearchPrediction>> PredictBatchAsync(
        IReadOnlyList<SearchObservation> observations, CancellationToken cancellationToken = default)
    {
        var outputs = new SearchPrediction[observations.Count];
        for (int sample = 0; sample < observations.Count; sample++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var observation = observations[sample];
            if (observation.LegalActionIds.Length == 0) throw new ArgumentException("A policy request needs legal actions.");
            ulong hash = SearchHash.Add(SearchHash.Add(observation.InformationKey, seed), observation.Head);
            var priors = new float[observation.LegalActionIds.Length]; double total = 0;
            for (int action = 0; action < priors.Length; action++)
            {
                priors[action] = (float)(.01 + SearchHash.Unit(SearchHash.Add(hash, observation.LegalActionIds[action])));
                total += priors[action];
            }
            for (int action = 0; action < priors.Length; action++) priors[action] /= (float)total;
            outputs[sample] = new(priors, (float)(2 * SearchHash.Unit(SearchHash.Add(hash, "value")) - 1));
        }
        return ValueTask.FromResult<IReadOnlyList<SearchPrediction>>(outputs);
    }
}

public sealed record InferenceBatchStatistics(long Requests, long Batches, int MaximumBatch,
    double AverageBatch, double ModelMilliseconds, double AverageQueueMilliseconds,
    IReadOnlyDictionary<int, long>? BatchSizes = null, long CancelledRequests = 0, int MaximumHeadFamilies = 0);

/// <summary>Many independent trees share this inference queue. Each model call
/// can contain different head families: a real model batches its shared board
/// trunk first and dispatches small conditional heads afterward. Bounded ingress,
/// timeout batching, owned buffers, and exception/cancellation propagation are explicit.</summary>
public sealed class InferenceBatcher : IAsyncDisposable
{
    private sealed record Request(SearchObservation Observation, long Enqueued,
        TaskCompletionSource<SearchPrediction> Completion, CancellationToken Cancellation);
    private readonly Channel<Request> queue;
    private readonly IBatchedSearchModel model;
    private readonly int maximumBatch;
    private readonly TimeSpan maximumDelay;
    private readonly Task pump;
    private long requests, batches, modelTicks, queueTicks, cancelledRequests;
    private readonly long[] batchSizes;
    private int observedMaximum, maximumHeadFamilies;
    private int disposed;
    /// <summary>Immutable prediction-cache compatibility identifier. Use a new
    /// batcher or version whenever weights or feature/policy schemas change.</summary>
    public string ModelVersion { get; }

    public InferenceBatcher(IBatchedSearchModel model, int maximumBatch = 32, TimeSpan? maximumDelay = null,
        string modelVersion = "unversioned")
    {
        ArgumentNullException.ThrowIfNull(model);
        if (maximumBatch < 1) throw new ArgumentOutOfRangeException(nameof(maximumBatch));
        if (string.IsNullOrWhiteSpace(modelVersion)) throw new ArgumentException("A model version must be nonempty.", nameof(modelVersion));
        this.model = model; this.maximumBatch = maximumBatch;
        ModelVersion = modelVersion; batchSizes = new long[maximumBatch + 1];
        this.maximumDelay = maximumDelay ?? TimeSpan.FromMilliseconds(1);
        if (this.maximumDelay < TimeSpan.Zero || this.maximumDelay > TimeSpan.FromSeconds(1))
            throw new ArgumentOutOfRangeException(nameof(maximumDelay));
        queue = Channel.CreateBounded<Request>(new BoundedChannelOptions(maximumBatch * 16)
        { SingleReader = true, SingleWriter = false, FullMode = BoundedChannelFullMode.Wait });
        pump = Task.Run(PumpAsync);
    }

    public async ValueTask<SearchPrediction> PredictAsync(SearchObservation observation,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref disposed) != 0, this);
        var completion = new TaskCompletionSource<SearchPrediction>(TaskCreationOptions.RunContinuationsAsynchronously);
        var request = new Request(observation, Stopwatch.GetTimestamp(), completion, cancellationToken);
        await queue.Writer.WriteAsync(request, cancellationToken).ConfigureAwait(false);
        return await completion.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
    }

    public InferenceBatchStatistics Statistics
    {
        get
        {
            long count = Interlocked.Read(ref requests), calls = Interlocked.Read(ref batches);
            return new(count, calls, Volatile.Read(ref observedMaximum), calls == 0 ? 0 : (double)count / calls,
                Interlocked.Read(ref modelTicks) * 1000.0 / Stopwatch.Frequency,
                count == 0 ? 0 : Interlocked.Read(ref queueTicks) * 1000.0 / Stopwatch.Frequency / count,
                Enumerable.Range(1, maximumBatch).Select(size => (size, count: Interlocked.Read(ref batchSizes[size])))
                    .Where(sample => sample.count != 0).ToDictionary(sample => sample.size, sample => sample.count),
                Interlocked.Read(ref cancelledRequests), Volatile.Read(ref maximumHeadFamilies));
        }
    }

    private async Task PumpAsync()
    {
        var pending = new List<Request>(maximumBatch);
        try
        {
            while (await queue.Reader.WaitToReadAsync().ConfigureAwait(false))
            {
                pending.Clear();
                if (!queue.Reader.TryRead(out var first)) continue;
                pending.Add(first);
                long deadline = Stopwatch.GetTimestamp() + (long)(maximumDelay.TotalSeconds * Stopwatch.Frequency);
                while (pending.Count < maximumBatch)
                {
                    while (pending.Count < maximumBatch && queue.Reader.TryRead(out var next)) pending.Add(next);
                    if (pending.Count == maximumBatch || maximumDelay == TimeSpan.Zero || Stopwatch.GetTimestamp() >= deadline) break;
                    TimeSpan remaining = TimeSpan.FromSeconds((deadline - Stopwatch.GetTimestamp()) / (double)Stopwatch.Frequency);
                    if (remaining <= TimeSpan.Zero) break;
                    using var timeout = new CancellationTokenSource(remaining);
                    try { if (!await queue.Reader.WaitToReadAsync(timeout.Token).ConfigureAwait(false)) break; }
                    catch (OperationCanceledException) { break; }
                }
                // A canceled environment must not leave a permanent batch hole.
                // Remove canceled requests before model execution; each completion
                // belongs to this exact request, never a mutable environment slot.
                for (int index = pending.Count - 1; index >= 0; index--)
                {
                    var request = pending[index];
                    if (!request.Cancellation.IsCancellationRequested) continue;
                    request.Completion.TrySetCanceled(request.Cancellation); pending.RemoveAt(index);
                    Interlocked.Increment(ref cancelledRequests);
                }
                while (pending.Count < maximumBatch && queue.Reader.TryRead(out var replacement))
                {
                    if (replacement.Cancellation.IsCancellationRequested)
                    {
                        replacement.Completion.TrySetCanceled(replacement.Cancellation);
                        Interlocked.Increment(ref cancelledRequests);
                    }
                    else pending.Add(replacement);
                }
                if (pending.Count > 0)
                {
                    var batch = pending.ToArray(); long started = Stopwatch.GetTimestamp();
                    try
                    {
                        var predictions = await model.PredictBatchAsync(batch.Select(request => request.Observation).ToArray()).ConfigureAwait(false);
                        if (predictions.Count != batch.Length) throw new InvalidOperationException("Model batch output count does not match its input.");
                        for (int index = 0; index < batch.Length; index++) Validate(predictions[index], batch[index].Observation);
                        Interlocked.Add(ref modelTicks, Stopwatch.GetTimestamp() - started);
                        Interlocked.Increment(ref batches); Interlocked.Add(ref requests, batch.Length);
                        Interlocked.Increment(ref batchSizes[batch.Length]);
                        observedMaximum = Math.Max(observedMaximum, batch.Length);
                        maximumHeadFamilies = Math.Max(maximumHeadFamilies, batch.Select(request => request.Observation.Head).Distinct(StringComparer.Ordinal).Count());
                        foreach (var request in batch) Interlocked.Add(ref queueTicks, started - request.Enqueued);
                        for (int index = 0; index < batch.Length; index++) batch[index].Completion.TrySetResult(predictions[index]);
                    }
                    catch (Exception error) { foreach (var request in batch) request.Completion.TrySetException(error); }
                }
            }
        }
        catch (Exception error)
        {
            queue.Writer.TryComplete(error);
            foreach (var request in pending) request.Completion.TrySetException(error);
            while (queue.Reader.TryRead(out var request)) request.Completion.TrySetException(error);
        }
    }

    private static void Validate(SearchPrediction prediction, SearchObservation observation)
    {
        if (prediction.Priors.Length != observation.LegalActionIds.Length ||
            !float.IsFinite(prediction.ObserverValue) || Math.Abs(prediction.ObserverValue) > 1)
            throw new InvalidOperationException("Model output has an invalid shape or value.");
        double sum = 0;
        foreach (float prior in prediction.Priors)
        {
            if (!float.IsFinite(prior) || prior < 0) throw new InvalidOperationException("Model priors must be finite and nonnegative.");
            sum += prior;
        }
        if (!(sum > 0)) throw new InvalidOperationException("Model assigned zero mass to all legal actions.");
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0) return;
        queue.Writer.TryComplete(); await pump.ConfigureAwait(false);
    }
}
