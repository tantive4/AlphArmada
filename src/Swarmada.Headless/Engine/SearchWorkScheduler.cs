namespace Swarmada.Headless.Search;

/// <summary>Bounds synchronous game/tree work independently from the number of
/// live environments and the inference batch size. Release the lease before
/// awaiting inference, then acquire a new lease before touching the workspace.
/// A lease must never cross a handoff to another owner of the same tree.</summary>
public sealed class SearchWorkScheduler
{
    private sealed class Lease(SearchWorkScheduler owner) : IDisposable
    {
        private SearchWorkScheduler? held = owner;
        public void Dispose()
        {
            var released = Interlocked.Exchange(ref held, null);
            if (released is null) return;
            Interlocked.Decrement(ref released.active);
            released.slots.Release();
        }
    }

    private readonly SemaphoreSlim slots;
    private int active, observedMaximum;
    private long entries;
    public int MaximumCpuWorkers { get; }
    public int ActiveWorkers => Volatile.Read(ref active);
    public int MaximumObservedWorkers => Volatile.Read(ref observedMaximum);
    public long Entries => Interlocked.Read(ref entries);

    public SearchWorkScheduler(int maximumCpuWorkers)
    {
        if (maximumCpuWorkers < 1) throw new ArgumentOutOfRangeException(nameof(maximumCpuWorkers));
        MaximumCpuWorkers = maximumCpuWorkers;
        slots = new(maximumCpuWorkers, maximumCpuWorkers);
    }

    public async ValueTask<IDisposable> EnterAsync(CancellationToken cancellationToken = default)
    {
        await slots.WaitAsync(cancellationToken).ConfigureAwait(false);
        int count = Interlocked.Increment(ref active), previous;
        Interlocked.Increment(ref entries);
        do
        {
            previous = Volatile.Read(ref observedMaximum);
            if (count <= previous) break;
        } while (Interlocked.CompareExchange(ref observedMaximum, count, previous) != previous);
        return new Lease(this);
    }
}
