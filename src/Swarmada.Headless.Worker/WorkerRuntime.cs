using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Swarmada.Headless.Worker;

public sealed class WorkerLog : IDisposable
{
    private readonly object gate = new();
    private readonly WorkerOptions options;
    private readonly string path;
    private StreamWriter writer;
    private const long MaximumLogBytes = 16L * 1024 * 1024;
    public WorkerLog(WorkerOptions options)
    {
        this.options = options; Directory.CreateDirectory(Path.Combine(options.Output, "logs"));
        path = Path.Combine(options.Output, "logs", options.RunId + "." + options.WorkerId + "." + options.CellId + ".jsonl");
        writer = Open();
    }
    private StreamWriter Open() => new(new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read)) { AutoFlush = true };
    public void Write(string eventType, object data)
    {
        string line = JsonSerializer.Serialize(new { timestampUtc = DateTimeOffset.UtcNow, eventType,
            options.RunId, options.WorkerId, options.CellId, data }, WorkerOptions.Json);
        lock (gate)
        {
            if (writer.BaseStream.Length >= MaximumLogBytes)
            { writer.Dispose(); File.Move(path, path + ".1", overwrite: true); writer = Open(); }
            writer.WriteLine(line); Console.WriteLine(line);
        }
    }
    public void FlushDurably() { lock (gate) { writer.Flush(); ((FileStream)writer.BaseStream).Flush(true); } }
    public void Dispose() { lock (gate) writer.Dispose(); }
}

public sealed class WorkerLifetime : IDisposable
{
    private readonly CancellationTokenSource stopNew = new(), cancelGames = new(), monitoring = new();
    private readonly WorkerLog log;
    private readonly WorkerOptions options;
    private readonly List<IDisposable> registrations = [];
    private int stopRequested, signals;
    public CancellationToken StopNew => stopNew.Token;
    public CancellationToken CancelGames => cancelGames.Token;
    public CancellationToken Monitoring => monitoring.Token;
    public string StopReason { get; private set; } = "completed";
    public WorkerLifetime(WorkerOptions options, WorkerLog log)
    {
        this.options = options; this.log = log;
        Console.CancelKeyPress += OnInterrupt;
        if (OperatingSystem.IsLinux())
            foreach (var signal in new[] { PosixSignal.SIGTERM })
                registrations.Add(PosixSignalRegistration.Create(signal, context => { context.Cancel = true; Signal(signal.ToString()); }));
        if (options.DurationSeconds > 0) _ = DurationAsync();
    }
    private void OnInterrupt(object? sender, ConsoleCancelEventArgs args) { args.Cancel = true; Signal("console-interrupt"); }
    private void Signal(string reason)
    { if (Interlocked.Increment(ref signals) > 1) { cancelGames.Cancel(); return; } RequestStop(reason); }
    public void RequestStop(string reason, bool immediate = false)
    {
        if (Interlocked.CompareExchange(ref stopRequested, 1, 0) == 0)
        {
            StopReason = reason; log.Write("stopping", new { reason, options.DrainSeconds, immediate }); stopNew.Cancel();
            _ = DrainAsync();
        }
        if (immediate) cancelGames.Cancel();
    }
    private async Task DurationAsync()
    { try { await Task.Delay(TimeSpan.FromSeconds(options.DurationSeconds), monitoring.Token); RequestStop("duration-limit"); } catch (OperationCanceledException) { } }
    private async Task DrainAsync()
    { try { await Task.Delay(TimeSpan.FromSeconds(options.DrainSeconds), monitoring.Token); cancelGames.Cancel(); } catch (OperationCanceledException) { } }
    public void EndMonitoring() => monitoring.Cancel();
    public void Dispose()
    { monitoring.Cancel(); foreach (var registration in registrations) registration.Dispose(); Console.CancelKeyPress -= OnInterrupt; }
}

public sealed record ProcessSample(long RssBytes, long PeakRssBytes, long PrivateBytes, long ManagedBytes,
    long AllocatedBytes, long GcCommittedBytes, double CpuSeconds, int Threads, int Gen0, int Gen1, int Gen2)
{
    public static ProcessSample Capture()
    {
        using var process = Process.GetCurrentProcess(); process.Refresh(); var info = GC.GetGCMemoryInfo();
        return new(process.WorkingSet64, process.PeakWorkingSet64, process.PrivateMemorySize64, GC.GetTotalMemory(false),
            GC.GetTotalAllocatedBytes(false), info.TotalCommittedBytes, process.TotalProcessorTime.TotalSeconds,
            process.Threads.Count, GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2));
    }
}

public sealed class OutputBudget
{
    private long used;
    private readonly long maximum;
    public long UsedBytes => Interlocked.Read(ref used);
    public OutputBudget(string root, long maximum)
    {
        this.maximum = maximum; Directory.CreateDirectory(root);
        used = Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories).Sum(path => new FileInfo(path).Length);
    }
    public void Reserve(long bytes)
    { if (Interlocked.Add(ref used, bytes) > maximum) { Interlocked.Add(ref used, -bytes); throw new WorkerLimitException("output-limit", "Configured output byte budget reached."); } }
    public void Release(long bytes) => Interlocked.Add(ref used, -bytes);
}

public sealed class WorkerLimitException(string status, string message) : Exception(message)
{ public string Status { get; } = status; }
