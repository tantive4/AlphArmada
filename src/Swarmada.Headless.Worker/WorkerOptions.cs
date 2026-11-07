using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace Swarmada.Headless.Worker;

public sealed record WorkerOptions
{
    public string Data { get; init; } = Path.Combine(AppContext.BaseDirectory, "data");
    public string Output { get; init; } = "worker-output";
    public string RunId { get; init; } = "run-" + DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
    public string WorkerId { get; init; } = Environment.MachineName;
    public string CellId { get; init; } = "selfplay";
    public string SourceRevision { get; init; } = Environment.GetEnvironmentVariable("ALPHARMADA_SOURCE_REVISION") ?? "unspecified";
    public int Games { get; init; } = 1; // zero means until stopped
    public int Concurrency { get; init; } = 1;
    public int CpuWorkers { get; init; } = Math.Max(1, Environment.ProcessorCount - 1);
    public int BatchSize { get; init; } = 16;
    public double BatchDelayMs { get; init; } = 1;
    public int Simulations { get; init; } = 200;
    public int MaximumNativeActions { get; init; } = 96;
    public int MaximumDecisionEdges { get; init; } = 2048;
    public ulong Seed { get; init; } = 7331;
    public string Preset { get; init; } = "small";
    public string? Objective { get; init; }
    public string? Match { get; init; }
    public string? Setup { get; init; }
    public string? Model { get; init; }
    public double SetupEpsilon { get; init; } = .05;
    public double MaxGameSeconds { get; init; } = 1800;
    public int MaxSteps { get; init; } = 100000;
    public int MaxRecords { get; init; } = 20000;
    public long MaxShardBytes { get; init; } = 512L * 1024 * 1024;
    public long MaxOutputBytes { get; init; } = 20L * 1024 * 1024 * 1024;
    public long MemoryLimitMb { get; init; } = 10240;
    public double DurationSeconds { get; init; }
    public double DrainSeconds { get; init; } = 30;
    public double HeartbeatSeconds { get; init; } = 10;
    public int MaximumConsecutiveFailures { get; init; } = 8;
    public int SetupChoiceLimit { get; init; } = 2000;
    public int SetupRetries { get; init; } = 4;
    public bool Replay { get; init; }
    public bool SubtreeReuse { get; init; } = true;
    public string? StopFile { get; init; }
    public bool ValidateOnly { get; init; }

    public static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase, PropertyNameCaseInsensitive = true,
        IncludeFields = true, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        ReadCommentHandling = JsonCommentHandling.Skip
    };

    public static WorkerOptions Parse(string[] args)
    {
        var config = new WorkerOptions();
        for (int i = 0; i < args.Length; i++)
            if (args[i] == "--config")
            {
                if (++i == args.Length) throw new ArgumentException("--config needs a path.");
                config = JsonSerializer.Deserialize<WorkerOptions>(File.ReadAllText(args[i]), Json) ?? throw new ArgumentException("Empty worker configuration.");
            }
        for (int i = 0; i < args.Length; i++)
        {
            string option = args[i];
            if (option == "run") continue;
            if (option == "--config") { i++; continue; }
            if (option == "--replay") { config = config with { Replay = true }; continue; }
            if (option == "--no-subtree-reuse") { config = config with { SubtreeReuse = false }; continue; }
            if (option == "--validate-only") { config = config with { ValidateOnly = true }; continue; }
            if (++i == args.Length) throw new ArgumentException(option + " needs a value.");
            string value = args[i]; int Int() => int.Parse(value, CultureInfo.InvariantCulture);
            long Long() => long.Parse(value, CultureInfo.InvariantCulture);
            double Double() => double.Parse(value, CultureInfo.InvariantCulture);
            config = option switch
            {
                "--data" => config with { Data = value }, "--output" => config with { Output = value },
                "--run-id" => config with { RunId = value }, "--worker-id" => config with { WorkerId = value }, "--cell-id" => config with { CellId = value },
                "--source-revision" => config with { SourceRevision = value },
                "--games" => config with { Games = Int() }, "--concurrency" => config with { Concurrency = Int() }, "--cpu-workers" => config with { CpuWorkers = Int() },
                "--batch-size" => config with { BatchSize = Int() }, "--batch-delay-ms" => config with { BatchDelayMs = Double() },
                "--simulations" => config with { Simulations = Int() }, "--maximum-native-actions" => config with { MaximumNativeActions = Int() },
                "--maximum-decision-edges" => config with { MaximumDecisionEdges = Int() }, "--seed" => config with { Seed = ulong.Parse(value, CultureInfo.InvariantCulture) },
                "--preset" => config with { Preset = value }, "--objective" => config with { Objective = value }, "--match" => config with { Match = value },
                "--setup" => config with { Setup = value }, "--model" => config with { Model = value }, "--setup-epsilon" => config with { SetupEpsilon = Double() },
                "--max-game-seconds" => config with { MaxGameSeconds = Double() }, "--max-steps" => config with { MaxSteps = Int() },
                "--max-records" => config with { MaxRecords = Int() }, "--max-shard-bytes" => config with { MaxShardBytes = Long() },
                "--max-output-bytes" => config with { MaxOutputBytes = Long() }, "--memory-limit-mb" => config with { MemoryLimitMb = Long() },
                "--duration-seconds" => config with { DurationSeconds = Double() }, "--drain-seconds" => config with { DrainSeconds = Double() },
                "--heartbeat-seconds" => config with { HeartbeatSeconds = Double() }, "--stop-file" => config with { StopFile = value },
                "--maximum-consecutive-failures" => config with { MaximumConsecutiveFailures = Int() },
                "--setup-choice-limit" => config with { SetupChoiceLimit = Int() }, "--setup-retries" => config with { SetupRetries = Int() },
                _ => throw new ArgumentException("Unknown worker argument: " + option)
            };
        }
        config.Validate(); return config with { Data = Path.GetFullPath(config.Data), Output = Path.GetFullPath(config.Output) };
    }

    public void Validate()
    {
        foreach (string id in new[] { RunId, WorkerId, CellId })
            if (!Regex.IsMatch(id, "^[A-Za-z0-9._-]{1,64}$", RegexOptions.CultureInvariant))
                throw new ArgumentException("runId, workerId and cellId must use 1–64 ASCII letters, digits, dots, underscores or hyphens.");
        if (Games < 0 || Concurrency is < 1 or > 64 || CpuWorkers is < 1 or > 256 || BatchSize is < 1 or > 256 || Simulations < 1 ||
            MaximumNativeActions < 1 || MaximumDecisionEdges < 1 || MaxSteps < 1 || MaxRecords < 1 || SetupChoiceLimit < 1 || SetupRetries < 1 || MaximumConsecutiveFailures < 1)
            throw new ArgumentOutOfRangeException(nameof(Games), "Invalid worker count/budget/limit.");
        if (MaxShardBytes < 1024 || MaxOutputBytes < MaxShardBytes || MemoryLimitMb < 0) throw new ArgumentOutOfRangeException(nameof(MaxShardBytes));
        if (!double.IsFinite(SetupEpsilon) || SetupEpsilon is < 0 or > 1 || !double.IsFinite(BatchDelayMs) || BatchDelayMs is < 0 or > 1000 ||
            !double.IsFinite(MaxGameSeconds) || MaxGameSeconds <= 0 || !double.IsFinite(DurationSeconds) || DurationSeconds < 0 ||
            !double.IsFinite(DrainSeconds) || DrainSeconds < 0 || !double.IsFinite(HeartbeatSeconds) || HeartbeatSeconds <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxGameSeconds), "Invalid timing/exploration limit.");
    }
}
