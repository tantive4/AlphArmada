using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Learning;

namespace Swarmada.Headless.Worker;

public sealed record WorkerVersions(string SchemaVersion, string CatalogFingerprint, string CodeVersion,
    string ModelVersion, string PolicyFingerprint, string ActionCatalog, string RulesVariant, string SourceRevision);
public sealed record ShardFile(string Path, long Bytes, string Sha256);
public sealed record PublishedShard(string GameId, string Directory, int Decisions, long RawTensorBytes,
    long CompressedTensorBytes, long InputStreamBytes, long TargetsBytes, long PayloadBytes, long UploadBytes,
    long SparsePolicyBytes, string ManifestSha256);

/// <summary>Append-only disk spool. Neural inputs are released after each append;
/// only a genuine terminal result can publish a checksummed immutable READY shard.</summary>
public sealed class GameRecordSpool : IDisposable
{
    private readonly WorkerOptions options;
    private readonly OutputBudget budget;
    private readonly string gameId, stagingRoot, staging;
    private readonly FileStream inputs, metadata;
    private FileStream? actions;
    private long reserved;
    private bool closed, published;
    public int Records { get; private set; }
    public long RawTensorBytes { get; private set; }
    public long CompressedTensorBytes { get; private set; }
    public long SparsePolicyBytes { get; private set; }
    public long WrittenBytes => reserved;
    public GameRecordSpool(WorkerOptions options, OutputBudget budget, string identity, string gameId)
    {
        this.options = options; this.budget = budget; this.gameId = gameId;
        stagingRoot = Path.GetFullPath(Path.Combine(options.Output, ".staging", identity)); Directory.CreateDirectory(stagingRoot);
        staging = Path.Combine(stagingRoot, gameId); Directory.CreateDirectory(staging);
        inputs = new(Path.Combine(staging, "inputs.bin"), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        metadata = new(Path.Combine(staging, "targets.pending.jsonl"), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        Reserve(12); inputs.Write("SWAREC01"u8); using var writer = new BinaryWriter(inputs, Encoding.UTF8, leaveOpen: true); writer.Write(1u);
    }
    private void Reserve(long bytes)
    {
        if (reserved + bytes > options.MaxShardBytes) throw new WorkerLimitException("shard-byte-limit", "Configured per-game spool limit reached; no terminal data published.");
        budget.Reserve(bytes); reserved += bytes;
    }
    public void Append(TrainingDecisionRecord record, int searchSequence, string schemaVersion)
    {
        if (Records >= options.MaxRecords) throw new WorkerLimitException("record-limit", "Configured per-game record count reached.");
        if (record.Input.ToRecord().SchemaVersion != schemaVersion || record.Policy.CandidateIndices.Length != record.Policy.Visits.Length || record.Policy.TotalVisits <= 0)
            throw new InvalidDataException("Training input schema or policy visits are invalid.");
        for (int i = 0; i < record.Policy.Visits.Length; i++)
            if (record.Policy.Visits[i] <= 0 || (uint)record.Policy.CandidateIndices[i] >= (uint)record.Input.Policy.LegalMask.Length ||
                record.Input.Policy.LegalMask[record.Policy.CandidateIndices[i]] == 0)
                throw new InvalidDataException("Policy visits include an illegal/missing candidate.");
        byte[] payload = ReplayTensorCodec.Encode(record.Input.ToRecord(), compress: true);
        long offset = inputs.Position + 4;
        byte[] target = JsonSerializer.SerializeToUtf8Bytes(new
        {
            index = Records, searchSequence, inputOffset = offset, inputBytes = payload.Length,
            observer = record.Observer, prefix = record.Prefix,
            candidateIndices = record.Policy.CandidateIndices, visits = record.Policy.Visits,
            tensorBytes = record.Input.TensorBytes, candidateCount = record.Input.Policy.LegalMask.Length
        }, WorkerOptions.Json);
        Reserve(4L + payload.Length + target.Length + 1);
        using (var writer = new BinaryWriter(inputs, Encoding.UTF8, leaveOpen: true)) writer.Write(checked((uint)payload.Length));
        inputs.Write(payload); metadata.Write(target); metadata.WriteByte((byte)'\n');
        Records++; RawTensorBytes += record.Input.TensorBytes; CompressedTensorBytes += payload.Length; SparsePolicyBytes += record.Policy.ByteSize;
    }
    public void RecordAction(GameAction action)
    {
        if (!options.Replay) return;
        actions ??= new(Path.Combine(staging, "actions.jsonl"), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(action, WorkerOptions.Json); Reserve(bytes.Length + 1L); actions.Write(bytes); actions.WriteByte((byte)'\n');
    }
    public void WriteJsonPayload(string name, object value) => WriteBytes(name, JsonSerializer.SerializeToUtf8Bytes(value, WorkerOptions.Json));
    public void WriteBytes(string name, byte[] bytes)
    {
        if (Path.GetFileName(name) != name) throw new ArgumentException("A shard payload name must be one file.");
        Reserve(bytes.Length); using var stream = new FileStream(Path.Combine(staging, name), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        stream.Write(bytes); stream.Flush(true);
    }
    public PublishedShard Publish(GameSession game, object gameReport, WorkerVersions versions, byte[] encoderManifest)
    {
        var terminal = TerminalTrainingTargets.FromTerminal(game); // rejects unfinished games
        CloseStreams();
        WriteBytes("encoder-manifest.json.gz", encoderManifest); WriteJsonPayload("game.json", gameReport);
        WriteTargets(terminal, versions.SchemaVersion);
        long pendingBytes = new FileInfo(Path.Combine(staging, "targets.pending.jsonl")).Length;
        File.Delete(Path.Combine(staging, "targets.pending.jsonl")); reserved -= pendingBytes; budget.Release(pendingBytes);
        var files = Directory.EnumerateFiles(staging).Select(path => new ShardFile(Path.GetFileName(path), new FileInfo(path).Length, HashFile(path)))
            .OrderBy(file => file.Path, StringComparer.Ordinal).ToArray();
        WriteJsonPayload("manifest.json", new
        {
            format = "swarmada.worker-shard.v1", options.RunId, options.WorkerId, shardId = gameId,
            createdUtc = DateTimeOffset.UtcNow, versions,
            games = new[] { new { gameId, status = "terminal", decisions = Records } }, files
        });
        string manifestPath = Path.Combine(staging, "manifest.json"); string manifestHash = HashFile(manifestPath);
        WriteJsonPayload("READY", new { format = "swarmada.worker-shard-ready.v1", manifestBytes = new FileInfo(manifestPath).Length, manifestSha256 = manifestHash });
        SyncDirectory(staging);
        string destinationRoot = Path.Combine(options.Output, "shards"); Directory.CreateDirectory(destinationRoot);
        string destination = Path.Combine(destinationRoot, gameId); Directory.Move(staging, destination);
        SyncDirectory(destinationRoot); SyncDirectory(stagingRoot); published = true;
        long inputBytes = files.Single(file => file.Path == "inputs.bin").Bytes;
        long targetBytes = files.Single(file => file.Path == "targets.json").Bytes;
        return new(gameId, destination, Records, RawTensorBytes, CompressedTensorBytes, inputBytes, targetBytes,
            files.Sum(file => file.Bytes), reserved, SparsePolicyBytes, manifestHash);
    }
    private void WriteTargets(TerminalTrainingTargets terminal, string schemaVersion)
    {
        string path = Path.Combine(staging, "targets.json");
        long upperBound = new FileInfo(Path.Combine(staging, "targets.pending.jsonl")).Length +
            JsonSerializer.SerializeToUtf8Bytes(terminal, WorkerOptions.Json).Length + Encoding.UTF8.GetByteCount(schemaVersion) + Encoding.UTF8.GetByteCount(gameId) + 2048;
        Reserve(upperBound);
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject(); writer.WriteString("format", "swarmada.training-targets.v1"); writer.WriteString("gameId", gameId);
            writer.WriteString("schemaVersion", schemaVersion); writer.WritePropertyName("terminalTargets"); JsonSerializer.Serialize(writer, terminal, WorkerOptions.Json);
            writer.WriteStartArray("decisions");
            foreach (string line in File.ReadLines(Path.Combine(staging, "targets.pending.jsonl")))
            {
                using var document = JsonDocument.Parse(line); document.RootElement.WriteTo(writer);
            }
            writer.WriteEndArray(); writer.WriteEndObject(); writer.Flush();
        }
        stream.Flush(true);
        if (stream.Length > upperBound) throw new InvalidDataException("Target JSON exceeded its reserved serialization bound.");
        long unused = upperBound - stream.Length; reserved -= unused; budget.Release(unused);
    }
    private void CloseStreams()
    {
        if (closed) return; closed = true;
        inputs.Flush(true); metadata.Flush(true); actions?.Flush(true); inputs.Dispose(); metadata.Dispose(); actions?.Dispose();
    }
    public void Dispose()
    {
        CloseStreams();
        if (published) return;
        CleanupOwnedDirectory(stagingRoot, staging); budget.Release(reserved); reserved = 0;
    }
    public static void CleanupOwnedDirectory(string root, string directory)
    {
        string absolute = Path.GetFullPath(directory), prefix = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        if (!absolute.StartsWith(prefix, comparison)) throw new IOException("Refusing cleanup outside the worker staging directory.");
        if (!Directory.Exists(absolute)) return;
        if ((File.GetAttributes(absolute) & FileAttributes.ReparsePoint) != 0) throw new IOException("Refusing recursive cleanup of a staging symlink.");
        Directory.Delete(absolute, recursive: true);
    }
    public static string HashFile(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
    private static void SyncDirectory(string path)
    {
        if (!OperatingSystem.IsLinux()) return;
        int fd = NativeOpen(path, 65536); if (fd < 0) throw new IOException("Cannot open output directory for durable fsync.");
        try { if (NativeSync(fd) != 0) throw new IOException("Cannot durably fsync output directory."); } finally { NativeClose(fd); }
    }
    [DllImport("libc", EntryPoint = "open", SetLastError = true)] private static extern int NativeOpen([MarshalAs(UnmanagedType.LPUTF8Str)] string path, int flags);
    [DllImport("libc", EntryPoint = "fsync", SetLastError = true)] private static extern int NativeSync(int fd);
    [DllImport("libc", EntryPoint = "close", SetLastError = true)] private static extern int NativeClose(int fd);
}
