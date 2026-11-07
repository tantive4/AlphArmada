using System.Diagnostics;
using Swarmada.Headless.Catalog;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Search;

namespace Swarmada.Headless.Learning;

public sealed record EncodedSearchInput(StateEncoding State, SpatialEncoding Spatial, PolicyEncoding Policy)
{
    public IReadOnlyList<EncodedTensor> Tensors => [.. State.Tensors, .. SpatialTensors(Spatial), .. Policy.Tensors];
    public long TensorBytes => Tensors.Sum(tensor => tensor.ByteSize);
    public TensorRecord ToRecord() => new(ProductionSearchEncoder.SchemaFor(State.SchemaFingerprint, Spatial.Height, Spatial.Width, Spatial.TerrainRaster.Length != 0), Tensors);
    private static IEnumerable<EncodedTensor> SpatialTensors(SpatialEncoding spatial)
    {
        yield return new("spatial.ship_ship", [spatial.ShipCount, spatial.ShipCount, SpatialEncoding.ShipShipWidth], spatial.ShipShipRelations);
        yield return new("spatial.ship_squad", [spatial.ShipCount, spatial.SquadronCount, SpatialEncoding.ShipSquadWidth], spatial.ShipSquadRelations);
        yield return new("spatial.squad_squad", [spatial.SquadronCount, spatial.SquadronCount, SpatialEncoding.SquadSquadWidth], spatial.SquadSquadRelations);
        yield return new("spatial.ship_coordinates", [spatial.ShipCount, SpatialEncoding.ShipCoordinateWidth], spatial.ShipCoordinates);
        yield return new("spatial.squadron_coordinates", [spatial.SquadronCount, SpatialEncoding.SquadronCoordinateWidth], spatial.SquadronCoordinates);
        yield return new("spatial.squadron_radii", [spatial.SquadronCount, SpatialEncoding.SquadronRadiusWidth], spatial.SquadronRadii);
        yield return new("spatial.ship_on_board", [spatial.ShipCount], spatial.ShipOnBoard);
        yield return new("spatial.squadron_on_board", [spatial.SquadronCount], spatial.SquadronOnBoard);
        yield return new("spatial.terrain", [spatial.TerrainRaster.Length == 0 ? 0 : SpatialEncoding.TerrainChannels, spatial.Height, spatial.Width], spatial.TerrainRaster);
    }
}

public sealed record ProductionEncoderStatistics(long Encodings, long BoardBuilds, long BoardReuses,
    double CoreMilliseconds, double GeometryMilliseconds, double PolicyMilliseconds, double CopyMilliseconds,
    long OutputTensorBytes, long MaximumObservationBytes);

/// <summary>Visible state plus native conditional candidate features. This encoder
/// never reads game RNG or future deck order and never substitutes the diagnostic mock.
/// Only one search workspace may use an instance at a time.</summary>
public sealed class ProductionSearchEncoder : ISearchObservationEncoder
{
    public const string Version = "alpharmada.learning.v1";
    private readonly StateEncoder core;
    private readonly EncoderLimits limits;
    private readonly SpatialEncoderOptions spatialOptions;
    private SpatialEncoder? spatial;
    private object? topology;
    private StateEncoding? cachedState;
    private SpatialEncoding? cachedSpatial;
    private long generation = -1, knowledgeVersion, boundaryVersion;
    private SearchKnowledge? view, boundary;
    private int actor;
    private ulong boardKey;
    private long encodings, builds, reuses, coreTicks, geometryTicks, policyTicks, copyTicks, bytes, maximumBytes;
    public static string SchemaFor(string stateFingerprint, int height = 64, int width = 128, bool terrain = true)
        => Version + ":" + stateFingerprint + ":" + SpatialEncoding.Schema + ":" + height + ":" + width + ":" + terrain + ":" + PolicyEncoder.SchemaVersion;
    public string SchemaVersion => SchemaFor(core.Schema.Fingerprint, spatialOptions.Height, spatialOptions.Width, spatialOptions.IncludeTerrain);
    public bool RequiresNativeCandidates => true;
    public StateEncoderSchema Schema => core.Schema;
    public ProductionEncoderStatistics Statistics => new(encodings, builds, reuses, Ms(coreTicks), Ms(geometryTicks),
        Ms(policyTicks), Ms(copyTicks), bytes, maximumBytes);

    public ProductionSearchEncoder(FleetData data, EncoderLimits? limits = null, SpatialEncoderOptions? spatialOptions = null)
    { core = new(StateEncoderSchema.For(data)); this.limits = limits ?? new(); this.spatialOptions = spatialOptions ?? new(); }

    public void Invalidate() { generation = -1; cachedState = null; cachedSpatial = null; }

    public SearchObservation Encode(GameSession position, SearchKnowledge knowledge, PolicyDecision decision,
        IReadOnlyList<string> prefix, SearchKnowledge? hypothesisKnowledge = null)
    {
        if (decision.Actor != knowledge.Observer) throw new ArgumentException("Encoder view must belong to the deciding player.");
        limits.ValidateCounts(position.Context.State.Ships.Length, position.Context.State.Squadrons.Length, position.Context.State.Upgrades.Length);
        if (decision.Candidates is null || decision.Candidates.Count != decision.Choices.Count)
            throw new ArgumentException("Production policy encoding requires exact native candidate descriptors.");
        if (!ReferenceEquals(topology, position.Context.Topology))
        { topology = position.Context.Topology; spatial = new(position, spatialOptions); Invalidate(); }
        bool reuse = cachedState is not null && generation == position.PositionGeneration && actor == decision.Actor &&
            ReferenceEquals(view, knowledge) && knowledgeVersion == knowledge.Version && ReferenceEquals(boundary, hypothesisKnowledge) &&
            boundaryVersion == (hypothesisKnowledge?.Version ?? 0);
        if (!reuse)
        {
            long started = Stopwatch.GetTimestamp();
            var written = core.Encode(position, knowledge, decision.Actor, decision.Actor, hypothesisKnowledge);
            if (written.CoverageIssues.Length != 0)
                throw new SearchCoverageException("State encoder coverage: " + string.Join("; ", written.CoverageIssues.Select(issue => issue.Location + ": " + issue.Detail)));
            coreTicks += Stopwatch.GetTimestamp() - started;
            started = Stopwatch.GetTimestamp(); var physical = spatial!.EncodeBorrowed(position, decision.Actor);
            geometryTicks += Stopwatch.GetTimestamp() - started;
            started = Stopwatch.GetTimestamp(); cachedState = written.Clone(); cachedSpatial = physical.Clone();
            copyTicks += Stopwatch.GetTimestamp() - started;
            // A versioned feature hash discriminates inference observations; exact
            // internal rules/knowledge keys remain responsible for MCTS traversal.
            boardKey = FeatureHash(cachedState.Tensors.Concat(SpatialTensorList(cachedSpatial)));
            generation = position.PositionGeneration; actor = decision.Actor; view = knowledge; knowledgeVersion = knowledge.Version;
            boundary = hypothesisKnowledge; boundaryVersion = hypothesisKnowledge?.Version ?? 0; builds++;
        }
        else reuses++;
        long policyStarted = Stopwatch.GetTimestamp(); var policy = PolicyEncoder.Encode(position, decision, prefix);
        policyTicks += Stopwatch.GetTimestamp() - policyStarted;
        var input = new EncodedSearchInput(cachedState!, cachedSpatial!, policy);
        ulong information = SearchHash.Add(SearchHash.Add(boardKey, decision.Stage), FeatureHash(policy.Tensors));
        var legal = decision.Choices.Select(choice => choice.Id).ToArray();
        foreach (string id in legal) information = SearchHash.Add(information, id);
        long size = input.TensorBytes; encodings++; bytes += size; maximumBytes = Math.Max(maximumBytes, size);
        return new(policy.Head, decision.Actor, boardKey, information, [], policy.Context, legal, false, input);
    }

    private static IEnumerable<EncodedTensor> SpatialTensorList(SpatialEncoding value)
    {
        yield return new("ss", [value.ShipShipRelations.Length], value.ShipShipRelations);
        yield return new("sq", [value.ShipSquadRelations.Length], value.ShipSquadRelations);
        yield return new("qq", [value.SquadSquadRelations.Length], value.SquadSquadRelations);
        yield return new("terrain", [value.TerrainRaster.Length], value.TerrainRaster);
    }
    internal static ulong FeatureHash(IEnumerable<EncodedTensor> tensors)
    {
        ulong hash = SearchHash.Initial;
        foreach (var tensor in tensors)
        {
            hash = SearchHash.Add(hash, tensor.Name);
            foreach (int dimension in tensor.Shape) hash = SearchHash.Add(hash, unchecked((ulong)dimension));
            // Simple FNV mixing of raw feature bytes, deliberately outside the
            // engine's exact traversal hash and never fed to the neural model.
            switch (tensor.Data)
            {
                case float[] values: foreach (float value in values) { hash ^= BitConverter.SingleToUInt32Bits(value); hash *= 1099511628211UL; } break;
                case int[] values: foreach (int value in values) { hash ^= unchecked((uint)value); hash *= 1099511628211UL; } break;
                case byte[] values: foreach (byte value in values) { hash ^= value; hash *= 1099511628211UL; } break;
                case bool[] values: foreach (bool value in values) { hash ^= value ? 1UL : 0; hash *= 1099511628211UL; } break;
                default: throw new ArgumentException("Unsupported feature dtype.");
            }
        }
        return hash;
    }
    private static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
}

/// <summary>Consumes the real padded tensors and returns random policy/value;
/// no strength heuristic, command prediction or GPU timing is introduced.</summary>
public sealed class TensorRandomSearchModel(ulong seed = 1) : IBatchedSearchModel
{
    public long Samples { get; private set; }
    public long InputTensorBytes { get; private set; }
    public long MaximumPaddedBatchBytes { get; private set; }
    public ValueTask<IReadOnlyList<SearchPrediction>> PredictBatchAsync(IReadOnlyList<SearchObservation> observations,
        CancellationToken cancellationToken = default)
    {
        var inputs = observations.Select(observation => observation.Encoded ??
            throw new ArgumentException("The real-encoder mock refuses diagnostic mock observations.")).ToArray();
        var batch = EncoderBatchCollator.Collate(inputs); MaximumPaddedBatchBytes = Math.Max(MaximumPaddedBatchBytes, batch.ByteSize);
        // Validate every padded tensor and read its contents. This exercises the
        // actual tensor preparation path rather than using only action labels.
        foreach (var tensor in batch.Tensors) tensor.Validate();
        var predictions = new SearchPrediction[inputs.Length];
        for (int sample = 0; sample < inputs.Length; sample++)
        {
            cancellationToken.ThrowIfCancellationRequested(); var observation = observations[sample];
            ulong hash = SearchHash.Add(ProductionSearchEncoder.FeatureHash(inputs[sample].Tensors), seed);
            var priors = new float[observation.LegalActionIds.Length]; double sum = 0;
            for (int action = 0; action < priors.Length; action++)
            { priors[action] = (float)(.01 + SearchHash.Unit(SearchHash.Add(hash, (ulong)action))); sum += priors[action]; }
            for (int action = 0; action < priors.Length; action++) priors[action] /= (float)sum;
            predictions[sample] = new(priors, (float)(2 * SearchHash.Unit(SearchHash.Add(hash, "value")) - 1));
            Samples++; InputTensorBytes += inputs[sample].TensorBytes;
        }
        return ValueTask.FromResult<IReadOnlyList<SearchPrediction>>(predictions);
    }
}
