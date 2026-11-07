using Swarmada.Headless.Runtime;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Learning;

namespace Swarmada.Headless.Search;

/// <summary>Owned, visible-only buffers. Decision context is separate so an eventual
/// neural model can cache its board trunk across buffered argument choices.</summary>
public sealed record SearchObservation(string Head, int Actor, ulong BoardKey, ulong InformationKey,
    float[] BoardFeatures, float[] DecisionFeatures, string[] LegalActionIds, bool UseUct = false,
    EncodedSearchInput? Encoded = null);

/// <summary>One encoder belongs to one mutable search workspace. Returned observations
/// own their buffers and remain valid while another environment fills an inference batch.</summary>
public interface ISearchObservationEncoder
{
    string SchemaVersion { get; }
    bool RequiresNativeCandidates => false;
    void Invalidate();
    SearchObservation Encode(GameSession position, SearchKnowledge knowledge, PolicyDecision decision,
        IReadOnlyList<string> prefix, SearchKnowledge? hypothesisKnowledge = null);
}

public sealed record SearchPrediction(float[] Priors, float ObserverValue);

public interface IBatchedSearchModel
{
    ValueTask<IReadOnlyList<SearchPrediction>> PredictBatchAsync(
        IReadOnlyList<SearchObservation> observations, CancellationToken cancellationToken = default);
}

/// <summary>One mutable environment belongs to one running search. Reset samples
/// a fresh hypothetical world, never an actual-game random outcome.</summary>
public interface IMctsEnvironment
{
    PolicyDecision Current { get; }
    bool InBufferedPrefix { get; }
    void Reset(ulong simulationSeed);
    SearchObservation Encode();
    bool Select(string semanticId);
    void SampleChance(ref DeterministicRandom random);
}

/// <summary>Optional native fast path. Inspect settles rules and identifies the
/// current outcome without generating candidates or model tensors. A descriptor
/// may be rebound only after its complete traversal key has matched.</summary>
public interface ICachedMctsEnvironment : IMctsEnvironment
{
    bool SupportsCachedTraversal => true;
    string CompatibilityToken { get; }
    SearchTraversalState Inspect();
    object CaptureDecision();
    void BindDecision(object descriptor);
    bool Select(int index);
}

public sealed record SearchTraversalState(SearchTraversalKey Key, DecisionKind Kind, int Actor,
    TerminalOutcome? Outcome = null);

public sealed record MctsOptions(int Simulations = 200, double Exploration = 1.5,
    int MaximumNativeActions = 96, int MaximumDecisionEdges = 2048);

public sealed record SearchActionStatistics(string Id, float Prior, int Visits, double PlayerOneValueSum);
public sealed record SearchDecisionStatistics(int Actor, string Head, string[] Prefix,
    SearchActionStatistics[] Actions, bool HiddenHypothesis = false);

public sealed record MctsResult(int Simulations, string[] SelectedPath,
    SearchActionStatistics[] RootActions, long Nodes, long ActionSlots, long ModelRequests,
    long DecisionEdges, long NativeActions, long ChanceSamples, int MaximumNativeDepth,
    int MaximumPrefixDepth, long DepthLimitEvaluations, double ElapsedMilliseconds, long UctNodes = 0,
    int InheritedVisits = 0, int NewSimulations = 0, long CachedTraversals = 0,
    SearchDecisionStatistics[]? SelectedDecisions = null);

/// <summary>Stable little hash for reproducible mock outputs and local observation
/// discrimination; not a canonical engine/transposition hash or security primitive.</summary>
public static class SearchHash
{
    public const ulong Initial = 14695981039346656037UL;
    public static ulong Add(ulong hash, ulong value)
    {
        for (int index = 0; index < 8; index++) { hash ^= (byte)value; hash *= 1099511628211UL; value >>= 8; }
        return hash;
    }
    public static ulong Add(ulong hash, string text)
    {
        hash = Add(hash, (ulong)text.Length);
        foreach (char character in text) { hash ^= character; hash *= 1099511628211UL; }
        return hash;
    }
    public static ulong Add(ulong hash, ReadOnlySpan<float> values)
    {
        foreach (float value in values) hash = Add(hash, BitConverter.SingleToUInt32Bits(value));
        return hash;
    }
    public static double Unit(ulong hash)
    {
        hash ^= hash >> 30; hash *= 0xbf58476d1ce4e5b9UL;
        hash ^= hash >> 27; hash *= 0x94d049bb133111ebUL; hash ^= hash >> 31;
        return (hash >> 11) * (1.0 / 9007199254740992.0);
    }
}
