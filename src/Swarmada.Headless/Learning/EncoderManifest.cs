using Swarmada.Headless.Search;

namespace Swarmada.Headless.Learning;

/// <summary>Persist once beside replay shards and checkpoints. Integer categories
/// must be interpreted using this exact manifest/fingerprint, never a later catalog.</summary>
public static class EncoderManifest
{
    public static object Describe(StateEncoderSchema state) => new
    {
        schema = ProductionSearchEncoder.SchemaFor(state.Fingerprint),
        catalogFingerprint = state.Fingerprint,
        policyFingerprint = EncoderSymbols.Fingerprint,
        actionCatalog = ActionCatalogProfile.Version,
        ships = state.Ships.Values, squadrons = state.Squadrons.Values,
        upgrades = state.Upgrades.Values, objectives = state.Objectives.Values,
        keywords = state.Keywords.Values, semanticSymbols = state.Symbols.Values,
        spatialKinds = StateEncoder.SpatialKinds,
        semanticWords = state.SymbolWords.Values,
        policyWords = EncoderSymbols.Vocabulary, policyHeads = PolicyEncoder.Heads,
        stateFields = state.Fields,
        coordinates = "Observer-relative x,y divided by board half-extents. Player two rotates 180 degrees. Ships add sin/cos heading; squadrons have no heading. Raster dimensions H64,W128; no per-unit planes.",
        references = "Native entity indices remain stable across reserve/death/revival. Ship rows precede squadron rows. -1 means absent/padding. Positive semantic symbols index semanticSymbols; values <= -2 reference compositional core.symbol_tokens sequences. Token IDs are categorical, never numeric magnitudes.",
        padding = "Transport uses actual counts. Batch collation pads float/byte tensors with zero and integer references with -1; actual shapes and roster/on-board/legal masks remain distinct.",
        policy = new
        {
            context = PolicyEncoding.ContextWidth, candidate = PolicyEncoding.CandidateWidth, candidateReferences = PolicyEncoding.ReferenceWidth,
            references = new[] { "pointer entity", "native source", "native target", "physical index", "action kind", "known native-field bit mask" },
            candidateFields = new[] { "0 legal; 1 represented complete-action count/1024", "2:43 action-kind one-hot", "43:47 command", "47:53 source zone; 53:59 target zone", "59:62 payment; 62:68 defense type", "68:71 position known,x,y", "71:74 heading known,sin,cos", "74:76 speed known,value/4", "76:78 amount known,value/16", "78 reflected; 79:83 course clicks/2", "83 ordinary attack;84 native measurement valid;85:89 range one-hot", "89 native distance/1000;90 effective obstruction", "91 endpoint center outside board;92 displacement/1000;93 displacement known", "94 course length/4", "96:99 battery dice/16;99 ignition;100 mine;101 extreme range", "102 ignition dice-distance/1000;103:107 dice-range one-hot;107 distance known" },
            conditional = "Each prefix is a separate conditional policy distribution. Candidate descriptors mask arguments that still vary underneath that choice. UCT hidden hypotheses never enter neural policy inputs/targets."
        },
        auxiliaries = new[] { "terminal result/value", "final raw points for both players", "final hull by ship", "final hull by squadron", "final game round" }
    };
}
