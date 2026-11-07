using System.Security.Cryptography;
using System.Text;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Armada.Data;
using Swarmada.Headless.Catalog;

namespace Swarmada.Headless.Learning;

/// <summary>A finite, ordinal, versioned vocabulary. Index zero means unknown; identities are never hashed into floats.</summary>
public sealed class EncoderVocabulary
{
    private readonly Dictionary<string, int> indices;
    public IReadOnlyList<string> Values { get; }
    internal EncoderVocabulary(IEnumerable<string> values)
    {
        var sorted = values.Where(x => !string.IsNullOrEmpty(x)).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
        Values = Array.AsReadOnly(new[] { "<unknown>" }.Concat(sorted).ToArray());
        indices = Values.Select((value, index) => (value, index)).ToDictionary(x => x.value, x => x.index, StringComparer.Ordinal);
    }
    public int Index(string? value) => value is not null && indices.TryGetValue(value, out int index) ? index : 0;
}

public sealed record EncoderField(string Tensor, int Offset, int Width, string Name, string Meaning);
public sealed record EncoderCoverageIssue(string Code, string Location, string Detail);

/// <summary>
/// Versioned feature contract, shared by workers. Padding is zero; rows have actual counts.
/// Entity references are separate int arrays with -1 meaning absent. Float coordinates are
/// observer-relative, divided by half the board extent; ship heading is sin/cos, squads have none.
/// Scalars are divided by documented scales without clipping: custom games do not silently saturate.
/// One-hot identity blocks reserve explicit unknown slot zero and unused expansion slots.
/// </summary>
public sealed class StateEncoderSchema
{
    private static readonly ConditionalWeakTable<FleetData, StateEncoderSchema> Cache = new();
    public static StateEncoderSchema For(FleetData data) => Cache.GetValue(data, static value => new(value));
    public const string Version = "swarmada-headless.state-encoder.v3";
    public EncoderVocabulary Ships { get; }
    public EncoderVocabulary Squadrons { get; }
    public EncoderVocabulary Upgrades { get; }
    public EncoderVocabulary Objectives { get; }
    public EncoderVocabulary Keywords { get; }
    public EncoderVocabulary Symbols { get; }
    public EncoderVocabulary SymbolWords { get; }
    public string Fingerprint { get; }
    public IReadOnlyList<EncoderField> Fields { get; } = Array.AsReadOnly(new[]
    {
        new EncoderField("Global",0,256,"match","round/6; own-first, actor/current-player; map relocation; board metres; objective37; phase27; VP/400; counts; actor3; victory tokens/20; squadron phase progress; terminal/outcome; deck remaining/discarded; movement summary"),
        new EncoderField("Ships",0,68,"printed.identity","ship one-hot including unknown; fixed vocabulary slots, see schema.Ships"),
        new EncoderField("Ships",68,76,"printed.stats","faction4; points/400,hull/32,command/4,engineering/8,squadron/8; footprint metres; hull-zone existence and battery dice/10; flak/10; maneuver chart and availability"),
        new EncoderField("Ships",144,238,"visible.mutable","entity flags/coordinates/heading/objective; current/max hull and shields; first four queue slots6 (absent,unknown,4 commands); token+raid counts/4; full activation state; critical counts/4. All queue slots also have Commands rows."),
        new EncoderField("Ships",382,2,"scenario.kind","IsStation, IsScenarioUnit; explicit even for generated definitions outside the fleet catalog"),
        new EncoderField("Squads",0,109,"printed.identity","squadron one-hot including unknown"),
        new EncoderField("Squads",109,51,"printed.stats","faction4; cost,hull,speed,diameter,unique; battery/flak; keyword levels/5"),
        new EncoderField("Squads",160,96,"visible.mutable","entity flags/coordinates (no heading); hull,activation,command and each temporary flag; activation progress and pending destination"),
        new EncoderField("Cards",0,294,"printed.identity","printed upgrade one-hot including unknown"),
        new EncoderField("Cards",294,294,"effective.identity","effective upgrade one-hot including unknown; linked to physical card instance by row"),
        new EncoderField("Cards",588,52,"mutable","ready/exhausted/discarded; untyped+typed command counters/8; usage; focus zone; assigned-command visibility; stored die; scheduled rounds visibility. Private/public stores are separate linked rows."),
        new EncoderField("Pieces",0,64,"map-piece","presence/active; kind8 indexes StateEncoder.SpatialKinds (native0,1,2,3,100,101,102,10001); owner2; observer-relative pose; size; flip; objective hull/destroyed; payload codec mask. Identity/shape symbols and source/host indices are integer columns."),
        new EncoderField("Tokens",0,16,"physical-defense-slot","type6; present,ready,exhausted,discarded,locked,spent-this-attack; owner/physical slot in integer columns"),
        new EncoderField("Dice",0,24,"die","present; red/blue/black one-hot3 (native None is all zero); face6 only after roll; rolled/pending/effect-selected/effect-eligible; parent attack and physical slot in integer columns"),
        new EncoderField("Damages",0,32,"damage-card","present,faceup,known; identity23 (unknown plus22); owner/index in integer columns. Never exports an unknown facedown identity."),
        new EncoderField("Commands",0,8,"queued-dial","present,known,command4,own,enemy-seen (known and enemy-owned after placeholder masking); ship/queue-slot in integer columns. Search materialization/boundary status is never an input feature. An encoder-only boundary overlay may allow a materialized hypothetical owner's dial without marking it legally seen by the root observer."),
        new EncoderField("KnownDeckPeeks",0,24,"remembered-deck-prefix","present; 22-card identity; deck/position in integer columns. Unseen draw order/RNG is never read."),
        new EncoderField("Attacks",0,192,"attack-stack","complete public attack scalars, flags, categorical phases/ranges/zones; pending faceup damage identities; redirects/locks/dice have linked rows. Internal effect payloads have typed side rows or explicit coverage issues."),
        new EncoderField("Effects",0,32,"effect-record","kind one-hot16; visibility/known flags; scalar/categorical payload only where codec is registered. Rule/text vocabulary, source/target/slot/parent integer columns retain semantic bindings, never raw float IDs."),
        new EncoderField("Objective",0,64,"objective","victory tokens, invalid-setup status; entity/piece data is in linked rows; named flags and timing continuations are effect records"),
        new EncoderField("Stores",0,16,"card-store","kind4; known/present; command4 or round/6; card/entity/owner/slot integer columns. Opponent private command values are masked."),
    });
    public StateEncoderSchema(FleetData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        Ships = new(data.ShipEntries.Select(x => x.Definition.ShipId));
        Squadrons = new(data.SquadronEntries.Select(x => x.Definition.SquadronId));
        Upgrades = new(UpgradeCatalog.Cards.Select(x => x.id));
        Objectives = new(data.Objectives.Select(x => x.Id));
        Keywords = new(data.SquadronEntries.SelectMany(x => x.Definition.Keywords).Select(x => x.Split(' ')[0].ToLowerInvariant()));
        Symbols = new(StateEncoderSymbols.Values);
        SymbolWords = new(StateEncoderSymbols.Values.SelectMany(value => Regex.Matches(value, @"[\p{L}_][\p{L}\p{N}_]*|[^\w\s]", RegexOptions.CultureInvariant).Select(match => match.Value)).Concat(["#integer", "#entity"]));
        if (Ships.Values.Count > 68 || Squadrons.Values.Count > 109 || Upgrades.Values.Count > 294 || Objectives.Values.Count > 37 || Keywords.Values.Count > 25)
            throw new NotSupportedException("Catalog exceeds the versioned encoder identity blocks. Expand schema/version explicitly; never truncate.");
        Fingerprint = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(Version + "\n" +
            string.Join("\n", Ships.Values.Concat(Squadrons.Values).Concat(Upgrades.Values).Concat(Objectives.Values).Concat(Keywords.Values).Concat(Symbols.Values).Concat(SymbolWords.Values)))));
    }
}

public enum EncodedEffectKind { Continuation, AttackFlag, FlowFlag, ObjectiveFlag, RoundCommand, PalpatineToken, PalpatineRemaining, Movement, MovementDamage, Sequence, EntityLink, TokenHost, DeathTrigger, Register, AdditionalDial, ArrayItem }

/// <summary>Owned arrays when cloned; otherwise a writer workspace reused by the next Encode call.</summary>
public sealed class StateEncoding
{
    public string SchemaFingerprint { get; internal set; } = "";
    public int Observer { get; internal set; }
    public int Actor { get; internal set; }
    public float[] Global { get; internal set; } = new float[StateEncoder.GlobalWidth];
    public float[] Ships { get; internal set; } = [];
    public float[] Squads { get; internal set; } = [];
    public float[] Cards { get; internal set; } = [];
    public float[] Pieces { get; internal set; } = [];
    public float[] Tokens { get; internal set; } = [];
    public float[] Dice { get; internal set; } = [];
    public float[] Damages { get; internal set; } = [];
    public float[] Commands { get; internal set; } = [];
    public float[] KnownDeckPeeks { get; internal set; } = [];
    public float[] Attacks { get; internal set; } = [];
    public float[] Effects { get; internal set; } = [];
    public float[] Objective { get; internal set; } = new float[StateEncoder.ObjectiveWidth];
    public float[] Stores { get; internal set; } = [];
    public int[] CardHosts { get; internal set; } = [];
    public int[] SquadCommanders { get; internal set; } = [];
    public int[] PieceReferences { get; internal set; } = []; // source entity, shape symbol, id symbol, objective entity, host piece
    public int[] TokenReferences { get; internal set; } = []; // entity, type, physical slot
    public int[] DieReferences { get; internal set; } = []; // attack, physical slot
    public int[] DamageReferences { get; internal set; } = []; // entity, physical slot
    public int[] CommandReferences { get; internal set; } = []; // entity, queue slot
    public int[] PeekReferences { get; internal set; } = []; // observer-relative deck owner, slot
    public int[] AttackReferences { get; internal set; } = []; // attacker,defender,mine piece,intel entity,token type,slot
    public int[] EffectReferences { get; internal set; } = []; // kind,rule symbol,text symbol,source,target,slot,parent,aux
    public int[] StoreReferences { get; internal set; } = []; // card,entity,relative owner,slot
    /// <summary>sequence, ordinal, finite word category, exact integer argument (-1 absent).
    /// Symbol columns elsewhere: positive is Symbols index, zero is empty/unknown,
    /// -1 is padding, and -(sequence+2) refers to these compositional rows.</summary>
    public int[] SymbolTokens { get; internal set; } = [];
    public EncoderCoverageIssue[] CoverageIssues { get; internal set; } = [];
    public IReadOnlyList<EncodedTensor> Tensors { get; private set; } = [];
    public bool Complete => CoverageIssues.Length == 0;
    public int ShipCount => Ships.Length / StateEncoder.ShipWidth;
    public int SquadronCount => Squads.Length / StateEncoder.SquadWidth;
    public int UpgradeCount => Cards.Length / StateEncoder.CardWidth;
    public int PieceCount => Pieces.Length / StateEncoder.PieceWidth;
    public int TokenCount => Tokens.Length / StateEncoder.TokenWidth;
    public int DieCount => Dice.Length / StateEncoder.DieWidth;
    public int DamageCount => Damages.Length / StateEncoder.DamageWidth;
    public int CommandCount => Commands.Length / StateEncoder.CommandWidth;
    public int PeekCount => KnownDeckPeeks.Length / StateEncoder.PeekWidth;
    public int AttackCount => Attacks.Length / StateEncoder.AttackWidth;
    public int EffectCount => Effects.Length / StateEncoder.EffectWidth;
    public int StoreCount => Stores.Length / StateEncoder.StoreWidth;
    public long FloatElementCount => (long)Global.Length + Ships.Length + Squads.Length + Cards.Length + Pieces.Length + Tokens.Length + Dice.Length + Damages.Length + Commands.Length + KnownDeckPeeks.Length + Attacks.Length + Effects.Length + Objective.Length + Stores.Length;
    public long IntegerElementCount => (long)CardHosts.Length + SquadCommanders.Length + PieceReferences.Length + TokenReferences.Length + DieReferences.Length + DamageReferences.Length + CommandReferences.Length + PeekReferences.Length + AttackReferences.Length + EffectReferences.Length + StoreReferences.Length + SymbolTokens.Length;
    public long ByteSize => checked(4 * (FloatElementCount + IntegerElementCount));
    public StateEncoding Clone()
    {
        var copy = (StateEncoding)MemberwiseClone();
        copy.Global = (float[])Global.Clone(); copy.Ships = (float[])Ships.Clone(); copy.Squads = (float[])Squads.Clone(); copy.Cards = (float[])Cards.Clone(); copy.Pieces = (float[])Pieces.Clone(); copy.Tokens = (float[])Tokens.Clone(); copy.Dice = (float[])Dice.Clone(); copy.Damages = (float[])Damages.Clone(); copy.Commands = (float[])Commands.Clone(); copy.KnownDeckPeeks = (float[])KnownDeckPeeks.Clone(); copy.Attacks = (float[])Attacks.Clone(); copy.Effects = (float[])Effects.Clone(); copy.Objective = (float[])Objective.Clone(); copy.Stores = (float[])Stores.Clone();
        copy.CardHosts = (int[])CardHosts.Clone(); copy.SquadCommanders = (int[])SquadCommanders.Clone(); copy.PieceReferences = (int[])PieceReferences.Clone(); copy.TokenReferences = (int[])TokenReferences.Clone(); copy.DieReferences = (int[])DieReferences.Clone(); copy.DamageReferences = (int[])DamageReferences.Clone(); copy.CommandReferences = (int[])CommandReferences.Clone(); copy.PeekReferences = (int[])PeekReferences.Clone(); copy.AttackReferences = (int[])AttackReferences.Clone(); copy.EffectReferences = (int[])EffectReferences.Clone(); copy.StoreReferences = (int[])StoreReferences.Clone(); copy.SymbolTokens = (int[])SymbolTokens.Clone(); copy.CoverageIssues = (EncoderCoverageIssue[])CoverageIssues.Clone(); copy.RefreshTensors(); return copy;
    }

    internal void RefreshTensors()
    {
        // Retain descriptors while the dynamic shapes and backing arrays stay unchanged.
        if (Tensors.Count > 0 && ReferenceEquals(Tensors[1].Data, Ships) &&
            ReferenceEquals(Tensors[2].Data, Squads) && ReferenceEquals(Tensors[3].Data, Cards) &&
            ReferenceEquals(Tensors[4].Data, Pieces) && ReferenceEquals(Tensors[5].Data, Tokens) &&
            ReferenceEquals(Tensors[6].Data, Dice) && ReferenceEquals(Tensors[7].Data, Damages) &&
            ReferenceEquals(Tensors[8].Data, Commands) && ReferenceEquals(Tensors[9].Data, KnownDeckPeeks) &&
            ReferenceEquals(Tensors[10].Data, Attacks) && ReferenceEquals(Tensors[11].Data, Effects) &&
            ReferenceEquals(Tensors[13].Data, Stores) && ReferenceEquals(Tensors[25].Data, SymbolTokens) &&
            ReferenceEquals(Tensors[14].Data, CardHosts) && ReferenceEquals(Tensors[15].Data, SquadCommanders) &&
            ReferenceEquals(Tensors[16].Data, PieceReferences) && ReferenceEquals(Tensors[17].Data, TokenReferences) &&
            ReferenceEquals(Tensors[18].Data, DieReferences) && ReferenceEquals(Tensors[19].Data, DamageReferences) &&
            ReferenceEquals(Tensors[20].Data, CommandReferences) && ReferenceEquals(Tensors[21].Data, PeekReferences) &&
            ReferenceEquals(Tensors[22].Data, AttackReferences) && ReferenceEquals(Tensors[23].Data, EffectReferences) &&
            ReferenceEquals(Tensors[24].Data, StoreReferences)) return;
        Tensors = Array.AsReadOnly(new EncodedTensor[]
        {
            new("core.global", [StateEncoder.GlobalWidth], Global),
            new("core.ships", [ShipCount, StateEncoder.ShipWidth], Ships),
            new("core.squadrons", [SquadronCount, StateEncoder.SquadWidth], Squads),
            new("core.upgrades", [UpgradeCount, StateEncoder.CardWidth], Cards),
            new("core.pieces", [PieceCount, StateEncoder.PieceWidth], Pieces),
            new("core.defense_tokens", [TokenCount, StateEncoder.TokenWidth], Tokens),
            new("core.dice", [DieCount, StateEncoder.DieWidth], Dice),
            new("core.damage_cards", [DamageCount, StateEncoder.DamageWidth], Damages),
            new("core.commands", [CommandCount, StateEncoder.CommandWidth], Commands),
            new("core.deck_peeks", [PeekCount, StateEncoder.PeekWidth], KnownDeckPeeks),
            new("core.attacks", [AttackCount, StateEncoder.AttackWidth], Attacks),
            new("core.effects", [EffectCount, StateEncoder.EffectWidth], Effects),
            new("core.objective", [StateEncoder.ObjectiveWidth], Objective),
            new("core.stores", [StoreCount, StateEncoder.StoreWidth], Stores),
            new("core.card_hosts", [UpgradeCount], CardHosts),
            new("core.squadron_commanders", [SquadronCount], SquadCommanders),
            new("core.piece_references", [PieceCount, 5], PieceReferences),
            new("core.token_references", [TokenCount, 3], TokenReferences),
            new("core.die_references", [DieCount, 2], DieReferences),
            new("core.damage_references", [DamageCount, 2], DamageReferences),
            new("core.command_references", [CommandCount, 2], CommandReferences),
            new("core.peek_references", [PeekCount, 2], PeekReferences),
            new("core.attack_references", [AttackCount, 6], AttackReferences),
            new("core.effect_references", [EffectCount, 8], EffectReferences),
            new("core.store_references", [StoreCount, 4], StoreReferences),
            new("core.symbol_tokens", [SymbolTokens.Length / 4, 4], SymbolTokens),
        });
    }
}
