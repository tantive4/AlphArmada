using System.Globalization;
using System.Numerics;
using Armada.Core;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Runtime;
using Swarmada.Headless.Setup;

namespace Swarmada.Headless.Search;

public sealed record PolicyChoice(string Id, string Label, int Entity = -1, int Code = -1);
public sealed record PolicyDecision(DecisionKind Kind, int Actor, string Stage,
    IReadOnlyList<PolicyChoice> Choices, GamePhase Phase, TerminalOutcome? Outcome = null,
    InvalidSetupResult? InvalidSetup = null, IReadOnlyList<PolicyCandidateDescriptor>? Candidates = null);

[Flags]
public enum PolicyActionFields
{
    Kind = 1, Source = 2, Target = 4, FromZone = 8, ToZone = 16, Command = 32,
    SpendMode = 64, Token = 128, Index = 256, Amount = 512, Position = 1024,
    Yaw = 2048, Speed = 4096, Course = 8192, DiceIndices = 16384, Dice = 32768,
    Ability = 65536, Flipped = 131072, All = 262143
}

/// <summary>Fields shared by every complete legal action represented by one
/// conditional choice. Unselected arguments are masked, never guessed from an example.</summary>
public sealed record PolicyCandidateDescriptor(GameAction Example, PolicyActionFields KnownFields, int CompleteActions,
    PublicProviderArgument? Provider = null);

/// <summary>Recoverable conditional argument trie over exact legal native actions.
/// Only Select at a complete leaf mutates the engine. This is worker-local state.
/// No engine snapshots or speculative actions are retained in individual prefix nodes.</summary>
public sealed class DecisionAdapter
{
    internal sealed class Node(string stage)
    {
        public string Stage { get; set; } = stage;
        private SortedDictionary<string, Node>? children;
        private Dictionary<string, int>? entities;
        // Complete native-action leaves need neither map. Spatial domains can
        // contain hundreds of such leaves at each physical state.
        public SortedDictionary<string, Node> Children => children ??= new(StringComparer.Ordinal);
        public Dictionary<string, int> Entities => entities ??= new(StringComparer.Ordinal);
        public GameAction? Action { get; init; }
        public PolicyDecision? Decision { get; set; }
        public Node[]? IndexedChildren { get; set; }
        public GameAction[]? PendingActions { get; set; }
        public GameAction[]? CatalogActions { get; set; }
        public ActionCodec? Codec { get; set; }
        public PolicyCandidateDescriptor[]? CandidateDescriptors { get; set; }
        public IReadOnlyDictionary<GameAction, PublicProviderArgument>? ProviderArguments { get; init; }
        public int Depth { get; init; }
    }
    internal sealed record Field(string Stage, Func<GameAction, string> Value, Func<GameAction, int>? Entity = null);
    /// <summary>Opaque worker-local cached argument descriptor. The caller must
    /// establish exact rule/knowledge/prefix equivalence before binding it to a
    /// rebuilt state. It retains exact native payloads, not engine snapshots.
    /// Prefix descriptors share the same lazily materialized physical-state trie.</summary>
    public sealed class DecisionDescriptor
    {
        internal readonly GameDecision Native;
        internal readonly Node Root, Cursor;
        internal readonly Field[] Fields;
        internal readonly string[] Prefix;
        internal readonly int AtomicCount;
        internal readonly int Ships, Entities;
        /// <summary>Diagnostic/export reconstruction table. Created only on
        /// demand; normal cached traversal needs only numeric edge indices.</summary>
        public ActionCodec GetCodec() => Root.Codec ??= new(Root.CatalogActions ?? []);
        internal DecisionDescriptor(GameDecision native, Node root, Node cursor,
            Field[] fields, string[] prefix, int count, int ships, int entities)
        { Native = native; Root = root; Cursor = cursor; Fields = fields; Prefix = prefix;
          AtomicCount = count; Ships = ships; Entities = entities; }
    }
    private readonly ActionCatalog catalog;
    private readonly List<string> prefix = [];
    private readonly IReadOnlyList<string> prefixView;
    private GameDecision? native;
    private Node? root, cursor;
    private Field[] fields = [];
    private DecisionDescriptor? captured;
    private long revision = -1, generation = -1;
    public GameSession Game { get; }
    public GameAction? LastCommittedAction { get; private set; }
    public long CandidatesValidated => catalog.CandidatesValidated;
    public long CatalogsBuilt => catalog.CatalogsBuilt;
    public int AtomicActionCount { get; private set; }
    public int PrefixCount => prefix.Count;
    public IReadOnlyList<string> Prefix => prefixView;

    public DecisionAdapter(GameSession game, ActionCatalogProfile? profile = null)
    { Game = game ?? throw new ArgumentNullException(nameof(game)); catalog = new(profile); prefixView = prefix.AsReadOnly(); }

    public PolicyDecision Current
    {
        get
        {
            Ensure(); var decision = native!;
            if (decision.Kind != DecisionKind.Player)
                return new(decision.Kind, decision.Player, decision.Kind.ToString().ToLowerInvariant(), [],
                    decision.Phase, decision.Outcome, decision.InvalidSetup);
            var node = cursor!; Materialize(node);
            if (node.Decision is null)
            {
                var choices = new PolicyChoice[node.Children.Count]; node.IndexedChildren = new Node[choices.Length];
                int index = 0;
                foreach (var pair in node.Children)
                {
                    choices[index] = new(pair.Key, pair.Key, node.Entities.GetValueOrDefault(pair.Key, -1), index);
                    node.IndexedChildren[index++] = pair.Value;
                }
                node.Decision = new(decision.Kind, decision.Player, node.Stage, Array.AsReadOnly(choices), decision.Phase);
            }
            return node.Decision;
        }
    }

    /// <returns>True only when a complete native action was committed.</returns>
    public bool Select(string id)
    {
        ArgumentNullException.ThrowIfNull(id); Ensure();
        if (native!.Kind == DecisionKind.Player) Materialize(cursor!);
        if (native!.Kind != DecisionKind.Player || !cursor!.Children.TryGetValue(id, out var next))
            throw new ArgumentException("Choice is not legal at the current conditional stage: " + id, nameof(id));
        return SelectNext(id, next);
    }

    /// <summary>Node-local numeric edge selection. The code is meaningful only
    /// together with its descriptor; it is not a persistent global action ID.</summary>
    public bool Select(int index)
    {
        var decision = Current;
        if (decision.Kind != DecisionKind.Player || (uint)index >= (uint)decision.Choices.Count)
            throw new ArgumentOutOfRangeException(nameof(index));
        return SelectNext(decision.Choices[index].Id, cursor!.IndexedChildren![index]);
    }
    public bool Commits(int index)
    {
        var decision = Current;
        if (decision.Kind != DecisionKind.Player || (uint)index >= (uint)decision.Choices.Count)
            throw new ArgumentOutOfRangeException(nameof(index));
        return cursor!.IndexedChildren![index].Action is not null;
    }
    public bool Commits(string id)
    {
        ArgumentNullException.ThrowIfNull(id); Ensure();
        if (native!.Kind == DecisionKind.Player) Materialize(cursor!);
        if (native!.Kind != DecisionKind.Player || !cursor!.Children.TryGetValue(id, out var next))
            throw new ArgumentException("Choice is not legal at the current conditional stage: " + id, nameof(id));
        return next.Action is not null;
    }
    private bool SelectNext(string id, Node next)
    {
        if (next.Action is { } action)
        {
            // Native Apply rechecks the current rule offer. Cached descriptor
            // binding never bypasses the engine's authoritative legality guard.
            Game.Apply(action.DeepClone()); ResetPrefix(); LastCommittedAction = action.DeepClone(); return true;
        }
        prefix.Add(id); cursor = next; captured = null; return false;
    }

    public DecisionDescriptor CaptureDescriptor()
    {
        _ = Current;
        if (native!.Kind != DecisionKind.Player)
            throw new InvalidOperationException("Only player decisions have a reusable argument descriptor.");
        return captured ??= new(native!, root!, cursor!, fields, prefix.ToArray(), AtomicActionCount,
            Game.Context.State.Ships.Length, Game.Context.Topology.EntityCount);
    }

    public IReadOnlyList<PolicyCandidateDescriptor> DescribeChoices()
    {
        _ = Current;
        if (native!.Kind != DecisionKind.Player) return [];
        var node = cursor!;
        if (node.CandidateDescriptors is not null) return node.CandidateDescriptors;
        return node.CandidateDescriptors = node.IndexedChildren!.Select(Describe).ToArray();
    }

    private static IEnumerable<GameAction> ActionsUnder(Node node)
    {
        if (node.Action is { } action) { yield return action; yield break; }
        if (node.PendingActions is { } pending) { foreach (var item in pending) yield return item; yield break; }
        foreach (var child in node.Children.Values) foreach (var item in ActionsUnder(child)) yield return item;
    }

    private static PolicyCandidateDescriptor Describe(Node node)
    {
        using var actions = ActionsUnder(node).GetEnumerator();
        if (!actions.MoveNext()) throw new InvalidOperationException("Conditional candidate contains no native actions.");
        var first = actions.Current; var mask = PolicyActionFields.All; int count = 1;
        PublicProviderArgument? provider = node.ProviderArguments?.GetValueOrDefault(first);
        static bool Equal<T>(T[]? a, T[]? b) => a is null ? b is null : b is not null && a.SequenceEqual(b);
        while (actions.MoveNext())
        {
            count++; var next = actions.Current;
            provider = PublicProviderArgument.Merge(provider, node.ProviderArguments?.GetValueOrDefault(next));
            if (first.Kind != next.Kind) mask &= ~PolicyActionFields.Kind;
            if (first.Source != next.Source) mask &= ~PolicyActionFields.Source;
            if (first.Target != next.Target) mask &= ~PolicyActionFields.Target;
            if (first.FromZone != next.FromZone) mask &= ~PolicyActionFields.FromZone;
            if (first.ToZone != next.ToZone) mask &= ~PolicyActionFields.ToZone;
            if (first.Command != next.Command) mask &= ~PolicyActionFields.Command;
            if (first.SpendMode != next.SpendMode) mask &= ~PolicyActionFields.SpendMode;
            if (first.Token != next.Token) mask &= ~PolicyActionFields.Token;
            if (first.Index != next.Index) mask &= ~PolicyActionFields.Index;
            if (first.Amount != next.Amount) mask &= ~PolicyActionFields.Amount;
            if (first.Position != next.Position) mask &= ~PolicyActionFields.Position;
            if (BitConverter.SingleToInt32Bits(first.YawDegrees) != BitConverter.SingleToInt32Bits(next.YawDegrees)) mask &= ~PolicyActionFields.Yaw;
            if (first.Speed != next.Speed) mask &= ~PolicyActionFields.Speed;
            if (!Equal(first.Course, next.Course)) mask &= ~PolicyActionFields.Course;
            if (!Equal(first.DiceIndices, next.DiceIndices)) mask &= ~PolicyActionFields.DiceIndices;
            if (!Equal(first.Dice, next.Dice)) mask &= ~PolicyActionFields.Dice;
            if (first.AbilityId != next.AbilityId) mask &= ~PolicyActionFields.Ability;
            if (first.Flipped != next.Flipped) mask &= ~PolicyActionFields.Flipped;
        }
        return new(first.DeepClone(), mask, count, provider);
    }
    public void BindDescriptor(DecisionDescriptor descriptor)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        if (descriptor.Ships != Game.Context.State.Ships.Length || descriptor.Entities != Game.Context.Topology.EntityCount)
            throw new ArgumentException("Cached descriptor has a different entity topology.", nameof(descriptor));
        native = descriptor.Native; root = descriptor.Root; cursor = descriptor.Cursor; fields = descriptor.Fields;
        prefix.Clear(); prefix.AddRange(descriptor.Prefix); AtomicActionCount = descriptor.AtomicCount; captured = descriptor;
        revision = Game.Context.State.Revision; generation = Game.PositionGeneration; LastCommittedAction = null;
    }

    public string[] CapturePrefix() => prefix.ToArray();
    /// <summary>Discard buffered intent and invalidate catalog after external CopyFrom,
    /// RestoreSnapshot or sampled chance. No game state is changed.</summary>
    public void ResetPrefix()
    { prefix.Clear(); native = null; root = cursor = null; fields = []; captured = null; revision = generation = -1; AtomicActionCount = 0; LastCommittedAction = null; }

    public void RestorePrefix(IEnumerable<string> saved)
    {
        ArgumentNullException.ThrowIfNull(saved); var values = saved.ToArray(); ResetPrefix(); Ensure();
        foreach (var id in values)
        {
            if (native!.Kind == DecisionKind.Player) Materialize(cursor!);
            if (native!.Kind != DecisionKind.Player || !cursor!.Children.TryGetValue(id, out var next) || next.Action is not null)
            { ResetPrefix(); throw new ArgumentException("Saved prefix is invalid or would commit a native action.", nameof(saved)); }
            prefix.Add(id); cursor = next; captured = null;
        }
    }

    private void Ensure()
    {
        if (native is not null && revision == Game.Context.State.Revision && generation == Game.PositionGeneration) return;
        if (prefix.Count != 0) throw new InvalidOperationException("Engine changed while a partial action was buffered; restore/reset its prefix explicitly.");
        native = Game.GetDecision();
        if (native.Kind == DecisionKind.Player)
        {
            var actions = catalog.Build(Game, native); AtomicActionCount = actions.Count;
            Dictionary<GameAction, PublicProviderArgument>? providers = null;
            foreach (var action in actions)
                if (PublicProviderArguments.TryResolve(Game.Context, action.AbilityId, out var provider))
                    (providers ??= new(ExactGameActionComparer.Instance)).Add(action, provider);
            fields = Fields(actions, providers).ToArray(); var candidates = actions.ToArray(); root = Build(candidates, 0, providers); cursor = root;
            if (root.Action is { } only)
            {
                root = new(Family(native.Phase) + "/select_action") { ProviderArguments = providers };
                root.Children.Add(Operation(only), new("commit") { Action = only, ProviderArguments = providers }); cursor = root;
            }
            root.CatalogActions = candidates;
        }
        revision = Game.Context.State.Revision; generation = Game.PositionGeneration;
    }

    private Node Build(GameAction[] actions, int depth, IReadOnlyDictionary<GameAction, PublicProviderArgument>? providers)
    {
        if (actions.Length == 1) return new("commit") { Action = actions[0], ProviderArguments = providers };
        return new("pending") { PendingActions = actions, Depth = depth, ProviderArguments = providers };
    }

    private void Materialize(Node node)
    {
        if (node.PendingActions is not { } actions) return;
        int depth = node.Depth;
        while (depth < fields.Length)
        {
            var field = fields[depth]; var groups = actions.GroupBy(field.Value, StringComparer.Ordinal).ToArray();
            if (groups.Length == 1) { depth++; continue; }
            string stage = field.Stage;
            if (field.Entity is not null)
            {
                int[] entities = actions.Select(field.Entity).Distinct().ToArray();
                string type = entities.All(Game.Context.IsShip) ? "ship_pointer" :
                    entities.All(entity => entity >= Game.Context.State.Ships.Length && entity < Game.Context.Topology.EntityCount)
                        ? "squadron_pointer" : "entity_pointer";
                stage = type + stage[stage.IndexOf('/')..];
            }
            // Stage is fixed when this descriptor is first visited, not while
            // recursively expanding every descendant argument combination.
            node.Stage = stage;
            foreach (var group in groups)
            {
                var entries = group.ToArray(); node.Children.Add(group.Key, Build(entries, depth + 1, node.ProviderArguments));
                if (field.Entity is not null) node.Entities.Add(group.Key, field.Entity(entries[0]));
            }
            node.PendingActions = null; return;
        }
        throw new SearchCoverageException("Distinct native actions alias the same complete argument sequence.");
    }

    private List<Field> Fields(IReadOnlyList<GameAction> actions, IReadOnlyDictionary<GameAction, PublicProviderArgument>? providers)
    {
        string family = Family(native!.Phase);
        // Freeze only the timing discriminator. Cached selectors must not consult
        // an adapter's later native decision/flow, and retaining a label dictionary
        // for every spatial candidate would unnecessarily inflate node memory.
        bool commNoise = IsCommNoise();
        var fields = new List<Field> { new(family + "/select_action", action => Operation(action, commNoise)) };
        // Numeric equality rejects constant argument dimensions before allocating
        // any formatted keys. Movement therefore avoids scanning/formatting the
        // command, payment, hull-zone, token, and other unrelated argument fields.
        void Add<T>(string stage, Func<GameAction, T> value, Func<GameAction, string> format,
            Func<GameAction, int>? entity = null)
        {
            T first = value(actions[0]); var comparer = EqualityComparer<T>.Default;
            for (int i = 1; i < actions.Count; i++)
                if (!comparer.Equals(first, value(actions[i]))) { fields.Add(new(stage, format, entity)); return; }
        }
        Add("ship_pointer/select_source", a => a.Source, a => "source:" + a.Source, a => a.Source);
        Add("command/select_command", a => a.Command, a => "command:" + (int)a.Command);
        Add("ship_pointer/select_target", a => a.Target, a => "target:" + a.Target, a => a.Target);
        if (providers is not null)
            Add("entity_pointer/select_target", a => providers.GetValueOrDefault(a)?.ProviderEntity ?? -1,
                a => "target:" + (providers.GetValueOrDefault(a)?.ProviderEntity ?? -1),
                a => providers.GetValueOrDefault(a)?.ProviderEntity ?? -1);
        Add("payment/select_mode", a => a.SpendMode, a => "payment:" + (int)a.SpendMode);
        Add("zone/select_source_zone", a => a.FromZone, a => "from_zone:" + (int)a.FromZone);
        Add("zone/select_target_zone", a => a.ToZone, a => "to_zone:" + (int)a.ToZone);
        int Token(GameAction action) => providers?.GetValueOrDefault(action)?.TokenType is { } token ? (int)token : (int)action.Token;
        Add("token_pointer/select_token_type", Token, a => "token:" + Token(a));
        // Draven's unordered two-token effect is one native action. Its second
        // type is transported as Amount=command+1, but uses the same conditional
        // command head as the first type, rather than a numeric amount head.
        bool commandPair = actions.Any(a => a.AbilityId == DravenAbilityEffects.PairAction);
        Add(commandPair ? "command/select_command" : "amount/select_amount", a => a.Amount,
            a => commandPair && a.AbilityId == DravenAbilityEffects.PairAction ? "command:" + (a.Amount - 1) : "amount:" + a.Amount);
        Add("speed/select_speed", a => a.Speed, a => "speed:" + a.Speed);
        Add("spatial/select_destination", a => a.Position, a => "position:" + Position(a.Position));
        Add("heading/select_heading", a => a.YawDegrees, a => "heading:" + F(a.YawDegrees));
        Add("face/select_face", a => a.Flipped, a => "face:" + (a.Flipped ? "1" : "0"));
        int joints = actions.Max(a => a.Course?.Length ?? 0);
        for (int joint = 0; joint < joints; joint++)
        {
            int index = joint;
            Add("course/select_joint_" + joint, a => a.Course is { } c && index < c.Length ? c[index] : int.MinValue,
                a => "click:" + (a.Course is { } c && index < c.Length ? c[index].ToString(CultureInfo.InvariantCulture) : "_"));
        }
        Add(joints > 0 ? "course/select_tool_side" : "instance/select_index", a => a.Index, a => "index:" + a.Index);
        int[] dice = actions.SelectMany(a => a.DiceIndices ?? []).Distinct().Order().ToArray();
        foreach (int die in dice)
            Add("dice_subset/select_die_" + die, a => a.DiceIndices?.Contains(die) == true,
                a => "include:" + (a.DiceIndices?.Contains(die) == true ? "1" : "0"));
        // Exact final identities preserve unusual native payloads. They intentionally
        // remain visible as atomic-native choices until an explicit effect decoder is
        // registered; numeric suffixes are never stripped heuristically.
        fields.Add(new("native_atomic/select_payload", a => "native:" + ActionCatalog.Key(a)));
        return fields;
    }

    private bool IsCommNoise() => native!.Phase == GamePhase.ResolveCritical
        && Game.Context.Flow.Attacks.LastOrDefault()?.PendingDamageCard == CriticalDamageCard.CommNoise;
    private string Operation(GameAction action) => Operation(action, IsCommNoise());
    private static string Operation(GameAction action, bool commNoise)
    {
        if (action.AbilityId.StartsWith("reveal.comms-net.", StringComparison.Ordinal)) return "use:comms-net";
        if (PublicProviderArguments.TryFamily(action.AbilityId, out string family)) return "effect:" + action.Kind + ":" + family;
        if (commNoise)
            return action.Kind switch { GameActionKind.ChangeSpeed => "comm-noise:reduce-speed", GameActionKind.ChooseCommand => "comm-noise:replace-dial", _ => "comm-noise:decline" };
        return action.AbilityId.Length == 0 ? "action:" + action.Kind : "effect:" + action.Kind + ":" + action.AbilityId;
    }

    private string Family(GamePhase phase) => phase switch
    {
        GamePhase.CommandPlanning => "command_programming",
        GamePhase.RevealCommand or GamePhase.ResolveCommand or GamePhase.Repair or GamePhase.SquadronCommand => "command_window",
        GamePhase.SelectShip => "activation",
        GamePhase.SelectAttack => "attack_declaration",
        GamePhase.RollAttack or GamePhase.ModifyAttack => "attack_effects",
        GamePhase.SpendDefense or GamePhase.SpendAccuracy or GamePhase.EvadeDice or GamePhase.RedirectDamage => "defense",
        GamePhase.ResolveCritical or GamePhase.DealDamage or GamePhase.ChooseDamageCard => "damage",
        GamePhase.SelectCourse => "maneuver",
        GamePhase.SquadronActivation or GamePhase.SquadronPhase or GamePhase.SquadronMove or GamePhase.SquadronDisplacement => "squadron",
        GamePhase.AbilityChoice => TimingFamily(),
        _ => phase.ToString().ToLowerInvariant()
    };
    private string TimingFamily()
    {
        var frame = Game.Context.Flow.Continuations.LastOrDefault();
        if (frame is null) throw new SearchCoverageException("AbilityChoice has no recoverable timing context.");
        if (frame.Rule.StartsWith("general.reveal.", StringComparison.Ordinal) ||
            frame.Text is "before-reveal-command" or "before-command-reveal" or "command-reveal" or "after-command-reveal") return "command_window";
        if (frame.Rule.StartsWith("death.", StringComparison.Ordinal) || DamageEvents.Handles(frame.Rule)) return "damage";
        if (frame.Rule.StartsWith("movement.", StringComparison.Ordinal)) return "maneuver";
        if (frame.Rule.StartsWith("round.", StringComparison.Ordinal) || frame.Text is "start-round" or "start-ship-phase" or "start-squadron-phase" or "end-phase" or "status") return "boundary_window";
        return "timing_window";
    }
    private static string F(float value) => value.ToString("R", CultureInfo.InvariantCulture);
    private static string Position(Vector2? value) => value is { } v ? F(v.X) + "," + F(v.Y) : "_";
}
