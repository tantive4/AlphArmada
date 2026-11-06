using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using Armada.Rules;
using Swarmada.Headless.Engine;

namespace Swarmada.Headless.Search;

/// <summary>Exact native-action equality without diagnostic serialization. Arrays
/// are compared by contents; coordinates retain their original float bits.</summary>
public sealed class ExactGameActionComparer : IEqualityComparer<GameAction>
{
    public static ExactGameActionComparer Instance { get; } = new();
    public bool Equals(GameAction? a, GameAction? b)
    {
        if (ReferenceEquals(a, b)) return true;
        if (a is null || b is null) return false;
        return a.Kind == b.Kind && a.Source == b.Source && a.Target == b.Target && a.FromZone == b.FromZone
            && a.ToZone == b.ToZone && a.Command == b.Command && a.SpendMode == b.SpendMode && a.Token == b.Token
            && a.Index == b.Index && a.Amount == b.Amount && a.Speed == b.Speed && a.Flipped == b.Flipped
            && Bits(a.YawDegrees) == Bits(b.YawDegrees) && StringComparer.Ordinal.Equals(a.AbilityId, b.AbilityId)
            && a.Position.HasValue == b.Position.HasValue
            && (!a.Position.HasValue || Bits(a.Position.Value.X) == Bits(b.Position!.Value.X)
                && Bits(a.Position.Value.Y) == Bits(b.Position.Value.Y))
            && Equal(a.Course, b.Course) && Equal(a.DiceIndices, b.DiceIndices) && EqualDice(a.Dice, b.Dice);
    }
    public int GetHashCode(GameAction action)
    {
        var hash = new HashCode();
        hash.Add(action.Kind); hash.Add(action.Source); hash.Add(action.Target); hash.Add(action.FromZone); hash.Add(action.ToZone);
        hash.Add(action.Command); hash.Add(action.SpendMode); hash.Add(action.Token); hash.Add(action.Index); hash.Add(action.Amount);
        hash.Add(action.Speed); hash.Add(action.Flipped); hash.Add(Bits(action.YawDegrees)); hash.Add(action.AbilityId, StringComparer.Ordinal);
        hash.Add(action.Position.HasValue);
        if (action.Position is { } position) { hash.Add(Bits(position.X)); hash.Add(Bits(position.Y)); }
        Add(ref hash, action.Course); Add(ref hash, action.DiceIndices);
        hash.Add(action.Dice?.Length ?? -1);
        if (action.Dice is { } dice) foreach (var die in dice) { hash.Add(die.Color); hash.Add(die.Face); }
        return hash.ToHashCode();
    }
    private static int Bits(float value) => BitConverter.SingleToInt32Bits(value);
    private static bool Equal(int[]? a, int[]? b) => a is null ? b is null : b is not null && a.AsSpan().SequenceEqual(b);
    private static bool EqualDice(AttackDie[]? a, AttackDie[]? b)
    {
        if (a is null) return b is null;
        if (b is null || a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i].Color != b[i].Color || a[i].Face != b[i].Face) return false;
        return true;
    }
    private static void Add(ref HashCode hash, int[]? values)
    { hash.Add(values?.Length ?? -1); if (values is not null) foreach (int value in values) hash.Add(value); }
}

/// <summary>Dense action index, meaningful only with its exact versioned decision
/// catalogue. The catalogue retains original positions/courses and effect IDs.
/// This is a diagnostic/persistence codec, separate from a stale-decision token.</summary>
public readonly record struct CompactAction(int SchemaVersion, string CatalogSignature, int Index);
public sealed record ActionCodecTable(int SchemaVersion, string CatalogVersion, string CatalogSignature, GameAction[] Actions);

/// <summary>Lossless context table for compact action reconstruction. Retain or
/// persist the table with any exported IDs. Indices are never decoded against a
/// later unit position or a newly generated catalogue. No quantization is added.</summary>
public sealed class ActionCodec
{
    public const int SchemaVersion = 1;
    private readonly GameAction[] actions;
    private readonly Dictionary<GameAction, int> indices = new(ExactGameActionComparer.Instance);
    private readonly Lazy<string> signature;
    public string CatalogVersion { get; }
    public string CatalogSignature => signature.Value;
    public int Count => actions.Length;

    public ActionCodec(IEnumerable<GameAction> candidates, string catalogVersion = ActionCatalogProfile.Version)
    {
        ArgumentNullException.ThrowIfNull(candidates); ArgumentException.ThrowIfNullOrEmpty(catalogVersion);
        CatalogVersion = catalogVersion; actions = candidates.Select(action => action.DeepClone()).ToArray();
        for (int i = 0; i < actions.Length; i++)
            if (!indices.TryAdd(actions[i], i)) throw new ArgumentException("Codec catalogue contains duplicate exact actions.", nameof(candidates));
        // Diagnostic SHA/string work is lazy and never needed during normal
        // cached-tree selection, which uses node-local numeric edge indices.
        signature = new(() =>
        {
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            void Append(string value)
            {
                byte[] bytes = Encoding.UTF8.GetBytes(value);
                hash.AppendData(Encoding.ASCII.GetBytes(bytes.Length.ToString(CultureInfo.InvariantCulture) + ":"));
                hash.AppendData(bytes);
            }
            Append(SchemaVersion.ToString(CultureInfo.InvariantCulture)); Append(CatalogVersion);
            foreach (var action in actions) Append(ActionCatalog.Key(action));
            return Convert.ToHexString(hash.GetHashAndReset());
        });
    }
    public CompactAction Encode(GameAction action)
    {
        ArgumentNullException.ThrowIfNull(action);
        if (!indices.TryGetValue(action, out int index)) throw new ArgumentException("Action is absent from this exact catalogue.", nameof(action));
        return new(SchemaVersion, CatalogSignature, index);
    }
    public GameAction Decode(CompactAction compact)
    {
        if (compact.SchemaVersion != SchemaVersion || !StringComparer.Ordinal.Equals(compact.CatalogSignature, CatalogSignature))
            throw new ArgumentException("Compact action schema or catalogue does not match its saved context.", nameof(compact));
        if ((uint)compact.Index >= (uint)actions.Length) throw new ArgumentOutOfRangeException(nameof(compact));
        return actions[compact.Index].DeepClone();
    }
    public ActionCodecTable CaptureTable() => new(SchemaVersion, CatalogVersion, CatalogSignature,
        actions.Select(action => action.DeepClone()).ToArray());
    public static ActionCodec RestoreTable(ActionCodecTable table)
    {
        ArgumentNullException.ThrowIfNull(table);
        if (table.SchemaVersion != SchemaVersion) throw new ArgumentException("Unsupported compact action schema.", nameof(table));
        var codec = new ActionCodec(table.Actions, table.CatalogVersion);
        if (!StringComparer.Ordinal.Equals(codec.CatalogSignature, table.CatalogSignature))
            throw new ArgumentException("Saved compact action catalogue signature is invalid.", nameof(table));
        return codec;
    }
    /// <summary>Readable export for logs/UI. Private programmed commands are
    /// masked by default; callers may opt in only for an authorized observer or
    /// an explicitly authoritative debugging view. Normal selection builds no log.</summary>
    public string Describe(CompactAction compact, Func<int, string>? entityName = null, bool includePrivateCommands = false)
    {
        var action = Decode(compact); string Entity(int entity) => entityName?.Invoke(entity) ?? "entity " + entity;
        if (action.AbilityId.StartsWith("reveal.comms-net.", StringComparison.Ordinal))
            return Entity(action.Source) + ": Comms Net; transfer " + action.Command + " token to " + Entity(action.Target);
        string text = action.Source >= 0 ? Entity(action.Source) + ": " + action.Kind : action.Kind.ToString();
        if (action.Target >= 0) text += " → " + Entity(action.Target);
        if (action.Kind is GameActionKind.AssignCommand or GameActionKind.ChooseCommand)
            text += "; command " + (includePrivateCommands ? action.Command.ToString() : "unrevealed");
        else if (action.Kind is GameActionKind.BankCommand or GameActionKind.ResolveCommand)
            text += "; " + action.Command;
        if (action.Position is { } position) text += "; position (" + F(position.X) + ", " + F(position.Y) + ")";
        if (action.Course is { } course) text += "; course [" + string.Join(", ", course) + "]; side " + action.Index;
        if (action.AbilityId.Length != 0 && (includePrivateCommands || action.Kind is not (GameActionKind.AssignCommand or GameActionKind.ChooseCommand)))
            text += "; effect " + action.AbilityId;
        return text;
    }
    public static string DescribePrefix(DecisionAdapter.DecisionDescriptor descriptor, bool includePrivateCommands = false)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        bool privateContext = descriptor.Native.Phase == GamePhase.CommandPlanning || descriptor.Native.Actions.Any(option =>
            option.Template.Kind is GameActionKind.AssignCommand or GameActionKind.ChooseCommand);
        return DescribePrefixCore(descriptor.Prefix, descriptor.Cursor.Stage, includePrivateCommands, privateContext);
    }
    public static string DescribePrefix(IReadOnlyList<string> prefix, string pendingStage, bool includePrivateCommands = false)
    {
        ArgumentNullException.ThrowIfNull(prefix); ArgumentNullException.ThrowIfNull(pendingStage);
        return DescribePrefixCore(prefix, pendingStage, includePrivateCommands, prefix.Any(id => id.StartsWith("command:", StringComparison.Ordinal)));
    }
    private static string DescribePrefixCore(IReadOnlyList<string> prefix, string stage, bool includePrivateCommands, bool privateContext)
    {
        bool publicCommsToken = prefix.Contains("use:comms-net");
        var parts = new List<string>();
        foreach (string id in prefix)
        {
            if (id == "use:comms-net") parts.Add("Comms Net");
            else if (id.StartsWith("command:", StringComparison.Ordinal) && int.TryParse(id.AsSpan(8), out int command))
                parts.Add(includePrivateCommands || publicCommsToken ? ((ShipCommand)command) + " selected" : "command selected (unrevealed)");
            else if (!includePrivateCommands && privateContext && id.StartsWith("effect:", StringComparison.Ordinal)) parts.Add("effect selected");
            else parts.Add(id);
        }
        parts.Add(stage switch
        {
            "ship_pointer/select_target" => "recipient pending",
            "command/select_command" => "command pending",
            "spatial/select_destination" => "destination pending",
            _ => stage + " pending"
        });
        return string.Join("; ", parts);
    }
    private static string F(float value) => value.ToString("R", CultureInfo.InvariantCulture);
}
