using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Search;

namespace Swarmada.Headless.Learning;

/// <summary>Conditional policy input. Candidate rows describe legal choices only;
/// pointer indices and categorical symbols are separate from numeric features.</summary>
public sealed class PolicyEncoding
{
    public const int ContextWidth = 96, CandidateWidth = 112, ReferenceWidth = 6;
    public required string Head { get; init; }
    public required string Stage { get; init; }
    public required float[] Context { get; init; }
    public required float[] Candidates { get; init; }
    public required int[] CandidateReferences { get; init; }
    public required int[] PrefixSymbols { get; init; }
    public required float[] PrefixNumbers { get; init; }
    public required int[] PrefixOffsets { get; init; }
    public required int[] CandidateSymbols { get; init; }
    public required float[] CandidateNumbers { get; init; }
    public required int[] CandidateOffsets { get; init; }
    public required byte[] LegalMask { get; init; }
    public required float[] ArrayValues { get; init; }
    public required int[] ArrayReferences { get; init; }
    public int CandidateCount => LegalMask.Length;
    public IReadOnlyList<EncodedTensor> Tensors => [
        new("policy.context", [ContextWidth], Context),
        new("policy.candidates", [CandidateCount, CandidateWidth], Candidates),
        new("policy.references", [CandidateCount, ReferenceWidth], CandidateReferences),
        new("policy.prefix_symbols", [PrefixSymbols.Length], PrefixSymbols),
        new("policy.prefix_numbers", [PrefixNumbers.Length], PrefixNumbers),
        new("policy.prefix_offsets", [PrefixOffsets.Length], PrefixOffsets),
        new("policy.candidate_symbols", [CandidateSymbols.Length], CandidateSymbols),
        new("policy.candidate_numbers", [CandidateNumbers.Length], CandidateNumbers),
        new("policy.candidate_offsets", [CandidateOffsets.Length], CandidateOffsets),
        new("policy.legal_mask", [CandidateCount], LegalMask),
        new("policy.array_values", [ArrayValues.Length / 12, 12], ArrayValues),
        new("policy.array_references", [ArrayReferences.Length / 4, 4], ArrayReferences)
    ];
}

public static class PolicyEncoder
{
    public const string Version = "conditional-policy-v4";
    private static readonly string[] heads = ["activation", "command_programming", "command_window", "attack_declaration",
        "defense", "damage", "maneuver", "squadron", "timing_window", "boundary_window", "ship_pointer", "squadron_pointer",
        "entity_pointer", "command", "payment", "zone", "token_pointer", "amount", "speed", "spatial", "heading", "face",
        "course", "instance", "dice_subset", "native_atomic", "deployment", "status", "startround", "startshipphase", "attack_effects"];
    public static IReadOnlyList<string> Heads { get; } = Array.AsReadOnly(heads);
    public static string SchemaVersion => Version + ":" + PolicyEncoding.ContextWidth + ":" + PolicyEncoding.CandidateWidth
        + ":" + string.Join(',', heads) + ":" + EncoderSymbols.Fingerprint;

    public static PolicyEncoding Encode(GameSession position, PolicyDecision decision, IReadOnlyList<string> prefix)
    {
        if (decision.Kind != Runtime.DecisionKind.Player || decision.Stage.StartsWith("hidden_command/", StringComparison.Ordinal))
            throw new ArgumentException("Only actual player policy choices need neural candidate tensors.");
        string head = decision.Stage.Split('/')[0]; int headIndex = Array.IndexOf(heads, head);
        if (headIndex < 0) throw new SearchCoverageException("Register a shared policy head: " + head);
        var context = new float[PolicyEncoding.ContextWidth]; context[(int)decision.Phase] = 1;
        context[27 + headIndex] = 1; context[88] = prefix.Count / 16f;
        context[89] = position.Context.Topology.FirstPlayer == decision.Actor ? 1 : 0;
        context[90] = position.Context.Flow.CurrentPlayer == decision.Actor ? 1 : 0;
        var candidateFeatures = new float[decision.Choices.Count * PolicyEncoding.CandidateWidth];
        var references = new int[decision.Choices.Count * PolicyEncoding.ReferenceWidth]; Array.Fill(references, -1);
        var prefixSymbols = new List<int>(); var prefixNumbers = new List<float>(); var prefixOffsets = new List<int> { 0 };
        foreach (string identity in new[] { decision.Stage }.Concat(prefix)) Append(identity, prefixSymbols, prefixNumbers, prefixOffsets, position.Context, decision.Actor);
        // A unique static family may collapse before a provider/token pointer.
        // Retain the same public family context even when no static edge was
        // needed; the provider argument never replaces native combatants.
        string? commonFamily = null;
        if (decision.Candidates is { Count: > 0 } candidates)
        {
            commonFamily = Provider(position.Context, candidates[0])?.Family;
            if (commonFamily is not null && candidates.Any(candidate => Provider(position.Context, candidate)?.Family != commonFamily)) commonFamily = null;
        }
        if (commonFamily is not null) Append("effect:" + commonFamily, prefixSymbols, prefixNumbers, prefixOffsets, position.Context, decision.Actor);
        var symbols = new List<int>(); var numbers = new List<float>(); var offsets = new List<int> { 0 };
        var arrayValues = new List<float>(); var arrayReferences = new List<int>();
        for (int index = 0; index < decision.Choices.Count; index++)
        {
            var choice = decision.Choices[index];
            var row = candidateFeatures.AsSpan(index * PolicyEncoding.CandidateWidth, PolicyEncoding.CandidateWidth);
            var refs = references.AsSpan(index * PolicyEncoding.ReferenceWidth, PolicyEncoding.ReferenceWidth);
            row[0] = 1; refs[0] = choice.Entity;
            var descriptor = decision.Candidates?[index];
            if (descriptor is not null && descriptor.Provider is null && Provider(position.Context, descriptor) is { } provider)
                descriptor = descriptor with { Provider = provider };
            if (descriptor is not null)
            {
                WriteNative(position, decision.Actor, descriptor, row, refs);
                WriteArrays(index, descriptor, arrayValues, arrayReferences);
            }
            // Native fallback identities contain a diagnostic base64 serialization.
            // Encode its typed native fields and public effect identity instead.
            // A spatial choice already has exact normalized coordinates and its
            // native references in the typed candidate row. Re-tokenizing its
            // formatted engine identity would duplicate those coordinates and
            // run hundreds of regex/float parses for one movement head.
            string identity = decision.Stage.StartsWith("spatial/", StringComparison.Ordinal)
                && descriptor is { Example.Position: not null } && (descriptor.KnownFields & PolicyActionFields.Position) != 0 ? "position"
                : choice.Id.StartsWith("native:", StringComparison.Ordinal)
                ? descriptor is null ? throw new SearchCoverageException("Native policy payload lacks typed descriptor.")
                    : "native:" + descriptor.Example.Kind + ":" + descriptor.Example.AbilityId
                : choice.Id;
            Append(identity, symbols, numbers, offsets, position.Context, decision.Actor);
        }
        return new()
        {
            Head = head, Stage = decision.Stage, Context = context, Candidates = candidateFeatures,
            CandidateReferences = references, PrefixSymbols = prefixSymbols.ToArray(), PrefixNumbers = prefixNumbers.ToArray(),
            PrefixOffsets = prefixOffsets.ToArray(), CandidateSymbols = symbols.ToArray(), CandidateNumbers = numbers.ToArray(),
            CandidateOffsets = offsets.ToArray(), LegalMask = Enumerable.Repeat((byte)1, decision.Choices.Count).ToArray(),
            ArrayValues = arrayValues.ToArray(), ArrayReferences = arrayReferences.ToArray()
        };
    }

    private static void Append(string identity, List<int> symbols, List<float> numbers, List<int> offsets, GameContext game, int observer)
    {
        identity = PublicProviderArguments.NormalizeIdentity(game, identity);
        if (TryIntelTarget(identity, game, out var intel, out int entity, out int effectStart))
        {
            // The canonical native ID transports a public RuntimeUnitId as
            // base64url. That transport is neither a categorical rule symbol nor
            // a numeric model feature. Keep the public target/token arguments,
            // replacing only the opaque ID with its exact topology entity index.
            // This also handles effect-prefixed identities buffered by a caller.
            identity = identity[..effectStart] + UnitAbilityEngine.IntelOfficerActionPrefix
                + FormattableString.Invariant($"target:{(int)intel.UnitKind}:{entity}:{(int)intel.Token.Token}:{intel.Token.Slot}:{(intel.Token.Exhausted ? 1 : 0)}");
        }
        var encoded = EncoderSymbols.Encode(identity);
        // Public semantic identities retain exact engine coordinates outside the
        // tensors. Their numeric model channels use the same observer frame as
        // entity/candidate coordinates, including after buffering a destination.
        if (identity.StartsWith("position:", StringComparison.Ordinal))
        {
            int coordinate = 0; float sign = observer == 1 ? 1 : -1;
            for (int i = 0; i < encoded.Symbols.Length; i++) if (encoded.Symbols[i] == 1)
            {
                if (coordinate >= 2) throw new SearchCoverageException("Spatial policy identity has unexpected numeric fields.");
                encoded.Numbers[i] *= sign * 2 / (coordinate++ == 0 ? game.Topology.BoardSizeMillimetres.X : game.Topology.BoardSizeMillimetres.Y);
            }
        }
        else if (identity.StartsWith("heading:", StringComparison.Ordinal))
            for (int i = 0; i < encoded.Symbols.Length; i++) if (encoded.Symbols[i] == 1)
            {
                float degrees = encoded.Numbers[i] + (observer == 2 ? 180 : 0);
                encoded.Numbers[i] = ((degrees % 360 + 540) % 360 - 180) / 180;
            }
        symbols.AddRange(encoded.Symbols); numbers.AddRange(encoded.Numbers); offsets.Add(symbols.Count);
    }

    private static PublicProviderArgument? Provider(GameContext game, PolicyCandidateDescriptor descriptor)
        => descriptor.Provider ?? ((descriptor.KnownFields & PolicyActionFields.Ability) != 0
            && PublicProviderArguments.TryResolve(game, descriptor.Example.AbilityId, out var provider) ? provider : null);

    private static bool TryIntelTarget(string identity, GameContext game, out IntelOfficerTokenTarget target,
        out int entity, out int effectStart)
    {
        target = default; entity = -1;
        effectStart = identity.IndexOf(UnitAbilityEngine.IntelOfficerActionPrefix, StringComparison.Ordinal);
        if (effectStart < 0) return false;
        string effect = identity[effectStart..];
        if (effect == UnitAbilityEngine.IntelOfficerSkipAction) return false;
        if (!IntelOfficerTargetRules.TryParseChoiceId(effect, UnitAbilityEngine.IntelOfficerActionPrefix, out target))
            throw new SearchCoverageException("Intel Officer policy identity is not a canonical native target choice.");
        if (target.UnitKind == RuntimeUnitKind.Ship)
        {
            foreach (var ship in game.Topology.Ships)
                if (StringComparer.Ordinal.Equals(ship.RuntimeId, target.UnitId.Value)) { entity = ship.EntityIndex; break; }
        }
        else
        {
            foreach (var squadron in game.Topology.Squadrons)
                if (StringComparer.Ordinal.Equals(squadron.RuntimeId, target.UnitId.Value)) { entity = squadron.EntityIndex; break; }
        }
        if (entity < 0)
            throw new SearchCoverageException("Intel Officer policy target is absent from its typed topology.");
        return true;
    }

    private static void WriteArrays(int candidate, PolicyCandidateDescriptor descriptor, List<float> values, List<int> references)
    {
        bool Has(PolicyActionFields field) => (descriptor.KnownFields & field) != 0;
        var action = descriptor.Example;
        if (descriptor.Provider is { } provider)
        {
            // Kind 3 uses an explicit provider reference, distinct from native
            // source/target/index. Masks remain valid when arguments still vary.
            references.AddRange([candidate, 3, provider.ProviderEntity, (int)provider.ProviderKind]);
            var row = new float[12]; row[0] = provider.ProviderEntity >= 0 ? 1 : 0;
            if (provider.TokenType is { } token) { row[1] = 1; row[2 + (int)token] = 1; }
            values.AddRange(row);
        }
        if (Has(PolicyActionFields.Course) && action.Course is { } course)
            for (int i = 0; i < course.Length; i++)
            {
                references.AddRange([candidate, 0, i, -1]); var row = new float[12]; row[0] = course[i] / 2f; values.AddRange(row);
            }
        if (Has(PolicyActionFields.DiceIndices) && action.DiceIndices is { } indices)
            for (int i = 0; i < indices.Length; i++)
            { references.AddRange([candidate, 1, i, indices[i]]); values.AddRange(new float[12]); }
        if (Has(PolicyActionFields.Dice) && action.Dice is { } dice)
            for (int i = 0; i < dice.Length; i++)
            {
                references.AddRange([candidate, 2, i, -1]); var row = new float[12];
                if (dice[i].Color != AttackDieColor.None) row[(int)dice[i].Color] = 1;
                row[4 + (int)dice[i].Face] = 1; values.AddRange(row);
            }
    }

    private static void WriteNative(GameSession position, int observer, PolicyCandidateDescriptor descriptor, Span<float> row, Span<int> refs)
    {
        var a = descriptor.Example; var fields = descriptor.KnownFields; var game = position.Context;
        bool Has(PolicyActionFields field) => (fields & field) == field;
        row[1] = descriptor.CompleteActions / 1024f; refs[5] = (int)fields;
        if (Has(PolicyActionFields.Kind)) { row[2 + (int)a.Kind] = 1; refs[4] = (int)a.Kind; }
        if (Has(PolicyActionFields.Source)) refs[1] = a.Source;
        if (Has(PolicyActionFields.Target)) refs[2] = a.Target;
        if (Has(PolicyActionFields.Ability) && TryIntelTarget(a.AbilityId, game, out _, out int intelEntity, out _))
        {
            // Native Intel Officer offers explicitly target the defender. Do not
            // silently encode an ID/typed-reference mismatch as another action.
            if (!Has(PolicyActionFields.Target) || a.Target != intelEntity)
                throw new SearchCoverageException("Intel Officer native target differs from its encoded public target.");
            refs[2] = intelEntity;
        }
        if (Has(PolicyActionFields.Index)) refs[3] = a.Index;
        if (Has(PolicyActionFields.Command)) row[43 + (int)a.Command] = 1;
        if (Has(PolicyActionFields.FromZone)) row[47 + (int)a.FromZone] = 1;
        if (Has(PolicyActionFields.ToZone)) row[53 + (int)a.ToZone] = 1;
        if (Has(PolicyActionFields.SpendMode)) row[59 + (int)a.SpendMode] = 1;
        if (Has(PolicyActionFields.Token)) row[62 + (int)a.Token] = 1;
        float sign = observer == 1 ? 1 : -1;
        if (Has(PolicyActionFields.Position) && a.Position is { } p)
        {
            row[68] = 1; row[69] = sign * p.X * 2 / game.Topology.BoardSizeMillimetres.X;
            row[70] = sign * p.Y * 2 / game.Topology.BoardSizeMillimetres.Y;
            row[91] = MathF.Abs(row[69]) > 1 || MathF.Abs(row[70]) > 1 ? 1 : 0;
            if (a.Source >= 0 && a.Source < game.Topology.EntityCount && !game.IsDestroyed(a.Source) && !game.IsReserved(a.Source)
                && game.State.Registers[game.Topology.Registers[$"entity.{a.Source}.placed"]] != 0)
            { row[92] = System.Numerics.Vector2.Distance(p, game.State.Poses[a.Source].Position) / 1000; row[93] = 1; }
        }
        bool squadronPlacement = a.Kind is GameActionKind.MoveSquadron or GameActionKind.DisplaceSquadron ||
            a.Kind is GameActionKind.Deploy or GameActionKind.PlaceReservedUnit &&
            (a.Source >= game.State.Ships.Length && a.Source < game.Topology.EntityCount ||
             a.Target >= game.State.Ships.Length && a.Target < game.Topology.EntityCount);
        if (Has(PolicyActionFields.Yaw) && Has(PolicyActionFields.Position) && a.Position.HasValue && !squadronPlacement)
        {
            row[71] = 1; float angle = (a.YawDegrees + (observer == 2 ? 180 : 0)) * MathF.PI / 180;
            row[72] = MathF.Sin(angle); row[73] = MathF.Cos(angle);
        }
        if (Has(PolicyActionFields.Speed) && a.Speed >= 0) { row[74] = 1; row[75] = a.Speed / 4f; }
        if (Has(PolicyActionFields.Amount)) { row[76] = 1; row[77] = a.Amount / 16f; }
        if (Has(PolicyActionFields.Flipped)) row[78] = a.Flipped ? 1 : 0;
        if (Has(PolicyActionFields.Course) && a.Course is { } course)
        {
            if (course.Length > 4) throw new SearchCoverageException("Expand the maneuver-course feature schema.");
            for (int joint = 0; joint < course.Length; joint++) row[79 + joint] = course[joint] / 2f;
            row[94] = course.Length / 4f;
        }
        var tuple = PolicyActionFields.Kind | PolicyActionFields.Source | PolicyActionFields.Target | PolicyActionFields.FromZone | PolicyActionFields.ToZone | PolicyActionFields.Ability;
        if (Has(tuple) && a.Kind == GameActionKind.Attack && a.Source >= 0 && a.Target >= 0 && a.AbilityId.Length == 0)
        {
            using var scope = new GameScope(game);
            row[83] = 1;
            bool valid = CombatModule.TryAttackGeometry(game, a.Source, a.Target, a.FromZone, a.ToZone, out var range, out bool obstructed, out float distance);
            row[84] = valid ? 1 : 0;
            if (valid)
            {
                row[85 + (int)range] = 1; row[89] = distance / 1000; row[90] = obstructed ? 1 : 0; row[107] = 1;
                var battery = CombatModule.Battery(game, a.Source, a.Target, a.FromZone, range, a.ToZone);
                row[96] = battery.Red / 16f; row[97] = battery.Blue / 16f; row[98] = battery.Black / 16f;
            }
        }
        else if (Has(tuple) && a.Kind == GameActionKind.Attack && a.Source >= 0 && a.Target >= 0 && a.AbilityId == "attack.ignition")
        {
            using var scope = new GameScope(game); row[99] = 1;
            bool valid = IgnitionAttackGeometry.TryMeasureAttack(game, a.Source, a.Target, a.ToZone, out var measurement);
            row[84] = valid ? 1 : 0;
            if (valid)
            {
                row[85 + (int)measurement.AttackRange] = 1; row[89] = measurement.AttackDistance / 1000;
                row[90] = measurement.Obstructed ? 1 : 0; row[101] = measurement.IsExtremeRange ? 1 : 0;
                row[102] = measurement.DiceDistance / 1000; row[103 + (int)measurement.DiceRange] = 1; row[107] = 1;
                row[96] = measurement.Battery.RedDice / 16f; row[97] = measurement.Battery.BlueDice / 16f; row[98] = measurement.Battery.BlackDice / 16f;
            }
        }
        else if (Has(PolicyActionFields.Kind | PolicyActionFields.Source | PolicyActionFields.FromZone | PolicyActionFields.Index | PolicyActionFields.Ability)
            && a.Kind == GameActionKind.Attack && a.Source >= 0 && a.AbilityId == "attack.mine")
        {
            using var scope = new GameScope(game); row[100] = 1;
            bool valid = CombatModule.TryMineAttackGeometry(game, a.Source, a.Index, a.FromZone, out var range, out bool obstructed, out var battery);
            row[84] = valid ? 1 : 0;
            if (valid)
            {
                row[85 + (int)range] = 1; row[90] = obstructed ? 1 : 0;
                row[96] = battery.Red / 16f; row[97] = battery.Blue / 16f; row[98] = battery.Black / 16f;
            }
        }
    }
}
