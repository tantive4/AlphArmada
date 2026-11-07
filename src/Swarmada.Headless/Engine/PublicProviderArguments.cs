using System.Globalization;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Engine;

namespace Swarmada.Headless.Search;

/// <summary>Public effect arguments separate from the native combatants. A
/// negative provider is masked while a conditional choice still contains more
/// than one provider. Null TokenType means unknown/not applicable.</summary>
public sealed record PublicProviderArgument(string Family, int ProviderEntity,
    RuntimeUnitKind ProviderKind, DefenseTokenType? TokenType = null)
{
    public static PublicProviderArgument? Merge(PublicProviderArgument? first, PublicProviderArgument? next)
        => first is null || next is null || first.Family != next.Family ? null : first with
        {
            ProviderEntity = first.ProviderEntity == next.ProviderEntity ? first.ProviderEntity : -1,
            ProviderKind = first.ProviderKind == next.ProviderKind ? first.ProviderKind : 0,
            TokenType = first.TokenType == next.TokenType ? first.TokenType : null
        };
}

/// <summary>Bounded, exact parsing for five native public provider families.
/// Original GameActions and engine IDs remain unchanged. No hashes, guesses,
/// private information or rule queries participate in this projection.</summary>
public static class PublicProviderArguments
{
    public const string Version = "public-provider-arguments-v1";
    private static readonly (string Family, string Prefix, RuntimeUnitKind Kind, bool Token)[] Families =
    [
        ("remaining.attack-dice.clone-gunners", "remaining.attack-dice.clone-gunners.", RuntimeUnitKind.Ship, false),
        ("remaining.attack-dice.anakin", "remaining.attack-dice.anakin.", RuntimeUnitKind.Ship, true),
        ("attack.dice.task-force-organa", "attack.dice.task-force-organa.", RuntimeUnitKind.Ship, false),
        ("remaining.attack-dice.toryn-farr", "remaining.attack-dice.toryn-farr.", RuntimeUnitKind.Ship, false),
        ("clone-wars.axe", "clone-wars.axe.", RuntimeUnitKind.Squadron, false)
    ];
    public static bool TryFamily(string abilityId, out string family)
    {
        foreach (var item in Families)
            if (abilityId.StartsWith(item.Prefix, StringComparison.Ordinal)) { family = item.Family; return true; }
        family = ""; return false;
    }

    public static bool TryResolve(GameContext game, string abilityId, out PublicProviderArgument argument)
    {
        foreach (var item in Families)
        {
            string prefix = item.Prefix;
            if (!abilityId.StartsWith(prefix, StringComparison.Ordinal)) continue;
            string runtimeId = abilityId[prefix.Length..]; DefenseTokenType? token = null;
            if (item.Token)
            {
                int split = runtimeId.LastIndexOf('.');
                if (split < 1 || !int.TryParse(runtimeId.AsSpan(split + 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out int type)
                    || !Enum.IsDefined((DefenseTokenType)type))
                    throw new SearchCoverageException("Public provider effect has an invalid defense-token argument.");
                token = (DefenseTokenType)type; runtimeId = runtimeId[..split];
            }
            if (!game.Topology.EntityByRuntimeId.TryGetValue(runtimeId, out int entity)
                || (game.IsShip(entity) ? RuntimeUnitKind.Ship : RuntimeUnitKind.Squadron) != item.Kind)
                throw new SearchCoverageException("Public provider effect has an unknown or wrong-kind topology entity.");
            argument = new(item.Family, entity, item.Kind, token); return true;
        }
        argument = null!; return false;
    }

    public static string NormalizeIdentity(GameContext game, string identity)
    {
        foreach (var item in Families)
        {
            int start = identity.IndexOf(item.Prefix, StringComparison.Ordinal);
            if (start < 0) continue;
            TryResolve(game, identity[start..], out var value);
            return identity[..start] + value.Family
                + FormattableString.Invariant($":source:{value.ProviderEntity}:kind:{(int)value.ProviderKind}")
                + (value.TokenType.HasValue ? FormattableString.Invariant($":token:{(int)value.TokenType.Value}") : "");
        }
        return identity;
    }
}
