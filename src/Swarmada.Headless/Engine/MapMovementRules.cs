namespace Swarmada.Headless.Engine;

/// <summary>Versioned training variant: map-piece relocation is opt-in. Initial
/// placement, marker creation, token collection/removal and unit movement retain
/// their own rules. This policy never supplies a substitute destination.</summary>
public static class MapMovementRules
{
    public const string DisabledVariant = "static-map-v1";
    public const string EnabledVariant = "map-relocation-v1";
    public static bool Enabled(GameContext game) => game.Topology.MapMovementEnabled;
    public static string Variant(GameContext game) => Enabled(game) ? EnabledVariant : DisabledVariant;
    public static void RequireEnabled(GameContext game)
    {
        ArgumentNullException.ThrowIfNull(game);
        if (!Enabled(game)) throw new ArgumentException("Obstacle and map-token relocation is disabled in the static-map training variant.");
    }
}
