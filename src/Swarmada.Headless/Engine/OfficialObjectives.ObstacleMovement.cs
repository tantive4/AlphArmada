using System.Numerics;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

public static partial class OfficialObjectives
{
    private const float ObstacleMoveTolerance = PlayAreaGeometry.SourceContainmentTolerance;

    /// <summary>Validate only this caller-supplied final placement and its physical
    /// range, collisions, attachments, and non-backward toward direction. Minimum
    /// progress and preference among sideways alternatives belong to caller policy.
    /// The final footprint is checked; the swept path is not a collision constraint.</summary>
    public static void ValidateObstacleMove(GameContext game, int index, Pose destination, float maximumDistance,
        int towardPiece = -1, float minimumDistance = 0, bool? flipped = null, bool allowSquadronOverlap = false)
    {
        ArgumentNullException.ThrowIfNull(game);
        if (!CheckObstacleMove(game, index, destination, maximumDistance, towardPiece, minimumDistance, flipped, allowSquadronOverlap, out string failure))
            throw new ArgumentException(failure);
    }

    /// <summary>Pure candidate-mask query. Checks one supplied pose and hand; never
    /// chooses a replacement, searches alternatives, or changes branch/RNG state.</summary>
    public static bool IsLegalObstacleMove(GameContext game, int index, Pose destination, float maximumDistance,
        int towardPiece = -1, float minimumDistance = 0, bool? flipped = null, bool allowSquadronOverlap = false)
    {
        if (game is null) return false;
        try { return CheckObstacleMove(game, index, destination, maximumDistance, towardPiece, minimumDistance, flipped, allowSquadronOverlap, out _); }
        catch (ArgumentException) { return false; }
    }

    private static bool CheckObstacleMove(GameContext game, int index, Pose destination, float maximumDistance,
        int towardPiece, float minimumDistance, bool? flipped, bool allowSquadronOverlap, out string failure)
    {
        failure = "Obstacle relocation is disabled in the static-map training variant.";
        if (!MapMovementRules.Enabled(game)) return false;
        failure = "Obstacle movement distances must be finite and nonnegative.";
        if (!float.IsFinite(maximumDistance) || maximumDistance < 0 || !float.IsFinite(minimumDistance) || minimumDistance < 0) return false;
        failure = "Obstacle positions and angles must be finite.";
        if (!float.IsFinite(destination.Position.X) || !float.IsFinite(destination.Position.Y) || !float.IsFinite(destination.YawDegrees)) return false;
        failure = "Only an active obstacle can be moved.";
        if (index < 0 || index >= game.State.SpatialObjects.Count || game.State.SpatialObjects[index] is not { Active: true, Kind: 0 }) return false;
        failure = "An active, distinct obstacle or token is required as the toward component.";
        if (!ValidObstacleTowardTarget(game, index, towardPiece)) return false;
        int migrationExit = MigrationObstacleDestination(game, index);
        if (migrationExit >= 0) towardPiece = migrationExit;
        if (!ValidObstacleTowardTarget(game, index, towardPiece)) return false;

        var original = game.State.SpatialObjects[index];
        // Retain the explicit stay-put allowance. Every changed pose or physical
        // hand must pass complete final-placement checks, including old overlaps.
        if (destination == original.Pose && (flipped ?? original.Flipped) == original.Flipped) return true;
        var context = new PhysicalObstacleMoveContext(game, index, maximumDistance, towardPiece, flipped, allowSquadronOverlap);
        failure = "The physical obstacle or attached token is blocked, outside the board, beyond movement range, or backward from its toward ruler.";
        if (towardPiece < 0) return context.Valid(destination, null, out _);
        foreach (var ruler in context.Rulers)
            if (context.Progress(ruler, destination) >= -ObstacleMoveTolerance && context.Valid(destination, ruler, out _)) return true;
        return false;
    }

    private static bool ValidObstacleTowardTarget(GameContext game, int index, int toward)
        => toward == -1 || toward >= 0 && toward < game.State.SpatialObjects.Count && toward != index &&
            game.State.SpatialObjects[toward].Active &&
            (game.State.SpatialObjects[toward].Kind == 0 || TokenObstacleAttachments.IsTokenKind(game.State.SpatialObjects[toward].Kind));

    /// <summary>Apply exactly the caller's validated pose and hand, carrying only
    /// causally attached tokens. No movement planner or fallback destination runs.</summary>
    public static void MoveObstacle(GameContext game, int index, Pose destination, float maximumDistance,
        int towardPiece = -1, float minimumDistance = 0, bool? flipped = null, bool allowSquadronOverlap = false)
    {
        ValidateObstacleMove(game, index, destination, maximumDistance, towardPiece, minimumDistance, flipped, allowSquadronOverlap);
        var original = game.State.SpatialObjects[index]; bool mirror = (flipped ?? original.Flipped) != original.Flipped;
        var carried = new List<int>();
        foreach (var pair in game.Flow.TokenHosts)
            if (game.State.SpatialObjects[pair.Key].Active && Array.IndexOf(pair.Value, index) >= 0) carried.Add(pair.Key);
        foreach (int token in carried)
        {
            var piece = game.State.SpatialObjects[token];
            var local = original.Pose.InverseTransformPoint(piece.Pose.Position); if (mirror) local.X = -local.X;
            float yaw = destination.YawDegrees + (mirror ? original.Pose.YawDegrees - piece.Pose.YawDegrees : piece.Pose.YawDegrees - original.Pose.YawDegrees);
            game.State.SpatialObjects[token] = piece with { Pose = new(destination.TransformPoint(local), yaw) };
        }
        game.State.SpatialObjects[index] = original with { Pose = destination, Flipped = flipped ?? original.Flipped };
        foreach (int token in carried) TokenObstacleAttachments.PruneHosts(game, token);
        if (State(game)?.PieceEntities.TryGetValue(index, out int station) == true) game.State.Poses[station] = destination;
        int exit = MigrationObstacleDestination(game, index);
        if (exit >= 0 && Geometry2D.Distance(PieceCircle(game, exit), ObstacleGeometry.Shape(game, index)) <= ObstacleMoveTolerance)
        {
            game.State.SpatialObjects[index] = game.State.SpatialObjects[index] with { Active = false };
            TokenObstacleAttachments.RemoveObstacle(game, index);
        }
        MissingSquadronLifecycleEffects.ObserveObjectiveTokens(game);
    }

    private static int MigrationObstacleDestination(GameContext game, int index)
    {
        if (!Is(game, "hyperspace-migration") || !ObstacleId(game, index).StartsWith("purrgil", StringComparison.Ordinal) ||
            State(game) is not { } state || !state.Flags.TryGetValue("creature.migration.entry", out long entry)) return -1;
        for (int i = 0; i < 2; i++)
            if (state.Flags.TryGetValue("navigation.migration.token." + i, out long token) && token != entry) return (int)token;
        return -1;
    }
}
