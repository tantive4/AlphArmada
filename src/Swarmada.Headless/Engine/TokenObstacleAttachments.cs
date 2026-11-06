using System.Numerics;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

/// <summary>Causal attachment of physical tokens to obstacles. Only a token's own
/// placement or movement establishes hosts; rigid obstacle movement preserves them.</summary>
public static class TokenObstacleAttachments
{
    public static bool IsTokenKind(int kind) => kind is 1 or SetupModule.GravWellKind or
        SetupModule.GravShiftKind or SetupModule.ProximityMineKind or IgnitionAttackGeometry.TargetTokenKind;

    public static void InitializePreparedTokens(GameContext game)
    {
        for (int token = 0; token < game.State.SpatialObjects.Count; token++)
            if (game.State.SpatialObjects[token].Active && IsTokenKind(game.State.SpatialObjects[token].Kind)) RecordPlacement(game, token);
    }

    public static void RecordPlacement(GameContext game, int token)
    {
        if (token < 0 || token >= game.State.SpatialObjects.Count) throw new ArgumentOutOfRangeException(nameof(token));
        var piece = game.State.SpatialObjects[token];
        if (!piece.Active || !IsTokenKind(piece.Kind)) { game.Flow.TokenHosts.Remove(token); return; }
        var circle = new Circle(piece.Pose.Position, MathF.Max(piece.Size.X, piece.Size.Y) / 2);
        List<int>? hosts = null;
        for (int obstacle = 0; obstacle < game.State.SpatialObjects.Count; obstacle++)
        {
            var candidate = game.State.SpatialObjects[obstacle];
            if (candidate.Active && candidate.Kind == 0 && Geometry2D.Overlaps(circle, ObstacleGeometry.Shape(game, obstacle)))
                (hosts ??= []).Add(obstacle);
        }
        if (hosts is null) game.Flow.TokenHosts.Remove(token); else game.Flow.TokenHosts[token] = hosts.ToArray();
    }

    public static void MoveToken(GameContext game, int token, Pose destination)
    {
        MapMovementRules.RequireEnabled(game);
        var piece = game.State.SpatialObjects[token];
        if (!piece.Active || !IsTokenKind(piece.Kind)) throw new ArgumentException("An active physical token is required.", nameof(token));
        if (!float.IsFinite(destination.Position.X) || !float.IsFinite(destination.Position.Y) || !float.IsFinite(destination.YawDegrees))
            throw new ArgumentException("A finite token destination is required.", nameof(destination));
        game.State.SpatialObjects[token] = piece with { Pose = destination };
        RecordPlacement(game, token);
        MissingSquadronLifecycleEffects.ObserveObjectiveTokens(game);
    }

    public static bool IsCarriedBy(GameContext game, int token, int obstacle)
        => game.Flow.TokenHosts.TryGetValue(token, out var hosts) && Array.IndexOf(hosts, obstacle) >= 0;

    public static void ForgetToken(GameContext game, int token) => game.Flow.TokenHosts.Remove(token);

    /// <summary>Removing an obstacle leaves its tokens at their last physical positions.</summary>
    public static void RemoveObstacle(GameContext game, int obstacle)
    {
        List<int>? affected = null;
        foreach (var pair in game.Flow.TokenHosts)
            if (Array.IndexOf(pair.Value, obstacle) >= 0) (affected ??= []).Add(pair.Key);
        if (affected is null) return;
        foreach (int token in affected)
        {
            var hosts = game.Flow.TokenHosts[token];
            if (hosts.Length == 1) game.Flow.TokenHosts.Remove(token);
            else game.Flow.TokenHosts[token] = hosts.Where(host => host != obstacle).ToArray();
        }
    }

    /// <summary>Retain existing hosts only. A rigid move cannot attach a token to an
    /// obstacle that it happens to cross, or transfer it to a new obstacle.</summary>
    public static void PruneHosts(GameContext game, int token)
    {
        if (!game.Flow.TokenHosts.TryGetValue(token, out var hosts)) return;
        var piece = game.State.SpatialObjects[token]; var circle = new Circle(piece.Pose.Position, MathF.Max(piece.Size.X, piece.Size.Y) / 2);
        int count = 0;
        foreach (int host in hosts)
            if (game.State.SpatialObjects[host] is { Active: true, Kind: 0 } obstacle &&
                Geometry2D.Overlaps(circle, ObstacleGeometry.Shape(game, host))) count++;
        if (count == hosts.Length) return;
        if (count == 0) { game.Flow.TokenHosts.Remove(token); return; }
        var remaining = new int[count]; int next = 0;
        foreach (int host in hosts)
            if (game.State.SpatialObjects[host] is { Active: true, Kind: 0 } obstacle &&
                Geometry2D.Overlaps(circle, ObstacleGeometry.Shape(game, host))) remaining[next++] = host;
        game.Flow.TokenHosts[token] = remaining;
    }
}
