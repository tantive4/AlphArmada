using System.Numerics;
using Armada.Session;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Setup;

public sealed record MapSetupSelection(int Sequence, string Context, int CandidateId,
    GameAction Action, bool Exploration);
public sealed record MapSetupAudit(string PolicyVersion, int Seed, double Epsilon,
    IReadOnlyList<MapSetupSelection> Selections);

/// <summary>A bounded caller-policy domain was exhausted. This is not an engine
/// proof that setup is impossible, and must never be converted into a game reward.</summary>
public sealed class MapSetupCoverageException(string message) : InvalidOperationException(message);

/// <summary>Optional caller-side map generator. GameSession never invokes it.
/// Authored neutral presets, or supplied saved-map poses, take precedence over
/// epsilon exploration among a bounded finite set of legal candidates. These are
/// not an original SWArmada AI: that program has no initial map-placement policy.
/// One instance belongs to one setup coordinator and is not thread safe.</summary>
public sealed class MapSetupPolicy
{
    public const string Version = "authored-map-setup-v1";
    public const int MaximumCandidates = 16_384;
    private const int BatchSize = 64;
    private readonly DiscreteActionAdapter adapter = new();
    private readonly List<MapSetupSelection> selections = [];
    private readonly SavedPlacement[] baseline;
    private DeterministicRandom random;
    public int Seed { get; }
    public double Epsilon { get; }
    public IReadOnlyList<MapSetupSelection> Selections => Array.AsReadOnly(selections
        .Select(item => item with { Action = item.Action.DeepClone() }).ToArray());

    public MapSetupPolicy(int seed, double epsilon = .05, MatchConfigDto? baseline = null)
    {
        if (!double.IsFinite(epsilon) || epsilon < 0 || epsilon > 1)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be finite and in [0, 1].");
        Seed = seed; Epsilon = epsilon;
        random = new DeterministicRandom(unchecked((ulong)(long)seed));
        this.baseline = (baseline?.map?.placements ?? []).Where(piece => piece is not null)
            .Select(piece => new SavedPlacement(piece.placementId, piece.contentId, piece.kind,
                new Pose(new Vector2(TabletopMeasurements.FromWorldUnits(piece.positionX),
                    TabletopMeasurements.FromWorldUnits(piece.positionZ)), piece.yawDegrees), piece.flipped)).ToArray();
    }

    public MapSetupAudit CaptureAudit() => new(Version, Seed, Epsilon, Selections);

    /// <summary>True only for map-generation choices. Unit deployment, reserves,
    /// objective-ship selection, commands, and tactical abilities remain caller/MCTS
    /// decisions. Some setup rules are scheduled after deployment; the coordinator
    /// must call this method at their actual rule timing. Runtime is never owned.</summary>
    public static bool OwnsDecision(GameSession game, GameDecision decision)
    {
        ArgumentNullException.ThrowIfNull(game); ArgumentNullException.ThrowIfNull(decision);
        if (decision.Kind != DecisionKind.Player || decision.Actions.Count == 0) return false;
        var context = game.Context; var frame = context.Flow.Continuations.LastOrDefault();
        bool setup = context.State.Registers[context.Topology.Registers["setup.completed"]] == 0;
        if (!setup) return false;
        if (frame?.Rule is "objective.common.obstacles" or "objective.assault.beacons"
            or "official-nav.place-tokens" or "objective.defense.place-tokens") return true;
        if (frame?.Rule == "objective.creature.place-exogorths" && frame.Text == "initial") return true;
        if (frame?.Rule == "objective.creature.choose-exogorth-host" && context.Flow.Continuations
            .Any(parent => parent.Rule == "objective.creature.place-exogorths" && parent.Text == "initial")) return true;
        if (frame?.Rule == "objective.creature.migration-spawn" && context.State.Round == 1) return true;
        return decision.Phase == GamePhase.Deployment && decision.Actions
            .Any(option => option.Template.AbilityId == "setup.place-token") && decision.Actions
            .All(option => option.Template.Kind == GameActionKind.Pass || option.Template.AbilityId == "setup.place-token");
    }

    /// <summary>Apply one explicitly selected map setup action, or return false at
    /// the next non-map boundary. Validation uses isolated native scratch states;
    /// policy randomness is independent of the game's random stream. Exhausting the
    /// finite policy domain raises a coverage error, not an impossible-setup claim.</summary>
    public bool TryApplyNext(GameSession game)
    {
        ArgumentNullException.ThrowIfNull(game);
        var decision = game.GetDecision();
        if (!OwnsDecision(game, decision)) return false;
        string context = ContextName(game, decision);
        bool explore = Epsilon > 0 && (random.NextUInt64() >> 11) * (1.0 / 9_007_199_254_740_992) < Epsilon;
        DiscreteActionCandidate? selected = null; int legalCount = 0;
        foreach (var batch in Candidates(game, decision).Chunk(BatchSize))
        {
            var mask = adapter.GetDecision(game, batch);
            if (mask.InvalidSetup is { } invalid) throw new InvalidSetupException(invalid);
            foreach (var candidate in mask.Actions)
            {
                if (!candidate.Legal) continue;
                legalCount++;
                if (!explore)
                {
                    adapter.Apply(game, candidate.Id, mask.DecisionToken);
                    Record(context, candidate.Id, candidate.Action, false); return true;
                }
                // Uniform reservoir sampling across ALL legal entries in the
                // bounded domain; illegal proposals receive no probability mass.
                if (random.NextInt(legalCount) == 0) selected = new(candidate.Id, candidate.Action);
            }
        }
        if (selected is null) throw new MapSetupCoverageException(
            $"{Version} found no legal candidate in its bounded map domain for objective '{OfficialObjectives.Id(game.Context)}', {context}, preset {Math.Abs(Seed % 6)}. " +
            "This is a policy-coverage failure, not proof that the official setup is impossible.");
        var chosen = adapter.GetDecision(game, [selected]);
        if (chosen.InvalidSetup is { } invalidSetup) throw new InvalidSetupException(invalidSetup);
        adapter.Apply(game, selected.Id, chosen.DecisionToken);
        Record(context, selected.Id, selected.Action, true); return true;
    }

    private void Record(string context, int id, GameAction action, bool exploration)
        => selections.Add(new(selections.Count, context, id, action.DeepClone(), exploration));
    private static string ContextName(GameSession game, GameDecision decision)
        => game.Context.Flow.Continuations.LastOrDefault()?.Rule ?? decision.Actions[0].Template.AbilityId;

    private IEnumerable<DiscreteActionCandidate> Candidates(GameSession session, GameDecision decision)
    {
        var game = session.Context; var frame = game.Flow.Continuations.LastOrDefault();
        var seen = new HashSet<GameAction>(); int id = 0;
        // Placement precedes the explicit optional skip during exploitation.
        // Exploration can select any offered legal entry, including an optional skip.
        foreach (var option in decision.Actions.OrderBy(option => option.Template.Kind == GameActionKind.Pass ? 1 : 0))
            foreach (var action in Actions(game, option, frame))
            {
                if (!seen.Add(action)) continue;
                yield return new(id++, action);
                if (id >= MaximumCandidates) yield break;
            }
    }

    private IEnumerable<GameAction> Actions(GameContext game, GameActionOption option, GameContinuation? frame)
    {
        var action = option.Template;
        if (!option.AcceptsPosition) { yield return action; yield break; }
        foreach (var saved in SavedPoses(game, option, frame))
            yield return action with { Position = saved.Pose.Position,
                YawDegrees = option.AcceptsYaw ? saved.Pose.YawDegrees : action.YawDegrees,
                Flipped = option.AcceptsFlip ? saved.Flipped : action.Flipped };
        if (frame?.Rule == "objective.creature.place-exogorths")
            foreach (var placement in ExogorthContacts(game, action)) yield return placement;
        var poses = Positions(game, option, frame, Math.Abs(Seed % 6)).ToArray();
        // Prefer the entire authored orientation-zero layout before trying
        // rotated alternatives. Greedily rotating an early border pose can
        // otherwise leave no room for the later mandatory pieces.
        foreach (float turn in option.AcceptsYaw ? new[] { 0f, 90, 180, 270 } : new[] { 0f })
            foreach (var pose in poses)
                foreach (bool flip in option.AcceptsFlip ? new[] { action.Flipped, !action.Flipped } : new[] { action.Flipped })
                    yield return action with { Position = pose.Position,
                        YawDegrees = option.AcceptsYaw ? pose.YawDegrees + turn : action.YawDegrees, Flipped = flip };
    }

    private IEnumerable<SavedPlacement> SavedPoses(GameContext game, GameActionOption option, GameContinuation? frame)
    {
        int piece = option.Template.AbilityId == "objective.setup.obstacle" ? option.Template.Index : option.Template.Target;
        if (piece >= 0 && piece < game.State.SpatialObjects.Count && game.State.SpatialObjects[piece].Kind == 0)
        {
            var prepared = game.Topology.MapPlacements.FirstOrDefault(item => item.SpatialIndex == piece);
            if (prepared is not null)
                foreach (var saved in baseline.Where(saved => saved.Id == prepared.PlacementId && saved.Kind == MapPlacementKind.Obstacle)) yield return saved;
            string kind = Canonical(OfficialObjectives.ObstacleId(game, piece));
            int ordinal = Enumerable.Range(0, piece).Count(index => game.State.SpatialObjects[index].Kind == 0 && Canonical(OfficialObjectives.ObstacleId(game, index)) == kind);
            var matches = baseline.Where(saved => saved.Kind == MapPlacementKind.Obstacle && Canonical(saved.Content) == kind).ToArray();
            if (ordinal < matches.Length) yield return matches[ordinal];
        }
        else
        {
            int ordinal = frame?.Stage ?? selections.Count(item => item.Context == "setup.place-token");
            var matches = baseline.Where(saved => saved.Kind == MapPlacementKind.Objective).ToArray();
            if (ordinal >= 0 && ordinal < matches.Length) yield return matches[ordinal];
        }
    }

    private static string Canonical(string kind) => kind switch
    {
        "nebula-field" or "space-station" => "station",
        "dust-1" or "dust-2" => "dust-field",
        "purrgil-1" or "purrgil-2" => "purrgil",
        "exogorth-1" or "exogorth-2" => "exogorth",
        _ => kind
    };

    private static IEnumerable<Pose> Positions(GameContext game, GameActionOption option, GameContinuation? frame, int preset)
    {
        var action = option.Template; string objective = OfficialObjectives.Id(game);
        if (action.AbilityId == "objective.setup.obstacle")
        {
            var candidates = ObstacleCandidates(objective, preset).ToArray();
            int offset = preset * 17 % candidates.Length;
            foreach (var point in candidates.Skip(offset).Concat(candidates.Take(offset)).Distinct()
                .OrderByDescending(point => NearestObstacle(game, point))) yield return new(point);
            yield break;
        }
        if (frame?.Rule == "objective.creature.migration-spawn" && OfficialObjectives.State(game)!.Flags.TryGetValue("creature.migration.entry", out long entry))
        {
            var point = game.State.SpatialObjects[(int)entry].Pose.Position;
            foreach (float distance in new[] { 80f, 110, 140 }) yield return new(point + new Vector2(point.X < 0 ? distance : -distance, 0));
        }
        if (objective == "hyperspace-migration" && frame?.Rule == "official-nav.place-tokens")
            yield return new(new Vector2(frame.Stage == 0 ? -880 : 880, 0));
        if (objective == "salvage-run" && frame?.Rule == "official-nav.place-tokens")
            foreach (var point in new[] { new Vector2(-110, 0), new(110, 0), new(0, -110), new(0, 110) }) yield return new(point);
        if (action.Target >= 0 && action.Target < game.State.SpatialObjects.Count) yield return game.State.SpatialObjects[action.Target].Pose;
        foreach (int obstacle in OfficialObjectives.Obstacles(game))
        {
            var piece = game.State.SpatialObjects[obstacle]; yield return piece.Pose;
            foreach (var delta in new[] { new Vector2(100, 0), new(-100, 0), new(0, 100), new(0, -100) }) yield return new(piece.Pose.Position + delta);
        }
        foreach (float x in new[] { 0f, -500, -250, 250, 500, -880, 880 })
            foreach (float y in new[] { 0f, -100, 100, -200, 200, -350, 350 }) yield return new(new Vector2(x, y));
    }

    private static IEnumerable<Vector2> ObstacleCandidates(string id, int preset)
    {
        if (id == "minefields")
            foreach (float x in new[] { -470f, 0, 470 }) foreach (float y in new[] { -230f, 230 }) yield return new(x, y);
        foreach (float x in new[] { -300f, -220, 0, 220, 300 }) foreach (float y in new[] { 0f, -190, 190, -210, 210 }) yield return new(x, y);
        var columns = preset % 2 == 0 ? new[] { -500f, -300, -100, 100, 300, 500 } : new[] { -450f, -225, 0, 225, 450 };
        foreach (float x in columns)
            foreach (float y in preset % 3 == 0 ? new[] { -80f, 80, -170, 170, 0 } : new[] { -170f, 170, -80, 80, 0 }) yield return new(x, y);
        foreach (float x in new[] { -500f, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500 })
            foreach (float y in new[] { -350f, -250, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250, 350 }) yield return new(x, y);
    }

    private static IEnumerable<GameAction> ExogorthContacts(GameContext game, GameAction action)
    {
        foreach (int host in OfficialObjectives.Obstacles(game).Where(index => !OfficialObjectives.ObstacleId(game, index).StartsWith("exogorth", StringComparison.Ordinal)))
        {
            var obstacle = ObstacleGeometry.Shape(game, host);
            foreach (float yaw in new[] { obstacle.Pose.YawDegrees, obstacle.Pose.YawDegrees + 90, obstacle.Pose.YawDegrees + 180, obstacle.Pose.YawDegrees + 270 })
                foreach (bool flip in new[] { action.Flipped, !action.Flipped })
                {
                    var shape = ObstacleGeometry.Shape(game, action.Target, new Pose(Vector2.Zero, yaw), flip);
                    foreach (var direction in new[] { Vector2.UnitX, -Vector2.UnitX, Vector2.UnitY, -Vector2.UnitY })
                        yield return action with { Position = Support(obstacle, direction) - Support(shape, -direction), YawDegrees = yaw, Flipped = flip };
                }
        }
    }

    private static float NearestObstacle(GameContext game, Vector2 point)
    {
        float distance = float.PositiveInfinity;
        foreach (int index in OfficialObjectives.Obstacles(game)) distance = MathF.Min(distance, Vector2.Distance(point, game.State.SpatialObjects[index].Pose.Position));
        return distance;
    }
    private static Vector2 Support(PolygonFootprint shape, Vector2 direction)
    {
        var best = shape.Vertex(0); float distance = Vector2.Dot(best, direction);
        for (int index = 1; index < shape.Count; index++)
        {
            var point = shape.Vertex(index); float next = Vector2.Dot(point, direction);
            if (next > distance) { best = point; distance = next; }
        }
        return best;
    }
    private sealed record SavedPlacement(string Id, string Content, MapPlacementKind Kind, Pose Pose, bool Flipped);
}
