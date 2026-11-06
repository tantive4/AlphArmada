using System.Numerics;
using Armada.Core;
using Armada.Session;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public static partial class OfficialObjectives
{
    public static OrientedRectangle SetupArea(GameContext game) => new(new(Vector2.Zero),
        Is(game, "blockade-run") ? game.Topology.BoardSizeMillimetres :
        new(MathF.Min(game.Topology.BoardSizeMillimetres.X, 1219.2f), game.Topology.BoardSizeMillimetres.Y));
    public static OrientedRectangle DeploymentZone(GameContext game, int owner)
    {
        var board = game.Topology.BoardSizeMillimetres;
        if (Is(game, "blockade-run"))
        {
            float sign = owner == FirstPlayer(game) ? -1 : 1;
            return new(new(new Vector2(sign * (board.X / 2 - TabletopMeasurements.LongRange), 0)),
                new(TabletopMeasurements.LongRange * 2, board.Y));
        }
        var original = game.Topology.DeploymentZones.Single(z => z.Owner == owner);
        var minimum = Vector2.Max(original.MinimumMillimetres, -SetupArea(game).HalfSize);
        var maximum = Vector2.Min(original.MaximumMillimetres, SetupArea(game).HalfSize);
        return new(new((minimum + maximum) / 2), maximum - minimum);
    }
    public static float DistanceFromPlayerEdge(GameContext game, Vector2 point, int owner)
    {
        if (Is(game, "blockade-run")) return game.Topology.BoardSizeMillimetres.X / 2 + (owner == FirstPlayer(game) ? point.X : -point.X);
        var zone = game.Topology.DeploymentZones.Single(z => z.Owner == owner);
        return game.Topology.BoardSizeMillimetres.Y / 2 + ((zone.MinimumMillimetres.Y + zone.MaximumMillimetres.Y) < 0 ? point.Y : -point.Y);
    }
    public static float DistanceFromPlayerEdge(GameContext game, OrientedRectangle shape, int owner)
    { Span<Vector2> corners = stackalloc Vector2[4]; shape.CopyCorners(corners); float result = float.PositiveInfinity; foreach (var point in corners) result = MathF.Min(result, DistanceFromPlayerEdge(game, point, owner)); return result; }
    public static float DistanceFromPlayerEdge(GameContext game, PolygonFootprint shape, int owner)
    { float result = float.PositiveInfinity; for (int i = 0; i < shape.Count; i++) result = MathF.Min(result, DistanceFromPlayerEdge(game, shape.Vertex(i), owner)); return result; }
    public static string NextObstacleShapeId(GameContext game, string kind)
    {
        if (kind is not ("dust-field" or "purrgil" or "exogorth")) return ObstacleShapeCatalog.CanonicalId(kind);
        int count = 0;
        for (int i = 0; i < game.State.SpatialObjects.Count; i++)
            if (game.State.SpatialObjects[i].Kind == 0 && ObstacleId(game, i) == kind) count++;
        return ObstacleShapeForOrdinal(kind, count);
    }
    private static string ObstacleShapeForOrdinal(string kind, int ordinal) => kind is "dust-field" or "purrgil" or "exogorth"
        ? (kind == "dust-field" ? "dust" : kind) + "-" + (ordinal % 2 + 1) : ObstacleShapeCatalog.CanonicalId(kind);
    public static Vector2 PieceSize(GameContext game, string kind)
        => ObstacleShapeCatalog.Get(NextObstacleShapeId(game, kind)).Size;
    public static bool TryBeginBeforeDeployment(GameContext game)
    {
        var state = State(game); if (state is null || state.Flags.ContainsKey("setup.before")) return false;
        if (!state.Flags.ContainsKey("setup.obstacles"))
        {
            state.Flags["setup.obstacles"] = 1; PrepareObstacles(game);
            return QueueChoices(game, [new("objective.common.obstacles", Value: Is(game, "doomed-station") ? FirstPlayer(game) : SecondPlayer(game))], GamePhase.Deployment);
        }
        state.Flags["setup.before"] = 1;
        if (Is(game, "fleet-ambush"))
        {
            Span<Vector2> corners = stackalloc Vector2[4]; AmbushZone(game).CopyCorners(corners);
            foreach (var corner in corners) PlaceToken(game, corner, id: "fleet-ambush-corner");
        }
        if (Is(game, "blockade-run")) OfficialAssaultObjectives.SetupChoices(game, false).ToArray();
        return false;
    }
    public static IEnumerable<GameContinuation> SetupChoices(GameContext game, bool afterDeployment) =>
        OfficialAssaultObjectives.SetupChoices(game, afterDeployment).Concat(OfficialNavigationObjectives.SetupChoices(game, afterDeployment))
            .Concat(OfficialDefenseObjectives.SetupChoices(game, afterDeployment)).Concat(OfficialCreatureObjectives.SetupChoices(game, afterDeployment));
    private static void PrepareObstacles(GameContext game)
    {
        var state = State(game)!; string id = state.Id;
        foreach (var original in game.Topology.MapPlacements)
            if (original.Kind == MapPlacementKind.Obstacle) state.PieceKinds[original.SpatialIndex] =
                original.ContentId is "nebula-field" or "space-station" ? "station" : original.ContentId;
        for (int i = 0; i < game.State.SpatialObjects.Count; i++)
            if (game.State.SpatialObjects[i].Kind is 0 or 1)
            {
                if (game.State.SpatialObjects[i].Kind == 0) TokenObstacleAttachments.RemoveObstacle(game, i);
                else TokenObstacleAttachments.ForgetToken(game, i);
                game.State.SpatialObjects[i] = game.State.SpatialObjects[i] with { Active = false };
            }
        var kinds = new List<string> { "asteroid-field-a", "asteroid-field-b", "asteroid-field-c", "debris-field-a", "debris-field-b", "station" };
        if (id is "asteroid-tactics" or "hyperspace-migration" or "infested-fields" or "ion-storm" or "marked-for-destruction" or
            "rift-ambush" or "rift-assault" or "station-assault" or "volatile-deposits" or "jamming-barrier") kinds.Remove("station");
        if (id is "abandoned-mining-facility" or "marked-for-destruction") kinds.RemoveAll(k => k.StartsWith("asteroid", StringComparison.Ordinal));
        if (id == "jamming-barrier") kinds.RemoveAll(k => k.StartsWith("debris", StringComparison.Ordinal));
        if (id is "abandoned-mining-facility" or "close-range-intel-scan" or "jamming-barrier" or "marked-for-destruction" or "rift-ambush" or "salvage-run" or "volatile-deposits")
            kinds.AddRange(["dust-field", "dust-field"]);
        if (id is "doomed-station" or "rift-ambush" or "rift-assault") kinds.Add("gravity-rift");
        if (id is "abandoned-mining-facility" or "marked-for-destruction") kinds.AddRange(["purrgil", "purrgil"]);
        if (id == "station-assault") kinds.AddRange(["unarmed-station", "unarmed-station"]);
        var reused = new HashSet<int>(); var variantOrdinals = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (string kind in kinds)
        {
            int ordinal = variantOrdinals.GetValueOrDefault(kind); variantOrdinals[kind] = ordinal + 1;
            string shapeId = ObstacleShapeForOrdinal(kind, ordinal); var size = ObstacleShapeCatalog.Get(shapeId).Size;
            int existing = state.PieceKinds.FirstOrDefault(p => p.Value == kind && !reused.Contains(p.Key), new(-1, "")).Key;
            int index = existing >= 0 ? existing : AddObstacle(game, kind, Vector2.Zero, size, active: false);
            var prepared = game.State.SpatialObjects[index];
            bool customSize = game.Topology.MapPlacements.Any(piece => piece.SpatialIndex == index && piece.UseCustomSize);
            if (prepared.ShapeId != "custom-rectangle") game.State.SpatialObjects[index] = prepared with { ShapeId = shapeId, Size = customSize ? prepared.Size : size };
            reused.Add(index); state.Flags["setup.piece." + index] = 0;
            if (kind == "unarmed-station")
            {
                state.PieceHull[index] = 10;
                int entity = game.Topology.Ships.Where(s => s.IsStation).ElementAt(state.PieceEntities.Count).EntityIndex;
                state.PieceEntities[index] = entity;
            }
            if ((id is "abandoned-mining-facility" or "salvage-run") && kind == "station" || id == "doomed-station" && kind == "gravity-rift")
            { game.State.SpatialObjects[index] = game.State.SpatialObjects[index] with { Pose = new(Vector2.Zero), Active = true }; state.Flags["setup.piece." + index] = 1; }
        }
    }
    private static bool PendingObstacle(GameContext game, int index) => State(game)!.Flags.TryGetValue("setup.piece." + index, out long placed) && placed == 0;
    private static IEnumerable<int> AvailableObstacles(GameContext game)
    {
        var state = State(game)!; string id = Id(game);
        var remaining = state.PieceKinds.Keys.Where(index => PendingObstacle(game, index)).ToArray();
        if (id == "surprise-attack" && remaining.Any(i => ObstacleId(game, i) == "station")) return remaining.Where(i => ObstacleId(game, i) == "station");
        if (id is "contested-outpost" or "doomed-station" or "navigational-hazards" or "station-assault" && remaining.Any(i => ObstacleId(game, i) is not ("station" or "unarmed-station")))
            return remaining.Where(i => ObstacleId(game, i) is not ("station" or "unarmed-station"));
        if (id == "marked-for-destruction" && remaining.Any(i => ObstacleId(game, i) != "purrgil")) return remaining.Where(i => ObstacleId(game, i) != "purrgil");
        return remaining;
    }
    private static int ObstaclePlacer(GameContext game, GameContinuation frame, int[] available)
    {
        string id = Id(game);
        if (id is "asteroid-tactics" or "blockade-run" or "ion-storm" or "minefields" or "rift-ambush") return SecondPlayer(game);
        if (available.Length > 0 && (id is "contested-outpost" or "doomed-station" or "navigational-hazards" or "station-assault") &&
            available.All(i => ObstacleId(game, i) is "station" or "unarmed-station")) return SecondPlayer(game);
        if (id == "surprise-attack" && available.Any(i => ObstacleId(game, i) == "station") || id == "marked-for-destruction" && available.All(i => ObstacleId(game, i) == "purrgil")) return SecondPlayer(game);
        return (int)frame.Value;
    }
    private static GameDecision SetupDecision(GameContext game, GameContinuation frame)
    {
        int[] remaining = AvailableObstacles(game).ToArray();
        if (remaining.Length == 0) return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), "Obstacle placement complete")]);
        int owner = ObstaclePlacer(game, frame, remaining);
        return new(DecisionKind.Player, owner, GamePhase.AbilityChoice, remaining.Select(index =>
            new GameActionOption(new(GameActionKind.PlaceToken, Target: index, Index: index, AbilityId: "objective.setup.obstacle"),
                "Place " + ObstacleId(game, index), AcceptsPosition: true, AcceptsYaw: true, AcceptsFlip: true)).ToArray());
    }
    private static void ApplySetup(GameContext game, GameContinuation frame, GameAction action)
    {
        var offered = SetupDecision(game, frame);
        if (offered.Kind == DecisionKind.Forced)
        { EngineEffects.ReturnFromChoice(game); game.Flow.Phase = GamePhase.AbilityChoice; return; }
        if (!offered.Actions.Any(o => GameSession.Matches(o, action)) || action.Position is not { } position) throw new ArgumentException("An offered obstacle and destination are required.");
        var old = game.State.SpatialObjects[action.Index]; var pose = new Pose(position, action.YawDegrees);
        ValidateObstacleSetup(game, action.Index, pose, action.Flipped);
        game.State.SpatialObjects[action.Index] = old with { Pose = pose, Active = true, Flipped = action.Flipped };
        if (State(game)!.PieceEntities.TryGetValue(action.Index, out int station)) game.State.Poses[station] = pose;
        State(game)!.Flags["setup.piece." + action.Index] = 1;
        int next = 3 - offered.Player;
        if (Id(game) == "surprise-attack" && ObstacleId(game, action.Index) == "station") next = SecondPlayer(game);
        if (Id(game) == "doomed-station" && State(game)!.Flags.Count(p => p.Key.StartsWith("setup.piece.", StringComparison.Ordinal) && p.Value == 1) == 1) next = FirstPlayer(game);
        game.Flow.Continuations[^1] = frame with { Value = next };
    }
    public static void ValidateObstacleSetup(GameContext game, int index, Pose pose, bool? flipped = null)
    {
        var shape = ObstacleGeometry.Shape(game, index, pose, flipped);
        if (!PlayAreaGeometry.Contains(SetupArea(game), shape)) throw new ArgumentException("Obstacle must be wholly in the setup area.");
        string id = Id(game), kind = ObstacleId(game, index); var half = game.Topology.BoardSizeMillimetres / 2;
        float playerMargin = id switch { "dangerous-territory" or "ion-storm" => TabletopMeasurements.DistanceBand(5), _ => TabletopMeasurements.DistanceBand(3) };
        if (id == "marked-for-destruction" && kind == "purrgil" || id == "rift-ambush" && kind == "gravity-rift") playerMargin = TabletopMeasurements.DistanceBand(5);
        if (kind is "station" or "unarmed-station" && id is "contested-outpost" or "doomed-station" or "navigational-hazards") playerMargin = TabletopMeasurements.DistanceBand(5);
        if (id == "blockade-run") playerMargin = TabletopMeasurements.LongRange * 2;
        if (id == "minefields") playerMargin = -1;
        if (id == "surprise-attack" && kind == "station")
        {
            float distance = DistanceFromPlayerEdge(game, shape, FirstPlayer(game));
            if (distance > TabletopMeasurements.DistanceBand(5)) throw new ArgumentException("Surprise Attack station must be at distance 1 to 5 of the first player edge.");
        }
        else if (playerMargin >= 0 && (DistanceFromPlayerEdge(game, shape, 1) <= playerMargin || DistanceFromPlayerEdge(game, shape, 2) <= playerMargin))
            throw new ArgumentException("Obstacle is too close to a player edge.");
        if (id != "blockade-run" && id != "minefields")
        {
            for (int i = 0; i < shape.Count; i++)
            {
                var corner = shape.Vertex(i);
                if (MathF.Abs(corner.X) >= half.X - TabletopMeasurements.DistanceBand(3))
                    throw new ArgumentException("Obstacle is too close to a short play-area edge.");
            }
        }
        float spacing = id == "minefields" ? TabletopMeasurements.DistanceBand(5) : TabletopMeasurements.DistanceOne;
        foreach (int other in Obstacles(game))
            if (other != index && Geometry2D.Distance(shape, ObstacleGeometry.Shape(game, other)) <= spacing)
                throw new ArgumentException("Obstacles are too close to one another.");
        if (id is "abandoned-mining-facility" or "salvage-run")
        {
            int station = Obstacles(game).First(i => ObstacleId(game, i) == "station");
            float distance = Geometry2D.Distance(shape, ObstacleGeometry.Shape(game, station));
            if (distance > TabletopMeasurements.DistanceBand(5) || id == "abandoned-mining-facility" && distance <= TabletopMeasurements.DistanceOne)
                throw new ArgumentException("Obstacle is outside the specified station distance bands.");
        }
    }
}
