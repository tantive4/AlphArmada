using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Armada.Session;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Official objective routing, stable map identities, and mandatory timing sequences.</summary>
public static partial class OfficialObjectives
{
    public static readonly string[] SupportedIds = [
        "abandoned-mining-facility", "advanced-gunnery", "asteroid-tactics", "blockade-run", "capture-the-vip",
        "close-range-intel-scan", "contested-outpost", "dangerous-territory", "doomed-station", "fighter-ambush",
        "fire-lanes", "fleet-ambush", "fleet-in-being", "hyperspace-assault", "hyperspace-migration", "infested-fields",
        "intel-sweep", "ion-storm", "jamming-barrier", "marked-for-destruction", "minefields", "most-wanted",
        "navigational-hazards", "opening-salvo", "planetary-ion-cannon", "precision-strike", "rift-ambush", "rift-assault",
        "salvage-run", "sensor-net", "solar-corona", "station-assault", "superior-positions", "surprise-attack",
        "targeting-beacons", "volatile-deposits"];
    public static OfficialObjectiveState? State(GameContext game) => game.State.Objective as OfficialObjectiveState;
    public static string Id(GameContext game) => State(game)?.Id ?? string.Empty;
    public static bool Is(GameContext game, string id) => State(game)?.Id == id;
    public static int FirstPlayer(GameContext game) => game.Topology.FirstPlayer;
    public static int SecondPlayer(GameContext game) => 3 - game.Topology.FirstPlayer;
    public static int TokenCount(GameContext game, int entity) => State(game)?.EntityTokens[entity] ?? 0;
    public static void SetTokens(GameContext game, int entity, int count)
    { if (count < 0) throw new ArgumentOutOfRangeException(nameof(count)); State(game)!.EntityTokens[entity] = count; }
    public static bool IsObjectiveShip(GameContext game, int entity) => State(game)?.ObjectiveShips[entity] == true;
    public static void SetObjectiveShip(GameContext game, int entity, bool chosen = true) => State(game)!.ObjectiveShips[entity] = chosen;
    public static int VictoryValue(string id) => id switch {
        "capture-the-vip" => 50, "station-assault" => 40,
        "blockade-run" or "contested-outpost" or "doomed-station" or "hyperspace-migration" or "salvage-run" => 20,
        "abandoned-mining-facility" or "close-range-intel-scan" or "rift-assault" => 10,
        "dangerous-territory" or "fighter-ambush" or "fire-lanes" or "fleet-in-being" or "infested-fields" or "ion-storm" or
        "marked-for-destruction" or "navigational-hazards" or "precision-strike" or "sensor-net" or "superior-positions" or "volatile-deposits" => 15,
        _ => 0 };
    public static void AwardVictory(GameContext game, int player, int count = 1)
    {
        if (player is not (1 or 2) || count < 0) throw new ArgumentOutOfRangeException(nameof(player));
        var state = State(game) ?? throw new InvalidOperationException("No official objective is active.");
        checked { state.VictoryTokens[player - 1] += count; game.State.VictoryPoints[player - 1] += count * VictoryValue(state.Id); }
    }
    public static IEnumerable<int> Obstacles(GameContext game)
    { for (int i = 0; i < game.State.SpatialObjects.Count; i++) if (game.State.SpatialObjects[i] is { Active: true, Kind: 0 }) yield return i; }
    public static string ObstacleId(GameContext game, int index)
    {
        if (State(game)?.PieceKinds.TryGetValue(index, out var id) == true) return id;
        var original = game.Topology.MapPlacements.FirstOrDefault(p => p.SpatialIndex == index);
        return original?.ContentId is "nebula-field" or "space-station" ? "station" : original?.ContentId ?? game.State.SpatialObjects[index].Id;
    }
    public static IEnumerable<int> TokenIndices(GameContext game)
    { for (int i = 0; i < game.State.SpatialObjects.Count; i++) if (game.State.SpatialObjects[i] is { Active: true, Kind: 1 }) yield return i; }
    public static Circle PieceCircle(GameContext game, int index)
    { var piece = game.State.SpatialObjects[index]; return new(piece.Pose.Position, MathF.Max(piece.Size.X, piece.Size.Y) / 2); }
    public static float DistanceToPiece(GameContext game, int entity, int index)
    {
        var piece = game.State.SpatialObjects[index];
        if (game.IsShip(entity) && game.Topology.Ships[entity].IsStation)
            return TokenObstacleAttachments.IsTokenKind(piece.Kind) ? Geometry2D.Distance(PieceCircle(game, index), CombatModule.StationShape(game, entity))
                : Geometry2D.Distance(CombatModule.StationShape(game, entity), ObstacleGeometry.Shape(game, index));
        if (TokenObstacleAttachments.IsTokenKind(piece.Kind)) return game.IsShip(entity) ? Geometry2D.Distance(PieceCircle(game, index), game.ShipBase(entity))
            : Geometry2D.Distance(PieceCircle(game, index), game.SquadronBase(entity));
        var shape = ObstacleGeometry.Shape(game, index);
        return game.IsShip(entity) ? Geometry2D.Distance(game.ShipBase(entity), shape) : Geometry2D.Distance(game.SquadronBase(entity), shape);
    }
    public static int PlaceToken(GameContext game, Vector2 position, int owner = 0, string id = "", float radius = 12.7f)
    {
        Geometry2D.RequireFinite(position, nameof(position));
        int index = game.State.SpatialObjects.Count;
        game.State.SpatialObjects.Add(new(1, new(position), new(radius * 2), owner, Id: id));
        if (State(game) is { } state) state.PieceKinds[index] = "objective-token";
        TokenObstacleAttachments.RecordPlacement(game, index);
        return index;
    }
    public static void RemoveToken(GameContext game, int index)
    {
        var token = game.State.SpatialObjects[index];
        if (!token.Active) return;
        // Observer owns Moralo notification for tracked map tokens, avoiding duplicate events.
        game.State.SpatialObjects[index] = token with { Active = false };
        TokenObstacleAttachments.ForgetToken(game, index);
        MissingSquadronLifecycleEffects.ObserveObjectiveTokens(game);
    }
    public static int AddObstacle(GameContext game, string kind, Vector2 position, Vector2 size, float yaw = 0, bool active = true)
    {
        int index = game.State.SpatialObjects.Count;
        string shapeId = NextObstacleShapeId(game, kind);
        game.State.SpatialObjects.Add(new(0, new(position, yaw), size, Active: active, Id: kind, ShapeId: shapeId));
        State(game)!.PieceKinds[index] = kind; return index;
    }
    public static bool Live(GameContext game, int entity) => entity >= 0 && entity < game.Topology.EntityCount && !game.IsDestroyed(entity) && !game.IsReserved(entity);
    public static IEnumerable<int> LiveShips(GameContext game, int owner = 0) => game.Topology.Ships.Where(s => !s.IsScenarioUnit &&
        (owner == 0 || game.State.Owners[s.EntityIndex] == owner) && Live(game, s.EntityIndex)).Select(s => s.EntityIndex);
    public static bool Handles(string rule) => rule.StartsWith("objective.", StringComparison.Ordinal) || OfficialNavigationObjectives.Handles(rule);
    public static bool IsMandatoryAction(GameActionOption action) => action.Template.AbilityId is "objective.timing.resolve" or "objective.creature.exogorth-start"
        || action.Template.AbilityId.StartsWith("setup.order.", StringComparison.Ordinal);
    public static void PrepareTimingWindow(GameContext game, GameContinuation window)
    {
        if (State(game) is not { } state) return;
        var frames = TimingChoices(game, window.Text, window.Source, window.Target).ToArray();
        if (frames.Length == 0) return;
        var decision = GetDecision(game, frames[0]);
        int owner = decision.Player is 1 or 2 ? decision.Player : frames[0].Source >= 0 && frames[0].Source < game.Topology.EntityCount
            ? game.State.Owners[frames[0].Source] : SecondPlayer(game);
        state.TimingWindows[window.Value] = new(owner, frames);
    }
    public static IEnumerable<GameActionOption> TimingActions(GameContext game, GameContinuation window)
    {
        if (State(game)?.TimingWindows.TryGetValue(window.Value, out var sequence) == true)
        {
            int source = Enumerable.Range(0, game.Topology.EntityCount).First(e => game.State.Owners[e] == sequence.Player);
            yield return new(new(GameActionKind.UseAbility, source, AbilityId: "objective.timing.resolve"), "Resolve " + Id(game) + " timing effects");
        }
    }
    public static bool QueueChoices(GameContext game, IEnumerable<GameContinuation> choices, GamePhase returnPhase)
    {
        var frames = choices.ToArray(); if (frames.Length == 0) return false;
        game.Flow.Continuations.Add(new("objective.common.resume", Stage: (int)returnPhase));
        for (int i = frames.Length - 1; i >= 0; i--) game.Flow.Continuations.Add(frames[i]);
        game.Flow.Phase = GamePhase.AbilityChoice; return true;
    }
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        if (frame.Rule == "objective.common.resume") return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), "Continue after objective effects")]);
        if (frame.Rule.StartsWith("objective.common.", StringComparison.Ordinal)) return SetupDecision(game, frame);
        if (OfficialAssaultObjectives.Handles(frame.Rule)) return OfficialAssaultObjectives.GetDecision(game, frame);
        if (OfficialNavigationObjectives.Handles(frame.Rule)) return OfficialNavigationObjectives.GetDecision(game, frame);
        if (OfficialCreatureObjectives.Handles(frame.Rule)) return OfficialCreatureObjectives.GetDecision(game, frame);
        return OfficialDefenseObjectives.GetDecision(game, frame);
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (frame.Rule == "objective.common.resume") { EngineEffects.ReturnFromChoice(game); game.Flow.Phase = (GamePhase)frame.Stage; return; }
        if (frame.Rule.StartsWith("objective.common.", StringComparison.Ordinal)) { ApplySetup(game, frame, action); return; }
        if (OfficialAssaultObjectives.Handles(frame.Rule)) OfficialAssaultObjectives.Apply(game, frame, action);
        else if (OfficialNavigationObjectives.Handles(frame.Rule)) OfficialNavigationObjectives.Apply(game, frame, action);
        else if (OfficialCreatureObjectives.Handles(frame.Rule)) OfficialCreatureObjectives.Apply(game, frame, action);
        else OfficialDefenseObjectives.Apply(game, frame, action);
    }
    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
        => State(game) is null ? [] : OfficialAssaultObjectives.GetActions(game, timing, source, target)
            .Concat(OfficialNavigationObjectives.GetActions(game, timing, source, target)).Concat(OfficialDefenseObjectives.GetActions(game, timing, source, target))
            .Concat(OfficialCreatureObjectives.GetActions(game, timing, source, target));
    public static bool TryApply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (State(game) is not { } state) return false;
        if (action.AbilityId == "objective.timing.resolve")
        { QueueChoices(game, state.TimingWindows[frame.Value].Frames.Select(f => f.DeepClone()), GamePhase.AbilityChoice); return true; }
        return OfficialAssaultObjectives.TryApply(game, frame, action) || OfficialNavigationObjectives.TryApply(game, frame, action) ||
            OfficialDefenseObjectives.TryApply(game, frame, action) || OfficialCreatureObjectives.TryApply(game, frame, action);
    }
    public static IEnumerable<GameContinuation> TimingChoices(GameContext game, string timing, int source, int target)
        => State(game) is null ? [] : OfficialAssaultObjectives.TimingChoices(game, timing, source, target)
            .Concat(OfficialNavigationObjectives.TimingChoices(game, timing, source, target)).Concat(OfficialDefenseObjectives.TimingChoices(game, timing, source, target))
            .Concat(OfficialCreatureObjectives.TimingChoices(game, timing, source, target));
    public static int ScoreBonus(GameContext game, int player) => State(game) is null ? 0 :
        OfficialAssaultObjectives.ScoreBonus(game, player) + OfficialNavigationObjectives.ScoreBonus(game, player) + OfficialDefenseObjectives.ScoreBonus(game, player);
    public static void Observe(GameContext game)
    {
        if (State(game) is not { } state) return;
        foreach (var pair in state.PieceEntities)
        {
            state.PieceHull[pair.Key] = game.State.Ships[pair.Value].CurrentHull;
            if (game.IsDestroyed(pair.Value)) state.DestroyedPieces.Add(pair.Key); else state.DestroyedPieces.Remove(pair.Key);
        }
        OfficialAssaultObjectives.Observe(game);
        OfficialDefenseObjectives.Observe(game);
    }
    public static bool StationCanRepair(GameContext game, int entity, int obstacle)
    {
        string kind = ObstacleId(game, obstacle);
        if (kind is not ("station" or "unarmed-station")) return false;
        if (Is(game, "contested-outpost")) return false;
        if (State(game)?.PieceEntities.TryGetValue(obstacle, out int station) == true && game.IsDestroyed(station)) return false;
        return !Is(game, "station-assault") || game.State.Owners[entity] == SecondPlayer(game);
    }
}
