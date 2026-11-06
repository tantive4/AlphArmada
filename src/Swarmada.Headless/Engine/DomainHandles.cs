using System.Numerics;
using Armada.Core;
using Armada.Data;
using Armada.Rules;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

/// <summary>Scoped dependency for recovered rule queries that formerly searched the Unity scene.</summary>
public readonly struct GameScope : IDisposable
{
    private static readonly AsyncLocal<GameContext?> Context = new();
    private readonly GameContext? previous;
    private readonly bool changed;
    public static GameContext Current => Context.Value ?? throw new InvalidOperationException("A source rule requires a scoped game context.");
    public GameScope(GameContext game)
    { previous = Context.Value; changed = !ReferenceEquals(previous, game); if (changed) Context.Value = game; }
    public void Dispose() { if (changed) Context.Value = previous; }
}
public sealed partial class GameContext
{
    private ShipView[]? shipViews;
    private SquadronView[]? squadronViews;
    private ShipView[]? sceneShipViews;
    private SquadronView[]? sceneSquadronViews;
    public ShipView[] ShipViews => shipViews ??= Topology.Ships.Select(x => new ShipView(this, x.EntityIndex)).ToArray();
    public SquadronView[] SquadronViews => squadronViews ??= Topology.Squadrons.Select(x => new SquadronView(this, x.EntityIndex)).ToArray();
    public ShipView[] SceneShipViews
    {
        get
        {
            var all = ShipViews; int count = 0; bool same = sceneShipViews is not null;
            for (int i = 0; i < all.Length; i++)
                if (!IsReserved(all[i].Index) && !all[i].IsDestroyed)
                { same &= sceneShipViews is not null && count < sceneShipViews.Length && ReferenceEquals(sceneShipViews[count], all[i]); count++; }
            if (same && sceneShipViews!.Length == count) return sceneShipViews;
            var result = new ShipView[count]; count = 0;
            for (int i = 0; i < all.Length; i++) if (!IsReserved(all[i].Index) && !all[i].IsDestroyed) result[count++] = all[i];
            return sceneShipViews = result;
        }
    }
    public SquadronView[] SceneSquadronViews
    {
        get
        {
            var all = SquadronViews; int count = 0; bool same = sceneSquadronViews is not null;
            for (int i = 0; i < all.Length; i++)
                if (!IsReserved(all[i].Index) && !all[i].IsDestroyed)
                { same &= sceneSquadronViews is not null && count < sceneSquadronViews.Length && ReferenceEquals(sceneSquadronViews[count], all[i]); count++; }
            if (same && sceneSquadronViews!.Length == count) return sceneSquadronViews;
            var result = new SquadronView[count]; count = 0;
            for (int i = 0; i < all.Length; i++) if (!IsReserved(all[i].Index) && !all[i].IsDestroyed) result[count++] = all[i];
            return sceneSquadronViews = result;
        }
    }
}

/// <summary>Pure domain handle retaining source rule signatures. No rendering or Unity objects.</summary>
public sealed class ShipView
{
    private readonly ShipLoadoutState loadout;
    private readonly UnitTransform unitTransform;
    private readonly GameplayFootprintView footprint;
    private readonly string deferredDestructionKey;
    public GameContext Game { get; }
    public int Index { get; }
    public ShipView(GameContext game, int index)
    { Game = game; Index = index; loadout = new(game, index); unitTransform = new(game, index); footprint = new(game, index); deferredDestructionKey = "rieekan.ship." + index; }
    public ShipRuntimeState State => Game.State.Ships[Index];
    public ShipDefinition Definition => Game.Topology.Ships[Index].CatalogEntry.Definition;
    public int OwnerId => Game.State.Owners[Index];
    public bool IsDestroyed => State.Destroyed && !Game.Flow.Flags.ContainsKey(deferredDestructionKey);
    public string SizeClass => Game.Topology.Ships[Index].CatalogEntry.Metadata.Size;
    public RuntimeUnitId RuntimeId => new(Game.Topology.Ships[Index].RuntimeId);
    public CommandStackState Commands => Game.State.Commands[Index];
    public ShipLoadoutState Loadout => loadout;
    public UnitTransform transform => unitTransform;
    public static implicit operator RuntimeUnitId(ShipView ship) => ship.RuntimeId;
    public T? GetComponent<T>() where T : class => footprint as T;
    public HullZone DetermineFiringZone(Vector3 point)
    {
        if (Game.Topology.Ships[Index].IsStation) return HullZone.Front;
        var local = transform.InverseTransformPoint(point);
        var ratio = Definition.GameplayFootprint.X / Definition.GameplayFootprint.Y;
        if (MathF.Abs(local.X) <= MathF.Abs(local.Z) * ratio) return local.Z >= 0 ? HullZone.Front : HullZone.Rear;
        if (Definition.HasAuxiliaryHullZones && local.Z < -Definition.GameplayFootprint.Y * 0.07121825f)
            return local.X < 0 ? HullZone.LeftAuxiliary : HullZone.RightAuxiliary;
        return local.X < 0 ? HullZone.Left : HullZone.Right;
    }
    public HullZone DetermineHullZone(Vector3 point) => DetermineFiringZone(point);
    public HullZoneGeometry HullGeometryFor(HullZone zone) => Game.Topology.Ships[Index].IsStation ? CombatModule.StationZone(Game, Index, Game.State.Poses[Index].Position) : Definition.HasAuxiliaryHullZones ?
        HullGeometry.Huge(Game.ShipBase(Index), zone, Definition.ShipId.Contains("prototype", StringComparison.Ordinal)) :
        HullGeometry.Standard(Game.ShipBase(Index), zone);
    public Vector3[] GetHullZoneEdgesWorld(HullZone zone)
    {
        var edges = HullGeometryFor(zone).Edges; var points = new Vector3[edges.Length + 1];
        for (int i = 0; i < edges.Length; i++) points[i] = EngineVectors.ToWorld(edges[i].Start);
        points[^1] = EngineVectors.ToWorld(edges[^1].End); return points;
    }
    public Vector3 GetGameplayTargetingPoint(HullZone zone) => EngineVectors.ToWorld(HullGeometryFor(zone).TargetingPoint);
    public void GetHullZoneBoundaryDirections(HullZone zone, out Vector3 start, out Vector3 end)
    {
        var geometry = HullGeometryFor(zone);
        start = new(geometry.FirstBoundary.X, 0, geometry.FirstBoundary.Y);
        end = new(geometry.SecondBoundary.X, 0, geometry.SecondBoundary.Y);
    }
}
public sealed class SquadronView
{
    private readonly UnitTransform unitTransform;
    private readonly GameplayFootprintView footprint;
    public GameContext Game { get; }
    public int Index { get; }
    public SquadronView(GameContext game, int index)
    { Game = game; Index = index; unitTransform = new(game, index); footprint = new(game, index); }
    public SquadronRuntimeState State => Game.State.Squadrons[Game.SquadronIndex(Index)];
    public SquadronDefinition Definition => Game.Topology.Squadrons[Game.SquadronIndex(Index)].CatalogEntry.Definition;
    public int OwnerId => Game.State.Owners[Index];
    public bool IsDestroyed => State.Destroyed;
    public string LeaderUpgradeId => Game.Topology.Squadrons[Game.SquadronIndex(Index)].LeaderUpgradeIndex is int index && index >= 0 ?
        Game.State.Upgrades[index].Definition.id : string.Empty;
    public RuntimeUnitId RuntimeId => new(Game.Topology.Squadrons[Game.SquadronIndex(Index)].RuntimeId);
    public UnitTransform transform => unitTransform;
    public T? GetComponent<T>() where T : class => footprint as T;
    public void MarkActivated(bool commanded) => State.MarkActivated(commanded);
    public int ApplyDamage(int amount) => State.ApplyDamage(amount);
}
public sealed class UnitTransform(GameContext game, int entity)
{
    public Vector3 position
    {
        get => EngineVectors.ToWorld(game.State.Poses[entity].Position);
        set => game.State.Poses[entity] = new(EngineVectors.FromWorld(value), game.State.Poses[entity].YawDegrees);
    }
    public Vector3 right => new(game.State.Poses[entity].Right.X, 0, game.State.Poses[entity].Right.Y);
    public Vector3 forward => new(game.State.Poses[entity].Forward.X, 0, game.State.Poses[entity].Forward.Y);
    public Quaternion rotation => Quaternion.CreateFromAxisAngle(Vector3.UnitY, game.State.Poses[entity].YawDegrees * MathF.PI / 180);
    public Vector3 TransformPoint(Vector3 point) => EngineVectors.ToWorld(game.State.Poses[entity].TransformPoint(EngineVectors.FromWorld(point)));
    public Vector3 InverseTransformPoint(Vector3 point) => EngineVectors.ToWorld(game.State.Poses[entity].InverseTransformPoint(EngineVectors.FromWorld(point)));
    public Vector3 InverseTransformDirection(Vector3 direction) => Vector3.Transform(direction, Quaternion.Inverse(rotation));
}
public sealed class GameplayFootprintView(GameContext game, int entity)
{
    private readonly UnitTransform unitTransform = new(game, entity);
    public Vector2 SourceSize => game.IsShip(entity) ? game.Topology.Ships[entity].CatalogEntry.Definition.GameplayFootprint :
        game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition.GameplayFootprint;
    public UnitTransform transform => unitTransform;
    public bool IsConfigured => true;
}
public sealed class MapObstacleView(GameContext game, int index)
{
    public bool isActiveAndEnabled => game.State.SpatialObjects[index].Active;
    public ObstacleTransform transform => new(game, index);
    public static IEnumerable<MapObstacleView> ActiveObstacles => GameScope.Current.State.SpatialObjects
        .Select((x, i) => (State: x, Index: i)).Where(x => x.State.Active && x.State.Kind == (int)Armada.Session.MapPlacementKind.Obstacle)
        .Select(x => new MapObstacleView(GameScope.Current, x.Index));
}
public sealed class ObstacleTransform(GameContext game, int index)
{
    public Vector3 position => EngineVectors.ToWorld(game.State.SpatialObjects[index].Pose.Position);
}
public static class EngineVectors
{
    public static Vector3 ToWorld(Vector2 mm) => new(mm.X / TabletopMeasurements.MillimetresPerWorldUnit, 0,
        mm.Y / TabletopMeasurements.MillimetresPerWorldUnit);
    public static Vector2 FromWorld(Vector3 world) => new(world.X * TabletopMeasurements.MillimetresPerWorldUnit,
        world.Z * TabletopMeasurements.MillimetresPerWorldUnit);
    public static Vector3 ProjectOnPlane(Vector3 v, Vector3 normal) => v - normal * (Vector3.Dot(v, normal) / normal.LengthSquared());
    public static Vector2 NormalizeSafe(this Vector2 v) => v.LengthSquared() > 0 ? Vector2.Normalize(v) : Vector2.Zero;
    public static Vector3 NormalizeSafe(this Vector3 v) => v.LengthSquared() > 0 ? Vector3.Normalize(v) : Vector3.Zero;
    public static float Angle(Vector3 a, Vector3 b)
    {
        float magnitude = MathF.Sqrt(a.LengthSquared() * b.LengthSquared());
        return magnitude < 1E-15f ? 0 : MathF.Acos(Math.Clamp(Vector3.Dot(a, b) / magnitude, -1, 1)) * 180 / MathF.PI;
    }
}
public static class EngineMath
{
    public const float PI = MathF.PI, Deg2Rad = MathF.PI / 180, Rad2Deg = 180 / MathF.PI, Infinity = float.PositiveInfinity;
    public static float Abs(float x) => MathF.Abs(x);
    public static int Abs(int x) => Math.Abs(x);
    public static float Max(float x, float y) => MathF.Max(x, y);
    public static int Max(int x, int y) => Math.Max(x, y);
    public static float Min(float x, float y) => MathF.Min(x, y);
    public static int Min(int x, int y) => Math.Min(x, y);
    public static float Clamp(float x, float min, float max) => Math.Clamp(x, min, max);
    public static int Clamp(int x, int min, int max) => Math.Clamp(x, min, max);
    public static float Clamp01(float x) => Math.Clamp(x, 0, 1);
    public static float Sqrt(float x) => MathF.Sqrt(x);
    public static float Sign(float x) => x >= 0 ? 1 : -1;
    public static float Lerp(float a, float b, float t) => a + (b - a) * Clamp01(t);
    public static int FloorToInt(float x) => (int)MathF.Floor(x);
    public static int CeilToInt(float x) => (int)MathF.Ceiling(x);
    public static int RoundToInt(float x) => (int)MathF.Round(x, MidpointRounding.ToEven);
    public static float Atan2(float y, float x) => MathF.Atan2(y, x);
}
internal sealed class ScopedEffectSet(Dictionary<string, long> flags, string prefix)
{
    public bool Add(string key) { bool was = Contains(key); flags[prefix + key] = 1; return !was; }
    public bool Contains(string key) => flags.GetValueOrDefault(prefix + key) != 0;
    public void Clear() { foreach (var key in flags.Keys.Where(x => x.StartsWith(prefix, StringComparison.Ordinal)).ToArray()) flags.Remove(key); }
}
