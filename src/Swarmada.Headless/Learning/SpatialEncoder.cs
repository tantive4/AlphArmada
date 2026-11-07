using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Search;

namespace Swarmada.Headless.Learning;

/// <summary>Single-owner direct game-object spatial encoder. The first eight SS/
/// SQ pair fields are presence, same owner, normalized observer dx/dy, footprint
/// distance/long range, center distance/board diagonal, and source-local bearing.
/// Each four-field zone tuple is physical eligibility, distance/long range,
/// geometric range enum/4 and RAW physical obstruction. Zero eligibility masks
/// the remaining tuple, rather than claiming a clear line of sight. Ship/ship
/// has all 6x6 zone tuples; ship/squad has six zones in each direction.
/// QQ fields: pair presence, same owner, dx/dy, footprint distance, distance-one,
/// raw engagement, movement inhibition, engagement permission to attack ships,
/// physical long-range eligibility, physical obstructed, physical clear.
/// These are never complete native attack legality; effective card/damage/range,
/// ignition and phase permissions belong to the decision/candidate encoder.</summary>
public sealed partial class SpatialEncoder
{
    private readonly record struct Blocker(LineOfSightObstacle Shape, int Ship, int Station);
    private readonly object topologyKey;
    private readonly int ships, squadrons;
    private readonly int[] placed;
    private readonly bool[] live;
    private readonly OrientedRectangle[] rectangles;
    private readonly Circle[] circles;
    private readonly HullZoneGeometry?[] zones;
    private readonly PolygonFootprint?[] stations;
    private Blocker[] blockers = [];
    private LineOfSightObstacle[] pairBlockers = [];
    private GameContext? activeSquadronContext;
    private SquadronView[] activeSquadrons = [];
    private ulong lastStamp;
    private bool initialized;
    private readonly SpatialEncoding output;
    public bool GeometryCacheHit { get; private set; }
    public long GeometryBuilds { get; private set; }
    public long GeometryQueries { get; private set; }
    public long OwnedCopies { get; private set; }
    public SpatialEncoderOptions Options { get; }

    public SpatialEncoder(GameSession prepared, SpatialEncoderOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(prepared); Options = options ?? new();
        if (Options.Height < 8 || Options.Width < 8) throw new ArgumentOutOfRangeException(nameof(options));
        topologyKey = prepared.Context.Topology; ships = prepared.Context.State.Ships.Length;
        squadrons = prepared.Context.State.Squadrons.Length;
        output = new(ships, squadrons, prepared.Context.Topology.BoardSizeMillimetres, Options);
        placed = new int[ships + squadrons]; live = new bool[placed.Length];
        rectangles = new OrientedRectangle[ships]; circles = new Circle[squadrons];
        zones = new HullZoneGeometry?[ships * 6]; stations = new PolygonFootprint?[ships];
        for (int entity = 0; entity < placed.Length; entity++) placed[entity] = prepared.Context.Topology.Registers[$"entity.{entity}.placed"];
        InitializeTerrain();
    }

    public SpatialEncoding Encode(GameSession position, int observer)
    { var borrowed = EncodeBorrowed(position, observer); OwnedCopies++; return borrowed.Clone(); }

    /// <summary>Borrowed arrays are overwritten by the next call on this worker.
    /// Call Clone before queueing inference, keeping a tree observation or replay.</summary>
    public SpatialEncoding EncodeBorrowed(GameSession position, int observer)
    {
        ArgumentNullException.ThrowIfNull(position);
        if (observer is not (1 or 2)) throw new ArgumentOutOfRangeException(nameof(observer));
        var game = position.Context;
        if (!ReferenceEquals(topologyKey, game.Topology)) throw new ArgumentException("Use one spatial encoder per prepared match.", nameof(position));
        using var scope = new GameScope(game);
        ulong stamp = SceneStamp(game, observer);
        GeometryCacheHit = initialized && stamp == lastStamp;
        if (!GeometryCacheHit)
        {
            PrepareScene(game); EncodePairs(game, observer); EncodePhysicalTuples(game);
            EncodeTerrain(game, observer);
            lastStamp = stamp; initialized = true; GeometryBuilds++;
        }
        else TerrainCacheHit = true;
        RefreshInhibition(game); RefreshMetadata(game, observer); output.Observer = observer;
        return output;
    }

    private ulong SceneStamp(GameContext game, int observer)
    {
        ulong hash = SearchHash.Add(SearchHash.Initial, (ulong)observer);
        hash = SearchHash.Add(hash, OfficialObjectives.Id(game));
        void Number(float value) => hash = SearchHash.Add(hash, BitConverter.SingleToUInt32Bits(value));
        for (int entity = 0; entity < live.Length; entity++)
        {
            live[entity] = game.State.Registers[placed[entity]] != 0 && !game.IsReserved(entity) && !game.IsDestroyed(entity);
            hash = SearchHash.Add(hash, live[entity] ? 1UL : 0); hash = SearchHash.Add(hash, (ulong)game.State.Owners[entity]);
            // Nonboard coordinates are unknown/irrelevant to physical features.
            if (!live[entity]) continue;
            var pose = game.State.Poses[entity]; Number(pose.Position.X); Number(pose.Position.Y); Number(pose.YawDegrees);
        }
        foreach (var ship in game.Topology.Ships)
        { Number(ship.FootprintMillimetres.X); Number(ship.FootprintMillimetres.Y); hash = SearchHash.Add(hash, game.State.Ships[ship.ShipIndex].HasAuxiliaryHullZones ? 1UL : 0); }
        for (int index = 0; index < game.State.SpatialObjects.Count; index++)
        {
            var piece = game.State.SpatialObjects[index]; if (piece.Kind != 0) continue;
            hash = SearchHash.Add(hash, (ulong)index); hash = SearchHash.Add(hash, piece.Active ? 1UL : 0);
            if (!piece.Active) continue;
            Number(piece.Pose.Position.X); Number(piece.Pose.Position.Y); Number(piece.Pose.YawDegrees);
            Number(piece.Size.X); Number(piece.Size.Y); hash = SearchHash.Add(hash, piece.Flipped ? 1UL : 0);
            hash = SearchHash.Add(hash, piece.ShapeId); hash = SearchHash.Add(hash, OfficialObjectives.ObstacleId(game, index));
        }
        if (OfficialObjectives.State(game) is { } objective)
            foreach (var pair in objective.PieceEntities) { hash = SearchHash.Add(hash, (ulong)pair.Key); hash = SearchHash.Add(hash, (ulong)pair.Value); }
        return hash;
    }

    private void PrepareScene(GameContext game)
    {
        Array.Clear(zones); Array.Clear(stations);
        var scene = new List<Blocker>();
        for (int ship = 0; ship < ships; ship++)
        {
            rectangles[ship] = game.ShipBase(ship);
            if (!live[ship]) continue;
            foreach (var zone in game.State.Ships[ship].HullZones)
                zones[ship * 6 + (int)zone] = game.State.Ships[ship].HasAuxiliaryHullZones
                    ? HullGeometry.Huge(rectangles[ship], zone, game.Topology.Ships[ship].CatalogEntry.Definition.ShipType == "prototype-super-star-destroyer")
                    : HullGeometry.Standard(rectangles[ship], zone);
            if (game.Topology.Ships[ship].IsStation)
            {
                if (OfficialObjectives.State(game) is { } objective)
                    foreach (var pair in objective.PieceEntities) if (pair.Value == ship) { stations[ship] = ObstacleGeometry.Shape(game, pair.Key); break; }
                stations[ship] ??= new(ObstacleShapeCatalog.Get("station"), game.State.Poses[ship], game.Topology.Ships[ship].FootprintMillimetres);
            }
            else scene.Add(new(new(rectangles[ship]), ship, -1));
        }
        for (int squad = 0; squad < squadrons; squad++) circles[squad] = game.SquadronBase(ships + squad);
        for (int index = 0; index < game.State.SpatialObjects.Count; index++)
        {
            var piece = game.State.SpatialObjects[index];
            if (!piece.Active || piece.Kind != 0 || OfficialObjectives.Is(game, "contested-outpost") && OfficialObjectives.ObstacleId(game, index) == "station") continue;
            int station = OfficialObjectives.State(game)?.PieceEntities.GetValueOrDefault(index, -1) ?? -1;
            scene.Add(new(new(ObstacleGeometry.Shape(game, index)), -1, station));
        }
        blockers = scene.ToArray(); if (pairBlockers.Length < blockers.Length) pairBlockers = new LineOfSightObstacle[blockers.Length];
        BindActiveSquadrons(game);
    }

    private void EncodePairs(GameContext game, int observer)
    {
        Array.Clear(output.ShipShipRelations); Array.Clear(output.ShipSquadRelations); Array.Clear(output.SquadSquadRelations);
        for (int source = 0; source < ships; source++) if (live[source])
        {
            for (int target = 0; target < ships; target++) if (source != target && live[target])
                Pair(game, source, target, Geometry2D.Distance(rectangles[source], rectangles[target]),
                    output.ShipShipRelations.AsSpan((source * ships + target) * SpatialEncoding.ShipShipWidth, 8), observer);
            for (int squad = 0; squad < squadrons; squad++) if (live[ships + squad])
                Pair(game, source, ships + squad, Geometry2D.Distance(circles[squad], rectangles[source]),
                    output.ShipSquadRelations.AsSpan((source * squadrons + squad) * SpatialEncoding.ShipSquadWidth, 8), observer);
        }
        var board = game.Topology.BoardSizeMillimetres;
        for (int source = 0; source < squadrons; source++) for (int target = 0; target < squadrons; target++)
        {
            int entity = ships + source, other = ships + target;
            if (source == target || !live[entity] || !live[other]) continue;
            var row = output.SquadSquadRelations.AsSpan((source * squadrons + target) * SpatialEncoding.SquadSquadWidth, SpatialEncoding.SquadSquadWidth);
            var delta = game.State.Poses[other].Position - game.State.Poses[entity].Position;
            float distance = Geometry2D.Distance(circles[source], circles[target]);
            row[0] = 1; row[1] = game.State.Owners[entity] == game.State.Owners[other] ? 1 : 0;
            row[2] = (observer == 1 ? delta.X : -delta.X) / board.X; row[3] = (observer == 1 ? delta.Y : -delta.Y) / board.Y;
            row[4] = distance / TabletopMeasurements.LongRange; row[5] = distance <= TabletopMeasurements.DistanceOne ? 1 : 0;
            row[6] = SquadronAbilityEngine.IsEngaged(game.SquadronViews[source], game.SquadronViews[target]) ? 1 : 0;
        }
    }
    private static void Pair(GameContext game, int source, int target, float distance, Span<float> row, int observer)
    {
        var board = game.Topology.BoardSizeMillimetres; var delta = game.State.Poses[target].Position - game.State.Poses[source].Position;
        float length = delta.Length(); var direction = length > 0 ? delta / length : Vector2.Zero;
        row[0] = 1; row[1] = game.State.Owners[source] == game.State.Owners[target] ? 1 : 0;
        row[2] = (observer == 1 ? delta.X : -delta.X) / board.X; row[3] = (observer == 1 ? delta.Y : -delta.Y) / board.Y;
        row[4] = distance / TabletopMeasurements.LongRange; row[5] = length / board.Length();
        row[6] = Vector2.Dot(direction, game.State.Poses[source].Right); row[7] = Vector2.Dot(direction, game.State.Poses[source].Forward);
    }

    private void EncodePhysicalTuples(GameContext game)
    {
        GeometryQueries = 0;
        for (int source = 0; source < ships; source++) if (live[source])
        {
            for (int target = 0; target < ships; target++)
            {
                if (source == target || !live[target] || game.State.Owners[source] == game.State.Owners[target]) continue;
                int count = PreparePair(source, target);
                foreach (var from in game.State.Ships[source].HullZones) foreach (var to in game.State.Ships[target].HullZones)
                {
                    if (stations[target] is not null && to != HullZone.Front) continue;
                    var attack = zones[source * 6 + (int)from]!;
                    var defense = stations[target] is { } station ? StationZone(station, attack.TargetingPoint) : zones[target * 6 + (int)to]!;
                    GeometryQueries++;
                    if (stations[target] is { } targetStation && Geometry2D.Overlaps(rectangles[source], targetStation)) continue;
                    var measurement = HullGeometry.MeasureAttack(attack, defense);
                    if (!measurement.HasEligibleDefenderEdge || stations[target] is null && !HullGeometry.IsTargetZoneExposed(attack.TargetingPoint, defense, rectangles[target])) continue;
                    Fill(output.ShipShipRelations.AsSpan((source * ships + target) * SpatialEncoding.ShipShipWidth + 8 + ((int)from * 6 + (int)to) * 4, 4),
                        measurement.Distance, new(attack.TargetingPoint, defense.TargetingPoint), count);
                }
            }
            for (int squad = 0; squad < squadrons; squad++)
            {
                int entity = ships + squad;
                if (!live[entity] || game.State.Owners[source] == game.State.Owners[entity]) continue;
                int count = PreparePair(source, entity), offset = (source * squadrons + squad) * SpatialEncoding.ShipSquadWidth + 8;
                foreach (var zone in game.State.Ships[source].HullZones)
                {
                    var hull = zones[source * 6 + (int)zone]!; var circle = circles[squad]; GeometryQueries++;
                    if (hull.ContainsInArc(circle.Center) || ArcIntersectsCircle(hull, circle))
                        Fill(output.ShipSquadRelations.AsSpan(offset + (int)zone * 4, 4), HullGeometry.MeasureToSquadron(hull, circle).Distance,
                            new(hull.TargetingPoint, circle.Center), count);
                    GeometryQueries++;
                    if (stations[source] is not null && zone != HullZone.Front) continue;
                    var defense = stations[source] is { } station ? StationZone(station, circle.Center) : hull;
                    if (stations[source] is { } stationShape && Geometry2D.Overlaps(circle, stationShape)) continue;
                    if (stations[source] is null && !HullGeometry.IsTargetZoneExposed(circle.Center, defense, rectangles[source])) continue;
                    float distance = float.PositiveInfinity;
                    foreach (var edge in defense.Edges) distance = MathF.Min(distance,
                        MathF.Max(0, Vector2.Distance(circle.Center, Geometry2D.ClosestPoint(edge, circle.Center)) - circle.Radius));
                    Fill(output.ShipSquadRelations.AsSpan(offset + 24 + (int)zone * 4, 4), distance, new(circle.Center, defense.TargetingPoint), count);
                }
            }
        }
        int squadBlockers = PreparePair(-1, -1);
        for (int source = 0; source < squadrons; source++) for (int target = 0; target < squadrons; target++)
        {
            int entity = ships + source, other = ships + target;
            if (source == target || !live[entity] || !live[other] || game.State.Owners[entity] == game.State.Owners[other]) continue;
            GeometryQueries++; float distance = Geometry2D.Distance(circles[source], circles[target]);
            if (distance > TabletopMeasurements.LongRange) continue;
            bool obstructed = Obstructed(new(circles[source].Center, circles[target].Center), squadBlockers);
            int at = (source * squadrons + target) * SpatialEncoding.SquadSquadWidth;
            output.SquadSquadRelations[at + 9] = 1; output.SquadSquadRelations[at + 10] = obstructed ? 1 : 0;
            output.SquadSquadRelations[at + 11] = obstructed ? 0 : 1;
        }
    }

    private void RefreshInhibition(GameContext game)
    {
        // Handles retain a GameContext. A physically equivalent clone may still
        // have different card/temporary state, so rebind when its context changes.
        if (!ReferenceEquals(activeSquadronContext, game)) BindActiveSquadrons(game);
        for (int source = 0; source < squadrons; source++)
        {
            if (!live[ships + source]) continue;
            bool prevented = SquadronAbilityEngine.IsPreventedFromMoving(game.SquadronViews[source], activeSquadrons);
            bool attackShip = SquadronAbilityEngine.CanAttackShip(game.SquadronViews[source], activeSquadrons);
            for (int target = 0; target < squadrons; target++)
            {
                int at = (source * squadrons + target) * SpatialEncoding.SquadSquadWidth;
                if (output.SquadSquadRelations[at] == 0) continue;
                output.SquadSquadRelations[at + 7] = prevented ? 1 : 0; output.SquadSquadRelations[at + 8] = attackShip ? 1 : 0;
            }
        }
    }
    private void BindActiveSquadrons(GameContext game)
    {
        int count = 0;
        for (int squad = 0; squad < squadrons; squad++) if (live[ships + squad]) count++;
        if (activeSquadrons.Length != count) activeSquadrons = new SquadronView[count];
        count = 0;
        for (int squad = 0; squad < squadrons; squad++) if (live[ships + squad]) activeSquadrons[count++] = game.SquadronViews[squad];
        activeSquadronContext = game;
    }

    private void RefreshMetadata(GameContext game, int observer)
    {
        Array.Clear(output.ShipCoordinates); Array.Clear(output.SquadronCoordinates); Array.Clear(output.SquadronRadii);
        var half = game.Topology.BoardSizeMillimetres / 2;
        for (int entity = 0; entity < live.Length; entity++)
        {
            bool squadron = entity >= ships, known = game.State.Registers[placed[entity]] != 0 && !game.IsReserved(entity);
            var raw = game.State.Poses[entity]; var pose = known ? Canonical(raw, observer) : new Pose(Vector2.Zero);
            var size = squadron ? game.Topology.Squadrons[entity - ships].FootprintMillimetres : game.Topology.Ships[entity].FootprintMillimetres;
            output.UnitPoses[entity] = new(entity, RelativeOwner(game.State.Owners[entity], observer), known, live[entity], squadron,
                pose, size, squadron ? "circle" : game.Topology.Ships[entity].CatalogEntry.Definition.ShipType);
            if (!squadron)
            {
                output.ShipOnBoard[entity] = live[entity] ? (byte)1 : (byte)0;
                if (!live[entity]) continue;
                int at = entity * SpatialEncoding.ShipCoordinateWidth;
                output.ShipCoordinates[at] = pose.Position.X / half.X; output.ShipCoordinates[at + 1] = pose.Position.Y / half.Y;
                float angle = pose.YawDegrees * MathF.PI / 180;
                output.ShipCoordinates[at + 2] = MathF.Sin(angle); output.ShipCoordinates[at + 3] = MathF.Cos(angle);
            }
            else
            {
                int index = entity - ships; output.SquadronOnBoard[index] = live[entity] ? (byte)1 : (byte)0;
                if (!live[entity]) continue;
                output.SquadronCoordinates[index * 2] = pose.Position.X / half.X; output.SquadronCoordinates[index * 2 + 1] = pose.Position.Y / half.Y;
                output.SquadronRadii[index * 2] = size.X / 2 / half.X; output.SquadronRadii[index * 2 + 1] = size.X / 2 / half.Y;
            }
        }
        if (output.Pieces.Length != game.State.SpatialObjects.Count) output.Pieces = new SpatialPiecePose[game.State.SpatialObjects.Count];
        for (int index = 0; index < output.Pieces.Length; index++)
        {
            var piece = game.State.SpatialObjects[index];
            string shape = piece.Kind == 0 ? ObstacleShapeCatalog.CanonicalId(string.IsNullOrEmpty(piece.ShapeId)
                ? OfficialObjectives.ObstacleId(game, index) : piece.ShapeId) : piece.ShapeId;
            output.Pieces[index] = new(index, piece.Kind, RelativeOwner(piece.Owner, observer), piece.SourceEntity, piece.Active,
                Canonical(piece.Pose, observer), piece.Size, piece.Flipped, piece.Id, shape);
        }
    }
    private static Pose Canonical(Pose pose, int observer) => observer == 1 ? pose : new(-pose.Position, pose.YawDegrees + 180);
    private static int RelativeOwner(int owner, int observer) => owner == 0 ? 0 : owner == observer ? 1 : -1;
    private int PreparePair(int source, int target)
    {
        int count = 0;
        foreach (var blocker in blockers)
            if ((blocker.Ship < 0 || blocker.Ship != source && blocker.Ship != target) &&
                (blocker.Station < 0 || blocker.Station != source && blocker.Station != target)) pairBlockers[count++] = blocker.Shape;
        return count;
    }
    private void Fill(Span<float> row, float distance, Segment line, int blockerCount)
    {
        if (!float.IsFinite(distance) || distance > TabletopMeasurements.LongRange) return;
        row[0] = 1; row[1] = distance / TabletopMeasurements.LongRange;
        row[2] = (int)HullGeometry.ClassifyRange(distance) / 4f; row[3] = Obstructed(line, blockerCount) ? 1 : 0;
    }
    private bool Obstructed(Segment line, int count) => LineOfSightGeometry.Evaluate(line, pairBlockers.AsSpan(0, count)) == LineOfSightResult.Obstructed;
    private static HullZoneGeometry StationZone(PolygonFootprint shape, Vector2 reference)
    {
        var edges = new Segment[shape.Count]; for (int i = 0; i < edges.Length; i++) edges[i] = shape.Edge(i);
        return new(HullZone.Front, shape.Center, new(-1, 1), new(1, 1), shape.ClosestBoundaryPoint(reference), edges);
    }
    private static bool ArcIntersectsCircle(HullZoneGeometry zone, Circle circle)
        => DistanceToRay(zone.ArcOrigin, zone.FirstBoundary, circle.Center) <= circle.Radius || DistanceToRay(zone.ArcOrigin, zone.SecondBoundary, circle.Center) <= circle.Radius;
    private static float DistanceToRay(Vector2 origin, Vector2 direction, Vector2 point)
        => Vector2.Distance(point, origin + direction * MathF.Max(0, Vector2.Dot(point - origin, direction)));
}
