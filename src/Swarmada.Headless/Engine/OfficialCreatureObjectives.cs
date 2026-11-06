using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Armada.Session;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;
using PlayAreaGeometry = Swarmada.Headless.Geometry.PlayAreaGeometry;

namespace Swarmada.Headless.Engine;

/// <summary>Gravity rifts and neutral creatures use their selected physical polygon
/// footprints; no creature is a mutable fleet entity.</summary>
public static class OfficialCreatureObjectives
{
    private const string Prefix = "objective.creature.";
    public static bool Handles(string rule) => rule.StartsWith(Prefix, StringComparison.Ordinal);
    private static IEnumerable<int> Pieces(GameContext game, string kind, bool active = true)
    {
        for (int i = 0; i < game.State.SpatialObjects.Count; i++)
            if (game.State.SpatialObjects[i].Kind == 0 && (!active || game.State.SpatialObjects[i].Active)
                && OfficialObjectives.ObstacleId(game, i).StartsWith(kind, StringComparison.Ordinal)) yield return i;
    }
    private static int Anchor(GameContext game, int owner) => game.Topology.Ships.First(s => s.Owner == owner).EntityIndex;
    private static int Second(GameContext game) => OfficialObjectives.SecondPlayer(game);
    public static IEnumerable<GameContinuation> SetupChoices(GameContext game, bool afterDeployment = false)
    {
        if (!afterDeployment && OfficialObjectives.Is(game, "hyperspace-migration")) EnsureMigrationPurrgils(game);
        if (afterDeployment && OfficialObjectives.Is(game, "hyperspace-migration") && !MapMovementRules.Enabled(game))
        {
            // The static map must be complete before command planning/search.
            // Keep the initial ships in place so creature placement still checks
            // their actual bases, then never offer a ship-phase spawn afterward.
            EnsureMigrationPurrgils(game);
            yield return new(Prefix + "migration-spawn", Anchor(game, Second(game)), Text: "initial");
        }
        if (!afterDeployment && OfficialObjectives.Is(game, "infested-fields"))
        { EnsureExogorths(game); yield return new(Prefix + "place-exogorths", Anchor(game, Second(game)), Indices: [], Text: "initial"); }
    }
    public static IEnumerable<GameContinuation> TimingChoices(GameContext game, string timing, int source, int target)
    {
        string id = OfficialObjectives.Id(game);
        if (timing == "start-ship-phase" && id == "hyperspace-migration" && MapMovementRules.Enabled(game) &&
            game.State.Round is (1 or 3))
        {
            EnsureMigrationPurrgils(game);
            yield return new(Prefix + "migration-spawn", Anchor(game, Second(game)));
        }
        if (timing == "after-start-squadron-phase" && id is ("infested-fields" or "asteroid-tactics") && MapMovementRules.Enabled(game))
            yield return new(Prefix + "remove-exogorths", Anchor(game, Second(game)));
        if (timing == "end-round")
        {
            if (MapMovementRules.Enabled(game) && id is ("infested-fields" or "asteroid-tactics"))
            { EnsureExogorths(game); yield return new(Prefix + "place-exogorths", Anchor(game, Second(game)), Indices: []); }
            if (MapMovementRules.Enabled(game) && Pieces(game, "purrgil").Any()) yield return new(Prefix + "purrgil-round", Anchor(game, Second(game)), Value: Second(game), Indices: []);
        }
    }
    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        if (timing == "start-squadron-phase" && Pieces(game, "exogorth").Any())
            yield return new(new(GameActionKind.UseAbility, Anchor(game, Second(game)), AbilityId: Prefix + "exogorth-start"), "Resolve mandatory exogorth attacks");
    }
    public static bool TryApply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (action.AbilityId != Prefix + "exogorth-start") return false;
        EngineEffects.PushChoice(game, new(Prefix + "exogorth-round", action.Source, Value: Second(game), Indices: []), []); return true;
    }
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        if (IsDisabledMapContinuation(game, frame)) return Forced("Skip disabled map-piece relocation");
        if (frame.Rule is Prefix + "ship-obstacles" or Prefix + "squadron-move" && !OfficialObjectives.Live(game, frame.Source))
            return Forced("Finish destroyed unit's obstacle effects");
        var options = new List<GameActionOption>(); int player = frame.Source >= 0 ? game.State.Owners[frame.Source] : Second(game);
        switch (frame.Rule)
        {
            case Prefix + "place-exogorths":
                foreach (int piece in Pieces(game, "exogorth", false).Where(i => !game.State.SpatialObjects[i].Active))
                    options.Add(new(new(GameActionKind.PlaceToken, frame.Source, piece, AbilityId: frame.Rule),
                        "Place " + game.State.SpatialObjects[piece].ShapeId + " touching an eligible obstacle", true, true, AcceptsFlip: true));
                if (options.Count == 0) return Forced("Finish exogorth placement");
                break;
            case Prefix + "choose-exogorth-host":
                foreach (int host in frame.Indices ?? [])
                    options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, host, Index: frame.Target, AbilityId: frame.Rule), "Choose this exogorth's persistent initial host"));
                break;
            case Prefix + "exogorth-displace":
                player = 3 - player;
                foreach (int squad in frame.Indices ?? [])
                    if (OfficialObjectives.Live(game, squad)) options.Add(new(new(GameActionKind.DisplaceSquadron, squad, frame.Target, AbilityId: frame.Rule), "Place overlapped squadron touching the flat obstacle", true));
                if (options.Count == 0) return Forced("Finish obstacle displacement");
                break;
            case Prefix + "exogorth-round":
                player = (int)frame.Value;
                foreach (int exo in Pieces(game, "exogorth").Where(i => !(frame.Indices ?? []).Contains(i)))
                    options.Add(new(new(GameActionKind.ChooseEntity, Anchor(game, player), exo, AbilityId: frame.Rule), "Choose an exogorth to attack"));
                if (options.Count == 0) return Forced("Finish exogorth attacks");
                break;
            case Prefix + "exogorth-piece":
                foreach (var squad in game.Topology.Squadrons)
                    if (OfficialObjectives.Live(game, squad.EntityIndex) && !(frame.Indices ?? []).Contains(squad.EntityIndex)
                        && OfficialObjectives.DistanceToPiece(game, squad.EntityIndex, frame.Target) <= TabletopMeasurements.DistanceOne)
                        options.Add(new(new(GameActionKind.Attack, frame.Source, squad.EntityIndex, Index: frame.Target, AbilityId: frame.Rule), "Exogorth attacks this squadron"));
                if (options.Count == 0) return Forced("Finish this exogorth's attacks");
                break;
            case Prefix + "remove-exogorths": return Forced("Remove exogorths after their attacks");
            case Prefix + "infested-host":
                options.Add(new(new(GameActionKind.UseAbility, frame.Source, frame.Target, AbilityId: frame.Rule), "Move the exogorth's former host within distance 1-2", true, true, AcceptsFlip: true)); break;
            case Prefix + "purrgil-round":
                player = (int)frame.Value;
                foreach (int purrgil in Pieces(game, "purrgil").Where(i => !(frame.Indices ?? []).Contains(i)))
                    options.Add(new(new(GameActionKind.ChooseEntity, Anchor(game, player), purrgil, AbilityId: frame.Rule), "Choose a purrgil to move"));
                if (options.Count == 0) return Forced("Finish purrgil movement");
                break;
            case Prefix + "infested-hosts":
                foreach (int host in frame.Indices ?? []) options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, host, AbilityId: frame.Rule), "Choose the next former exogorth host to move"));
                if (options.Count == 0) return Forced("Finish former host movement");
                break;
            case Prefix + "purrgil-move":
                options.Add(new(new(GameActionKind.UseAbility, frame.Source, frame.Target, AbilityId: frame.Rule), "Move purrgil within distance 1-2", true, true, AcceptsFlip: true)); break;
            case Prefix + "migration-spawn":
                int[] available = MigrationPurrgils(game).Where(i => !game.State.SpatialObjects[i].Active).ToArray();
                if (available.Length == 0) return Forced("No purrgil is outside the play area");
                if (!OfficialObjectives.State(game)!.Flags.ContainsKey("creature.migration.entry"))
                    foreach (int token in MigrationTokens(game)) options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, token, AbilityId: frame.Rule), "Choose the migration entry token for both spawns"));
                else if (frame.Target < 0)
                    foreach (int purrgil in available)
                        options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, purrgil, AbilityId: frame.Rule), "Choose " + game.State.SpatialObjects[purrgil].ShapeId + " outside the play area"));
                else options.Add(new(new(GameActionKind.PlaceToken, frame.Source, frame.Target, AbilityId: frame.Rule), "Place the chosen purrgil at distance 1 of migration entry token", true, true, AcceptsFlip: true));
                break;
            case Prefix + "squadron-move":
            case Prefix + "ship-obstacles":
                foreach (int obstacle in frame.Indices ?? []) options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, obstacle, AbilityId: frame.Rule), "Choose the next obstacle effect to resolve"));
                if (options.Count == 0) return Forced("Finish obstacle overlap effects");
                break;
            case Prefix + "squadron-obstacle": return Forced("Resolve squadron obstacle overlap");
            case Prefix + "ship-overlap": return Forced("Resolve ship obstacle effects");
            case Prefix + "station-repair":
                var state = game.State.Ships[frame.Source];
                if (state.FacedownDamageCardCount > 0) options.Add(new(new(GameActionKind.RepairHull, frame.Source, AbilityId: frame.Rule), "Discard one facedown damage card"));
                foreach (var card in state.GetFaceupCriticalCards().Distinct())
                    options.Add(new(new(GameActionKind.RepairHull, frame.Source, Index: (int)card, Amount: 1, AbilityId: frame.Rule), "Discard faceup " + card));
                options.Add(new(new(GameActionKind.Pass, frame.Source), "Decline station repair"));
                break;
            case Prefix + "station-squadron-repair":
                options.Add(new(new(GameActionKind.UseAbility, frame.Source, AbilityId: frame.Rule), "Recover one squadron hull point"));
                options.Add(new(new(GameActionKind.Pass, frame.Source), "Decline station repair"));
                break;
            default: throw new NotSupportedException(frame.Rule);
        }
        return new(options.Count == 1 && options[0].Template.Kind is GameActionKind.ChooseEntity or GameActionKind.Attack ? DecisionKind.Forced : DecisionKind.Player,
            player, GamePhase.AbilityChoice, options);
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!GetDecision(game, frame).Actions.Any(o => GameSession.Matches(o, action))) throw new ArgumentException("Illegal creature rule choice.");
        if (IsDisabledMapContinuation(game, frame)) { Pop(game, frame); return; }
        switch (frame.Rule)
        {
            case Prefix + "place-exogorths":
                if (action.Kind == GameActionKind.Pass) { Pop(game, frame); return; }
                PlaceExogorth(game, frame, action); return;
            case Prefix + "choose-exogorth-host":
                OfficialObjectives.State(game)!.ExogorthHosts[frame.Target] = action.Target;
                Pop(game, frame); QueueExogorthDisplacement(game, frame.Source, frame.Target); return;
            case Prefix + "exogorth-displace":
                if (action.Kind == GameActionKind.Pass) { Pop(game, frame); game.Flow.Phase = (GamePhase)frame.Stage; return; }
                DisplaceSquadron(game, frame, action); return;
            case Prefix + "exogorth-round":
                if (action.Kind == GameActionKind.Pass) { Pop(game, frame); return; }
                Replace(game, frame, frame with { Value = 3 - frame.Value, Indices = [..frame.Indices ?? [], action.Target] });
                EngineEffects.PushChoice(game, new(Prefix + "exogorth-piece", action.Source, action.Target, Indices: []), []); return;
            case Prefix + "exogorth-piece":
                if (action.Kind == GameActionKind.Pass) { Pop(game, frame); return; }
                Replace(game, frame, frame with { Indices = [..frame.Indices ?? [], action.Target] });
                CombatModule.QueueExogorthAttack(game, frame.Target, action.Target, GamePhase.AbilityChoice); return;
            case Prefix + "remove-exogorths":
                int[] exos = Pieces(game, "exogorth").ToArray(); var hosts = new List<int>();
                foreach (int exo in exos)
                {
                    if (!OfficialObjectives.Is(game, "infested-fields")) continue;
                    if (OfficialObjectives.State(game)!.ExogorthHosts.TryGetValue(exo, out int host) &&
                        host >= 0 && host < game.State.SpatialObjects.Count && game.State.SpatialObjects[host] is { Active: true, Kind: 0 } &&
                        !hosts.Contains(host)) hosts.Add(host);
                }
                foreach (int exo in exos)
                {
                    game.State.SpatialObjects[exo] = game.State.SpatialObjects[exo] with { Active = false };
                    TokenObstacleAttachments.RemoveObstacle(game, exo);
                }
                Pop(game, frame);
                if (hosts.Count > 0) EngineEffects.PushChoice(game, new(Prefix + "infested-hosts", frame.Source, Indices: hosts.ToArray()), []);
                return;
            case Prefix + "infested-hosts":
                if (action.Kind == GameActionKind.Pass) { Pop(game, frame); return; }
                Replace(game, frame, frame with { Indices = frame.Indices!.Where(i => i != action.Target).ToArray() });
                EngineEffects.PushChoice(game, new(Prefix + "infested-host", frame.Source, action.Target), []); return;
            case Prefix + "infested-host":
                OfficialObjectives.MoveObstacle(game, frame.Target, new(action.Position!.Value, action.YawDegrees), TabletopMeasurements.DistanceBand(2), flipped: action.Flipped);
                Pop(game, frame); return;
            case Prefix + "purrgil-round":
                if (action.Kind == GameActionKind.Pass) { Pop(game, frame); return; }
                Replace(game, frame, frame with { Value = 3 - frame.Value, Indices = [..frame.Indices ?? [], action.Target] });
                EngineEffects.PushChoice(game, new(Prefix + "purrgil-move", action.Source, action.Target), []); return;
            case Prefix + "purrgil-move":
                OfficialObjectives.MoveObstacle(game, frame.Target, new(action.Position!.Value, action.YawDegrees), TabletopMeasurements.DistanceBand(2), flipped: action.Flipped);
                Pop(game, frame);
                OfficialObjectives.QueueChoices(game, OfficialAssaultObjectives.PurrgilMovedChoices(game, frame.Target, game.State.Owners[frame.Source]), GamePhase.AbilityChoice); return;
            case Prefix + "migration-spawn": SpawnMigrationPurrgil(game, frame, action); return;
            case Prefix + "squadron-move":
                if (action.Kind == GameActionKind.Pass) { ResolveSquadronMove(game, frame); return; }
                Replace(game, frame, frame with { Indices = frame.Indices!.Where(i => i != action.Target).ToArray() });
                EngineEffects.PushChoice(game, new(Prefix + "squadron-obstacle", frame.Source, action.Target), []); return;
            case Prefix + "squadron-obstacle": ResolveSquadronObstacle(game, frame); return;
            case Prefix + "ship-obstacles":
                if (action.Kind == GameActionKind.Pass) { Pop(game, frame); return; }
                Replace(game, frame, frame with { Indices = frame.Indices!.Where(i => i != action.Target).ToArray() });
                EngineEffects.PushChoice(game, new(Prefix + "ship-overlap", frame.Source, action.Target, Value: frame.Value), []); return;
            case Prefix + "ship-overlap": ResolveShipOverlap(game, frame); return;
            case Prefix + "station-repair":
                if (action.Kind != GameActionKind.Pass)
                {
                    var ship = game.State.Ships[frame.Source]; CriticalDamageCard discarded;
                    if (action.Amount == 1)
                    { discarded = (CriticalDamageCard)action.Index; if (ship.RepairFaceupDamage(discarded) != 1) throw new ArgumentException("Faceup card unavailable."); }
                    else if (!ship.TryRepairFacedownDamage(out discarded)) throw new ArgumentException("Facedown card unavailable.");
                    game.State.Decks[game.State.Owners[frame.Source] - 1].Discard(discarded);
                }
                Pop(game, frame); return;
            case Prefix + "station-squadron-repair":
                if (action.Kind != GameActionKind.Pass)
                    game.State.Squadrons[game.SquadronIndex(frame.Source)].RecoverHull(1, game.Topology.Squadrons[game.SquadronIndex(frame.Source)].CatalogEntry.Definition.Hull);
                Pop(game, frame); return;
        }
    }
    private static bool IsDisabledMapContinuation(GameContext game, GameContinuation frame)
        => !MapMovementRules.Enabled(game) && (frame.Rule is Prefix + "remove-exogorths" or Prefix + "infested-hosts" or
            Prefix + "infested-host" or Prefix + "purrgil-round" or Prefix + "purrgil-move" ||
            frame.Rule == Prefix + "place-exogorths" && (OfficialObjectives.Is(game, "asteroid-tactics") || frame.Text != "initial" || InitialSetupComplete(game)) ||
            frame.Rule == Prefix + "migration-spawn" && (game.State.Round != 1 || frame.Text != "initial" || InitialSetupComplete(game)));
    private static bool InitialSetupComplete(GameContext game)
        => game.State.Registers[game.Topology.Registers["setup.completed"]] != 0;
    private static void EnsureExogorths(GameContext game)
    {
        int count = Pieces(game, "exogorth", false).Count();
        for (; count < 2; count++) OfficialObjectives.AddObstacle(game, "exogorth", Vector2.Zero, OfficialObjectives.PieceSize(game, "exogorth"), active: false);
    }
    private static void PlaceExogorth(GameContext game, GameContinuation frame, GameAction action)
    {
        int exo = action.Target; var token = game.State.SpatialObjects[exo]; var pose = new Pose(action.Position!.Value, action.YawDegrees);
        var shape = ObstacleGeometry.Shape(game, exo, pose, action.Flipped);
        if (!PlayAreaGeometry.Contains(MovementModule.Board(game), shape)) throw new ArgumentException("Exogorth must be completely in the play area.");
        var hosts = new List<int>();
        foreach (int other in OfficialObjectives.Obstacles(game))
        {
            var obstacle = ObstacleGeometry.Shape(game, other);
            if (Geometry2D.Overlaps(shape, obstacle)) throw new ArgumentException("Exogorth cannot overlap another obstacle.");
            if (Geometry2D.Distance(shape, obstacle) <= Geometry2D.DefaultTolerance)
            {
                if (OfficialObjectives.ObstacleId(game, other).StartsWith("exogorth", StringComparison.Ordinal)) throw new ArgumentException("Exogorths cannot touch each other.");
                if (!Pieces(game, "exogorth").Any(existing => OfficialObjectives.State(game)!.ExogorthHosts.GetValueOrDefault(existing, -1) == other ||
                    Geometry2D.Distance(obstacle, ObstacleGeometry.Shape(game, existing)) <= Geometry2D.DefaultTolerance))
                    hosts.Add(other);
            }
        }
        if (hosts.Count == 0) throw new ArgumentException("Exogorth must touch an obstacle not assigned as another exogorth's host.");
        foreach (int ship in OfficialObjectives.LiveShips(game)) if (Geometry2D.Overlaps(shape, game.ShipBase(ship))) throw new ArgumentException("Exogorth cannot overlap a ship.");
        foreach (int objectiveToken in OfficialObjectives.TokenIndices(game)) if (Geometry2D.Overlaps(OfficialObjectives.PieceCircle(game, objectiveToken), shape)) throw new ArgumentException("Exogorth cannot overlap a token.");
        game.State.SpatialObjects[exo] = token with { Pose = pose, Active = true, Flipped = action.Flipped };
        var state = OfficialObjectives.State(game)!;
        state.ExogorthHosts.Remove(exo);
        if (hosts.Count > 1)
            EngineEffects.PushChoice(game, new(Prefix + "choose-exogorth-host", frame.Source, exo, Indices: hosts.ToArray()), []);
        else
        {
            state.ExogorthHosts[exo] = hosts[0]; QueueExogorthDisplacement(game, frame.Source, exo);
        }
    }
    private static void QueueExogorthDisplacement(GameContext game, int source, int exo)
    {
        BeginFlatObstacleDisplacements(game, exo, game.State.Owners[source], GamePhase.AbilityChoice);
    }
    /// <summary>The opponent of the player placing/moving the flat obstacle chooses
    /// the order and physical positions of its overlapped, lifted squadron bases.</summary>
    public static bool BeginFlatObstacleDisplacements(GameContext game, int obstacleIndex, int placingPlayer, GamePhase returnPhase)
    {
        if (placingPlayer is not (1 or 2)) throw new ArgumentOutOfRangeException(nameof(placingPlayer));
        var shape = ObstacleGeometry.Shape(game, obstacleIndex);
        int[] displaced = game.Topology.Squadrons.Where(s => OfficialObjectives.Live(game, s.EntityIndex) && Geometry2D.Overlaps(game.SquadronBase(s.EntityIndex), shape)).Select(s => s.EntityIndex).ToArray();
        if (displaced.Length == 0) return false;
        EngineEffects.PushChoice(game, new(Prefix + "exogorth-displace", Anchor(game, placingPlayer), obstacleIndex,
            Stage: (int)returnPhase, Indices: displaced), []);
        return true;
    }
    private static void DisplaceSquadron(GameContext game, GameContinuation frame, GameAction action)
    {
        int squad = action.Source; var position = action.Position!.Value; var pending = frame.Indices ?? [];
        if (!pending.Contains(squad)) throw new ArgumentException("This squadron was not overlapped.");
        var circle = new Circle(position, game.SquadronBase(squad).Radius);
        var shape = ObstacleGeometry.Shape(game, frame.Target);
        if (!SquadronDisplacementGeometry.IsLegalPosition(game, squad, position, pending)) throw new ArgumentException("Illegal exogorth displacement.");
        bool touching = SquadronDisplacementGeometry.Touches(circle, shape, flatObstacle: true);
        if (!touching && !SquadronDisplacementGeometry.CanPlaceTouching(game, squad, shape, pending, flatObstacle: true)) foreach (var other in game.Topology.Squadrons)
        {
            if (other.EntityIndex == squad || pending.Contains(other.EntityIndex) || !OfficialObjectives.Live(game, other.EntityIndex)) continue;
            var otherCircle = game.SquadronBase(other.EntityIndex);
            if (SquadronDisplacementGeometry.Touches(otherCircle, shape, flatObstacle: true) && Geometry2D.Contact(circle, otherCircle) == ContactKind.Touching) touching = true;
        }
        if (!touching) throw new ArgumentException("Displaced squadron must touch the exogorth or a touching squadron.");
        game.State.Poses[squad] = new(position, game.State.Poses[squad].YawDegrees);
        Replace(game, frame, frame with { Indices = pending.Where(e => e != squad).ToArray() });
    }
    private static IEnumerable<int> MigrationTokens(GameContext game)
    {
        var state = OfficialObjectives.State(game)!;
        for (int i = 0; i < 2; i++) if (state.Flags.TryGetValue("navigation.migration.token." + i, out long token)) yield return (int)token;
    }
    private static IEnumerable<int> MigrationPurrgils(GameContext game)
    {
        var state = OfficialObjectives.State(game)!;
        for (int i = 0; i < 2; i++)
            if (state.Flags.TryGetValue("creature.migration.piece." + i, out long piece)) yield return (int)piece;
    }
    private static void EnsureMigrationPurrgils(GameContext game)
    {
        var state = OfficialObjectives.State(game)!; var claimed = new HashSet<int>();
        for (int i = 0; i < 2; i++)
        {
            string key = "creature.migration.piece." + i, shapeId = "purrgil-" + (i + 1);
            if (state.Flags.TryGetValue(key, out long retained)) { claimed.Add((int)retained); continue; }
            int existing = Pieces(game, "purrgil", false).FirstOrDefault(index => !claimed.Contains(index) &&
                ObstacleGeometry.Shape(game, index).Definition.Id == shapeId, -1);
            if (existing < 0)
            {
                var definition = ObstacleShapeCatalog.Get(shapeId);
                existing = OfficialObjectives.AddObstacle(game, "purrgil", Vector2.Zero, definition.Size, active: false);
                game.State.SpatialObjects[existing] = game.State.SpatialObjects[existing] with { ShapeId = shapeId, Size = definition.Size };
            }
            state.Flags[key] = existing; claimed.Add(existing);
        }
    }
    private static void SpawnMigrationPurrgil(GameContext game, GameContinuation frame, GameAction action)
    {
        var state = OfficialObjectives.State(game)!;
        if (action.Kind == GameActionKind.Pass) { Pop(game, frame); return; }
        if (action.Kind == GameActionKind.ChooseEntity)
        {
            if (!state.Flags.ContainsKey("creature.migration.entry")) state.Flags["creature.migration.entry"] = action.Target;
            else Replace(game, frame, frame with { Target = action.Target });
            return;
        }
        int entry = (int)state.Flags["creature.migration.entry"]; var selected = game.State.SpatialObjects[frame.Target];
        var shape = ObstacleGeometry.Shape(game, frame.Target, new(action.Position!.Value, action.YawDegrees), action.Flipped);
        if (!PlayAreaGeometry.Contains(MovementModule.Board(game), shape) || Geometry2D.Distance(OfficialObjectives.PieceCircle(game, entry), shape) > TabletopMeasurements.DistanceOne)
            throw new ArgumentException("Purrgil must be placed at distance 1 of its migration entry token.");
        foreach (int obstacle in OfficialObjectives.Obstacles(game))
            if (Geometry2D.Overlaps(shape, ObstacleGeometry.Shape(game, obstacle))) throw new ArgumentException("Purrgil cannot overlap an obstacle.");
        foreach (int ship in OfficialObjectives.LiveShips(game)) if (Geometry2D.Overlaps(shape, game.ShipBase(ship))) throw new ArgumentException("Purrgil cannot overlap a ship.");
        foreach (var squad in game.Topology.Squadrons) if (OfficialObjectives.Live(game, squad.EntityIndex) && Geometry2D.Overlaps(game.SquadronBase(squad.EntityIndex), shape)) throw new ArgumentException("Purrgil cannot overlap a squadron.");
        game.State.SpatialObjects[frame.Target] = selected with { Pose = shape.Pose, Active = true, Flipped = action.Flipped }; Pop(game, frame);
    }
    public static void ApplyDeploymentSpeed(GameContext game, int ship)
    {
        if (IsNearGravityRift(game, ship))
            game.State.Ships[ship].SetSpeed(0, game.Topology.Ships[ship].CatalogEntry.Definition.MaximumSpeed);
    }
    public static bool IsNearGravityRift(GameContext game, int ship)
        => Pieces(game, "gravity-rift").Any(piece => OfficialObjectives.DistanceToPiece(game, ship, piece) <= TabletopMeasurements.DistanceBand(2));
    public static void ApplyGravityReduction(GameContext game, int ship)
    {
        string key = game.Topology.GravityRestoreKeys[ship];
        if (game.Flow.Flags.ContainsKey(key) || game.Flow.Flags.ContainsKey("movement.gravity-ignore." + ship)) return;
        if (!Pieces(game, "gravity-rift").Any(piece => OfficialObjectives.DistanceToPiece(game, ship, piece) <= TabletopMeasurements.DistanceBand(2))) return;
        game.Flow.Flags[key] = game.State.Ships[ship].CurrentSpeed + 1;
        game.State.Ships[ship].SetSpeed(Math.Max(0, game.State.Ships[ship].CurrentSpeed - 1), game.Topology.Ships[ship].CatalogEntry.Definition.MaximumSpeed);
    }
    public static void RestoreGravitySpeed(GameContext game, int ship)
    {
        if (game.Flow.Flags.Remove(game.Topology.GravityRestoreKeys[ship], out long original))
            game.State.Ships[ship].SetSpeed((int)original - 1, game.Topology.Ships[ship].CatalogEntry.Definition.MaximumSpeed);
    }
    public static void ApplyGravityOverlap(GameContext game, int ship, IReadOnlyList<int> obstacles)
    {
        if (!obstacles.Any(i => OfficialObjectives.ObstacleId(game, i) == "gravity-rift")) return;
        game.State.Ships[ship].SetSpeed(0, game.Topology.Ships[ship].CatalogEntry.Definition.MaximumSpeed);
        if (game.Flow.ExtraManeuvers.LastOrDefault() is { } extra && extra.Ship == ship) extra.OriginalSpeed = 0;
        if (game.Flow.ForcedManeuvers.LastOrDefault() is { } forced && forced.Ship == ship) forced.OriginalSpeed = 0;
    }
    /// <summary>Normal and bonus squadron moves resolve obstacle effects before after-move card effects.</summary>
    public static bool BeginSquadronMoveConsequences(GameContext game, int squadron, GamePhase returnPhase)
    {
        if (game.State.Outcome is not null || !OfficialObjectives.Live(game, squadron)) return false;
        int[] obstacles = OfficialObjectives.Obstacles(game).Where(i => Geometry2D.Overlaps(game.SquadronBase(squadron), ObstacleGeometry.Shape(game, i))).ToArray();
        bool mauler = game.Topology.Squadrons[game.SquadronIndex(squadron)].CatalogEntry.Definition.SquadronId == "mauler-mithel";
        if (obstacles.Length == 0 && !mauler) return false;
        EngineEffects.PushChoice(game, new(Prefix + "squadron-move", squadron, Value: (int)returnPhase, Indices: obstacles), []); return true;
    }
    public static bool BeginShipObstacleConsequences(GameContext game, MovementExecution move, GamePhase returnPhase)
    {
        if (move.OverlappedObstacles.Length == 0 || move.IgnoreObstacleEffects &&
            (move.IgnoredByJaina || !game.State.Ships[move.Ship].DealsExtraDamageWhenOverlapping)) return false;
        long flags = (move.IgnoreObstacleEffects ? 1L : 0L) | (move.IgnoredByJaina ? 2L : 0L) | ((long)move.ExecutedCourse.Length << 2);
        return OfficialObjectives.QueueChoices(game, [new(Prefix + "ship-obstacles", move.Ship, Value: flags, Indices: move.OverlappedObstacles.ToArray())], returnPhase);
    }
    private static void ResolveShipOverlap(GameContext game, GameContinuation frame)
    {
        if (!OfficialObjectives.Live(game, frame.Source)) { Pop(game, frame); return; }
        var next = frame with { Stage = frame.Stage + 1 }; Replace(game, frame, next);
        bool ignored = (frame.Value & 1) != 0, jaina = (frame.Value & 2) != 0;
        if (frame.Stage == 0 && !jaina && game.State.Ships[frame.Source].DealsExtraDamageWhenOverlapping)
        { CombatModule.QueueDamage(game, frame.Source, 1, false, true, GamePhase.AbilityChoice, rule: "damaged-controls.obstacle"); return; }
        if (frame.Stage == 1 && !ignored)
        {
            string kind = OfficialObjectives.ObstacleId(game, frame.Target);
            if (kind == "gravity-rift") ApplyGravityOverlap(game, frame.Source, [frame.Target]);
            if (!OfficialNavigationObjectives.IgnoresObstacleDamage(game, frame.Source, frame.Target) && !OfficialDefenseObjectives.IgnoresObstacleDamage(game, frame.Source, frame.Target))
            {
                int amount = MapPlacementCatalog.IsAsteroid(kind) ? 1 : MapPlacementCatalog.IsDebris(kind) ? 2 : kind == "purrgil" ? (int)(frame.Value >> 2) : kind == "exogorth" ? 1 : 0;
                if (amount > 0)
                {
                    string rule = MapPlacementCatalog.IsAsteroid(kind) ? "asteroid-overlap" : MapPlacementCatalog.IsDebris(kind) ? "debris-overlap" : "obstacle." + kind;
                    CombatModule.QueueDamage(game, frame.Source, amount, MapPlacementCatalog.IsAsteroid(kind), MapPlacementCatalog.IsAsteroid(kind), GamePhase.AbilityChoice, rule: rule); return;
                }
            }
            return;
        }
        if (frame.Stage == 2 && !ignored)
        { OfficialObjectives.QueueChoices(game, OfficialDefenseObjectives.ObstacleChoices(game, frame.Source, frame.Target, false), GamePhase.AbilityChoice); return; }
        if (frame.Stage == 3 && !ignored)
        { OfficialNavigationObjectives.OnObstacleOverlap(game, frame.Source, frame.Target, false, GamePhase.AbilityChoice); return; }
        if (frame.Stage == 4 && !ignored && OfficialObjectives.StationCanRepair(game, frame.Source, frame.Target) && game.State.Ships[frame.Source].DamageCards.Count > 0)
        { EngineEffects.PushChoice(game, new(Prefix + "station-repair", frame.Source, frame.Target), []); return; }
        if (frame.Stage < 5) return;
        Pop(game, next);
    }
    private static void ResolveSquadronMove(GameContext game, GameContinuation frame)
    {
        if (!OfficialObjectives.Live(game, frame.Source)) { Pop(game, frame); game.Flow.Phase = (GamePhase)frame.Value; return; }
        var obstacles = frame.Indices ?? [];
        if (obstacles.Length > 0) throw new InvalidOperationException("Choose an obstacle before resolving squadron overlap.");
        using (var scope = new GameScope(game)) SquadronAbilityEngine.ResolveMaulerMithelAfterMove(game.SquadronViews[game.SquadronIndex(frame.Source)], game.SceneSquadronViews);
        Pop(game, frame); game.Flow.Phase = (GamePhase)frame.Value;
    }
    private static void ResolveSquadronObstacle(GameContext game, GameContinuation frame)
    {
        if (!OfficialObjectives.Live(game, frame.Source)) { Pop(game, frame); return; }
        var next = frame with { Stage = frame.Stage + 1 }; Replace(game, frame, next);
        if (frame.Stage == 0)
        {
            string id = OfficialObjectives.ObstacleId(game, frame.Target);
            if (id == "purrgil") { CombatModule.QueueDamage(game, frame.Source, 1, false, true, GamePhase.AbilityChoice, rule: "obstacle.purrgil"); return; }
            if (OfficialObjectives.StationCanRepair(game, frame.Source, frame.Target) && game.State.Squadrons[game.SquadronIndex(frame.Source)].CurrentHull < game.Topology.Squadrons[game.SquadronIndex(frame.Source)].CatalogEntry.Definition.Hull)
                EngineEffects.PushChoice(game, new(Prefix + "station-squadron-repair", frame.Source, frame.Target), []);
            return;
        }
        if (frame.Stage == 1) { OfficialObjectives.QueueChoices(game, OfficialDefenseObjectives.ObstacleChoices(game, frame.Source, frame.Target, false), GamePhase.AbilityChoice); return; }
        if (frame.Stage == 2) { OfficialNavigationObjectives.OnObstacleOverlap(game, frame.Source, frame.Target, false, GamePhase.AbilityChoice); return; }
        Pop(game, next);
    }
    private static GameDecision Forced(string label) => new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), label)]);
    private static void Replace(GameContext game, GameContinuation old, GameContinuation next)
    { int index = game.Flow.Continuations.IndexOf(old); if (index < 0) throw new InvalidOperationException("Creature continuation absent."); game.Flow.Continuations[index] = next; }
    private static void Pop(GameContext game, GameContinuation frame) { game.Flow.Continuations.Remove(frame); game.Flow.Phase = GamePhase.AbilityChoice; }
}
