using System.Numerics;
using System.Collections.Concurrent;
using Armada.Core;
using Armada.Rules;
using Armada.Session;
using Swarmada.Headless.Effects;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Runtime;
using ManeuverGeometry = Swarmada.Headless.Geometry.ManeuverGeometry;
using PlayAreaGeometry = Swarmada.Headless.Geometry.PlayAreaGeometry;

namespace Swarmada.Headless.Engine;

/// <summary>Complete analytic chart constraints, independent of any policy discretization.
/// Independent yaw grants compose, but no joint may exceed two clicks.</summary>
public sealed record ManeuverCourseConstraints(int DialedSpeed, IReadOnlyList<int> JointAllowances,
    bool OneExtraYaw, bool HugeShip, int ExtraYawBudget = 0);

/// <summary>Synchronous port of MovementPreviewView's articulated course/overlap rules and
/// BattleTurnController's movement resolution. No UI, frame clock, course grid or learner policy is used.</summary>
public sealed class MovementModule : IGameplayModule
{
    private readonly bool compactDecisions;
    public MovementModule(bool compactDecisions = false) => this.compactDecisions = compactDecisions;
    // Chart-only data is immutable and shared by all search branches. The key space is bounded
    // by four joints, three allowance values, and the independently available yaw grants.
    private static readonly ConcurrentDictionary<(int Length, int Allowances, int Budget), int[][]> CourseCache = new();
    private static readonly ConcurrentDictionary<(int Length, int Allowances, int Budget), IReadOnlyList<ReadOnlyMemory<int>>> OrderedCourseCache = new();
    public bool Handles(GamePhase phase) => phase is GamePhase.SelectCourse or GamePhase.SquadronDisplacement;
    public GameDecision GetDecision(GameContext game)
    {
        if (game.Flow.Phase == GamePhase.SquadronDisplacement) return DisplacementDecision(game);
        int ship = game.Flow.ActiveShip;
        if (!game.IsShip(ship) || game.IsDestroyed(ship)) return Forced(game, "Complete destroyed ship activation");
        if (game.Flow.Movement is null && !game.Flow.Flags.ContainsKey($"movement.attack-step-complete.{game.State.Round}.{ship}"))
            return Forced(game, "Resolve end of attack step");
        if (game.Flow.Movement is null && !game.Flow.Flags.ContainsKey($"movement.before-determine.{game.State.Round}.{ship}"))
            return Forced(game, "Resolve effects before Determine Course");
        if (game.Flow.Movement is null && CommandModule.BeginHeldNavigateForManeuver(game)) return Forced(game, "Resolve held Navigate command");
        if (game.Flow.Movement is null && !game.Flow.Flags.ContainsKey($"movement.determine-course.{game.State.Round}.{ship}"))
            return PendingSpeedDecision(game, ship) ?? Forced(game, "Resolve Determine Course abilities");
        if (game.Flow.Movement is { } executing)
        {
            if (executing.Stage == 2 && executing.ObjectiveChoiceT >= 0 && !executing.ObjectiveChoiceResolved)
                return Player(game, [new(new(GameActionKind.AcceptObjective, ship), "Stop and begin objective"),
                    new(new(GameActionKind.DeclineObjective, ship), "Continue plotted maneuver")]);
            if (executing.Stage == 3 && NeedsChartOfficerChoice(game, executing))
            {
                var choices = new List<GameActionOption>();
                if (game.HasUpgrade(ship, "jainas-light-title")) choices.Add(new(new(GameActionKind.UseAbility, ship, AbilityId: "jainas-light"), "Ignore all obstacle effects with Jaina's Light"));
                if (game.HasUpgrade(ship, "chart-officer")) choices.Add(new(new(GameActionKind.UseAbility, ship, AbilityId: "chart-officer"), "Discard Chart Officer to ignore all obstacle effects"));
                choices.Add(new(new(GameActionKind.Pass, ship), "Resolve obstacle effects normally"));
                return Player(game, choices);
            }
            return Forced(game, "Resolve maneuver");
        }
        var options = new List<GameActionOption>();
        if ((game.Flow.ShipActivations[ship].ResolvedCommands & (1 << (int)ShipCommand.Navigate)) == 0
            && (game.State.Commands[ship].HasToken(ShipCommand.Navigate) || HasBeck(game, ship)))
            options.AddRange(CommandModule.ActivationCommandOptions(game, GamePhase.SelectCourse));
        if (compactDecisions && game.State.Ships[ship].CurrentSpeed > 0)
            options.Add(new(new(GameActionKind.SetCourse, ship, Speed: game.State.Ships[ship].CurrentSpeed, AbilityId: "movement.course"),
                "Choose articulated course and legal tool side; see GetCourseConstraints", AcceptsCourse: true));
        else foreach (var movement in EnumerateMoves(game, ship))
                options.Add(new(new(GameActionKind.SetCourse, ship, Index: (int)movement.GuideSide, YawDegrees: movement.YawDelta,
                    Speed: movement.DialedSpeed, Course: movement.PlottedCourse,
                    AbilityId: movement.StandstillYaw ? "movement.standstill-yaw" : "movement.course"),
                    movement.EndsOutsidePlayArea ? "Execute maneuver and leave the play area" : movement.StandstillYaw ? $"Resolve speed-zero yaw {movement.YawDelta} degrees"
                        : movement.CollidedShip >= 0 ? $"Execute reduced speed {movement.ExecutedCourse.Length} after overlap" : "Execute plotted maneuver"));
        if (options.Count == 0) throw new InvalidOperationException("No legal source maneuver exists for this ship.");
        return Player(game, options);
    }
    public void Apply(GameContext game, GameAction action)
    {
        if (game.Flow.Phase == GamePhase.SquadronDisplacement) { ApplyDisplacement(game, action); return; }
        int ship = game.Flow.ActiveShip;
        if (!game.IsShip(ship) || (game.IsDestroyed(ship) && game.Flow.Movement is null)) { Complete(game); return; }
        if (game.Flow.Movement is null)
        {
            if (CommandModule.ResolveActivationCommand(game, action, GamePhase.SelectCourse))
            {
                if (action.Command == ShipCommand.Navigate)
                {
                    game.Flow.Flags.Remove($"movement.determine-course.{game.State.Round}.{ship}");
                    game.Flow.Flags.Remove(SpeedChoiceKey(game, ship, "movement.navigate-speed"));
                }
                return;
            }
            var attackStepKey = $"movement.attack-step-complete.{game.State.Round}.{ship}";
            if (!game.Flow.Flags.ContainsKey(attackStepKey))
            {
                game.Flow.Flags[attackStepKey] = 1;
                EngineEffects.BeginWindow(game, "end-attack-step", ship, resumePhase: GamePhase.SelectCourse); return;
            }
            var beforeDetermineKey = $"movement.before-determine.{game.State.Round}.{ship}";
            if (!game.Flow.Flags.ContainsKey(beforeDetermineKey))
            {
                if (MovementAbilityEffects.BeginG8Choice(game, ship)) return;
                OfficialCreatureObjectives.ApplyGravityReduction(game, ship);
                game.Flow.Flags[beforeDetermineKey] = 1; return;
            }
            var determineKey = $"movement.determine-course.{game.State.Round}.{ship}";
            if (!game.Flow.Flags.ContainsKey(determineKey))
            {
                if (PendingSpeedDecision(game, ship) is { } speedDecision)
                {
                    if (!speedDecision.Actions.Any(o => GameSession.Matches(o, action))) throw new ArgumentException("Choose a speed change or keep speed.", nameof(action));
                    if (action.AbilityId is "movement.navigate-token.yaw" or "movement.navigate-token.speed")
                        game.Flow.ShipActivations[ship].NavigateTokenUse = action.AbilityId.EndsWith(".yaw", StringComparison.Ordinal) ? 2 : 1;
                    else if (action.Kind == GameActionKind.ChangeSpeed) ChangeSpeed(game, action);
                    else game.Flow.Flags[SpeedChoiceKey(game, ship, action.AbilityId[..^5])] = 1;
                    return;
                }
                game.Flow.Flags[determineKey] = 1;
                EngineEffects.BeginWindow(game, "determine-course", ship, resumePhase: GamePhase.SelectCourse);
                return;
            }
            if (action.Kind == GameActionKind.ChangeSpeed) { ChangeSpeed(game, action); return; }
            if (action.Kind != GameActionKind.SetCourse) throw new ArgumentException("A plotted course is required.", nameof(action));
            MovementExecution? move;
            if (action.Speed != game.State.Ships[ship].CurrentSpeed || action.Course is null) throw new ArgumentException("Illegal ship maneuver.", nameof(action));
            if (game.State.Ships[ship].CurrentSpeed == 0)
            {
                float clicks = action.YawDegrees / TabletopMeasurements.ManeuverClickDegrees;
                if (action.AbilityId != "movement.standstill-yaw" || action.Course.Length != 0 || action.Index != (int)GuideSide.Center
                    || clicks != MathF.Truncate(clicks) || Math.Abs(clicks) > StandstillYawAllowance(game, ship)) throw new ArgumentException("Illegal standstill yaw.", nameof(action));
                move = EvaluateStandstill(game, ship, (int)clicks);
            }
            else if (action.AbilityId != "movement.course" || action.YawDegrees != 0
                || !TryEvaluateLegalCourse(game, ship, action.Course, (GuideSide)action.Index, out move)) throw new ArgumentException("Illegal ship maneuver.", nameof(action));
            game.Flow.Movement = move ?? throw new ArgumentException("Illegal ship maneuver.", nameof(action));
        }
        var execution = game.Flow.Movement;
        if (execution.Stage == 0)
        {
            execution.Stage = 1;
            if (EngineEffects.BeginWindow(game, "before-maneuver", ship, resumePhase: GamePhase.SelectCourse)) return;
        }
        if (execution.Stage == 1) { FindObjectiveOffer(game, execution); execution.Stage = 2; }
        if (execution.Stage == 2)
        {
            if (execution.ObjectiveChoiceT >= 0 && !execution.ObjectiveChoiceResolved)
            {
                if (action.Kind is not (GameActionKind.AcceptObjective or GameActionKind.DeclineObjective)) return;
                execution.ObjectiveChoiceResolved = true;
                execution.ObjectiveAccepted = action.Kind == GameActionKind.AcceptObjective;
                if (!execution.ObjectiveAccepted)
                {
                    if (game.State.Objective is IntelRetrievalObjective intel) intel.ContinuePastIntelZone(ship, execution.ObjectiveToken);
                    else if (game.State.Objective is SeizeStationObjective station) station.ContinuePastCaptureZone(ship);
                }
            }
            CommitMovement(game, execution);
            execution.Stage = 3;
        }
        if (execution.Stage == 3)
        {
            if (NeedsChartOfficerChoice(game, execution))
            {
                if (action.Kind == GameActionKind.UseAbility && action.AbilityId == "chart-officer")
                {
                    var card = game.UpgradesFor(ship).First(c => c.Active && c.Definition.id == "chart-officer");
                    card.Discard(); execution.IgnoreObstacleEffects = true;
                }
                else if (action.Kind == GameActionKind.UseAbility && action.AbilityId == "jainas-light" && game.HasUpgrade(ship, "jainas-light-title"))
                { execution.IgnoreObstacleEffects = true; execution.IgnoredByJaina = true; }
                else if (action.Kind != GameActionKind.Pass) return;
                execution.ChartOfficerChoiceResolved = true;
            }
            if (execution.DamageEffects.Count == 0 && execution.DamageEffectIndex == 0) BuildDamageEffects(game, execution);
            if (execution.DamageEffects.Count > 0) game.Flow.Flags["movement.damage-batch"] = 1;
            execution.Stage = 4;
        }
        if (execution.Stage == 4)
        {
            while (execution.DamageEffectIndex < execution.DamageEffects.Count)
            {
                var damage = execution.DamageEffects[execution.DamageEffectIndex++];
                if (game.IsDestroyed(damage.Entity)) continue;
                CombatModule.QueueDamage(game, damage.Entity, damage.Amount, damage.Faceup, damage.IgnoreShields,
                    GamePhase.SelectCourse, damage.Zone, damage.Rule);
                return;
            }
            game.Flow.Flags.Remove("movement.damage-batch");
            RoundModule.ObserveObjectives(game); RoundModule.CheckTerminal(game);
            if (game.State.Outcome is not null) return;
            if (game.IsDestroyed(ship)) { Complete(game); return; }
            if (!execution.ObstacleRewardsResolved)
            {
                execution.ObstacleRewardsResolved = true;
                if (OfficialCreatureObjectives.BeginShipObstacleConsequences(game, execution, GamePhase.SelectCourse)) return;
            }
            if (!execution.MinesGathered)
            {
                execution.MinesGathered = true;
                execution.MineIndices = Enumerable.Range(0, game.State.SpatialObjects.Count)
                    .Where(i => game.State.SpatialObjects[i].Active && game.State.SpatialObjects[i].Kind == SetupModule.ProximityMineKind
                        && Vector2.Distance(game.State.SpatialObjects[i].Pose.Position, game.ShipBase(ship).ClosestPoint(game.State.SpatialObjects[i].Pose.Position))
                            - TabletopMeasurements.FromWorldUnits(.26f) <= TabletopMeasurements.DistanceOne)
                    .OrderBy(i => game.State.SpatialObjects[i].Id, StringComparer.Ordinal).ThenBy(i => i).ToArray();
            }
            while (execution.MineIndex < execution.MineIndices.Length)
            {
                int mine = execution.MineIndices[execution.MineIndex++];
                if (!game.State.SpatialObjects[mine].Active) continue;
                CombatModule.QueueMineExplosion(game, ship, mine, GamePhase.SelectCourse); return;
            }
            if (game.State.Objective is IntelRetrievalObjective delivery) delivery.ResolveAfterManeuver(ObjectiveInput(game, ship), game.State.Round);
            execution.Stage = 5;
            if (game.Flow.PendingDisplacements.Length > 0) { game.Flow.Phase = GamePhase.SquadronDisplacement; return; }
        }
        if (execution.Stage == 5)
        {
            if (!execution.AfterManeuverDamageResolved)
            {
                execution.AfterManeuverDamageResolved = true;
                if (game.State.Ships[ship].DamageAfterManeuver > 0)
                { CombatModule.QueueDamage(game, ship, 1, false, false, GamePhase.SelectCourse, rule: "ruptured-engine"); return; }
            }
            if (game.IsDestroyed(ship)) { Complete(game); return; }
            execution.Stage = 6;
            if (EngineEffects.BeginWindow(game, "after-maneuver", ship, resumePhase: GamePhase.SelectCourse)) return;
        }
        if (execution.Stage == 6)
        {
            execution.Stage = 7;
            if (EngineEffects.BeginWindow(game, "after-maneuver-final", ship, resumePhase: GamePhase.SelectCourse)) return;
        }
        Complete(game);
    }
    private static void AddSpeedChoices(GameContext game, int ship, int change, string id, List<GameActionOption> options)
    {
        var max = game.Topology.Ships[ship].CatalogEntry.Definition.MaximumSpeed;
        int physical = PhysicalDialSpeed(game, ship);
        for (var speed = Math.Max(0, physical - change); speed <= Math.Min(max, physical + change); speed++)
            if (speed != physical) options.Add(new(new(GameActionKind.ChangeSpeed, ship, Speed: speed, AbilityId: id), $"Change physical speed dial to {speed}"));
    }
    private static GameDecision? PendingSpeedDecision(GameContext game, int ship)
    {
        var activation = game.Flow.ShipActivations[ship];
        if (activation.NavigateToken && !activation.NavigateManeuverConsumed && activation.NavigateTokenUse == 0 && game.HasUpgrade(ship, "nav-team"))
            return Player(game, [new(new(GameActionKind.UseAbility, ship, AbilityId: "movement.navigate-token.yaw"), "Use Navigate token to increase yaw with Nav Team"),
                new(new(GameActionKind.UseAbility, ship, AbilityId: "movement.navigate-token.speed"), "Use Navigate token to change speed")]);
        int change = 1; string? id = null;
        bool tokenSpeed = activation.NavigateToken && activation.NavigateTokenUse != 2;
        if (!activation.NavigateManeuverConsumed && (activation.NavigateDial || tokenSpeed) && !game.Flow.Flags.ContainsKey(SpeedChoiceKey(game, ship, "movement.navigate-speed")))
        { id = "movement.navigate-speed"; change = activation.NavigateDial && tokenSpeed ? 2 : 1; }
        else
        {
            using var scope = new GameScope(game);
            if (!game.Flow.Flags.ContainsKey(SpeedChoiceKey(game, ship, "movement.ozzel-speed")) && CommanderRules.GrantsDetermineCourseSpeedChange(game.ShipViews[ship])) id = "movement.ozzel-speed";
            else
            {
                var fleet = FleetMask(game, ship);
                if (!game.Flow.Flags.ContainsKey(SpeedChoiceKey(game, ship, "movement.entrapment-speed")) && FleetCommandEffects.AllowsDetermineCourseSpeedChange(fleet)) id = "movement.entrapment-speed";
            }
        }
        if (id is null) return null;
        var options = new List<GameActionOption>(); AddSpeedChoices(game, ship, change, id, options);
        options.Add(new(new(GameActionKind.Pass, ship, AbilityId: id + ".skip"), "Keep current speed"));
        return Player(game, options);
    }
    private static string SpeedChoiceKey(GameContext game, int ship, string id) => $"{(id == "movement.navigate-speed" ? "movement.nav-speed" : id)}.{game.State.Round}.{ship}";
    private void ChangeSpeed(GameContext game, GameAction action)
    {
        if (!GetDecision(game).Actions.Any(o => o.Template.Kind == GameActionKind.ChangeSpeed
            && o.Template.Speed == action.Speed && o.Template.AbilityId == action.AbilityId)) throw new ArgumentException("Illegal speed change.", nameof(action));
        int ship = game.Flow.ActiveShip; var state = game.State.Ships[ship]; int previous = PhysicalDialSpeed(game, ship);
        SetPhysicalDialSpeed(game, ship, action.Speed);
        game.Flow.Flags[SpeedChoiceKey(game, ship, action.AbilityId)] = 1;
        RoundModule.ObserveObjectives(game);
        if (previous != action.Speed && state.HasFaceupCritical(CriticalDamageCard.ThrusterFissure))
            CombatModule.QueueDamage(game, ship, 1, false, false, GamePhase.SelectCourse, rule: "thruster-fissure");
    }
    public static IReadOnlyList<MovementExecution> EnumerateMoves(GameContext game, int ship)
    {
        var result = new List<MovementExecution>(); var state = game.State.Ships[ship];
        var definition = game.Topology.Ships[ship].CatalogEntry.Definition;
        if (state.CurrentSpeed == 0)
        {
            int yaw = StandstillYawAllowance(game, ship);
            for (int click = -yaw; click <= yaw; click++) result.Add(EvaluateStandstill(game, ship, click));
            return result;
        }
        var (allowances, extraYaw) = EffectiveYaw(game, ship);
        var courses = EnumerateCourses(allowances, extraYaw);
        var holds = new HashSet<int>();
        foreach (var course in courses)
        {
            var guide = ManeuverGeometry.GuideSideForCourse(course);
            if (guide == GuideSide.Center && definition.HasAuxiliaryHullZones) guide = GuideSide.Port;
            int sides = course.All(click => click == 0) ? 1 : 2;
            for (int index = 0; index < sides; index++)
            {
                var side = sides == 1 ? guide : index == 0 ? GuideSide.Port : GuideSide.Starboard;
                var move = EvaluateCourse(game, ship, course, side);
                if (move is null) continue;
                if (move.ExecutedCourse.Length == 0 && !holds.Add(move.PlayAreaHold ? -1 : move.CollidedShip)) continue;
                result.Add(move);
            }
        }
        return result;
    }
    public static IReadOnlyList<GameActionOption> GetForcedManeuverActions(GameContext game, int ship, bool allowSpeedOneYaw, bool ignoreGravityRift = false)
    {
        if (!game.IsShip(ship) || game.Topology.Ships[ship].IsStation || game.IsDestroyed(ship)) return [];
        if (!ignoreGravityRift && OfficialCreatureObjectives.IsNearGravityRift(game, ship))
            return [new(new(GameActionKind.SetCourse, ship, Index: (int)GuideSide.Center, Speed: 0, Course: [], AbilityId: "movement.forced"), "Resolve granted maneuver at speed zero due to gravity rift")];
        int yaw = allowSpeedOneYaw ? Math.Max(0, game.Topology.Ships[ship].CatalogEntry.Definition.GetManeuverRow(1).JointYaw[0] -
            (PhysicalDialSpeed(game, ship) == 1 ? game.State.Ships[ship].FinalYawPenalty : 0)) : 0;
        if (allowSpeedOneYaw && game.HasUpgrade(ship, "harrow-title")) yaw = 2;
        if (allowSpeedOneYaw && PhysicalDialSpeed(game, ship) == 1 && FleetCommandEffects.IncreasesFinalYaw(FleetMask(game, ship))) yaw = Math.Min(2, yaw + 1);
        var options = new List<GameActionOption>();
        for (int click = -yaw; click <= yaw; click++)
        {
            int[] course = [click];
            var sides = click == 0 ? new[] { game.Topology.Ships[ship].CatalogEntry.Definition.HasAuxiliaryHullZones ? GuideSide.Port : GuideSide.Center } : new[] { GuideSide.Port, GuideSide.Starboard };
            foreach (var side in sides)
                if (EvaluateCourse(game, ship, course, side) is not null)
                    options.Add(new(new(GameActionKind.SetCourse, ship, Index: (int)side, Speed: 1, Course: course, AbilityId: "movement.forced"), "Execute granted speed-one maneuver"));
        }
        return options;
    }
    public static void BeginForcedManeuver(GameContext game, int ship, GameAction courseAction, bool allowSpeedOneYaw, GamePhase returnPhase, bool ignoreGravityRift = false)
    {
        if (!GetForcedManeuverActions(game, ship, allowSpeedOneYaw, ignoreGravityRift).Any(o => GameSession.Matches(o, courseAction))) throw new ArgumentException("Illegal forced maneuver.");
        int speed = courseAction.Speed;
        var move = speed == 0 ? EvaluateStandstill(game, ship, 0) : EvaluateCourse(game, ship, courseAction.Course!, (GuideSide)courseAction.Index)!; move.DialedSpeed = speed;
        game.Flow.ForcedManeuvers.Add(new() { Ship = ship, OriginalSpeed = game.State.Ships[ship].CurrentSpeed,
            ActiveShip = game.Flow.ActiveShip, ActiveSquadron = game.Flow.ActiveSquadron, ReturnPhase = returnPhase, ParentMovement = game.Flow.Movement?.DeepClone() });
        game.Flow.Continuations.Add(new("movement.forced-return", ship));
        game.State.Ships[ship].SetSpeed(speed, game.Topology.Ships[ship].CatalogEntry.Definition.MaximumSpeed);
        game.Flow.ActiveShip = ship; game.Flow.ActiveSquadron = -1; game.Flow.Movement = move; game.Flow.Phase = GamePhase.SelectCourse;
    }
    private static int StandstillYawAllowance(GameContext game, int ship)
    {
        var activation = game.Flow.ShipActivations[ship];
        if (!game.HasUpgrade(ship, "rakehell-title") || (activation.ResolvedCommands & (1 << (int)ShipCommand.Navigate)) == 0) return 0;
        int yaw = Math.Max(0, 1 - game.State.Ships[ship].FinalYawPenalty);
        if (game.Flow.Flags.ContainsKey($"movement.jerjerrod.{game.State.Round}.{ship}")) yaw = 2;
        yaw += ExtraYawBudget(game, ship);
        if (PhysicalDialSpeed(game, ship) == 0 && FleetCommandEffects.IncreasesFinalYaw(FleetMask(game, ship))) yaw++;
        return Math.Min(2, yaw);
    }
    private static Pose StandstillPose(GameContext game, int ship, Pose start, float yaw)
    {
        if (yaw == 0) return start;
        var size = game.Topology.Ships[ship].FootprintMillimetres;
        var guide = new Vector2(MathF.Sign(yaw) * size.X * 0.5f, size.Y * 0.5f);
        var endRotation = new Pose(Vector2.Zero, start.YawDegrees + yaw);
        return new(start.TransformPoint(guide) - endRotation.TransformDirection(guide), endRotation.YawDegrees);
    }
    private static MovementExecution EvaluateStandstill(GameContext game, int ship, int clicks)
    {
        var start = game.State.Poses[ship]; float yaw = clicks * TabletopMeasurements.ManeuverClickDegrees;
        var end = StandstillPose(game, ship, start, yaw); int validity = Fits(game, ship, end, out int blocked);
        if (validity == 2)
        {
            Span<bool> rammed = game.State.Ships.Length <= 128 ? stackalloc bool[game.State.Ships.Length] : new bool[game.State.Ships.Length];
            rammed.Clear();
            RecordOverlappedShips(game, ship, end, rammed);
            blocked = ClosestRammedShip(game, ship, start, rammed);
        }
        return new() { Ship = ship, Start = start, End = validity == 2 ? start : end, DialedSpeed = 0,
            GuideSide = GuideSide.Center, StandstillYaw = true, YawDelta = yaw,
            CollidedShip = validity == 2 ? blocked : -1, EndsOutsidePlayArea = validity == 1 };
    }
    public static ManeuverCourseConstraints GetCourseConstraints(GameContext game, int ship)
    {
        var (allowances, extraYaw) = EffectiveYaw(game, ship);
        return new(game.State.Ships[ship].CurrentSpeed, Array.AsReadOnly(allowances), extraYaw > 0,
            game.Topology.Ships[ship].CatalogEntry.Definition.HasAuxiliaryHullZones, extraYaw);
    }
    /// <summary>Chart-valid patterns only, in ascending base-five order with joint zero
    /// varying fastest. Geometry and tool-side legality still require native validation.
    /// Immutable chart/budget data is shared between games; copy a pattern before editing it.</summary>
    public static IReadOnlyList<ReadOnlyMemory<int>> GetChartValidCourses(ManeuverCourseConstraints constraints)
    {
        ArgumentNullException.ThrowIfNull(constraints);
        if (constraints.JointAllowances.Count > 6 || constraints.ExtraYawBudget < 0)
            throw new ArgumentOutOfRangeException(nameof(constraints));
        int encoded = 0, place = 1;
        foreach (int allowance in constraints.JointAllowances)
        {
            if (allowance is < 0 or > 2) throw new ArgumentOutOfRangeException(nameof(constraints));
            encoded += allowance * place; place *= 3;
        }
        return OrderedCourseCache.GetOrAdd((constraints.JointAllowances.Count, encoded, constraints.ExtraYawBudget), static key =>
        {
            var source = CourseCache.GetOrAdd(key, static value => BuildCourses(value));
            var ordered = new ReadOnlyMemory<int>[source.Length];
            for (int i = 0; i < source.Length; i++) ordered[i] = source[i];
            Array.Sort(ordered, static (first, second) =>
            {
                var a = first.Span; var b = second.Span;
                for (int i = a.Length - 1; i >= 0; i--)
                { int comparison = a[i].CompareTo(b[i]); if (comparison != 0) return comparison; }
                return 0;
            });
            return Array.AsReadOnly(ordered);
        });
    }
    public static bool TryEvaluateLegalCourse(GameContext game, int ship, int[] course, GuideSide side, out MovementExecution? movement)
    {
        movement = null;
        var (allowances, extraYaw) = EffectiveYaw(game, ship);
        if (course.Length != allowances.Length || course.Length == 0) return false;
        int extraSpent = 0;
        for (int i = 0; i < course.Length; i++)
        {
            if (course[i] is < -2 or > 2) return false;
            int magnitude = Math.Abs(course[i]);
            if (magnitude > 2) return false;
            extraSpent += Math.Max(0, magnitude - allowances[i]);
            if (extraSpent > extraYaw) return false;
        }
        var guide = ManeuverGeometry.GuideSideForCourse(course);
        if (guide == GuideSide.Center && game.Topology.Ships[ship].CatalogEntry.Definition.HasAuxiliaryHullZones) guide = GuideSide.Port;
        bool straight = course.All(click => click == 0);
        if (straight ? side != guide : side is not (GuideSide.Port or GuideSide.Starboard)) return false;
        movement = EvaluateCourse(game, ship, course, side); return movement is not null;
    }
    private static (int[] Allowances, int ExtraYaw) EffectiveYaw(GameContext game, int ship)
    {
        var state = game.State.Ships[ship]; var definition = game.Topology.Ships[ship].CatalogEntry.Definition;
        var fleet = FleetMask(game, ship);
        bool physicalChart = PhysicalDialSpeed(game, ship) == state.CurrentSpeed;
        var allowances = MovementResolver.GetEffectiveAllowances(definition, state.CurrentSpeed, physicalChart ? state.FinalYawPenalty : 0, out _);
        if (allowances.Length > 0 && state.CurrentSpeed == 1 && game.HasUpgrade(ship, "harrow-title")) allowances[0] = 2;
        if (allowances.Length > 0 && game.Flow.Flags.ContainsKey($"movement.jerjerrod.{game.State.Round}.{ship}")) allowances[0] = 2;
        if (allowances.Length > 0 && physicalChart && FleetCommandEffects.IncreasesFinalYaw(fleet)) allowances[^1] = Math.Min(2, allowances[^1] + 1);
        return (allowances, ExtraYawBudget(game, ship));
    }
    private static int ExtraYawBudget(GameContext game, int ship)
    {
        var activation = game.Flow.ShipActivations[ship]; int budget = 0;
        if (!activation.NavigateManeuverConsumed)
        {
            if (activation.NavigateDial) budget++;
            if (activation.NavigateToken && activation.NavigateTokenUse == 2 && game.HasUpgrade(ship, "nav-team")) budget++;
        }
        if (game.Flow.Flags.ContainsKey($"movement.extra-yaw.{game.State.Round}.{ship}")) budget++;
        using var scope = new GameScope(game);
        if (activation.ManeuversExecuted == 0 && CommanderRules.GrantsExtraManeuverYaw(game.ShipViews[ship])) budget++;
        return budget;
    }
    /// <summary>The physical dial survives temporary G-8/rift reductions and card-granted maneuvers.</summary>
    public static int PhysicalDialSpeed(GameContext game, int ship)
    {
        for (int i = game.Flow.ForcedManeuvers.Count - 1; i >= 0; i--)
            if (game.Flow.ForcedManeuvers[i].Ship == ship) return game.Flow.ForcedManeuvers[i].OriginalSpeed;
        for (int i = game.Flow.ExtraManeuvers.Count - 1; i >= 0; i--)
            if (game.Flow.ExtraManeuvers[i].Ship == ship) return game.Flow.ExtraManeuvers[i].OriginalSpeed;
        if (game.Flow.Flags.TryGetValue(game.Topology.ProjectorRestoreKeys[ship], out long projector)) return (int)projector - 1;
        if (game.Flow.Flags.TryGetValue(game.Topology.GravityRestoreKeys[ship], out long gravity)) return (int)gravity - 1;
        return game.State.Ships[ship].CurrentSpeed;
    }
    public static void SetPhysicalDialSpeed(GameContext game, int ship, int speed)
    {
        int maximum = game.Topology.Ships[ship].CatalogEntry.Definition.MaximumSpeed;
        if (speed < 0 || speed > maximum) throw new ArgumentOutOfRangeException(nameof(speed));
        var forced = game.Flow.ForcedManeuvers.LastOrDefault(m => m.Ship == ship);
        var extra = game.Flow.ExtraManeuvers.LastOrDefault(m => m.Ship == ship);
        if (forced is not null) forced.OriginalSpeed = speed;
        if (extra is not null) extra.OriginalSpeed = speed;
        int maneuver = forced is not null || extra is not null ? 1 : speed;
        bool projector = game.Flow.Flags.ContainsKey(game.Topology.ProjectorRestoreKeys[ship]), gravity = game.Flow.Flags.ContainsKey(game.Topology.GravityRestoreKeys[ship]);
        if (projector) game.Flow.Flags[game.Topology.ProjectorRestoreKeys[ship]] = maneuver + 1;
        if (gravity) game.Flow.Flags[game.Topology.GravityRestoreKeys[ship]] = Math.Max(0, maneuver - (projector ? 1 : 0)) + 1;
        game.State.Ships[ship].SetSpeed(Math.Max(0, maneuver - (projector ? 1 : 0) - (gravity ? 1 : 0)), maximum);
    }
    private static FleetCommandMask FleetMask(GameContext game, int ship) => (FleetCommandMask)game.State.Registers[game.Topology.Registers[
        game.State.Owners[ship] == 1 ? "round.fleetCommandMask.player1" : "round.fleetCommandMask.player2"]];
    private static bool HasBeck(GameContext game, int ship)
    {
        for (int i = 0; i < game.State.Upgrades.Length; i++)
            if (game.State.Upgrades[i].Active && game.State.Upgrades[i].Definition.id == "commander-beck-officer"
                && game.State.Registers[game.Topology.Registers.UpgradeOwner(i)] == ship + 1) return true;
        return false;
    }
    private static int[][] EnumerateCourses(int[] allowances, int budget)
    {
        int encoded = 0, place = 1;
        foreach (int allowance in allowances) { encoded += allowance * place; place *= 3; }
        return CourseCache.GetOrAdd((allowances.Length, encoded, budget), static key => BuildCourses(key));
    }
    private static int[][] BuildCourses((int Length, int Allowances, int Budget) key)
    {
        var allowances = new int[key.Length]; int encoded = key.Allowances;
        for (int i = 0; i < allowances.Length; i++) { allowances[i] = encoded % 3; encoded /= 3; }
        var result = new List<int[]>(); var course = new int[allowances.Length];
        void Add(int joint, int remaining)
        {
            if (joint == course.Length) { result.Add(course.ToArray()); return; }
            for (int click = -2; click <= 2; click++)
            {
                int cost = Math.Max(0, Math.Abs(click) - allowances[joint]);
                if (cost > remaining) continue;
                course[joint] = click; Add(joint + 1, remaining - cost);
            }
        }
        Add(0, key.Budget);
        return result.ToArray();
    }
    public static MovementExecution? EvaluateCourse(GameContext game, int ship, int[] course, GuideSide side)
    {
        var definition = game.Topology.Ships[ship].CatalogEntry.Definition; var start = game.State.Poses[ship];
        var size = game.Topology.Ships[ship].FootprintMillimetres;
        var collision = -1;
        Span<bool> rammed = game.State.Ships.Length <= 128 ? stackalloc bool[game.State.Ships.Length] : new bool[game.State.Ships.Length];
        rammed.Clear();
        for (var speed = course.Length; speed >= 0; speed--)
        {
            var executed = course.AsSpan(0, speed);
            // Reduced-speed overlap maneuvers may cross the tool, including huge ships.
            if (speed == course.Length && definition.HasAuxiliaryHullZones && ManeuverGeometry.HugeShipOverlapsTool(start, executed, size, side)) return null;
            var end = ManeuverGeometry.CalculateEndPose(start, executed, size, side, definition.HasAuxiliaryHullZones);
            if (speed == course.Length && !definition.HasAuxiliaryHullZones && FinalBaseOverlapsTool(start, end, executed, size, side)) return null;
            var validity = Fits(game, ship, end, out var blockedBy);
            if (validity == 2) { collision = blockedBy; RecordOverlappedShips(game, ship, end, rammed); continue; }
            if (collision >= 0) collision = ClosestRammedShip(game, ship, end, rammed);
            return new() { Ship = ship, Start = start, End = end, PlottedCourse = course.ToArray(), ExecutedCourse = executed.ToArray(),
                GuideSide = side, DialedSpeed = game.State.Ships[ship].CurrentSpeed, CollidedShip = collision, EndsOutsidePlayArea = validity == 1 };
        }
        return collision < 0 ? null : new MovementExecution { Ship = ship, Start = start, End = start,
            PlottedCourse = course.ToArray(), ExecutedCourse = [], GuideSide = side,
            DialedSpeed = game.State.Ships[ship].CurrentSpeed, CollidedShip = ClosestRammedShip(game, ship, start, rammed) };
    }
    private static void RecordOverlappedShips(GameContext game, int ship, Pose proposed, Span<bool> rammed)
    {
        var footprint = new OrientedRectangle(proposed, game.Topology.Ships[ship].FootprintMillimetres);
        for (int other = 0; other < game.State.Ships.Length; other++)
            if (other != ship && !game.Topology.Ships[other].IsStation && !game.IsDestroyed(other) && !game.IsReserved(other)
                && Geometry2D.Overlaps(footprint, game.ShipBase(other))) rammed[other] = true;
    }
    private static int ClosestRammedShip(GameContext game, int ship, Pose final, ReadOnlySpan<bool> rammed)
    {
        var footprint = new OrientedRectangle(final, game.Topology.Ships[ship].FootprintMillimetres);
        int closest = -1; float distance = float.PositiveInfinity;
        for (int other = 0; other < rammed.Length; other++)
        {
            if (!rammed[other]) continue;
            float current = Geometry2D.Distance(footprint, game.ShipBase(other));
            if (current < distance) { closest = other; distance = current; }
        }
        return closest;
    }
    private static bool FinalBaseOverlapsTool(Pose start, Pose end, ReadOnlySpan<int> course, Vector2 size, GuideSide side)
    {
        var physicalSide = side == GuideSide.Center ? GuideSide.Port : side;
        var guide = start.TransformPoint(new(physicalSide == GuideSide.Port ? -size.X / 2 : size.X / 2, size.Y / 2));
        float yaw = start.YawDegrees;
        var finalBase = new OrientedRectangle(end, size);
        foreach (int click in course)
        {
            var next = guide + new Pose(Vector2.Zero, yaw).Forward * TabletopMeasurements.ManeuverSegment;
            if (Geometry2D.SegmentCrossesInterior(new(guide, next), finalBase, .02285f)) return true;
            guide = next; yaw += click * TabletopMeasurements.ManeuverClickDegrees;
        }
        return false;
    }
    // 0 legal, 1 final base outside board, 2 ship collision (reduce speed before checking final containment).
    private static int Fits(GameContext game, int ship, Pose pose, out int blockingShip)
    {
        blockingShip = -1; var footprint = new OrientedRectangle(pose, game.Topology.Ships[ship].FootprintMillimetres);
        for (var other = 0; other < game.State.Ships.Length; other++)
            if (other != ship && !game.Topology.Ships[other].IsStation && !game.IsDestroyed(other) && !game.IsReserved(other) && Geometry2D.Overlaps(footprint, game.ShipBase(other)))
            { blockingShip = other; return 2; }
        return PlayAreaGeometry.Contains(Board(game), footprint) ? 0 : 1;
    }
    internal static OrientedRectangle Board(GameContext game) => new(new Pose(Vector2.Zero), game.Topology.BoardSizeMillimetres);
    internal static ObjectiveShip ObjectiveInput(GameContext game, int ship) => new(ship, game.State.Owners[ship], game.ShipBase(ship),
        game.State.Ships[ship].CurrentSpeed, game.IsDestroyed(ship), game.IsReserved(ship), game.Topology.Ships[ship].CatalogEntry.Metadata.Size);
    internal static ObjectiveShip[] ObjectiveInputs(GameContext game) => Enumerable.Range(0, game.State.Ships.Length).Select(s => ObjectiveInput(game, s)).ToArray();
    private static Pose PathPose(GameContext game, MovementExecution execution, float fraction)
        => execution.StandstillYaw ? execution.CollidedShip >= 0 ? execution.Start : StandstillPose(game, execution.Ship, execution.Start, execution.YawDelta * fraction)
            : ManeuverGeometry.SamplePose(execution.Start, execution.ExecutedCourse, game.Topology.Ships[execution.Ship].FootprintMillimetres,
                fraction, execution.GuideSide, game.Topology.Ships[execution.Ship].CatalogEntry.Definition.HasAuxiliaryHullZones);
    private static bool MeetsObjective(GameContext game, MovementExecution move, float t, float fraction, int? requiredToken, out int token)
    {
        token = -1;
        var ship = ObjectiveInput(game, move.Ship) with { Base = new(PathPose(game, move, t), game.Topology.Ships[move.Ship].FootprintMillimetres) };
        if (game.State.Objective is IntelRetrievalObjective intel && intel.GetDownloadTokenAtPose(ship, fraction, requiredToken) is { } sequence)
        { token = sequence; return true; }
        return game.State.Objective is SeizeStationObjective station && station.CanBeginCaptureAtPose(ship, fraction);
    }
    private static void FindObjectiveOffer(GameContext game, MovementExecution move)
    {
        if (move.CollidedShip >= 0 || move.PlayAreaHold || game.State.Objective is not (IntelRetrievalObjective or SeizeStationObjective)) return;
        for (var i = 1; i <= 192; i++)
        {
            if (!MeetsObjective(game, move, i / 192f, 0.5f, null, out var token)) continue;
            move.ObjectiveToken = token;
            float Refine(float low, float high, float required)
            {
                for (var iteration = 0; iteration < 12; iteration++)
                { var mid = (low + high) * 0.5f; if (MeetsObjective(game, move, mid, required, token >= 0 ? token : null, out _)) high = mid; else low = mid; }
                return high;
            }
            move.ObjectiveChoiceT = Refine((i - 1) / 192f, i / 192f, 0.5f);
            if (MeetsObjective(game, move, 1, 0.5f, token >= 0 ? token : null, out _)) { move.ObjectiveStopT = 1; return; }
            move.ObjectiveStopT = move.ObjectiveChoiceT;
            for (var j = i; j <= 192; j++)
                if (MeetsObjective(game, move, j / 192f, 1, token >= 0 ? token : null, out _))
                { move.ObjectiveStopT = Refine(MathF.Max(move.ObjectiveChoiceT, (j - 1) / 192f), j / 192f, 1); break; }
            return;
        }
    }
    private static void CommitMovement(GameContext game, MovementExecution move)
    {
        string maneuverKey = $"movement.maneuvers.{game.State.Round}.{move.Ship}";
        game.Flow.Flags[maneuverKey] = game.Flow.Flags.GetValueOrDefault(maneuverKey) + 1;
        var activation = game.Flow.ShipActivations[move.Ship]; activation.ManeuversExecuted++;
        if (activation.NavigateDial || activation.NavigateToken) activation.NavigateManeuverConsumed = true;
        var endT = move.ObjectiveAccepted ? move.ObjectiveStopT : 1;
        if (game.State.Objective is IntelRetrievalObjective or SeizeStationObjective)
            for (var i = 1; i <= 192; i++)
            {
                var t = MathF.Min(endT, i / 192f);
                game.State.Poses[move.Ship] = PathPose(game, move, t);
                var input = ObjectiveInput(game, move.Ship);
                if (game.State.Objective is SeizeStationObjective station) station.TrackDepartureAtPose(input);
                RoundModule.ObserveObjectives(game);
                if (t >= endT) break;
            }
        game.State.Poses[move.Ship] = move.ObjectiveAccepted ? PathPose(game, move, endT) : move.End;
        if (PlayAreaDestruction.DestroyIfOutside(game, move.Ship))
        {
            move.EndsOutsidePlayArea = true; move.OverlappedObstacles = []; game.Flow.PendingDisplacements = [];
            MovementAbilityEffects.RestoreTemporarySpeedAfterMove(game, move.Ship); RoundModule.ObserveObjectives(game); return;
        }
        OfficialAssaultObjectives.OnShipMoved(game, move.Ship);
        if (move.ObjectiveAccepted)
        {
            var input = ObjectiveInput(game, move.Ship); var ships = ObjectiveInputs(game);
            var choice = game.State.Objective is IntelRetrievalObjective intel ? intel.BeginDownload(input, move.ObjectiveToken, game.State.Round, ships)
                : ((SeizeStationObjective)game.State.Objective!).BeginCapture(input, game.State.Round, ships);
            if (!choice.Accepted) throw new InvalidOperationException("Committed objective stop became invalid.");
            game.State.Ships[move.Ship].SetSpeed(0, game.Topology.Ships[move.Ship].CatalogEntry.Definition.MaximumSpeed);
        }
        RoundModule.ObserveObjectives(game);
        MovementAbilityEffects.RestoreTemporarySpeedAfterMove(game, move.Ship);
        move.OverlappedObstacles = OfficialObjectives.Obstacles(game).Where(index => Geometry2D.Overlaps(game.ShipBase(move.Ship), ObstacleGeometry.Shape(game, index)))
            .OrderBy(index => game.State.SpatialObjects[index].Id, StringComparer.Ordinal).ThenBy(index => index).ToArray();
        ProcessDisplacements(game, move.Ship);
    }
    private static bool NeedsChartOfficerChoice(GameContext game, MovementExecution move) => !move.ChartOfficerChoiceResolved
        && (game.HasUpgrade(move.Ship, "jainas-light-title") || game.HasUpgrade(move.Ship, "chart-officer")) && move.OverlappedObstacles.Length > 0;
    private static void BuildDamageEffects(GameContext game, MovementExecution move)
    {
        int ship = move.Ship;
        if (move.CollidedShip >= 0 && !move.ObjectiveAccepted)
        {
            var target = move.CollidedShip;
            bool moverFlotilla = game.Topology.Ships[ship].CatalogEntry.Metadata.IsFlotilla == true, targetFlotilla = game.Topology.Ships[target].CatalogEntry.Metadata.IsFlotilla == true;
            if ((!targetFlotilla || moverFlotilla) && !OffensiveRetrofitEffects.PreventsCollisionDamage(game.HasUpgrade(ship, "hardened-bulkheads"), game.Topology.Ships[ship].CatalogEntry.Metadata.Size, game.Topology.Ships[target].CatalogEntry.Metadata.Size))
                move.DamageEffects.Add(new(ship, 1, game.State.Owners[ship] != game.State.Owners[target] && game.HasUpgrade(target, "eclipse-title"), Rule: "collision.mover"));
            if (game.State.Ships[ship].DealsExtraDamageWhenOverlapping) move.DamageEffects.Add(new(ship, 1, Rule: "damaged-controls"));
            if ((!moverFlotilla || targetFlotilla) && !OffensiveRetrofitEffects.PreventsCollisionDamage(game.HasUpgrade(target, "hardened-bulkheads"), game.Topology.Ships[target].CatalogEntry.Metadata.Size, game.Topology.Ships[ship].CatalogEntry.Metadata.Size))
                move.DamageEffects.Add(new(target, 1, game.State.Owners[ship] != game.State.Owners[target] && game.HasUpgrade(ship, "garels-honor-title"), Rule: "collision.rammed"));
            OfficerCombatEffects.DiscardMontferratAfterRam(game.UpgradesFor(ship).FirstOrDefault(c => c.Definition.id == "admiral-montferrat-officer"));
        }
    }
    public static void ProcessDisplacements(GameContext game, int ship)
    {
        var pending = new List<int>(); game.Flow.DisplacementShip = ship;
        foreach (var topology in game.Topology.Squadrons)
        {
            var squad = topology.EntityIndex;
            if (game.IsDestroyed(squad) || game.IsReserved(squad) || !Geometry2D.Overlaps(game.SquadronBase(squad), game.ShipBase(ship))) continue;
            pending.Add(squad);
        }
        game.Flow.PendingDisplacements = pending.ToArray();
    }
    public static Vector2 TouchingPlacement(OrientedRectangle ship, Circle squad, Vector2 desired)
        => SquadronDisplacementGeometry.TouchingPosition(ship, squad, desired);
    private static GameDecision DisplacementDecision(GameContext game)
    {
        if (game.Flow.PendingDisplacements.Length == 0) return Forced(game, "Resume ship activation");
        var owner = 3 - game.State.Owners[game.Flow.DisplacementShip];
        return new(DecisionKind.Player, owner, GamePhase.SquadronDisplacement,
            game.Flow.PendingDisplacements.Select(squad => new GameActionOption(new(GameActionKind.DisplaceSquadron, squad, Target: game.Flow.DisplacementShip),
                "Place overlapped squadron touching moving ship", AcceptsPosition: true)).ToArray());
    }
    private static void ApplyDisplacement(GameContext game, GameAction action)
    {
        if (game.Flow.PendingDisplacements.Length == 0) { FinishDisplacements(game); return; }
        int squad = action.Source, ship = game.Flow.DisplacementShip;
        var pending = game.Flow.PendingDisplacements;
        if (action.Kind != GameActionKind.DisplaceSquadron || !pending.Contains(squad) || action.Target != ship || action.Position is not { } position
            || !SquadronDisplacementGeometry.IsLegalPosition(game, squad, position, pending)) throw new ArgumentException("Illegal squadron displacement.", nameof(action));
        var movingBase = game.ShipBase(ship); var circle = new Circle(position, game.SquadronBase(squad).Radius);
        var touching = SquadronDisplacementGeometry.Touches(circle, movingBase);
        if (!touching && !SquadronDisplacementGeometry.CanPlaceTouching(game, squad, movingBase, pending)) foreach (var other in game.Topology.Squadrons)
        {
            if (pending.Contains(other.EntityIndex) || !OfficialObjectives.Live(game, other.EntityIndex)) continue;
            var anchor = game.SquadronBase(other.EntityIndex);
            if (SquadronDisplacementGeometry.Touches(anchor, movingBase) && Geometry2D.Contact(circle, anchor) == ContactKind.Touching) { touching = true; break; }
        }
        if (!touching) throw new ArgumentException("Displaced squadron must touch the moving ship or an eligible touching squadron.", nameof(action));
        game.State.Poses[squad] = new(position, game.State.Poses[squad].YawDegrees);
        game.Flow.PendingDisplacements = pending.Where(s => s != squad).ToArray(); RoundModule.ObserveObjectives(game);
        if (game.Flow.PendingDisplacements.Length == 0) FinishDisplacements(game);
    }
    private static void FinishDisplacements(GameContext game)
    {
        if (game.Flow.Flags.Remove("objective.displacement.return", out long phase)) game.Flow.Phase = (GamePhase)phase;
        else game.Flow.Phase = game.Flow.Flags.Remove("round.reserve-displacement") ? GamePhase.StartRound : GamePhase.SelectCourse;
    }
    private static void Complete(GameContext game)
    {
        if (game.Flow.ForcedManeuvers.Count > 0 && game.Flow.Continuations.LastOrDefault()?.Rule == "movement.forced-return")
        {
            var parent = game.Flow.ForcedManeuvers[^1]; game.Flow.ForcedManeuvers.RemoveAt(game.Flow.ForcedManeuvers.Count - 1);
            game.Flow.Continuations.RemoveAt(game.Flow.Continuations.Count - 1);
            game.State.Ships[parent.Ship].SetSpeed(parent.OriginalSpeed, game.Topology.Ships[parent.Ship].CatalogEntry.Definition.MaximumSpeed);
            game.Flow.ActiveShip = parent.ActiveShip; game.Flow.ActiveSquadron = parent.ActiveSquadron; game.Flow.Movement = parent.ParentMovement;
            game.Flow.Phase = parent.ReturnPhase; return;
        }
        if (MovementAbilityEffects.FinishExtraManeuver(game)) return;
        if (game.IsShip(game.Flow.ActiveShip))
        { game.Flow.ShipActivations[game.Flow.ActiveShip].ManeuverComplete = true; game.State.Activated[game.Flow.ActiveShip] = true; }
        game.Flow.Movement = null; game.Flow.DisplacementShip = -1;
        RoundModule.EndShipActivation(game);
    }
    private static GameDecision Forced(GameContext game, string label) => new(DecisionKind.Forced, 0, game.Flow.Phase, [new(new(GameActionKind.Pass), label)]);
    private static GameDecision Player(GameContext game, IReadOnlyList<GameActionOption> options) => new(DecisionKind.Player,
        game.State.Owners[game.Flow.ActiveShip], game.Flow.Phase, options);
}
