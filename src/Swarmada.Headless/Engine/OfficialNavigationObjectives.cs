using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Armada.Session;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Executable navigation objectives. Mandatory choices are continuations;
/// persistent counters and markers live in the objective snapshot, never static state.</summary>
public static class OfficialNavigationObjectives
{
    private const string Prefix = "official-nav.";
    private const float TokenRadius = 12.7f;
    private const float Epsilon = .0115f;
    public static bool Supports(string id) => id is "dangerous-territory" or "doomed-station" or "hyperspace-migration" or
        "infested-fields" or "intel-sweep" or "minefields" or "navigational-hazards" or "salvage-run" or "sensor-net" or
        "solar-corona" or "superior-positions" or "volatile-deposits";
    public static bool Handles(string rule) => rule.StartsWith(Prefix, StringComparison.Ordinal);

    public static IEnumerable<GameContinuation> SetupChoices(GameContext game, bool afterDeployment)
    {
        string id = OfficialObjectives.Id(game);
        if (!Supports(id)) yield break;
        if (afterDeployment)
        {
            if (id == "salvage-run") yield return new(Prefix + "salvage-tokens");
            if (id == "solar-corona") yield return new(Prefix + "corona-edge", Source: Anchor(game, Second(game)));
            yield break;
        }
        if (id is "dangerous-territory" or "infested-fields") yield return new(Prefix + "mark-obstacles");
        if (id is "intel-sweep" or "minefields" or "salvage-run" or "sensor-net" or "hyperspace-migration")
            yield return new(Prefix + "place-tokens", Source: Anchor(game, id == "sensor-net" ? First(game) : Second(game)), Text: id);
        if (id == "intel-sweep") yield return new(Prefix + "objective-ships", Source: Anchor(game, First(game)));
    }

    public static IEnumerable<GameContinuation> TimingChoices(GameContext game, string timing, int source, int target)
    {
        string id = OfficialObjectives.Id(game);
        if (!Supports(id)) yield break;
        if (timing == "command-reveal" && id == "sensor-net" && game.IsShip(source) && NearbyTokens(game, source).Any())
            yield return new(Prefix + "sensor-net", source);
        if (timing == "start-round" && id == "navigational-hazards") yield return new(Prefix + "clear-hazard-marks");
        if (timing == "end-round")
        {
            if (id == "doomed-station") yield return new(Prefix + "remove-rift-touching");
            if (MapMovementRules.Enabled(game) && id is ("doomed-station" or "navigational-hazards"))
                yield return new(Prefix + "move-obstacles", Source: Anchor(game, Second(game)), Target: -1, Text: id);
            else if (id == "doomed-station") yield return new(Prefix + "doomed-station-score");
            if (id == "hyperspace-migration") yield return new(Prefix + "migration-score");
            if (id == "volatile-deposits") yield return new(Prefix + "volatile-score");
        }
    }

    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        string id = OfficialObjectives.Id(game);
        if (timing != "command-reveal" || !game.IsShip(source) || !OfficialObjectives.Live(game, source) ||
            !(id == "salvage-run" || id == "intel-sweep" && OfficialObjectives.IsObjectiveShip(game, source))) yield break;
        foreach (int token in NearbyTokens(game, source))
            yield return new(new(GameActionKind.UseAbility, source, token, AbilityId: Prefix + "collect"), "Collect objective token " + token);
    }

    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (action.AbilityId != Prefix + "collect") return false;
        if (!GetActions(game, window.Text, window.Source, window.Target).Any(o => GameSession.Matches(o, action)))
            throw new ArgumentException("No eligible objective token can be collected.");
        OfficialObjectives.RemoveToken(game, action.Target);
        OfficialObjectives.AwardVictory(game, game.State.Owners[action.Source]);
        return true;
    }

    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = new List<GameActionOption>(); int owner = frame.Source >= 0 ? game.State.Owners[frame.Source] : First(game);
        void Add(GameAction action, string label, bool position = false, bool yaw = false) => options.Add(new(action, label, position, yaw));
        switch (frame.Rule)
        {
            case Prefix + "place-tokens":
                Add(new(GameActionKind.PlaceToken, frame.Source, AbilityId: frame.Rule), "Place " + frame.Text + " objective token", true); break;
            case Prefix + "objective-ships":
                foreach (int ship in Ships(game, owner)) Add(new(GameActionKind.ChooseEntity, frame.Source, ship, AbilityId: frame.Rule), "Choose objective ship");
                break;
            case Prefix + "corona-edge":
                Add(new(GameActionKind.ChooseEntity, frame.Source, Index: -1, AbilityId: frame.Rule), "Choose left three-foot edge as the corona");
                Add(new(GameActionKind.ChooseEntity, frame.Source, Index: 1, AbilityId: frame.Rule), "Choose right three-foot edge as the corona"); break;
            case Prefix + "sensor-net":
                if (frame.Stage == 0)
                    foreach (int token in NearbyTokens(game, frame.Source)) Add(new(GameActionKind.ChooseEntity, frame.Source, token, AbilityId: frame.Rule),
                        MapMovementRules.Enabled(game) ? "Score Sensor Net and choose token to move" : "Score Sensor Net at this token");
                else if (!MapMovementRules.Enabled(game)) Add(new(GameActionKind.Pass, frame.Source, AbilityId: frame.Rule), "Skip disabled Sensor Net token relocation");
                else
                {
                    owner = 3 - game.State.Owners[frame.Source];
                    Add(new(GameActionKind.PlaceToken, frame.Source, frame.Target, AbilityId: frame.Rule), "Move the chosen Sensor Net token", true);
                }
                break;
            case Prefix + "overlap-claim":
                foreach (int token in frame.Indices ?? [])
                    if (TokenAvailableOn(game, token, frame.Target)) Add(new(GameActionKind.UseAbility, frame.Source, token, AbilityId: frame.Rule), "Claim an obstacle's objective token");
                Add(new(GameActionKind.Pass, frame.Source, AbilityId: frame.Rule), "Decline the obstacle's objective token"); break;
            case Prefix + "move-obstacles":
                if (!MapMovementRules.Enabled(game)) Add(new(GameActionKind.Pass, frame.Source, AbilityId: frame.Rule), "Skip disabled obstacle movement");
                else if (frame.Target < 0)
                {
                    foreach (int obstacle in MovableObstacles(game, frame.Text))
                        Add(new(GameActionKind.ChooseEntity, frame.Source, obstacle, AbilityId: frame.Rule), "Choose obstacle to move");
                    if (options.Count == 0) Add(new(GameActionKind.Pass, frame.Source, AbilityId: frame.Rule), "Finish moving obstacles and score station");
                }
                else
                {
                    options.Add(new(new(GameActionKind.PlaceToken, frame.Source, frame.Target, AbilityId: frame.Rule, Flipped: game.State.SpatialObjects[frame.Target].Flipped),
                        "Move the selected obstacle", AcceptsPosition: true, AcceptsYaw: true, AcceptsFlip: true));
                    Add(new(GameActionKind.ChooseEntity, frame.Source, Target: -1, AbilityId: frame.Rule), "Choose a different obstacle");
                }
                break;
            case Prefix + "volatile":
                foreach (int entity in frame.Indices ?? [])
                {
                    if (entity >= 0 && OfficialObjectives.Live(game, entity)) Add(new(GameActionKind.ChooseEntity, frame.Source, entity, AbilityId: frame.Rule), "Resolve Volatile Deposits damage to unit " + entity);
                }
                if (options.Count == 0) Add(new(GameActionKind.Pass, frame.Source, AbilityId: frame.Rule), "Finish Volatile Deposits splash damage");
                break;
            case Prefix + "minefields":
                foreach (int token in frame.Indices ?? [])
                    if (game.State.SpatialObjects[token].Active) Add(new(GameActionKind.ChooseEntity, frame.Source, token, AbilityId: frame.Rule), "Resolve nearby Minefields token");
                if (options.Count == 0) Add(new(GameActionKind.Pass, frame.Source, AbilityId: frame.Rule), "Finish Minefields explosions");
                break;
            default:
                Add(new(GameActionKind.Pass, AbilityId: frame.Rule), "Resolve " + frame.Rule[Prefix.Length..]); break;
        }
        bool forced = options.Count == 1 && options[0].Template.Kind == GameActionKind.Pass;
        return new(forced ? DecisionKind.Forced : DecisionKind.Player, forced ? 0 : owner, GamePhase.AbilityChoice, options);
    }

    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!GetDecision(game, frame).Actions.Any(o => GameSession.Matches(o, action))) throw new ArgumentException("Illegal navigation objective choice.");
        var state = OfficialObjectives.State(game)!;
        switch (frame.Rule)
        {
            case Prefix + "place-tokens": PlaceSetupToken(game, frame, action); return;
            case Prefix + "objective-ships":
                OfficialObjectives.SetObjectiveShip(game, action.Target);
                if (frame.Stage == 0) { Replace(game, frame, frame with { Source = Anchor(game, 3 - game.State.Owners[frame.Source]), Stage = 1 }); return; }
                break;
            case Prefix + "corona-edge": state.Flags["navigation.corona.edge"] = action.Index; break;
            case Prefix + "sensor-net":
                if (frame.Stage == 0)
                {
                    OfficialObjectives.AwardVictory(game, game.State.Owners[frame.Source]);
                    if (!MapMovementRules.Enabled(game)) break;
                    Replace(game, frame, frame with { Target = action.Target, Stage = 1 }); return;
                }
                if (!MapMovementRules.Enabled(game)) break;
                if (action.Position is null) throw new ArgumentException("Sensor Net requires a destination.");
                int mover = 3 - game.State.Owners[frame.Source];
                var token = game.State.SpatialObjects[frame.Target];
                ValidateTokenMove(game, token, action.Position.Value, mover == Second(game) ? 4 : 2);
                TokenObstacleAttachments.MoveToken(game, frame.Target, new(action.Position.Value, token.Pose.YawDegrees)); break;
            case Prefix + "mark-obstacles":
                foreach (int index in OfficialObjectives.Obstacles(game))
                    OfficialObjectives.PlaceToken(game, game.State.SpatialObjects[index].Pose.Position, id: "navigation.obstacle." + index);
                break;
            case Prefix + "salvage-tokens":
                foreach (int ship in OfficialObjectives.LiveShips(game, Second(game))) TokenAssignments.TryAssign(game, ship, ShipCommand.Navigate);
                break;
            case Prefix + "overlap-claim":
                if (action.Kind != GameActionKind.Pass)
                { OfficialObjectives.RemoveToken(game, action.Target); OfficialObjectives.AwardVictory(game, game.State.Owners[frame.Source]); }
                break;
            case Prefix + "clear-hazard-marks":
                foreach (string key in state.Flags.Keys.Where(k => k.StartsWith("navigation.moved.", StringComparison.Ordinal)).ToArray()) state.Flags.Remove(key);
                foreach (int marker in OfficialObjectives.TokenIndices(game).ToArray()) OfficialObjectives.RemoveToken(game, marker);
                break;
            case Prefix + "remove-rift-touching": RemoveObstaclesTouchingRift(game); break;
            case Prefix + "doomed-station-score": FinishDoomedStationRound(game); break;
            case Prefix + "move-obstacles":
                if (action.Kind == GameActionKind.Pass)
                {
                    if (frame.Text == "doomed-station")
                        FinishDoomedStationRound(game);
                    break;
                }
                if (action.Kind == GameActionKind.ChooseEntity)
                { Replace(game, frame, frame with { Target = action.Target }); return; }
                if (action.Position is null) throw new ArgumentException("Obstacle movement requires a destination.");
                MoveObjectiveObstacle(game, frame, action);
                state.Flags["navigation.moved." + frame.Target] = game.State.Round;
                OfficialObjectives.PlaceToken(game, game.State.SpatialObjects[frame.Target].Pose.Position, id: "navigation.moved." + frame.Target);
                if (frame.Text == "doomed-station") RemoveObstaclesTouchingRift(game);
                Replace(game, frame, frame with { Source = Anchor(game, 3 - game.State.Owners[frame.Source]), Target = -1 }); return;
            case Prefix + "migration-score":
                foreach (int ship in OfficialObjectives.LiveShips(game))
                {
                    if (OfficialObjectives.TokenCount(game, ship) > 0)
                    { OfficialObjectives.SetTokens(game, ship, 0); OfficialObjectives.AwardVictory(game, game.State.Owners[ship]); }
                    else if (OfficialObjectives.Obstacles(game).Any(i => OfficialObjectives.ObstacleId(game, i) == "purrgil" && OfficialObjectives.DistanceToPiece(game, ship, i) <= TabletopMeasurements.DistanceOne + Epsilon))
                        OfficialObjectives.SetTokens(game, ship, 1);
                }
                break;
            case Prefix + "volatile-score":
                foreach (int asteroid in OfficialObjectives.Obstacles(game).Where(i => MapPlacementCatalog.IsAsteroid(OfficialObjectives.ObstacleId(game, i)))) AwardCommandMajority(game, asteroid);
                break;
            case Prefix + "volatile":
                if (action.Kind == GameActionKind.Pass) break;
                Replace(game, frame, frame with { Indices = (frame.Indices ?? []).Where(i => i != action.Target).ToArray() });
                CombatModule.QueueDamage(game, action.Target, (int)frame.Value, false, false, GamePhase.AbilityChoice, rule: "objective.volatile-deposits", source: frame.Source); return;
            case Prefix + "minefields":
                if (action.Kind == GameActionKind.Pass) break;
                OfficialObjectives.RemoveToken(game, action.Target);
                Replace(game, frame, frame with { Indices = (frame.Indices ?? []).Where(i => i != action.Target).ToArray() });
                CombatModule.QueueUnmodifiedDiceDamage(game, -1, frame.Source, 2, GamePhase.AbilityChoice, "mine.explosion.objective"); return;
            default: throw new NotSupportedException("Unknown navigation objective continuation " + frame.Rule);
        }
        Pop(game, frame);
    }

    public static int ScoreBonus(GameContext game, int player)
    {
        if (!OfficialObjectives.Is(game, "intel-sweep")) return 0;
        var state = OfficialObjectives.State(game)!;
        return state.VictoryTokens[player - 1] > state.VictoryTokens[2 - player] ? 75 : 0;
    }

    /// <summary>Queue the optional obstacle reward only when its overlap effect is actually resolved.</summary>
    public static bool OnObstacleOverlap(GameContext game, int entity, int obstacle, bool effectsIgnored, GamePhase resumePhase)
    {
        string id = OfficialObjectives.Id(game);
        if (effectsIgnored || id is not ("dangerous-territory" or "infested-fields") || id == "dangerous-territory" && !game.IsShip(entity)) return false;
        int[] tokens = OfficialObjectives.TokenIndices(game).Where(i => TokenAvailableOn(game, i, obstacle)).ToArray();
        if (tokens.Length == 0) return false;
        return OfficialObjectives.QueueChoices(game, [new(Prefix + "overlap-claim", entity, obstacle, Indices: tokens)], resumePhase);
    }

    public static bool IgnoresObstacleDamage(GameContext game, int ship, int obstacle) => OfficialObjectives.Is(game, "dangerous-territory") &&
        game.State.Owners[ship] == Second(game) && (MapPlacementCatalog.IsAsteroid(OfficialObjectives.ObstacleId(game, obstacle)) || MapPlacementCatalog.IsDebris(OfficialObjectives.ObstacleId(game, obstacle)));

    public static void ObserveObstacleDamage(GameContext game, int ship, int actualAppliedDamage, int cardsDealt)
    {
        if (OfficialObjectives.Is(game, "navigational-hazards") && (actualAppliedDamage > 0 || cardsDealt > 0))
            OfficialObjectives.AwardVictory(game, 3 - game.State.Owners[ship]);
    }

    public static bool BeginMinefields(GameContext game, int ship, GamePhase returnPhase)
    {
        if (!OfficialObjectives.Is(game, "minefields") || !game.IsShip(ship)) return false;
        int[] mines = NearbyTokens(game, ship).ToArray();
        return mines.Length > 0 && OfficialObjectives.QueueChoices(game, [new(Prefix + "minefields", ship, Indices: mines)], returnPhase);
    }

    public static IEnumerable<GameActionOption> SolarCoronaOptions(GameContext game, AttackState attack)
    {
        if (!OfficialObjectives.Is(game, "solar-corona") || !game.IsShip(attack.Attacker) || attack.EnvironmentalDamage ||
            attack.EffectFlags.ContainsKey("objective.solar.resolved") || !CoronaInArc(game, attack)) yield break;
        for (int index = 0; index < attack.Dice.Length; index++)
            if (attack.Dice[index].Face == DieFace.Accuracy)
                yield return new(new(GameActionKind.ModifyDice, attack.Attacker, attack.Defender, Index: index, AbilityId: Prefix + "solar-remove"), "Solar Corona: discard one accuracy die");
    }
    public static bool ApplySolarCorona(GameContext game, AttackState attack, GameAction action)
    {
        if (action.AbilityId != Prefix + "solar-remove") return false;
        if (!SolarCoronaOptions(game, attack).Any(o => GameSession.Matches(o, action))) throw new ArgumentException("Choose an accuracy die for Solar Corona.");
        attack.Dice = attack.Dice.Where((_, i) => i != action.Index).ToArray(); attack.EffectFlags["objective.solar.resolved"] = 1; return true;
    }
    public static IEnumerable<GameActionOption> CriticalOptions(GameContext game, AttackState attack)
    {
        if (!OfficialObjectives.Is(game, "volatile-deposits") || !game.IsShip(attack.Attacker) || !game.IsShip(attack.Defender) ||
            attack.EnvironmentalDamage || attack.IsSalvoAttack || attack.Scatter || attack.EffectFlags.ContainsKey("objective.volatile.used")) yield break;
        bool eligible = attack.Dice.Any(d => (d.Face is DieFace.Critical or DieFace.HitCritical) &&
            (d.Color == AttackDieColor.Blue || game.State.Owners[attack.Defender] == First(game)));
        if (!eligible) yield break;
        foreach (int obstacle in OfficialObjectives.Obstacles(game))
            if (OfficialObjectives.DistanceToPiece(game, attack.Defender, obstacle) <= TabletopMeasurements.DistanceOne + Epsilon)
                yield return new(new(GameActionKind.ChooseCritical, attack.Attacker, attack.Defender, Index: obstacle, AbilityId: Prefix + "volatile-critical"), "Volatile Deposits critical around obstacle " + obstacle);
    }
    public static bool ApplyCritical(GameContext game, AttackState attack, GameAction action)
    {
        if (action.AbilityId != Prefix + "volatile-critical") return false;
        if (!CriticalOptions(game, attack).Any(o => GameSession.Matches(o, action))) throw new ArgumentException("Illegal Volatile Deposits critical.");
        int damage = (attack.Dice.Count(d => d.Face is DieFace.Critical or DieFace.HitCritical) + 1) / 2;
        int[] targets = Enumerable.Range(0, game.Topology.EntityCount).Where(i => OfficialObjectives.Live(game, i) &&
            OfficialObjectives.DistanceToPiece(game, i, action.Index) <= TabletopMeasurements.DistanceOne + Epsilon).ToArray();
        attack.EffectFlags["objective.volatile.used"] = 1;
        OfficialObjectives.QueueChoices(game, [new(Prefix + "volatile", attack.Attacker, action.Index, Value: damage, Indices: targets)], GamePhase.ResolveCritical);
        return true;
    }
    public static void ObserveAttack(GameContext game, AttackState attack)
    {
        if (!OfficialObjectives.Is(game, "superior-positions") || attack.EnvironmentalDamage || attack.Attacker < 0 ||
            !game.IsShip(attack.Defender) || attack.ToZone != HullZone.Rear || attack.AppliedDamage < 1 || attack.EffectFlags.ContainsKey("objective.superior.scored")) return;
        attack.EffectFlags["objective.superior.scored"] = 1; OfficialObjectives.AwardVictory(game, game.State.Owners[attack.Attacker]);
    }

    private static void PlaceSetupToken(GameContext game, GameContinuation frame, GameAction action)
    {
        if (action.Position is null) throw new ArgumentException("Objective token position is required.");
        ValidateSetupToken(game, frame.Text, frame.Stage, action.Position.Value);
        int token = OfficialObjectives.PlaceToken(game, action.Position.Value, id: frame.Text + "." + frame.Stage);
        if (frame.Text == "hyperspace-migration") OfficialObjectives.State(game)!.Flags["navigation.migration.token." + frame.Stage] = token;
        int count = frame.Text switch { "intel-sweep" => 5, "minefields" => 6, "hyperspace-migration" => 2, _ => 4 };
        if (frame.Stage + 1 == count) { Pop(game, frame); return; }
        bool alternate = frame.Text is "intel-sweep" or "sensor-net";
        Replace(game, frame, frame with { Stage = frame.Stage + 1, Source = alternate ? Anchor(game, 3 - game.State.Owners[frame.Source]) : frame.Source });
    }
    private static void ValidateSetupToken(GameContext game, string id, int stage, Vector2 position)
    {
        Geometry2D.RequireFinite(position, nameof(position)); var half = game.Topology.BoardSizeMillimetres / 2;
        float limitX = id == "hyperspace-migration" ? half.X : MathF.Min(half.X, 609.6f);
        if (MathF.Abs(position.X) + TokenRadius > limitX + Epsilon || MathF.Abs(position.Y) + TokenRadius > half.Y + Epsilon)
            throw new ArgumentException("Objective token must be wholly inside its placement area.");
        var circle = new Circle(position, TokenRadius);
        if (id is "intel-sweep" or "sensor-net")
        {
            if (half.Y - MathF.Abs(position.Y) - TokenRadius <= TabletopMeasurements.DistanceBand(5) + Epsilon) throw new ArgumentException("Token must be beyond distance five of both players' edges.");
            if (OfficialObjectives.TokenIndices(game).Any(i => Geometry2D.Distance(circle, OfficialObjectives.PieceCircle(game, i)) <= TabletopMeasurements.DistanceBand(3) + Epsilon)) throw new ArgumentException("Tokens must be beyond distance three of each other.");
        }
        else if (id is "minefields" or "salvage-run")
        {
            var obstacles = OfficialObjectives.Obstacles(game).Where(i => id == "minefields" || OfficialObjectives.ObstacleId(game, i) == "station");
            if (!obstacles.Any(i => Geometry2D.Distance(circle, ObstacleGeometry.Shape(game, i)) <= TabletopMeasurements.DistanceOne + Epsilon)) throw new ArgumentException("Token must be at distance one of its obstacle.");
            if (OfficialObjectives.TokenIndices(game).Any(i => Geometry2D.Distance(circle, OfficialObjectives.PieceCircle(game, i)) <= TabletopMeasurements.DistanceOne + Epsilon)) throw new ArgumentException("Tokens must be beyond distance one of each other.");
        }
        else if (id == "hyperspace-migration")
        {
            int side;
            if (stage == 0) side = position.X < 0 ? -1 : 1;
            else
            {
                int firstToken = (int)OfficialObjectives.State(game)!.Flags["navigation.migration.token.0"];
                side = game.State.SpatialObjects[firstToken].Pose.Position.X < 0 ? 1 : -1;
            }
            if (half.X - position.X * side - TokenRadius > TabletopMeasurements.DistanceOne + Epsilon) throw new ArgumentException("Migration tokens must lie at opposite three-foot edges.");
        }
    }
    private static void ValidateTokenMove(GameContext game, SpatialObjectState original, Vector2 destination, int maximumBand)
    {
        Geometry2D.RequireFinite(destination, nameof(destination)); float radius = MathF.Max(original.Size.X, original.Size.Y) / 2;
        var half = game.Topology.BoardSizeMillimetres / 2;
        if (MathF.Abs(destination.X) + radius > half.X + Epsilon || MathF.Abs(destination.Y) + radius > half.Y + Epsilon) throw new ArgumentException("Moved token must remain in the play area.");
        if (Vector2.Distance(destination, original.Pose.Position) > TabletopMeasurements.DistanceBand(maximumBand) + Epsilon) throw new ArgumentException("Token movement exceeds its allowed distance.");
    }
    private static IEnumerable<int> NearbyTokens(GameContext game, int ship) => OfficialObjectives.TokenIndices(game)
        .Where(i => OfficialObjectives.DistanceToPiece(game, ship, i) <= TabletopMeasurements.DistanceOne + Epsilon);
    private static bool TokenAvailableOn(GameContext game, int token, int obstacle) => game.State.SpatialObjects[token].Active && game.State.SpatialObjects[obstacle].Active &&
        Geometry2D.Overlaps(OfficialObjectives.PieceCircle(game, token), ObstacleGeometry.Shape(game, obstacle));
    private static IEnumerable<int> MovableObstacles(GameContext game, string id) => OfficialObjectives.Obstacles(game).Where(i =>
        !OfficialObjectives.TokenIndices(game).Any(token => TokenAvailableOn(game, token, i)) &&
        (id == "doomed-station" ? OfficialObjectives.ObstacleId(game, i) != "gravity-rift" : MapPlacementCatalog.IsAsteroid(OfficialObjectives.ObstacleId(game, i)) || MapPlacementCatalog.IsDebris(OfficialObjectives.ObstacleId(game, i))));
    private static void AwardCommandMajority(GameContext game, int obstacle)
    {
        int one = 0, two = 0;
        foreach (int ship in OfficialObjectives.LiveShips(game))
            if (OfficialObjectives.DistanceToPiece(game, ship, obstacle) <= TabletopMeasurements.DistanceOne + Epsilon)
            { if (game.State.Owners[ship] == 1) one += game.Topology.Ships[ship].CatalogEntry.Definition.Command; else two += game.Topology.Ships[ship].CatalogEntry.Definition.Command; }
        if (one != two) OfficialObjectives.AwardVictory(game, one > two ? 1 : 2);
    }
    private static void RemoveObstaclesTouchingRift(GameContext game)
    {
        int[] rifts = OfficialObjectives.Obstacles(game).Where(i => OfficialObjectives.ObstacleId(game, i) == "gravity-rift").ToArray();
        foreach (int obstacle in OfficialObjectives.Obstacles(game).ToArray())
        {
            if (rifts.Contains(obstacle)) continue;
            var piece = game.State.SpatialObjects[obstacle];
            if (rifts.Any(i => Geometry2D.Contact(ObstacleGeometry.Shape(game, obstacle), ObstacleGeometry.Shape(game, i)) != ContactKind.Separated))
            {
                game.State.SpatialObjects[obstacle] = piece with { Active = false };
                TokenObstacleAttachments.RemoveObstacle(game, obstacle);
            }
        }
    }
    private static void FinishDoomedStationRound(GameContext game)
    {
        RemoveObstaclesTouchingRift(game);
        foreach (int station in OfficialObjectives.Obstacles(game).Where(i => OfficialObjectives.ObstacleId(game, i) == "station")) AwardCommandMajority(game, station);
        var state = OfficialObjectives.State(game)!;
        foreach (string key in state.Flags.Keys.Where(k => k.StartsWith("navigation.moved.", StringComparison.Ordinal)).ToArray()) state.Flags.Remove(key);
        foreach (int marker in OfficialObjectives.TokenIndices(game).ToArray()) OfficialObjectives.RemoveToken(game, marker);
    }
    private static void MoveObjectiveObstacle(GameContext game, GameContinuation frame, GameAction action)
    {
        var destination = new Pose(action.Position!.Value, action.YawDegrees);
        OfficialObjectives.MoveObstacle(game, frame.Target, destination, TabletopMeasurements.DistanceBand(2),
            frame.Text == "doomed-station" ? OfficialObjectives.Obstacles(game).First(i => OfficialObjectives.ObstacleId(game, i) == "gravity-rift") : -1, flipped: action.Flipped);
    }
    private static bool CoronaInArc(GameContext game, AttackState attack)
    {
        if (!OfficialObjectives.State(game)!.Flags.TryGetValue("navigation.corona.edge", out long side)) return false;
        var shape = game.ShipBase(attack.Attacker);
        var arc = game.State.Ships[attack.Attacker].HasAuxiliaryHullZones ? HullGeometry.Huge(shape, attack.FromZone,
            game.Topology.Ships[attack.Attacker].CatalogEntry.Definition.ShipType == "prototype-super-star-destroyer") : HullGeometry.Standard(shape, attack.FromZone);
        var half = game.Topology.BoardSizeMillimetres / 2; float x = side * half.X;
        return HullGeometry.ClipToArc(new Segment(new(x, -half.Y), new(x, half.Y)), arc, out _);
    }
    private static int First(GameContext game) => OfficialObjectives.FirstPlayer(game);
    private static int Second(GameContext game) => OfficialObjectives.SecondPlayer(game);
    private static int Anchor(GameContext game, int owner) => game.Topology.Ships.First(x => x.Owner == owner && !x.IsScenarioUnit).EntityIndex;
    private static IEnumerable<int> Ships(GameContext game, int owner) => game.Topology.Ships.Where(x => x.Owner == owner && !x.IsScenarioUnit).Select(x => x.EntityIndex);
    private static void Replace(GameContext game, GameContinuation frame, GameContinuation next)
    { int index = game.Flow.Continuations.IndexOf(frame); if (index < 0) throw new InvalidOperationException("Objective continuation is absent."); game.Flow.Continuations[index] = next; }
    private static void Pop(GameContext game, GameContinuation frame)
    { game.Flow.Continuations.Remove(frame); game.Flow.Phase = GamePhase.AbilityChoice; }
}
