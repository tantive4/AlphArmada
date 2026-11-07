using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Executable official Assault objectives. Objective choice is an input supplied
/// before match preparation; physical choices remain natural engine decisions.</summary>
public static class OfficialAssaultObjectives
{
    private const string Prefix = "objective.assault.";
    public const string SurpriseRevealRule = Prefix + "surprise-reveal";
    public const string SurpriseRevealAction = SurpriseRevealRule;
    private static OfficialObjectiveState State(GameContext game) => OfficialObjectives.State(game)!;
    public static bool Handles(string rule) => rule.StartsWith(Prefix, StringComparison.Ordinal);
    public static IEnumerable<GameContinuation> SetupChoices(GameContext game, bool afterDeployment)
    {
        string id = OfficialObjectives.Id(game);
        int first = OfficialObjectives.FirstPlayer(game), second = OfficialObjectives.SecondPlayer(game);
        if (!afterDeployment)
        {
            if (id == "blockade-run" && !State(game).Flags.ContainsKey("blockade.assigned"))
            {
                State(game).Flags["blockade.assigned"] = 1;
                foreach (var ship in game.Topology.Ships.Where(ship => !ship.IsScenarioUnit && game.State.Owners[ship.EntityIndex] == second))
                    OfficialObjectives.SetTokens(game, ship.EntityIndex, 1);
            }
            if (id == "targeting-beacons") yield return new(Prefix + "beacons", Target: second);
            yield break;
        }
        if (id == "advanced-gunnery")
        {
            yield return new(Prefix + "objective-ship", Source: first, Target: first);
            yield return new(Prefix + "objective-ship", Source: second, Target: second);
        }
        if (id == "most-wanted")
        {
            yield return new(Prefix + "objective-ship", Source: second, Target: second);
            yield return new(Prefix + "objective-ship", Source: second, Target: first);
        }
        if (id == "opening-salvo") foreach (int ship in OfficialObjectives.LiveShips(game)) OfficialObjectives.SetTokens(game, ship, 1);
        if (id == "precision-strike") foreach (int ship in OfficialObjectives.LiveShips(game, second)) TokenAssignments.TryAssign(game, ship, ShipCommand.ConcentrateFire);
        if (id == "surprise-attack") yield return new(Prefix + "surprise-dials", Target: second);
    }
    public static IEnumerable<GameContinuation> TimingChoices(GameContext game, string timing, int source, int target)
    {
        if (OfficialObjectives.Is(game, "surprise-attack") && timing == "start-ship-phase" && game.State.Round <= 3
            && State(game).Flags.ContainsKey("surprise.dial." + (game.State.Round - 1)))
            yield return new(Prefix + "surprise-reveal", Target: OfficialObjectives.SecondPlayer(game));
        if (OfficialObjectives.Is(game, "rift-assault") && timing == "end-round")
            foreach (int ship in OfficialObjectives.LiveShips(game).Where(ship => game.State.Ships[ship].CurrentSpeed <= 1
                && OfficialObjectives.Obstacles(game).Any(piece => OfficialObjectives.ObstacleId(game, piece) == "gravity-rift"
                    && OfficialObjectives.DistanceToPiece(game, ship, piece) <= TabletopMeasurements.DistanceBand(2))))
                yield return new(Prefix + "rift-damage", Source: ship);
    }
    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target) => [];
    public static bool TryApply(GameContext game, GameContinuation window, GameAction action) => false;
    public static IEnumerable<GameActionOption> GetEngineeringActions(GameContext game, int ship)
    {
        if (OfficialObjectives.Is(game, "ion-storm") && OfficialObjectives.TokenCount(game, ship) > 0)
            yield return new(new(GameActionKind.UseAbility, ship, AbilityId: Prefix + "ion-repair"), "Discard Ion Storm objective token");
    }
    public static bool TryApplyEngineering(GameContext game, GameAction action)
    {
        if (action.AbilityId != Prefix + "ion-repair" || !OfficialObjectives.Is(game, "ion-storm")
            || OfficialObjectives.TokenCount(game, action.Source) == 0) return false;
        OfficialObjectives.SetTokens(game, action.Source, 0); return true;
    }
    public static void OnShipMoved(GameContext game, int ship)
    {
        if (!OfficialObjectives.Is(game, "ion-storm") || !OfficialObjectives.Live(game, ship) || OfficialObjectives.TokenCount(game, ship) != 0) return;
        float safe = game.State.Owners[ship] == OfficialObjectives.FirstPlayer(game) ? TabletopMeasurements.DistanceOne : TabletopMeasurements.DistanceBand(2);
        if (OfficialObjectives.Obstacles(game).All(index => OfficialObjectives.DistanceToPiece(game, ship, index) > safe)) OfficialObjectives.SetTokens(game, ship, 1);
    }
    public static void Observe(GameContext game)
    {
        if (!OfficialObjectives.Is(game, "blockade-run")) return;
        foreach (var ship in game.Topology.Ships)
            if (!ship.IsScenarioUnit && game.State.Owners[ship.EntityIndex] == OfficialObjectives.SecondPlayer(game)
                && game.IsDestroyed(ship.EntityIndex) && OfficialObjectives.TokenCount(game, ship.EntityIndex) > 0
                && !State(game).Flags.ContainsKey("blockade.dead." + ship.EntityIndex))
            {
                State(game).Flags["blockade.dead." + ship.EntityIndex] = 1;
                OfficialObjectives.AwardVictory(game, OfficialObjectives.FirstPlayer(game));
            }
    }
    public static int ScoreBonus(GameContext game, int player)
    {
        string id = OfficialObjectives.Id(game); int bonus = 0;
        if (id is "advanced-gunnery" or "most-wanted")
            foreach (var ship in game.Topology.Ships)
                if (game.State.Owners[ship.EntityIndex] != player && OfficialObjectives.IsObjectiveShip(game, ship.EntityIndex)
                    && TournamentRules.CountsAsDestroyed(game, ship.EntityIndex)) bonus += ship.CatalogEntry.Definition.Points;
        if (id == "opening-salvo")
            foreach (int ship in OfficialObjectives.LiveShips(game, 3 - player))
                if (game.State.Ships[ship].DamageCards.Count > 0) bonus += (TournamentRules.ShipPointCost(game, ship) + 1) / 2;
        if (id == "blockade-run" && player == OfficialObjectives.SecondPlayer(game))
        {
            var rectangle = OfficialObjectives.DeploymentZone(game, OfficialObjectives.FirstPlayer(game));
            foreach (int ship in OfficialObjectives.LiveShips(game, player))
                if (OfficialObjectives.TokenCount(game, ship) > 0 && ObjectiveGeometry.RectangleOverlap(game.ShipBase(ship), rectangle) > 0) bonus += 20;
        }
        if (id == "station-assault")
            foreach (var station in State(game).PieceHull)
            {
                if (OfficialObjectives.ObstacleId(game, station.Key) != "unarmed-station") continue;
                bool destroyed = State(game).PieceEntities.TryGetValue(station.Key, out int stationEntity)
                    ? game.IsDestroyed(stationEntity) : State(game).DestroyedPieces.Contains(station.Key);
                if (destroyed ? player == OfficialObjectives.FirstPlayer(game) : player == OfficialObjectives.SecondPlayer(game)) bonus += 40;
            }
        // Marked for Destruction has no end-game doubling rule. The extracted upstream
        // text copied Most Wanted's paragraph; the original printed card is authoritative.
        return bonus;
    }
    public static bool IsAdvancedGunnery(GameContext game, int ship) => OfficialObjectives.Is(game, "advanced-gunnery") && OfficialObjectives.IsObjectiveShip(game, ship);
    public static bool AllowsZoneReuse(GameContext game, int ship) => IsAdvancedGunnery(game, ship);
    private static string AttackKey(GameContext game, int attacker, int defender, HullZone from, HullZone to)
        => $"ag.{game.State.Round}.{attacker}.{(int)from}.{defender}.{(int)to}";
    public static bool AllowsTarget(GameContext game, int attacker, int defender, HullZone from, HullZone to)
        => !IsAdvancedGunnery(game, attacker) || game.State.Owners[attacker] != OfficialObjectives.FirstPlayer(game)
            || !State(game).Flags.ContainsKey(AttackKey(game, attacker, defender, from, to));
    public static void OnAttackDeclared(GameContext game, AttackState attack)
    {
        if (game.IsShip(attack.Attacker) && IsAdvancedGunnery(game, attack.Attacker))
            State(game).Flags[AttackKey(game, attack.Attacker, attack.Defender, attack.FromZone, attack.ToZone)] = 1;
    }
    public static void CleanupRound(GameContext game)
    {
        if (OfficialObjectives.State(game) is not { } state) return;
        foreach (string key in state.Flags.Keys.Where(key => key.StartsWith("ag.", StringComparison.Ordinal)).ToArray()) state.Flags.Remove(key);
    }
    public static IEnumerable<GameContinuation> PurrgilMovedChoices(GameContext game, int piece, int player)
    {
        if (OfficialObjectives.Is(game, "marked-for-destruction") && OfficialObjectives.ObstacleId(game, piece).StartsWith("purrgil", StringComparison.Ordinal)
            && MarkedTargets(game, piece).Any()) yield return new(Prefix + "marked-purrgil", Source: piece, Target: player);
    }
    private static IEnumerable<int> MarkedTargets(GameContext game, int piece) => OfficialObjectives.LiveShips(game)
        .Where(ship => OfficialObjectives.TokenCount(game, ship) > 0 && OfficialObjectives.DistanceToPiece(game, ship, piece) <= TabletopMeasurements.DistanceOne);
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = new List<GameActionOption>(); int owner = frame.Target;
        switch (frame.Rule)
        {
            case Prefix + "objective-ship":
                owner = frame.Source;
                foreach (int ship in OfficialObjectives.LiveShips(game, frame.Target)) options.Add(new(new(GameActionKind.ChooseEntity, Target: ship), "Choose objective ship " + ship));
                break;
            case Prefix + "surprise-dials":
                foreach (var command in Enum.GetValues<ShipCommand>()) options.Add(new(new(GameActionKind.ChooseCommand, Command: command), "Place facedown " + command + " dial " + (frame.Stage + 1)));
                break;
            case Prefix + "beacons":
                options.Add(new(new(GameActionKind.PlaceToken), "Place Targeting Beacon " + (frame.Stage + 1), AcceptsPosition: true)); break;
            case Prefix + "marked-purrgil":
                foreach (int ship in MarkedTargets(game, frame.Source)) options.Add(new(new(GameActionKind.ChooseEntity, Target: ship), "Discard token and damage marked ship " + ship));
                options.Add(new(new(GameActionKind.Pass), "Decline Marked for Destruction")); break;
            case Prefix + "rift-damage":
                return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), "Resolve " + frame.Rule)]);
            case SurpriseRevealRule:
                // An explicit singleton exposes a legal search suspension point
                // before this private dial becomes public. Real play advances it
                // without a policy/model decision through ForcedActionAdvancer.
                var revealCommand = (ShipCommand)State(game).Flags["surprise.dial." + (game.State.Round - 1)];
                return new(DecisionKind.Player, OfficialObjectives.SecondPlayer(game), GamePhase.AbilityChoice,
                    [new(new(GameActionKind.RevealCommand, Command: revealCommand, AbilityId: SurpriseRevealAction), "Reveal Surprise Attack command")]);
            default: throw new NotSupportedException("Unknown Assault objective continuation " + frame.Rule);
        }
        if (options.Count == 0) return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), "No eligible objective ship")]);
        return new(DecisionKind.Player, owner, GamePhase.AbilityChoice, options);
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!GetDecision(game, frame).Actions.Any(option => GameSession.Matches(option, action))) throw new ArgumentException("Illegal Assault objective choice.");
        switch (frame.Rule)
        {
            case Prefix + "objective-ship":
                if (action.Kind != GameActionKind.Pass) OfficialObjectives.SetObjectiveShip(game, action.Target); break;
            case Prefix + "surprise-dials":
                State(game).Flags["surprise.dial." + frame.Stage] = (int)action.Command;
                if (frame.Stage < 2) { game.Flow.Continuations[^1] = frame with { Stage = frame.Stage + 1 }; return; } break;
            case SurpriseRevealRule:
                State(game).Flags["surprise.revealed." + (game.State.Round - 1)] = 1;
                foreach (int ship in OfficialObjectives.LiveShips(game, OfficialObjectives.FirstPlayer(game))) game.State.Commands[ship].TryGainRaidToken(action.Command); break;
            case Prefix + "beacons":
                RequireSetupPosition(game, action.Position, new Vector2(25.4f), action.YawDegrees);
                OfficialObjectives.PlaceToken(game, action.Position!.Value, id: "targeting-beacon");
                if (frame.Stage < 3) { game.Flow.Continuations[^1] = frame with { Stage = frame.Stage + 1, Target = 3 - frame.Target }; return; } break;
            case Prefix + "marked-purrgil":
                if (action.Kind == GameActionKind.Pass) break;
                OfficialObjectives.SetTokens(game, action.Target, OfficialObjectives.TokenCount(game, action.Target) - 1);
                OfficialObjectives.AwardVictory(game, 3 - game.State.Owners[action.Target]);
                EngineEffects.ReturnFromChoice(game);
                CombatModule.QueueDamage(game, action.Target, 1, false, true, GamePhase.AbilityChoice, rule: "objective.marked.purrgil"); return;
            case Prefix + "rift-damage":
                EngineEffects.ReturnFromChoice(game);
                CombatModule.QueueDamage(game, frame.Source, 1, false, true, GamePhase.AbilityChoice, rule: "objective.rift.end-round"); return;
        }
        EngineEffects.ReturnFromChoice(game);
    }
    private static void RequireSetupPosition(GameContext game, Vector2? position, Vector2 size, float yaw)
    {
        if (position is not { } point) throw new ArgumentException("Objective placement requires a position.");
        Geometry2D.RequireFinite(point, nameof(position));
        var rectangle = new OrientedRectangle(new(point, yaw), size); var half = game.Topology.BoardSizeMillimetres / 2;
        half.X = Math.Min(half.X, 609.6f);
        Span<Vector2> corners = stackalloc Vector2[4]; rectangle.CopyCorners(corners);
        foreach (var corner in corners) if (MathF.Abs(corner.X) > half.X || MathF.Abs(corner.Y) > half.Y) throw new ArgumentException("Objective placement is outside the setup area.");
    }
    public static void ObserveFaceupCard(GameContext game, int defender)
    {
        if (!OfficialObjectives.Is(game, "precision-strike")) return;
        for (int i = game.Flow.Attacks.Count - 1; i >= 0; i--)
        {
            var attack = game.Flow.Attacks[i];
            if (!attack.EnvironmentalDamage && attack.Defender == defender)
            { attack.EffectFlags["objective.precision.faceups"] = attack.EffectFlags.GetValueOrDefault("objective.precision.faceups") + 1; return; }
        }
    }
    public static void ObserveAttack(GameContext game, AttackState attack)
    {
        if (!OfficialObjectives.Is(game, "precision-strike") || attack.EnvironmentalDamage || attack.Attacker < 0
            || !game.IsShip(attack.Attacker) && !SquadronAbilityEngine.HasBomber(game.SquadronViews[game.SquadronIndex(attack.Attacker)])
            || attack.EffectFlags.ContainsKey("objective.precision.scored")) return;
        attack.EffectFlags["objective.precision.scored"] = 1;
        OfficialObjectives.AwardVictory(game, game.State.Owners[attack.Attacker], (int)attack.EffectFlags.GetValueOrDefault("objective.precision.faceups"));
    }
}
