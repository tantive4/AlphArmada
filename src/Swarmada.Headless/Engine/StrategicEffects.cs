using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>RRG 1.6, p19: one optional physical objective-token move after a
/// Strategic squadron finishes moving. The parent timing window owns usage and order.</summary>
public static class StrategicEffects
{
    public const string AbilityId = "keyword.strategic";
    private const string SelectToken = "strategic.choice.token";
    private const string MoveToken = "strategic.choice.destination";

    public static bool Handles(string rule) => rule is SelectToken or MoveToken;

    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        if (MapMovementRules.Enabled(game) && timing == "after-squadron-move" && IsEligibleSquadron(game, source) && EligibleTokens(game, source).Any())
            yield return new(new(GameActionKind.UseAbility, source, AbilityId: AbilityId),
                "Strategic: move one nearby objective token");
    }

    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (action.AbilityId != AbilityId) return false;
        if (!MapMovementRules.Enabled(game)) throw new ArgumentException("Objective-token relocation is disabled for this match.", nameof(action));
        EngineEffects.PushChoice(game, new(SelectToken, action.Source), []);
        return true;
    }

    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = new List<GameActionOption>();
        if (MapMovementRules.Enabled(game) && IsEligibleSquadron(game, frame.Source))
        {
            if (frame.Rule == SelectToken)
            {
                foreach (int token in EligibleTokens(game, frame.Source))
                    options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, Index: token, AbilityId: AbilityId),
                        "Choose an objective token at distance one"));
            }
            else if (frame.Rule == MoveToken && IsEligibleToken(game, frame.Source, frame.Target))
                options.Add(new(new(GameActionKind.PlaceToken, frame.Source, Index: frame.Target, AbilityId: AbilityId),
                    "Move the objective token to distance one of the squadron", AcceptsPosition: true));
        }
        options.Add(new(new(GameActionKind.Pass, frame.Source, AbilityId: AbilityId), "Decline Strategic"));
        return new(options.Count == 1 ? DecisionKind.Forced : DecisionKind.Player,
            options.Count == 1 ? 0 : game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
    }

    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!GetDecision(game, frame).Actions.Any(option => GameSession.Matches(option, action)))
            throw new ArgumentException("Illegal Strategic choice.", nameof(action));
        if (action.Kind == GameActionKind.Pass) { EngineEffects.ReturnFromChoice(game); return; }
        if (frame.Rule == SelectToken)
        {
            game.Flow.Continuations[^1] = frame with { Rule = MoveToken, Target = action.Index };
            return;
        }
        var position = action.Position ?? throw new ArgumentException("Strategic requires a token destination.", nameof(action));
        Geometry2D.RequireFinite(position, nameof(action));
        var token = game.State.SpatialObjects[frame.Target];
        var destination = new Circle(position, MathF.Max(token.Size.X, token.Size.Y) / 2);
        if (!PlayAreaGeometry.Contains(MovementModule.Board(game), destination)
            || Geometry2D.Distance(game.SquadronBase(frame.Source), destination) > TabletopMeasurements.DistanceOne + Geometry2D.DefaultTolerance)
            throw new ArgumentException("Strategic destination must be in the play area and at distance one of the squadron.", nameof(action));
        // Remaining at the same position creates no movement/removal event for Moralo.
        // Tokens may overlap ships, squadrons, obstacles, and other tokens (RRG FAQ).
        if (position != token.Pose.Position)
            TokenObstacleAttachments.MoveToken(game, frame.Target, new(position, token.Pose.YawDegrees));
        EngineEffects.ReturnFromChoice(game);
    }

    private static bool IsEligibleSquadron(GameContext game, int source) => source >= game.State.Ships.Length
        && source < game.Topology.EntityCount && !game.IsDestroyed(source) && !game.IsReserved(source)
        && SquadronAbilityEngine.HasKeyword(game.Topology.Squadrons[game.SquadronIndex(source)].CatalogEntry.Definition, "STRATEGIC");

    private static IEnumerable<int> EligibleTokens(GameContext game, int source)
    {
        foreach (int token in OfficialObjectives.TokenIndices(game))
            if (IsEligibleToken(game, source, token)) yield return token;
    }

    private static bool IsEligibleToken(GameContext game, int source, int token) => token >= 0
        && token < game.State.SpatialObjects.Count && game.State.SpatialObjects[token] is { Active: true, Kind: 1 }
        && OfficialObjectives.DistanceToPiece(game, source, token) <= TabletopMeasurements.DistanceOne + Geometry2D.DefaultTolerance;
}
