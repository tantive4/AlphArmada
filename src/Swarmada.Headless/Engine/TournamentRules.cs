using Armada.Session;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public enum TournamentEndReason { FleetDestroyed = 1, MutualDestruction, RoundLimit, TimeLimit, Concession }

public sealed record TournamentPlayerScore(int Player, int DestroyedFleetPoints,
    int CrippledHugeShipPoints, int TablingAdjustment, int AdditionalPoints, int MercyMissionPoints)
{
    public int Total => DestroyedFleetPoints + CrippledHugeShipPoints + TablingAdjustment + AdditionalPoints + MercyMissionPoints;
}

public sealed record TournamentResult(int Winner, TournamentEndReason Reason, int CompletedRound,
    TournamentPlayerScore PlayerOne, TournamentPlayerScore PlayerTwo, int MarginOfVictory,
    int PlayerOneTournamentPoints, int PlayerTwoTournamentPoints);

/// <summary>
/// Standard tournament game completion, based on AMG Tournament Regulations 4.0
/// (27 January 2025), page 7, and Rules Reference 1.6, Huge Ship/Scoring/Set Aside.
/// Official objective-card effects remain outside this module. Existing custom scenarios
/// retain their scenario-specific outcomes.
/// </summary>
public static class TournamentRules
{
    public const int RoundLimit = 6;
    public const int MinimumTabledFleetValue = 400;
    private const string ReasonKey = "tournament.end-reason";
    private const string TimeKey = "tournament.time-called";
    private const string ConcessionKey = "tournament.conceding-player";

    public static bool Applies(GameContext game) => game.State.Objective is null or Swarmada.Headless.Objective.OfficialObjectiveState &&
        string.Equals(MapScenarioCatalog.Normalize(game.Topology.ScenarioId), MapScenarioCatalog.DefaultId, StringComparison.Ordinal);

    /// <summary>Call when a tournament clock expires; completion waits for this round's Status phase.</summary>
    public static void RequestTimeLimit(GameContext game)
    {
        RequireDefaultGame(game);
        if (game.State.Outcome is null) game.Flow.Flags[TimeKey] = 1;
    }

    /// <summary>Concession is an explicit caller operation, not a generated policy action.</summary>
    public static TournamentResult Concede(GameContext game, int player)
    {
        RequireDefaultGame(game);
        if (player is not (1 or 2)) throw new ArgumentOutOfRangeException(nameof(player));
        if (game.State.Outcome is not null) throw new InvalidOperationException("The game has already ended.");
        game.Flow.Flags[ConcessionKey] = player;
        Finish(game, 3 - player, TournamentEndReason.Concession);
        return GetResult(game)!;
    }

    /// <summary>Called once the full Status phase and all destruction reactions have resolved.</summary>
    internal static bool ResolveEndOfRound(GameContext game)
    {
        RequireDefaultGame(game);
        if (game.State.Outcome is not null) return true;
        bool one = HasNonFlotillaInPlay(game, 1), two = HasNonFlotillaInPlay(game, 2);
        if (!one || !two)
        {
            Finish(game, one ? 1 : two ? 2 : 3 - game.Topology.FirstPlayer,
                one || two ? TournamentEndReason.FleetDestroyed : TournamentEndReason.MutualDestruction);
            return true;
        }
        bool timed = game.Flow.Flags.ContainsKey(TimeKey);
        if (game.State.Round < RoundLimit && !timed) return false;
        var first = CalculateScore(game, 1); var second = CalculateScore(game, 2);
        int winner = first.Total > second.Total ? 1 : second.Total > first.Total ? 2 : 3 - game.Topology.FirstPlayer;
        Finish(game, winner, timed ? TournamentEndReason.TimeLimit : TournamentEndReason.RoundLimit, first, second);
        return true;
    }

    /// <summary>
    /// Calculates a final-state score without changing the position. Set-aside units count
    /// as destroyed at game completion. Do not use this as a nonterminal evaluation function.
    /// </summary>
    public static TournamentPlayerScore CalculateScore(GameContext game, int player)
    {
        RequireValidSetup(game);
        if (player is not (1 or 2)) throw new ArgumentOutOfRangeException(nameof(player));
        int enemy = 3 - player, destroyed = 0, crippled = 0, enemyFleetValue = 0;
        for (int index = 0; index < game.Topology.Ships.Count; index++)
        {
            var ship = game.Topology.Ships[index];
            if (ship.IsScenarioUnit || game.State.Owners[index] != enemy) continue;
            int cost = ShipPointCost(game, index); enemyFleetValue += cost;
            if (CountsAsDestroyed(game, index)) destroyed += cost;
            else if (string.Equals(ship.CatalogEntry.Metadata.Size, "huge", StringComparison.OrdinalIgnoreCase) &&
                game.State.Ships[index].DamageCards.Count >= (game.State.Ships[index].MaximumHull + 1) / 2)
                crippled += (cost + 1) / 2;
        }
        for (int index = 0; index < game.Topology.Squadrons.Count; index++)
        {
            var squadron = game.Topology.Squadrons[index];
            if (game.State.Owners[squadron.EntityIndex] != enemy) continue;
            int cost = squadron.CatalogEntry.Definition.Points; enemyFleetValue += cost;
            if (CountsAsDestroyed(game, squadron.EntityIndex)) destroyed += cost;
        }
        // Tabling values the entire fleet, including surviving flotillas/squadrons, at
        // its actual value or 400, whichever is greater. Objective/card bonuses are added afterwards.
        int table = !HasNonFlotillaInPlay(game, enemy) ? Math.Max(MinimumTabledFleetValue, enemyFleetValue) - destroyed - crippled : 0;
        int mercy = 0;
        foreach (var award in MapAbilityEffects.GetMercyMissionAwards(game)) if (award.Player == player) mercy += award.Points;
        return new(player, destroyed, crippled, table, game.State.VictoryPoints[player - 1] + OfficialObjectives.ScoreBonus(game, player), mercy);
    }

    public static bool CountsAsDestroyed(GameContext game, int entity) => game.IsReserved(entity) || game.IsDestroyed(entity);

    public static bool HasNonFlotillaInPlay(GameContext game, int player)
    {
        for (int index = 0; index < game.Topology.Ships.Count; index++)
        {
            var ship = game.Topology.Ships[index];
            if (game.State.Owners[index] == player && !ship.IsScenarioUnit && ship.CatalogEntry.Metadata.IsFlotilla != true &&
                !game.IsReserved(index) && !game.IsDestroyed(index)) return true;
        }
        return false;
    }

    /// <summary>Includes every equipped upgrade at its current owner, including discarded cards.</summary>
    public static int ShipPointCost(GameContext game, int ship)
    {
        if (!game.IsShip(ship)) throw new ArgumentOutOfRangeException(nameof(ship));
        int cost = game.Topology.Ships[ship].CatalogEntry.Definition.Points;
        for (int index = 0; index < game.State.Upgrades.Length; index++)
            if (game.State.Registers[game.Topology.Registers.UpgradeOwner(index)] == ship + 1)
                cost += game.State.Upgrades[index].EffectiveDefinition.points;
        return cost;
    }

    /// <summary>Full tournament results are materialized only on request, never at MCTS action boundaries.</summary>
    public static TournamentResult? GetResult(GameContext game)
    {
        if (OfficialObjectives.State(game)?.InvalidSetup is not null) return null;
        if (game.State.Outcome is not { } outcome || !game.Flow.Flags.TryGetValue(ReasonKey, out long reasonValue)) return null;
        var reason = (TournamentEndReason)reasonValue;
        int margin = reason == TournamentEndReason.MutualDestruction ? 0 : Math.Clamp((int)(outcome.Winner == 1
            ? outcome.PlayerOneScore - outcome.PlayerTwoScore : outcome.PlayerTwoScore - outcome.PlayerOneScore), 0, 400);
        if (reason == TournamentEndReason.Concession) margin = Math.Max(140, margin);
        int winningPoints = margin < 60 ? 6 : margin < 140 ? 7 : margin < 220 ? 8 : margin < 300 ? 9 : 10;
        int losingPoints = reason == TournamentEndReason.Concession ? 0 : 11 - winningPoints;
        return new(outcome.Winner, reason, game.State.Round, CalculateScore(game, 1), CalculateScore(game, 2), margin,
            outcome.Winner == 1 ? winningPoints : losingPoints, outcome.Winner == 2 ? winningPoints : losingPoints);
    }

    private static void Finish(GameContext game, int winner, TournamentEndReason reason,
        TournamentPlayerScore? first = null, TournamentPlayerScore? second = null)
    {
        first ??= CalculateScore(game, 1); second ??= CalculateScore(game, 2);
        game.Flow.Flags[ReasonKey] = (int)reason;
        game.State.Outcome = new(winner, first.Total, second.Total);
        game.Flow.Phase = GamePhase.Terminal;
    }
    private static void RequireDefaultGame(GameContext game)
    {
        RequireValidSetup(game);
        if (!Applies(game)) throw new InvalidOperationException("Tournament completion applies only to the default game mode.");
    }
    private static void RequireValidSetup(GameContext game)
    {
        if (OfficialObjectives.State(game)?.InvalidSetup is { } rejected) throw new InvalidSetupException(rejected);
    }
}
