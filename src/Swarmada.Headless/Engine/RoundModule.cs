using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public sealed partial class RoundModule : IGameplayModule
{
    public bool Handles(GamePhase phase) => phase is GamePhase.SelectShip or GamePhase.Status or GamePhase.Terminal or GamePhase.StartRound or GamePhase.StartShipPhase;
    public GameDecision GetDecision(GameContext game)
    {
        if (game.State.Outcome is { } outcome) return new(DecisionKind.Terminal, 0, GamePhase.Terminal, [], outcome);
        if (game.Flow.Phase is GamePhase.Status or GamePhase.StartRound or GamePhase.StartShipPhase) return RoundPhaseDecision(game);
        if (CheckTerminal(game)) return new(DecisionKind.Terminal, 0, GamePhase.Terminal, [], game.State.Outcome);
        var ready = ReadyShips(game, game.Flow.CurrentPlayer).ToArray();
        if (ready.Length == 0) return Forced(game, "Advance ship activation");
        return new(DecisionKind.Player, game.Flow.CurrentPlayer, GamePhase.SelectShip,
            ready.Select(entity => new GameActionOption(new(GameActionKind.ActivateShip, entity),
                "Activate " + game.Topology.Ships[entity].CatalogEntry.Definition.DisplayName)).ToArray());
    }
    public void Apply(GameContext game, GameAction action)
    {
        if (game.Flow.Phase is GamePhase.Status or GamePhase.StartRound or GamePhase.StartShipPhase)
        { ApplyRoundChoice(game, CurrentRoundChoice(game) ?? throw new InvalidOperationException("No pending phase choice."), action); return; }
        if (game.Flow.Phase != GamePhase.SelectShip) throw new InvalidOperationException("Cannot activate in this phase.");
        if (!ReadyShips(game, game.Flow.CurrentPlayer).Any())
        {
            int other = 3 - game.Flow.CurrentPlayer;
            if (ReadyShips(game, other).Any()) { game.Flow.CurrentPlayer = other; return; }
            game.Flow.ActiveShip = -1; game.Flow.SquadronPhasePlayer = game.Topology.FirstPlayer;
            game.Flow.SquadronPhaseRemaining = 2; game.Flow.Phase = GamePhase.SquadronPhase;
            game.Flow.Flags["round.squadron-phase-start"] = 0;
            if (EngineEffects.BeginWindow(game, "end-ship-phase", -1, resumePhase: GamePhase.SquadronPhase)) return;
            game.Flow.Flags["round.squadron-phase-start"] = 1;
            EngineEffects.BeginWindow(game, "before-start-squadron-phase", -1, resumePhase: GamePhase.SquadronPhase); return;
        }
        if (action.Kind != GameActionKind.ActivateShip || !ReadyShips(game, game.Flow.CurrentPlayer).Contains(action.Source))
            throw new ArgumentException("Choose an eligible ship.", nameof(action));
        int ship = action.Source;
        if (ReactionAbilityEffects.BeginActivationResponses(game, ship)) return;
        game.State.Activated[ship] = true; game.Flow.ActiveShip = ship;
        game.Flow.Flags["activation.serial"] = game.Flow.Flags.GetValueOrDefault("activation.serial") + 1;
        game.Flow.Flags["activation.in-progress"] = ship + 1;
        OfficialObjectives.State(game)?.Flags.Remove("rift.used." + ship);
        game.Flow.ShipActivations[ship] = new();
        DravenAbilityEffects.ResetActivation(game, ship);
        game.Flow.Flags.Remove($"crew-panic.{ship}");
        foreach (string key in game.Flow.Flags.Keys.Where(x => x.StartsWith($"wolffe.{ship}.", StringComparison.Ordinal) || x.StartsWith($"ahsoka.{ship}.", StringComparison.Ordinal)).ToArray()) game.Flow.Flags.Remove(key);
        foreach (var card in game.UpgradesFor(ship)) { card.ResetActivationUsage(); card.ResetAttackUsage(); }
        foreach (var squadron in game.State.Squadrons) squadron.ClearShipActivationMovementRestriction();
        game.Flow.Phase = GamePhase.RevealCommand;
        EngineEffects.BeginWindow(game, "start-activation", ship, resumePhase: GamePhase.RevealCommand);
    }
    private static IEnumerable<int> ReadyShips(GameContext game, int owner)
    {
        bool anyRegular = game.Topology.Ships.Any(x => !x.IsScenarioUnit && !game.IsDestroyed(x.EntityIndex)
            && !game.IsReserved(x.EntityIndex) && !game.State.Activated[x.EntityIndex]);
        foreach (var ship in game.Topology.Ships)
            if (!ship.IsStation && ship.Owner == owner && !game.IsDestroyed(ship.EntityIndex) && !game.IsReserved(ship.EntityIndex)
                && !game.State.Activated[ship.EntityIndex] && (!ship.IsScenarioUnit || !anyRegular)
                && (!game.Flow.Flags.ContainsKey($"raddus.block-first.{ship.EntityIndex}") ||
                    game.Topology.Ships.Any(x => x.Owner == owner && game.State.Activated[x.EntityIndex])))
                yield return ship.EntityIndex;
    }
    public static void EndShipActivation(GameContext game)
    {
        int ship = game.Flow.ActiveShip;
        if (ship >= 0)
        {
            DravenAbilityEffects.ResetActivation(game, ship);
            game.Flow.Flags.Remove("activation.in-progress");
            game.Flow.ActiveShip = -1;
            foreach (var squadron in game.State.Squadrons) squadron.ClearShipActivationMovementRestriction();
            game.Flow.CurrentPlayer = 3 - game.State.Owners[ship];
        }
        game.Flow.Phase = GamePhase.SelectShip;
        EngineEffects.BeginWindow(game, "end-activation", ship, resumePhase: GamePhase.SelectShip);
    }
    public static ObjectiveShip[] ObjectiveShips(GameContext game) => game.Topology.Ships.Select(ship =>
        new ObjectiveShip(ship.EntityIndex, game.State.Owners[ship.EntityIndex], game.ShipBase(ship.EntityIndex),
            game.State.Ships[ship.EntityIndex].CurrentSpeed, game.IsDestroyed(ship.EntityIndex),
            game.IsReserved(ship.EntityIndex), ship.CatalogEntry.Metadata.Size)).ToArray();
    public static void ObserveObjectives(GameContext game)
    {
        DamageEvents.Observe(game);
        if (game.State.Objective is null || game.Flow.Flags.ContainsKey("movement.damage-batch")) return;
        var ships = ObjectiveShips(game);
        switch (game.State.Objective)
        {
            case EscortVipObjective escort: escort.Observe(ships); break;
            case IntelRetrievalObjective intel: intel.Observe(ships); break;
            case SeizeStationObjective station: station.Observe(ships); break;
        }
    }
    public static bool CheckTerminal(GameContext game)
    {
        DamageEvents.Observe(game);
        if (game.State.Outcome is not null) return true;
        if (game.Flow.Phase == GamePhase.Deployment || game.Flow.Flags.ContainsKey("movement.damage-batch") || DamageEvents.HasPending(game)) return false;
        // Standard tournament games finish the entire game round. In particular, surviving
        // squadrons and flotillas still act after the last capital ship has been destroyed.
        if (TournamentRules.Applies(game)) return false;
        ObserveObjectives(game);
        ObjectiveOutcome? objective = game.State.Objective switch
        { EscortVipObjective e => e.Outcome, IntelRetrievalObjective i => i.Outcome, SeizeStationObjective s => s.Outcome, _ => null };
        int winner;
        if (objective is not null) winner = objective.Winner;
        else
        {
            bool one = game.Topology.Ships.Any(x => !x.IsScenarioUnit && x.Owner == 1 && !game.IsReserved(x.EntityIndex) && !game.ShipViews[x.EntityIndex].IsDestroyed);
            bool two = game.Topology.Ships.Any(x => !x.IsScenarioUnit && x.Owner == 2 && !game.IsReserved(x.EntityIndex) && !game.ShipViews[x.EntityIndex].IsDestroyed);
            if (one && two) return false;
            // Escort transport victory has its own interceptor elimination and VIP loss precedence.
            if (game.State.Objective is EscortVipObjective escort && one != two)
            {
                var ships = ObjectiveShips(game);
                foreach (var ship in ships.Where(x => x.Destroyed)) escort.NotifyShipDestroyed(ship, ships);
                if (escort.Outcome is null) return false;
                winner = escort.Outcome.Winner;
            }
            else winner = one ? 1 : two ? 2 : 0;
        }
        game.State.Outcome = new(winner, game.State.VictoryPoints[0], game.State.VictoryPoints[1]);
        game.Flow.Phase = GamePhase.Terminal; return true;
    }
    private static GameDecision Forced(GameContext game, string label) => new(DecisionKind.Forced, 0, game.Flow.Phase,
        [new(new(GameActionKind.Pass), label)]);
}
