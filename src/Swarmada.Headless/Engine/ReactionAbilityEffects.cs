using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Enemy activation and observed-speed reactions, without scene singletons.</summary>
public static class ReactionAbilityEffects
{
    public static bool SupportsUpgrade(string id) => id is "suppressor-title" or "the-grand-inquisitor-officer";
    public static void SnapshotSpeeds(GameContext game)
    {
        for (int ship = 0; ship < game.State.Ships.Length; ship++) game.Flow.Flags["speed.observed." + ship] = game.State.Ships[ship].CurrentSpeed;
        game.Flow.Flags["speed.observed.initialized"] = 1;
    }
    public static bool BeginActivationResponses(GameContext game, int selectedShip)
    {
        if (!game.Flow.Flags.ContainsKey("speed.observed.initialized")) { SnapshotSpeeds(game); return false; }
        var pairs = new List<int>();
        foreach (var carrier in game.SceneShipViews.Where(ship => ship.Loadout.Find("the-grand-inquisitor-officer") is { Active: true, Exhausted: false }))
            foreach (var enemy in game.SceneShipViews.Where(enemy => enemy.OwnerId != carrier.OwnerId
                && game.Flow.Flags.GetValueOrDefault("speed.observed." + enemy.Index) != enemy.State.CurrentSpeed
                && RaidRules.FootprintDistance(carrier, enemy) <= 13.260394f))
            { pairs.Add(carrier.Index); pairs.Add(enemy.Index); }
        if (pairs.Count == 0) { SnapshotSpeeds(game); return false; }
        EngineEffects.PushChoice(game, new("general.inquisitor", selectedShip, Indices: pairs.ToArray()), []); return true;
    }
    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        if (timing != "end-activation" || !game.IsShip(source) || game.IsDestroyed(source)) yield break;
        foreach (var carrier in game.SceneShipViews.Where(carrier => carrier.OwnerId != game.State.Owners[source]
            && carrier.Loadout.Find("suppressor-title") is { Active: true, Exhausted: false }
            && RaidRules.FootprintDistance(carrier, game.ShipViews[source]) <= 8.09628f))
            foreach (var token in Enum.GetValues<DefenseTokenType>().Where(game.State.Ships[source].IsDefenseTokenReady))
                yield return new(new(GameActionKind.UseAbility, carrier.Index, source, Token: token, AbilityId: "reaction.suppressor"), "Exhaust enemy " + token + " with Suppressor");
    }
    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (action.AbilityId != "reaction.suppressor") return false;
        if (game.ShipViews[action.Source].Loadout.Find("suppressor-title")?.Exhaust() != true || !game.State.Ships[action.Target].SpendDefenseToken(action.Token))
            throw new InvalidOperationException("Suppressor cannot exhaust the chosen token.");
        return true;
    }
    public static bool Handles(string rule) => rule == "general.inquisitor";
    private static IEnumerable<GameActionOption> Options(GameContext game, GameContinuation frame)
    {
        if (frame.Indices is null || frame.Value >= frame.Indices.Length)
        { yield return new(new(GameActionKind.Pass, frame.Source, AbilityId: "reaction.inquisitor.finish"), "Begin selected ship activation"); yield break; }
        int carrier = frame.Indices[(int)frame.Value], enemy = frame.Indices[(int)frame.Value + 1];
        if (!game.IsDestroyed(carrier) && !game.IsDestroyed(enemy) && game.ShipViews[carrier].Loadout.Find("the-grand-inquisitor-officer") is { Active: true, Exhausted: false })
            foreach (int change in new[] { -1, 1 })
            {
                int speed = game.State.Ships[carrier].CurrentSpeed + change;
                if (speed >= 0 && speed <= game.Topology.Ships[carrier].CatalogEntry.Definition.MaximumSpeed)
                    yield return new(new(GameActionKind.ChangeSpeed, carrier, enemy, Amount: change, Speed: speed, AbilityId: "reaction.inquisitor.speed"), "Respond to enemy speed change: " + change.ToString("+0;-0"));
            }
        yield return new(new(GameActionKind.Pass, carrier, enemy, AbilityId: "reaction.inquisitor.skip"), "Skip this Grand Inquisitor response");
    }
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = Options(game, frame).ToArray(); bool forced = options.Length == 1;
        return new(forced ? DecisionKind.Forced : DecisionKind.Player, forced ? 0 : game.State.Owners[options[0].Template.Source], GamePhase.AbilityChoice, options);
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!Options(game, frame).Any(option => GameSession.Matches(option, action))) throw new ArgumentException("Illegal Grand Inquisitor response.");
        if (action.AbilityId == "reaction.inquisitor.finish")
        {
            SnapshotSpeeds(game); game.Flow.Continuations.RemoveAt(game.Flow.Continuations.Count - 1); game.Flow.PendingOptions = []; game.Flow.Phase = GamePhase.SelectShip;
            if (!game.IsDestroyed(frame.Source) && !RoundModule.CheckTerminal(game)) new RoundModule().Apply(game, new(GameActionKind.ActivateShip, frame.Source));
            return;
        }
        game.Flow.Continuations[^1] = frame with { Value = frame.Value + 2 };
        if (action.Kind == GameActionKind.Pass) return;
        int previous = game.State.Ships[action.Source].CurrentSpeed;
        if (game.ShipViews[action.Source].Loadout.Find("the-grand-inquisitor-officer")?.Exhaust() != true) throw new InvalidOperationException("Grand Inquisitor unavailable.");
        game.State.Ships[action.Source].SetSpeed(action.Speed, game.Topology.Ships[action.Source].CatalogEntry.Definition.MaximumSpeed);
        RoundModule.ObserveObjectives(game);
        if (game.State.Ships[action.Source].DamageWhenChangingSpeed(previous) > 0)
            CombatModule.QueueDamage(game, action.Source, 1, false, false, GamePhase.AbilityChoice, rule: "grand-inquisitor.thruster-fissure");
    }
}
