using System.Globalization;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Source command-token assignment, including Tantive IV's replacement recipient choice.</summary>
public static class TokenAssignments
{
    private const string PendingPrefix = "token.pending.";
    public static bool SupportsUpgrade(string id) => id == "tantive-iv-title";

    public static bool CanAssign(GameContext game, int entity, ShipCommand command) =>
        CanReceive(game, entity, command) && (game.State.Commands[entity].CanStoreToken(command) ||
            (game.HasUpgrade(entity, "tantive-iv-title") && Recipients(game, entity, command).Any()));

    public static bool TryAssign(GameContext game, int entity, ShipCommand command)
    {
        if (!CanReceive(game, entity, command)) return false;
        if (game.HasUpgrade(entity, "tantive-iv-title") && Recipients(game, entity, command).Any())
        {
            bool stored = game.State.Commands[entity].TryStoreToken(command);
            long serial = game.Flow.Flags.GetValueOrDefault("token.next-id") + 1;
            game.Flow.Flags["token.next-id"] = serial;
            game.Flow.Flags[PendingPrefix + serial.ToString(CultureInfo.InvariantCulture)] = ((long)(uint)entity << 4) | ((long)(uint)command << 1) | (stored ? 1L : 0L);
            return true;
        }
        return game.State.Commands[entity].TryStoreToken(command);
    }

    public static bool TryBeginNext(GameContext game)
    {
        // A custom nested choice owns its stack until it returns. Ordinary timing windows
        // may receive the title's nested choice at the next synchronous decision boundary.
        if (game.Flow.Phase == GamePhase.AbilityChoice && game.Flow.Continuations.LastOrDefault() is { } top && top.Rule != "effects.window") return false;
        while (true)
        {
            string? selected = null; long earliest = long.MaxValue;
            foreach (var entry in game.Flow.Flags)
                if (entry.Key.StartsWith(PendingPrefix, StringComparison.Ordinal) &&
                    long.TryParse(entry.Key.AsSpan(PendingPrefix.Length), NumberStyles.None, CultureInfo.InvariantCulture, out long serial) && serial < earliest)
                { selected = entry.Key; earliest = serial; }
            if (selected is null) return false;
            long payload = game.Flow.Flags[selected]; game.Flow.Flags.Remove(selected);
            int source = (int)(payload >> 4); var command = (ShipCommand)((payload >> 1) & 7); bool stored = (payload & 1) != 0;
            if (!game.IsShip(source) || game.IsDestroyed(source) || game.IsReserved(source) || !game.HasUpgrade(source, "tantive-iv-title")) continue;
            if (!stored && !Recipients(game, source, command).Any()) continue;
            var frame = new GameContinuation("token.tantive", source, (int)command, (int)game.Flow.Phase, stored ? 1 : 0);
            EngineEffects.PushChoice(game, frame, []); return true;
        }
    }

    public static bool Handles(string rule) => rule == "token.tantive";
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = Options(game, frame).ToArray();
        return new(options.Length == 1 && options[0].Template.Kind == GameActionKind.Pass ? DecisionKind.Forced : DecisionKind.Player,
            game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
    }
    private static IEnumerable<GameActionOption> Options(GameContext game, GameContinuation frame)
    {
        var command = (ShipCommand)frame.Target;
        foreach (int recipient in Recipients(game, frame.Source, command))
            yield return new(new(GameActionKind.ChooseEntity, frame.Source, recipient, Command: command, AbilityId: "token.tantive.transfer"), "Assign token to " + game.Topology.Ships[recipient].CatalogEntry.Definition.DisplayName);
        if (frame.Value != 0) yield return new(new(GameActionKind.Pass, frame.Source, Command: command, AbilityId: "token.tantive.keep"), "Keep token on Tantive IV");
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!Options(game, frame).Any(x => x.Template == action)) throw new ArgumentException("Tantive IV recipient is unavailable.");
        if (action.Kind != GameActionKind.Pass)
        {
            // The original transfer deliberately stores directly on the recipient, without
            // recursively triggering a second Tantive title's replacement choice.
            if (!game.State.Commands[action.Target].TryStoreToken(action.Command)) throw new ArgumentException("Recipient cannot store this command token.");
            if (frame.Value != 0) game.State.Commands[frame.Source].TrySpendToken(action.Command);
        }
        if (game.Flow.Continuations.Count == 0 || game.Flow.Continuations[^1] != frame) throw new InvalidOperationException("Token replacement choices must close in stack order.");
        game.Flow.Continuations.RemoveAt(game.Flow.Continuations.Count - 1); game.Flow.PendingOptions = [];
        game.Flow.Phase = (GamePhase)frame.Stage;
    }

    private static bool CanReceive(GameContext game, int entity, ShipCommand command) => game.IsShip(entity) && !game.Topology.Ships[entity].IsStation && !game.IsDestroyed(entity) && !game.IsReserved(entity) &&
        game.State.Ships[entity].CanHaveCommandTokens && CommandModule.CommandAllowed(game, entity, command);
    private static IEnumerable<int> Recipients(GameContext game, int source, ShipCommand command) => game.Topology.Ships
        .Where(x => x.EntityIndex != source && x.Owner == game.State.Owners[source] && CanReceive(game, x.EntityIndex, command) &&
            game.State.Commands[x.EntityIndex].CanStoreToken(command) && Geometry2D.Distance(game.ShipBase(source), game.ShipBase(x.EntityIndex)) <= TabletopMeasurements.LongRange)
        .Select(x => x.EntityIndex);
}
