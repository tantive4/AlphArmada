using System.Globalization;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Source command-token assignment, including Tantive IV's replacement recipient choice.</summary>
public static class TokenAssignments
{
    private const string PendingPrefix = "token.pending.";
    private const string ReplacementPrefix = "token.replace.pending.";
    public const string ReplacementRule = "token.replace";
    public static bool SupportsUpgrade(string id) => id == "tantive-iv-title";

    public static bool CanAssign(GameContext game, int entity, ShipCommand command) =>
        CanReceive(game, entity, command);

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
        return TryAssignDirect(game, entity, command);
    }

    private static bool TryAssignDirect(GameContext game, int entity, ShipCommand command)
    {
        if (!CanReceive(game, entity, command)) return false;
        var stack = game.State.Commands[entity];
        if (stack.TryStoreToken(command)) return true;
        // Tokens have no physical identity: replacing an already held unique
        // token by another of that type yields exactly the same game state.
        if (!stack.AllowsDuplicateTokens && stack.HasToken(command)) return true;
        long serial = game.Flow.Flags.GetValueOrDefault("token.next-id") + 1;
        game.Flow.Flags["token.next-id"] = serial;
        game.Flow.Flags[ReplacementPrefix + serial.ToString(CultureInfo.InvariantCulture)] = ((long)(uint)entity << 4) | (uint)command;
        return true;
    }

    public static bool TryBeginNext(GameContext game)
    {
        // A mandatory capacity replacement interrupts an unrelated custom choice
        // immediately, then resumes that same frame. Never interrupt an active
        // token assignment with another token assignment. Tantive's optional
        // replacement remains deferred while another custom effect owns its stack.
        bool customChoice = game.Flow.Phase == GamePhase.AbilityChoice &&
            game.Flow.Continuations.LastOrDefault() is { Rule: not "effects.window" };
        if (customChoice && Handles(game.Flow.Continuations[^1].Rule)) return false;
        while (true)
        {
            string? selected = null; long earliest = long.MaxValue; bool replacement = false;
            foreach (var entry in game.Flow.Flags)
            {
                bool replace = entry.Key.StartsWith(ReplacementPrefix, StringComparison.Ordinal);
                if (customChoice && !replace) continue;
                int prefix = replace ? ReplacementPrefix.Length : entry.Key.StartsWith(PendingPrefix, StringComparison.Ordinal) ? PendingPrefix.Length : -1;
                if (prefix >= 0 && long.TryParse(entry.Key.AsSpan(prefix), NumberStyles.None, CultureInfo.InvariantCulture, out long serial) && serial < earliest)
                { selected = entry.Key; earliest = serial; replacement = replace; }
            }
            if (selected is null) return false;
            long payload = game.Flow.Flags[selected]; game.Flow.Flags.Remove(selected);
            if (replacement)
            {
                int recipient = (int)(payload >> 4); var incoming = (ShipCommand)(payload & 15);
                if (!CanReceive(game, recipient, incoming)) continue;
                EngineEffects.PushChoice(game, new(ReplacementRule, recipient, (int)incoming, (int)game.Flow.Phase), []);
                return true;
            }
            int source = (int)(payload >> 4); var command = (ShipCommand)((payload >> 1) & 7); bool stored = (payload & 1) != 0;
            if (!game.IsShip(source) || game.IsDestroyed(source) || game.IsReserved(source) || !game.HasUpgrade(source, "tantive-iv-title")) continue;
            var frame = new GameContinuation("token.tantive", source, (int)command, (int)game.Flow.Phase, stored ? 1 : 0);
            EngineEffects.PushChoice(game, frame, []); return true;
        }
    }

    public static bool Handles(string rule) => rule is "token.tantive" or ReplacementRule;
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = Options(game, frame).ToArray();
        return new(options.Length == 1 && options[0].Template.Kind == GameActionKind.Pass ? DecisionKind.Forced : DecisionKind.Player,
            game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
    }
    private static IEnumerable<GameActionOption> Options(GameContext game, GameContinuation frame)
    {
        if (frame.Rule == ReplacementRule)
        {
            foreach (var discard in game.State.Commands[frame.Source].Tokens)
                yield return new(new(GameActionKind.ChooseCommand, frame.Source, Command: discard, Index: frame.Target,
                    AbilityId: "token.replace.discard"), "Discard existing " + discard + " token and keep the gained token");
            yield return new(new(GameActionKind.Pass, frame.Source, Command: (ShipCommand)frame.Target,
                AbilityId: "token.replace.discard-incoming"), "Discard the gained token");
            yield break;
        }
        var command = (ShipCommand)frame.Target;
        foreach (int recipient in Recipients(game, frame.Source, command))
            yield return new(new(GameActionKind.ChooseEntity, frame.Source, recipient, Command: command, AbilityId: "token.tantive.transfer"), "Assign token to " + game.Topology.Ships[recipient].CatalogEntry.Definition.DisplayName);
        yield return new(new(GameActionKind.Pass, frame.Source, Command: command, AbilityId: "token.tantive.keep"), "Keep token on Tantive IV");
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!Options(game, frame).Any(x => x.Template == action)) throw new ArgumentException("Command-token assignment choice is unavailable.");
        if (frame.Rule == ReplacementRule)
        {
            if (action.Kind != GameActionKind.Pass &&
                (!game.State.Commands[frame.Source].TrySpendToken(action.Command) ||
                 !game.State.Commands[frame.Source].TryStoreToken((ShipCommand)frame.Target)))
                throw new InvalidOperationException("Command-token replacement became illegal.");
        }
        else if (action.Kind != GameActionKind.Pass)
        {
            // The original transfer deliberately stores directly on the recipient, without
            // recursively triggering a second Tantive title's replacement choice.
            if (!TryAssignDirect(game, action.Target, action.Command)) throw new ArgumentException("Recipient cannot store this command token.");
            if (frame.Value != 0) game.State.Commands[frame.Source].TrySpendToken(action.Command);
        }
        else if (frame.Value == 0 && !TryAssignDirect(game, frame.Source, action.Command))
            throw new InvalidOperationException("Tantive IV cannot keep its gained token.");
        if (game.Flow.Continuations.Count == 0 || game.Flow.Continuations[^1] != frame) throw new InvalidOperationException("Token replacement choices must close in stack order.");
        game.Flow.Continuations.RemoveAt(game.Flow.Continuations.Count - 1); game.Flow.PendingOptions = [];
        game.Flow.Phase = (GamePhase)frame.Stage;
    }

    private static bool CanReceive(GameContext game, int entity, ShipCommand command) => game.IsShip(entity) && !game.Topology.Ships[entity].IsStation && !game.IsDestroyed(entity) && !game.IsReserved(entity) &&
        game.State.Ships[entity].CanHaveCommandTokens && CommandModule.CommandAllowed(game, entity, command);
    private static IEnumerable<int> Recipients(GameContext game, int source, ShipCommand command) => game.Topology.Ships
        .Where(x => x.EntityIndex != source && x.Owner == game.State.Owners[source] && CanReceive(game, x.EntityIndex, command) &&
            Geometry2D.Distance(game.ShipBase(source), game.ShipBase(x.EntityIndex)) <= TabletopMeasurements.LongRange)
        .Select(x => x.EntityIndex);
}
