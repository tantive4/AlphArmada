using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Additional source command-reveal officers and fleet-support transfers.</summary>
public static class CommandRevealEffects
{
    public const string IsardAction = "reveal.isard";
    private static readonly HashSet<string> Cards = new(StringComparer.Ordinal)
    { "clone-navigation-officer", "leia-organa-officer", "commandant-aresko-officer", "tikkes-officer",
        "passel-argente-officer", "rune-haako-officer", "comms-net", "munitions-resupply", "parts-resupply", "director-isard-officer" };
    public static bool SupportsUpgrade(string id) => Cards.Contains(id);
    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    {
        if (!game.IsShip(source)) yield break;
        if (timing == "after-command-reveal")
        {
            foreach (var option in AfterRevealActions(game, source)) yield return option;
            yield break;
        }
        if (timing != "command-reveal") yield break;
        var ship = game.ShipViews[source]; var activation = game.Flow.ShipActivations[source];
        if ((activation.ActualRevealedCommand ?? activation.RevealedCommand) is not { } command) yield break;
        if (ship.Loadout.HasActive("director-isard-officer"))
            foreach (var enemy in game.SceneShipViews.Where(other => other.OwnerId != ship.OwnerId))
                yield return new(new(GameActionKind.UseAbility, source, enemy.Index, AbilityId: IsardAction),
                    "Look at " + enemy.Definition.DisplayName + " command dials with Director Isard");
        var friends = game.SceneShipViews.Where(other => other.OwnerId == ship.OwnerId && RaidRules.FootprintDistance(ship, other) <= 13.260394f).ToArray();
        if (command != ShipCommand.Engineering && Ready(ship, "clone-navigation-officer"))
            foreach (var friend in friends.Where(friend => CanStore(game, friend.Index, command)))
                yield return Option("reveal.clone-navigation", source, friend.Index, command, "Give matching token to " + friend.Definition.DisplayName);
        if (ship.Loadout.HasActive("leia-organa-officer"))
            foreach (var friend in friends.Where(friend => friend.Index != source && friend.Commands.Count > 0))
                yield return Option("reveal.leia", source, friend.Index, command, "Change " + friend.Definition.DisplayName + " top dial to " + command);
        foreach (var friend in friends.Where(friend => friend.Index != source && Ready(friend, "commandant-aresko-officer")
            && RaidRules.FootprintDistance(ship, friend) <= 8.09628f && CanStore(game, friend.Index, command)))
            yield return Option("reveal.aresko", source, friend.Index, command, "Aresko gains matching token");
        if (ship.Loadout.HasActive("tikkes-officer"))
            foreach (var token in Enum.GetValues<ShipCommand>().Where(token => CanStore(game, source, token)))
                yield return Option("reveal.tikkes", source, source, token, "Tikkes stores revealed command and gains " + token);
        if (ship.Loadout.HasActive("passel-argente-officer"))
            yield return Option("reveal.passel-store", source, source, command, "Store matching token on Passel Argente");
        if (ship.Loadout.HasActive("rune-haako-officer"))
            foreach (var friend in friends.Where(friend => friend.Index != source && Enum.GetValues<ShipCommand>().Any(token => CanStore(game, source, token)
                && (friend.Commands.HasToken(token) || ship.Loadout.Find("rune-haako-officer")!.GetTypedCommandTokenCount(token) > 0))))
                yield return Option("reveal.rune", source, friend.Index, command, "Transfer command tokens from " + friend.Definition.DisplayName + " or Rune Haako");
    }

    private static IEnumerable<GameActionOption> AfterRevealActions(GameContext game, int source)
    {
        var ship = game.ShipViews[source];
        var friends = game.SceneShipViews.Where(other => other.OwnerId == ship.OwnerId && RaidRules.FootprintDistance(ship, other) <= 13.260394f).ToArray();
        if (ship.Loadout.HasActive("comms-net"))
            foreach (var token in ship.Commands.Tokens)
                foreach (var friend in friends.Where(friend => friend.Index != source && CanStore(game, friend.Index, token)))
                    yield return Option($"reveal.comms-net.{(int)token}.{friend.Index}", source, friend.Index, token, "Transfer " + token + " to " + friend.Definition.DisplayName);
        foreach (var cardId in new[] { "munitions-resupply", "parts-resupply" })
        {
            var card = ship.Loadout.Find(cardId); var token = cardId == "munitions-resupply" ? ShipCommand.ConcentrateFire : ShipCommand.Engineering;
            if (card is { Active: true, Exhausted: false, CommandTokens: > 0 } && friends.Any(friend => friend.Index != source && CanStore(game, friend.Index, token)))
                yield return Option("reveal.resupply." + cardId, source, source, token, "Distribute " + token + " tokens");
        }
    }
    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        string id = action.AbilityId; if (!id.StartsWith("reveal.", StringComparison.Ordinal)) return false;
        var ship = game.ShipViews[action.Source]; var activation = game.Flow.ShipActivations[action.Source];
        switch (id)
        {
            // The engine validates and records use of the timing effect. The private
            // observation belongs to SearchKnowledge, not to public flow/labels.
            case IsardAction: return true;
            case "reveal.clone-navigation": ship.Loadout.Find("clone-navigation-officer")!.Exhaust(); TokenAssignments.TryAssign(game, action.Target, action.Command); return true;
            case "reveal.leia": game.State.Commands[action.Target].ReplaceTop(action.Command); return true;
            case "reveal.aresko": game.ShipViews[action.Target].Loadout.Find("commandant-aresko-officer")!.Exhaust(); TokenAssignments.TryAssign(game, action.Target, action.Command); return true;
            case "reveal.tikkes":
                var tikkes = ship.Loadout.Find("tikkes-officer")!; tikkes.ClearTypedCommandTokens(); tikkes.AddTypedCommandToken((activation.ActualRevealedCommand ?? activation.RevealedCommand)!.Value);
                TokenAssignments.TryAssign(game, action.Source, action.Command); return true;
            case "reveal.passel-store":
                var passel = ship.Loadout.Find("passel-argente-officer")!; passel.AddTypedCommandToken((activation.ActualRevealedCommand ?? activation.RevealedCommand)!.Value);
                if (Enum.GetValues<ShipCommand>().All(command => passel.GetTypedCommandTokenCount(command) > 0))
                    Push(game, new("general.reveal.passel", action.Source, Value: window.Value));
                return true;
            case "reveal.rune": Push(game, new("general.reveal.rune", action.Source, action.Target)); return true;
        }
        if (id.StartsWith("reveal.comms-net.", StringComparison.Ordinal))
        {
            if (!ship.Commands.TrySpendToken(action.Command) || !TokenAssignments.TryAssign(game, action.Target, action.Command)) throw new InvalidOperationException("Comms Net transfer became illegal.");
            return true;
        }
        if (id.StartsWith("reveal.resupply.", StringComparison.Ordinal))
        { Push(game, new("general.reveal.resupply", action.Source, Text: id["reveal.resupply.".Length..])); return true; }
        return false;
    }
    public static bool Handles(string rule) => rule.StartsWith("general.reveal.", StringComparison.Ordinal);
    public static GameDecision GetDecision(GameContext game, GameContinuation frame)
    {
        var options = Options(game, frame).ToArray(); bool forced = options.Length == 1 && options[0].Template.Kind == GameActionKind.Pass;
        return new(forced ? DecisionKind.Forced : DecisionKind.Player, forced ? 0 : game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
    }
    private static IEnumerable<GameActionOption> Options(GameContext game, GameContinuation frame)
    {
        var ship = game.ShipViews[frame.Source];
        if (frame.Rule == "general.reveal.passel")
        {
            foreach (var command in Enum.GetValues<ShipCommand>()) yield return Option("reveal.passel-dial", frame.Source, -1, command, "Gain additional " + command + " dial");
        }
        else if (frame.Rule == "general.reveal.rune")
        {
            var card = ship.Loadout.Find("rune-haako-officer")!;
            foreach (var command in Enum.GetValues<ShipCommand>().Where(command => CanStore(game, frame.Source, command)))
            {
                if (game.State.Commands[frame.Target].HasToken(command)) yield return Option("reveal.rune-ship", frame.Source, frame.Target, command, "Transfer " + command + " from ship");
                if (card.GetTypedCommandTokenCount(command) > 0) yield return Option("reveal.rune-card", frame.Source, frame.Target, command, "Transfer " + command + " from Rune Haako");
            }
        }
        else if (frame.Rule == "general.reveal.resupply")
        {
            var card = ship.Loadout.Find(frame.Text)!; var token = frame.Text == "munitions-resupply" ? ShipCommand.ConcentrateFire : ShipCommand.Engineering;
            var recipients = game.SceneShipViews.Where(other => other.Index != frame.Source && other.OwnerId == ship.OwnerId
                && RaidRules.FootprintDistance(ship, other) <= 13.260394f && CanStore(game, other.Index, token)).ToArray();
            if (frame.Stage == 0)
            {
                int maximum = Math.Min(card.CommandTokens, recipients.Length);
                for (int quantity = 1; quantity <= maximum; quantity++)
                    yield return new(new(GameActionKind.UseAbility, frame.Source, Amount: quantity, AbilityId: "reveal.resupply-quantity"), "Distribute " + quantity + " tokens");
            }
            else
                foreach (var recipient in recipients.Where(other => !(frame.Indices ?? []).Contains(other.Index)))
                    yield return Option("reveal.resupply-recipient", frame.Source, recipient.Index, token, "Assign token to " + recipient.Definition.DisplayName);
        }
        else throw new NotSupportedException("Unknown command-reveal continuation " + frame.Rule);
        yield return new(new(GameActionKind.Pass, frame.Source, AbilityId: "reveal.finish"), "Finish or cancel");
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (!Options(game, frame).Any(option => GameSession.Matches(option, action))) throw new ArgumentException("Illegal command-reveal choice.");
        if (action.Kind == GameActionKind.Pass) { EngineEffects.ReturnFromChoice(game); return; }
        if (frame.Rule == "general.reveal.passel")
        {
            EngineEffects.ReturnFromChoice(game);
            ActivationAbilityEffects.EnqueueDial(game, action.Command, "Passel Argente"); return;
        }
        if (frame.Rule == "general.reveal.rune")
        {
            bool paid = action.AbilityId == "reveal.rune-card" ? game.ShipViews[frame.Source].Loadout.Find("rune-haako-officer")!.TrySpendTypedCommandToken(action.Command)
                : game.State.Commands[frame.Target].TrySpendToken(action.Command);
            if (!paid || !TokenAssignments.TryAssign(game, frame.Source, action.Command)) throw new InvalidOperationException("Rune Haako transfer failed.");
            return;
        }
        if (action.AbilityId == "reveal.resupply-quantity")
        { game.Flow.Continuations[^1] = frame with { Stage = 1, Value = action.Amount, Indices = [] }; return; }
        var selected = (frame.Indices ?? []).Append(action.Target).ToArray();
        if (selected.Length < frame.Value) { game.Flow.Continuations[^1] = frame with { Indices = selected }; return; }
        var card = game.ShipViews[frame.Source].Loadout.Find(frame.Text)!;
        if (!card.Exhaust()) throw new InvalidOperationException("Resupply card unavailable.");
        foreach (int recipient in selected)
            if (!card.TrySpendStoredCommandToken() || !TokenAssignments.TryAssign(game, recipient, action.Command)) throw new InvalidOperationException("Resupply transfer failed.");
        EngineEffects.ReturnFromChoice(game);
    }
    private static void Push(GameContext game, GameContinuation frame) => EngineEffects.PushChoice(game, frame, Options(game, frame));
    private static bool Ready(ShipView ship, string id) => ship.Loadout.Find(id) is { Active: true, Exhausted: false };
    private static bool CanStore(GameContext game, int ship, ShipCommand command) => game.State.Ships[ship].CanHaveCommandTokens
        && CommandModule.CommandAllowed(game, ship, command) && TokenAssignments.CanAssign(game, ship, command);
    private static GameActionOption Option(string id, int source, int target, ShipCommand command, string label)
        => new(new(GameActionKind.UseAbility, source, target, Command: command, AbilityId: id), label);
}
