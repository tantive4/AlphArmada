using Armada.Rules;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public static partial class ActivationAbilityEffects
{
    /// <summary>Gained dials may become tokens before raid cleanup and after-reveal effects.
    /// Choosing which dial to process first is a player decision; no capacity heuristic is used.</summary>
    private static GameDecision? RevealBankingDecision(GameContext game)
    {
        int ship = game.Flow.ActiveShip; var state = Active(game); var options = new List<GameActionOption>();
        if (!state.NativeBankingDecided && state.RevealedDialAvailable && !state.NativeDialConsumedBeforePrelude && state.RevealedCommand is { } native)
        {
            Add(options, new(GameActionKind.UseAbility, ship, Command: native, AbilityId: "activation.reveal-keep-native"), "Keep the ship's revealed dial for command resolution");
            if (TokenAssignments.CanAssign(game, ship, native))
                Add(options, new(GameActionKind.BankCommand, ship, Command: native, AbilityId: "activation.reveal-bank-native"), "Convert the ship's revealed dial to a command token");
        }
        for (int index = 0; index < state.AdditionalCommandDials.Count; index++)
        {
            var dial = state.AdditionalCommandDials[index]; if (dial.BankingDecided) continue;
            Add(options, new(GameActionKind.UseAbility, ship, Index: index, Command: dial.Command, AbilityId: "activation.reveal-keep-extra"), "Keep " + dial.Source + " dial for command resolution");
            if (TokenAssignments.CanAssign(game, ship, dial.Command))
                Add(options, new(GameActionKind.BankCommand, ship, Index: index, Command: dial.Command, AbilityId: "activation.reveal-bank-extra"), "Convert " + dial.Source + " dial to a command token");
        }
        return options.Count == 0 ? null : Decision(game, options);
    }

    internal static bool NeedsRevealRaidDecision(GameContext game)
    {
        int ship = game.Flow.ActiveShip; var state = Active(game);
        return state.RevealPreludeActive && state.CommandRevealStep == 3 && state.RevealWhenWindowOpened &&
            state.RevealBankingClosed && !state.RevealAfterWindowOpened && !state.RaidInterferenceResolved &&
            game.State.Commands[ship].RaidTokenCount > 0;
    }

    private static GameDecision? RevealDialOrderDecision(GameContext game)
    {
        int ship = game.Flow.ActiveShip; var state = Active(game); var options = new List<GameActionOption>();
        bool native = state.RevealedDialAvailable && !state.NativeDialConsumedBeforePrelude;
        for (int index = 0; index < state.AdditionalCommandDials.Count; index++)
        {
            var dial = state.AdditionalCommandDials[index];
            if (!CommandModule.CommandAllowed(game, ship, dial.Command)) continue;
            Add(options, new(GameActionKind.UseAbility, ship, Index: index, Command: dial.Command,
                AbilityId: dial.Source == "Grand Admiral Thrawn" ? "activation.thrawn-first" : "activation.additional-first"),
                "Resolve " + dial.Source + " " + dial.Command + " dial first");
        }
        if (options.Count == 0) return null;
        if (native) options.Insert(0, new(new(GameActionKind.UseAbility, ship, Command: state.RevealedCommand!.Value,
            AbilityId: "activation.native-first"), "Resolve the ship's revealed dial first"));
        return new(options.Count == 1 ? DecisionKind.Forced : DecisionKind.Player, options.Count == 1 ? 0 : game.State.Owners[ship],
            game.Flow.Phase, options.AsReadOnly());
    }

    private static GameDecision? NextAdditionalDialDecision(GameContext game)
    {
        int ship = game.Flow.ActiveShip; var state = Active(game); var options = new List<GameActionOption>();
        for (int index = 0; index < state.AdditionalCommandDials.Count; index++)
        {
            var dial = state.AdditionalCommandDials[index];
            if (CommandModule.CommandAllowed(game, ship, dial.Command))
                Add(options, new(GameActionKind.UseAbility, ship, Index: index, Command: dial.Command, AbilityId: "activation.next-dial"),
                    "Resolve " + dial.Source + " " + dial.Command + " dial next");
        }
        return options.Count == 0 ? null : new(options.Count == 1 ? DecisionKind.Forced : DecisionKind.Player,
            options.Count == 1 ? 0 : game.State.Owners[ship], game.Flow.Phase, options.AsReadOnly());
    }

    private static void SelectAdditionalDial(GameContext game, int index, bool retainNative)
    {
        var state = Active(game); var selected = state.AdditionalCommandDials[index];
        state.AdditionalCommandDials.RemoveAt(index);
        if (retainNative)
            state.AdditionalCommandDials.Insert(0, new(state.RevealedCommand!.Value, "Revealed Command", BankingDecided: true));
        state.RevealedCommand = selected.Command; state.RevealedDialAvailable = true;
        state.AdditionalDialActive = selected.Source != "Revealed Command";
        state.NativeDialConsumedBeforePrelude = false; state.CommandRevealStep = 5;
        state.CommandResolutionCompleted = false; state.CommandCompletionPending = false;
        game.Flow.Phase = GamePhase.ResolveCommand;
    }
}
