using Armada.Core;
using Armada.Data;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Additional command dials and the choices that complete an individual command.</summary>
public static partial class ActivationAbilityEffects
{
    public static bool SupportsUpgrade(string id) => id is "bail-organa-commander" or "commander-vanto-officer" or
        "san-hill-officer" or "adar-tallon-officer" or "resolute-title" or "leia-organa-commander" or
        "clone-commander-wolffe-officer" or "captain-rex-officer" or "flight-commander-officer" or "commander-beck-officer" or "chimaera-title";

    public static GameDecision? GetPlanningDecision(GameContext game)
    {
        int owner = game.State.Owners[game.Flow.PlanningShip]; string key = $"round.chimaera.{owner}";
        if (game.Flow.Flags.GetValueOrDefault(key) == game.State.Round) return null;
        var carrier = game.Topology.Ships.FirstOrDefault(x => x.Owner == owner && !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex) && game.HasUpgrade(x.EntityIndex, "chimaera-title"));
        if (carrier is null) { game.Flow.Flags[key] = game.State.Round; return null; }
        var entry = game.Topology.Upgrades.FirstOrDefault(x => game.State.Registers[game.Topology.Registers.UpgradeOwner(x.UpgradeIndex)] == carrier.EntityIndex + 1 &&
            game.State.Upgrades[x.UpgradeIndex].Active && game.State.Upgrades[x.UpgradeIndex].EffectiveDefinition.slot == "fleet-command");
        if (entry is null || !FleetCommandRules.IsSupported(game.State.Upgrades[entry.UpgradeIndex].EffectiveDefinitionId))
        { game.Flow.Flags[key] = game.State.Round; return null; }
        var current = game.State.Upgrades[entry.UpgradeIndex];
        var unavailable = game.Topology.Upgrades.Where(x => x.UpgradeIndex != entry.UpgradeIndex &&
            game.State.Owners[(int)game.State.Registers[game.Topology.Registers.UpgradeOwner(x.UpgradeIndex)] - 1] == owner)
            .Select(x => game.State.Upgrades[x.UpgradeIndex].EffectiveDefinition).Where(x => x.slot == "fleet-command").Select(x => x.id).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var replacements = UpgradeCatalog.Cards.Where(x => x.slot == "fleet-command" && FleetCommandRules.IsSupported(x.id) &&
            UpgradeCatalog.IsFactionLegal(x, carrier.CatalogEntry.Definition.Faction) && x.id != current.EffectiveDefinitionId && !unavailable.Contains(x.id))
            .OrderBy(x => x.points).ThenBy(x => x.name, StringComparer.OrdinalIgnoreCase).ToArray();
        if (replacements.Length == 0) { game.Flow.Flags[key] = game.State.Round; return null; }
        var options = new List<GameActionOption>
        { new(new(GameActionKind.Pass, carrier.EntityIndex, Index: entry.UpgradeIndex, AbilityId: "command.chimaera.keep"), "Keep current fleet-command card") };
        foreach (var replacement in replacements)
            options.Add(new(new(GameActionKind.UseAbility, carrier.EntityIndex, Index: entry.UpgradeIndex, AbilityId: "command.chimaera.replace." + replacement.id), "Equip " + replacement.name));
        return new(DecisionKind.Player, owner, GamePhase.CommandPlanning, options.AsReadOnly());
    }

    public static bool TryApplyPlanningChoice(GameContext game, GameAction action)
    {
        if (!action.AbilityId.StartsWith("command.chimaera.", StringComparison.Ordinal)) return false;
        var decision = GetPlanningDecision(game);
        if (decision is null || !decision.Actions.Any(x => x.Template == action)) throw new ArgumentException("Chimaera replacement is unavailable.");
        if (action.AbilityId.StartsWith("command.chimaera.replace.", StringComparison.Ordinal) &&
            !game.State.Upgrades[action.Index].TryReplaceEffectiveDefinition(action.AbilityId["command.chimaera.replace.".Length..])) throw new ArgumentException("Illegal Chimaera replacement.");
        game.Flow.Flags[$"round.chimaera.{game.State.Owners[action.Source]}"] = game.State.Round; return true;
    }

    public static bool CanUseBeck(GameContext game, ShipCommand command, bool allowResolvedDial)
    {
        int ship = game.Flow.ActiveShip;
        if (!game.IsShip(ship) || !CanResolveCommand(game, ship)) return false;
        var state = Active(game); int bit = 1 << (int)command;
        return game.UpgradesFor(ship).Any(x => x.Active && x.Definition.id == "commander-beck-officer" && x.UsesThisActivation < 2) &&
            game.State.Commands[ship].CanGainRaidToken(command) && (state.BeckCommands & bit) == 0 &&
            ((state.ResolvedCommands & bit) == 0 || (allowResolvedDial && (state.ResolvedDialCommands & bit) != 0));
    }

    public static void CommitBeckCommand(GameContext game, ShipCommand command)
    {
        if (!CanUseBeck(game, command, allowResolvedDial: true)) throw new ArgumentException("Commander Beck is unavailable for this command.");
        var card = game.UpgradesFor(game.Flow.ActiveShip).First(x => x.Active && x.Definition.id == "commander-beck-officer");
        if (!card.TryUseThisActivation(2) || !game.State.Commands[game.Flow.ActiveShip].TryGainRaidToken(command)) throw new ArgumentException("Commander Beck payment became illegal.");
        Active(game).BeckCommands |= 1 << (int)command;
    }

    public static bool TryWindowApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (action.AbilityId != "remaining.command-reveal.squadron-keyword.clone-commander-wolffe-officer") return false;
        int ship = action.Source;
        var card = ReadyCard(game, ship, "clone-commander-wolffe-officer");
        if (card is null || !card.TryUseThisActivation() || !card.Exhaust()) throw new ArgumentException("Clone Commander Wolffe is unavailable.");
        foreach (string key in game.Flow.Flags.Keys.Where(x => x.StartsWith($"wolffe.{ship}.", StringComparison.Ordinal)).ToArray()) game.Flow.Flags.Remove(key);
        game.Flow.Flags[$"wolffe.{ship}.enabled"] = 1; return true;
    }

    public static bool ApplyCommandedKeywords(GameContext game, int entity)
    {
        int ship = game.Flow.ActiveShip;
        if (!game.IsShip(ship) || !game.Flow.Flags.ContainsKey($"wolffe.{ship}.enabled")) return false;
        var definition = game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition;
        if (CommandModule.KeywordValue(definition.Keywords, "ADEPT") > 0 || game.Flow.Flags.ContainsKey($"wolffe.{ship}.{entity}") ||
            game.Flow.Flags.Keys.Count(x => x.StartsWith($"wolffe.{ship}.", StringComparison.Ordinal) && !x.EndsWith(".enabled", StringComparison.Ordinal)) >= 3) return false;
        game.Flow.Flags[$"wolffe.{ship}.{entity}"] = 1;
        game.State.Squadrons[game.SquadronIndex(entity)].GrantTemporaryAssault(); return true;
    }

    public static void BeginSquadronCommand(GameContext game)
    {
        var state = Active(game); state.SquadronCommandPreludeStep = 0;
        foreach (string key in game.Flow.Flags.Keys.Where(x => x.StartsWith($"wolffe.{game.Flow.ActiveShip}.", StringComparison.Ordinal) && !x.EndsWith(".enabled", StringComparison.Ordinal)).ToArray()) game.Flow.Flags.Remove(key);
    }

    public static GameDecision? GetSquadronCommandDecision(GameContext game)
    {
        int ship = game.Flow.ActiveShip; var state = Active(game);
        if (state.SquadronCommandPreludeStep == 0)
        {
            var rex = game.UpgradesFor(ship).FirstOrDefault(x => x.Active && x.Definition.id == "captain-rex-officer" && x.UsesThisActivation == 0);
            var options = new List<GameActionOption>();
            if (rex is not null)
                foreach (var enemy in game.Topology.Ships.Where(x => x.Owner != game.State.Owners[ship] && !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex) &&
                    Geometry2D.Distance(game.ShipBase(ship), game.ShipBase(x.EntityIndex)) <= TabletopMeasurements.CloseRange))
                    foreach (var command in Enum.GetValues<ShipCommand>().Where(x => game.State.Commands[enemy.EntityIndex].CanGainRaidToken(x)))
                        Add(options, new(GameActionKind.ChooseCommand, ship, enemy.EntityIndex, Command: command, AbilityId: "activation.rex-raid"), "Give enemy " + command + " raid with Captain Rex");
            if (options.Count > 0)
            {
                Add(options, new(GameActionKind.Pass, ship, AbilityId: "activation.rex-skip"), "Skip Captain Rex");
                return Decision(game, options);
            }
            state.SquadronCommandPreludeStep = 1;
        }
        if (state.SquadronCommandPreludeStep == 1)
        {
            if (game.HasUpgrade(ship, "flight-commander-officer") && !state.ManeuverComplete)
                return Decision(game,
                [new(new(GameActionKind.UseAbility, ship, AbilityId: "activation.flight-now"), "Resolve Squadron command now"),
                 new(new(GameActionKind.UseAbility, ship, AbilityId: "activation.flight-defer"), "Hold Squadron command until after a maneuver")]);
            state.SquadronCommandPreludeStep = 2;
        }
        if (state.SquadronCommandPreludeStep == 2)
        {
            state.SquadronCommandPreludeStep = 3;
            if (MapAbilityEffects.BeginReservedSquadronRelease(game, ship, "invisible-hand-title", GamePhase.SquadronCommand)) return null;
        }
        if (state.SquadronCommandPreludeStep == 3)
        {
            state.SquadronCommandPreludeStep = 4;
            MovementAbilityEffects.BeginAhsokaSnipeChoices(game, ship);
        }
        return null;
    }

    public static void EnqueueDial(GameContext game, ShipCommand command, string source) =>
        game.Flow.ShipActivations[game.Flow.ActiveShip].AdditionalCommandDials.Add(new(command, source));

    public static void BeginReveal(GameContext game, bool hasNativeReveal = true)
    {
        var state = Active(game); state.CommandRevealStep = 0; state.BailDialSelection = 0;
        state.ThrawnDialQueued = false; state.AdditionalDialActive = false; state.CurrentCommandResolved = false;
        state.RaidInterferenceResolved = false;
        state.NativeDialConsumedBeforePrelude = false;
        state.RapidLaunchReleaseChecked = false;
        state.ActualRevealedCommand = hasNativeReveal ? state.RevealedCommand : null;
        state.RevealPreludeActive = true; state.RevealWhenWindowOpened = false;
        state.RevealAfterWindowOpened = false; state.RevealBankingClosed = false; state.NativeBankingDecided = false;
        if (!hasNativeReveal)
        {
            state.CommandRevealStep = 2; state.RevealedCommand = null; state.RevealedDialAvailable = false;
            state.NativeBankingDecided = true; state.NativeDialConsumedBeforePrelude = true;
        }
    }

    public static GameDecision? GetCommandDecision(GameContext game)
    {
        using var scope = new GameScope(game);
        int ship = game.Flow.ActiveShip; var state = Active(game);
        if (state.CommandCompletionPending) return CompletionDecision(game);
        while (state.CommandRevealStep < 5)
        {
            var options = new List<GameActionOption>();
            if (state.CommandRevealStep == 0)
            {
                var bail = FindFleetCard(game, ship, "bail-organa-commander");
                if (bail is not null)
                {
                    bool navigate = bail.GetTypedCommandTokenCount(ShipCommand.Navigate) > 0;
                    bool engineering = bail.GetTypedCommandTokenCount(ShipCommand.Engineering) > 0;
                    if (navigate) Add(options, new(GameActionKind.UseAbility, ship, Command: ShipCommand.Navigate, AbilityId: "activation.bail-use-navigate"), "Gain Bail's Navigate dial");
                    if (engineering) Add(options, new(GameActionKind.UseAbility, ship, Command: ShipCommand.Engineering, AbilityId: "activation.bail-use-engineering"), "Gain Bail's Engineering dial");
                    if (navigate && engineering) Add(options, new(GameActionKind.UseAbility, ship, AbilityId: "activation.bail-use-both"), "Gain both Bail dials");
                }
                if (options.Count > 0)
                {
                    Add(options, new(GameActionKind.Pass, ship, AbilityId: "activation.bail-finish"), "Skip Bail Organa");
                    return Decision(game, options);
                }
                state.CommandRevealStep++; continue;
            }
            if (state.CommandRevealStep == 1)
            {
                if (game.HasUpgrade(ship, "admiral-tarkin-commander") && game.Flow.SourceRules.AdmiralTarkinDials.TryGetValue(game.State.Owners[ship], out var command))
                {
                    Add(options, new(GameActionKind.UseAbility, ship, Command: command, AbilityId: "activation.tarkin-dial"), "Use Admiral Tarkin's gained dial");
                    if (CanGain(game, ship, command)) Add(options, new(GameActionKind.BankCommand, ship, Command: command, AbilityId: "activation.tarkin-token"), "Convert Admiral Tarkin's gained dial to a token");
                    Add(options, new(GameActionKind.Pass, ship, AbilityId: "activation.tarkin-save"), "Keep Admiral Tarkin's gained dial unused");
                    return Decision(game, options);
                }
                state.CommandRevealStep++; continue;
            }
            if (state.CommandRevealStep == 2)
            {
                if (!state.ThrawnDialQueued)
                {
                    state.ThrawnDialQueued = true;
                    if (game.Flow.SourceRules.ThrawnCommands.TryGetValue(game.State.Owners[ship], out var thrawn)) EnqueueDial(game, thrawn, "Grand Admiral Thrawn");
                }
                state.CommandRevealStep++; continue;
            }
            if (state.CommandRevealStep == 3)
            {
                if (state.RevealPreludeActive)
                {
                    if (!state.RevealWhenWindowOpened)
                    {
                        state.RevealWhenWindowOpened = true;
                        if (state.ActualRevealedCommand.HasValue && EngineEffects.BeginWindow(game, "command-reveal", ship, resumePhase: GamePhase.ResolveCommand))
                            return new(DecisionKind.Forced, 0, game.Flow.Phase, []);
                    }
                    if (!state.RevealBankingClosed)
                    {
                        if (RevealBankingDecision(game) is { } banking) return banking;
                        state.RevealBankingClosed = true;
                    }
                    if (NeedsRevealRaidDecision(game)) return null;
                    if (!state.RevealAfterWindowOpened)
                    {
                        state.RevealAfterWindowOpened = true;
                        if (state.ActualRevealedCommand.HasValue && EngineEffects.BeginWindow(game, "after-command-reveal", ship, resumePhase: GamePhase.ResolveCommand))
                            return new(DecisionKind.Forced, 0, game.Flow.Phase, []);
                    }
                }
                if (!state.RapidLaunchReleaseChecked)
                {
                    state.RapidLaunchReleaseChecked = true;
                    if (MapAbilityEffects.BeginReservedSquadronRelease(game, ship, "rapid-launch-bays", GamePhase.ResolveCommand))
                        return new(DecisionKind.Forced, 0, game.Flow.Phase, []);
                }
                state.CommandRevealStep++; continue;
            }
            if (RevealDialOrderDecision(game) is { } order) return order;
            state.CommandRevealStep++;
        }
        state.RevealPreludeActive = false;
        if (state.NativeDialConsumedBeforePrelude)
        {
            state.NativeDialConsumedBeforePrelude = false;
            CompleteCommand(game);
            if (state.CommandCompletionPending) return CompletionDecision(game);
        }
        return null;
    }

    public static bool TryApplyCommandChoice(GameContext game, GameAction action)
    {
        if (!action.AbilityId.StartsWith("activation.", StringComparison.Ordinal)) return false;
        var decision = game.Flow.Phase == GamePhase.SquadronCommand ? GetSquadronCommandDecision(game) : GetCommandDecision(game);
        if (decision is null) throw new ArgumentException("No activation command choice is pending.");
        if (!decision.Actions.Any(x => x.Template == action)) throw new ArgumentException("Illegal activation command choice.");
        var state = Active(game); int ship = game.Flow.ActiveShip, owner = game.State.Owners[ship];
        switch (action.AbilityId)
        {
            case "activation.reveal-keep-native": state.NativeBankingDecided = true; return true;
            case "activation.reveal-bank-native":
                state.NativeBankingDecided = true; state.RevealedDialAvailable = false; state.NativeDialConsumedBeforePrelude = true;
                if (!TokenAssignments.TryAssign(game, ship, action.Command)) throw new InvalidOperationException("The revealed command token cannot be gained.");
                return true;
            case "activation.reveal-keep-extra":
                state.AdditionalCommandDials[action.Index] = state.AdditionalCommandDials[action.Index] with { BankingDecided = true }; return true;
            case "activation.reveal-bank-extra":
                state.AdditionalCommandDials.RemoveAt(action.Index);
                if (!TokenAssignments.TryAssign(game, ship, action.Command)) throw new InvalidOperationException("The gained command token cannot be gained.");
                return true;
            case "activation.rex-raid":
                game.UpgradesFor(ship).First(x => x.Active && x.Definition.id == "captain-rex-officer").TryUseThisActivation();
                game.State.Commands[action.Target].TryGainRaidToken(action.Command); state.SquadronCommandPreludeStep = 1;
                GetSquadronCommandDecision(game); return true;
            case "activation.rex-skip": state.SquadronCommandPreludeStep = 1; GetSquadronCommandDecision(game); return true;
            case "activation.flight-now": state.SquadronCommandPreludeStep = 2; GetSquadronCommandDecision(game); return true;
            case "activation.flight-defer":
                state.DeferredSquadronActivations = state.CommandedSquadronsRemaining;
                state.DeferredSquadronSpendMode = game.Flow.EffectiveCommandSpendMode;
                state.CommandedSquadronsRemaining = 0; state.SquadronCommandPreludeStep = 4;
                state.CurrentCommandResolved = false; state.CurrentResolvedCommand = null;
                CompleteCommand(game); return true;
            case "activation.bail-use-navigate":
            case "activation.bail-use-engineering":
            case "activation.bail-use-both":
                var bailCard = FindFleetCard(game, ship, "bail-organa-commander")!;
                foreach (var selected in action.AbilityId == "activation.bail-use-both"
                    ? new[] { ShipCommand.Navigate, ShipCommand.Engineering } : new[] { action.Command })
                {
                    if (!bailCard.TrySpendTypedCommandToken(selected)) throw new InvalidOperationException("Bail's selected token is unavailable.");
                    EnqueueDial(game, selected, "Bail Organa");
                }
                state.CommandRevealStep = 1; return true;
            case "activation.bail-finish":
                var bail = FindFleetCard(game, ship, "bail-organa-commander");
                foreach (var command in Enum.GetValues<ShipCommand>())
                    if ((state.BailDialSelection & (1 << (int)command)) != 0 && bail!.TrySpendTypedCommandToken(command)) EnqueueDial(game, command, "Bail Organa");
                state.CommandRevealStep = 1; return true;
            case "activation.tarkin-dial":
                game.Flow.SourceRules.AdmiralTarkinDials.Remove(owner); EnqueueDial(game, action.Command, "Admiral Tarkin"); state.CommandRevealStep = 2; return true;
            case "activation.tarkin-token":
                game.Flow.SourceRules.AdmiralTarkinDials.Remove(owner); TokenAssignments.TryAssign(game, ship, action.Command); state.CommandRevealStep = 2; return true;
            case "activation.tarkin-save": state.CommandRevealStep = 2; return true;
            case "activation.native-first": state.CommandRevealStep = 5; return true;
            case "activation.thrawn-first":
            case "activation.additional-first":
                SelectAdditionalDial(game, action.Index, retainNative: state.RevealedDialAvailable && !state.NativeDialConsumedBeforePrelude); return true;
            case "activation.next-dial": SelectAdditionalDial(game, action.Index, retainNative: false); return true;
            case "activation.adar-ready":
                game.UpgradesFor(ship).First(x => x.Active && x.Definition.id == "adar-tallon-officer").Exhaust();
                game.State.Squadrons[game.SquadronIndex(action.Target)].ReadyActivation();
                game.State.Activated[action.Target] = false; state.CommandCompletionStep = 1; break;
            case "activation.resolute-token":
                var resolute = game.UpgradesFor(ship).First(x => x.Active && x.Definition.id == "resolute-title");
                resolute.TrySpendTypedCommandToken(action.Command); TokenAssignments.TryAssign(game, ship, action.Command); state.CommandCompletionStep = 2; break;
            case "activation.vanto-token":
                game.UpgradesFor(ship).First(x => x.Active && x.Definition.id == "commander-vanto-officer").Exhaust();
                TokenAssignments.TryAssign(game, ship, action.Command); state.VantoOffered = true; state.CommandCompletionStep = 3; break;
            case "activation.completion-skip":
                if (state.CommandCompletionStep == 2) state.VantoOffered = true;
                state.CommandCompletionStep++; break;
            default: throw new NotSupportedException("Unknown activation command choice " + action.AbilityId);
        }
        CompletionDecision(game); return true;
    }

    public static void CompleteCommand(GameContext game)
    {
        var state = Active(game);
        state.CommandCompletionPending = true; state.CommandCompletionStep = 0;
        state.CommandCompletionReturnPhase = game.Flow.CommandReturnPhase;
        game.Flow.CommandReturnPhase = GamePhase.SelectAttack; game.Flow.Phase = GamePhase.ResolveCommand;
        CompletionDecision(game);
    }

    private static GameDecision? CompletionDecision(GameContext game)
    {
        int ship = game.Flow.ActiveShip; var state = Active(game);
        while (state.CommandCompletionStep < 4)
        {
            if (state.CommandCompletionStep == 3)
            {
                // Engineering and Squadron remain unresolved while their nested
                // choices run. Enter this window only at actual completion and
                // advance first so resuming the window cannot trigger it twice.
                state.CommandCompletionStep++;
                if (state.CurrentCommandResolved && state.CurrentResolvedCommand is { } completed &&
                    !DravenAbilityEffects.DeferCompletion(game, ship, completed) &&
                    DravenAbilityEffects.BeginReaction(game, ship, completed, GamePhase.ResolveCommand))
                    return new(DecisionKind.Forced, 0, game.Flow.Phase, []);
                continue;
            }
            var options = new List<GameActionOption>();
            if (state.CurrentCommandResolved)
            {
                if (state.CommandCompletionStep == 0 && state.CurrentResolvedCommand == ShipCommand.Squadron && ReadyCard(game, ship, "adar-tallon-officer") is not null)
                    foreach (int entity in state.CommandedSquadrons.Distinct().Where(x => !game.IsDestroyed(x) && game.State.Squadrons[game.SquadronIndex(x)].Activated))
                        Add(options, new(GameActionKind.ChooseEntity, ship, entity, AbilityId: "activation.adar-ready"), "Ready a squadron activated by this command");
                if (state.CommandCompletionStep == 1 && state.CurrentResolvedCommand is { } resolved && (state.ResolvedDialCommands & (1 << (int)resolved)) != 0 && CanGain(game, ship, resolved) &&
                    game.UpgradesFor(ship).Any(x => x.Active && x.Definition.id == "resolute-title" && x.GetTypedCommandTokenCount(resolved) > 0))
                    Add(options, new(GameActionKind.ChooseCommand, ship, Command: resolved, AbilityId: "activation.resolute-token"), "Gain matching Resolute token");
                if (state.CommandCompletionStep == 2 && !state.VantoOffered && ReadyCard(game, ship, "commander-vanto-officer") is not null)
                    foreach (var command in Enum.GetValues<ShipCommand>().Where(x => CanGain(game, ship, x)))
                        Add(options, new(GameActionKind.ChooseCommand, ship, Command: command, AbilityId: "activation.vanto-token"), "Exhaust Commander Vanto to gain " + command);
            }
            if (options.Count > 0)
            {
                Add(options, new(GameActionKind.Pass, ship, AbilityId: "activation.completion-skip"), "Skip this optional effect");
                return Decision(game, options);
            }
            if (state.CommandCompletionStep == 2 && state.CurrentCommandResolved) state.VantoOffered = true;
            state.CommandCompletionStep++;
        }
        if (state.CurrentResolvedCommand == ShipCommand.Squadron)
            foreach (string key in game.Flow.Flags.Keys.Where(x => x.StartsWith($"wolffe.{ship}.", StringComparison.Ordinal) || x.StartsWith($"ahsoka.{ship}.", StringComparison.Ordinal)).ToArray()) game.Flow.Flags.Remove(key);
        state.CurrentCommandResolved = false; state.CurrentResolvedCommand = null; state.CommandedSquadrons.Clear();
        state.CommandResolutionCompleted = true;
        if (state.CommandCompletionReturnPhase == GamePhase.SelectAttack)
            if (NextAdditionalDialDecision(game) is { } next) return next;
        state.CommandCompletionPending = false;
        game.Flow.Phase = state.CommandCompletionReturnPhase;
        return null;
    }

    public static int ConsumeSanHillToken(GameContext game, int ship)
    {
        foreach (var carrier in game.Topology.Ships.Where(x => x.Owner == game.State.Owners[ship] && !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex)))
            if (Geometry2D.Distance(game.ShipBase(carrier.EntityIndex), game.ShipBase(ship)) <= TabletopMeasurements.LongRange)
            {
                var card = game.UpgradesFor(carrier.EntityIndex).FirstOrDefault(x => x.Active && x.Definition.id == "san-hill-officer" && x.GetTypedCommandTokenCount(ShipCommand.Squadron) > 0);
                if (card is not null && card.TrySpendTypedCommandToken(ShipCommand.Squadron)) return 1;
            }
        return 0;
    }

    public static bool ResumeDeferredSquadrons(GameContext game, GamePhase returnPhase)
    {
        var state = Active(game);
        if (state.DeferredSquadronActivations <= 0) return false;
        state.CommandedSquadronsRemaining = state.DeferredSquadronActivations; state.DeferredSquadronActivations = 0;
        state.CurrentCommandResolved = true; state.CurrentResolvedCommand = ShipCommand.Squadron;
        state.SquadronCommandPreludeStep = 2; state.CommandedSquadrons.Clear();
        game.Flow.EffectiveCommandSpendMode = state.DeferredSquadronSpendMode;
        game.Flow.CommandReturnPhase = returnPhase; game.Flow.Phase = GamePhase.SquadronCommand;
        GetSquadronCommandDecision(game); return true;
    }

    public static bool CanResolveCommand(GameContext game, int ship) => game.Flow.Flags.GetValueOrDefault($"round.leia-blocked.{ship}") != game.State.Round;
    public static bool CanBoostLeia(GameContext game, int ship) => CanResolveCommand(game, ship) &&
        game.Flow.Flags.GetValueOrDefault($"round.resolved-command.{ship}") != game.State.Round && FindFleetCard(game, ship, "leia-organa-commander") is not null;
    private static ShipActivationState Active(GameContext game) => game.Flow.ShipActivations[game.Flow.ActiveShip];
    private static UpgradeCardRuntimeState? FindFleetCard(GameContext game, int ship, string id) => game.Topology.Ships
        .Where(x => x.Owner == game.State.Owners[ship] && !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex))
        .SelectMany(x => game.UpgradesFor(x.EntityIndex)).FirstOrDefault(x => x.Active && x.Definition.id == id);
    private static UpgradeCardRuntimeState? ReadyCard(GameContext game, int ship, string id) => game.UpgradesFor(ship).FirstOrDefault(x => x.Active && !x.Exhausted && x.Definition.id == id);
    private static bool CanGain(GameContext game, int ship, ShipCommand command) => TokenAssignments.CanAssign(game, ship, command);
    private static GameDecision Decision(GameContext game, List<GameActionOption> options) => new(DecisionKind.Player, game.State.Owners[game.Flow.ActiveShip], game.Flow.Phase, options.AsReadOnly());
    private static void Add(List<GameActionOption> options, GameAction action, string label) => options.Add(new(action, label));
}
