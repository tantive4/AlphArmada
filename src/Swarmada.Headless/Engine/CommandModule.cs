using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Effects;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Command FIFO planning, reveal, explicit dial/token payment, repairs and squadron orders.</summary>
public sealed class CommandModule : IGameplayModule
{
    public bool Handles(GamePhase phase) => phase is GamePhase.CommandPlanning or GamePhase.RevealCommand or
        GamePhase.ResolveCommand or GamePhase.Repair or GamePhase.SquadronCommand;

    public GameDecision GetDecision(GameContext game)
    {
        using var scope = new GameScope(game);
        var flow = game.Flow;
        if (flow.Phase == GamePhase.CommandPlanning) return Planning(game);
        int ship = flow.ActiveShip;
        if (!game.IsShip(ship)) throw new InvalidOperationException("A command requires an active ship.");
        // Command costs and nested reactions can destroy the activating ship
        // before its next command decision (for example, lethal Crew Panic).
        // The session resolves death replacements first, so a Rieekan-deferred
        // ship still proceeds normally instead of losing its pending command.
        if (game.IsDestroyed(ship))
            return new(DecisionKind.Forced, 0, flow.Phase,
                [new(new(GameActionKind.Pass, ship, AbilityId: "command.end-destroyed"), "Complete destroyed ship activation")]);
        var activation = flow.ShipActivations[ship];
        int owner = game.State.Owners[ship];
        var options = new List<GameActionOption>();
        if (flow.Phase == GamePhase.RevealCommand)
        {
            if (!activation.ShieldsToMaximumChecked)
            {
                if (FleetCommandRules.AllowsPreRevealShieldRecovery(game.ShipViews[ship]))
                    foreach (var zone in game.State.Ships[ship].HullZones)
                        if (game.State.Ships[ship].GetShields(zone) < UnitAbilityEngine.GetEffectiveShieldMaximum(game.ShipViews[ship], zone) &&
                            !(game.State.Ships[ship].GetShields(zone) == 0 && game.State.Ships[ship].HasFaceupCritical(CriticalDamageCard.CapacitorFailure)))
                            Add(options, new(GameActionKind.RecoverShield, ship, ToZone: zone, AbilityId: "command.fleet-shields.recover"), "Recover one shield before revealing command");
                if (options.Count > 0)
                {
                    Add(options, new(GameActionKind.Pass, ship, AbilityId: "command.fleet-shields.skip"), "Skip Shields to Maximum");
                    return new(DecisionKind.Player, owner, flow.Phase, options.AsReadOnly());
                }
                activation.ShieldsToMaximumChecked = true;
            }
            if (game.State.Ships[ship].MustResolveCrewPanicBeforeCommand &&
                flow.Flags.GetValueOrDefault($"crew-panic.{ship}") != game.State.Round && game.State.Commands[ship].Count > 0)
            {
                Add(options, new(GameActionKind.UseAbility, ship, AbilityId: "critical.crew-panic.damage"), "Suffer one facedown damage card");
                Add(options, new(GameActionKind.UseAbility, ship, AbilityId: "critical.crew-panic.discard"), "Discard the top command dial");
            }
            else Add(options, new(GameActionKind.RevealCommand, ship), "Reveal top command dial");
        }
        else if (flow.Phase == GamePhase.ResolveCommand)
        {
            if (ActivationAbilityEffects.GetCommandDecision(game) is { } commandChoice) return commandChoice;
            if (flow.Phase != GamePhase.ResolveCommand) return new(DecisionKind.Forced, 0, flow.Phase, []);
            if (ActivationAbilityEffects.NeedsRevealRaidDecision(game) ||
                !activation.RevealPreludeActive && activation.CommandRevealStep == 2 && game.State.Commands[ship].RaidTokenCount > 0 && !activation.RaidInterferenceResolved)
            {
                var raidCommand = activation.RevealedCommand ?? activation.AdditionalCommandDials.FirstOrDefault().Command;
                AddRaidOptions(game, ship, raidCommand, options);
                Add(options, new(GameActionKind.UseAbility, ship, Command: raidCommand, AbilityId: "raid.keep"), "Keep raid tokens and continue if this dial is unblocked");
                return new(DecisionKind.Player, owner, flow.Phase, options.AsReadOnly());
            }
            if (activation.RevealedCommand is not { } command) throw new InvalidOperationException("No revealed command is pending.");
            bool resolved = (activation.ResolvedCommands & (1 << (int)command)) != 0;
            bool allowed = CommandAllowed(game, ship, command) && ActivationAbilityEffects.CanResolveCommand(game, ship) &&
                !(activation.AdditionalDialActive && command == ShipCommand.Navigate && activation.PendingCommand == ShipCommand.Navigate);
            bool raided = game.State.Commands[ship].RaidTokenCountOf(command) > 0;
            if (!resolved && allowed && !raided)
                foreach (CommandSpendMode mode in Enum.GetValues<CommandSpendMode>())
                    if ((!CommandSpendRules.UsesDial(mode) || activation.RevealedDialAvailable) &&
                        (!CommandSpendRules.UsesToken(mode) || game.State.Commands[ship].HasToken(command)))
                        Add(options, new(GameActionKind.ResolveCommand, ship, Command: command, SpendMode: mode), "Resolve " + command + " using " + mode);
            if (!resolved && allowed && !raided && game.State.Commands[ship].HasToken(command))
                foreach (int upgrade in TokenStrengthUpgradeIndices(game, command))
                    Add(options, new(GameActionKind.ResolveCommand, ship, Command: command, SpendMode: CommandSpendMode.Token, Index: upgrade,
                        AbilityId: "command.resolve-strengthened-token"), "Spend token at dial strength with " + game.State.Upgrades[upgrade].Definition.name);
            if (!resolved && allowed && !raided && activation.RevealedDialAvailable && ActivationAbilityEffects.CanBoostLeia(game, ship))
                Add(options, new(GameActionKind.UseAbility, ship, Command: command, AbilityId: "commander.leia-organa.resolve-boosted"), "Resolve dial at dial-and-token strength with Leia Organa");
            if (!resolved && allowed && !raided && activation.RevealedDialAvailable && ActivationAbilityEffects.CanUseBeck(game, command, false) &&
                (command is ShipCommand.Engineering or ShipCommand.Squadron || (command == ShipCommand.Navigate && flow.CommandReturnPhase == GamePhase.SelectCourse)))
                Add(options, new(GameActionKind.UseAbility, ship, Command: command, AbilityId: "command.beck-combined"), "Resolve dial with Beck's supplied token and gain matching raid");
            if (activation.RevealedDialAvailable && !activation.RevealBankingClosed && !raided && TokenAssignments.CanAssign(game, ship, command))
                Add(options, new(GameActionKind.BankCommand, ship, Command: command), "Bank dial as " + command + " token");
            if (activation.RevealedDialAvailable && command == ShipCommand.Navigate && flow.CommandReturnPhase != GamePhase.SelectCourse)
                Add(options, new(GameActionKind.SkipCommand, ship, Command: command, AbilityId: "command.hold-navigate"), "Hold Navigate dial until Determine Course");
            Add(options, new(GameActionKind.SkipCommand, ship, Command: command), "Discard command without resolving");
        }
        else if (flow.Phase == GamePhase.Repair) RepairOptions(game, ship, options);
        else
        {
            if (ActivationAbilityEffects.GetSquadronCommandDecision(game) is { } squadronChoice) return squadronChoice;
            if (flow.Phase != GamePhase.SquadronCommand) return new(DecisionKind.Forced, 0, flow.Phase, []);
            if (activation.CommandedSquadronsRemaining > 0)
                for (int squadron = 0; squadron < game.State.Squadrons.Length; squadron++)
                {
                    int entity = game.State.Ships.Length + squadron;
                    if (SquadronModule.CanActivate(game, entity, owner, commanded: true))
                        foreach (var relay in CommandRelaySources(game, ship, entity))
                            Add(options, new(GameActionKind.ActivateSquadron, ship, entity, Index: relay.Keyword, Amount: relay.Title + 1,
                                AbilityId: relay.Matchstick ? "command.matchstick" : ""),
                                "Activate " + game.Topology.Squadrons[squadron].CatalogEntry.Definition.DisplayName + (relay.Keyword < 0 && relay.Title < 0 && !relay.Matchstick ? "" : " through Relay"));
                }
            Add(options, new(GameActionKind.Pass, ship, AbilityId: "command.finish-squadrons"), "Finish Squadron command");
        }
        return new(DecisionKind.Player, owner, flow.Phase, options.AsReadOnly());
    }

    public void Apply(GameContext game, GameAction action)
    {
        var decision = GetDecision(game);
        if (!decision.Actions.Any(x => x.Template == action)) throw new ArgumentException("Illegal command action.", nameof(action));
        if (action.AbilityId == "command.end-destroyed")
        { RoundModule.EndShipActivation(game); return; }
        if (game.Flow.Phase == GamePhase.CommandPlanning && ActivationAbilityEffects.TryApplyPlanningChoice(game, action)) return;
        if (ActivationAbilityEffects.TryApplyCommandChoice(game, action)) return;
        var flow = game.Flow;
        if (flow.Phase == GamePhase.CommandPlanning)
        {
            if (!game.State.Commands[action.Source].QueueAtBottom(action.Command)) throw new InvalidOperationException("Command stack is already full.");
            if (game.State.Commands[action.Source].CommandsNeeded == 0 && ForcedPlanningCommand(game, action.Source) is not null)
                game.UpgradesFor(action.Source).First(x => x.Active && x.Definition.id == "tikkes-officer").ClearTypedCommandTokens();
            AdvancePlanning(game); return;
        }
        int ship = flow.ActiveShip;
        var activation = flow.ShipActivations[ship];
        var commands = game.State.Commands[ship];
        if (flow.Phase == GamePhase.RevealCommand)
        {
            if (action.AbilityId.StartsWith("command.fleet-shields.", StringComparison.Ordinal))
            {
                activation.ShieldsToMaximumChecked = true;
                if (action.Kind == GameActionKind.RecoverShield) game.State.Ships[ship].RestoreShield(action.ToZone, 1, UnitAbilityEngine.HasAuxiliaryShieldsTeam(game.ShipViews[ship]));
                return;
            }
            if (action.AbilityId.StartsWith("critical.crew-panic.", StringComparison.Ordinal))
            {
                flow.Flags[$"crew-panic.{ship}"] = game.State.Round;
                if (action.AbilityId.EndsWith("discard", StringComparison.Ordinal))
                { commands.TryDiscardTop(out _); ActivationAbilityEffects.BeginReveal(game, hasNativeReveal: false); flow.Phase = GamePhase.ResolveCommand; }
                else game.State.Ships[ship].DealFacedownDamageCard(game.State.Decks[game.State.Owners[ship] - 1].Draw());
                return;
            }
            if (!commands.TryRevealTop(out var revealed))
            { ActivationAbilityEffects.BeginReveal(game, hasNativeReveal: false); flow.Phase = GamePhase.ResolveCommand; return; }
            activation.RevealedCommand = revealed; activation.RevealedDialAvailable = true;
            ActivationAbilityEffects.BeginReveal(game);
            activation.CommandResolutionCompleted = false; flow.Phase = GamePhase.ResolveCommand; return;
        }
        if (flow.Phase == GamePhase.Repair)
        { if (!RepairAbilityEffects.TryApply(game, action)) ApplyRepair(game, action); return; }
        if (flow.Phase == GamePhase.SquadronCommand)
        {
            if (action.Kind == GameActionKind.Pass) { activation.CommandedSquadronsRemaining = 0; FinishCommand(game); return; }
            if (action.Index >= 0) game.Flow.Flags[$"relay.{ship}.{action.Index}"] = game.Flow.Flags.GetValueOrDefault($"relay.{ship}.{action.Index}") + 1;
            if (action.Amount > 0) game.Flow.Flags[$"title-relay.{ship}.{action.Amount - 1}"] = game.Flow.Flags.GetValueOrDefault($"title-relay.{ship}.{action.Amount - 1}") + 1;
            if (action.AbilityId == "command.matchstick") game.Flow.Flags[$"matchstick.{ship}.{action.Target}"] = 1;
            activation.UncommittedRelayKeyword = action.Index; activation.UncommittedRelayTitle = action.Amount - 1;
            activation.UncommittedRelayTarget = action.Target; activation.UncommittedMatchstick = action.AbilityId == "command.matchstick";
            activation.CommandedSquadrons.Add(action.Target);
            SquadronModule.BeginActivation(game, action.Target, commanded: true); return;
        }
        if (action.Kind == GameActionKind.BankCommand)
        { TokenAssignments.TryAssign(game, ship, action.Command); activation.RevealedDialAvailable = false; FinishCommand(game); return; }
        if (action.Kind == GameActionKind.SkipCommand)
        {
            if (action.AbilityId == "command.hold-navigate")
                activation.PendingCommand = ShipCommand.Navigate;
            activation.RevealedDialAvailable = false; FinishCommand(game); return;
        }
        if (action.AbilityId.StartsWith("raid.", StringComparison.Ordinal))
        {
            activation.RaidInterferenceResolved = true;
            if (action.AbilityId == "raid.keep")
            {
                if (activation.RevealedCommand is { } blocked && commands.RaidTokenCountOf(blocked) > 0)
                { activation.RevealedDialAvailable = false; FinishCommand(game); }
            }
            else if (action.AbilityId == "raid.discard-dial")
            { activation.RevealedDialAvailable = false; commands.DiscardRaidTokensForDial(); FinishCommand(game); }
            else if (action.AbilityId == "raid.discard-dial-one")
            { activation.RevealedDialAvailable = false; commands.DiscardRaidToken(action.Command); FinishCommand(game); }
            else
            {
                commands.TrySpendToken(action.Command); commands.DiscardRaidToken(action.Command);
                if (activation.RevealedCommand is { } pending && commands.RaidTokenCountOf(pending) > 0) FinishCommand(game);
            }
            return;
        }
        if (action.AbilityId == "commander.leia-organa.resolve-boosted")
        {
            flow.Flags[$"round.leia-blocked.{ship}"] = game.State.Round;
            Resolve(game, action.Command, CommandSpendMode.Dial, forcedMode: CommandSpendMode.DialAndToken); return;
        }
        if (action.AbilityId == "command.beck-combined")
        {
            ActivationAbilityEffects.CommitBeckCommand(game, action.Command);
            Resolve(game, action.Command, CommandSpendMode.DialAndToken, payResources: false, consumeRevealedDial: true, forcedMode: CommandSpendMode.DialAndToken); return;
        }
        if (action.AbilityId == "command.resolve-strengthened-token")
        {
            if (!game.State.Upgrades[action.Index].Exhaust()) throw new ArgumentException("Token-strength upgrade is unavailable.");
            Resolve(game, action.Command, CommandSpendMode.Token, forcedMode: CommandSpendMode.Dial); return;
        }
        Resolve(game, action.Command, action.SpendMode);
    }

    private static void AddRaidOptions(GameContext game, int ship, ShipCommand command, List<GameActionOption> options)
    {
        var activation = game.Flow.ShipActivations[ship]; var commands = game.State.Commands[ship];
        if (activation.RevealedDialAvailable)
        {
            if (RaidRules.DiscardLimitFor(game.ShipViews[ship]) >= commands.RaidTokenCount)
                Add(options, new(GameActionKind.UseAbility, ship, Command: command, AbilityId: "raid.discard-dial"), "Discard dial to remove all raid tokens");
            else foreach (var raid in commands.RaidTokens.Distinct())
                Add(options, new(GameActionKind.UseAbility, ship, Command: raid, AbilityId: "raid.discard-dial-one"), "Discard dial to remove one " + raid + " raid token");
        }
        foreach (var raid in commands.RaidTokens.Distinct().Where(commands.HasToken))
            Add(options, new(GameActionKind.UseAbility, ship, Command: raid, AbilityId: "raid.discard-token"), "Discard matching command token to remove " + raid + " raid");
    }

    public static IReadOnlyList<GameActionOption> AdditionalTokenOptions(GameContext game, GamePhase returnPhase)
    {
        int ship = game.Flow.ActiveShip;
        if (!game.IsShip(ship)) return Array.Empty<GameActionOption>();
        var activation = game.Flow.ShipActivations[ship];
        var commands = Enum.GetValues<ShipCommand>().Where(command => CommandAllowed(game, ship, command) && ActivationAbilityEffects.CanResolveCommand(game, ship) &&
            game.State.Commands[ship].HasToken(command) && (activation.ResolvedCommands & (1 << (int)command)) == 0 &&
            game.State.Commands[ship].RaidTokenCountOf(command) == 0).ToArray();
        var options = new List<GameActionOption>();
        foreach (var command in commands)
        {
            Add(options, new(GameActionKind.UseAbility, ship, Command: command, SpendMode: CommandSpendMode.Token,
                AbilityId: "command.resolve-additional-token"), "Resolve " + command + " token");
            if (command != ShipCommand.ConcentrateFire)
                foreach (int upgrade in TokenStrengthUpgradeIndices(game, command))
                    Add(options, new(GameActionKind.UseAbility, ship, Command: command, SpendMode: CommandSpendMode.Token, Index: upgrade,
                        AbilityId: "command.resolve-strengthened-token"), "Spend " + command + " token at dial strength with " + game.State.Upgrades[upgrade].Definition.name);
        }
        return options;
    }

    public static void ResolveAdditionalToken(GameContext game, GameAction action, GamePhase returnPhase)
    {
        if (!AdditionalTokenOptions(game, returnPhase).Any(x => x.Template == action)) throw new ArgumentException("Illegal additional command token.");
        game.Flow.CommandReturnPhase = returnPhase;
        if (action.AbilityId == "command.resolve-strengthened-token")
        {
            if (!game.State.Upgrades[action.Index].Exhaust()) throw new ArgumentException("Token-strength upgrade is unavailable.");
            Resolve(game, action.Command, CommandSpendMode.Token, consumeRevealedDial: false, forcedMode: CommandSpendMode.Dial);
        }
        else Resolve(game, action.Command, CommandSpendMode.Token, consumeRevealedDial: false);
    }

    public static IReadOnlyList<GameActionOption> ActivationCommandOptions(GameContext game, GamePhase phase)
    {
        int ship = game.Flow.ActiveShip;
        if (!game.IsShip(ship) || !ActivationAbilityEffects.CanResolveCommand(game, ship)) return [];
        bool Eligible(ShipCommand command) => phase == GamePhase.SelectCourse ? command == ShipCommand.Navigate :
            phase == GamePhase.SelectAttack && game.Flow.ShipActivations[ship].AttacksResolved == 0 && command is ShipCommand.Engineering or ShipCommand.Squadron;
        var options = AdditionalTokenOptions(game, phase).Where(x => Eligible(x.Template.Command)).ToList();
        foreach (var command in Enum.GetValues<ShipCommand>().Where(x => Eligible(x) && CommandAllowed(game, ship, x) && ActivationAbilityEffects.CanUseBeck(game, x, false)))
            Add(options, new(GameActionKind.UseAbility, ship, Command: command, AbilityId: "command.beck-token"), "Resolve " + command + " at Beck token strength and gain matching raid");
        return options;
    }

    public static bool ResolveActivationCommand(GameContext game, GameAction action, GamePhase returnPhase)
    {
        if (action.AbilityId is not ("command.resolve-additional-token" or "command.resolve-strengthened-token" or "command.beck-token")) return false;
        if (!ActivationCommandOptions(game, returnPhase).Any(x => x.Template == action)) throw new ArgumentException("Command is unavailable at this activation timing.");
        if (action.AbilityId != "command.beck-token") ResolveAdditionalToken(game, action, returnPhase);
        else
        {
            ActivationAbilityEffects.CommitBeckCommand(game, action.Command);
            ResolveGrantedCommand(game, action.Command, CommandSpendMode.Token, returnPhase);
        }
        return true;
    }

    public static IReadOnlyList<int> TokenStrengthUpgradeIndices(GameContext game, ShipCommand command)
    {
        int ship = game.Flow.ActiveShip;
        if (!game.IsShip(ship) || !game.State.Commands[ship].HasToken(command)) return [];
        int piett = -1, droid = -1;
        foreach (var upgrade in game.Topology.Upgrades)
        {
            var card = game.State.Upgrades[upgrade.UpgradeIndex];
            if (!card.Active || card.Exhausted) continue;
            int carrier = (int)game.State.Registers[game.Topology.Registers.UpgradeOwner(upgrade.UpgradeIndex)] - 1;
            if (!game.IsShip(carrier) || game.IsDestroyed(carrier) || game.IsReserved(carrier) || game.State.Owners[carrier] != game.State.Owners[ship]) continue;
            if (piett < 0 && card.Definition.id == "admiral-piett-commander") piett = upgrade.UpgradeIndex;
            if (droid < 0 && carrier == ship && command != ShipCommand.ConcentrateFire && card.Definition.id == "t-series-tactical-droid-officer") droid = upgrade.UpgradeIndex;
        }
        return piett < 0 ? droid < 0 ? [] : [droid] : droid < 0 ? [piett] : [piett, droid];
    }

    public static void ResolveStrengthenedToken(GameContext game, GameAction action, GamePhase returnPhase)
    {
        int ship = game.Flow.ActiveShip;
        if (!game.IsShip(ship) || action.Source != ship || !ActivationAbilityEffects.CanResolveCommand(game, ship) ||
            !CommandAllowed(game, ship, action.Command) || !TokenStrengthUpgradeIndices(game, action.Command).Contains(action.Index) ||
            game.State.Commands[ship].RaidTokenCountOf(action.Command) > 0 ||
            (game.Flow.ShipActivations[ship].ResolvedCommands & (1 << (int)action.Command)) != 0)
            throw new ArgumentException("Strengthened command token is unavailable.");
        if (!game.State.Upgrades[action.Index].Exhaust()) throw new ArgumentException("Token-strength upgrade is unavailable.");
        game.Flow.CommandReturnPhase = returnPhase;
        Resolve(game, action.Command, CommandSpendMode.Token, consumeRevealedDial: false, forcedMode: CommandSpendMode.Dial);
    }

    /// <summary>Resolves a command whose dial/token payment has already been handled by an ability.
    /// Preserve an unrelated revealed dial unless the granting effect explicitly consumes it.</summary>
    public static void ResolveGrantedCommand(GameContext game, ShipCommand command, CommandSpendMode effectMode,
        GamePhase returnPhase, bool consumeRevealedDial = false, CommandSpendMode? paidMode = null)
    {
        if (!CommandAllowed(game, game.Flow.ActiveShip, command)) throw new ArgumentException("This command is forbidden for the active ship.");
        game.Flow.CommandReturnPhase = returnPhase;
        Resolve(game, command, paidMode ?? effectMode, payResources: false, consumeRevealedDial, forcedMode: effectMode);
    }

    public static bool BeginHeldNavigateForManeuver(GameContext game)
    {
        int ship = game.Flow.ActiveShip;
        if (!game.IsShip(ship) || game.Flow.ShipActivations[ship].PendingCommand != ShipCommand.Navigate) return false;
        var activation = game.Flow.ShipActivations[ship];
        activation.PendingCommand = null; activation.RevealedCommand = ShipCommand.Navigate;
        activation.RevealedDialAvailable = true; activation.CommandResolutionCompleted = false;
        game.Flow.CommandReturnPhase = GamePhase.SelectCourse;
        game.Flow.Phase = GamePhase.ResolveCommand;
        return true;
    }

    private static GameDecision Planning(GameContext game)
    {
        AdvancePlanning(game);
        if (game.Flow.Phase != GamePhase.CommandPlanning)
            return new(DecisionKind.Forced, 0, game.Flow.Phase, Array.Empty<GameActionOption>());
        int ship = game.Flow.PlanningShip;
        if (ActivationAbilityEffects.GetPlanningDecision(game) is { } chimaeraChoice) return chimaeraChoice;
        var forced = ForcedPlanningCommand(game, ship);
        var actions = Enum.GetValues<ShipCommand>().Where(command => CommandAllowed(game, ship, command) && (forced is null || command == forced)).Select(command =>
            new GameActionOption(new GameAction(GameActionKind.AssignCommand, ship, Command: command), "Queue " + command + " at bottom")).ToArray();
        return new(DecisionKind.Player, game.State.Owners[ship], GamePhase.CommandPlanning, actions);
    }

    private static ShipCommand? ForcedPlanningCommand(GameContext game, int ship)
    {
        if (game.Topology.Ships[ship].IsScenarioUnit) return null;
        var tikkes = game.UpgradesFor(ship).FirstOrDefault(x => x.Active && x.Definition.id == "tikkes-officer");
        if (tikkes is null) return null;
        foreach (var command in Enum.GetValues<ShipCommand>())
            if (tikkes.GetTypedCommandTokenCount(command) > 0) return command;
        return null;
    }

    private static void AdvancePlanning(GameContext game)
    {
        // Present each player's hidden planning choices in initiative order. Existing FIFO dials remain.
        foreach (int owner in new[] { game.Topology.FirstPlayer, 3 - game.Topology.FirstPlayer })
            for (int ship = 0; ship < game.State.Ships.Length; ship++)
                if (game.State.Owners[ship] == owner && !game.IsDestroyed(ship) && game.State.Commands[ship].CommandsNeeded > 0)
                { game.Flow.PlanningShip = ship; game.Flow.DecisionPlayer = owner; return; }
        game.Flow.CurrentPlayer = game.Topology.FirstPlayer; game.Flow.Phase = GamePhase.StartShipPhase;
    }

    private static void Resolve(GameContext game, ShipCommand command, CommandSpendMode mode, bool payResources = true, bool consumeRevealedDial = true, CommandSpendMode? forcedMode = null)
    {
        using var scope = new GameScope(game);
        int ship = game.Flow.ActiveShip;
        var activation = game.Flow.ShipActivations[ship];
        if ((activation.ResolvedCommands & (1 << (int)command)) != 0) throw new InvalidOperationException("A command can resolve only once per activation.");
        if (payResources && CommandSpendRules.UsesToken(mode) && !game.State.Commands[ship].TrySpendToken(command)) throw new InvalidOperationException("No matching token to spend.");
        if (payResources && CommandSpendRules.UsesToken(mode))
        {
            var wullf = game.UpgradesFor(ship).FirstOrDefault(x => x.Active && !x.Exhausted && x.Definition.id == "wullf-yularen-officer");
            if (wullf is not null && TokenAssignments.CanAssign(game, ship, command))
            { wullf.Exhaust(); TokenAssignments.TryAssign(game, ship, command); }
        }
        activation.ResolvedCommands |= 1 << (int)command;
        activation.CurrentCommandResolved = true;
        activation.CurrentResolvedCommand = command;
        game.Flow.Flags[$"round.resolved-command.{ship}"] = game.State.Round;
        if (CommandSpendRules.UsesDial(mode)) activation.ResolvedDialCommands |= 1 << (int)command;
        if (consumeRevealedDial) activation.RevealedDialAvailable = false;
        var paidMode = mode;
        if (forcedMode is not null) mode = forcedMode.Value;
        else if (payResources)
        {
            mode = CommanderRules.UpgradeDialSpendMode(game.ShipViews[ship], mode);
            if (paidMode == CommandSpendMode.Token && RemainingUpgradeRules.TrenchUpgradesToken(game.State.Owners[ship], command)) mode = CommandSpendMode.DialAndToken;
        }
        game.Flow.EffectiveCommandSpendMode = mode;
        switch (command)
        {
            case ShipCommand.Navigate:
                activation.NavigateDial |= CommandSpendRules.UsesDial(mode);
                activation.NavigateToken |= CommandSpendRules.UsesToken(paidMode);
                activation.NavigateSpeedLimit = Math.Max(activation.NavigateSpeedLimit, CommandSpendRules.NavigateSpeedLimit(mode));
                FinishCommand(game); break;
            case ShipCommand.ConcentrateFire:
                game.Flow.Flags["activation.cf-mode." + ship] = (int)mode;
                activation.ConcentrateFireDie |= CommandSpendRules.GrantsConcentrateFireDie(mode) || (mode == CommandSpendMode.Token && game.HasUpgrade(ship, "ravager-title"));
                activation.ConcentrateFireReroll |= CommandSpendRules.GrantsConcentrateFireReroll(mode);
                activation.ConcentrateFireRavagerChoice = mode == CommandSpendMode.Token && game.HasUpgrade(ship, "ravager-title");
                FinishCommand(game); break;
            case ShipCommand.Engineering:
                int points = game.State.Ships[ship].GetEngineeringValue(CommandSpendRules.EngineeringPoints(game.Topology.Ships[ship].CatalogEntry.Definition.Engineering, mode));
                TitleRules.AdditionalEngineeringPoint(game.ShipViews[ship], ref points);
                activation.EngineeringPoints = points; game.Flow.Phase = GamePhase.Repair;
                RepairAbilityEffects.BeginEngineering(game, ship); break;
            case ShipCommand.Squadron:
                foreach (string key in game.Flow.Flags.Keys.Where(x => x.StartsWith($"relay.{ship}.", StringComparison.Ordinal)).ToArray()) game.Flow.Flags.Remove(key);
                foreach (string key in game.Flow.Flags.Keys.Where(x => x.StartsWith($"title-relay.{ship}.", StringComparison.Ordinal) || x.StartsWith($"matchstick.{ship}.", StringComparison.Ordinal)).ToArray()) game.Flow.Flags.Remove(key);
                foreach (string key in game.Flow.Flags.Keys.Where(x => x.StartsWith($"ahsoka.{ship}.", StringComparison.Ordinal)).ToArray()) game.Flow.Flags.Remove(key);
                int value = CommandUpgradeRules.SquadronValue(game.ShipViews[ship]);
                int count = CommandSpendRules.SquadronActivations(value, mode);
                count += TitleRules.AdditionalSquadronActivations(game.ShipViews[ship], paidMode) + CommanderRules.AdditionalSquadronActivations(game.ShipViews[ship], paidMode);
                count += ActivationAbilityEffects.ConsumeSanHillToken(game, ship);
                activation.CommandedSquadrons.Clear();
                activation.CommandedSquadronsRemaining = count; game.Flow.Phase = GamePhase.SquadronCommand;
                ActivationAbilityEffects.BeginSquadronCommand(game);
                ActivationAbilityEffects.GetSquadronCommandDecision(game); break;
        }
    }

    private static void RepairOptions(GameContext game, int ship, List<GameActionOption> options)
    {
        var state = game.State.Ships[ship]; int points = game.Flow.ShipActivations[ship].EngineeringPoints;
        bool reserves = game.UpgradesFor(ship).Any(x => x.Active && x.Exhausted && x.Definition.id == "battle-droid-reserves");
        int facedownCost = reserves ? 2 : 3;
        if (points >= facedownCost && state.FacedownDamageCardCount > 0)
            Add(options, new(GameActionKind.RepairHull, ship, Index: -1, Amount: facedownCost), "Discard one facedown damage card");
        if (points >= 5)
            foreach (var critical in state.GetFaceupCriticalCards().Distinct())
                Add(options, new(GameActionKind.RepairHull, ship, Index: (int)critical, Amount: 5), "Discard " + critical);
        bool auxiliary = game.HasUpgrade(ship, "auxiliary-shields-team");
        foreach (var zone in state.HullZones)
        {
            int maximum = EffectiveMaximum(state, zone, auxiliary);
            if (points >= 2 && state.GetShields(zone) < maximum && !(state.GetShields(zone) == 0 && state.HasFaceupCritical(CriticalDamageCard.CapacitorFailure)))
                Add(options, new(GameActionKind.RecoverShield, ship, ToZone: zone, Amount: 2), "Recover one " + zone + " shield");
            if (points >= 1 && state.GetShields(zone) > 0)
                foreach (var destination in state.HullZones.Where(x => x != zone && state.GetShields(x) < EffectiveMaximum(state, x, auxiliary)))
                    if (!(state.GetShields(destination) == 0 && state.HasFaceupCritical(CriticalDamageCard.CapacitorFailure)))
                        Add(options, new(GameActionKind.TransferShield, ship, FromZone: zone, ToZone: destination, Amount: 1), "Move shield from " + zone + " to " + destination);
        }
        if (FleetHasUpgrade(game, game.State.Owners[ship], "admiral-yularen-commander") && points >= 2)
            foreach (var squadron in game.Topology.Squadrons)
                if (squadron.Owner == game.State.Owners[ship] && !game.IsDestroyed(squadron.EntityIndex) && !game.IsReserved(squadron.EntityIndex) &&
                    game.State.Squadrons[squadron.SquadronIndex].CurrentHull < squadron.CatalogEntry.Definition.Hull &&
                    Geometry2D.Distance(game.SquadronBase(squadron.EntityIndex), game.ShipBase(ship)) <= TabletopMeasurements.MediumRange)
                    Add(options, new(GameActionKind.RepairHull, ship, squadron.EntityIndex, AbilityId: "repair.yularen", Amount: 2), "Repair friendly squadron");
        options.AddRange(RepairAbilityEffects.GetActions(game, ship));
        options.AddRange(OfficialAssaultObjectives.GetEngineeringActions(game, ship));
        options.AddRange(OfficialDefenseObjectives.GetEngineeringActions(game, ship));
        Add(options, new(GameActionKind.Pass, ship, AbilityId: "command.finish-repair"), "Finish Engineering command");
    }

    private static void ApplyRepair(GameContext game, GameAction action)
    {
        int ship = game.Flow.ActiveShip; var state = game.State.Ships[ship];
        var activation = game.Flow.ShipActivations[ship];
        if (OfficialAssaultObjectives.TryApplyEngineering(game, action) || OfficialDefenseObjectives.TryApplyEngineering(game, action)) return;
        if (action.Kind == GameActionKind.Pass) { activation.EngineeringPoints = 0; FinishCommand(game); return; }
        if (action.AbilityId == "repair.yularen")
            game.State.Squadrons[game.SquadronIndex(action.Target)].RecoverHull(1, game.Topology.Squadrons[game.SquadronIndex(action.Target)].CatalogEntry.Definition.Hull);
        else if (action.Kind == GameActionKind.RecoverShield) state.RestoreShield(action.ToZone, 1, game.HasUpgrade(ship, "auxiliary-shields-team"));
        else if (action.Kind == GameActionKind.TransferShield) state.TransferShield(action.FromZone, action.ToZone, game.HasUpgrade(ship, "auxiliary-shields-team"));
        else if (action.Index < 0)
        {
            if (!state.TryRepairFacedownDamage(out var card)) throw new InvalidOperationException("No facedown damage card to repair.");
            game.State.Decks[game.State.Owners[ship] - 1].Discard(card);
        }
        else
        {
            var card = (CriticalDamageCard)action.Index;
            if (state.RepairFaceupDamage(card) != 1) throw new InvalidOperationException("Faceup damage card is absent.");
            game.State.Decks[game.State.Owners[ship] - 1].Discard(card);
        }
        activation.EngineeringPoints -= action.Amount;
    }

    public static bool CanCommandSquadron(GameContext game, int ship, int entity)
    {
        if (!SquadronModule.CanActivate(game, entity, game.State.Owners[ship], commanded: true)) return false;
        return CommandRelaySources(game, ship, entity).Any();
    }
    private readonly record struct CommandRelayRoute(int Keyword = -1, int Title = -1, bool Matchstick = false);
    private static IEnumerable<CommandRelayRoute> CommandRelaySources(GameContext game, int ship, int entity)
    {
        float range = CommandUpgradeEffects.SquadronCommandRangeMillimetres(game.HasUpgrade(ship, "boosted-comms"));
        if (Geometry2D.Distance(game.SquadronBase(entity), game.ShipBase(ship)) <= range) { yield return new(-1, -1); yield break; }
        if (SquadronAbilityEngine.HasBomber(game.SquadronViews[game.SquadronIndex(entity)]) && game.Flow.Flags.Keys.Count(x => x.StartsWith($"matchstick.{ship}.", StringComparison.Ordinal)) < 3 &&
            game.Topology.Squadrons.Any(x => x.Owner == game.State.Owners[ship] && !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex) && x.CatalogEntry.Id == "matchstick-btl-b-y-wing-squadron" &&
                Geometry2D.Distance(game.SquadronBase(x.EntityIndex), game.ShipBase(ship)) <= range && Geometry2D.Distance(game.SquadronBase(x.EntityIndex), game.SquadronBase(entity)) <= TabletopMeasurements.DistanceBand(3)))
            yield return new(-1, -1, true);
        foreach (var relay in game.Topology.Squadrons)
        {
            if (relay.Owner != game.State.Owners[ship] || game.IsDestroyed(relay.EntityIndex) || game.IsReserved(relay.EntityIndex)) continue;
            int limit = KeywordValue(relay.CatalogEntry.Definition.Keywords, "RELAY");
            if (limit > 0 && game.Flow.Flags.GetValueOrDefault($"relay.{ship}.{relay.EntityIndex}") < limit &&
                Geometry2D.Distance(game.SquadronBase(relay.EntityIndex), game.ShipBase(ship)) <= range &&
                Geometry2D.Distance(game.SquadronBase(entity), game.SquadronBase(relay.EntityIndex)) <= TabletopMeasurements.DistanceBand(3)) yield return new(relay.EntityIndex, -1);
        }
        foreach (var carrier in game.Topology.Ships.Where(x => x.EntityIndex != ship && x.Owner == game.State.Owners[ship] && !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex)))
        {
            int used = (int)game.Flow.Flags.GetValueOrDefault($"title-relay.{ship}.{carrier.EntityIndex}");
            if (game.HasUpgrade(carrier.EntityIndex, "centicore-title") && used < 2)
            {
                if (Geometry2D.Distance(game.SquadronBase(entity), game.ShipBase(carrier.EntityIndex)) <= TabletopMeasurements.MediumRange) yield return new(-1, carrier.EntityIndex);
                foreach (var relay in game.Topology.Squadrons.Where(x => x.Owner == game.State.Owners[ship] && !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex)))
                {
                    int limit = KeywordValue(relay.CatalogEntry.Definition.Keywords, "RELAY");
                    if (limit > game.Flow.Flags.GetValueOrDefault($"relay.{ship}.{relay.EntityIndex}") && Geometry2D.Distance(game.SquadronBase(relay.EntityIndex), game.ShipBase(carrier.EntityIndex)) <= TabletopMeasurements.MediumRange &&
                        Geometry2D.Distance(game.SquadronBase(relay.EntityIndex), game.SquadronBase(entity)) <= TabletopMeasurements.DistanceBand(3)) yield return new(relay.EntityIndex, carrier.EntityIndex);
                }
            }
            if (game.State.Round > 1 && game.HasUpgrade(carrier.EntityIndex, "triumphant-title") && !RemainingUpgradeRules.IsFlotilla(game.ShipViews[ship]) && used < 3 &&
                KeywordValue(game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition.Keywords, "ADEPT") == 0 &&
                Geometry2D.Distance(game.SquadronBase(entity), game.ShipBase(carrier.EntityIndex)) <= TabletopMeasurements.CloseRange) yield return new(-1, carrier.EntityIndex);
        }
    }

    public static void CancelSquadronSelection(GameContext game, int entity)
    {
        int ship = game.Flow.ActiveShip; if (!game.IsShip(ship)) return;
        var state = game.Flow.ShipActivations[ship]; if (state.UncommittedRelayTarget != entity) return;
        void Refund(string key) { if (game.Flow.Flags.GetValueOrDefault(key) <= 1) game.Flow.Flags.Remove(key); else game.Flow.Flags[key]--; }
        if (state.UncommittedRelayKeyword >= 0) Refund($"relay.{ship}.{state.UncommittedRelayKeyword}");
        if (state.UncommittedRelayTitle >= 0) Refund($"title-relay.{ship}.{state.UncommittedRelayTitle}");
        if (state.UncommittedMatchstick) game.Flow.Flags.Remove($"matchstick.{ship}.{entity}");
        int index = state.CommandedSquadrons.LastIndexOf(entity); if (index >= 0) state.CommandedSquadrons.RemoveAt(index);
        CommitSquadronSelection(game, entity);
    }

    public static void CommitSquadronSelection(GameContext game, int entity)
    {
        int ship = game.Flow.ActiveShip; if (!game.IsShip(ship)) return;
        var state = game.Flow.ShipActivations[ship]; if (state.UncommittedRelayTarget != entity) return;
        state.UncommittedRelayKeyword = state.UncommittedRelayTitle = state.UncommittedRelayTarget = -1; state.UncommittedMatchstick = false;
    }
    internal static int KeywordValue(IReadOnlyList<string> keywords, string name)
    {
        foreach (string keyword in keywords)
        {
            var parts = keyword.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length > 0 && string.Equals(parts[0], name, StringComparison.OrdinalIgnoreCase))
                return parts.Length > 1 && int.TryParse(parts[1], out int value) ? value : 1;
        }
        return 0;
    }
    internal static bool FleetHasUpgrade(GameContext game, int owner, string id) => game.Topology.Ships.Any(x => x.Owner == owner && !game.IsDestroyed(x.EntityIndex) && game.HasUpgrade(x.EntityIndex, id));
    private static bool FriendlyShipWithUpgrade(GameContext game, int ship, string id, float range) => game.Topology.Ships.Any(x => x.Owner == game.State.Owners[ship] &&
        !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex) && game.HasUpgrade(x.EntityIndex, id) && Geometry2D.Distance(game.ShipBase(ship), game.ShipBase(x.EntityIndex)) <= range);
    internal static bool CommandAllowed(GameContext game, int ship, ShipCommand command) => !game.Topology.Ships[ship].IsStation &&
        (!game.Topology.Ships[ship].IsScenarioUnit || command == ShipCommand.Navigate);
    private static int EffectiveMaximum(ShipRuntimeState ship, HullZone zone, bool auxiliary) => auxiliary && zone is HullZone.Left or HullZone.Right ? Math.Min(4, ship.GetMaximumShields(zone) + 1) : ship.GetMaximumShields(zone);
    private static void FinishCommand(GameContext game)
    {
        var state = game.Flow.ShipActivations[game.Flow.ActiveShip];
        if (state.RevealPreludeActive && !state.RevealAfterWindowOpened && !state.CurrentCommandResolved)
        {
            state.NativeDialConsumedBeforePrelude = true;
            state.RevealedDialAvailable = false;
            return;
        }
        ActivationAbilityEffects.CompleteCommand(game);
    }
    private static void Add(List<GameActionOption> options, GameAction action, string label) => options.Add(new(action, label));
}
