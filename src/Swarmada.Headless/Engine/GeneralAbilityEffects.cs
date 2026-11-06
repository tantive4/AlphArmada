using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public static class GeneralAbilityEffects
{
    public const string AhsokaActionPrefix = "remaining.command-reveal.ahsoka.";
    private static readonly HashSet<string> OwnedCards = new(StringComparer.Ordinal)
    {
        "interdictor-title", "hand-of-justice-title", "beast-of-burden-title", "disposable-capacitors", "phylon-q7-tractor-beams",
        "navigation-officer", "engineering-captain-officer", "tactical-expert-officer", "wing-commander-officer",
        "veteran-captain-officer", "raymus-antilles-officer", "taskmaster-grint-officer", "defense-liaison-officer", "weapons-liaison-officer",
        "skilled-first-officer", "fb-88-title", "walex-blissex-officer", "support-officer", "pursuant-title",
        "t-series-tactical-droid-officer", "admiral-piett-commander", "swivel-mount-batteries",
        "ahsoka-tano-officer-rebel", "wullf-yularen-officer", "boosted-comms", "expanded-hangar-bay", "raymus-antilles-officer"
    };
    public static bool SupportsUpgrade(string id) => OwnedCards.Contains(id) || BoardingAbilityEffects.SupportsUpgrade(id) || CommandRevealEffects.SupportsUpgrade(id)
        || RepairAbilityEffects.SupportsUpgrade(id) || RaidAbilityEffects.SupportsUpgrade(id) || ReactionAbilityEffects.SupportsUpgrade(id);
    public static IEnumerable<GameActionOption> GetActions(GameContext game, string timing, int source, int target)
    { return CommandRevealEffects.GetActions(game, timing, source, target).Concat(RepairAbilityEffects.GetWindowActions(game, timing, source, target))
        .Concat(ReactionAbilityEffects.GetActions(game, timing, source, target)).Concat(AhsokaActions(game, timing, source)); }
    private static IEnumerable<GameActionOption> AhsokaActions(GameContext game, string timing, int source)
    {
        if (timing != "command-reveal" || !game.IsShip(source) ||
            game.ShipViews[source].Loadout.Find("ahsoka-tano-officer-rebel") is not { Active: true, Exhausted: false }) yield break;
        foreach (var gain in Enum.GetValues<ShipCommand>())
            if (AhsokaPayments(game, source, gain).Any())
                yield return new(new(GameActionKind.UseAbility, source, AbilityId: AhsokaActionPrefix + (int)gain),
                    "Ahsoka: choose a token to discard and gain " + gain);
    }
    private static IEnumerable<ShipCommand> AhsokaPayments(GameContext game, int source, ShipCommand gain)
    {
        if (!game.IsShip(source) || game.IsDestroyed(source) || game.IsReserved(source) ||
            !game.State.Ships[source].CanHaveCommandTokens || !CommandModule.CommandAllowed(game, source, gain)) yield break;
        var commands = game.State.Commands[source];
        foreach (var pay in Enum.GetValues<ShipCommand>())
            if (commands.HasToken(pay) && (TokenAssignments.CanAssign(game, source, gain) ||
                commands.TokenCount - 1 < commands.TokenCapacity &&
                (commands.AllowsDuplicateTokens || !commands.HasToken(gain) || pay == gain))) yield return pay;
    }
    public static bool TryApply(GameContext game, GameContinuation window, GameAction action)
    {
        if (BoardingAbilityEffects.TryApply(game, window, action)) return true;
        if (CommandRevealEffects.TryApply(game, window, action)) return true;
        if (RepairAbilityEffects.TryWindowApply(game, window, action)) return true;
        if (RaidAbilityEffects.TryApply(game, window, action)) return true;
        if (ReactionAbilityEffects.TryApply(game, window, action)) return true;
        if (!game.IsShip(action.Source)) return false;
        var ship = game.ShipViews[action.Source]; string id = action.AbilityId;
        var activation = game.Flow.ShipActivations[action.Source];
        if (id.StartsWith(UnitAbilityEngine.CommandChangeActionPrefix, StringComparison.Ordinal))
        {
            var values = id[UnitAbilityEngine.CommandChangeActionPrefix.Length..].Split('.', 2);
            if (values.Length != 2 || !int.TryParse(values[0], out int command) || !Enum.IsDefined((ShipCommand)command)
                || !ship.Loadout.HasActive(values[1]) || !ship.Commands.ReplaceTop((ShipCommand)command)) throw new ArgumentException("Cannot change command.");
            return true;
        }
        if (id is UnitAbilityEngine.RaymusAntillesAction or UnitAbilityEngine.TaskmasterGrintAction)
        {
            string cardId = id == UnitAbilityEngine.RaymusAntillesAction ? "raymus-antilles-officer" : "taskmaster-grint-officer";
            Require(activation.RevealedCommand is { } && TokenAssignments.CanAssign(game, action.Source, activation.RevealedCommand.Value), "No matching token capacity.");
            Require(ship.Loadout.Find(cardId)?.TryUseThisActivation() == true, "Matching-token officer already used.");
            TokenAssignments.TryAssign(game, action.Source, activation.RevealedCommand!.Value); return true;
        }
        if (id.StartsWith(UnitAbilityEngine.VeteranCaptainActionPrefix, StringComparison.Ordinal))
        {
            var command = CommandSuffix(id, UnitAbilityEngine.VeteranCaptainActionPrefix);
            Require(TokenAssignments.CanAssign(game, action.Source, command) && ship.Loadout.Find("veteran-captain-officer")?.Discard() == true, "Veteran Captain unavailable.");
            TokenAssignments.TryAssign(game, action.Source, command); return true;
        }
        if (id.StartsWith(UnitAbilityEngine.LiaisonActionPrefix, StringComparison.Ordinal))
        {
            var values = id[UnitAbilityEngine.LiaisonActionPrefix.Length..].Split('.');
            Require(values.Length == 3, "Invalid liaison choice.");
            var command = (ShipCommand)int.Parse(values[0]); var token = (ShipCommand)int.Parse(values[1]);
            Require(ship.Commands.HasToken(token) && ship.Loadout.Find(values[2])?.TryUseThisActivation() == true, "Liaison cannot spend this token.");
            SpendCommandToken(game, action.Source, token); activation.RevealedCommand = command; return true;
        }
        if (id.StartsWith(UnitAbilityEngine.DiscardTopCommandActionPrefix, StringComparison.Ordinal))
        {
            string cardId = id[UnitAbilityEngine.DiscardTopCommandActionPrefix.Length..]; var card = ship.Loadout.Find(cardId);
            Require(ship.Commands.Count > 0 && card is not null && (cardId == "skilled-first-officer" ? card.Discard() : card.TryUseThisActivation()), "Cannot discard top command.");
            ship.Commands.TryDiscardTop(out _); return true;
        }
        if (id.StartsWith(UnitAbilityEngine.WalexBlissexActionPrefix, StringComparison.Ordinal))
        {
            var token = (DefenseTokenType)int.Parse(id[UnitAbilityEngine.WalexBlissexActionPrefix.Length..]);
            Require(ship.State.GetDiscardedDefenseTokenCount(token) > 0 && ship.Loadout.Find("walex-blissex-officer")?.Discard() == true, "Walex cannot recover that token.");
            ship.State.TryRecoverDiscardedDefenseToken(token); return true;
        }
        if (id == UnitAbilityEngine.SupportOfficerAction)
        { Require(ship.Commands.Count > 0 && ship.Loadout.Find("support-officer")?.Discard() == true, "Support Officer unavailable."); ship.Commands.DiscardAllCommands(); return true; }
        if (id == UnitAbilityEngine.PursuantAction)
        {
            Require(activation.RevealedCommand != ShipCommand.Squadron && ship.Loadout.Find("pursuant-title")?.Discard() == true, "Pursuant unavailable.");
            GrantCommand(game, window, action.Source, ShipCommand.Squadron, CommandSpendMode.Dial, consumeDial: false); return true;
        }
        if (id.StartsWith(UnitAbilityEngine.TokenAsDialActionPrefix, StringComparison.Ordinal))
        {
            string cardId = id[UnitAbilityEngine.TokenAsDialActionPrefix.Length..];
            var card = cardId == "admiral-piett-commander" ? CommanderRules.FindFleetUpgrade(ship, cardId) : ship.Loadout.Find(cardId);
            Require(activation.RevealedCommand.HasValue && ship.Commands.HasToken(activation.RevealedCommand.Value) && card?.Exhaust() == true, "Token-as-dial unavailable.");
            var command = activation.RevealedCommand!.Value; SpendCommandToken(game, action.Source, command);
            GrantCommand(game, window, action.Source, command, CommandSpendMode.Dial, consumeDial: true, paidMode: CommandSpendMode.Token); return true;
        }
        if (id.StartsWith(AhsokaActionPrefix, StringComparison.Ordinal))
        {
            var command = CommandSuffix(id, AhsokaActionPrefix);
            Require(ship.Loadout.Find("ahsoka-tano-officer-rebel") is { Active: true, Exhausted: false } &&
                AhsokaPayments(game, action.Source, command).Any(), "Ahsoka unavailable.");
            Push(game, new("general.ahsoka-payment", action.Source, (int)command)); return true;
        }
        if (id.StartsWith("remaining.start-activation.interdictor.", StringComparison.Ordinal))
        {
            var card = ship.Loadout.Find(id["remaining.start-activation.interdictor.".Length..]);
            if (card is null || !card.Exhausted || ship.Loadout.Find("interdictor-title")?.Exhaust() != true) throw new ArgumentException("Interdictor cannot ready this card.");
            card.Ready(); return true;
        }
        if (id == UnitAbilityEngine.DisposableCapacitorsAction)
        {
            if (ship.Loadout.Find("disposable-capacitors")?.Discard() != true) throw new ArgumentException("Disposable Capacitors unavailable.");
            game.Flow.Flags[$"activation.disposable-capacitors.{game.State.Round}.{action.Source}"] = 1; return true;
        }
        if (id.StartsWith(UnitAbilityEngine.SwivelMountActionPrefix, StringComparison.Ordinal)
            && int.TryParse(id[UnitAbilityEngine.SwivelMountActionPrefix.Length..], out int zone))
        { var card = ship.Loadout.Find("swivel-mount-batteries"); if (card?.Exhaust() != true || !card.SetFocusZone((HullZone)zone)) throw new ArgumentException("Invalid focused zone."); return true; }
        if (id.StartsWith("remaining.start-activation.hand-of-justice.target.", StringComparison.Ordinal))
        {
            var body = id["remaining.start-activation.hand-of-justice.target.".Length..]; int split = body.LastIndexOf(".ready.", StringComparison.Ordinal);
            Require(split > 0 && game.Topology.EntityByRuntimeId.ContainsKey(body[..split]), "Unknown Hand of Justice target.");
            int target = game.Topology.EntityByRuntimeId[body[..split]]; var token = (DefenseTokenType)int.Parse(body[(split + 7)..]);
            Require(ship.Loadout.Find("hand-of-justice-title")?.Exhaust() == true && game.State.Ships[target].TryReadyDefenseToken(token), "Cannot ready target token."); return true;
        }
        if (id.StartsWith("remaining.start-activation.beast-of-burden.target.", StringComparison.Ordinal))
        {
            var body = id["remaining.start-activation.beast-of-burden.target.".Length..]; int split = body.LastIndexOf(".spend.", StringComparison.Ordinal);
            Require(split > 0 && game.Topology.EntityByRuntimeId.ContainsKey(body[..split]), "Unknown Beast of Burden target.");
            int target = game.Topology.EntityByRuntimeId[body[..split]];
            Push(game, new("general.beast", action.Source, target)); return true;
        }
        if (id == UnitAbilityEngine.PhylonQ7Action)
        { Push(game, new("general.phylon", action.Source)); return true; }
        return false;
    }
    public static GameDecision GetDecision(GameContext game, GameContinuation continuation)
    {
        if (BoardingAbilityEffects.Handles(continuation.Rule)) return BoardingAbilityEffects.GetDecision(game, continuation);
        if (CommandRevealEffects.Handles(continuation.Rule)) return CommandRevealEffects.GetDecision(game, continuation);
        if (RepairAbilityEffects.Handles(continuation.Rule)) return RepairAbilityEffects.GetDecision(game, continuation);
        if (RaidAbilityEffects.Handles(continuation.Rule)) return RaidAbilityEffects.GetDecision(game, continuation);
        if (ReactionAbilityEffects.Handles(continuation.Rule)) return ReactionAbilityEffects.GetDecision(game, continuation);
        var options = ChoiceOptions(game, continuation).ToArray();
        return new(continuation.Rule == "general.command-return" ? DecisionKind.Forced : DecisionKind.Player,
            continuation.Rule == "general.command-return" ? 0 : game.State.Owners[continuation.Source], GamePhase.AbilityChoice, options);
    }
    private static IEnumerable<GameActionOption> ChoiceOptions(GameContext game, GameContinuation frame)
    {
        if (frame.Rule == "general.command-return") { yield return new(new(GameActionKind.Pass, frame.Source), "Finish granted command"); yield break; }
        if (frame.Rule == "general.ahsoka-payment")
        {
            if (game.ShipViews[frame.Source].Loadout.Find("ahsoka-tano-officer-rebel") is not { Active: true, Exhausted: false }) yield break;
            foreach (var pay in AhsokaPayments(game, frame.Source, (ShipCommand)frame.Target))
                yield return new(new(GameActionKind.ChooseCommand, frame.Source, Command: pay, AbilityId: "general.ahsoka.discard"),
                    "Discard " + pay + " token; gain " + (ShipCommand)frame.Target);
            yield break;
        }
        if (frame.Rule == "general.phylon")
        {
            var source = game.ShipViews[frame.Source];
            foreach (var target in game.SceneShipViews.Where(target => target.OwnerId != source.OwnerId && RemainingUpgradeRules.SizeRank(target) <= RemainingUpgradeRules.SizeRank(source)
                && RaidRules.FootprintDistance(source, target) <= UnitAbilityEngine.PhylonQ7MaximumDistance))
                yield return new(new(GameActionKind.ChooseEntity, frame.Source, target.Index, AbilityId: "general.phylon.target"), "Tractor " + target.Definition.DisplayName);
            yield return new(new(GameActionKind.Pass, frame.Source), "Cancel tractor targeting"); yield break;
        }
        if (frame.Rule == "general.beast")
        {
            var source = game.State.Ships[frame.Source]; var target = game.State.Ships[frame.Target];
            if (frame.Stage == 0)
            {
                int spent = Enum.GetValues<DefenseTokenType>().Sum(token => target.GetAvailableDefenseTokenCount(token) - target.GetReadyDefenseTokenCount(token));
                if (frame.Value < spent)
                    foreach (var token in Enum.GetValues<DefenseTokenType>())
                        foreach (var instance in source.GetDefenseTokenInstances(token).Where(source.IsDefenseTokenInstanceAvailable))
                            yield return new(new(GameActionKind.ChooseToken, frame.Source, frame.Target, Token: token, Index: instance.Slot,
                                Amount: instance.Exhausted ? 1 : 0, AbilityId: "general.beast.spend"), "Spend " + token + " token " + instance.Slot);
                yield return new(new(GameActionKind.Pass, frame.Source, frame.Target), frame.Value == 0 ? "Cancel Beast of Burden" : "Choose tokens to ready");
            }
            else
                foreach (var token in Enum.GetValues<DefenseTokenType>().Where(token => target.GetAvailableDefenseTokenCount(token) > target.GetReadyDefenseTokenCount(token)))
                    yield return new(new(GameActionKind.ChooseToken, frame.Source, frame.Target, Token: token, AbilityId: "general.beast.ready"), "Ready " + token);
            yield break;
        }
        throw new NotSupportedException("Unknown ability continuation " + frame.Rule);
    }
    public static void Apply(GameContext game, GameContinuation frame, GameAction action)
    {
        if (BoardingAbilityEffects.Handles(frame.Rule)) { BoardingAbilityEffects.Apply(game, frame, action); return; }
        if (CommandRevealEffects.Handles(frame.Rule)) { CommandRevealEffects.Apply(game, frame, action); return; }
        if (RepairAbilityEffects.Handles(frame.Rule)) { RepairAbilityEffects.Apply(game, frame, action); return; }
        if (RaidAbilityEffects.Handles(frame.Rule)) { RaidAbilityEffects.Apply(game, frame, action); return; }
        if (ReactionAbilityEffects.Handles(frame.Rule)) { ReactionAbilityEffects.Apply(game, frame, action); return; }
        Require(ChoiceOptions(game, frame).Any(x => GameSession.Matches(x, action)), "Illegal ability choice.");
        if (frame.Rule == "general.ahsoka-payment")
        {
            Require(game.ShipViews[frame.Source].Loadout.Find("ahsoka-tano-officer-rebel")?.Exhaust() == true &&
                game.State.Commands[frame.Source].TrySpendToken(action.Command), "Ahsoka's selected discard is unavailable.");
            // A discard does not trigger Wullf Yularen's 'spend' response. Removing
            // this exact token also frees capacity before receiving the replacement.
            Require(TokenAssignments.TryAssign(game, frame.Source, (ShipCommand)frame.Target), "Ahsoka cannot gain the replacement token.");
            EngineEffects.ReturnFromChoice(game); return;
        }
        if (frame.Rule == "general.command-return")
        {
            var activation = game.Flow.ShipActivations[frame.Source];
            activation.RevealedCommand = frame.Target >= 0 ? (ShipCommand)frame.Target : null;
            activation.RevealedDialAvailable = frame.Value != 0;
            EngineEffects.ReturnFromChoice(game); return;
        }
        if (frame.Rule == "general.phylon")
        {
            if (action.Kind == GameActionKind.Pass) { EngineEffects.ReturnFromChoice(game); return; }
            var target = game.ShipViews[action.Target];
            Require(game.ShipViews[frame.Source].Loadout.Find("phylon-q7-tractor-beams")?.Exhaust() == true, "Tractor beams unavailable.");
            int previous = target.State.CurrentSpeed;
            if (target.Commands.HasToken(ShipCommand.Navigate)) SpendCommandToken(game, action.Target, ShipCommand.Navigate);
            else if (previous > 1) target.State.SetSpeed(previous - 1, target.Definition.MaximumSpeed);
            EngineEffects.ReturnFromChoice(game);
            RoundModule.ObserveObjectives(game);
            if (target.State.DamageWhenChangingSpeed(previous) > 0)
                CombatModule.QueueDamage(game, action.Target, 1, faceup: false, ignoreShields: false, returnPhase: GamePhase.AbilityChoice, rule: "phylon.thruster-fissure");
            return;
        }
        if (frame.Rule == "general.beast")
        {
            if (frame.Stage == 0 && action.Kind == GameActionKind.Pass)
            {
                if (frame.Value == 0) { EngineEffects.ReturnFromChoice(game); return; }
                Require(game.ShipViews[frame.Source].Loadout.Find("beast-of-burden-title")?.Exhaust() == true, "Beast of Burden unavailable.");
                game.Flow.Continuations[^1] = frame with { Stage = 1 }; return;
            }
            if (frame.Stage == 0)
            {
                Require(game.State.Ships[frame.Source].TryResolveDefenseToken(new DefenseTokenInstance(action.Token, action.Index, action.Amount != 0), out _), "Cannot spend this physical token.");
                game.Flow.Continuations[^1] = frame with { Value = frame.Value + 1 }; return;
            }
            Require(game.State.Ships[frame.Target].TryReadyDefenseToken(action.Token), "No exhausted target token.");
            if (frame.Value <= 1) EngineEffects.ReturnFromChoice(game); else game.Flow.Continuations[^1] = frame with { Value = frame.Value - 1 };
            return;
        }
    }
    private static void Push(GameContext game, GameContinuation frame) => EngineEffects.PushChoice(game, frame, ChoiceOptions(game, frame));
    internal static void GrantCommand(GameContext game, GameContinuation window, int source, ShipCommand command, CommandSpendMode mode, bool consumeDial, CommandSpendMode? paidMode = null)
    {
        var activation = game.Flow.ShipActivations[source];
        var original = activation.RevealedCommand;
        bool available = activation.RevealedDialAvailable && !consumeDial;
        if (consumeDial)
        {
            // Consuming the native dial still leaves Bail, Tarkin and Thrawn dials
            // to be prepared after the remaining command-reveal timing choices.
            activation.NativeDialConsumedBeforePrelude = true;
        }
        Push(game, new("general.command-return", source, original.HasValue ? (int)original.Value : -1, Value: available ? 1 : 0));
        CommandModule.ResolveGrantedCommand(game, command, mode, GamePhase.AbilityChoice, consumeDial, paidMode);
    }
    public static bool SpendCommandToken(GameContext game, int entity, ShipCommand command)
    {
        if (!game.State.Commands[entity].TrySpendToken(command)) return false;
        var wullf = game.UpgradesFor(entity).FirstOrDefault(x => x.Active && !x.Exhausted && x.Definition.id == "wullf-yularen-officer");
        if (wullf is not null && TokenAssignments.CanAssign(game, entity, command)) { wullf.Exhaust(); TokenAssignments.TryAssign(game, entity, command); }
        return true;
    }
    private static ShipCommand CommandSuffix(string id, string prefix)
    { int number = int.Parse(id[prefix.Length..]); Require(Enum.IsDefined((ShipCommand)number), "Invalid command."); return (ShipCommand)number; }
    private static void Require(bool condition, string message) { if (!condition) throw new ArgumentException(message); }
}
