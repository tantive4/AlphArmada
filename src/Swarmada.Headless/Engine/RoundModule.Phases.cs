using System.Numerics;
using Armada.Core;
using Armada.Data;
using Armada.Fleet;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public sealed partial class RoundModule
{
    private static GameDecision RoundPhaseDecision(GameContext game)
    {
        using var scope = new GameScope(game);
        var phase = game.Flow.Phase;
        PreparePhaseChoices(game, phase);
        while (CurrentRoundChoice(game) is { } choice)
        {
            if (choice.Rule == "round.window")
            {
                RemoveRoundChoice(game);
                if (EngineEffects.BeginWindow(game, choice.Text, -1, resumePhase: phase))
                    return new(DecisionKind.Forced, 0, game.Flow.Phase, []);
                continue;
            }
            if (choice.Rule == "round.refresh")
            {
                RemoveRoundChoice(game);
                foreach (var ship in game.ShipViews.Where(x => !x.IsDestroyed && !game.IsReserved(x.Index)))
                { ship.State.RefreshDefenseTokens(); ship.Loadout.ReadyForStatusPhase(ship.Commands); }
                foreach (string key in game.Flow.Flags.Keys.Where(x => x.StartsWith("round.early-warning.", StringComparison.Ordinal)).ToArray()) game.Flow.Flags.Remove(key);
                continue;
            }
            if (game.IsDestroyed(choice.Source) || game.IsReserved(choice.Source)) { RemoveRoundChoice(game); continue; }
            var decision = RoundChoiceDecision(game, choice);
            if (decision.Actions.Count == 0) { RemoveRoundChoice(game); continue; }
            return decision;
        }
        if (phase == GamePhase.StartRound)
        { game.Flow.Phase = GamePhase.CommandPlanning; game.Flow.PlanningShip = 0; }
        else if (phase == GamePhase.StartShipPhase)
        { game.Flow.Phase = GamePhase.SelectShip; game.Flow.CurrentPlayer = game.Topology.FirstPlayer; }
        else CompleteStatus(game);
        return new(DecisionKind.Forced, 0, game.Flow.Phase, []);
    }

    private static void PreparePhaseChoices(GameContext game, GamePhase phase)
    {
        string key = "round.prepared." + phase;
        if (game.Flow.Flags.GetValueOrDefault(key) == game.State.Round) return;
        game.Flow.Flags[key] = game.State.Round;
        var cards = CurrentCards(game).ToArray();
        void Add(string rule, int source, int upgrade, int target = -1, long value = 0, string text = "") =>
            game.Flow.Continuations.Add(new(rule, source, upgrade, Value: value, Text: text, Indices: target < 0 ? null : [target]));
        if (phase == GamePhase.StartRound)
        {
            RemainingUpgradeRules.ResetRoundEffects(); FleetCommandRules.ResetRound();
            foreach (var entry in cards.Where(x => x.Card.Definition.id is "admiral-raddus-commander" or "profundity-title"))
                if (ReservedFor(game, entry.Index).Any()) Add("round.reserve", entry.Source, entry.Index, text: entry.Card.Definition.id);
            game.Flow.Continuations.Add(new("round.window", Text: "start-round")); return;
        }
        if (phase == GamePhase.Status)
        {
            foreach (var entry in cards.Where(x => x.Card.Definition.id == "redundant-shields")) Add("round.redundant", entry.Source, entry.Index);
            foreach (var commander in cards.Where(x => x.Card.Definition.id == "admiral-konstantine-commander"))
                foreach (var enemy in LiveShips(game).Where(x => x.Owner != game.State.Owners[commander.Source]))
                {
                    var supporting = LiveShips(game).Where(x => x.Owner == game.State.Owners[commander.Source] &&
                        !RemainingUpgradeRules.IsFlotilla(game.ShipViews[x.EntityIndex]) &&
                        Geometry2D.Distance(game.ShipBase(x.EntityIndex), game.ShipBase(enemy.EntityIndex)) <= TabletopMeasurements.LongRange).ToArray();
                    if (supporting.Length >= 2 && supporting.Any(x => RemainingUpgradeRules.SizeRank(game.ShipViews[x.EntityIndex]) >= 2))
                        Add("round.konstantine", commander.Source, commander.Index, enemy.EntityIndex);
                }
            game.Flow.Continuations.Add(new("round.refresh"));
            game.Flow.Continuations.Add(new("round.window", Text: "status")); return;
        }
        game.Flow.Continuations.Add(new("round.window", Text: "end-command-phase"));
        game.Flow.Continuations.Add(new("round.window", Text: "start-ship-phase"));
        foreach (var entry in cards)
        {
            if (entry.Card.Definition.id == "garm-bel-iblis-commander" && entry.Card.IsScheduledFor(game.State.Round))
                foreach (var ship in LiveShips(game).Where(x => x.Owner == game.State.Owners[entry.Source] && game.State.Ships[x.EntityIndex].CanHaveCommandTokens))
                    Add("round.garm", entry.Source, entry.Index, ship.EntityIndex, game.State.Commands[ship.EntityIndex].Capacity);
            if (entry.Card.Definition.id == "general-tagge-commander" && entry.Card.IsScheduledFor(game.State.Round))
                foreach (var ship in LiveShips(game).Where(x => x.Owner == game.State.Owners[entry.Source])) Add("round.tagge", entry.Source, entry.Index, ship.EntityIndex);
        }
        foreach (var entry in cards.Where(x => x.Card.Definition.id == "sovereign-title")) Add("round.sovereign", entry.Source, entry.Index, value: 3);
        foreach (var entry in cards.Where(x => x.Card.Definition.id is "admiral-tarkin-commander" or "admiral-trench-commander" or "general-draven-commander" or "grand-admiral-thrawn-commander" or "count-dooku-commander"))
            if (entry.Card.TypedCommandTokens.Count > 0) Add("round.stored", entry.Source, entry.Index, text: entry.Card.Definition.id);
        foreach (var entry in cards.Where(x => x.Card.Definition.id == "grand-moff-tarkin-commander")) Add("round.tarkin", entry.Source, entry.Index);
        foreach (var entry in cards.Where(x => x.Card.Definition.id == "emperor-palpatine-commander")) Add("round.palpatine", entry.Source, entry.Index);
        foreach (var entry in cards.Where(x => x.Card.Definition.id == "hondo-ohnaka-officer")) Add("round.hondo", entry.Source, entry.Index);
        foreach (var entry in cards.Where(x => FleetCommandRules.IsSupported(x.Card.EffectiveDefinition.id))) Add("round.fleet-command", entry.Source, entry.Index, text: entry.Card.EffectiveDefinition.id);
        foreach (var entry in cards.Where(x => x.Card.Definition.id is "early-warning-system" or "reinforced-blast-doors"))
            Add(entry.Card.Definition.id == "early-warning-system" ? "round.early-warning" : "round.blast-doors", entry.Source, entry.Index);
        game.Flow.Continuations.Add(new("round.window", Text: "after-start-ship-phase"));
    }

    internal static GameDecision RoundChoiceDecision(GameContext game, GameContinuation frame)
    {
        using var scope = new GameScope(game);
        var options = new List<GameActionOption>(); int source = frame.Source, owner = game.State.Owners[source];
        int recipient = frame.Indices is { Length: > 0 } ? frame.Indices[0] : -1;
        var card = game.State.Upgrades[frame.Target];
        void Add(GameAction action, string label, bool position = false, bool yaw = false) => options.Add(new(action, label, position, yaw));
        switch (frame.Rule)
        {
            case "round.reserve":
                foreach (int entity in ReservedFor(game, frame.Target).Where(game.IsShip))
                    Add(new(GameActionKind.PlaceReservedUnit, source, entity, Index: frame.Target, Speed: 1, AbilityId: frame.Rule), "Deploy set-aside ship", true, true);
                break;
            case "round.profundity-transfer":
                string slot = frame.Stage == 0 ? "commander" : "officer";
                foreach (var upgrade in game.Topology.Upgrades.Where(x => game.State.Registers[game.Topology.Registers.UpgradeOwner(x.UpgradeIndex)] == source + 1))
                    if (game.State.Upgrades[upgrade.UpgradeIndex].Definition.slot == slot && CanTransferProfundity(game, recipient, game.State.Upgrades[upgrade.UpgradeIndex]))
                        Add(new(GameActionKind.UseAbility, source, recipient, Index: upgrade.UpgradeIndex, AbilityId: frame.Rule), "Transfer " + game.State.Upgrades[upgrade.UpgradeIndex].Definition.name);
                break;
            case "round.garm":
                if (frame.Value > 0)
                    foreach (var command in Enum.GetValues<ShipCommand>().Where(x => CanGain(game, recipient, x)))
                        Add(new(GameActionKind.ChooseCommand, source, recipient, Command: command, Index: frame.Target, AbilityId: frame.Rule), "Gain " + command + " token");
                break;
            case "round.tagge":
                foreach (var token in Enum.GetValues<DefenseTokenType>().Where(x => game.State.Ships[recipient].GetDiscardedDefenseTokenCount(x) > 0))
                    Add(new(GameActionKind.ChooseToken, source, recipient, Token: token, Index: frame.Target, AbilityId: frame.Rule), "Recover discarded " + token);
                break;
            case "round.sovereign":
                if (frame.Stage > 0)
                    foreach (var command in Enum.GetValues<ShipCommand>().Where(x => CanGain(game, frame.Stage - 1, x)))
                        Add(new(GameActionKind.ChooseCommand, source, frame.Stage - 1, Command: command, Index: frame.Target, AbilityId: "round.sovereign-gain"), "Gain " + command + " token");
                else if (frame.Value > 0)
                    foreach (var ship in LiveShips(game).Where(x => x.Owner == owner && !(frame.Indices ?? []).Contains(x.EntityIndex) &&
                        Geometry2D.Distance(game.ShipBase(source), game.ShipBase(x.EntityIndex)) <= TabletopMeasurements.LongRange))
                        foreach (var command in game.State.Commands[ship.EntityIndex].Tokens)
                            Add(new(GameActionKind.ChooseCommand, source, ship.EntityIndex, Command: command, Index: frame.Target, AbilityId: "round.sovereign-spend"), "Convert " + command + " on " + ship.CatalogEntry.Definition.DisplayName);
                break;
            case "round.stored":
                if (frame.Stage > 0)
                {
                    Add(new(GameActionKind.UseAbility, source, Index: frame.Target, Command: (ShipCommand)(frame.Stage - 1), AbilityId: "round.tarkin-dial"), "Store additional matching dial");
                    Add(new(GameActionKind.UseAbility, source, Index: frame.Target, Command: (ShipCommand)(frame.Stage - 1), AbilityId: "round.tarkin-fleet-tokens"), "Issue matching tokens to fleet");
                }
                else foreach (var command in Enum.GetValues<ShipCommand>().Where(x => card.GetTypedCommandTokenCount(x) > 0))
                    Add(new(GameActionKind.ChooseCommand, source, Index: frame.Target, Command: command, AbilityId: frame.Rule), "Reveal stored " + command);
                break;
            case "round.tarkin":
                foreach (var command in Enum.GetValues<ShipCommand>().Where(command => LiveShips(game).Any(x => x.Owner == owner && CanGain(game, x.EntityIndex, command))))
                    Add(new(GameActionKind.ChooseCommand, source, Command: command, Index: frame.Target, AbilityId: frame.Rule), "Issue " + command + " tokens to fleet");
                break;
            case "round.palpatine":
                foreach (var token in RemainingUpgradeRules.GetRemainingPalpatineTokens(owner).Order())
                    Add(new(GameActionKind.ChooseToken, source, Token: token, Index: frame.Target, AbilityId: frame.Rule), "Discard Palpatine's " + token);
                break;
            case "round.hondo": HondoOptions(game, frame, options, out owner); break;
            case "round.fleet-command":
                var required = FleetCommandRules.RequiredToken(frame.Text);
                if (!game.HasUpgrade(source, "flag-bridge") && !game.HasUpgrade(source, "liberator-title") && game.State.Commands[source].HasToken(required))
                    Add(new(GameActionKind.UseAbility, source, Command: required, Index: frame.Target, AbilityId: "round.fleet-command-token"), "Spend matching command token");
                Add(new(GameActionKind.UseAbility, source, Index: frame.Target, AbilityId: "round.fleet-command-discard"), "Discard fleet-command card");
                break;
            case "round.early-warning":
                foreach (var zone in game.State.Ships[source].HullZones) Add(new(GameActionKind.ChooseZone, source, ToZone: zone, Index: frame.Target, AbilityId: frame.Rule), "Mark " + zone + " with chaff");
                break;
            case "round.blast-doors":
                if (game.State.Ships[source].FacedownDamageCardCount > 0) Add(new(GameActionKind.UseAbility, source, Index: frame.Target, AbilityId: frame.Rule), "Discard Blast Doors and repair up to three facedown cards");
                break;
            case "round.redundant":
                foreach (var zone in game.State.Ships[source].HullZones)
                    if (game.State.Ships[source].GetShields(zone) < UnitAbilityEngine.GetEffectiveShieldMaximum(game.ShipViews[source], zone) &&
                        !(game.State.Ships[source].GetShields(zone) == 0 && game.State.Ships[source].HasFaceupCritical(CriticalDamageCard.CapacitorFailure)))
                        Add(new(GameActionKind.RecoverShield, source, ToZone: zone, Index: frame.Target, AbilityId: frame.Rule), "Recover one " + zone + " shield");
                break;
            case "round.konstantine":
                if (game.State.Commands[recipient].HasToken(ShipCommand.Navigate)) Add(new(GameActionKind.UseAbility, source, recipient, Index: frame.Target, AbilityId: "round.konstantine-token"), "Discard enemy Navigate token");
                foreach (int delta in new[] { -1, 1 })
                { int speed = game.State.Ships[recipient].CurrentSpeed + delta; if (speed >= 0 && speed <= game.Topology.Ships[recipient].CatalogEntry.Definition.MaximumSpeed) Add(new(GameActionKind.ChangeSpeed, source, recipient, Speed: speed, Index: frame.Target, AbilityId: frame.Rule), "Set enemy speed " + speed); }
                break;
            default: throw new NotSupportedException("Unrecognized round continuation " + frame.Rule);
        }
        if (!(frame.Rule == "round.sovereign" && frame.Stage > 0) && !(frame.Rule == "round.hondo" && frame.Stage > 0))
            Add(new(GameActionKind.Pass, source, Index: frame.Target, AbilityId: "round.skip"), "Skip this optional effect");
        return new(DecisionKind.Player, owner, game.Flow.Phase, options.AsReadOnly());
    }

    internal static void ApplyRoundChoice(GameContext game, GameContinuation frame, GameAction action)
    {
        using var scope = new GameScope(game);
        if (!RoundChoiceDecision(game, frame).Actions.Any(x => RoundMatches(x, action))) throw new ArgumentException("Illegal round choice.");
        var card = game.State.Upgrades[frame.Target]; int owner = game.State.Owners[frame.Source];
        if (action.Kind == GameActionKind.Pass)
        {
            if (frame.Rule == "round.profundity-transfer" && frame.Stage == 0) { UpdateRoundChoice(game, frame with { Stage = 1 }); return; }
            RemoveRoundChoice(game); return;
        }
        switch (frame.Rule)
        {
            case "round.reserve": ReleaseReservedShip(game, frame, action); return;
            case "round.profundity-transfer":
                game.State.Registers[game.Topology.Registers.UpgradeOwner(action.Index)] = action.Target + 1;
                if (frame.Stage == 0) { UpdateRoundChoice(game, frame with { Stage = 1 }); return; } break;
            case "round.garm":
                TokenAssignments.TryAssign(game, action.Target, action.Command);
                if (frame.Value > 1) { UpdateRoundChoice(game, frame with { Value = frame.Value - 1 }); return; } break;
            case "round.tagge": game.State.Ships[action.Target].TryRecoverDiscardedDefenseToken(action.Token); break;
            case "round.sovereign":
                if (frame.Stage == 0) { game.State.Commands[action.Target].TrySpendToken(action.Command); UpdateRoundChoice(game, frame with { Stage = action.Target + 1 }); return; }
                TokenAssignments.TryAssign(game, action.Target, action.Command);
                if (frame.Value > 1) { UpdateRoundChoice(game, frame with { Stage = 0, Value = frame.Value - 1, Indices = (frame.Indices ?? []).Append(action.Target).ToArray() }); return; } break;
            case "round.stored":
                if (card.Definition.id == "admiral-tarkin-commander" && frame.Stage == 0) { UpdateRoundChoice(game, frame with { Stage = (int)action.Command + 1 }); return; }
                if (!card.TrySpendTypedCommandToken(action.Command)) throw new ArgumentException("Stored command is unavailable.");
                switch (card.Definition.id)
                {
                    case "admiral-tarkin-commander":
                        if (action.AbilityId == "round.tarkin-dial") RemainingUpgradeRules.ActivateAdmiralTarkinDial(owner, action.Command);
                        else IssueFleetTokens(game, owner, action.Command); break;
                    case "admiral-trench-commander": RemainingUpgradeRules.ActivateTrench(owner, action.Command); break;
                    case "general-draven-commander": RemainingUpgradeRules.ActivateDraven(owner, action.Command); break;
                    case "grand-admiral-thrawn-commander": RemainingUpgradeRules.ActivateThrawn(owner, action.Command); break;
                    case "count-dooku-commander": foreach (var ship in LiveShips(game).Where(x => x.Owner != owner)) game.State.Commands[ship.EntityIndex].TryGainRaidToken(action.Command); break;
                } break;
            case "round.tarkin": IssueFleetTokens(game, owner, action.Command); break;
            case "round.palpatine": RemainingUpgradeRules.TryActivatePalpatineToken(owner, action.Token); break;
            case "round.hondo": ApplyHondo(game, frame, action); return;
            case "round.fleet-command":
                if (action.AbilityId == "round.fleet-command-token") game.State.Commands[frame.Source].TrySpendToken(action.Command); else card.Discard();
                FleetCommandRules.Activate(owner, frame.Text); break;
            case "round.early-warning": game.Flow.Flags[$"round.early-warning.{frame.Source}"] = (int)action.ToZone + 1; card.SetFocusZone(action.ToZone); break;
            case "round.blast-doors":
                card.Discard(); for (int i = 0; i < 3 && game.State.Ships[frame.Source].TryRepairFacedownDamage(out var repaired); i++) game.State.Decks[owner - 1].Discard(repaired); break;
            case "round.redundant": game.State.Ships[frame.Source].RestoreShield(action.ToZone, 1, UnitAbilityEngine.HasAuxiliaryShieldsTeam(game.ShipViews[frame.Source])); break;
            case "round.konstantine":
                if (action.AbilityId == "round.konstantine-token") game.State.Commands[action.Target].TrySpendToken(ShipCommand.Navigate);
                else game.State.Ships[action.Target].SetSpeed(action.Speed, game.Topology.Ships[action.Target].CatalogEntry.Definition.MaximumSpeed); break;
        }
        RemoveRoundChoice(game);
    }

    private static void ReleaseReservedShip(GameContext game, GameContinuation frame, GameAction action)
    {
        if (action.Position is null || !game.IsShip(action.Target)) throw new ArgumentException("Reserved ship placement needs a position.");
        var pose = new Pose(action.Position.Value, action.YawDegrees);
        var shape = new OrientedRectangle(pose, game.Topology.Ships[action.Target].FootprintMillimetres);
        var board = new OrientedRectangle(new Pose(Vector2.Zero), game.Topology.BoardSizeMillimetres);
        Span<Vector2> corners = stackalloc Vector2[4]; shape.CopyCorners(corners);
        foreach (var corner in corners) if (!board.Contains(corner)) throw new ArgumentException("Reserved ship must remain on the board.");
        if (Geometry2D.Distance(shape, game.ShipBase(frame.Source)) > TabletopMeasurements.DistanceOne + .0115f) throw new ArgumentException("Reserved ship must be at distance one.");
        foreach (var ship in LiveShips(game)) if (ship.EntityIndex != action.Target && Geometry2D.Overlaps(shape, game.ShipBase(ship.EntityIndex))) throw new ArgumentException("Reserved ship cannot overlap another ship.");
        var overlapping = game.Topology.Squadrons.Where(x => !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex) && Geometry2D.Overlaps(game.SquadronBase(x.EntityIndex), shape)).Select(x => x.EntityIndex).ToArray();
        if (frame.Text != "profundity-title" && overlapping.Length > 0) throw new ArgumentException("Raddus ship cannot overlap squadrons.");
        int maximum = game.Topology.Ships[action.Target].CatalogEntry.Definition.ChartedMaximumSpeed;
        if (action.Speed < 1 || action.Speed > maximum) throw new ArgumentException("Reserved ship must enter at a legal nonzero speed.");
        game.State.Poses[action.Target] = pose;
        int enteringSpeed = game.State.SpatialObjects.Any(x => x.Active && x.Kind == SetupModule.GravWellKind &&
            Vector2.Distance(x.Pose.Position, shape.ClosestPoint(x.Pose.Position)) <= TabletopMeasurements.MediumRange) ? 0 : action.Speed;
        game.State.Ships[action.Target].SetSpeed(enteringSpeed, maximum);
        game.State.Registers[game.Topology.Registers.EntityReserved(action.Target)] = 0;
        game.State.Registers[game.Topology.Registers[$"entity.{action.Target}.placed"]] = 1;
        game.State.Upgrades[frame.Target].RemoveStoredUnit(game.Topology.Ships[action.Target].RuntimeId);
        if (frame.Text == "admiral-raddus-commander") game.Flow.Flags[$"raddus.block-first.{action.Target}"] = 1;
        if (frame.Text == "profundity-title")
        {
            // The same mutable card instances move via snapshot-owned ownership registers.
            UpdateRoundChoice(game, frame with { Rule = "round.profundity-transfer", Stage = 0, Indices = [action.Target] });
            MovementModule.ProcessDisplacements(game, action.Target);
            if (game.Flow.PendingDisplacements.Length > 0)
            {
                game.Flow.Flags["round.reserve-displacement"] = 1;
                game.Flow.Phase = GamePhase.SquadronDisplacement;
            }
            return;
        }
        RemoveRoundChoice(game);
    }

    private static bool CanTransferProfundity(GameContext game, int target, UpgradeCardRuntimeState card)
    {
        var entry = game.Topology.Ships[target].CatalogEntry; var definition = card.Definition;
        if (!UpgradeCatalog.IsFactionLegal(definition, entry.Definition.Faction) ||
            (!string.IsNullOrWhiteSpace(definition.boundShipType) && !string.Equals(definition.boundShipType, entry.Definition.ShipType, StringComparison.OrdinalIgnoreCase)) ||
            (definition.requiredSizes is { Length: > 0 } && !definition.requiredSizes.Contains(entry.Metadata.Size, StringComparer.OrdinalIgnoreCase)) ||
            (definition.requiredTraits is { Length: > 0 } && !definition.requiredTraits.Any(x => entry.Metadata.Traits.Contains(x, StringComparer.OrdinalIgnoreCase))) ||
            (definition.disallowedSlots is { Length: > 0 } && definition.disallowedSlots.Any(x => entry.Definition.UpgradeSlots.Contains(x, StringComparer.OrdinalIgnoreCase))) ||
            (definition.slot == "commander" && entry.Metadata.IsFlotilla == true)) return false;
        var existing = game.UpgradesFor(target);
        if (definition.flagshipOnly && definition.slot != "commander" && !existing.Any(x => x.Definition.slot == "commander")) return false;
        var ship = FleetContentAdapter.Describe(entry.Definition, entry.Metadata);
        var upgrades = existing.Select(x => OfficialContentPolicy.Describe(x.Definition)).Append(OfficialContentPolicy.Describe(definition)).ToArray();
        return UpgradeSlotResolver.Resolve(ship, upgrades, "profundity-target").IsValid;
    }

    private static void HondoOptions(GameContext game, GameContinuation frame, List<GameActionOption> options, out int owner)
    {
        owner = game.State.Owners[frame.Source];
        if (frame.Stage == 0) { options.Add(new(new(GameActionKind.UseAbility, frame.Source, Index: frame.Target, AbilityId: "round.hondo-use"), "Discard Hondo and exchange tokens")); return; }
        if (frame.Stage >= 3) owner = 3 - owner;
        int[] history = frame.Indices ?? [];
        var usedTypes = history.Where((_, index) => index % 2 == 0).ToArray();
        var usedShipsThisStep = history.Where((_, index) => index % 2 == 1 && (frame.Stage < 3 || index >= 5)).ToArray();
        foreach (var command in Enum.GetValues<ShipCommand>().Where(x => !usedTypes.Contains((int)x)))
            foreach (var ship in LiveShips(game).Where(x => !usedShipsThisStep.Contains(x.EntityIndex) && CanGain(game, x.EntityIndex, command)))
                options.Add(new(new(GameActionKind.ChooseCommand, frame.Source, ship.EntityIndex, Command: command, Index: frame.Target, AbilityId: "round.hondo-place"), "Give " + command + " to " + ship.CatalogEntry.Definition.DisplayName));
    }
    private static void ApplyHondo(GameContext game, GameContinuation frame, GameAction action)
    {
        if (frame.Stage == 0) { game.State.Upgrades[frame.Target].Discard(); UpdateRoundChoice(game, frame with { Stage = 1, Indices = [] }); return; }
        TokenAssignments.TryAssign(game, action.Target, action.Command);
        if (frame.Stage >= 4) { RemoveRoundChoice(game); return; }
        UpdateRoundChoice(game, frame with { Stage = frame.Stage + 1, Indices = (frame.Indices ?? []).Concat([(int)action.Command, action.Target]).ToArray() });
    }

    private static void CompleteStatus(GameContext game)
    {
        if (game.Flow.Flags.GetValueOrDefault("status.finalized") != game.State.Round)
        {
        game.Flow.Flags["status.finalized"] = game.State.Round;
        foreach (var squadron in game.State.Squadrons) { squadron.Refresh(); squadron.FinalizeDeferredDestruction(); }
        foreach (string key in game.Flow.Flags.Keys.Where(x => x.StartsWith("rieekan.ship.", StringComparison.Ordinal) ||
            x.StartsWith("rieekan.squadron.", StringComparison.Ordinal)).ToArray())
        {
            int entity = int.Parse(key.AsSpan(key.LastIndexOf('.') + 1), System.Globalization.CultureInfo.InvariantCulture);
            if (game.IsShip(entity)) game.State.Ships[entity].DestroyWithoutDamage();
            else game.State.Squadrons[game.SquadronIndex(entity)].DestroyWithoutDamage();
            // The destruction was already observed when Rieekan deferred it. Intervening
            // repair must not generate a second Grievous/Reserve/death-response event.
            if (entity < game.Flow.DeathRecorded.Length) game.Flow.DeathRecorded[entity] = true;
            game.Flow.Flags.Remove(key);
        }
        ObserveObjectives(game);
        if (game.State.Objective is not null)
        {
            var ships = ObjectiveShips(game);
            switch (game.State.Objective)
            { case EscortVipObjective e: e.ResolveEndOfRound(game.State.Round, ships); break; case IntelRetrievalObjective i: i.ResolveEndOfRound(game.State.Round, ships); break; case SeizeStationObjective s: s.ResolveEndOfRound(game.State.Round, ships); break; }
        }
        }
        OfficialObjectives.Observe(game);
        if (game.Flow.Phase != GamePhase.Status) return;
        if (DamageEvents.HasPending(game)) return;
        if (OfficialObjectives.State(game) is not null && game.Flow.Flags.GetValueOrDefault("status.objective-end-round") != game.State.Round)
        {
            game.Flow.Flags["status.objective-end-round"] = game.State.Round;
            if (EngineEffects.BeginWindow(game, "end-round", -1, resumePhase: GamePhase.Status)) return;
        }
        if (TournamentRules.Applies(game) && TournamentRules.ResolveEndOfRound(game)) return;
        if (CheckTerminal(game)) return;
        Array.Clear(game.State.Activated); game.Flow.ActiveShip = -1; game.Flow.ActiveSquadron = -1;
        EngineEffects.CleanupRound(game, game.State.Round);
        OfficialAssaultObjectives.CleanupRound(game);
        game.State.Round++; game.Flow.CurrentPlayer = game.Topology.FirstPlayer; game.Flow.PlanningShip = 0;
        foreach (string key in game.Flow.Flags.Keys.Where(x => x.StartsWith("round.", StringComparison.Ordinal) || x.StartsWith("status.", StringComparison.Ordinal) ||
            x.StartsWith("raddus.block-first.", StringComparison.Ordinal)).ToArray()) game.Flow.Flags.Remove(key);
        switch (game.State.Objective)
        { case EscortVipObjective e: e.BeginRound(game.State.Round); break; case IntelRetrievalObjective i: i.BeginRound(game.State.Round); break; case SeizeStationObjective s: s.BeginRound(game.State.Round); break; }
        game.Flow.Phase = GamePhase.StartRound;
    }

    private static IEnumerable<(int Source, int Index, UpgradeCardRuntimeState Card)> CurrentCards(GameContext game) => game.Topology.Upgrades.Select(upgrade =>
        (Source: (int)game.State.Registers[game.Topology.Registers.UpgradeOwner(upgrade.UpgradeIndex)] - 1, Index: upgrade.UpgradeIndex, Card: game.State.Upgrades[upgrade.UpgradeIndex]))
        .Where(x => game.IsShip(x.Source) && !game.IsDestroyed(x.Source) && !game.IsReserved(x.Source) && x.Card.Active)
        .OrderBy(x => game.State.Owners[x.Source] == game.Topology.FirstPlayer ? 0 : 1).ThenBy(x => x.Source).ThenBy(x => x.Index);
    private static IEnumerable<Swarmada.Headless.Setup.PreparedShipTopology> LiveShips(GameContext game) => game.Topology.Ships.Where(x => !game.IsDestroyed(x.EntityIndex) && !game.IsReserved(x.EntityIndex));
    private static IEnumerable<int> ReservedFor(GameContext game, int upgrade) => Enumerable.Range(0, game.Topology.EntityCount).Where(entity => game.IsReserved(entity) && game.State.Registers[game.Topology.Registers[$"entity.{entity}.reserveUpgradePlusOne"]] == upgrade + 1);
    private static bool CanGain(GameContext game, int entity, ShipCommand command) => TokenAssignments.CanAssign(game, entity, command);
    private static void IssueFleetTokens(GameContext game, int owner, ShipCommand command) { foreach (var ship in LiveShips(game).Where(x => x.Owner == owner && CanGain(game, x.EntityIndex, command))) TokenAssignments.TryAssign(game, ship.EntityIndex, command); }
    private static GameContinuation? CurrentRoundChoice(GameContext game) => game.Flow.Continuations.FirstOrDefault(x => x.Rule.StartsWith("round.", StringComparison.Ordinal));
    private static void RemoveRoundChoice(GameContext game) { int index = game.Flow.Continuations.FindIndex(x => x.Rule.StartsWith("round.", StringComparison.Ordinal)); if (index >= 0) game.Flow.Continuations.RemoveAt(index); }
    private static void UpdateRoundChoice(GameContext game, GameContinuation frame) { int index = game.Flow.Continuations.FindIndex(x => x.Rule.StartsWith("round.", StringComparison.Ordinal)); game.Flow.Continuations[index] = frame; }
    private static bool RoundMatches(GameActionOption option, GameAction action)
    {
        var expected = option.Template;
        if (option.AcceptsPosition) expected = expected with { Position = action.Position, Speed = action.Speed };
        if (option.AcceptsYaw) expected = expected with { YawDegrees = action.YawDegrees };
        return expected == action;
    }
}
