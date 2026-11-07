using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public static partial class MovementAbilityEffects
{
    // Ported from BattleTurnController: Hondo/Jendon (26774-26884), Clone Wars movement
    // reactions (27108-27266), Hera (26178-26265), and Kickback (28766-28822).
    private static IEnumerable<GameActionOption> GetSquadronEffectActions(GameContext game, string timing, int source, int target)
    {
        if (timing == "during-squadron-activation" && game.Flow.SquadronProgress is { } progress && progress.Entity == source)
        {
            var context = new AbilityQueryContext(AbilityTiming.DuringActivation, moveUsed: progress.Moved, attackUsed: progress.Attacked);
            foreach (var action in SquadronAbilityEngine.GetActions(game.SquadronViews[game.SquadronIndex(source)], context, game.SceneSquadronViews))
                yield return Option(source, action.AbilityId, action.Label);
        }
        if (timing == "after-squadron-move" && IsSquadron(game, source, CloneWarsSquadronRules.Ahsoka)
            && NearbySquadrons(game, source, TabletopMeasurements.DistanceOne, friendly: true).Any())
            yield return Option(source, "movement.ahsoka-after-move", "Ahsoka: another friendly squadron may make a two-blue attack");
        if (timing is "before-squadron-move" or "before-maneuver" && !WindowEffectUsed(game, timing, "haor"))
            foreach (var squad in game.Topology.Squadrons)
                if (IsSquadron(game, squad.EntityIndex, CloneWarsSquadronRules.HaorChall) && squad.Owner != game.State.Owners[source]
                    && FootprintDistance(game, squad.EntityIndex, source) <= TabletopMeasurements.DistanceOne)
                    yield return new(new(GameActionKind.UseAbility, squad.EntityIndex, source, AbilityId: "movement.haor-before-move"), "Haor Chall: attack before enemy movement");
        if (timing == "after-squadron-attack" && IsSquadron(game, source, CloneWarsSquadronRules.Kickback))
            yield return Option(source, "clone-wars.kickback.use", "Kickback: move distance one even while engaged");
        if (timing == "start-squadron-phase")
            foreach (var squad in game.Topology.Squadrons)
                if (IsSquadron(game, squad.EntityIndex, SquadronAbilityEngine.HeraSyndullaGhost)
                    && NearbySquadrons(game, squad.EntityIndex, TabletopMeasurements.DistanceBand(2), true).Any())
                    yield return Option(squad.EntityIndex, "movement.hera-rogue", "Hera: grant Rogue to up to two friendly squadrons");
    }

    /// <summary>Activation actions are queried again after movement, as in the source action bar.</summary>
    public static IEnumerable<GameActionOption> GetActivationActions(GameContext game, int squadron)
        => GetActions(game, "during-squadron-activation", squadron, -1);
    public static bool ApplyActivationAction(GameContext game, GameAction action)
        => TryApply(game, new("movement.direct", action.Source, Stage: (int)GamePhase.SquadronActivation), action);
    public static IEnumerable<GameActionOption> GetSquadronPhaseActions(GameContext game)
    {
        if (game.Flow.SquadronBatchKind != SquadronBatchKind.Normal) yield break;
        using var scope = new GameScope(game);
        int owner = game.Flow.SquadronPhasePlayer;
        foreach (var ship in game.Topology.Ships)
            if (ship.Owner == owner && game.IsOnBoard(ship.EntityIndex)
                && Card(game, ship.EntityIndex, "hyperwave-signal-boost") is { Active: true, Exhausted: false }
                && HyperwaveCandidates(game, ship.EntityIndex).Any())
                yield return Option(ship.EntityIndex, "movement.hyperwave-batch.begin", "Choose Hyperwave Signal Boost squadron turn");
    }
    public static bool ApplySquadronPhaseAction(GameContext game, GameAction action)
    {
        if (action.AbilityId != "movement.hyperwave-batch.begin") return false;
        using var scope = new GameScope(game);
        int limit = Math.Max(0, CommandUpgradeRules.SquadronValue(game.ShipViews[action.Source]));
        EngineEffects.PushChoice(game, new("movement-choice.squadron-batch", action.Source, Stage: (int)GamePhase.SquadronPhase,
            Value: limit, Text: "hyperwave", Indices: []), []); return true;
    }

    private static bool TryApplySquadronEffect(GameContext game, GameContinuation window, GameAction action)
    {
        int source = action.Source;
        int resume = window.Rule == "movement.direct" ? (int)GamePhase.SquadronActivation : (int)GamePhase.AbilityChoice;
        switch (action.AbilityId)
        {
            case SquadronAbilityEngine.HondoActivationAction:
            case SquadronAbilityEngine.JendonAttackAction:
                EngineEffects.PushChoice(game, new("movement-choice.squadron-target", source, Stage: resume, Text: action.AbilityId), []); return true;
            case CloneWarsSquadronRules.AnakinIgnoreEngagementAction:
                EngineEffects.PushChoice(game, new("movement-choice.anakin-token", source, Stage: resume), []); return true;
            case "movement.ahsoka-after-move":
                EngineEffects.PushChoice(game, new("movement-choice.ahsoka", source, Stage: resume), []); return true;
            case "movement.haor-before-move":
                game.Flow.Flags[$"effects.used.{window.Value}.haor.single"] = 1;
                QueueSquadronGrant(game, source, action.Target, null, false, resume, false); return true;
            case "clone-wars.kickback.use":
                EngineEffects.PushChoice(game, new("movement-choice.kickback", source, Stage: resume), []); return true;
            case "movement.hera-rogue":
                EngineEffects.PushChoice(game, new("movement-choice.hera", source, Stage: resume, Indices: []), []); return true;
        }
        return false;
    }

    private static GameDecision? GetSquadronChoiceDecision(GameContext game, GameContinuation frame)
    {
        using var scope = new GameScope(game);
        var options = new List<GameActionOption>();
        switch (frame.Rule)
        {
            case "movement-choice.squadron-target":
                bool jendon = frame.Text == SquadronAbilityEngine.JendonAttackAction;
                foreach (int squad in NearbySquadrons(game, frame.Source, jendon ? TabletopMeasurements.CloseRange : TabletopMeasurements.DistanceOne, jendon ? true : null))
                    options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, squad), jendon ? "Grant immediate attack" : "Apply Hondo's favor"));
                break;
            case "movement-choice.hondo-position":
                options.Add(new(new(GameActionKind.MoveSquadron, frame.Source, frame.Target), "Reposition within distance one of Hondo", AcceptsPosition: true)); break;
            case "movement-choice.anakin-token":
                foreach (var token in Enum.GetValues<DefenseTokenType>())
                    foreach (var instance in game.State.Squadrons[game.SquadronIndex(frame.Source)].GetDefenseTokenInstances(token))
                        options.Add(new(new(GameActionKind.ChooseToken, frame.Source, Token: token, Index: instance.Slot, Amount: instance.Exhausted ? 1 : 0), "Spend " + token + (instance.Exhausted ? " (discard)" : "")));
                break;
            case "movement-choice.ahsoka":
                foreach (int squad in NearbySquadrons(game, frame.Target < 0 ? frame.Source : frame.Target, TabletopMeasurements.DistanceOne, frame.Target < 0))
                    options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, squad), frame.Target < 0 ? "Choose friendly firing squadron" : "Choose enemy squadron target"));
                break;
            case "movement-choice.kickback":
                options.Add(new(new(GameActionKind.MoveSquadron, frame.Source), "Move up to distance one", AcceptsPosition: true)); break;
            case "movement-choice.hera":
                var selected = frame.Indices ?? [];
                foreach (int squad in NearbySquadrons(game, frame.Source, TabletopMeasurements.DistanceBand(2), true))
                    if (selected.Contains(squad) || selected.Length < 2)
                        options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, squad), selected.Contains(squad) ? "Remove Rogue recipient" : "Choose Rogue recipient"));
                break;
            case "movement-choice.granted-attack":
                return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), "Restore parent activation after granted attack")]);
            case "movement-choice.fenn-select":
                var fennSelected = frame.Indices ?? [];
                foreach (int squad in FennCandidates(game, frame.Source))
                    if (fennSelected.Contains(squad) || fennSelected.Length < 2)
                        options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, squad), fennSelected.Contains(squad) ? "Remove bonus activation recipient" : "Choose bonus activation recipient"));
                break;
            case "movement-choice.fenn-offer":
                int recipient = NextFennRecipient(game, frame);
                if (recipient < 0) return new(DecisionKind.Forced, 0, GamePhase.AbilityChoice, [new(new(GameActionKind.Pass), "Finish Fenn Rau offers")]);
                if (frame.Text == "offer" && CanOfferFennSnipe(game, frame.Target, recipient))
                    return new(DecisionKind.Player, game.State.Owners[frame.Target], GamePhase.AbilityChoice,
                        [new(new(GameActionKind.UseAbility, frame.Target, recipient, AbilityId: "movement.fenn-snipe.grant"), "Grant Ahsoka Snipe for bonus activation"),
                         new(new(GameActionKind.UseAbility, frame.Target, recipient, AbilityId: "movement.fenn-snipe.skip"), "Skip Snipe grant")]);
                return new(DecisionKind.Player, game.State.Owners[recipient], GamePhase.AbilityChoice,
                    [new(new(GameActionKind.ActivateSquadron, recipient), "Accept bonus commanded activation"), new(new(GameActionKind.Pass, recipient), "Decline bonus activation")]);
            case "movement-choice.ahsoka-snipe":
                var snipeSelected = frame.Indices ?? [];
                foreach (int squad in AhsokaCandidates(game, frame.Source))
                    if (snipeSelected.Contains(squad) || snipeSelected.Length < frame.Value)
                        options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, squad), snipeSelected.Contains(squad) ? "Remove Snipe recipient" : "Choose Snipe recipient"));
                break;
            case "movement-choice.squadron-batch":
                var batch = frame.Indices ?? [];
                foreach (int squad in BatchCandidates(game, frame))
                    if (batch.Contains(squad) || batch.Length < frame.Value)
                        options.Add(new(new(GameActionKind.ChooseEntity, frame.Source, squad), batch.Contains(squad) ? "Remove chosen squadron" : "Choose squadron for turn"));
                if (batch.Length > 0) options.Add(new(new(GameActionKind.Pass, frame.Source, AbilityId: "movement.batch.confirm"), "Confirm chosen squadron turn"));
                options.Add(new(new(GameActionKind.Pass, frame.Source, AbilityId: "movement.batch.cancel"), "Return to normal squadron turn"));
                return new(DecisionKind.Player, game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
            default: return null;
        }
        // Selecting Jendon commits the attack alternative. A UI back button here
        // would create a zero-progress use/cancel loop in the native action tree;
        // the outer activation already permits choosing a different action.
        bool committedJendon = frame.Rule == "movement-choice.squadron-target"
            && frame.Text == SquadronAbilityEngine.JendonAttackAction && options.Count > 0;
        if (!committedJendon)
            options.Add(new(new(GameActionKind.Pass, frame.Source), frame.Rule == "movement-choice.hera" ? "Confirm Rogue recipients" : "Cancel or skip"));
        return new(options.Count == 1 ? DecisionKind.Forced : DecisionKind.Player, options.Count == 1 ? 0 : game.State.Owners[frame.Source], GamePhase.AbilityChoice, options);
    }

    private static bool ApplySquadronChoice(GameContext game, GameContinuation frame, GameAction action)
    {
        if (frame.Rule == "movement-choice.granted-attack")
        {
            var saved = game.Flow.GrantedSquadronAttacks[^1];
            bool completed = game.Flow.Flags.GetValueOrDefault("combat.granted.completed") != 0;
            if (saved.Stage == 0 && saved.TriggerAfterAttack && completed)
            {
                saved.Stage = 1;
                if (EngineEffects.BeginWindow(game, "after-squadron-attack", frame.Target, resumePhase: GamePhase.AbilityChoice)) return true;
            }
            game.Flow.ActiveSquadron = saved.ParentSquadron;
            game.Flow.SquadronCanMove = saved.ParentCanMove; game.Flow.SquadronCanAttack = saved.ParentCanAttack;
            game.Flow.SquadronCommanded = saved.ParentCommanded;
            if (saved.ConsumeParentAttack && completed && game.Flow.SquadronProgress is { } progress)
            {
                progress.Attacked = true; game.Flow.SquadronCanAttack = false;
                if (!progress.CanMoveAndAttack) game.Flow.SquadronCanMove = false;
            }
            game.Flow.GrantedSquadronAttacks.RemoveAt(game.Flow.GrantedSquadronAttacks.Count - 1);
            game.Flow.Flags.Remove("combat.granted.completed");
            ReturnSquadronChoice(game, frame); return true;
        }
        if (GetSquadronChoiceDecision(game, frame) is null) return false;
        if (frame.Rule == "movement-choice.squadron-batch")
        {
            if (action.Kind == GameActionKind.ChooseEntity)
            {
                var chosen = frame.Indices ?? [];
                game.Flow.Continuations[^1] = frame with { Indices = chosen.Contains(action.Target) ? chosen.Where(s => s != action.Target).ToArray() : [..chosen, action.Target] }; return true;
            }
            if (action.AbilityId == "movement.batch.confirm")
            {
                int[] chosen = (frame.Indices ?? []).Where(s => BatchCandidates(game, frame).Contains(s)).ToArray();
                if (chosen.Length == 0) throw new ArgumentException("Choose at least one squadron.");
                if (Card(game, frame.Source, "hyperwave-signal-boost")?.Exhaust() != true) throw new ArgumentException("Hyperwave Signal Boost unavailable.");
                game.Flow.SquadronBatchKind = SquadronBatchKind.HyperwaveSignalBoost;
                game.Flow.SquadronBatchSource = frame.Source; game.Flow.SquadronBatchSelections = chosen; game.Flow.SquadronPhaseRemaining = chosen.Length;
            }
            ReturnSquadronChoice(game, frame); return true;
        }
        if (frame.Rule == "movement-choice.fenn-offer")
        {
            int recipient = NextFennRecipient(game, frame);
            if (recipient < 0) { ReturnSquadronChoice(game, frame); return true; }
            int index = Array.IndexOf(frame.Indices ?? [], recipient);
            if (action.Kind == GameActionKind.UseAbility)
            {
                if (action.AbilityId == "movement.fenn-snipe.grant") game.Flow.Flags[$"ahsoka.{frame.Target}.{recipient}"] = 1;
                game.Flow.Continuations[^1] = frame with { Value = index, Text = action.AbilityId == "movement.fenn-snipe.grant" ? "offer-granted" : "offer-nosnipe" };
                return true;
            }
            game.Flow.Continuations[^1] = frame with { Value = index, Text = action.Kind == GameActionKind.Pass ? "offer" : "active" };
            if (action.Kind == GameActionKind.Pass)
            {
                if (frame.Text == "offer-granted") game.Flow.Flags.Remove($"ahsoka.{frame.Target}.{recipient}");
                game.Flow.Continuations[^1] = game.Flow.Continuations[^1] with { Value = index + 1 };
            }
            else
            {
                game.Flow.ShipActivations[game.Flow.ActiveShip].CommandedSquadrons.Add(recipient);
                SquadronModule.BeginActivation(game, recipient, true);
            }
            return true;
        }
        if (action.Kind == GameActionKind.Pass)
        {
            if (frame.Rule == "movement-choice.fenn-select")
            { game.Flow.Continuations[^1] = frame with { Rule = "movement-choice.fenn-offer", Text = "offer", Value = 0 }; return true; }
            if (frame.Rule == "movement-choice.ahsoka-snipe")
                foreach (int squad in frame.Indices ?? [])
                    if (AhsokaCandidates(game, frame.Source).Contains(squad)) game.Flow.Flags[$"ahsoka.{frame.Source}.{squad}"] = 1;
            if (frame.Rule == "movement-choice.hera")
                foreach (int squad in frame.Indices ?? [])
                    if (NearbySquadrons(game, frame.Source, TabletopMeasurements.DistanceBand(2), true).Contains(squad))
                        game.State.Squadrons[game.SquadronIndex(squad)].GrantTemporaryRogue();
            ReturnSquadronChoice(game, frame); return true;
        }
        switch (frame.Rule)
        {
            case "movement-choice.squadron-target":
                if (frame.Text == SquadronAbilityEngine.JendonAttackAction)
                {
                    ReturnSquadronChoice(game, frame);
                    // The other squadron may decline its attack; Jendon still
                    // spent his own attack by choosing the recipient instead.
                    ConsumeSquadronAttack(game);
                    QueueSquadronGrant(game, action.Target, null, null, false, frame.Stage, true); return true;
                }
                var targetState = game.State.Squadrons[game.SquadronIndex(action.Target)];
                if (!targetState.Activated)
                {
                    targetState.MarkActivated(false); game.State.Activated[action.Target] = true;
                    ConsumeSquadronAttack(game); ReturnSquadronChoice(game, frame); return true;
                }
                game.Flow.Continuations[^1] = frame with { Rule = "movement-choice.hondo-position", Target = action.Target }; return true;
            case "movement-choice.hondo-position":
                var hondoPosition = action.Position ?? throw new ArgumentException("Hondo destination required.");
                if (System.Numerics.Vector2.Distance(hondoPosition, game.State.Poses[frame.Source].Position) > TabletopMeasurements.DistanceOne + Geometry2D.DefaultTolerance
                    || !SquadronModule.IsLegalMoveDestination(game, frame.Target, hondoPosition)) throw new ArgumentException("Illegal Hondo destination.");
                game.State.Poses[frame.Target] = new(hondoPosition, game.State.Poses[frame.Target].YawDegrees);
                PlayAreaDestruction.DestroyIfOutside(game, frame.Target);
                ConsumeSquadronAttack(game); RoundModule.ObserveObjectives(game); ReturnSquadronChoice(game, frame);
                MissingSquadronLifecycleEffects.BeginBonusMoveConsequences(game, frame.Target, (GamePhase)frame.Stage); return true;
            case "movement-choice.anakin-token":
                var state = game.State.Squadrons[game.SquadronIndex(frame.Source)];
                if (!state.TryResolveDefenseToken(new(action.Token, action.Index, action.Amount != 0), out _)) throw new ArgumentException("Defense token unavailable.");
                state.GrantTemporaryIgnoreEngagement();
                if (game.Flow.SquadronProgress is { Moved: false }) game.Flow.SquadronCanMove = !state.TemporaryAttackOnly;
                ReturnSquadronChoice(game, frame); return true;
            case "movement-choice.ahsoka":
                if (frame.Target < 0) { game.Flow.Continuations[^1] = frame with { Target = action.Target }; return true; }
                ReturnSquadronChoice(game, frame); QueueSquadronGrant(game, frame.Target, action.Target, 2, false, frame.Stage, false); return true;
            case "movement-choice.kickback":
                var position = action.Position ?? throw new ArgumentException("Kickback destination required.");
                if (System.Numerics.Vector2.Distance(position, game.State.Poses[frame.Source].Position) > TabletopMeasurements.DistanceOne + Geometry2D.DefaultTolerance
                    || !SquadronModule.IsLegalMoveDestination(game, frame.Source, position)) throw new ArgumentException("Illegal Kickback destination.");
                game.State.Poses[frame.Source] = new(position, game.State.Poses[frame.Source].YawDegrees);
                PlayAreaDestruction.DestroyIfOutside(game, frame.Source);
                RoundModule.ObserveObjectives(game); ReturnSquadronChoice(game, frame);
                MissingSquadronLifecycleEffects.BeginBonusMoveConsequences(game, frame.Source, (GamePhase)frame.Stage); return true;
            case "movement-choice.hera":
                var selected = frame.Indices ?? [];
                game.Flow.Continuations[^1] = frame with { Indices = selected.Contains(action.Target) ? selected.Where(s => s != action.Target).ToArray() : [..selected, action.Target] }; return true;
            case "movement-choice.fenn-select":
                var fenn = frame.Indices ?? [];
                game.Flow.Continuations[^1] = frame with { Indices = fenn.Contains(action.Target) ? fenn.Where(s => s != action.Target).ToArray() : [..fenn, action.Target] }; return true;
            case "movement-choice.ahsoka-snipe":
                var snipers = frame.Indices ?? [];
                game.Flow.Continuations[^1] = frame with { Indices = snipers.Contains(action.Target) ? snipers.Where(s => s != action.Target).ToArray() : [..snipers, action.Target] }; return true;
        }
        return false;
    }

    private static void QueueSquadronGrant(GameContext game, int attacker, int? defender, int? blueDice, bool consumeParentAttack, int resume, bool triggerAfterAttack)
    {
        game.Flow.GrantedSquadronAttacks.Add(new() { ParentSquadron = game.Flow.ActiveSquadron,
            ParentCanMove = game.Flow.SquadronCanMove, ParentCanAttack = game.Flow.SquadronCanAttack,
            ParentCommanded = game.Flow.SquadronCommanded, ConsumeParentAttack = consumeParentAttack, TriggerAfterAttack = triggerAfterAttack });
        EngineEffects.PushChoice(game, new("movement-choice.granted-attack", game.Flow.ActiveSquadron, attacker, resume), []);
        game.Flow.ActiveSquadron = attacker; game.Flow.SquadronCanMove = false; game.Flow.SquadronCanAttack = true;
        CombatModule.QueueGrantedAttack(game, attacker, defender, blueDice, GamePhase.AbilityChoice);
    }
    private static void ConsumeSquadronAttack(GameContext game)
    {
        if (game.Flow.SquadronProgress is not { } progress) throw new InvalidOperationException("No parent squadron activation.");
        if (progress.Commanded) CommandModule.CommitSquadronSelection(game, progress.Entity);
        progress.Attacked = true; game.Flow.SquadronCanAttack = false;
        if (!progress.CanMoveAndAttack) game.Flow.SquadronCanMove = false;
    }
    private static void ReturnSquadronChoice(GameContext game, GameContinuation frame)
    {
        EngineEffects.ReturnFromChoice(game); game.Flow.Phase = (GamePhase)frame.Stage;
        if (frame.Rule == "movement-choice.fenn-offer" && game.Flow.Phase == GamePhase.SelectAttack) ActivationAbilityEffects.CompleteCommand(game);
    }
    private static bool IsSquadron(GameContext game, int entity, string id) => entity >= game.State.Ships.Length && entity < game.Topology.EntityCount
        && game.IsOnBoard(entity) && game.Topology.Squadrons[game.SquadronIndex(entity)].CatalogEntry.Definition.SquadronId == id;
    private static IEnumerable<int> NearbySquadrons(GameContext game, int source, float distance, bool? friendly)
        => game.Topology.Squadrons.Where(s => s.EntityIndex != source && game.IsOnBoard(s.EntityIndex)
            && (friendly is null || (s.Owner == game.State.Owners[source]) == friendly.Value)
            && FootprintDistance(game, source, s.EntityIndex) <= distance).Select(s => s.EntityIndex);
    private static float FootprintDistance(GameContext game, int a, int b) => game.IsShip(b)
        ? Geometry2D.Distance(game.SquadronBase(a), game.ShipBase(b)) : Geometry2D.Distance(game.SquadronBase(a), game.SquadronBase(b));
    private static bool WindowEffectUsed(GameContext game, string timing, string name)
    {
        var window = game.Flow.Continuations.LastOrDefault(c => c.Rule == "effects.window" && c.Text == timing);
        return window is not null && game.Flow.Flags.ContainsKey($"effects.used.{window.Value}.{name}.single");
    }
    public static bool BeginFennRauOffers(GameContext game, int completed, GamePhase resumePhase)
    {
        if (!IsSquadron(game, completed, "fenn-rau-mandalorian-gauntlet-fighter") || !FennCandidates(game, completed).Any()) return false;
        EngineEffects.PushChoice(game, new("movement-choice.fenn-select", completed, game.Flow.ActiveShip, (int)resumePhase, Indices: []), []); return true;
    }
    public static bool FinishFennRauBonusActivation(GameContext game)
    {
        var frame = game.Flow.Continuations.LastOrDefault();
        if (frame is not { Rule: "movement-choice.fenn-offer", Text: "active" }) return false;
        game.Flow.Continuations[^1] = frame with { Text = "offer", Value = frame.Value + 1 };
        game.Flow.Phase = GamePhase.AbilityChoice; return true;
    }
    private static IEnumerable<int> FennCandidates(GameContext game, int source) => NearbySquadrons(game, source, TabletopMeasurements.DistanceOne, null)
        .Where(s => !game.Topology.Squadrons[game.SquadronIndex(s)].CatalogEntry.Definition.IsUnique
            && !game.State.Squadrons[game.SquadronIndex(s)].Activated && !game.State.Activated[s]);
    private static int NextFennRecipient(GameContext game, GameContinuation frame) => (frame.Indices ?? []).Skip((int)frame.Value)
        .FirstOrDefault(s => game.IsOnBoard(s) && !game.State.Squadrons[game.SquadronIndex(s)].Activated && !game.State.Activated[s], -1);
    public static bool BeginAhsokaSnipeChoices(GameContext game, int ship)
    {
        if (!game.HasUpgrade(ship, "ahsoka-tano-officer-republic")) return false;
        int limit = Math.Min(3, Math.Min(game.Flow.ShipActivations[ship].CommandedSquadronsRemaining, AhsokaCandidates(game, ship).Count()));
        if (limit <= 0) return false;
        EngineEffects.PushChoice(game, new("movement-choice.ahsoka-snipe", ship, Stage: (int)GamePhase.SquadronCommand, Value: limit, Indices: []), []); return true;
    }
    public static void ApplyCommandedSquadronGrants(GameContext game, int ship, int squadron)
    {
        var definition = game.Topology.Squadrons[game.SquadronIndex(squadron)].CatalogEntry.Definition;
        var state = game.State.Squadrons[game.SquadronIndex(squadron)];
        if (!definition.IsUnique && game.Flow.Flags.ContainsKey($"ahsoka.{ship}.{squadron}")) state.GrantTemporarySnipe(Math.Min(3, definition.AntiSquadronArmament.TotalDice));
        if (!definition.IsUnique && game.HasUpgrade(ship, "nevoota-bee-title")) state.GrantTemporarySwarm();
    }
    private static IEnumerable<int> AhsokaCandidates(GameContext game, int ship) => game.Topology.Squadrons.Select(s => s.EntityIndex)
        .Where(s => CanGrantSnipe(game, s) && CommandModule.CanCommandSquadron(game, ship, s));
    private static bool CanGrantSnipe(GameContext game, int squad) => game.IsOnBoard(squad)
        && !game.Topology.Squadrons[game.SquadronIndex(squad)].CatalogEntry.Definition.IsUnique
        && game.Topology.Squadrons[game.SquadronIndex(squad)].CatalogEntry.Definition.AntiSquadronArmament.TotalDice > 0;
    private static bool CanOfferFennSnipe(GameContext game, int ship, int squad) => game.HasUpgrade(ship, "ahsoka-tano-officer-republic")
        && CanGrantSnipe(game, squad) && !game.Flow.Flags.ContainsKey($"ahsoka.{ship}.{squad}")
        && game.Flow.Flags.Keys.Count(k => k.StartsWith($"ahsoka.{ship}.", StringComparison.Ordinal)) < 3;
    private static IEnumerable<int> HyperwaveCandidates(GameContext game, int ship) => game.Topology.Squadrons
        .Where(s => s.Owner == game.State.Owners[ship] && game.IsOnBoard(s.EntityIndex)
            && !game.State.Squadrons[s.SquadronIndex].Activated && !game.State.Activated[s.EntityIndex]
            && Geometry2D.Distance(game.SquadronBase(s.EntityIndex), game.ShipBase(ship)) <= TabletopMeasurements.LongRange).Select(s => s.EntityIndex);
    private static IEnumerable<int> BatchCandidates(GameContext game, GameContinuation frame)
    {
        return HyperwaveCandidates(game, frame.Source);
    }
}
