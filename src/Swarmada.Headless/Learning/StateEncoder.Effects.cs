using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Search;

namespace Swarmada.Headless.Learning;

public sealed partial class StateEncoder
{
    // Numeric native references stay in integer columns. The scalar block never contains
    // registry hashes, runtime IDs, serial counters, packed bit fields, or secret placeholders.
    private Writer Effect(EncodedEffectKind kind, string rule, string text = "", int source = -1,
        int target = -1, int slot = -1, int parent = -1, int aux = -1)
    {
        int start = effects.Count; CollectionsMarshal.SetCount(effects, start + EffectWidth);
        var span = CollectionsMarshal.AsSpan(effects).Slice(start, EffectWidth); span.Clear();
        effectRefs.Add((int)kind); effectRefs.Add(Symbol(rule, "effect.rule")); effectRefs.Add(Symbol(text, "effect.text")); effectRefs.Add(source); effectRefs.Add(target); effectRefs.Add(slot); effectRefs.Add(parent); effectRefs.Add(aux);
        var writer = new Writer(span); writer.Hot(16, (int)kind); return writer;
    }
    private void Link(string rule, int source, int target, int slot = -1)
    { if (source < 0 && target < 0) return; var e = Effect(EncodedEffectKind.EntityLink, rule, source: source, target: target, slot: slot); e.B(true); }
    private void Store(int kind, bool known, int card, int entity, int owner, int slot, int command = -1, float scalar = 0)
    {
        int start = stores.Count; CollectionsMarshal.SetCount(stores, start + StoreWidth); var span = CollectionsMarshal.AsSpan(stores).Slice(start, StoreWidth); span.Clear();
        var w = new Writer(span); w.Hot(4, kind); w.B(known); w.B(true); w.Hot(4, known ? command : -1); w.N(known ? scalar : 0);
        storeRefs.Add(card); storeRefs.Add(known ? entity : -1); storeRefs.Add(owner); storeRefs.Add(slot);
    }

    private int Symbol(string value, string location)
    {
        if (string.IsNullOrEmpty(value)) return 0;
        bool nativeReference = false;
        if (!entityIdsHaveSeparators || value.Contains('|'))
            foreach (var entity in entityIds)
                if (value.Contains(entity.Key, StringComparison.Ordinal)) { value = value.Replace(entity.Key, "__entity__ " + entity.Value.ToString(CultureInfo.InvariantCulture) + " ", StringComparison.Ordinal); nativeReference = true; }
        if (!nativeReference)
        { int exact = Schema.Symbols.Index(value); if (exact > 0) return exact; }
        string normalized = NormalizeSymbol(value); int symbol = Schema.Symbols.Index(normalized);
        if (symbol > 0 && !nativeReference && !IntegerSegments().IsMatch(value)) return symbol;
        if (currentSymbolSequences.TryGetValue(value, out int previous)) return -previous - 2;
        int sequence = currentSymbolSequences.Count; currentSymbolSequences.Add(value, sequence); int ordinal = 0; bool nextEntity = false;
        foreach (Match token in LexicalTokens().Matches(value))
        {
            if (token.Value == "__entity__") { nextEntity = true; continue; }
            int argument = -1, category;
            if (int.TryParse(token.Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int number))
            { argument = number; category = Schema.SymbolWords.Index(nextEntity ? "#entity" : "#integer"); nextEntity = false; }
            else category = Schema.SymbolWords.Index(token.Value);
            if (category == 0) issues.Add(new("unknown-symbol-word", location, token.Value));
            symbolTokens.Add(sequence); symbolTokens.Add(ordinal++); symbolTokens.Add(category); symbolTokens.Add(argument);
        }
        return -sequence - 2;
    }
    private static string NormalizeSymbol(string value)
    {
        // A registry symbol describes the semantic family, while numeric suffixes are
        // explicit integer references. Interpolated native IDs are never a scalar feature.
        value = Interpolation().Replace(value, "");
        return NumericSegment().Replace(value.Trim(), "").Trim('.', '-', ':', ' ', '/');
    }
    [GeneratedRegex(@"\{[^{}]*\}", RegexOptions.CultureInvariant)] private static partial Regex Interpolation();
    [GeneratedRegex(@"(?<![A-Za-z])\b-?\d+\b(?![A-Za-z])", RegexOptions.CultureInvariant)] private static partial Regex NumericSegment();
    [GeneratedRegex(@"(?<![A-Za-z])-?\d+(?![A-Za-z])", RegexOptions.CultureInvariant)] private static partial Regex IntegerSegments();
    [GeneratedRegex(@"[\p{L}_][\p{L}\p{N}_]*|-?\d+|[^\w\s]", RegexOptions.CultureInvariant)] private static partial Regex LexicalTokens();

    private void Flag(EncodedEffectKind kind, string key, long value, GameContext game, int observer, int parent = -1, SearchKnowledge? boundary = null)
    {
        // These are purely allocator counters. Their identity has no gameplay meaning.
        if (key is "effects.serial" or "token.next-id" or "activation.serial") return;
        if (key.StartsWith("token.pending.", StringComparison.Ordinal))
        {
            int source = (int)(value >> 4), command = (int)((value >> 1) & 7); bool stored = (value & 1) != 0;
            var token = Effect(kind, "token.pending", source: source, aux: command, parent: parent); token.B(true); token.B(stored); return;
        }
        int[] numbers = HasDigit(key) ? IntegerSegments().Matches(key).Select(match => int.TryParse(match.Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int number) ? number : -1).ToArray() : [];
        int sourceIndex = numbers.Length > 0 ? numbers[0] : -1, targetIndex = numbers.Length > 1 ? numbers[1] : -1, slotIndex = numbers.Length > 2 ? numbers[2] : -1;
        if (key.StartsWith("effects.used.", StringComparison.Ordinal) && numbers.Length > 0)
        {
            int window = game.Flow.Continuations.FindIndex(x => x.Rule == "effects.window" && x.Value == numbers[0]);
            if (window < 0) return; // Retired serial namespaces cannot affect any future window.
            sourceIndex = window;
            // Preserve the live window link, not an arbitrary allocation serial.
            var serial = IntegerSegments().Match(key); key = key[..serial.Index] + window.ToString(CultureInfo.InvariantCulture) + key[(serial.Index + serial.Length)..];
        }
        if (key.StartsWith("rift.used.", StringComparison.Ordinal))
        { if (value != game.Flow.Flags.GetValueOrDefault("activation.serial")) return; value = 1; }
        bool known = !key.StartsWith("surprise.dial.", StringComparison.Ordinal) ||
            observer == 3 - game.Topology.FirstPlayer && (boundary is null || boundary.Observer == observer) ||
            game.State.Objective is OfficialObjectiveState objective && objective.Flags.ContainsKey("surprise.revealed." + sourceIndex);
        if (value is < int.MinValue or > int.MaxValue)
        { issues.Add(new("unregistered-flag-payload", key, "Packed 64-bit flag requires an explicit codec; omitted.")); return; }
        var e = Effect(kind, key, source: sourceIndex, target: targetIndex, slot: slotIndex, parent: parent, aux: known ? (int)value : -1); e.B(true); e.B(known);
    }

    private static bool HasDigit(string value)
    { foreach (char character in value) if (char.IsAsciiDigit(character)) return true; return false; }

    private void EncodeEffects(GameContext game, int observer, SearchKnowledge? boundary)
    {
        var flow = game.Flow;
        foreach (var flag in flow.Flags) Flag(EncodedEffectKind.FlowFlag, flag.Key, flag.Value, game, observer);
        for (int i = 0; i < flow.Continuations.Count; i++) Continuation(game, flow.Continuations[i], observer, boundary, i);
        if (game.State.Frames.Count > 0) issues.Add(new("low-level-rule-frames", "battle.frames", "Low-level RuleFrame payloads require a registered handler encoder."));
        var round = flow.SourceRules;
        RoundCommands(round.TrenchCommands, "trench-command", observer, boundary); RoundCommands(round.DravenCommands, "draven-command", observer, boundary); RoundCommands(round.ThrawnCommands, "thrawn-command", observer, boundary); RoundCommands(round.AdmiralTarkinDials, "tarkin-dial", observer, boundary);
        foreach (var entry in round.PalpatineTokens) { var e = Effect(EncodedEffectKind.PalpatineToken, "palpatine-token", source: entry.Key == observer ? 1 : 2, aux: (int)entry.Value); e.B(true); }
        foreach (var entry in round.PalpatineRemaining) foreach (var token in entry.Value.Order()) { var e = Effect(EncodedEffectKind.PalpatineRemaining, "palpatine-remaining", source: entry.Key == observer ? 1 : 2, aux: (int)token); e.B(true); }
        if (game.State.Objective is OfficialObjectiveState objective)
        {
            foreach (var flag in objective.Flags) Flag(EncodedEffectKind.ObjectiveFlag, flag.Key, flag.Value, game, observer, boundary: boundary);
            int index = 0; foreach (var item in objective.TimingWindows.OrderBy(x => x.Key))
            {
                int currentWindow = flow.Continuations.FindIndex(x => x.Value == item.Key && x.Rule == "effects.window");
                var e = Effect(EncodedEffectKind.Sequence, "objective-timing-sequence", source: item.Value.Player == observer ? 1 : 2, slot: index++, parent: currentWindow); e.N(item.Value.Frames.Length / 16f);
                for (int frame = 0; frame < item.Value.Frames.Length; frame++) Continuation(game, item.Value.Frames[frame], observer, boundary, frame, currentWindow);
            }
        }
        { var ruleset = Effect(EncodedEffectKind.Register, "ruleset", game.Topology.Ruleset); ruleset.B(true); }
        { var scenario = Effect(EncodedEffectKind.Register, "scenario", game.Topology.ScenarioId); scenario.B(true); }
        { var escort = Effect(EncodedEffectKind.Register, "escort-player", source: game.Topology.EscortPlayer == 0 ? 0 : game.Topology.EscortPlayer == observer ? 1 : 2); escort.N(game.Topology.PlayPlaneMillimetres / 1000); }
        for (int i = 0; i < game.Topology.EntityCount; i++)
        {
            Link("reserved-card", i, checked((int)game.State.Registers[game.Topology.Registers[$"entity.{i}.reserveUpgradePlusOne"]] - 1));
            Link("reserved-source", i, checked((int)game.State.Registers[game.Topology.Registers[$"entity.{i}.reserveSourcePlusOne"]] - 1));
            var e = Effect(EncodedEffectKind.Register, "scout-granted", source: i); e.B(game.State.Registers[game.Topology.Registers[$"entity.{i}.scoutGranted"]] != 0);
        }
        // Setup and round masks are typed; private unfinished scheduled choices are owner-masked.
        foreach (string name in new[] { "setup.stage", "setup.completed", "setup.currentPlayer", "setup.choiceActive", "setup.choiceSelectedCount", "setup.postDeploymentApplied", "setup.objectiveSelectionCompleted", "round.started", "round.commandPlanningCompleted" })
        { var e = Effect(EncodedEffectKind.Register, name, aux: checked((int)game.State.Registers[game.Topology.Registers[name]])); e.B(true); }
        for (int owner = 1; owner <= 2; owner++)
        { var e = Effect(EncodedEffectKind.Register, "fleet-command-mask", source: owner == observer ? 1 : 2, aux: checked((int)game.State.Registers[game.Topology.Registers[$"round.fleetCommandMask.player{owner}"]])); e.B(true); }
        Link("planning-ship", flow.PlanningShip, -1); Link("deployment-entity", flow.DeploymentEntity, -1); Link("displacement-ship", flow.DisplacementShip, -1); Link("squadron-batch-source", flow.SquadronBatchSource, -1);
        for (int i = 0; i < flow.PendingDisplacements.Length; i++) Link("pending-displacement", flow.DisplacementShip, flow.PendingDisplacements[i], i);
        for (int i = 0; i < flow.SquadronBatchSelections.Length; i++) Link("squadron-batch-selection", flow.SquadronBatchSource, flow.SquadronBatchSelections[i], i);
        { var e = Effect(EncodedEffectKind.Sequence, "squadron-batch-kind", aux: (int)flow.SquadronBatchKind); e.B(true); }
        for (int i = 0; i < flow.DeathRecorded.Length; i++) { var e = Effect(EncodedEffectKind.DeathTrigger, "death-recorded", source: i); e.B(flow.DeathRecorded[i]); }
        for (int i = 0; i < flow.DeathQueue.Count; i++) { var death = flow.DeathQueue[i]; var e = Effect(EncodedEffectKind.DeathTrigger, "death-queue", source: death.Source, target: death.Entity, slot: i, aux: death.Effects); e.B(true); }
        if (flow.Movement is not null) Movement(game, flow.Movement, observer, "movement-current");
        if (flow.SquadronProgress is not null) Progress(game, flow.SquadronProgress, observer, "current-squadron-progress", -1);
        for (int i = 0; i < flow.ExtraManeuvers.Count; i++) { var item = flow.ExtraManeuvers[i]; var e = Effect(EncodedEffectKind.Sequence, "extra-maneuver", source: item.Ship, slot: i); e.N(item.OriginalSpeed / 4f); if (item.ParentMovement is not null) Movement(game, item.ParentMovement, observer, "movement-extra-parent", i); }
        for (int i = 0; i < flow.ForcedManeuvers.Count; i++) { var item = flow.ForcedManeuvers[i]; var e = Effect(EncodedEffectKind.Sequence, "forced-maneuver", source: item.Ship, target: item.ActiveShip, slot: i, aux: item.ActiveSquadron); e.N(item.OriginalSpeed / 4f); var phase = Effect(EncodedEffectKind.Sequence, "forced-maneuver-return-phase", source: item.Ship, slot: i, aux: (int)item.ReturnPhase); phase.B(true); if (item.ParentMovement is not null) Movement(game, item.ParentMovement, observer, "movement-forced-parent", i); }
        for (int i = 0; i < flow.GrantedSquadronAttacks.Count; i++) { var item = flow.GrantedSquadronAttacks[i]; var e = Effect(EncodedEffectKind.Sequence, "granted-squadron-attack", source: item.ParentSquadron, slot: i); e.B(item.ParentCanMove); e.B(item.ParentCanAttack); e.B(item.ParentCommanded); e.B(item.ConsumeParentAttack); e.B(item.TriggerAfterAttack); e.N(item.Stage / 16f); }
        for (int i = 0; i < flow.StandaloneSquadronActivations.Count; i++) { var item = flow.StandaloneSquadronActivations[i]; var e = Effect(EncodedEffectKind.Sequence, "standalone-squadron-activation", source: item.ActiveShip, target: item.ActiveSquadron, slot: i, aux: (int)item.ReturnPhase); e.B(item.CanMove); e.B(item.CanAttack); e.B(item.Commanded); if (item.Progress is not null) Progress(game, item.Progress, observer, "standalone-parent-progress", i); }
        foreach (var item in flow.TokenHosts) for (int i = 0; i < item.Value.Length; i++) { var e = Effect(EncodedEffectKind.TokenHost, "token-obstacle-host", source: item.Key, target: item.Value[i], slot: i); e.B(true); }
        for (int i = 0; i < flow.ObservedObjectiveTokens.Count; i++) { var item = flow.ObservedObjectiveTokens[i]; var e = Effect(EncodedEffectKind.ArrayItem, "observed-objective-token", source: item.Index, slot: i, aux: item.Kind); e.B(item.Active); e.B(item.OnObstacle); e.N(item.Radius / 1000); Point(game, observer, item.Position, ref e); }
        for (int i = 0; i < flow.PendingMoraloTriggers.Count; i++) { var item = flow.PendingMoraloTriggers[i]; var e = Effect(EncodedEffectKind.ArrayItem, "moralo-trigger", source: item.Squadron, slot: i); Point(game, observer, item.TokenPosition, ref e); }
    }

    private void RoundCommands(Dictionary<int, ShipCommand> commands, string rule, int observer, SearchKnowledge? boundary)
    {
        foreach (var entry in commands)
        { bool visible = entry.Key == observer && (boundary is null || boundary.Observer == observer); var e = Effect(EncodedEffectKind.RoundCommand, rule, source: entry.Key == observer ? 1 : 2); e.B(true); e.B(visible); e.Hot(4, visible ? (int)entry.Value : -1); }
    }

    private void Continuation(GameContext game, GameContinuation frame, int observer, SearchKnowledge? boundary, int ordinal, int parent = -1)
    {
        bool privateSetup = frame.Rule is "setup.store-commands" or "setup.schedule" or "setup.assign-command";
        bool visible = !privateSetup || frame.Source >= 0 && game.State.Owners[frame.Source] == observer && (boundary is null || boundary.Observer == observer);
        int value = 0;
        if (frame.Rule != "effects.window")
        {
            if (frame.Value is >= int.MinValue and <= int.MaxValue) value = visible ? (int)frame.Value : -1;
            else { issues.Add(new("continuation-payload", frame.Rule, "Packed 64-bit continuation payload requires an explicit codec.")); value = -1; }
        }
        if (frame.Rule == "general.reveal.passel") value = game.Flow.Continuations.FindIndex(candidate => candidate.Rule == "effects.window" && candidate.Value == frame.Value);
        var e = Effect(EncodedEffectKind.Continuation, frame.Rule, visible ? frame.Text : "", frame.Source, frame.Target, ordinal, parent, value); e.B(true); e.B(visible); e.N(frame.Stage / 32f);
        if (frame.Rule is "objective.creature.ship-obstacles" or "objective.creature.ship-overlap")
        { var flags = Effect(EncodedEffectKind.ArrayItem, "ship-obstacle-flags", source: frame.Source, parent: ordinal); flags.B((frame.Value & 1) != 0); flags.B((frame.Value & 2) != 0); flags.N((frame.Value >> 2) / 4f); }
        if (visible && frame.Indices is not null) for (int i = 0; i < frame.Indices.Length; i++) { var index = Effect(EncodedEffectKind.ArrayItem, "continuation-index", source: frame.Indices[i], slot: i, parent: ordinal); index.B(true); }
    }

    private void Movement(GameContext game, MovementExecution m, int observer, string rule, int parent = -1)
    {
        var e = Effect(EncodedEffectKind.Movement, rule, source: m.Ship, target: m.CollidedShip, parent: parent, aux: m.ObjectiveToken); Point(game, observer, m.Start.Position, ref e); Heading(observer, m.Start.YawDegrees, ref e); Point(game, observer, m.End.Position, ref e); Heading(observer, m.End.YawDegrees, ref e); e.N(m.DialedSpeed / 4f); e.N(m.Stage / 16f); e.B(m.PlayAreaHold); e.B(m.EndsOutsidePlayArea); e.B(m.StandstillYaw); e.N(m.YawDelta / 180f); e.N(m.ObjectiveChoiceT); e.N(m.ObjectiveStopT);
        var flags = Effect(EncodedEffectKind.Movement, "movement-flags", source: m.Ship, parent: parent, aux: (int)m.GuideSide); flags.B(m.ObjectiveChoiceResolved); flags.B(m.ObjectiveAccepted); flags.B(m.ChartOfficerChoiceResolved); flags.B(m.IgnoreObstacleEffects); flags.B(m.IgnoredByJaina); flags.B(m.ObstacleRewardsResolved); flags.B(m.AfterManeuverDamageResolved); flags.N(m.DamageEffectIndex / 32f); flags.N(m.MineIndex / 32f); flags.B(m.MinesGathered);
        for (int i = 0; i < m.PlottedCourse.Length; i++) { var row = Effect(EncodedEffectKind.ArrayItem, "plotted-course", source: m.Ship, slot: i, parent: parent); row.N(m.PlottedCourse[i] / 2f); }
        for (int i = 0; i < m.ExecutedCourse.Length; i++) { var row = Effect(EncodedEffectKind.ArrayItem, "executed-course", source: m.Ship, slot: i, parent: parent); row.N(m.ExecutedCourse[i] / 2f); }
        for (int i = 0; i < m.OverlappedObstacles.Length; i++) { var row = Effect(EncodedEffectKind.ArrayItem, "overlapped-obstacle", source: m.Ship, target: m.OverlappedObstacles[i], slot: i, parent: parent); row.B(true); }
        for (int i = 0; i < m.MineIndices.Length; i++) { var row = Effect(EncodedEffectKind.ArrayItem, "pending-mine", source: m.Ship, target: m.MineIndices[i], slot: i, parent: parent); row.B(true); }
        for (int i = 0; i < m.DamageEffects.Count; i++) { var damage = m.DamageEffects[i]; var row = Effect(EncodedEffectKind.MovementDamage, damage.Rule, source: m.Ship, target: damage.Entity, slot: i, parent: parent, aux: damage.Zone.HasValue ? (int)damage.Zone.Value : -1); row.N(damage.Amount / 32f); row.B(damage.Faceup); row.B(damage.IgnoreShields); }
    }
    private void Progress(GameContext game, SquadronActivationProgress p, int observer, string rule, int parent)
    {
        var e = Effect(EncodedEffectKind.Sequence, rule, source: p.Entity, parent: parent); e.B(p.Moved); e.B(p.Attacked); e.B(p.Commanded); e.B(p.CanMoveAndAttack); e.B(p.WolffeGrantAdded); e.B(p.ActivationCommitted); e.B(p.Standalone); e.B(p.PreserveActivation); e.N(p.MaximumMoveDistance / 1000); e.N(p.Stage / 16f); e.B(p.PendingMove.HasValue); Point(game, observer, p.PendingMove ?? System.Numerics.Vector2.Zero, ref e);
        if (p.BeforeSelection is not { } before) return;
        var old = Effect(EncodedEffectKind.Sequence, "squadron-before-selection", source: p.Entity, target: entityIds.GetValueOrDefault(before.CommandingShipRuntimeId, -1), parent: parent);
        old.N(before.CurrentHull / 16f); old.B(before.Activated); old.B(before.Commanded); old.B(before.TemporaryGrit); old.N(before.TemporarySnipeValue / 5f); old.B(before.TemporaryAssault); old.B(before.TemporarySwarm); old.N(before.TemporarySpeedOverride / 5f); old.B(before.TemporaryCannotAttack); old.B(before.TemporaryAttackOnly); old.B(before.MovementPreventedForShipActivation); old.B(before.TemporaryYavarisBonusDie); old.B(before.TemporaryCloak); old.B(before.TemporaryRogue); old.B(before.TemporaryIgnoresEngagement); old.B(before.TemporaryCommanded);
        var deferred = Effect(EncodedEffectKind.Sequence, "squadron-before-destruction-deferred", source: p.Entity, parent: parent); deferred.B(before.DestructionDeferred);
        foreach (var type in before.DefenseTokenSlots) for (int slot = 0; slot < type.Slots.Length; slot++) { var token = Effect(EncodedEffectKind.ArrayItem, "squadron-before-defense-slot", source: p.Entity, slot: slot, parent: parent, aux: (int)type.Token); token.B(type.Slots[slot].Exhausted); token.B(type.Slots[slot].Discarded); }
    }
}
