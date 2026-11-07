using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Search;

namespace Swarmada.Headless.Learning;

public sealed partial class StateEncoder
{
    private void EncodeGlobal(GameContext game, int observer, int actor)
    {
        var state = game.State; var flow = game.Flow; var w = new Writer(output.Global);
        w.N(state.Round / 6f); w.B(game.Topology.FirstPlayer == observer); w.B(actor == observer); w.B(flow.CurrentPlayer == observer); w.B(game.Topology.MapMovementEnabled); w.N(game.Topology.BoardSizeMillimetres.X / 2000); w.N(game.Topology.BoardSizeMillimetres.Y / 2000);
        w.Hot(37, Schema.Objectives.Index(game.Topology.ObjectiveId)); w.Hot(27, (int)flow.Phase); w.N(state.VictoryPoints[observer - 1] / 400f); w.N(state.VictoryPoints[2 - observer] / 400f); w.N(state.Ships.Length / 32f); w.N(state.Squadrons.Length / 128f); w.Hot(3, actor == observer ? 1 : actor == 3 - observer ? 2 : 0);
        if (state.Objective is OfficialObjectiveState objective) { w.N(objective.VictoryTokens[observer - 1] / 20f); w.N(objective.VictoryTokens[2 - observer] / 20f); } else { w.N(0); w.N(0); }
        w.B(flow.SquadronPhasePlayer == observer); w.N(flow.SquadronPhaseRemaining / 2f); w.B(flow.SquadronCanMove); w.B(flow.SquadronCanAttack); w.B(flow.SquadronCommanded); w.Hot(27, (int)flow.CommandReturnPhase); w.Hot(3, (int)flow.EffectiveCommandSpendMode);
        w.B(state.Outcome is not null); w.Hot(3, state.Outcome is null || state.Outcome.Winner == 0 ? 0 : state.Outcome.Winner == observer ? 1 : 2);
        w.N((float)(state.Outcome is null ? 0 : observer == 1 ? state.Outcome.PlayerOneScore : state.Outcome.PlayerTwoScore) / 400f); w.N((float)(state.Outcome is null ? 0 : observer == 1 ? state.Outcome.PlayerTwoScore : state.Outcome.PlayerOneScore) / 400f);
        for (int relative = 0; relative < 2; relative++) { int deck = relative == 0 ? observer - 1 : 2 - observer; w.N(state.Decks[deck].Remaining / 52f); w.N(state.Decks[deck].Discarded / 52f); }
        w.B(flow.Movement is not null); w.N(flow.ExtraManeuvers.Count / 8f); w.N(flow.ForcedManeuvers.Count / 8f); w.N(flow.GrantedSquadronAttacks.Count / 8f); w.N(flow.StandaloneSquadronActivations.Count / 8f); w.N(flow.PendingDisplacements.Length / 32f); w.N(flow.Attacks.Count / 8f); w.N(flow.Continuations.Count / 16f);
        int discardOffset = w.Position; w.Skip(44);
        for (int relative = 0; relative < 2; relative++) foreach (var card in state.Decks[relative == 0 ? observer - 1 : 2 - observer].CapturePublicDiscardPile()) w.Data[discardOffset + relative * 22 + (int)card] += 1 / 4f;
    }

    private void Activation(GameContext game, int ship, int observer, SearchKnowledge? boundary, ref Writer w)
    {
        var a = game.Flow.ShipActivations[ship]; bool privateVisible = game.State.Owners[ship] == observer && (boundary is null || boundary.Observer == observer);
        w.B(a.NavigateDial); w.B(a.NavigateToken); w.N(a.ManeuversExecuted / 4f); w.B(a.NavigateManeuverConsumed); w.N(a.NavigateTokenUse / 4f); w.B(a.ConcentrateFireRavagerChoice); w.N(a.NavigateSpeedLimit / 4f); w.B(a.CommandResolutionCompleted);
        w.Hot(6, a.PendingCommand.HasValue ? 2 + (int)a.PendingCommand.Value : 0); w.N(a.ResolvedDialCommands / 4f); w.N(a.CommandRevealStep / 8f); w.B(a.RapidLaunchReleaseChecked); for (int command = 0; command < 4; command++) w.B(privateVisible && (a.BailDialSelection & (1 << command)) != 0); w.B(a.ThrawnDialQueued); w.B(a.AdditionalDialActive); w.B(a.CurrentCommandResolved); w.Hot(5, a.CurrentResolvedCommand.HasValue ? 1 + (int)a.CurrentResolvedCommand.Value : 0); w.B(a.CommandCompletionPending); w.N(a.CommandCompletionStep / 8f); w.Hot(27, (int)a.CommandCompletionReturnPhase); w.B(a.VantoOffered); w.N(a.SquadronCommandPreludeStep / 8f); w.B(a.RaidInterferenceResolved); w.B(a.NativeDialConsumedBeforePrelude); w.N(a.DeferredSquadronActivations / 8f); w.Hot(3, (int)a.DeferredSquadronSpendMode); w.N(a.BeckCommands / 4f); w.B(a.UncommittedMatchstick); w.B(a.ShieldsToMaximumChecked);
        if (a.ActualRevealedCommand.HasValue || a.RevealPreludeActive || a.RevealWhenWindowOpened ||
            a.RevealAfterWindowOpened || a.RevealBankingClosed || a.NativeBankingDecided)
        {
            // Public reveal history is separate from whichever gained dial is
            // currently being resolved. Sparse side rows keep ShipWidth stable.
            var reveal = Effect(EncodedEffectKind.Sequence, "command-reveal-prelude", source: ship);
            reveal.B(true); reveal.B(a.ActualRevealedCommand.HasValue);
            reveal.Hot(4, a.ActualRevealedCommand.HasValue ? (int)a.ActualRevealedCommand.Value : -1);
            reveal.B(a.RevealPreludeActive); reveal.B(a.RevealWhenWindowOpened); reveal.B(a.RevealAfterWindowOpened);
            reveal.B(a.RevealBankingClosed); reveal.B(a.NativeBankingDecided);
        }
        for (int i = 0; i < a.AdditionalCommandDials.Count; i++)
        {
            var dial = a.AdditionalCommandDials[i];
            // These dials were gained from a public reveal, spent public card token,
            // or committed public choice. Their type is already known before the
            // additional dial becomes the command currently being resolved.
            bool visible = privateVisible || a.AdditionalDialActive || dial.Source is
                "Grand Admiral Thrawn" or "Bail Organa" or "Admiral Tarkin" or "Passel Argente" or "Revealed Command";
            var e = Effect(EncodedEffectKind.AdditionalDial, "additional-command-dial", dial.Source, ship, slot: i);
            e.B(visible); e.Hot(4, visible ? (int)dial.Command : -1); e.B(visible && dial.BankingDecided);
        }
        for (int i = 0; i < a.CommandedSquadrons.Count; i++) Link("commanded-squadron", ship, a.CommandedSquadrons[i], i);
        for (int i = 0; i < a.AttackedShips.Length; i++) Link("attacked-ship", ship, a.AttackedShips[i], i);
        Link("relay-keyword", ship, a.UncommittedRelayKeyword); Link("relay-title", ship, a.UncommittedRelayTitle); Link("relay-target", ship, a.UncommittedRelayTarget);
    }

    private void EncodeCards(GameContext game, int observer, SearchKnowledge? boundary)
    {
        var state = game.State;
        for (int i = 0; i < state.Upgrades.Length; i++)
        {
            var card = state.Upgrades[i]; var w = new Writer(output.Cards.AsSpan(i * CardWidth, CardWidth)); w.Hot(294, Schema.Upgrades.Index(card.Definition.id)); w.Hot(294, Schema.Upgrades.Index(card.EffectiveDefinitionId));
            int host = checked((int)state.Registers[upgradeOwners[i]] - 1); output.CardHosts[i] = host; bool own = host >= 0 && host < state.Owners.Length && state.Owners[host] == observer;
            bool boundaryVisible = boundary is null || boundary.Observer == observer;
            bool privateAssigned = card.Definition.id is "grand-admiral-thrawn-commander" or "admiral-trench-commander";
            // A simulated owner's inventory may be a fixed UCT hypothesis. Its
            // materialization marker permits that owner to observe it; it never
            // makes an enemy's unrevealed inventory visible to the other player.
            bool assignmentKnown = !privateAssigned || own && (boundaryVisible || boundary!.HasMaterializedCommanderInventory(i));
            w.B(true); w.B(!card.Exhausted && !card.Discarded); w.B(card.Exhausted); w.B(card.Discarded); w.N(card.CommandTokens / 8f); w.N(card.UsesThisActivation / 4f); w.N(card.UsesThisAttack / 4f); w.N(card.LastUsedRound / 6f);
            for (int c = 0; c < 4; c++) w.N(privateAssigned && !assignmentKnown ? 0 : card.TypedCommandTokens.GetValueOrDefault((ShipCommand)c) / 8f);
            w.Hot(7, card.FocusZone.HasValue ? 1 + (int)card.FocusZone.Value : 0); w.B(own); w.Hot(6, !assignmentKnown ? 1 : card.AssignedCommandToken.HasValue ? 2 + (int)card.AssignedCommandToken.Value : 0); w.B(card.SetupInitialized); w.B(card.HasStoredDie); w.Hot(3, card.HasStoredDie ? (int)card.StoredDieColor - 1 : -1); w.Hot(6, card.HasStoredDie ? (int)card.StoredDieFace : -1);
            // Republic Bail commander has public fixed command tokens, not the
            // round-selection mechanic of the separate Rebel Bail officer.
            const bool scheduledKnown = true;
            w.N(card.StoredUnitIds.Count / 16f); w.N(card.ScheduledRounds.Count / 6f); for (int round = 1; round <= 6; round++) w.B(card.ScheduledRounds.Contains(round));
            for (int slot = 0; slot < card.StoredUnitIds.Count; slot++)
            {
                string value = card.StoredUnitIds[slot];
                if (entityIds.TryGetValue(value, out int entity)) Store(0, true, i, entity, own ? 1 : 2, slot);
                else if (Enum.TryParse<ShipCommand>(value, true, out var command) && Enum.IsDefined(command)) Store(1, assignmentKnown, i, -1, own ? 1 : 2, slot, assignmentKnown ? (int)command : -1);
                else if (int.TryParse(value, out int encodedCommand) && Enum.IsDefined((ShipCommand)encodedCommand)) Store(1, assignmentKnown, i, -1, own ? 1 : 2, slot, assignmentKnown ? encodedCommand : -1);
                else issues.Add(new("unregistered-card-store", card.Definition.id, "Stored value has no registered entity/command codec; value is not exposed."));
            }
            for (int slot = 0; slot < card.ScheduledRounds.Count; slot++) Store(2, scheduledKnown, i, -1, own ? 1 : 2, slot, scalar: scheduledKnown ? card.ScheduledRounds[slot] / 6f : 0);
        }
    }

    private void EncodePieces(GameContext game, int observer)
    {
        var objective = game.State.Objective as OfficialObjectiveState;
        for (int i = 0; i < game.State.SpatialObjects.Count; i++)
        {
            var piece = game.State.SpatialObjects[i]; var w = new Writer(output.Pieces.AsSpan(i * PieceWidth, PieceWidth)); w.B(true); w.B(piece.Active);
            int kind = Array.IndexOf(SpatialKindValues, piece.Kind);
            if (kind < 0) issues.Add(new("spatial-kind", piece.Id, "Register native map-piece kind " + piece.Kind + " before training."));
            w.Hot(8, kind); w.B(piece.Owner == observer); w.B(piece.Owner == 3 - observer); Point(game, observer, piece.Pose.Position, ref w); Heading(observer, piece.Pose.YawDegrees, ref w); w.N(piece.Size.X / 1000); w.N(piece.Size.Y / 1000); w.B(piece.Flipped); w.N(objective?.PieceHull.GetValueOrDefault(i) / 32f ?? 0); w.B(objective?.DestroyedPieces.Contains(i) == true);
            // Payloads for mines/tokens encode native references; never feed packed longs as continuous features.
            w.B(piece.Payload == 0); if (piece.Payload != 0) PiecePayload(game, i, piece, ref w);
            int at = i * 5; output.PieceReferences[at] = piece.SourceEntity; output.PieceReferences[at + 1] = Symbol(piece.ShapeId, "piece.shape"); output.PieceReferences[at + 2] = Symbol(piece.Id, "piece.id"); output.PieceReferences[at + 3] = objective?.PieceEntities.GetValueOrDefault(i, -1) ?? -1; output.PieceReferences[at + 4] = objective?.ExogorthHosts.GetValueOrDefault(i, -1) ?? -1;
        }
        if (objective is not null)
        {
            var w = new Writer(output.Objective); w.N(objective.VictoryTokens[observer - 1] / 20f); w.N(objective.VictoryTokens[2 - observer] / 20f); w.B(objective.InvalidSetup is not null); w.N(objective.ReservedEntities.Count / 32f); w.N(objective.TimingWindows.Count / 16f);
            if (objective.InvalidSetup is { } invalid) { w.B(invalid.Player == observer); w.N((int)invalid.Code / 8f); Link("invalid-setup-entity", invalid.Entity, -1); }
            foreach (int entity in objective.ReservedEntities) Link("objective-reserved", entity, -1);
            foreach (var entry in objective.PieceKinds) { var e = Effect(EncodedEffectKind.EntityLink, "objective-piece-kind", entry.Value, target: entry.Key); e.B(true); }
        }
        else if (game.State.Objective is not null) issues.Add(new("objective-runtime", game.Topology.ObjectiveId, "Objective runtime requires an explicit encoder adapter."));
    }

    private void PiecePayload(GameContext game, int index, Swarmada.Headless.Runtime.SpatialObjectState piece, ref Writer w)
    {
        // Existing spatial payload is a stable native reference, emitted only as an integer side record.
        if (piece.Payload is >= int.MinValue and <= int.MaxValue) { var e = Effect(EncodedEffectKind.ArrayItem, "spatial-payload", source: piece.SourceEntity, target: index, aux: (int)piece.Payload); e.B(true); w.B(true); }
        else issues.Add(new("spatial-payload", piece.Id, "Packed spatial payload needs a registered codec."));
    }

    private void EncodeAttacks(GameContext game, int observer)
    {
        int dieRow = 0;
        for (int i = 0; i < game.Flow.Attacks.Count; i++)
        {
            var a = game.Flow.Attacks[i]; var w = new Writer(output.Attacks.AsSpan(i * AttackWidth, AttackWidth)); w.B(true); w.Hot(6, (int)a.Window); w.Hot(5, (int)a.Range); w.Hot(5, (int)a.DiceRange); w.Hot(6, (int)a.FromZone); w.Hot(6, (int)a.ToZone); w.B(a.Obstructed); w.B(a.Counter); w.B(a.Salvo); w.B(a.Brace); w.B(a.Scatter); w.B(a.Contain); w.B(a.Redirect); w.B(a.Evade); w.B(a.CriticalResolved); w.N(a.Damage / 32f); w.N(a.RedDice / 16f); w.N(a.BlueDice / 16f); w.N(a.BlackDice / 16f); w.Hot(27, (int)a.ResumePhase); w.B(a.EnvironmentalDamage); w.B(a.IsSalvoAttack); w.B(a.SnipeAttack); w.B(a.IgnitionAttack); w.B(a.ExtremeRange); w.B(a.IgnoreShields); w.B(a.NeedsDamageZone); w.B(a.Rolled); w.B(a.PreRollComplete); w.N(a.MandatoryRemovals / 16f); w.Hot(27, (int)a.AfterRollPhase); w.B(a.SpentEvade.HasValue); w.B(a.EvadeWasDiscarded); w.N(a.EvadeBaseCount / 8f); w.B(a.EvadeCancels); w.N(a.PendingHullCards / 32f); w.N(a.HullCardsDealt / 32f); w.B(a.FirstHullCardFaceup); w.Hot(23, a.PendingDamageCard.HasValue ? 1 + (int)a.PendingDamageCard.Value : 0);
            // Incoming facedown card identity remains hidden even while the damage machinery holds it.
            bool incomingVisible = a.HullCardsDealt < Math.Max(a.FirstHullCardFaceup ? 1 : 0, a.EffectFlags.GetValueOrDefault("faceup.limit")) || a.PendingDamageCard.HasValue; w.Hot(23, incomingVisible && a.IncomingDamageCard.HasValue ? 1 + (int)a.IncomingDamageCard.Value : 0); w.B(a.DamageStarted); w.N(a.AppliedDamage / 32f); w.B(a.ResponseStarted); w.B(a.AfterAttackWindowEntered); w.N(a.EffectStage / 16f); w.B(a.EffectOwner == observer); w.N(a.EffectMinimum / 16f); w.N(a.EffectMaximum / 16f);
            for (int type = 0; type < 6; type++) w.B(a.SpentTokenTypes.Contains((DefenseTokenType)type));
            int redirect = w.Position; w.Skip(6); foreach (var item in a.RedirectAllocations) w.Data[redirect + (int)item.Key] += item.Value / 16f;
            int at = i * 6; output.AttackReferences[at] = a.Attacker; output.AttackReferences[at + 1] = a.Defender; output.AttackReferences[at + 2] = a.MineSpatialIndex; output.AttackReferences[at + 3] = a.IntelOfficerTarget.HasValue ? entityIds.GetValueOrDefault(a.IntelOfficerTarget.Value.UnitId.Value, -1) : -1; output.AttackReferences[at + 4] = a.IntelOfficerTarget.HasValue ? (int)a.IntelOfficerTarget.Value.Token.Token : -1; output.AttackReferences[at + 5] = a.IntelOfficerTarget?.Token.Slot ?? -1;
            for (int slot = 0; slot < a.Dice.Length; slot++) { var die = a.Dice[slot]; var dw = new Writer(output.Dice.AsSpan(dieRow * DieWidth, DieWidth)); dw.B(true); dw.Hot(3, (int)die.Color - 1); dw.Hot(6, a.Rolled ? (int)die.Face : -1); dw.B(a.Rolled); dw.B(a.PendingRollIndices.Contains(slot)); dw.B(a.EffectIndices.Contains(slot)); dw.B(a.EffectEligibleIndices.Contains(slot)); output.DieReferences[dieRow * 2] = i; output.DieReferences[dieRow * 2 + 1] = slot; dieRow++; }
            for (int slot = 0; slot < a.CriticalChosenZones.Length; slot++) { var e = Effect(EncodedEffectKind.ArrayItem, "critical-chosen-zone", source: a.Defender, slot: slot, parent: i); e.Hot(6, a.CriticalChosenZones[slot]); }
            for (int slot = 0; slot < a.EffectIndices.Length; slot++) { var e = Effect(EncodedEffectKind.ArrayItem, "attack-effect-index", source: a.EffectIndices[slot], slot: slot, parent: i); e.B(true); }
            for (int slot = 0; slot < a.EffectEligibleIndices.Length; slot++) { var e = Effect(EncodedEffectKind.ArrayItem, "attack-eligible-index", source: a.EffectEligibleIndices[slot], slot: slot, parent: i); e.B(true); }
            if (a.ParentResumePhase.HasValue) { var e = Effect(EncodedEffectKind.ArrayItem, "attack-parent-resume-phase", parent: i, aux: (int)a.ParentResumePhase.Value); e.B(true); }
            if (a.SpentEvade.HasValue) { var e = Effect(EncodedEffectKind.ArrayItem, "spent-evade-token", source: a.Defender, slot: a.SpentEvade.Value.Slot, parent: i, aux: (int)a.SpentEvade.Value.Token); e.B(a.SpentEvade.Value.Exhausted); }
            foreach (var token in a.LockedTokens) { var e = Effect(EncodedEffectKind.ArrayItem, "locked-defense-token", source: a.Defender, slot: token.Slot, parent: i, aux: (int)token.Token); e.B(token.Exhausted); }
            foreach (var entry in a.EffectFlags) Flag(EncodedEffectKind.AttackFlag, entry.Key, entry.Value, game, observer, parent: i);
            if (a.EffectRule.Length > 0) { var e = Effect(EncodedEffectKind.Continuation, a.EffectRule, a.EffectActionId, a.Attacker, a.Defender, a.EffectStage, i, a.EffectValue); e.B(true); }
            if (a.DamageRule.Length > 0) { var e = Effect(EncodedEffectKind.Continuation, a.DamageRule, source: a.Defender, parent: i); e.B(true); }
        }
    }
}
