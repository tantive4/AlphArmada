using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Search;

public sealed record KnownDamageCard(int Ship, int Index, CriticalDamageCard Card);
public sealed record KnownDeckPrefix(int Deck, CriticalDamageCard[] Cards);
public sealed record SearchKnowledgeSnapshot(int Observer, ShipCommand?[][] EnemyCommands,
    KnownDamageCard[] DamageCards, KnownDeckPrefix[] DeckPrefixes);
public sealed record VisibleCommandStack(int Ship, int Owner, ShipCommand?[] Commands,
    ShipCommand[] Tokens, ShipCommand[] RaidTokens);

/// <summary>Legally remembered information, separate from authoritative recovery state.
/// Own commands are read directly; other stacks contain only explicitly learned slots.
/// This is an initial command/deck-knowledge boundary, not a full card-privacy encoder.</summary>
public sealed class SearchKnowledge
{
    private readonly ShipCommand?[][] enemyCommands;
    private readonly Dictionary<(int Ship, int Index), CriticalDamageCard> damageCards = new();
    private readonly Dictionary<int, CriticalDamageCard[]> deckPrefixes = new();
    public int Observer { get; }
    public long Version { get; private set; }

    private SearchKnowledge(int observer, ShipCommand?[][] commands)
    {
        if (observer is not (1 or 2)) throw new ArgumentOutOfRangeException(nameof(observer));
        Observer = observer; enemyCommands = commands;
    }

    public static SearchKnowledge For(GameSession position, int observer, bool observePrivateCommandLooks = true)
    {
        ArgumentNullException.ThrowIfNull(position);
        var result = new SearchKnowledge(observer,
            position.Context.State.Commands.Select(stack => new ShipCommand?[stack.Count]).ToArray());
        result.ObserveCurrentDamagePeek(position);
        if (observePrivateCommandLooks) result.ObserveCurrentCommandLook(position);
        return result;
    }

    /// <summary>Register information legally exposed by a settled decision.
    /// Call before computing traversal keys or model input, including windows
    /// reached inside a simulation. This never settles or mutates the game.
    /// Only the actual chooser learns a private looked-at damage-card prefix.</summary>
    public void ObserveDecision(GameSession position, GameDecision decision, bool observePrivateCommandLooks = true)
    {
        ArgumentNullException.ThrowIfNull(position); ArgumentNullException.ThrowIfNull(decision); Validate(position);
        if (decision.Kind == DecisionKind.Player && decision.Player == Observer && decision.Phase == GamePhase.ChooseDamageCard)
            ObserveCurrentDamagePeek(position);
        if (observePrivateCommandLooks) ObserveCurrentCommandLook(position);
    }

    private void ObserveCurrentCommandLook(GameSession position)
    {
        // Choosing Slicer Tools' target privately exposes the old top command
        // before the controller chooses replacement versus a nonmatching raid.
        if (position.Context.Flow.Continuations.LastOrDefault() is
            { Rule: "movement-choice.slicer", Source: >= 0, Target: >= 0 } frame &&
            position.Context.State.Owners[frame.Source] == Observer &&
            position.Context.State.Commands[frame.Target].TryPeekTop(out var command))
            RememberCommand(frame.Target, 0, command);
    }

    private void ObserveCurrentDamagePeek(GameSession position)
    {
        // Dodonna's current choice legally exposes the looked-at prefix to the chooser.
        // It must be retained, whereas the rest of the actual future pile is never copied.
        var flow = position.Context.Flow;
        if (flow.Phase == GamePhase.ChooseDamageCard && flow.Attacks.Count > 0)
        {
            var attack = flow.Attacks[^1];
            if (!attack.IncomingDamageCard.HasValue && attack.Attacker >= 0 &&
                position.Context.State.Owners[attack.Attacker] == Observer)
            {
                bool faceup = attack.HullCardsDealt < Math.Max(attack.FirstHullCardFaceup ? 1 : 0,
                    attack.EffectFlags.GetValueOrDefault("faceup.limit"));
                bool dodonna = Enumerable.Range(0, position.Context.State.Ships.Length).Any(ship =>
                    !position.Context.IsDestroyed(ship) && !position.Context.IsReserved(ship) &&
                    position.Context.State.Owners[ship] == Observer && position.Context.HasUpgrade(ship, "general-dodonna-commander"));
                int deck = position.Context.State.Owners[attack.Defender] - 1;
                int look = Math.Min(4, position.Context.State.Decks[deck].Remaining);
                if (faceup && dodonna && look > 1)
                {
                    RememberDeckPrefix(deck, position.Context.State.Decks[deck].PeekTop(look));
                }
            }
        }
    }

    public ShipCommand?[] CommandsFor(GameSession position, int ship)
    {
        Validate(position); CheckShip(ship);
        var stack = position.Context.State.Commands[ship];
        if (position.Context.State.Owners[ship] == Observer)
            return stack.Stack.Select(command => (ShipCommand?)command).ToArray();
        var result = new ShipCommand?[stack.Count];
        Array.Copy(enemyCommands[ship], result, Math.Min(result.Length, enemyCommands[ship].Length));
        return result;
    }

    public VisibleCommandStack[] ProjectCommands(GameSession position)
    {
        Validate(position);
        return position.Context.State.Commands.Select((stack, ship) => new VisibleCommandStack(ship,
            position.Context.State.Owners[ship], CommandsFor(position, ship), stack.TokenInstances.ToArray(), stack.RaidTokens.ToArray())).ToArray();
    }

    /// <summary>Call only following a legally observed reveal/change/look effect.</summary>
    public void RememberCommand(int ship, int slot, ShipCommand command)
    {
        CheckShip(ship);
        if ((uint)slot >= (uint)enemyCommands[ship].Length) throw new ArgumentOutOfRangeException(nameof(slot));
        if (!Enum.IsDefined(command)) throw new ArgumentOutOfRangeException(nameof(command));
        if (enemyCommands[ship][slot] == command) return;
        enemyCommands[ship][slot] = command;
        Version++;
    }
    public void RememberStack(int ship, IReadOnlyList<ShipCommand> commands)
    {
        CheckShip(ship); ArgumentNullException.ThrowIfNull(commands);
        if (enemyCommands[ship].Length == commands.Count &&
            enemyCommands[ship].Select((known, slot) => known == commands[slot]).All(equal => equal)) return;
        enemyCommands[ship] = commands.Select(command => (ShipCommand?)command).ToArray();
        Version++;
    }
    public void RememberDamageCard(int ship, int index, CriticalDamageCard card)
    {
        CheckShip(ship); if (index < 0) throw new ArgumentOutOfRangeException(nameof(index));
        damageCards[(ship, index)] = card;
        Version++;
    }
    public void RememberDeckPrefix(int deck, IReadOnlyList<CriticalDamageCard> cards)
    {
        if (deck is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(deck));
        ArgumentNullException.ThrowIfNull(cards);
        if (deckPrefixes.TryGetValue(deck, out var previous) && previous.Length == cards.Count)
        {
            bool same = true;
            for (int i = 0; i < previous.Length; i++) if (previous[i] != cards[i]) { same = false; break; }
            if (same) return;
        }
        deckPrefixes[deck] = cards.ToArray(); Version++;
    }
    internal bool TryKnownDamage(int ship, int index, out CriticalDamageCard card)
        => damageCards.TryGetValue((ship, index), out card);
    internal IReadOnlyList<CriticalDamageCard> KnownDeckPrefix(int deck)
        => deckPrefixes.TryGetValue(deck, out var cards) ? cards : Array.Empty<CriticalDamageCard>();

    // Used by a search-only materialization overlay when forced settlement has
    // consumed/cleared queued dials. It never reads a hidden command's identity.
    internal void SynchronizeCommandCounts(GameSession position)
    {
        Validate(position);
        for (int ship = 0; ship < enemyCommands.Length; ship++)
        {
            int count = position.Context.State.Commands[ship].Count; var prior = enemyCommands[ship];
            if (prior.Length == count) continue;
            var next = new ShipCommand?[count]; int removed = Math.Max(0, prior.Length - count);
            Array.Copy(prior, removed, next, 0, Math.Min(prior.Length - removed, count));
            enemyCommands[ship] = next; Version++;
        }
    }

    internal ulong AddCommandMaterializationHash(ulong hash)
    {
        hash = SearchHash.Add(hash, "command-materialization-v1");
        foreach (var stack in enemyCommands)
        {
            hash = SearchHash.Add(hash, (ulong)stack.Length);
            foreach (var command in stack)
                hash = SearchHash.Add(hash, command.HasValue ? (ulong)(int)command.Value + 1 : 0);
        }
        return hash;
    }

    /// <summary>Update from an actual or simulated committed transition. Never compares unknown
    /// command values. Call after forced settlement as well as explicit player actions.
    /// Simulations defer a private command look until their non-neural hypothesis
    /// nodes resolve unknown slots; ordinary game transitions observe real dials.</summary>
    public void ObserveTransition(GameSession before, GameAction action, GameSession after,
        bool observePrivateCommandLook = true, bool commandCountsAlreadySynchronized = false)
    {
        ArgumentNullException.ThrowIfNull(before); ArgumentNullException.ThrowIfNull(action); Validate(after);
        Version++;
        for (int ship = 0; ship < enemyCommands.Length; ship++)
        {
            int oldCount = before.Context.State.Commands[ship].Count, newCount = after.Context.State.Commands[ship].Count;
            var prior = enemyCommands[ship]; var next = new ShipCommand?[newCount];
            // Native command removals consume from the top; newly programmed dials append.
            int removed = Math.Max(0, oldCount - newCount);
            if (prior.Length == oldCount)
                Array.Copy(prior, removed, next, 0, Math.Min(prior.Length - removed, newCount));
            else if (commandCountsAlreadySynchronized && prior.Length == newCount)
                Array.Copy(prior, next, newCount);
            enemyCommands[ship] = next;

            var oldCards = before.Context.State.Ships[ship].DamageCards;
            var newCards = after.Context.State.Ships[ship].DamageCards;
            if (oldCards.Count != newCards.Count)
                foreach (var key in damageCards.Keys.Where(key => key.Ship == ship).ToArray()) damageCards.Remove(key);
            for (int i = 0; i < newCards.Count; i++)
                if (newCards[i].FaceUp && newCards[i].CriticalCard is { } faceup) damageCards[(ship, i)] = faceup;
                else if (oldCards.Count == newCards.Count && oldCards[i].FaceUp && oldCards[i].CriticalCard is { } flipped)
                    damageCards[(ship, i)] = flipped;
        }
        // These choices reveal a replacement's identity without exposing the old dial.
        bool noise = before.Context.Flow.Attacks.LastOrDefault()?.PendingDamageCard == CriticalDamageCard.CommNoise;
        if (action.Target >= 0 && action.Target < enemyCommands.Length && enemyCommands[action.Target].Length > 0 &&
            (action.AbilityId is "reveal.leia" or "slicer.command" || noise && action.Kind == GameActionKind.ChooseCommand))
            enemyCommands[action.Target][0] = action.Command;
        if (action.AbilityId.StartsWith("boarding.command", StringComparison.Ordinal) && action.Target >= 0 &&
            action.Target < enemyCommands.Length && action.Index >= 0 && action.Index < enemyCommands[action.Target].Length)
            enemyCommands[action.Target][action.Index] = action.Command;
        // Fixed pre-reveal officers encode their publicly selected command in the
        // semantic ID; the native GameAction.Command field is not populated for them.
        if (action.AbilityId.StartsWith(UnitAbilityEngine.CommandChangeActionPrefix, StringComparison.Ordinal) &&
            action.Source >= 0 && action.Source < enemyCommands.Length && enemyCommands[action.Source].Length > 0)
        {
            string encoded = action.AbilityId[UnitAbilityEngine.CommandChangeActionPrefix.Length..].Split('.', 2)[0];
            if (int.TryParse(encoded, out int command) && Enum.IsDefined((ShipCommand)command))
                enemyCommands[action.Source][0] = (ShipCommand)command;
        }
        if (action.Kind == GameActionKind.AssignCommand && action.Source >= 0 && action.Source < enemyCommands.Length &&
            before.Context.UpgradesFor(action.Source).Any(card => card.Active && card.Definition.id == "tikkes-officer" &&
                card.GetTypedCommandTokenCount(action.Command) > 0) && enemyCommands[action.Source].Length > 0)
            enemyCommands[action.Source][^1] = action.Command;
        if (observePrivateCommandLook && action.AbilityId == CommandRevealEffects.IsardAction &&
            action.Source >= 0 && action.Source < enemyCommands.Length &&
            after.Context.State.Owners[action.Source] == Observer &&
            action.Target >= 0 && action.Target < enemyCommands.Length)
            RememberStack(action.Target, after.Context.State.Commands[action.Target].Stack);
        // A draw can consume, discard or reorder looked-at cards. Only keep a prefix
        // while its deck has not changed; future look effects can establish a new one.
        foreach (int deck in deckPrefixes.Keys.ToArray())
            if (before.Context.State.Decks[deck].Remaining != after.Context.State.Decks[deck].Remaining ||
                before.Context.State.Decks[deck].Discarded != after.Context.State.Decks[deck].Discarded)
                deckPrefixes.Remove(deck);
        // Settlement may enter another private look after the deck changed.
        // Register that new current prefix after invalidating the old one.
        // The phase/chooser guard uses flow only and never queries/settles again.
        ObserveCurrentDamagePeek(after);
        if (observePrivateCommandLook) ObserveCurrentCommandLook(after);
    }

    public SearchKnowledgeSnapshot CaptureSnapshot() => new(Observer,
        enemyCommands.Select(commands => commands.ToArray()).ToArray(),
        damageCards.OrderBy(entry => entry.Key.Ship).ThenBy(entry => entry.Key.Index)
            .Select(entry => new KnownDamageCard(entry.Key.Ship, entry.Key.Index, entry.Value)).ToArray(),
        deckPrefixes.OrderBy(entry => entry.Key).Select(entry => new KnownDeckPrefix(entry.Key, entry.Value.ToArray())).ToArray());
    public SearchKnowledge DeepClone() => FromSnapshot(CaptureSnapshot());
    // An inactive opponent's private card identities are hidden from this acting
    // information set. Retain their knowledge shape, plus command memory (which
    // points at the root player's already fixed own dials), without allocations.
    internal ulong AddInactivePrivatePatternHash(ulong hash)
    {
        hash = SearchHash.Add(hash, "inactive-private-knowledge-v1");
        hash = SearchHash.Add(hash, (ulong)Observer);
        hash = SearchHash.Add(hash, (ulong)enemyCommands.Length);
        foreach (var stack in enemyCommands)
        {
            hash = SearchHash.Add(hash, (ulong)stack.Length);
            foreach (var command in stack)
                hash = SearchHash.Add(hash, command.HasValue ? (ulong)(int)command.Value + 1 : 0);
        }
        ulong sum = 0, xor = 0;
        foreach (var slot in damageCards.Keys)
        {
            ulong item = SearchHash.Add(SearchHash.Add(SearchHash.Initial, (ulong)slot.Ship), (ulong)slot.Index);
            sum = unchecked(sum + item); xor ^= System.Numerics.BitOperations.RotateLeft(item, 23);
        }
        hash = SearchHash.Add(SearchHash.Add(SearchHash.Add(hash, sum), xor), (ulong)damageCards.Count);
        sum = xor = 0;
        foreach (var entry in deckPrefixes)
        {
            ulong item = SearchHash.Add(SearchHash.Add(SearchHash.Initial, (ulong)entry.Key), (ulong)entry.Value.Length);
            sum = unchecked(sum + item); xor ^= System.Numerics.BitOperations.RotateLeft(item, 23);
        }
        return SearchHash.Add(SearchHash.Add(SearchHash.Add(hash, sum), xor), (ulong)deckPrefixes.Count);
    }
    public static SearchKnowledge FromSnapshot(SearchKnowledgeSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        var result = new SearchKnowledge(snapshot.Observer, snapshot.EnemyCommands.Select(commands => commands.ToArray()).ToArray());
        foreach (var card in snapshot.DamageCards) result.RememberDamageCard(card.Ship, card.Index, card.Card);
        foreach (var prefix in snapshot.DeckPrefixes) result.RememberDeckPrefix(prefix.Deck, prefix.Cards);
        return result;
    }
    internal void Validate(GameSession position)
    {
        if (position.Context.State.Commands.Length != enemyCommands.Length)
            throw new ArgumentException("Knowledge belongs to another entity layout.", nameof(position));
    }
    private void CheckShip(int ship)
    {
        if ((uint)ship >= (uint)enemyCommands.Length) throw new ArgumentOutOfRangeException(nameof(ship));
    }
}
