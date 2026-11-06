using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Search;

/// <summary>Checkpoint format. Ordinary MCTS nodes retain edges/statistics instead of this graph.</summary>
public sealed record SearchRecoverySnapshot(string Schema, string TopologyStamp, GameSnapshot Engine,
    SearchKnowledgeSnapshot Knowledge, string[] DecisionPrefix);
public sealed record RestoredSearchContext(SearchKnowledge Knowledge, string[] DecisionPrefix);
public readonly record struct HiddenCommandSlot(int Ship, int Slot, int Observer, int Owner, int AllowedCommandMask = 15);
/// <summary>Optional search-owned hypothesis chooser. It receives no authoritative hidden value.</summary>
public interface IHiddenCommandSampler
{
    ShipCommand Sample(HiddenCommandSlot slot, ref DeterministicRandom random);
}

public static class SearchRecovery
{
    public const string Schema = "swarmada-headless.search-recovery.v1";
    public static SearchRecoverySnapshot Capture(GameSession position, SearchKnowledge knowledge, IEnumerable<string>? decisionPrefix = null)
    {
        ArgumentNullException.ThrowIfNull(position); ArgumentNullException.ThrowIfNull(knowledge); knowledge.Validate(position);
        return new(Schema, TopologyStamp(position), position.CaptureSnapshot(), knowledge.CaptureSnapshot(), decisionPrefix?.ToArray() ?? []);
    }
    /// <summary>The returned prefix must be restored by the policy adapter against the restored position.</summary>
    public static RestoredSearchContext Restore(GameSession position, SearchRecoverySnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(position); ArgumentNullException.ThrowIfNull(snapshot);
        if (snapshot.Schema != Schema || snapshot.TopologyStamp != TopologyStamp(position))
            throw new ArgumentException("Search checkpoint schema/topology differs from this match.", nameof(snapshot));
        var knowledge = SearchKnowledge.FromSnapshot(snapshot.Knowledge); knowledge.Validate(position);
        var prefix = snapshot.DecisionPrefix.ToArray();
        position.RestoreSnapshot(snapshot.Engine);
        return new(knowledge, prefix);
    }
    public static string TopologyStamp(GameSession position)
    {
        var topology = position.Context.Topology;
        string descriptor = JsonSerializer.Serialize(new
        {
            position.Context.State.SchemaId, topology.MatchId, topology.Ruleset, topology.FirstPlayer,
            topology.ObjectiveId, topology.ScenarioId, topology.MapMovementEnabled,
            BoardX = topology.BoardSizeMillimetres.X, BoardY = topology.BoardSizeMillimetres.Y,
            Ships = topology.Ships.Select(ship => new { ship.RuntimeId, ship.Owner, ship.CatalogEntry.Definition.ShipId,
                X = ship.FootprintMillimetres.X, Y = ship.FootprintMillimetres.Y }),
            Squadrons = topology.Squadrons.Select(squad => new { squad.RuntimeId, squad.Owner, squad.CatalogEntry.Definition.SquadronId }),
            topology.Upgrades, Registers = topology.Registers.Names
        });
        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(descriptor)));
    }
}

/// <summary>Root-sampling boundary for the current mock-search implementation.
/// Diagnostic roots can sample unknown enemy commands uniformly over legal commands.
/// Deferred-reveal roots instead use neutral legal placeholders until the search
/// wrapper explicitly resolves a hidden dial at its reveal event. Neither is a learned belief.
/// Future decks and hidden facedown damage identities are sampled independently of
/// authoritative order/RNG. This does not solve strategy fusion or opponent-safe ISMCTS.</summary>
public static class SearchRootFactory
{
    private static readonly CriticalDamageCard[] CanonicalCards = new DamageDeckState(0).CaptureSnapshot().DrawPile.Order().ToArray();
    private static readonly string[] UnsupportedPrivateCards = ["grand-admiral-thrawn-commander", "admiral-trench-commander",
        "general-draven-commander", "bail-organa-commander"];
    private static readonly string[] FixedCommandOfficers = ["navigation-officer", "engineering-captain-officer",
        "tactical-expert-officer", "wing-commander-officer"];

    public static GameSession Create(GameSession authority, SearchKnowledge knowledge, ulong seed,
        IHiddenCommandSampler? commandSampler = null, bool deferUnknownCommands = false)
    {
        var result = authority.DeepClone(); PopulateWorkspace(result, authority, knowledge, seed, commandSampler, deferUnknownCommands); return result;
    }
    public static void PopulateWorkspace(GameSession workspace, GameSession authority, SearchKnowledge knowledge, ulong seed,
        IHiddenCommandSampler? commandSampler = null, bool deferUnknownCommands = false)
    {
        ArgumentNullException.ThrowIfNull(workspace); ArgumentNullException.ThrowIfNull(authority); ArgumentNullException.ThrowIfNull(knowledge);
        if (ReferenceEquals(workspace, authority)) throw new ArgumentException("Search workspace cannot be the authoritative game.", nameof(workspace));
        knowledge.Validate(authority); RejectUnsupportedPrivateCards(authority);
        workspace.CopyFrom(authority);
        var random = new DeterministicRandom(seed);
        for (int ship = 0; ship < workspace.Context.State.Commands.Length; ship++)
        {
            if (workspace.Context.State.Owners[ship] == knowledge.Observer) continue;
            var known = knowledge.CommandsFor(authority, ship);
            var allowed = Enum.GetValues<ShipCommand>().Where(command => CommandModule.CommandAllowed(workspace.Context, ship, command)).ToArray();
            int allowedMask = allowed.Aggregate(0, (mask, command) => mask | (1 << (int)command));
            if (known.Length > 0 && allowed.Length == 0) throw new InvalidOperationException("A ship has queued commands but no legal command hypotheses.");
            for (int slot = 0; slot < known.Length; slot++)
            {
                var command = known[slot] ?? (deferUnknownCommands ? allowed[0] : commandSampler is null ? allowed[random.NextInt(allowed.Length)] :
                    commandSampler.Sample(new(ship, slot, knowledge.Observer, workspace.Context.State.Owners[ship], allowedMask), ref random));
                if (!Enum.IsDefined(command) || (allowedMask & (1 << (int)command)) == 0)
                    throw new InvalidOperationException("Search hypothesis chooser returned an invalid or forbidden command.");
                workspace.Context.State.Commands[ship].ReplaceAt(slot, command);
            }
        }
        SampleDecks(workspace, authority, knowledge, ref random);
        // Actual game RNG is secret future information even if command/deck fields are hidden.
        workspace.Context.State.Random = new DeterministicRandom(random.NextUInt64());
    }
    private static void RejectUnsupportedPrivateCards(GameSession position)
    {
        foreach (var card in position.Context.State.Upgrades)
            if (!card.Discarded && UnsupportedPrivateCards.Contains(card.Definition.id, StringComparer.Ordinal))
                throw new NotSupportedException("Search belief coverage is not registered for private commander-card state: " + card.Definition.id);
    }
    /// <summary>Deferred dials are placeholders before reveal. Do not let an unregistered
    /// private look/replacement read a placeholder or overwrite a prior simulated choice.</summary>
    public static void ValidateDeferredDecision(GameSession position, SearchKnowledge rootKnowledge, GameDecision? settledDecision = null)
    {
        ArgumentNullException.ThrowIfNull(position); ArgumentNullException.ThrowIfNull(rootKnowledge);
        bool UnknownEnemy(int ship) => ship >= 0 && ship < position.Context.State.Commands.Length &&
            position.Context.State.Owners[ship] != rootKnowledge.Observer && rootKnowledge.CommandsFor(position, ship).Any(command => command is null);
        bool UnknownEnemyTop(int ship)
        {
            if (ship < 0 || ship >= position.Context.State.Commands.Length || position.Context.State.Owners[ship] == rootKnowledge.Observer) return false;
            var commands = rootKnowledge.CommandsFor(position, ship); return commands.Length > 0 && !commands[0].HasValue;
        }
        void CheckFixedOfficer(int ship)
        {
            if (!UnknownEnemyTop(ship)) return;
            var officer = position.Context.UpgradesFor(ship).FirstOrDefault(card => card.Active && FixedCommandOfficers.Contains(card.Definition.id, StringComparer.Ordinal));
            if (officer is not null)
                throw new SearchCoverageException("Deferred-reveal command knowledge needs a registered pre-reveal fixed-command interaction: " + officer.Definition.id);
        }
        // A neutral Navigate placeholder can suppress Navigation Officer's native offer
        // entirely. Inspect active cards at the actual pre-reveal boundary as well as
        // offered actions, before the unknown dial is turned into a reveal hypothesis.
        var flow = position.Context.Flow;
        if (flow.Phase == GamePhase.RevealCommand && flow.ActiveShip >= 0 &&
            !flow.ShipActivations[flow.ActiveShip].RevealedCommand.HasValue) CheckFixedOfficer(flow.ActiveShip);
        if (flow.Continuations.LastOrDefault() is { Rule: "effects.window" } window &&
            window.Text is "start-activation" or "before-command-reveal" or "before-reveal-command") CheckFixedOfficer(window.Source);
        var decision = settledDecision ?? position.GetDecision();
        foreach (var option in decision.Actions)
        {
            var action = option.Template; string id = action.AbilityId;
            bool commandChange = id.StartsWith(UnitAbilityEngine.CommandChangeActionPrefix, StringComparison.Ordinal) && UnknownEnemyTop(action.Source);
            bool liaisonBeforeReveal = id.StartsWith(UnitAbilityEngine.LiaisonActionPrefix, StringComparison.Ordinal) && UnknownEnemyTop(action.Source) &&
                !position.Context.Flow.ShipActivations[action.Source].RevealedCommand.HasValue;
            bool privateBoarding = id == "boarding.command" && UnknownEnemy(action.Target);
            bool privateLook = id.Contains("isard", StringComparison.OrdinalIgnoreCase) &&
                Enumerable.Range(0, position.Context.State.Commands.Length).Any(UnknownEnemy);
            if (commandChange || liaisonBeforeReveal || privateBoarding || privateLook)
                throw new SearchCoverageException("Deferred-reveal command knowledge needs a registered private look/replacement interaction: " + id);
        }
    }
    private static void SampleDecks(GameSession workspace, GameSession authority, SearchKnowledge knowledge, ref DeterministicRandom random)
    {
        for (int deck = 0; deck < workspace.Context.State.Decks.Length; deck++)
        {
            // The authoritative draw pile is deliberately never inspected here.
            var discarded = authority.Context.State.Decks[deck].CapturePublicDiscardPile();
            var pool = new List<CriticalDamageCard>(CanonicalCards);
            foreach (var card in discarded) Remove(pool, card);
            var hiddenSlots = new List<(int Ship, int Index)>();
            for (int ship = 0; ship < workspace.Context.State.Ships.Length; ship++)
            {
                if (workspace.Context.State.Owners[ship] != deck + 1) continue;
                var cards = authority.Context.State.Ships[ship].DamageCards;
                for (int index = 0; index < cards.Count; index++)
                {
                    if (cards[index].FaceUp && cards[index].CriticalCard is { } faceup) Remove(pool, faceup);
                    else if (knowledge.TryKnownDamage(ship, index, out var known)) Remove(pool, known);
                    else hiddenSlots.Add((ship, index));
                }
            }
            foreach (var attack in workspace.Context.Flow.Attacks)
                if (attack.IncomingDamageCard is { } incoming && workspace.Context.State.Owners[attack.Defender] == deck + 1) Remove(pool, incoming);
            var prefix = knowledge.KnownDeckPrefix(deck);
            foreach (var card in prefix) Remove(pool, card);
            for (int index = pool.Count - 1; index > 0; index--)
            {
                int other = random.NextInt(index + 1); (pool[index], pool[other]) = (pool[other], pool[index]);
            }
            int next = 0;
            foreach (var group in hiddenSlots.GroupBy(slot => slot.Ship))
            {
                var snapshot = workspace.Context.State.Ships[group.Key].CaptureSnapshot();
                var cards = snapshot.DamageCards.ToArray();
                foreach (var slot in group)
                {
                    if (next >= pool.Count) throw new InvalidOperationException("Public damage records exceed the canonical deck composition.");
                    cards[slot.Index] = new(false, pool[next++]);
                }
                workspace.Context.State.Ships[group.Key].RestoreSnapshot(snapshot with { DamageCards = cards });
            }
            // Also restore remembered facedown identities rather than copying a changed secret.
            foreach (var known in knowledge.CaptureSnapshot().DamageCards)
                if (workspace.Context.State.Owners[known.Ship] == deck + 1)
                {
                    var snapshot = workspace.Context.State.Ships[known.Ship].CaptureSnapshot();
                    if ((uint)known.Index < (uint)snapshot.DamageCards.Length && !snapshot.DamageCards[known.Index].FaceUp)
                    {
                        snapshot.DamageCards[known.Index] = new(false, known.Card);
                        workspace.Context.State.Ships[known.Ship].RestoreSnapshot(snapshot);
                    }
                }
            int remaining = authority.Context.State.Decks[deck].Remaining;
            if (prefix.Count > remaining || remaining - prefix.Count > pool.Count - next)
                throw new InvalidOperationException("Deck count and legally known damage-card constraints are inconsistent.");
            var draw = new CriticalDamageCard[remaining];
            for (int index = 0; index < prefix.Count; index++) draw[index] = prefix[index];
            for (int index = prefix.Count; index < remaining; index++) draw[index] = pool[next++];
            workspace.Context.State.Decks[deck].RestoreSnapshot(new(draw, discarded));
        }
    }
    private static void Remove(List<CriticalDamageCard> pool, CriticalDamageCard card)
    {
        if (!pool.Remove(card)) throw new InvalidOperationException("Public damage records exceed the canonical deck composition for " + card);
    }
}
