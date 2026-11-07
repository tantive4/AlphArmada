using System.Diagnostics;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Search;

public enum SearchInformationMode
{
    /// <summary>Explicit testing mode only; permits full-state lookahead.</summary>
    PerfectInformationTest,
    /// <summary>Sample unknown commands/decks per simulation. Approximate root
    /// determinization; does not claim complete information-set search.</summary>
    UniformRootSampling,
    /// <summary>Do not assign unknown enemy commands until reveal. Pure UCT treats
    /// each first reveal/private look as an opponent-owned hypothetical choice,
    /// a pessimistic approximation. Previously seen slots remain fixed.</summary>
    DeferredUctReveal,
    /// <summary>Same delayed assignment, but uniformly average possible reveal outcomes.</summary>
    DeferredChanceReveal
}

public sealed record NativeSearchTimings(long Resets, double ResetMilliseconds,
    long Encodings, double EncodingMilliseconds, double CatalogMilliseconds,
    double ApplyMilliseconds, long CandidatesValidated, long CatalogsBuilt,
    long KeyBuilds = 0, double KeyMilliseconds = 0, long DescriptorBindings = 0);

/// <summary>Small diagnostic encoder, deliberately not the production feature schema.
/// It excludes future deck order, RNG, opponent unknown dials and private flow stores.
/// Its hash is a local stochastic-outcome discriminator, not a transposition key.</summary>
public sealed class MockSearchEncoder : ISearchObservationEncoder
{
    public string SchemaVersion => "diagnostic-mock-v1";
    private float[]? cachedBoard;
    private ulong cachedBoardKey;
    private long generation = -1;
    private int cachedActor;
    private SearchKnowledge? cachedKnowledge, cachedHypothesisKnowledge;
    private long cachedKnowledgeVersion, cachedHypothesisVersion;

    public void Invalidate() { generation = -1; cachedBoard = null; }

    public SearchObservation Encode(GameSession position, SearchKnowledge knowledge,
        PolicyDecision decision, IReadOnlyList<string> prefix, SearchKnowledge? hypothesisKnowledge = null)
    {
        if (decision.Kind != DecisionKind.Player) throw new ArgumentException("Only player decisions need mock observations.");
        if (cachedBoard is null || generation != position.PositionGeneration || cachedActor != decision.Actor ||
            !ReferenceEquals(cachedKnowledge, knowledge) || cachedKnowledgeVersion != knowledge.Version ||
            !ReferenceEquals(cachedHypothesisKnowledge, hypothesisKnowledge) ||
            cachedHypothesisVersion != (hypothesisKnowledge?.Version ?? 0))
        {
            var state = position.Context.State; var flow = position.Context.Flow; var board = new List<float>();
            board.AddRange([state.Round, position.Context.Topology.FirstPlayer, flow.CurrentPlayer,
                flow.ActiveShip, flow.ActiveSquadron, state.VictoryPoints[0], state.VictoryPoints[1]]);
            board.AddRange([(int)flow.Phase, state.Ships.Length, state.Squadrons.Length, state.Upgrades.Length]);
            for (int entity = 0; entity < state.Poses.Length; entity++)
            {
                var pose = state.Poses[entity];
                board.AddRange([state.Owners[entity], pose.Position.X, pose.Position.Y, pose.YawDegrees,
                    state.Activated[entity] ? 1 : 0]);
            }
            foreach (var ship in state.Ships)
            {
                var snapshot = ship.CaptureSnapshot();
                board.AddRange([snapshot.CurrentHull, snapshot.MaximumHull, snapshot.CurrentSpeed, snapshot.DamageCards.Length]);
                foreach (var zone in snapshot.Shields) board.AddRange([(int)zone.Key, zone.Value]);
                foreach (var type in snapshot.DefenseTokenSlots)
                    for (int slot = 0; slot < type.Slots.Length; slot++)
                    {
                        var token = type.Slots[slot];
                        board.AddRange([(int)type.Token, slot, token.Exhausted ? 1 : 0, token.Discarded ? 1 : 0]);
                    }
                foreach (var card in snapshot.DamageCards)
                    board.Add(card.FaceUp && card.CriticalCard is { } visible ? (int)visible + 1 : 0);
            }
            foreach (var squadron in state.Squadrons)
            {
                var snapshot = squadron.CaptureSnapshot();
                board.AddRange([snapshot.CurrentHull, snapshot.Activated ? 1 : 0, snapshot.Commanded ? 1 : 0,
                    snapshot.TemporaryCloak ? 1 : 0]);
            }
            foreach (var stack in knowledge.ProjectCommands(position))
            {
                var hypothetical = hypothesisKnowledge?.CommandsFor(position, stack.Ship);
                board.AddRange([stack.Ship, stack.Commands.Length]);
                for (int slot = 0; slot < stack.Commands.Length; slot++)
                {
                    var command = stack.Commands[slot];
                    if (hypothetical is not null && !hypothetical[slot].HasValue) command = null;
                    board.Add(command.HasValue ? (int)command.Value : -1);
                }
                board.Add(stack.Tokens.Length); foreach (var token in stack.Tokens) board.Add((int)token);
                board.Add(stack.RaidTokens.Length); foreach (var raid in stack.RaidTokens) board.Add((int)raid);
            }
            // A legal card peek is private observation, not unknown future order.
            for (int deck = 0; deck < state.Decks.Length; deck++)
            {
                var known = knowledge.KnownDeckPrefix(deck);
                if (known.Count > 0)
                {
                    board.AddRange([-2, deck, known.Count]); foreach (var card in known) board.Add((int)card);
                }
            }
            foreach (var card in state.Upgrades)
            {
                ulong identity = SearchHash.Add(SearchHash.Initial, card.Definition.id);
                board.AddRange([(uint)identity, (uint)(identity >> 32), card.Exhausted ? 1 : 0,
                    card.Discarded ? 1 : 0, card.CommandTokens, card.UsesThisActivation, card.UsesThisAttack]);
            }
            foreach (var spatial in state.SpatialObjects)
                board.AddRange([spatial.Kind, spatial.Owner, spatial.Pose.Position.X, spatial.Pose.Position.Y,
                    spatial.Pose.YawDegrees, spatial.Active ? 1 : 0, spatial.Flipped ? 1 : 0]);
            foreach (var activation in flow.ShipActivations)
                board.AddRange([activation.AttacksResolved, activation.ManeuverComplete ? 1 : 0,
                    activation.RevealedCommand.HasValue ? (int)activation.RevealedCommand.Value : -1,
                    activation.RevealedDialAvailable ? 1 : 0, activation.EngineeringPoints, activation.CommandedSquadronsRemaining]);
            foreach (var attack in flow.Attacks)
            {
                board.AddRange([attack.Attacker, attack.Defender, (int)attack.FromZone, (int)attack.ToZone,
                    attack.Damage, attack.PendingHullCards,
                    attack.PendingDamageCard.HasValue ? (int)attack.PendingDamageCard.Value : -1]);
                foreach (var die in attack.Dice) board.AddRange([(int)die.Color, (int)die.Face]);
            }
            ulong publicContextHash = AddPublicContext(position.Context, board);
            cachedBoard = board.ToArray(); cachedBoardKey = SearchHash.Add(SearchHash.Add(SearchHash.Initial, cachedBoard), publicContextHash);
            generation = position.PositionGeneration; cachedActor = decision.Actor;
            cachedKnowledge = knowledge; cachedKnowledgeVersion = knowledge.Version;
            cachedHypothesisKnowledge = hypothesisKnowledge; cachedHypothesisVersion = hypothesisKnowledge?.Version ?? 0;
        }
        ulong stageCode = SearchHash.Add(SearchHash.Initial, decision.Stage);
        var context = new List<float> { decision.Actor, (int)decision.Phase, prefix.Count,
            (uint)stageCode, (uint)(stageCode >> 32) };
        ulong information = SearchHash.Add(SearchHash.Add(cachedBoardKey, decision.Stage), (ulong)decision.Actor);
        foreach (string part in prefix)
        {
            ulong field = SearchHash.Add(SearchHash.Initial, part);
            context.Add((uint)field); context.Add((uint)(field >> 32)); information = SearchHash.Add(information, part);
        }
        var legal = decision.Choices.Select(choice => choice.Id).ToArray();
        foreach (string id in legal) information = SearchHash.Add(information, id);
        string head = decision.Stage.Split('/')[0];
        return new(head, decision.Actor, cachedBoardKey, information,
            cachedBoard!, context.ToArray(), legal, decision.Stage == "hidden_command/uct_reveal");
    }

    /// <summary>Public continuation/commitment context only. Generic frame Value,
    /// Text and Indices may carry hidden command information, so are not serialized.
    /// Keep an exact hash alongside lossy mock float features for string identities.</summary>
    private static ulong AddPublicContext(GameContext game, List<float> features)
    {
        ulong hash = SearchHash.Initial;
        void Number(int value)
        { features.Add(value); hash = SearchHash.Add(hash, unchecked((ulong)(long)value)); }
        void Scalar(float value)
        { features.Add(value); hash = SearchHash.Add(hash, unchecked((uint)BitConverter.SingleToInt32Bits(value))); }
        void Identity(string value)
        {
            ulong identity = SearchHash.Add(SearchHash.Initial, value);
            features.Add((uint)identity); features.Add((uint)(identity >> 32)); hash = SearchHash.Add(hash, value);
        }
        void Flag(bool value) => Number(value ? 1 : 0);
        var flow = game.Flow;
        Number(flow.Continuations.Count);
        foreach (var frame in flow.Continuations)
        {
            Identity(frame.Rule); Number(frame.Stage); Number(frame.Source); Number(frame.Target);
            // This window's Text is a predefined public timing name. Other Text
            // payloads are deliberately excluded instead of guessed to be public.
            if (frame.Rule == "effects.window") Identity(frame.Text);
        }
        Number((int)flow.CommandReturnPhase); Number(flow.PlanningShip); Number(flow.DisplacementShip);
        Number(flow.PendingDisplacements.Length); foreach (int entity in flow.PendingDisplacements) Number(entity);
        Number(flow.SquadronPhasePlayer); Number(flow.SquadronPhaseRemaining);
        Flag(flow.SquadronCanMove); Flag(flow.SquadronCanAttack); Flag(flow.SquadronCommanded);
        foreach (var activation in flow.ShipActivations)
        {
            Number(activation.CommandRevealStep); Number(activation.CommandCompletionStep);
            Number((int)activation.CommandCompletionReturnPhase); Flag(activation.CommandCompletionPending);
            Flag(activation.CommandResolutionCompleted); Flag(activation.CurrentCommandResolved);
            Flag(activation.RaidInterferenceResolved); Flag(activation.NavigateSpeedUsed);
            Flag(activation.NavigateManeuverConsumed); Flag(activation.CurrentResolvedCommand.HasValue);
            // CurrentResolvedCommand is public: it exists only for a resolved command.
            Number(activation.CurrentResolvedCommand.HasValue ? (int)activation.CurrentResolvedCommand.Value : -1);
        }
        foreach (var attack in flow.Attacks)
        {
            Number((int)attack.ResumePhase); Number((int)attack.Window); Identity(attack.EffectRule);
            Number(attack.EffectStage); Number(attack.EffectOwner); Number(attack.EffectMinimum); Number(attack.EffectMaximum);
            Flag(attack.DamageStarted); Flag(attack.ResponseStarted); Flag(attack.AfterAttackWindowEntered);
            Flag(attack.NeedsDamageZone); Flag(attack.Rolled); Flag(attack.PreRollComplete);
            Number(attack.AppliedDamage); Number(attack.HullCardsDealt);
            Number(attack.CriticalChosenZones.Length); foreach (int zone in attack.CriticalChosenZones) Number(zone);
            Number(attack.EffectIndices.Length); foreach (int index in attack.EffectIndices) Number(index);
        }
        Flag(flow.Movement is not null);
        if (flow.Movement is { } movement)
        {
            Number(movement.Ship); Number(movement.Stage); Number(movement.DialedSpeed); Number((int)movement.GuideSide);
            Scalar(movement.Start.Position.X); Scalar(movement.Start.Position.Y); Scalar(movement.Start.YawDegrees);
            Scalar(movement.End.Position.X); Scalar(movement.End.Position.Y); Scalar(movement.End.YawDegrees);
            Number(movement.PlottedCourse.Length); foreach (int click in movement.PlottedCourse) Number(click);
            Number(movement.ExecutedCourse.Length); foreach (int click in movement.ExecutedCourse) Number(click);
            Number(movement.CollidedShip); Number(movement.DamageEffectIndex); Number(movement.MineIndex);
            Flag(movement.EndsOutsidePlayArea); Flag(movement.AfterManeuverDamageResolved);
            Flag(movement.ObjectiveChoiceResolved); Flag(movement.ObjectiveAccepted);
        }
        return hash;
    }
}

public sealed class NativeMctsEnvironment : ICachedMctsEnvironment
{
    // A private look is resolved outside the native action/policy space. Each
    // unknown slot is selected once by pure UCT, then fixed for this continuation.
    private sealed record PrivateCommandLook(int Observer, int Ship, int[] Slots, int Cursor = 0,
        bool LegallyObserved = true);
    private sealed record ObserverKnowledgePattern(int Observer, bool[][] Commands,
        (int Ship, int Index)[] DamageSlots, (int Deck, int Count)[] DeckPrefixes);
    private sealed record CachedDecision(PolicyDecision Decision, DecisionAdapter.DecisionDescriptor? Native,
        GameAction? Reveal, ShipCommand[]? Commands, PrivateCommandLook? Look);
    private readonly GameSession root, before;
    private readonly SearchKnowledge rootKnowledge;
    private readonly SearchInformationMode mode;
    private readonly ISearchObservationEncoder encoder;
    private readonly ObserverKnowledgePattern? opposingKnowledgePattern;
    private SearchKnowledge[] knowledge;
    // Simulation world allocation is separate from legal observation. In
    // particular, an owner's pre-reveal hypothesis must not become enemy seen.
    private SearchKnowledge materializedCommands;
    private readonly HashSet<int> handledPreRevealWindows = [];
    private GameDecision? settled;
    private long settledGeneration = -1, settledRevision = -1, keyGeneration = -1;
    private long keyFirstKnowledgeVersion = -1, keySecondKnowledgeVersion = -1;
    private int keyActor = -1;
    private PolicyDecision? current;
    private GameAction? hypotheticalReveal;
    private ShipCommand[]? hypotheticalCommands;
    private PrivateCommandLook? pendingPrivateLook;
    private SearchTraversalKey baseKey;
    private bool prepared;
    private long keyBuilds, keyTicks, descriptorBindings;
    private long resets, encodings, resetTicks, encodeTicks, catalogTicks, applyTicks;
    public GameSession Position { get; }
    public DecisionAdapter Adapter { get; }
    public int RootObserver => rootKnowledge.Observer;
    // The diagnostic uniform-world mode retains its previous observation-based
    // traversal until world-dependent pre-reveal legality has a separate cache.
    public bool SupportsCachedTraversal => mode != SearchInformationMode.UniformRootSampling;
    public string CompatibilityToken { get; }
    private bool Deferred => mode is SearchInformationMode.DeferredUctReveal or SearchInformationMode.DeferredChanceReveal;

    public NativeMctsEnvironment(GameSession game, SearchKnowledge rootKnowledge,
        SearchInformationMode mode, ActionCatalogProfile? profile = null, ISearchObservationEncoder? encoder = null,
        IReadOnlyList<SearchKnowledge>? observerKnowledge = null)
    {
        ArgumentNullException.ThrowIfNull(game); ArgumentNullException.ThrowIfNull(rootKnowledge);
        if (!Enum.IsDefined(mode)) throw new ArgumentOutOfRangeException(nameof(mode));
        game.GetDecision(); this.root = game.DeepClone(); this.rootKnowledge = rootKnowledge.DeepClone();
        this.mode = mode; Position = game.DeepClone(); before = game.DeepClone(); Adapter = new(Position, profile);
        this.encoder = encoder ?? new MockSearchEncoder();
        if (observerKnowledge is not null)
        {
            if (observerKnowledge.Count != 2 || observerKnowledge.Select(view => view.Observer).Distinct().Count() != 2)
                throw new ArgumentException("Provide one knowledge history for each player.", nameof(observerKnowledge));
            foreach (var view in observerKnowledge) view.Validate(game);
            var other = observerKnowledge.Single(view => view.Observer != RootObserver).CaptureSnapshot();
            // Preserve who has seen which slot, never their actual private values.
            // Those values are rebound to the independently sampled search world.
            opposingKnowledgePattern = new(other.Observer,
                other.EnemyCommands.Select(stack => stack.Select(command => command.HasValue).ToArray()).ToArray(),
                other.DamageCards.Select(card => (card.Ship, card.Index)).ToArray(),
                other.DeckPrefixes.Select(prefix => (prefix.Deck, prefix.Cards.Length)).ToArray());
        }
        knowledge = CreateKnowledge(bindOpponent: false);
        materializedCommands = this.rootKnowledge.DeepClone();
        Position.Context.SuspendTimingWindow = SuspendPreRevealWindow;
        CompatibilityToken = SearchRecovery.TopologyStamp(game) + ":" + mode + ":" + RootObserver + ":" +
            (profile ?? new ActionCatalogProfile()) + ":" + this.encoder.SchemaVersion;
    }
    public PolicyDecision Current
    {
        get
        {
            long start = Stopwatch.GetTimestamp();
            Prepare();
            if (current is not null) return current;
            PolicyDecision decision;
            if (pendingPrivateLook is { } look)
            {
                var choices = RevealCommands(look.Ship).Select(command =>
                    new PolicyChoice("hidden_command:" + (int)command, command.ToString())).ToArray();
                decision = new(DecisionKind.Player, Position.Context.State.Owners[look.Ship],
                    look.LegallyObserved ? "hidden_command/uct_private_look" : "hidden_command/uct_pre_reveal",
                    choices, GamePhase.AbilityChoice);
            }
            else if (hypotheticalReveal is { } reveal)
            {
                var choices = hypotheticalCommands!.Select(command =>
                    new PolicyChoice("hidden_command:" + (int)command, command.ToString())).ToArray();
                decision = new(mode == SearchInformationMode.DeferredUctReveal ? DecisionKind.Player : DecisionKind.Chance,
                    mode == SearchInformationMode.DeferredUctReveal ? RevealOwner(reveal) : 0,
                    mode == SearchInformationMode.DeferredUctReveal ? "hidden_command/uct_reveal" : "hidden_command/chance_reveal",
                    choices, GamePhase.RevealCommand);
            }
            else decision = Adapter.Current;
            catalogTicks += Stopwatch.GetTimestamp() - start; return current = decision;
        }
    }
    public bool InBufferedPrefix => pendingPrivateLook is not null || Adapter.PrefixCount > 0;
    public NativeSearchTimings Timings => new(resets, Milliseconds(resetTicks), encodings, Milliseconds(encodeTicks),
        Milliseconds(catalogTicks), Milliseconds(applyTicks), Adapter.CandidatesValidated, Adapter.CatalogsBuilt,
        keyBuilds, Milliseconds(keyTicks), descriptorBindings);

    public void Reset(ulong simulationSeed)
    {
        long start = Stopwatch.GetTimestamp();
        if (mode != SearchInformationMode.PerfectInformationTest)
            SearchRootFactory.PopulateWorkspace(Position, root, rootKnowledge, simulationSeed, deferUnknownCommands: Deferred);
        else Position.CopyFrom(root);
        knowledge = CreateKnowledge();
        materializedCommands = rootKnowledge.DeepClone();
        handledPreRevealWindows.Clear();
        pendingPrivateLook = null;
        Adapter.ResetPrefix(); Invalidate(); resets++; resetTicks += Stopwatch.GetTimestamp() - start;
    }
    private SearchKnowledge[] CreateKnowledge(bool bindOpponent = true)
    {
        SearchKnowledge[] result = [SearchKnowledge.For(Position, 1, observePrivateCommandLooks: false),
            SearchKnowledge.For(Position, 2, observePrivateCommandLooks: false)];
        result[RootObserver - 1] = rootKnowledge.DeepClone();
        if (!bindOpponent || opposingKnowledgePattern is not { } pattern) return result;
        var other = result[pattern.Observer - 1];
        for (int ship = 0; ship < pattern.Commands.Length; ship++)
        {
            var commands = Position.Context.State.Commands[ship].Stack;
            if (pattern.Commands[ship].Length != commands.Count)
                throw new SearchCoverageException("Opponent command-knowledge slots do not match this search root.");
            for (int slot = 0; slot < commands.Count; slot++)
                if (pattern.Commands[ship][slot]) other.RememberCommand(ship, slot, commands[slot]);
        }
        foreach (var slot in pattern.DamageSlots)
        {
            var cards = Position.Context.State.Ships[slot.Ship].DamageCards;
            if ((uint)slot.Index >= (uint)cards.Count || cards[slot.Index].CriticalCard is not { } card)
                throw new SearchCoverageException("Opponent damage-card knowledge has no corresponding sampled card.");
            other.RememberDamageCard(slot.Ship, slot.Index, card);
        }
        foreach (var prefix in pattern.DeckPrefixes)
        {
            var deck = Position.Context.State.Decks[prefix.Deck];
            if (prefix.Count > deck.Remaining)
                throw new SearchCoverageException("Opponent private deck peek exceeds the sampled deck.");
            other.RememberDeckPrefix(prefix.Deck, deck.PeekTop(prefix.Count));
        }
        return result;
    }
    public SearchObservation Encode()
    {
        long start = Stopwatch.GetTimestamp(); var decision = Current;
        if (decision.Stage.StartsWith("hidden_command/uct_", StringComparison.Ordinal))
        {
            // Hypothesis choices are search mechanics, never production encoder
            // inputs, neural requests, replay policy targets or real game actions.
            var key = Inspect().Key;
            encodings++; encodeTicks += Stopwatch.GetTimestamp() - start;
            return new("hidden_command", decision.Actor, key.Rules,
                SearchHash.Add(key.Knowledge, key.Rules), [], [],
                decision.Choices.Select(choice => choice.Id).ToArray(), UseUct: true);
        }
        if (encoder.RequiresNativeCandidates)
            decision = decision with { Candidates = Adapter.DescribeChoices() };
        var result = encoder.Encode(Position, knowledge[decision.Actor - 1], decision, Adapter.Prefix,
            Deferred ? materializedCommands : null);
        encodings++; encodeTicks += Stopwatch.GetTimestamp() - start; return result;
    }
    public bool Select(string semanticId)
    {
        var decision = Current;
        for (int index = 0; index < decision.Choices.Count; index++)
            if (decision.Choices[index].Id == semanticId) return Select(index);
        throw new ArgumentException("Choice is not legal at the current stage.", nameof(semanticId));
    }
    public bool Select(int index)
    {
        long start = Stopwatch.GetTimestamp(); var decision = Current;
        if ((uint)index >= (uint)decision.Choices.Count) throw new ArgumentOutOfRangeException(nameof(index));
        if (pendingPrivateLook is { } look)
        {
            ApplyPrivateLookHypothesis(look, RevealCommands(look.Ship)[index]);
            applyTicks += Stopwatch.GetTimestamp() - start;
            return false;
        }
        if (hypotheticalReveal is { } reveal)
        {
            before.CopyFrom(Position);
            ApplyReveal(reveal, hypotheticalCommands![index]); applyTicks += Stopwatch.GetTimestamp() - start; return true;
        }
        if (Adapter.Commits(index)) before.CopyFrom(Position);
        bool committed = Adapter.Select(index);
        if (committed)
        {
            var action = Adapter.LastCommittedAction ?? throw new InvalidOperationException("Adapter did not expose its committed action.");
            ObserveMaterializedTransition(action);
            Invalidate(); GetSettled();
            foreach (var observer in knowledge) observer.ObserveTransition(before, action, Position,
                observePrivateCommandLook: !Deferred);
            if (Deferred && action.AbilityId == CommandRevealEffects.IsardAction)
                BeginPrivateLook(Position.Context.State.Owners[action.Source], action.Target,
                    Enumerable.Range(0, Position.Context.State.Commands[action.Target].Count).ToArray());
        }
        else current = null;
        applyTicks += Stopwatch.GetTimestamp() - start; return committed;
    }
    public void SampleChance(ref DeterministicRandom random)
    {
        Prepare();
        if (mode == SearchInformationMode.DeferredChanceReveal && hypotheticalReveal is { } reveal)
        {
            before.CopyFrom(Position); var commands = RevealCommands(reveal.Source);
            ApplyReveal(reveal, commands[random.NextInt(commands.Length)]); return;
        }
        before.CopyFrom(Position); var action = Position.SampleChance(ref random);
        ObserveMaterializedTransition(action); Invalidate(); GetSettled();
        foreach (var observer in knowledge) observer.ObserveTransition(before, action, Position,
            observePrivateCommandLook: !Deferred);
        Adapter.ResetPrefix();
    }

    public SearchTraversalState Inspect()
    {
        Prepare(); var native = GetSettled();
        if (keyGeneration != Position.PositionGeneration || keyActor != native.Player ||
            keyFirstKnowledgeVersion != knowledge[0].Version || keySecondKnowledgeVersion != knowledge[1].Version)
        {
            long start = Stopwatch.GetTimestamp();
            baseKey = SearchRuleKey.Create(Position, knowledge, RootObserver, Array.Empty<string>(), native.Player);
            keyGeneration = Position.PositionGeneration;
            keyActor = native.Player;
            keyFirstKnowledgeVersion = knowledge[0].Version; keySecondKnowledgeVersion = knowledge[1].Version;
            keyBuilds++; keyTicks += Stopwatch.GetTimestamp() - start;
        }
        ulong prefixKey = baseKey.Knowledge;
        if (Deferred) prefixKey = materializedCommands.AddCommandMaterializationHash(prefixKey);
        foreach (string part in Adapter.Prefix) prefixKey = SearchHash.Add(prefixKey, part);
        if (pendingPrivateLook is { } look)
        {
            prefixKey = SearchHash.Add(prefixKey, "private-command-look");
            prefixKey = SearchHash.Add(prefixKey, (ulong)look.Observer);
            prefixKey = SearchHash.Add(prefixKey, (ulong)look.Ship);
            prefixKey = SearchHash.Add(prefixKey, (ulong)look.Cursor);
            prefixKey = SearchHash.Add(prefixKey, look.LegallyObserved ? 1UL : 0UL);
            foreach (int slot in look.Slots) prefixKey = SearchHash.Add(prefixKey, (ulong)slot);
        }
        var kind = pendingPrivateLook is not null ? DecisionKind.Player : hypotheticalReveal is null ? native.Kind :
            mode == SearchInformationMode.DeferredUctReveal ? DecisionKind.Player : DecisionKind.Chance;
        int actor = pendingPrivateLook is { } privateLook ? Position.Context.State.Owners[privateLook.Ship] : hypotheticalReveal is null ? native.Player :
            mode == SearchInformationMode.DeferredUctReveal ? RevealOwner(hypotheticalReveal) : 0;
        return new(new(baseKey.Rules, prefixKey), kind, actor, native.Outcome);
    }
    public object CaptureDecision() => new CachedDecision(Current,
        hypotheticalReveal is null && pendingPrivateLook is null ? Adapter.CaptureDescriptor() : null,
        hypotheticalReveal, hypotheticalCommands, pendingPrivateLook);
    public void BindDecision(object descriptor)
    {
        if (descriptor is not CachedDecision saved) throw new ArgumentException("Descriptor belongs to another environment type.", nameof(descriptor));
        current = saved.Decision; hypotheticalReveal = saved.Reveal; hypotheticalCommands = saved.Commands;
        pendingPrivateLook = saved.Look;
        if (saved.Native is not null) Adapter.BindDescriptor(saved.Native);
        descriptorBindings++;
    }
    private GameDecision GetSettled()
    {
        materializedCommands.SynchronizeCommandCounts(Position);
        if (settled is null || settledGeneration != Position.PositionGeneration || settledRevision != Position.Context.State.Revision)
        {
            settled = Position.GetDecision(); settledGeneration = Position.PositionGeneration;
            settledRevision = Position.Context.State.Revision;
        }
        materializedCommands.SynchronizeCommandCounts(Position);
        foreach (var observer in knowledge) observer.ObserveDecision(Position, settled,
            observePrivateCommandLooks: !Deferred);
        return settled;
    }
    private void Prepare()
    {
        if (settledGeneration != Position.PositionGeneration || settledRevision != Position.Context.State.Revision)
        { current = null; settled = null; keyGeneration = -1; prepared = false; encoder.Invalidate(); }
        if (prepared) return;
        GetSettled();
        if (Deferred)
        {
            if (pendingPrivateLook is not null) { prepared = true; return; }
            SettleDeferredProgramming();
            if (Position.Context.Flow.Continuations.LastOrDefault() is
                { Rule: "movement-choice.slicer", Source: >= 0, Target: >= 0 } slicer &&
                Position.Context.State.Commands[slicer.Target].Count > 0)
            {
                BeginPrivateLook(Position.Context.State.Owners[slicer.Source], slicer.Target, [0]);
                if (pendingPrivateLook is not null) { prepared = true; return; }
            }
            SearchRootFactory.ValidateDeferredDecision(Position, materializedCommands, GetSettled(),
                handledPreRevealWindows.Contains(Position.Context.Flow.ActiveShip));
            hypotheticalReveal = UnknownReveal();
            hypotheticalCommands = hypotheticalReveal is { } reveal ? RevealCommands(reveal.Source) : null;
        }
        prepared = true;
    }
    private void Invalidate()
    {
        current = null; settled = null; settledGeneration = settledRevision = keyGeneration = -1;
        prepared = false;
        hypotheticalReveal = null; hypotheticalCommands = null; encoder.Invalidate();
    }

    private GameAction? UnknownReveal()
    {
        if (Position.Context.Flow.Continuations.LastOrDefault() is { Rule: OfficialAssaultObjectives.SurpriseRevealRule } &&
            RootObserver != OfficialObjectives.SecondPlayer(Position.Context) &&
            !OfficialObjectives.State(Position.Context)!.Flags.ContainsKey("surprise.revealed." + (Position.Context.State.Round - 1)))
            return GetSettled().Actions.Single(option => option.Template.AbilityId == OfficialAssaultObjectives.SurpriseRevealAction).Template;
        if (Position.Context.Flow.Phase != GamePhase.RevealCommand) return null;
        int ship = Position.Context.Flow.ActiveShip;
        if (ship < 0 || Position.Context.State.Owners[ship] == RootObserver) return null;
        var commands = materializedCommands.CommandsFor(Position, ship);
        if (commands.Length == 0 || commands[0].HasValue) return null;
        return GetSettled().Actions.FirstOrDefault(option => option.Template.Kind == GameActionKind.RevealCommand)?.Template;
    }
    private ShipCommand[] RevealCommands(int ship)
        => Enum.GetValues<ShipCommand>().Where(command => ship < 0 || CommandModule.CommandAllowed(Position.Context, ship, command)).ToArray();
    private int RevealOwner(GameAction reveal) => reveal.AbilityId == OfficialAssaultObjectives.SurpriseRevealAction
        ? OfficialObjectives.SecondPlayer(Position.Context) : Position.Context.State.Owners[reveal.Source];
    private void BeginPrivateLook(int observer, int ship, int[] slots)
    {
        var resolved = materializedCommands.CommandsFor(Position, ship);
        var unresolved = slots.Where(slot => !resolved[slot].HasValue).ToArray();
        foreach (int slot in slots.Where(slot => resolved[slot].HasValue))
            knowledge[observer - 1].RememberCommand(ship, slot, resolved[slot]!.Value);
        if (unresolved.Length > 0)
        {
            pendingPrivateLook = new(observer, ship, unresolved);
            current = null; keyGeneration = -1; encoder.Invalidate();
        }
    }
    private bool SuspendPreRevealWindow(GameContinuation window)
    {
        if (!Deferred) return false;
        if (window.Rule == "effects.window" && window.Text is ("start-activation" or "before-command-reveal" or "before-reveal-command") &&
            Position.Context.IsShip(window.Source)) handledPreRevealWindows.Add(window.Source);
        if (!SearchRootFactory.HasPreRevealCommandDependency(Position.Context, window) ||
            Position.Context.State.Owners[window.Source] == RootObserver) return false;
        materializedCommands.SynchronizeCommandCounts(Position);
        var commands = materializedCommands.CommandsFor(Position, window.Source);
        if (commands.Length == 0 || commands[0].HasValue) return false;
        if (pendingPrivateLook is null)
            pendingPrivateLook = new(Position.Context.State.Owners[window.Source], window.Source, [0], LegallyObserved: false);
        return true;
    }
    private void ObserveMaterializedTransition(GameAction action)
        => materializedCommands.ObserveTransition(before, action, Position,
            observePrivateCommandLook: false, commandCountsAlreadySynchronized: true);
    private void ApplyPrivateLookHypothesis(PrivateCommandLook look, ShipCommand command)
    {
        int slot = look.Slots[look.Cursor];
        Position.Context.State.Commands[look.Ship].ReplaceAt(slot, command);
        // Materialization fixes this simulation's value. Only a legal private
        // look also updates observation memory; a pre-reveal owner hypothesis
        // remains unseen by the other player until a later public/private event.
        materializedCommands.RememberCommand(look.Ship, slot, command);
        if (look.LegallyObserved) knowledge[look.Observer - 1].RememberCommand(look.Ship, slot, command);
        pendingPrivateLook = look.Cursor + 1 < look.Slots.Length ? look with { Cursor = look.Cursor + 1 } : null;
        Adapter.ResetPrefix(); Invalidate();
    }
    private void ApplyReveal(GameAction reveal, ShipCommand command)
    {
        if (reveal.AbilityId == OfficialAssaultObjectives.SurpriseRevealAction)
        {
            OfficialObjectives.State(Position.Context)!.Flags["surprise.dial." + (Position.Context.State.Round - 1)] = (int)command;
            reveal = reveal with { Command = command };
        }
        else Position.Context.State.Commands[reveal.Source].ReplaceTop(command);
        Adapter.ResetPrefix(); Position.Apply(reveal); ObserveMaterializedTransition(reveal); Invalidate(); GetSettled();
        foreach (var observer in knowledge) observer.ObserveTransition(before, reveal, Position,
            observePrivateCommandLook: !Deferred);
        Adapter.ResetPrefix();
    }
    private void SettleDeferredProgramming()
    {
        for (int count = 0; count < 256; count++)
        {
            var decision = GetSettled();
            if (decision.Phase != GamePhase.CommandPlanning || decision.Kind != DecisionKind.Player || decision.Player == RootObserver) return;
            // Dummy values reserve slots only. Their actual hypotheses are chosen at
            // reveal and remain masked from every model input until that point.
            if (decision.Actions.Any(option => option.Template.Kind != GameActionKind.AssignCommand || option.Template.AbilityId.Length != 0))
                throw new SearchCoverageException("Deferred command programming needs a registered private-card interaction.");
            before.CopyFrom(Position); var placeholder = decision.Actions[0].Template;
            Position.Apply(placeholder); ObserveMaterializedTransition(placeholder); Invalidate(); GetSettled();
            foreach (var observer in knowledge) observer.ObserveTransition(before, placeholder, Position,
                observePrivateCommandLook: !Deferred);
            Adapter.ResetPrefix();
        }
        throw new SearchCoverageException("Deferred command programming did not settle within its explicit bound.");
    }

    /// <summary>Apply exactly the path selected by search to an actual game. This
    /// uses the caller's actual RNG only when their subsequent loop samples chance.</summary>
    public static GameAction ApplySelectedPath(GameSession actual, IReadOnlyList<string> path,
        IReadOnlyList<SearchKnowledge> knowledge, ActionCatalogProfile? profile = null)
    {
        var adapter = new DecisionAdapter(actual, profile); var previous = actual.DeepClone();
        for (int index = 0; index < path.Count; index++)
            if (adapter.Select(path[index]))
            {
                if (index != path.Count - 1) throw new ArgumentException("Search path continues after a native action committed.", nameof(path));
                var action = adapter.LastCommittedAction ?? throw new InvalidOperationException("Missing committed action.");
                actual.GetDecision(); foreach (var observer in knowledge) observer.ObserveTransition(previous, action, actual);
                return action;
            }
        throw new ArgumentException("Search path did not complete a native action.", nameof(path));
    }
    private static double Milliseconds(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
}
