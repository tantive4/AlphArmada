using System.Numerics;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Objective;
using Swarmada.Headless.Runtime;
using Swarmada.Headless.Setup;

namespace Swarmada.Headless.Engine;

public enum GamePhase
{
    Deployment, CommandPlanning, SelectShip, RevealCommand, ResolveCommand, SquadronCommand,
    Repair, SelectAttack, RollAttack, ModifyAttack, SpendAccuracy, SpendDefense, EvadeDice,
    RedirectDamage, ResolveCritical, DealDamage, ChooseDamageCard, SelectCourse, SquadronPhase,
    SquadronActivation, SquadronMove, SquadronDisplacement, Status, AbilityChoice, Terminal,
    StartRound, StartShipPhase
}
public enum GameActionKind
{
    Deploy, AssignCommand, ActivateShip, RevealCommand, ResolveCommand, BankCommand,
    SkipCommand, RepairHull, RecoverShield, TransferShield, ActivateSquadron, MoveSquadron,
    Attack, SkipAttack, ResolveDice, ModifyDice, SpendAccuracy, FinishAccuracy,
    SpendDefenseToken, FinishDefense, ChooseEvadeDice, Redirect, FinishRedirect,
    ChooseCritical, ResolveDamage, DrawDamageCard, SetCourse, ChangeSpeed, EndActivation,
    Pass, UseAbility, ChooseEntity, ChooseCommand, ChooseDie, ChooseToken, ChooseZone,
    PlaceToken, PlaceReservedUnit, DisplaceSquadron, AcceptObjective, DeclineObjective
}

/// <summary>Natural gameplay action, independent of neural policy shapes or movement discretization.</summary>
public sealed record GameAction(GameActionKind Kind, int Source = -1, int Target = -1,
    HullZone FromZone = HullZone.Front, HullZone ToZone = HullZone.Front,
    ShipCommand Command = ShipCommand.Navigate, CommandSpendMode SpendMode = CommandSpendMode.Dial,
    DefenseTokenType Token = DefenseTokenType.Brace, int Index = -1, int Amount = 0,
    Vector2? Position = null, float YawDegrees = 0, int Speed = -1, int[]? Course = null,
    int[]? DiceIndices = null, AttackDie[]? Dice = null, string AbilityId = "", bool Flipped = false)
{
    public GameAction DeepClone() => Course is null && DiceIndices is null && Dice is null ? this : this with
    { Course = Course?.ToArray(), DiceIndices = DiceIndices?.ToArray(), Dice = Dice?.ToArray() };
}
public sealed record GameActionOption(GameAction Template, string Label, bool AcceptsPosition = false,
    bool AcceptsYaw = false, bool AcceptsCourse = false, bool AcceptsFlip = false);
public sealed record GameDecision(DecisionKind Kind, int Player, GamePhase Phase,
    IReadOnlyList<GameActionOption> Actions, TerminalOutcome? Outcome = null, ManeuverCourseConstraints? CourseConstraints = null,
    InvalidSetupResult? InvalidSetup = null);

/// <summary>Continuation data, never delegates or enumerators. A rule owns its argument schema.</summary>
public sealed record GameContinuation(string Rule, int Source = -1, int Target = -1,
    int Stage = 0, long Value = 0, string Text = "", int[]? Indices = null)
{
    public GameContinuation DeepClone() => Indices is null ? this : this with { Indices = Indices.ToArray() };
}
public sealed partial class ShipActivationState
{
    public int AttacksResolved { get; set; }
    public int UsedHullZones { get; set; }
    public int ResolvedCommands { get; set; }
    public bool ManeuverComplete { get; set; }
    public ShipCommand? RevealedCommand { get; set; }
    public bool NavigateDial { get; set; }
    public bool NavigateToken { get; set; }
    public int ManeuversExecuted { get; set; }
    public bool NavigateManeuverConsumed { get; set; }
    public int NavigateTokenUse { get; set; }
    public int EngineeringPoints { get; set; }
    public int CommandedSquadronsRemaining { get; set; }
    public int[] AttackedShips { get; set; } = [];
    public ShipActivationState DeepClone()
    {
        var copy = new ShipActivationState
        {
        AttacksResolved = AttacksResolved, UsedHullZones = UsedHullZones, ResolvedCommands = ResolvedCommands,
        ManeuverComplete = ManeuverComplete, RevealedCommand = RevealedCommand, NavigateDial = NavigateDial,
        NavigateToken = NavigateToken, EngineeringPoints = EngineeringPoints,
        ManeuversExecuted = ManeuversExecuted, NavigateManeuverConsumed = NavigateManeuverConsumed, NavigateTokenUse = NavigateTokenUse,
        CommandedSquadronsRemaining = CommandedSquadronsRemaining, AttackedShips = AttackedShips.ToArray()
        };
        copy.CopyCommandStateFrom(this); return copy;
    }
    partial void CopyCommandStateFrom(ShipActivationState other);
}
public sealed partial class AttackState
{
    public int Attacker { get; set; } = -1;
    public int Defender { get; set; } = -1;
    public HullZone FromZone { get; set; }
    public HullZone ToZone { get; set; }
    public RangeBand Range { get; set; }
    public bool Obstructed { get; set; }
    public bool Counter { get; set; }
    public bool Salvo { get; set; }
    public bool Brace { get; set; }
    public bool Scatter { get; set; }
    public bool Contain { get; set; }
    public bool Redirect { get; set; }
    public bool Evade { get; set; }
    public bool CriticalResolved { get; set; }
    public int Damage { get; set; }
    public int RedDice { get; set; }
    public int BlueDice { get; set; }
    public int BlackDice { get; set; }
    public AttackDie[] Dice { get; set; } = [];
    public DefenseTokenInstance[] LockedTokens { get; set; } = [];
    public DefenseTokenType[] SpentTokenTypes { get; set; } = [];
    public KeyValuePair<HullZone, int>[] RedirectAllocations { get; set; } = [];
    public AttackState DeepClone()
    {
        var copy = (AttackState)MemberwiseClone();
        copy.Dice = Dice.ToArray(); copy.LockedTokens = LockedTokens.ToArray();
        copy.SpentTokenTypes = SpentTokenTypes.ToArray(); copy.RedirectAllocations = RedirectAllocations.ToArray();
        copy.CloneCombatFieldsFrom(this);
        return copy;
    }
    partial void CloneCombatFieldsFrom(AttackState other);
}
public sealed partial class GameFlowState
{
    public GamePhase Phase { get; set; } = GamePhase.Deployment;
    public int DecisionPlayer { get; set; }
    public int CurrentPlayer { get; set; } = 1;
    public int ActiveShip { get; set; } = -1;
    public int ActiveSquadron { get; set; } = -1;
    public int PlanningShip { get; set; }
    public int DeploymentEntity { get; set; }
    public int SquadronPhasePlayer { get; set; } = 1;
    public int SquadronPhaseRemaining { get; set; } = 2;
    public bool SquadronCanMove { get; set; }
    public bool SquadronCanAttack { get; set; }
    public bool SquadronCommanded { get; set; }
    public int[] PendingDisplacements { get; set; } = [];
    public ShipActivationState[] ShipActivations { get; set; } = [];
    public List<AttackState> Attacks { get; set; } = [];
    public List<GameContinuation> Continuations { get; set; } = [];
    public GameActionOption[] PendingOptions { get; set; } = [];
    public Dictionary<string, long> Flags { get; set; } = new(StringComparer.Ordinal);
    public SourceRoundState SourceRules { get; set; } = new();
    public bool[] DeathRecorded { get; set; } = [];
    public List<DeathTrigger> DeathQueue { get; set; } = [];
    public GameFlowState DeepClone()
    {
        var copy = (GameFlowState)MemberwiseClone();
        copy.PendingDisplacements = PendingDisplacements.ToArray();
        copy.ShipActivations = ShipActivations.Select(x => x.DeepClone()).ToArray();
        copy.Attacks = Attacks.Select(x => x.DeepClone()).ToList();
        copy.Continuations = Continuations.Select(x => x.DeepClone()).ToList();
        copy.PendingOptions = PendingOptions.Select(x => x with { Template = x.Template.DeepClone() }).ToArray();
        copy.Flags = new(Flags, StringComparer.Ordinal);
        copy.SourceRules = SourceRules.DeepClone();
        copy.DeathRecorded = DeathRecorded.ToArray(); copy.DeathQueue = DeathQueue.ToList();
        copy.CopyCommandFlowStateFrom(this); copy.CopyMovementFlowStateFrom(this); return copy;
    }
    partial void CopyCommandFlowStateFrom(GameFlowState other);
    partial void CopyMovementFlowStateFrom(GameFlowState other);
}
public sealed record GameSnapshot(BattleSnapshot Physical, GameFlowState Flow);

/// <summary>Owns every mutable gameplay object and a shared immutable match topology.</summary>
public sealed partial class GameContext
{
    // Membership changes rarely, while recovered rules query attached cards many
    // times per decision. Cards themselves remain live mutable state references.
    private UpgradeCardRuntimeState[]? upgradeCacheSource;
    private UpgradeCardRuntimeState[]? upgradeCacheCards;
    private long[]? upgradeCacheOwners;
    private IReadOnlyList<UpgradeCardRuntimeState>[]? upgradeCacheByEntity;
    private int[]? upgradeCacheCardIndices;
    private int[]? upgradeCacheOwnerRegisterIndices;
    public MatchTopology Topology { get; }
    public BattleState State { get; }
    public GameFlowState Flow { get; private set; }
    // Optional worker-local rule-query boundary. It is deliberately absent from
    // snapshots/CopyFrom/DeepClone; authoritative games leave it unset.
    internal Func<GameContinuation, bool>? SuspendTimingWindow { get; set; }
    public GameContext(PreparedMatch prepared)
    {
        Topology = prepared.Topology; State = prepared.CreateState();
        State.Frames.Clear();
        Flow = new() { CurrentPlayer = Topology.FirstPlayer,
            ShipActivations = Enumerable.Range(0, State.Ships.Length).Select(_ => new ShipActivationState()).ToArray(), DeathRecorded = new bool[Topology.EntityCount] };
        TokenObstacleAttachments.InitializePreparedTokens(this);
    }
    private GameContext(MatchTopology topology, BattleState state, GameFlowState flow)
    { Topology = topology; State = state; Flow = flow; }
    public GameSnapshot CaptureSnapshot() => new(State.CaptureSnapshot(), Flow.DeepClone());
    public void RestoreSnapshot(GameSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        if (Topology.ObjectiveId.Length > 0 && (snapshot.Physical.Objective is not OfficialObjectiveSnapshot official ||
            official.Id != Topology.ObjectiveId || official.EntityTokens.Length != Topology.EntityCount))
            throw new ArgumentException("A snapshot must retain the match's one supplied objective and entity layout.", nameof(snapshot));
        var flow = snapshot.Flow.DeepClone(); State.RestoreSnapshot(snapshot.Physical); Flow = flow;
    }
    public GameContext DeepClone() => new(Topology, State.DeepClone(), Flow.DeepClone());
    /// <summary>Reset a reusable search position from trusted state with the same match schema.</summary>
    public void CopyFrom(GameContext source)
    {
        ArgumentNullException.ThrowIfNull(source); if (ReferenceEquals(this, source)) return;
        if (State.SchemaId != source.State.SchemaId) throw new ArgumentException("Search positions belong to different match schemas.", nameof(source));
        State.CopyFrom(source.State); Flow.CopyFrom(source.Flow);
    }
    public bool IsShip(int entity) => entity >= 0 && entity < State.Ships.Length;
    public int SquadronIndex(int entity) => entity - State.Ships.Length;
    public bool IsDestroyed(int entity) => Flow.Flags.ContainsKey(Topology.RemovedFromPlayAreaKeys[entity])
        || (IsShip(entity) ? State.Ships[entity].Destroyed && !Flow.Flags.ContainsKey("rieekan.ship." + entity) : State.Squadrons[SquadronIndex(entity)].Destroyed);
    public bool IsReserved(int entity) => State.Registers[Topology.Registers.EntityReserved(entity)] != 0;
    public OrientedRectangle ShipBase(int entity) => new(State.Poses[entity], Topology.Ships[entity].FootprintMillimetres);
    public Circle SquadronBase(int entity) => new(State.Poses[entity].Position,
        Topology.Squadrons[SquadronIndex(entity)].FootprintMillimetres.X / 2);
    public IReadOnlyList<UpgradeCardRuntimeState> UpgradesFor(int entity)
    {
        // Preserve the original owner-register predicate for unusual/debug IDs,
        // including owner zero queried as entity -1 and unchecked integer wrap.
        if ((uint)entity >= (uint)Topology.EntityCount)
        {
            var result = new List<UpgradeCardRuntimeState>(); int owner = unchecked(entity + 1);
            for (int i = 0; i < Topology.Upgrades.Count; i++)
            {
                int index = Topology.Upgrades[i].UpgradeIndex;
                if (State.Registers[Topology.Registers.UpgradeOwner(index)] == owner) result.Add(State.Upgrades[index]);
            }
            return result.ToArray();
        }
        if (Topology.Upgrades.Count == 0) return Array.Empty<UpgradeCardRuntimeState>();
        var currentCards = State.Upgrades;
        bool same = upgradeCacheByEntity is not null && ReferenceEquals(upgradeCacheSource, currentCards);
        if (same)
        {
            // The topology never changes. Validate against precomputed numeric
            // slots rather than traversing descriptor/register-table properties
            // for every card on every recovered-rule query.
            var owners = upgradeCacheOwners!; var cards = upgradeCacheCards!;
            var cardIndices = upgradeCacheCardIndices!;
            var ownerRegisters = upgradeCacheOwnerRegisterIndices!;
            var registers = State.Registers;
            for (int i = 0; i < owners.Length; i++)
            {
                if (owners[i] != registers[ownerRegisters[i]] || !ReferenceEquals(cards[i], currentCards[cardIndices[i]]))
                { same = false; break; }
            }
        }
        if (!same) RebuildUpgradeMembership();
        return upgradeCacheByEntity![entity];
    }
    private void RebuildUpgradeMembership()
    {
        if (upgradeCacheCardIndices is null)
        {
            var indices = new int[Topology.Upgrades.Count]; var registers = new int[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            { indices[i] = Topology.Upgrades[i].UpgradeIndex; registers[i] = Topology.Registers.UpgradeOwner(indices[i]); }
            upgradeCacheCardIndices = indices; upgradeCacheOwnerRegisterIndices = registers;
        }
        var cardIndices = upgradeCacheCardIndices;
        var ownerRegisters = upgradeCacheOwnerRegisterIndices!;
        int count = cardIndices.Length;
        var cards = new UpgradeCardRuntimeState[count]; var owners = new long[count];
        var counts = new int[Topology.EntityCount];
        var currentCards = State.Upgrades; var currentRegisters = State.Registers;
        for (int i = 0; i < count; i++)
        {
            cards[i] = currentCards[cardIndices[i]]; owners[i] = currentRegisters[ownerRegisters[i]];
            if (owners[i] >= 1 && owners[i] <= counts.Length) counts[(int)owners[i] - 1]++;
        }
        var grouped = new UpgradeCardRuntimeState[counts.Length][];
        for (int entity = 0; entity < grouped.Length; entity++)
        { grouped[entity] = counts[entity] == 0 ? Array.Empty<UpgradeCardRuntimeState>() : new UpgradeCardRuntimeState[counts[entity]]; counts[entity] = 0; }
        for (int i = 0; i < count; i++)
            if (owners[i] >= 1 && owners[i] <= counts.Length)
            { int entity = (int)owners[i] - 1; grouped[entity][counts[entity]++] = cards[i]; }
        var lists = new IReadOnlyList<UpgradeCardRuntimeState>[grouped.Length];
        for (int entity = 0; entity < lists.Length; entity++)
            lists[entity] = grouped[entity].Length == 0 ? Array.Empty<UpgradeCardRuntimeState>() : Array.AsReadOnly(grouped[entity]);
        upgradeCacheSource = State.Upgrades; upgradeCacheCards = cards;
        upgradeCacheOwners = owners; upgradeCacheByEntity = lists;
    }
    public bool HasUpgrade(int entity, string id)
    {
        for (var i = 0; i < Topology.Upgrades.Count; i++)
        {
            var index = Topology.Upgrades[i].UpgradeIndex;
            if (State.Registers[Topology.Registers.UpgradeOwner(index)] != entity + 1) continue;
            var card = State.Upgrades[index];
            if (card.Active && card.Definition.id == id) return true;
        }
        return false;
    }
}

public interface IGameplayModule
{
    bool Handles(GamePhase phase);
    GameDecision GetDecision(GameContext game);
    void Apply(GameContext game, GameAction action);
}
/// <summary>Internal dispatch after the session has matched the current legal offer.
/// Direct module callers retain their ordinary validation path.</summary>
internal interface IVerifiedDecisionModule
{
    void ApplyVerified(GameContext game, GameAction action);
}
public interface IChanceGameplayModule : IGameplayModule
{
    GameAction SampleChance(GameContext game, ref DeterministicRandom random);
    void AdvanceChanceRandom(GameContext game, ref DeterministicRandom random) => _ = SampleChance(game, ref random);
}
