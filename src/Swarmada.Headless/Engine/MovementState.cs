using Armada.Core;
using System.Numerics;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

public sealed record MovementDamageEffect(int Entity, int Amount, bool Faceup = false,
    bool IgnoreShields = true, HullZone? Zone = null, string Rule = "movement.damage");
public sealed class MovementExecution
{
    public int Ship { get; set; }
    public Pose Start { get; set; }
    public Pose End { get; set; }
    public int[] PlottedCourse { get; set; } = [];
    public int[] ExecutedCourse { get; set; } = [];
    public GuideSide GuideSide { get; set; }
    public int DialedSpeed { get; set; }
    public int CollidedShip { get; set; } = -1;
    public bool PlayAreaHold { get; set; }
    public bool EndsOutsidePlayArea { get; set; }
    public bool StandstillYaw { get; set; }
    public float YawDelta { get; set; }
    public int Stage { get; set; }
    public int ObjectiveToken { get; set; } = -1;
    public float ObjectiveChoiceT { get; set; } = -1;
    public float ObjectiveStopT { get; set; } = -1;
    public bool ObjectiveChoiceResolved { get; set; }
    public bool ObjectiveAccepted { get; set; }
    public bool ChartOfficerChoiceResolved { get; set; }
    public bool IgnoreObstacleEffects { get; set; }
    public bool IgnoredByJaina { get; set; }
    public bool ObstacleRewardsResolved { get; set; }
    public bool AfterManeuverDamageResolved { get; set; }
    public int[] OverlappedObstacles { get; set; } = [];
    public List<MovementDamageEffect> DamageEffects { get; set; } = [];
    public int DamageEffectIndex { get; set; }
    public int[] MineIndices { get; set; } = [];
    public int MineIndex { get; set; }
    public bool MinesGathered { get; set; }
    public void CopyFrom(MovementExecution other)
    {
        if (ReferenceEquals(this, other)) return;
        Ship = other.Ship; Start = other.Start; End = other.End; GuideSide = other.GuideSide; DialedSpeed = other.DialedSpeed;
        CollidedShip = other.CollidedShip; PlayAreaHold = other.PlayAreaHold; EndsOutsidePlayArea = other.EndsOutsidePlayArea; StandstillYaw = other.StandstillYaw; YawDelta = other.YawDelta; Stage = other.Stage;
        ObjectiveToken = other.ObjectiveToken; ObjectiveChoiceT = other.ObjectiveChoiceT; ObjectiveStopT = other.ObjectiveStopT;
        ObjectiveChoiceResolved = other.ObjectiveChoiceResolved; ObjectiveAccepted = other.ObjectiveAccepted; ChartOfficerChoiceResolved = other.ChartOfficerChoiceResolved;
        IgnoreObstacleEffects = other.IgnoreObstacleEffects; IgnoredByJaina = other.IgnoredByJaina; ObstacleRewardsResolved = other.ObstacleRewardsResolved; AfterManeuverDamageResolved = other.AfterManeuverDamageResolved;
        DamageEffectIndex = other.DamageEffectIndex; MineIndex = other.MineIndex; MinesGathered = other.MinesGathered;
        PlottedCourse = CopyArray(PlottedCourse, other.PlottedCourse); ExecutedCourse = CopyArray(ExecutedCourse, other.ExecutedCourse);
        OverlappedObstacles = CopyArray(OverlappedObstacles, other.OverlappedObstacles); MineIndices = CopyArray(MineIndices, other.MineIndices);
        DamageEffects.Clear(); DamageEffects.AddRange(other.DamageEffects);
    }
    private static int[] CopyArray(int[] destination, int[] source)
    { if (destination.Length != source.Length) destination = new int[source.Length]; source.CopyTo(destination, 0); return destination; }
    public MovementExecution DeepClone()
    {
        var copy = (MovementExecution)MemberwiseClone();
        copy.PlottedCourse = PlottedCourse.ToArray(); copy.ExecutedCourse = ExecutedCourse.ToArray();
        copy.OverlappedObstacles = OverlappedObstacles.ToArray();
        copy.MineIndices = MineIndices.ToArray();
        copy.DamageEffects = DamageEffects.ToList(); return copy;
    }
}

public sealed class SquadronActivationProgress
{
    public int Entity { get; set; }
    public bool Moved { get; set; }
    public bool Attacked { get; set; }
    public bool Commanded { get; set; }
    public bool CanMoveAndAttack { get; set; }
    public bool WolffeGrantAdded { get; set; }
    public bool ActivationCommitted { get; set; }
    public bool Standalone { get; set; }
    public bool PreserveActivation { get; set; }
    public float MaximumMoveDistance { get; set; } = -1;
    public int Stage { get; set; }
    public Vector2? PendingMove { get; set; }
    public SquadronRuntimeSnapshot? BeforeSelection { get; set; }
    public SquadronActivationProgress DeepClone() => new()
    {
        Entity = Entity, Moved = Moved, Attacked = Attacked, Commanded = Commanded,
        CanMoveAndAttack = CanMoveAndAttack, WolffeGrantAdded = WolffeGrantAdded, ActivationCommitted = ActivationCommitted,
        Standalone = Standalone, PreserveActivation = PreserveActivation, MaximumMoveDistance = MaximumMoveDistance, Stage = Stage, PendingMove = PendingMove,
        BeforeSelection = CopySelectionSnapshot(BeforeSelection, null)
    };
    public void CopyFrom(SquadronActivationProgress other)
    {
        if (ReferenceEquals(this, other)) return;
        Entity = other.Entity; Moved = other.Moved; Attacked = other.Attacked; Commanded = other.Commanded;
        CanMoveAndAttack = other.CanMoveAndAttack; WolffeGrantAdded = other.WolffeGrantAdded; ActivationCommitted = other.ActivationCommitted;
        Standalone = other.Standalone; PreserveActivation = other.PreserveActivation; MaximumMoveDistance = other.MaximumMoveDistance; Stage = other.Stage; PendingMove = other.PendingMove;
        BeforeSelection = CopySelectionSnapshot(other.BeforeSelection, BeforeSelection);
    }
    private static SquadronRuntimeSnapshot? CopySelectionSnapshot(SquadronRuntimeSnapshot? source, SquadronRuntimeSnapshot? target)
    {
        if (source is null) return null;
        bool reuse = target is not null && !ReferenceEquals(source, target) &&
            source.CurrentHull == target.CurrentHull && source.Activated == target.Activated && source.Commanded == target.Commanded &&
            source.CommandingShipRuntimeId == target.CommandingShipRuntimeId && source.TemporaryGrit == target.TemporaryGrit &&
            source.TemporarySnipeValue == target.TemporarySnipeValue && source.TemporaryAssault == target.TemporaryAssault &&
            source.TemporarySwarm == target.TemporarySwarm && source.TemporarySpeedOverride == target.TemporarySpeedOverride &&
            source.TemporaryCannotAttack == target.TemporaryCannotAttack && source.TemporaryAttackOnly == target.TemporaryAttackOnly &&
            source.MovementPreventedForShipActivation == target.MovementPreventedForShipActivation && source.TemporaryYavarisBonusDie == target.TemporaryYavarisBonusDie &&
            source.TemporaryCloak == target.TemporaryCloak && source.TemporaryRogue == target.TemporaryRogue &&
            source.TemporaryIgnoresEngagement == target.TemporaryIgnoresEngagement && source.TemporaryCommanded == target.TemporaryCommanded &&
            source.DestructionDeferred == target.DestructionDeferred && source.DefenseTokenCounts.Length == target.DefenseTokenCounts.Length &&
            source.SpentDefenseTokens.Length == target.SpentDefenseTokens.Length && source.DiscardedDefenseTokens.Length == target.DiscardedDefenseTokens.Length &&
            source.DefenseTokenSlots.Length == target.DefenseTokenSlots.Length;
        if (reuse)
            for (int i = 0; i < source.DefenseTokenSlots.Length; i++)
                if (source.DefenseTokenSlots[i].Token != target!.DefenseTokenSlots[i].Token || source.DefenseTokenSlots[i].Slots.Length != target.DefenseTokenSlots[i].Slots.Length)
                { reuse = false; break; }
        if (reuse)
        {
            source.DefenseTokenCounts.CopyTo(target!.DefenseTokenCounts, 0); source.SpentDefenseTokens.CopyTo(target.SpentDefenseTokens, 0);
            source.DiscardedDefenseTokens.CopyTo(target.DiscardedDefenseTokens, 0);
            for (int i = 0; i < source.DefenseTokenSlots.Length; i++) source.DefenseTokenSlots[i].Slots.CopyTo(target.DefenseTokenSlots[i].Slots, 0);
            return target;
        }
        var slots = new DefenseTokenTypeSnapshot[source.DefenseTokenSlots.Length];
        for (int i = 0; i < slots.Length; i++) slots[i] = source.DefenseTokenSlots[i] with { Slots = (DefenseTokenSlotSnapshot[])source.DefenseTokenSlots[i].Slots.Clone() };
        return source with { DefenseTokenCounts = (KeyValuePair<DefenseTokenType, int>[])source.DefenseTokenCounts.Clone(),
            SpentDefenseTokens = (KeyValuePair<DefenseTokenType, int>[])source.SpentDefenseTokens.Clone(),
            DiscardedDefenseTokens = (KeyValuePair<DefenseTokenType, int>[])source.DiscardedDefenseTokens.Clone(), DefenseTokenSlots = slots };
    }
}

/// <summary>An activation granted by a squadron card never consumes a ship command or squadron turn.</summary>
public sealed class StandaloneSquadronActivationState
{
    public int ActiveShip { get; set; }
    public int ActiveSquadron { get; set; }
    public bool CanMove { get; set; }
    public bool CanAttack { get; set; }
    public bool Commanded { get; set; }
    public GamePhase ReturnPhase { get; set; }
    public SquadronActivationProgress? Progress { get; set; }
    public StandaloneSquadronActivationState DeepClone() => new()
    {
        ActiveShip = ActiveShip, ActiveSquadron = ActiveSquadron, CanMove = CanMove, CanAttack = CanAttack, Commanded = Commanded,
        ReturnPhase = ReturnPhase, Progress = Progress?.DeepClone()
    };
    public void CopyFrom(StandaloneSquadronActivationState other)
    {
        ActiveShip = other.ActiveShip; ActiveSquadron = other.ActiveSquadron; CanMove = other.CanMove; CanAttack = other.CanAttack;
        Commanded = other.Commanded; ReturnPhase = other.ReturnPhase;
        if (other.Progress is null) Progress = null;
        else if (Progress is null) Progress = other.Progress.DeepClone(); else Progress.CopyFrom(other.Progress);
    }
}

public readonly record struct ObjectiveTokenObservation(int Kind, int Index, Vector2 Position, bool Active, bool OnObstacle, float Radius);
public readonly record struct MoraloTriggerState(int Squadron, Vector2 TokenPosition);

public sealed class ExtraManeuverState
{
    public int Ship { get; set; }
    public int OriginalSpeed { get; set; }
    public MovementExecution? ParentMovement { get; set; }
    public ExtraManeuverState DeepClone() => new() { Ship = Ship, OriginalSpeed = OriginalSpeed, ParentMovement = ParentMovement?.DeepClone() };
    public void CopyFrom(ExtraManeuverState other)
    {
        if (ReferenceEquals(this, other)) return;
        Ship = other.Ship; OriginalSpeed = other.OriginalSpeed;
        if (other.ParentMovement is null) ParentMovement = null;
        else if (ParentMovement is null) ParentMovement = other.ParentMovement.DeepClone();
        else ParentMovement.CopyFrom(other.ParentMovement);
    }
}

public sealed class ForcedManeuverState
{
    public int Ship { get; set; }
    public int OriginalSpeed { get; set; }
    public int ActiveShip { get; set; }
    public int ActiveSquadron { get; set; }
    public GamePhase ReturnPhase { get; set; }
    public MovementExecution? ParentMovement { get; set; }
    public ForcedManeuverState DeepClone() => new() { Ship = Ship, OriginalSpeed = OriginalSpeed, ActiveShip = ActiveShip,
        ActiveSquadron = ActiveSquadron, ReturnPhase = ReturnPhase, ParentMovement = ParentMovement?.DeepClone() };
    public void CopyFrom(ForcedManeuverState other)
    {
        Ship = other.Ship; OriginalSpeed = other.OriginalSpeed; ActiveShip = other.ActiveShip; ActiveSquadron = other.ActiveSquadron; ReturnPhase = other.ReturnPhase;
        if (other.ParentMovement is null) ParentMovement = null;
        else if (ParentMovement is null) ParentMovement = other.ParentMovement.DeepClone(); else ParentMovement.CopyFrom(other.ParentMovement);
    }
}

/// <summary>Parent activation flags saved across source-granted immediate attacks.</summary>
public sealed class GrantedSquadronAttackState
{
    public int ParentSquadron { get; set; } = -1;
    public bool ParentCanMove { get; set; }
    public bool ParentCanAttack { get; set; }
    public bool ParentCommanded { get; set; }
    public bool ConsumeParentAttack { get; set; }
    public bool TriggerAfterAttack { get; set; }
    public int Stage { get; set; }
    public GrantedSquadronAttackState DeepClone() => (GrantedSquadronAttackState)MemberwiseClone();
    public void CopyFrom(GrantedSquadronAttackState other)
    {
        ParentSquadron = other.ParentSquadron; ParentCanMove = other.ParentCanMove; ParentCanAttack = other.ParentCanAttack;
        ParentCommanded = other.ParentCommanded; ConsumeParentAttack = other.ConsumeParentAttack; TriggerAfterAttack = other.TriggerAfterAttack; Stage = other.Stage;
    }
}

public sealed partial class GameFlowState
{
    public SquadronBatchKind SquadronBatchKind { get; set; }
    public int SquadronBatchSource { get; set; } = -1;
    public int[] SquadronBatchSelections { get; set; } = [];
    public MovementExecution? Movement { get; set; }
    public SquadronActivationProgress? SquadronProgress { get; set; }
    public int DisplacementShip { get; set; } = -1;
    public List<ExtraManeuverState> ExtraManeuvers { get; set; } = [];
    public List<ForcedManeuverState> ForcedManeuvers { get; set; } = [];
    public List<GrantedSquadronAttackState> GrantedSquadronAttacks { get; set; } = [];
    public List<StandaloneSquadronActivationState> StandaloneSquadronActivations { get; set; } = [];
    public List<ObjectiveTokenObservation> ObservedObjectiveTokens { get; set; } = [];
    public List<MoraloTriggerState> PendingMoraloTriggers { get; set; } = [];
    /// <summary>Spatial token index to the obstacles it was explicitly placed on.
    /// An obstacle moving underneath a free token never creates an attachment.</summary>
    public Dictionary<int, int[]> TokenHosts { get; set; } = [];
    partial void CopyMovementFlowStateFrom(GameFlowState other)
    {
        Movement = other.Movement?.DeepClone(); SquadronProgress = other.SquadronProgress?.DeepClone();
        DisplacementShip = other.DisplacementShip;
        SquadronBatchKind = other.SquadronBatchKind; SquadronBatchSource = other.SquadronBatchSource; SquadronBatchSelections = other.SquadronBatchSelections.ToArray();
        ExtraManeuvers = other.ExtraManeuvers.Select(x => x.DeepClone()).ToList();
        ForcedManeuvers = other.ForcedManeuvers.Select(x => x.DeepClone()).ToList();
        GrantedSquadronAttacks = other.GrantedSquadronAttacks.Select(x => x.DeepClone()).ToList();
        StandaloneSquadronActivations = other.StandaloneSquadronActivations.Select(x => x.DeepClone()).ToList();
        ObservedObjectiveTokens = other.ObservedObjectiveTokens.ToList(); PendingMoraloTriggers = other.PendingMoraloTriggers.ToList();
        TokenHosts = other.TokenHosts.ToDictionary(pair => pair.Key, pair => pair.Value.ToArray());
    }
    partial void ReuseMovementFlowStateFrom(GameFlowState other)
    {
        if (other.Movement is null) Movement = null;
        else if (Movement is null) Movement = other.Movement.DeepClone();
        else Movement.CopyFrom(other.Movement);
        if (other.SquadronProgress is null) SquadronProgress = null;
        else if (SquadronProgress is null) SquadronProgress = other.SquadronProgress.DeepClone();
        else SquadronProgress.CopyFrom(other.SquadronProgress);
        DisplacementShip = other.DisplacementShip;
        SquadronBatchKind = other.SquadronBatchKind; SquadronBatchSource = other.SquadronBatchSource;
        SquadronBatchSelections = BranchCopy.Array(other.SquadronBatchSelections, SquadronBatchSelections);
        for (int i = 0; i < other.ExtraManeuvers.Count; i++)
            if (i < ExtraManeuvers.Count) ExtraManeuvers[i].CopyFrom(other.ExtraManeuvers[i]); else ExtraManeuvers.Add(other.ExtraManeuvers[i].DeepClone());
        if (ExtraManeuvers.Count > other.ExtraManeuvers.Count) ExtraManeuvers.RemoveRange(other.ExtraManeuvers.Count, ExtraManeuvers.Count - other.ExtraManeuvers.Count);
        for (int i = 0; i < other.ForcedManeuvers.Count; i++)
            if (i < ForcedManeuvers.Count) ForcedManeuvers[i].CopyFrom(other.ForcedManeuvers[i]); else ForcedManeuvers.Add(other.ForcedManeuvers[i].DeepClone());
        if (ForcedManeuvers.Count > other.ForcedManeuvers.Count) ForcedManeuvers.RemoveRange(other.ForcedManeuvers.Count, ForcedManeuvers.Count - other.ForcedManeuvers.Count);
        for (int i = 0; i < other.GrantedSquadronAttacks.Count; i++)
            if (i < GrantedSquadronAttacks.Count) GrantedSquadronAttacks[i].CopyFrom(other.GrantedSquadronAttacks[i]); else GrantedSquadronAttacks.Add(other.GrantedSquadronAttacks[i].DeepClone());
        if (GrantedSquadronAttacks.Count > other.GrantedSquadronAttacks.Count) GrantedSquadronAttacks.RemoveRange(other.GrantedSquadronAttacks.Count, GrantedSquadronAttacks.Count - other.GrantedSquadronAttacks.Count);
        for (int i = 0; i < other.StandaloneSquadronActivations.Count; i++)
            if (i < StandaloneSquadronActivations.Count) StandaloneSquadronActivations[i].CopyFrom(other.StandaloneSquadronActivations[i]); else StandaloneSquadronActivations.Add(other.StandaloneSquadronActivations[i].DeepClone());
        if (StandaloneSquadronActivations.Count > other.StandaloneSquadronActivations.Count) StandaloneSquadronActivations.RemoveRange(other.StandaloneSquadronActivations.Count, StandaloneSquadronActivations.Count - other.StandaloneSquadronActivations.Count);
        ObservedObjectiveTokens.Clear(); ObservedObjectiveTokens.AddRange(other.ObservedObjectiveTokens);
        PendingMoraloTriggers.Clear(); PendingMoraloTriggers.AddRange(other.PendingMoraloTriggers);
        if (ReferenceEquals(TokenHosts, other.TokenHosts)) TokenHosts = [];
        if (TokenHosts.Count != 0)
        {
            List<int>? stale = null;
            foreach (int token in TokenHosts.Keys)
                if (!other.TokenHosts.ContainsKey(token)) (stale ??= []).Add(token);
            if (stale is not null) foreach (int token in stale) TokenHosts.Remove(token);
        }
        foreach (var pair in other.TokenHosts)
        {
            TokenHosts.TryGetValue(pair.Key, out var existing);
            TokenHosts[pair.Key] = existing is null || ReferenceEquals(existing, pair.Value)
                ? pair.Value.ToArray() : BranchCopy.Array(pair.Value, existing);
        }
    }
}
