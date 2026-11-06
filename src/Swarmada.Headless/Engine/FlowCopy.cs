using Armada.Core;

namespace Swarmada.Headless.Engine;

internal static class BranchCopy
{
    internal static T[] Array<T>(T[] source, T[] destination)
    {
        if (destination.Length != source.Length) return (T[])source.Clone();
        source.CopyTo(destination, 0); return destination;
    }
}

public sealed partial class ShipActivationState
{
    public void CopyFrom(ShipActivationState other)
    {
        ArgumentNullException.ThrowIfNull(other); if (ReferenceEquals(this, other)) return;
        AttacksResolved = other.AttacksResolved; UsedHullZones = other.UsedHullZones;
        ResolvedCommands = other.ResolvedCommands; ManeuverComplete = other.ManeuverComplete;
        RevealedCommand = other.RevealedCommand; NavigateDial = other.NavigateDial; NavigateToken = other.NavigateToken;
        ManeuversExecuted = other.ManeuversExecuted; NavigateManeuverConsumed = other.NavigateManeuverConsumed; NavigateTokenUse = other.NavigateTokenUse;
        EngineeringPoints = other.EngineeringPoints; CommandedSquadronsRemaining = other.CommandedSquadronsRemaining;
        AttackedShips = BranchCopy.Array(other.AttackedShips, AttackedShips);
        CopyCommandStateFrom(other);
    }
}

public sealed partial class GameFlowState
{
    /// <summary>Copies trusted flow into reusable branch storage; no source collection is shared.</summary>
    public void CopyFrom(GameFlowState other)
    {
        ArgumentNullException.ThrowIfNull(other); if (ReferenceEquals(this, other)) return;
        Phase = other.Phase; DecisionPlayer = other.DecisionPlayer; CurrentPlayer = other.CurrentPlayer;
        ActiveShip = other.ActiveShip; ActiveSquadron = other.ActiveSquadron; PlanningShip = other.PlanningShip;
        DeploymentEntity = other.DeploymentEntity; SquadronPhasePlayer = other.SquadronPhasePlayer;
        SquadronPhaseRemaining = other.SquadronPhaseRemaining; SquadronCanMove = other.SquadronCanMove;
        SquadronCanAttack = other.SquadronCanAttack; SquadronCommanded = other.SquadronCommanded;
        PendingDisplacements = BranchCopy.Array(other.PendingDisplacements, PendingDisplacements);
        if (ShipActivations.Length != other.ShipActivations.Length)
        {
            ShipActivations = new ShipActivationState[other.ShipActivations.Length];
            for (int i = 0; i < ShipActivations.Length; i++) ShipActivations[i] = new();
        }
        for (int i = 0; i < ShipActivations.Length; i++) ShipActivations[i].CopyFrom(other.ShipActivations[i]);
        while (Attacks.Count > other.Attacks.Count) Attacks.RemoveAt(Attacks.Count - 1);
        for (int i = 0; i < other.Attacks.Count; i++)
            if (i < Attacks.Count) Attacks[i].CopyFrom(other.Attacks[i]);
            else Attacks.Add(other.Attacks[i].DeepClone());
        while (Continuations.Count > other.Continuations.Count) Continuations.RemoveAt(Continuations.Count - 1);
        for (int i = 0; i < other.Continuations.Count; i++)
        {
            var source = other.Continuations[i];
            if (source.Indices is null)
            { if (i < Continuations.Count) Continuations[i] = source; else Continuations.Add(source); continue; }
            if (i < Continuations.Count)
            {
                var target = Continuations[i];
                if (target.Rule == source.Rule && target.Source == source.Source && target.Target == source.Target &&
                    target.Stage == source.Stage && target.Value == source.Value && target.Text == source.Text &&
                    target.Indices is { } indices && indices.Length == source.Indices.Length &&
                    (indices.Length == 0 || !ReferenceEquals(indices, source.Indices)))
                { source.Indices.CopyTo(indices, 0); continue; }
                Continuations[i] = source.DeepClone();
            }
            else Continuations.Add(source.DeepClone());
        }
        if (PendingOptions.Length != other.PendingOptions.Length) PendingOptions = new GameActionOption[other.PendingOptions.Length];
        for (int i = 0; i < other.PendingOptions.Length; i++)
        {
            var option = other.PendingOptions[i]; var action = option.Template.DeepClone();
            PendingOptions[i] = ReferenceEquals(action, option.Template) ? option : option with { Template = action };
        }
        SnapshotCollections.CopyInto(other.Flags, Flags); SourceRules.CopyFrom(other.SourceRules);
        DeathRecorded = BranchCopy.Array(other.DeathRecorded, DeathRecorded);
        DeathQueue.Clear(); DeathQueue.AddRange(other.DeathQueue);
        CopyCommandFlowStateFrom(other); ReuseMovementFlowStateFrom(other);
    }

    partial void ReuseMovementFlowStateFrom(GameFlowState other);
}
