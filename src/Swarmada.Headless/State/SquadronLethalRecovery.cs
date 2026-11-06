namespace Armada.Core;

public sealed partial class SquadronRuntimeState
{
    /// <summary>Tel Trevura's explicit exception to ordinary recovery: recover after the
    /// defending attack even when the last hull point was lost. Activation state is preserved.</summary>
    public int RecoverHullIncludingLethal(int amount, int maximumHull)
    {
        int previous = CurrentHull;
        if (amount > 0 && maximumHull > 0) CurrentHull = System.Math.Min(maximumHull, CurrentHull + amount);
        if (CurrentHull > 0) DestructionDeferred = false;
        return CurrentHull - previous;
    }
}
