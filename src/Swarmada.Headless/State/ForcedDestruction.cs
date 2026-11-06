namespace Armada.Core;

public sealed partial class ShipRuntimeState
{
    /// <summary>Removes a unit by a destruction rule, without dealing damage or drawing cards.</summary>
    public void DestroyWithoutDamage() => CurrentHull = 0;
}

public sealed partial class SquadronRuntimeState
{
    /// <summary>Removes a unit by a destruction rule, regardless of any intervening hull recovery.</summary>
    public void DestroyWithoutDamage()
    {
        CurrentHull = 0;
        DestructionDeferred = false;
    }
}
