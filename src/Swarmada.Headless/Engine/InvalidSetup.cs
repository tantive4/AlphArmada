using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

public enum InvalidSetupCode { FleetAmbushNoLegalDeployment }

/// <summary>A rejected setup attempt, never a game result or training reward.</summary>
public sealed record InvalidSetupResult(InvalidSetupCode Code, string ObjectiveId, int Player, int Entity,
    string Message, bool Retryable = true);

public sealed class InvalidSetupException(InvalidSetupResult result) : InvalidOperationException(result.Message)
{
    public InvalidSetupResult Result { get; } = result;
}

public sealed record SetupRetryResult(GameSession? Session, int Attempts, InvalidSetupResult? LastInvalidSetup)
{
    public bool Succeeded => Session is not null;
}

/// <summary>Retries only explicitly retryable setup failures. The caller supplies
/// match construction and all setup choices, including its own random placement policy.</summary>
public static class SetupRetries
{
    public static SetupRetryResult Run(Func<ulong, GameSession> create, Action<GameSession, ulong> completeSetup,
        ref DeterministicRandom seeds, int maximumAttempts = 64)
    {
        ArgumentNullException.ThrowIfNull(create); ArgumentNullException.ThrowIfNull(completeSetup);
        if (maximumAttempts < 1) throw new ArgumentOutOfRangeException(nameof(maximumAttempts));
        InvalidSetupResult? last = null;
        for (int attempt = 1; attempt <= maximumAttempts; attempt++)
        {
            ulong seed = seeds.NextUInt64(); var game = create(seed);
            try { completeSetup(game, seed); }
            catch (InvalidSetupException error) when (error.Result.Retryable) { last = error.Result; continue; }
            if (game.InvalidSetup is { } rejected)
            {
                if (!rejected.Retryable) throw new InvalidSetupException(rejected);
                last = rejected; continue;
            }
            if (game.Context.State.Registers[game.Context.Topology.Registers["setup.completed"]] == 0)
                throw new InvalidOperationException("The setup policy returned before setup was complete.");
            return new(game, attempt, last);
        }
        return new(null, maximumAttempts, last);
    }
}
