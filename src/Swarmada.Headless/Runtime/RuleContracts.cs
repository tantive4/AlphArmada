namespace Swarmada.Headless.Runtime;

public enum DecisionKind { Forced, Player, Chance, Terminal, InvalidSetup }

/// <summary>Stable numeric action. Each registered rule owns its payload schema.</summary>
public readonly record struct EngineAction(int Kind, int Source = -1, int Target = -1,
    int Option = 0, long Payload = 0);

/// <summary>Serializable continuation. No closures, coroutine enumerators, or view objects.</summary>
public readonly record struct RuleFrame(int RuleId, int Owner = 0, int Source = -1,
    int Target = -1, int Stage = 0, long Payload = 0);

public sealed record TerminalOutcome(int Winner, double PlayerOneScore, double PlayerTwoScore);
public sealed record DecisionView(DecisionKind Kind, int Player, int RuleId, long Revision,
    IReadOnlyList<EngineAction> Actions, TerminalOutcome? Outcome);

/// <summary>Handlers must be stateless; all mutable data belongs to BattleState and its frames.</summary>
public interface IRuleHandler
{
    int Id { get; }
    DecisionKind GetKind(BattleState state, RuleFrame frame);
    void GetLegalActions(BattleState state, RuleFrame frame, List<EngineAction> output);
    bool IsLegalChanceOutcome(BattleState state, RuleFrame frame, EngineAction action);
    EngineAction SampleChance(BattleState state, RuleFrame frame, ref DeterministicRandom random);
    void Apply(BattleState state, RuleFrame frame, EngineAction action);
}

public sealed class RuleRegistry
{
    private readonly Dictionary<int, IRuleHandler> handlers;
    public RuleRegistry(IEnumerable<IRuleHandler> rules)
    {
        ArgumentNullException.ThrowIfNull(rules);
        handlers = new();
        foreach (var rule in rules)
        {
            ArgumentNullException.ThrowIfNull(rule);
            if (rule.Id < 1 || !handlers.TryAdd(rule.Id, rule))
                throw new ArgumentException("Rule IDs must be positive and unique.", nameof(rules));
        }
    }
    public IRuleHandler GetRequired(int id) => handlers.TryGetValue(id, out var handler) ? handler :
        throw new UnsupportedRuleException(id);
    public IReadOnlyList<int> RegisteredIds => handlers.Keys.Order().ToArray();
}

public sealed class UnsupportedRuleException(int ruleId) : NotSupportedException(
    $"Rule {ruleId} has no headless handler. Unported gameplay cannot be skipped.")
{
    public int RuleId { get; } = ruleId;
}
