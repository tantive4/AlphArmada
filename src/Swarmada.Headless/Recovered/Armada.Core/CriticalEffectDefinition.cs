#nullable disable
namespace Armada.Core;

public readonly struct CriticalEffectDefinition
{
	public CriticalDamageCard Card { get; }

	public string DisplayName { get; }

	public string RulesText { get; }

	public CriticalEffectTiming Timing { get; }

	public CriticalDecisionType DecisionType { get; }

	public CriticalDecisionOwner DecisionOwner { get; }

	public bool RemainsFaceup => Timing != CriticalEffectTiming.ImmediateAndFlip;

	public bool RequiresImmediatePrompt => Timing == CriticalEffectTiming.ImmediateAndFlip;

	public CriticalEffectDefinition(CriticalDamageCard card, string displayName, string rulesText, CriticalEffectTiming timing, CriticalDecisionType decisionType = CriticalDecisionType.None, CriticalDecisionOwner decisionOwner = CriticalDecisionOwner.AffectedPlayer)
	{
		Card = card;
		DisplayName = displayName;
		RulesText = rulesText;
		Timing = timing;
		DecisionType = decisionType;
		DecisionOwner = decisionOwner;
	}
}
