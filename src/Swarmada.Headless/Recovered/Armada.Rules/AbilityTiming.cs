#nullable disable
namespace Armada.Rules;

public enum AbilityTiming
{
	Setup,
	StartOfRound,
	StartOfShipPhase,
	StartOfSquadronPhase,
	StartOfActivation,
	CommandReveal,
	DuringActivation,
	InsteadOfAttack,
	BeforeAttack,
	ModifyAttackDice,
	ResolveAttackEffects,
	Defense,
	AfterDefense,
	DetermineCourse,
	AfterManeuver,
	AfterAttack,
	EndOfActivation,
	EndOfPhase,
	StatusPhase,
	AfterGatherAttackDice,
	AfterDeclareDefendingHullZone,
	StartResolveDamage
}
