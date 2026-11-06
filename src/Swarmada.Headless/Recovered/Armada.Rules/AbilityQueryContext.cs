#nullable disable
namespace Armada.Rules;

public readonly struct AbilityQueryContext
{
	public AbilityTiming Timing { get; }

	public RangeBand Range { get; }

	public bool MoveUsed { get; }

	public bool AttackUsed { get; }

	public bool IsCounterAttack { get; }

	public ShipCommand? RevealedCommand { get; }

	public AbilityQueryContext(AbilityTiming timing, RangeBand range = RangeBand.OutOfRange, bool moveUsed = false, bool attackUsed = false, ShipCommand? revealedCommand = null, bool isCounterAttack = false)
	{
		Timing = timing;
		Range = range;
		MoveUsed = moveUsed;
		AttackUsed = attackUsed;
		IsCounterAttack = isCounterAttack;
		RevealedCommand = revealedCommand;
	}
}
