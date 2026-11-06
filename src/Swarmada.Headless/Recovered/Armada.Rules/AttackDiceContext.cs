#nullable disable
using System;
using Armada.Core;
using Swarmada.Headless.Engine;

namespace Armada.Rules;

public readonly struct AttackDiceContext
{
	public ShipView Attacker { get; }

	public ShipView DefenderShip { get; }

	public HullZone AttackZone { get; }

	public HullZone TargetZone { get; }

	public RangeBand Range { get; }

	public bool DefenderPreviouslyAttacked { get; }

	public bool DefenderActivatedThisRound { get; }

	public int RoundNumber { get; }

	public int AttacksResolvedThisActivation { get; }

	public AttackDiceContext(ShipView attacker, ShipView defenderShip, HullZone attackZone, HullZone targetZone, RangeBand range, bool defenderPreviouslyAttacked = false, bool defenderActivatedThisRound = false, int roundNumber = 0, int attacksResolvedThisActivation = 0)
	{
		Attacker = attacker;
		DefenderShip = defenderShip;
		AttackZone = attackZone;
		TargetZone = targetZone;
		Range = range;
		DefenderPreviouslyAttacked = defenderPreviouslyAttacked;
		DefenderActivatedThisRound = defenderActivatedThisRound;
		RoundNumber = roundNumber;
		AttacksResolvedThisActivation = Math.Max(0, attacksResolvedThisActivation);
	}
}
