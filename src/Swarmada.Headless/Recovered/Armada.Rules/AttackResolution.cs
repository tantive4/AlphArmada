#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Rules;

public readonly struct AttackResolution
{
	public RangeBand Range { get; }

	public int DiceRolled { get; }

	public int Hits { get; }

	public int Criticals { get; }

	public int Accuracies { get; }

	public int Blanks { get; }

	public bool IsLegal { get; }

	public int CriticalDamagePerIcon { get; }

	public IReadOnlyList<AttackDie> Dice { get; }

	public int Damage => Hits + Criticals * CriticalDamagePerIcon;

	public bool IsMiss
	{
		get
		{
			if (IsLegal)
			{
				return Damage == 0;
			}
			return false;
		}
	}

	public bool CanResolveCriticalEffect => Criticals > 0;

	public bool IsMissAgainstSquadron
	{
		get
		{
			if (IsLegal)
			{
				return DamageAgainstSquadron() == 0;
			}
			return false;
		}
	}

	public AttackResolution(RangeBand range, int diceRolled, int hits, int criticals, int accuracies, int blanks, bool isLegal, int criticalDamagePerIcon = 1, IReadOnlyList<AttackDie> dice = null)
	{
		Dice = dice ?? Array.Empty<AttackDie>();
		Range = range;
		DiceRolled = diceRolled;
		Hits = Math.Max(0, hits);
		Criticals = Math.Max(0, criticals);
		Accuracies = Math.Max(0, accuracies);
		Blanks = Math.Max(0, blanks);
		IsLegal = isLegal;
		CriticalDamagePerIcon = Math.Max(1, criticalDamagePerIcon);
	}

	public int DamageAgainstSquadron()
	{
		return Hits;
	}

	public int DamageFromSquadronAgainstShip(bool bomber)
	{
		return Hits + (bomber ? Criticals : 0);
	}

	public AttackResolution WithRange(RangeBand range)
	{
		return new AttackResolution(range, DiceRolled, Hits, Criticals, Accuracies, Blanks, IsLegal, CriticalDamagePerIcon, Dice);
	}
}
