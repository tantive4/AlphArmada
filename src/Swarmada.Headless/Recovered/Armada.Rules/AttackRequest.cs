#nullable disable
using System;

namespace Armada.Rules;

public readonly struct AttackRequest
{
	public int RedDice { get; }

	public int BlueDice { get; }

	public int BlackDice { get; }

	public float Distance { get; }

	public int Seed { get; }

	public int Rerolls { get; }

	public int BlackRerolls { get; }

	public int CriticalDamagePerIcon { get; }

	public bool BlackDiceAtMedium { get; }

	public bool BlueDiceAtLong { get; }

	public bool DiceAlreadyGathered { get; }

	public AttackRequest(int redDice, int blueDice, int blackDice, float distance, int seed, int rerolls = 0, int blackRerolls = 0, int criticalDamagePerIcon = 1, bool blackDiceAtMedium = false, bool blueDiceAtLong = false, bool diceAlreadyGathered = false)
	{
		RedDice = Math.Max(0, redDice);
		BlueDice = Math.Max(0, blueDice);
		BlackDice = Math.Max(0, blackDice);
		Distance = Math.Max(0f, distance);
		Seed = seed;
		Rerolls = Math.Max(0, rerolls);
		BlackRerolls = Math.Max(0, blackRerolls);
		CriticalDamagePerIcon = Math.Max(1, criticalDamagePerIcon);
		BlackDiceAtMedium = blackDiceAtMedium;
		BlueDiceAtLong = blueDiceAtLong;
		DiceAlreadyGathered = diceAlreadyGathered;
	}
}
