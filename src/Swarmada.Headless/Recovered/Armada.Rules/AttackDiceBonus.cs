#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Rules;

public readonly struct AttackDiceBonus
{
	public int Red { get; }

	public int Blue { get; }

	public int Black { get; }

	public IReadOnlyList<string> Sources { get; }

	public bool Any
	{
		get
		{
			if (Red <= 0 && Blue <= 0)
			{
				return Black > 0;
			}
			return true;
		}
	}

	public AttackDiceBonus(int red, int blue, int black, IReadOnlyList<string> sources)
	{
		Red = red;
		Blue = blue;
		Black = black;
		Sources = sources ?? Array.Empty<string>();
	}
}
