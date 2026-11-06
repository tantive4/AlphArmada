#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Core;

public sealed partial class DamageDeckState
{
	private readonly List<CriticalDamageCard> drawPile = new List<CriticalDamageCard>(52);

	private readonly List<CriticalDamageCard> discardPile = new List<CriticalDamageCard>(52);

	public int Remaining => drawPile.Count;

	public int Discarded => discardPile.Count;

	public DamageDeckState(int seed)
	{
		foreach (CriticalDamageCard value in Enum.GetValues(typeof(CriticalDamageCard)))
		{
			drawPile.Add(value);
			drawPile.Add(value);
		}
		drawPile.Add(CriticalDamageCard.InjuredCrew);
		drawPile.Add(CriticalDamageCard.InjuredCrew);
		for (int i = 0; i < 6; i++)
		{
			drawPile.Add(CriticalDamageCard.StructuralDamage);
		}
		Shuffle(seed);
	}

	public CriticalDamageCard Draw()
	{
		if (drawPile.Count == 0)
		{
			return CriticalDamageCard.StructuralDamage;
		}
		CriticalDamageCard result = drawPile[0];
		drawPile.RemoveAt(0);
		return result;
	}

	public IReadOnlyList<CriticalDamageCard> PeekTop(int count)
	{
		List<CriticalDamageCard> list = new List<CriticalDamageCard>();
		int num = Math.Min(Math.Max(0, count), drawPile.Count);
		for (int i = 0; i < num; i++)
		{
			list.Add(drawPile[i]);
		}
		return list;
	}

	public CriticalDamageCard ChooseAndDrawFromTop(int chosenIndex, int lookCount)
	{
		int num = Math.Min(Math.Max(1, lookCount), drawPile.Count);
		if (num <= 0)
		{
			return Draw();
		}
		int num2 = Math.Max(0, Math.Min(chosenIndex, num - 1));
		CriticalDamageCard result = drawPile[num2];
		for (int num3 = num - 1; num3 >= 0; num3--)
		{
			CriticalDamageCard item = drawPile[num3];
			drawPile.RemoveAt(num3);
			if (num3 != num2)
			{
				discardPile.Add(item);
			}
		}
		return result;
	}

	public void Discard(CriticalDamageCard card)
	{
		discardPile.Add(card);
	}

	private void Shuffle(int seed)
	{
		Random random = new Random(seed);
		for (int num = drawPile.Count - 1; num > 0; num--)
		{
			int num2 = random.Next(num + 1);
			List<CriticalDamageCard> list = drawPile;
			int index = num;
			List<CriticalDamageCard> list2 = drawPile;
			int index2 = num2;
			CriticalDamageCard value = drawPile[num2];
			CriticalDamageCard value2 = drawPile[num];
			list[index] = value;
			list2[index2] = value2;
		}
	}
}
