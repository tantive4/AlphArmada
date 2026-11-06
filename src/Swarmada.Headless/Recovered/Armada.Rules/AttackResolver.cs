#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Rules;

public static class AttackResolver
{
	private readonly struct DieRoll
	{
		public AttackDieColor Color { get; }

		public DieFace Face { get; }

		public DieRoll(AttackDieColor color, DieFace face)
		{
			Color = color;
			Face = face;
		}

		public AttackDie ToPublic()
		{
			return new AttackDie(Color, Face);
		}
	}

	private const float MillimetresPerUnit = 22.85f;

	public const float CloseRange = 5.426696f;

	public const float MediumRange = 8.227571f;

	public const float LongRange = 13.260394f;

	public const float SquadronAttackDistance = 3.369803f;

	public const float SquadronEngagementDistance = 3.369803f;

	private static readonly DieFace[] RedFaces = new DieFace[8]
	{
		DieFace.Blank,
		DieFace.Blank,
		DieFace.Hit,
		DieFace.Hit,
		DieFace.Critical,
		DieFace.Critical,
		DieFace.Accuracy,
		DieFace.DoubleHit
	};

	private static readonly DieFace[] BlueFaces = new DieFace[8]
	{
		DieFace.Hit,
		DieFace.Hit,
		DieFace.Hit,
		DieFace.Hit,
		DieFace.Critical,
		DieFace.Critical,
		DieFace.Accuracy,
		DieFace.Accuracy
	};

	private static readonly DieFace[] BlackFaces = new DieFace[8]
	{
		DieFace.Blank,
		DieFace.Blank,
		DieFace.Hit,
		DieFace.Hit,
		DieFace.Hit,
		DieFace.Hit,
		DieFace.HitCritical,
		DieFace.HitCritical
	};

	public static RangeBand ClassifyRange(float distance)
	{
		if (distance <= 5.426696f)
		{
			return RangeBand.Close;
		}
		if (distance <= 8.227571f)
		{
			return RangeBand.Medium;
		}
		if (distance <= 13.260394f)
		{
			return RangeBand.Long;
		}
		return RangeBand.OutOfRange;
	}

	public static string GetWeaponClass(AttackDieColor color)
	{
		return GetWeaponProfile(color).WeaponClass;
	}

	public static WeaponDieProfile GetWeaponProfile(AttackDieColor color)
	{
		return color switch
		{
			AttackDieColor.Red => new WeaponDieProfile(color, "LONG", RangeBand.Long, 2, 0.25f, 0.375f, 0.25f, 0.125f, 0.125f, 0f, 0.75f),
			AttackDieColor.Blue => new WeaponDieProfile(color, "MIDRANGE", RangeBand.Medium, 1, 0f, 0.5f, 0.25f, 0.25f, 0f, 0f, 0.75f),
			AttackDieColor.Black => new WeaponDieProfile(color, "CLOSE", RangeBand.Close, 2, 0.25f, 0.75f, 0.25f, 0f, 0f, 0.25f, 1f),
			_ => new WeaponDieProfile(AttackDieColor.None, "NONE", RangeBand.OutOfRange, 0, 1f, 0f, 0f, 0f, 0f, 0f, 0f),
		};
	}

	public static AttackResolution Resolve(AttackRequest request)
	{
		RangeBand rangeBand = ClassifyRange(request.Distance);
		if (rangeBand == RangeBand.OutOfRange || (request.DiceAlreadyGathered ? (request.RedDice + request.BlueDice + request.BlackDice <= 0) : (!HasUsableDiceAtRange(request.RedDice, request.BlueDice, request.BlackDice, rangeBand, request.BlackDiceAtMedium, request.BlueDiceAtLong))))
		{
			return new AttackResolution(rangeBand, 0, 0, 0, 0, 0, isLegal: false);
		}
		int redDice = request.RedDice;
		int num = (request.DiceAlreadyGathered ? request.BlueDice : ((rangeBand != RangeBand.Long || request.BlueDiceAtLong) ? request.BlueDice : 0));
		int num2 = (request.DiceAlreadyGathered ? request.BlackDice : ((rangeBand == RangeBand.Close || (rangeBand == RangeBand.Medium && request.BlackDiceAtMedium)) ? request.BlackDice : 0));
		Random random = new Random(request.Seed);
		List<DieRoll> list = new List<DieRoll>(redDice + num + num2);
		RollDice(random, AttackDieColor.Red, redDice, RedFaces, list);
		RollDice(random, AttackDieColor.Blue, num, BlueFaces, list);
		RollDice(random, AttackDieColor.Black, num2, BlackFaces, list);
		for (int i = 0; i < request.Rerolls; i++)
		{
			int num3 = list.FindIndex((DieRoll roll) => roll.Face == DieFace.Blank);
			if (num3 < 0)
			{
				break;
			}
			DieRoll dieRoll = list[num3];
			list[num3] = new DieRoll(dieRoll.Color, RollFace(random, GetFaces(dieRoll.Color)));
		}
		for (int num4 = 0; num4 < request.BlackRerolls; num4++)
		{
			int num5 = list.FindIndex((DieRoll roll) => roll.Color == AttackDieColor.Black && roll.Face == DieFace.Blank);
			if (num5 < 0)
			{
				break;
			}
			list[num5] = new DieRoll(AttackDieColor.Black, RollFace(random, BlackFaces));
		}
		int num6 = 0;
		int num7 = 0;
		int num8 = 0;
		int num9 = 0;
		using (List<DieRoll>.Enumerator enumerator = list.GetEnumerator())
		{
			while (enumerator.MoveNext())
			{
				switch (enumerator.Current.Face)
				{
				case DieFace.Hit:
					num6++;
					break;
				case DieFace.DoubleHit:
					num6 += 2;
					break;
				case DieFace.Critical:
					num7++;
					break;
				case DieFace.HitCritical:
					num6++;
					num7++;
					break;
				case DieFace.Accuracy:
					num8++;
					break;
				default:
					num9++;
					break;
				}
			}
		}
		AttackDie[] array = new AttackDie[list.Count];
		for (int num10 = 0; num10 < list.Count; num10++)
		{
			array[num10] = list[num10].ToPublic();
		}
		return new AttackResolution(rangeBand, list.Count, num6, num7, num8, num9, isLegal: true, request.CriticalDamagePerIcon, array);
	}

	public static int CountDiceAtRange(int redDice, int blueDice, int blackDice, RangeBand range, bool blackDiceAtMedium = false, bool blueDiceAtLong = false)
	{
		if (range == RangeBand.OutOfRange)
		{
			return 0;
		}
		int num = Math.Max(0, redDice);
		if (range != RangeBand.Long || blueDiceAtLong)
		{
			num += Math.Max(0, blueDice);
		}
		if (range == RangeBand.Close || (range == RangeBand.Medium && blackDiceAtMedium))
		{
			num += Math.Max(0, blackDice);
		}
		return num;
	}

	public static bool HasUsableDiceAtRange(int redDice, int blueDice, int blackDice, RangeBand range, bool blackDiceAtMedium = false, bool blueDiceAtLong = false)
	{
		if (range == RangeBand.OutOfRange)
		{
			return false;
		}
		if (redDice > 0)
		{
			return true;
		}
		if (blueDice > 0 && (range != RangeBand.Long || blueDiceAtLong))
		{
			return true;
		}
		if (blackDice > 0)
		{
			if (range != RangeBand.Close)
			{
				return range == RangeBand.Medium && blackDiceAtMedium;
			}
			return true;
		}
		return false;
	}

	public static AttackResolution RerollHighestSquadronDamageDie(AttackResolution attack, int seed)
	{
		return RerollHighestSquadronDamageDice(attack, 1, seed);
	}

	public static AttackResolution RerollHighestSquadronDamageDice(AttackResolution attack, int count, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		AttackDie[] dice = new AttackDie[attack.Dice.Count];
		List<int> list = new List<int>(attack.Dice.Count);
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			dice[i] = attack.Dice[i];
			list.Add(i);
		}
		list.Sort((int first, int second) => SquadronDamage(dice[second].Face).CompareTo(SquadronDamage(dice[first].Face)));
		Random random = new Random(seed);
		int num = Math.Min(Math.Max(0, count), list.Count);
		for (int num2 = 0; num2 < num; num2++)
		{
			int num3 = list[num2];
			dice[num3] = new AttackDie(dice[num3].Color, RollFace(random, GetFaces(dice[num3].Color)));
		}
		return FromDice(attack, dice);
	}

	public static AttackResolution RerollHighestDamageDice(AttackResolution attack, int count, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count == 0 || count <= 0)
		{
			return attack;
		}
		AttackDie[] dice = new AttackDie[attack.Dice.Count];
		List<int> list = new List<int>(attack.Dice.Count);
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			dice[i] = attack.Dice[i];
			list.Add(i);
		}
		list.Sort((int first, int second) => ShipDamage(dice[second].Face).CompareTo(ShipDamage(dice[first].Face)));
		Random random = new Random(seed);
		int num = Math.Min(count, list.Count);
		for (int num2 = 0; num2 < num; num2++)
		{
			int num3 = list[num2];
			dice[num3] = new AttackDie(dice[num3].Color, RollFace(random, GetFaces(dice[num3].Color)));
		}
		return FromDice(attack, dice);
	}

	public static AttackResolution RemoveHighestDamageDice(AttackResolution attack, int count)
	{
		if (attack.Dice == null || attack.Dice.Count == 0 || count <= 0)
		{
			return attack;
		}
		List<AttackDie> list = new List<AttackDie>(attack.Dice);
		list.Sort((AttackDie left, AttackDie right) => ShipDamage(right.Face).CompareTo(ShipDamage(left.Face)));
		int count2 = Math.Min(count, list.Count);
		list.RemoveRange(0, count2);
		return FromDice(attack, list);
	}

	public static AttackResolution RerollBlankDice(AttackResolution attack, int count, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count == 0 || count <= 0)
		{
			return attack;
		}
		AttackDie[] array = new AttackDie[attack.Dice.Count];
		Random random = new Random(seed);
		int num = count;
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			array[i] = attack.Dice[i];
			if (num > 0 && array[i].Face == DieFace.Blank)
			{
				array[i] = new AttackDie(array[i].Color, RollFace(random, GetFaces(array[i].Color)));
				num--;
			}
		}
		return FromDice(attack, array);
	}

	public static bool HasDie(AttackResolution attack, AttackDieColor color, DieFace? face = null)
	{
		if (attack.Dice == null)
		{
			return false;
		}
		foreach (AttackDie die in attack.Dice)
		{
			if (die.Color == color && (!face.HasValue || die.Face == face.Value))
			{
				return true;
			}
		}
		return false;
	}

	public static bool HasCriticalIcon(AttackResolution attack, AttackDieColor color)
	{
		if (attack.Dice == null)
		{
			return false;
		}
		foreach (AttackDie die in attack.Dice)
		{
			if (die.Color == color && (die.Face == DieFace.Critical || die.Face == DieFace.HitCritical))
			{
				return true;
			}
		}
		return false;
	}

	public static AttackResolution RerollBlankDiceOfColor(AttackResolution attack, AttackDieColor color, int count, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count == 0 || count <= 0)
		{
			return attack;
		}
		AttackDie[] array = new AttackDie[attack.Dice.Count];
		Random random = new Random(seed);
		int num = count;
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			array[i] = attack.Dice[i];
			if (num > 0 && array[i].Color == color && array[i].Face == DieFace.Blank)
			{
				array[i] = new AttackDie(color, RollFace(random, GetFaces(color)));
				num--;
			}
		}
		return FromDice(attack, array);
	}

	public static AttackResolution RerollFaceDice(AttackResolution attack, DieFace face, int count, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count == 0 || count <= 0)
		{
			return attack;
		}
		AttackDie[] array = new AttackDie[attack.Dice.Count];
		Random random = new Random(seed);
		int num = count;
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			array[i] = attack.Dice[i];
			if (num > 0 && array[i].Face == face)
			{
				array[i] = new AttackDie(array[i].Color, RollFace(random, GetFaces(array[i].Color)));
				num--;
			}
		}
		return FromDice(attack, array);
	}

	public static AttackResolution RerollAllDice(AttackResolution attack, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		AttackDie[] array = new AttackDie[attack.Dice.Count];
		Random random = new Random(seed);
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			AttackDie attackDie = attack.Dice[i];
			array[i] = new AttackDie(attackDie.Color, RollFace(random, GetFaces(attackDie.Color)));
		}
		return FromDice(attack, array);
	}

	public static AttackResolution RerollLowestDieOfColor(AttackResolution attack, AttackDieColor color, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		AttackDie[] array = new AttackDie[attack.Dice.Count];
		int num = -1;
		int num2 = int.MaxValue;
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			array[i] = attack.Dice[i];
			if (array[i].Color == color)
			{
				int num3 = DieUtilityScore(array[i].Face);
				if (num3 < num2)
				{
					num = i;
					num2 = num3;
				}
			}
		}
		if (num < 0)
		{
			return attack;
		}
		Random random = new Random(seed);
		array[num] = new AttackDie(color, RollFace(random, GetFaces(color)));
		return FromDice(attack, array);
	}

	public static AttackResolution RerollLowestDie(AttackResolution attack, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		AttackDie[] array = new AttackDie[attack.Dice.Count];
		int num = 0;
		int num2 = int.MaxValue;
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			array[i] = attack.Dice[i];
			int num3 = DieUtilityScore(array[i].Face);
			if (num3 < num2)
			{
				num = i;
				num2 = num3;
			}
		}
		Random random = new Random(seed);
		array[num] = new AttackDie(array[num].Color, RollFace(random, GetFaces(array[num].Color)));
		return FromDice(attack, array);
	}

	public static AttackResolution RerollDiceOfColor(AttackResolution attack, AttackDieColor color, int maximumCount, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count == 0 || maximumCount <= 0)
		{
			return attack;
		}
		List<AttackDie> dice = new List<AttackDie>(attack.Dice);
		List<int> list = new List<int>();
		for (int i = 0; i < dice.Count; i++)
		{
			if (dice[i].Color == color)
			{
				list.Add(i);
			}
		}
		list.Sort((int left, int right) => DieUtilityScore(dice[left].Face).CompareTo(DieUtilityScore(dice[right].Face)));
		Random random = new Random(seed);
		int num = Math.Min(maximumCount, list.Count);
		for (int num2 = 0; num2 < num; num2++)
		{
			int index = list[num2];
			dice[index] = new AttackDie(color, RollFace(random, GetFaces(color)));
		}
		return FromDice(attack, dice);
	}

	public static AttackResolution ChangeLowestDieToBestFace(AttackResolution attack, bool allowAccuracy = false)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		List<AttackDie> list = new List<AttackDie>(attack.Dice);
		int index = 0;
		for (int i = 1; i < list.Count; i++)
		{
			if (DieUtilityScore(list[i].Face) < DieUtilityScore(list[index].Face))
			{
				index = i;
			}
		}
		AttackDie attackDie = list[index];
		DieFace face = attackDie.Color switch
		{
			AttackDieColor.Red => DieFace.DoubleHit,
			AttackDieColor.Black => DieFace.HitCritical,
			_ => DieFace.Hit,
		};
		if (allowAccuracy && attackDie.Color == AttackDieColor.Blue)
		{
			face = DieFace.Accuracy;
		}
		list[index] = new AttackDie(attackDie.Color, face);
		return FromDice(attack, list);
	}

	public static AttackResolution SpendLowestDieAndSetAccuracy(AttackResolution attack)
	{
		return SpendLowestDieAndSetFace(attack, DieFace.Accuracy);
	}

	public static AttackResolution ChangeAccuracyToCritical(AttackResolution attack)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			if (attack.Dice[i].Face == DieFace.Accuracy)
			{
				return ChangeAccuracyToCritical(attack, attack.Dice[i].Color);
			}
		}
		return attack;
	}

	public static AttackResolution ChangeAccuracyToCritical(AttackResolution attack, AttackDieColor color)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		AttackDie[] array = new AttackDie[attack.Dice.Count];
		bool flag = false;
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			array[i] = attack.Dice[i];
			if (!flag && array[i].Color == color && array[i].Face == DieFace.Accuracy)
			{
				array[i] = new AttackDie(array[i].Color, (array[i].Color == AttackDieColor.Black) ? DieFace.HitCritical : DieFace.Critical);
				flag = true;
			}
		}
		if (!flag)
		{
			return attack;
		}
		return FromDice(attack, array);
	}

	public static AttackResolution ChangeLowestDieOfColor(AttackResolution attack, AttackDieColor color, DieFace face)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		AttackDie[] array = new AttackDie[attack.Dice.Count];
		int num = -1;
		int num2 = int.MaxValue;
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			array[i] = attack.Dice[i];
			if (array[i].Color == color)
			{
				int num3 = DieUtilityScore(array[i].Face);
				if (num3 < num2)
				{
					num = i;
					num2 = num3;
				}
			}
		}
		if (num < 0)
		{
			return attack;
		}
		array[num] = new AttackDie(color, face);
		return FromDice(attack, array);
	}

	public static AttackResolution RollDieAndRemoveLowest(AttackResolution attack, AttackDieColor color, int seed)
	{
		List<AttackDie> list = new List<AttackDie>((attack.Dice?.Count ?? 0) + 1);
		if (attack.Dice != null)
		{
			list.AddRange(attack.Dice);
		}
		Random random = new Random(seed);
		list.Add(new AttackDie(color, RollFace(random, GetFaces(color))));
		if (list.Count <= 1)
		{
			return FromDice(attack, list);
		}
		int index = 0;
		int num = DieUtilityScore(list[0].Face);
		for (int i = 1; i < list.Count; i++)
		{
			int num2 = DieUtilityScore(list[i].Face);
			if (num2 < num)
			{
				index = i;
				num = num2;
			}
		}
		list.RemoveAt(index);
		return FromDice(attack, list);
	}

	public static AttackResolution SpendLowestDieAndRerollBlanks(AttackResolution attack, AttackDieColor spendColor, int rerollCount, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count <= 1)
		{
			return attack;
		}
		int num = -1;
		int num2 = int.MaxValue;
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			AttackDie attackDie = attack.Dice[i];
			if (attackDie.Color == spendColor)
			{
				int num3 = DieUtilityScore(attackDie.Face);
				if (num3 < num2)
				{
					num = i;
					num2 = num3;
				}
			}
		}
		if (num < 0)
		{
			return attack;
		}
		List<AttackDie> list = new List<AttackDie>(attack.Dice.Count - 1);
		for (int j = 0; j < attack.Dice.Count; j++)
		{
			if (j != num)
			{
				list.Add(attack.Dice[j]);
			}
		}
		return RerollBlankDice(FromDice(attack, list), rerollCount, seed);
	}

	public static AttackResolution SpendLowestDieAndSetCritical(AttackResolution attack)
	{
		if (attack.Dice == null || attack.Dice.Count <= 1)
		{
			return attack;
		}
		int num = 0;
		int num2 = DieUtilityScore(attack.Dice[0].Face);
		for (int i = 1; i < attack.Dice.Count; i++)
		{
			int num3 = DieUtilityScore(attack.Dice[i].Face);
			if (num3 < num2)
			{
				num = i;
				num2 = num3;
			}
		}
		List<AttackDie> list = new List<AttackDie>(attack.Dice.Count - 1);
		for (int j = 0; j < attack.Dice.Count; j++)
		{
			if (j != num)
			{
				list.Add(attack.Dice[j]);
			}
		}
		int index = 0;
		int num4 = DieUtilityScore(list[0].Face);
		for (int k = 1; k < list.Count; k++)
		{
			int num5 = DieUtilityScore(list[k].Face);
			if (num5 < num4)
			{
				index = k;
				num4 = num5;
			}
		}
		AttackDie attackDie = list[index];
		list[index] = new AttackDie(attackDie.Color, (attackDie.Color == AttackDieColor.Black) ? DieFace.HitCritical : DieFace.Critical);
		return FromDice(attack, list);
	}

	private static AttackResolution SpendLowestDieAndSetFace(AttackResolution attack, DieFace face)
	{
		if (attack.Dice == null || attack.Dice.Count <= 1)
		{
			return attack;
		}
		int num = 0;
		int num2 = DieUtilityScore(attack.Dice[0].Face);
		for (int i = 1; i < attack.Dice.Count; i++)
		{
			int num3 = DieUtilityScore(attack.Dice[i].Face);
			if (num3 < num2)
			{
				num = i;
				num2 = num3;
			}
		}
		List<AttackDie> list = new List<AttackDie>(attack.Dice.Count - 1);
		for (int j = 0; j < attack.Dice.Count; j++)
		{
			if (j != num)
			{
				list.Add(attack.Dice[j]);
			}
		}
		int index = 0;
		int num4 = DieUtilityScore(list[0].Face);
		for (int k = 1; k < list.Count; k++)
		{
			int num5 = DieUtilityScore(list[k].Face);
			if (num5 < num4)
			{
				index = k;
				num4 = num5;
			}
		}
		list[index] = new AttackDie(list[index].Color, face);
		return FromDice(attack, list);
	}

	public static AttackResolution ChangeAccuracyToHit(AttackResolution attack)
	{
		if (attack.Dice == null)
		{
			return attack;
		}
		AttackDie[] array = new AttackDie[attack.Dice.Count];
		bool flag = false;
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			array[i] = attack.Dice[i];
			if (!flag && array[i].Face == DieFace.Accuracy)
			{
				array[i] = new AttackDie(array[i].Color, (array[i].Color != AttackDieColor.Red) ? DieFace.Hit : DieFace.DoubleHit);
				flag = true;
			}
		}
		if (!flag)
		{
			return attack;
		}
		return FromDice(attack, array);
	}

	public static AttackResolution ChangeLowestDieToAccuracy(AttackResolution attack)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		AttackDie[] array = new AttackDie[attack.Dice.Count];
		int num = 0;
		int num2 = DieUtilityScore(attack.Dice[0].Face);
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			array[i] = attack.Dice[i];
			int num3 = DieUtilityScore(array[i].Face);
			if (num3 < num2)
			{
				num = i;
				num2 = num3;
			}
		}
		array[num] = new AttackDie(array[num].Color, DieFace.Accuracy);
		return FromDice(attack, array);
	}

	public static bool HasDieWithoutSingleHit(AttackResolution attack)
	{
		if (attack.Dice == null)
		{
			return false;
		}
		foreach (AttackDie die in attack.Dice)
		{
			if (die.Face != DieFace.Hit)
			{
				return true;
			}
		}
		return false;
	}

	public static AttackResolution ChangeLowestDieToSingleHit(AttackResolution attack)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		AttackDie[] array = new AttackDie[attack.Dice.Count];
		int num = -1;
		int num2 = int.MaxValue;
		for (int i = 0; i < attack.Dice.Count; i++)
		{
			array[i] = attack.Dice[i];
			if (array[i].Face != DieFace.Hit)
			{
				int num3 = DieUtilityScore(array[i].Face);
				if (num3 < num2)
				{
					num = i;
					num2 = num3;
				}
			}
		}
		if (num < 0)
		{
			return attack;
		}
		array[num] = new AttackDie(array[num].Color, DieFace.Hit);
		return FromDice(attack, array);
	}

	public static AttackResolution AddFixedDie(AttackResolution attack, AttackDieColor color, DieFace face)
	{
		List<AttackDie> list = new List<AttackDie>((attack.Dice?.Count ?? 0) + 1);
		if (attack.Dice != null)
		{
			list.AddRange(attack.Dice);
		}
		list.Add(new AttackDie(color, face));
		return FromDice(attack, list);
	}

	public static AttackResolution AddRolledDice(AttackResolution attack, AttackDieColor color, int count, int seed)
	{
		if (color == AttackDieColor.None || count <= 0)
		{
			return attack;
		}
		List<AttackDie> list = new List<AttackDie>((attack.Dice?.Count ?? 0) + count);
		if (attack.Dice != null)
		{
			list.AddRange(attack.Dice);
		}
		Random random = new Random(seed);
		for (int i = 0; i < count; i++)
		{
			list.Add(new AttackDie(color, RollFace(random, GetFaces(color))));
		}
		return FromDice(attack, list);
	}

	public static AttackResolution SpendAccuracyDie(AttackResolution attack)
	{
		return RemoveFirstMatchingDie(attack, (AttackDie die) => die.Face == DieFace.Accuracy);
	}

	public static AttackResolution SpendCriticalIconDie(AttackResolution attack, AttackDieColor? color = null)
	{
		return RemoveFirstMatchingDie(attack, (AttackDie die) => (!color.HasValue || die.Color == color.Value) && (die.Face == DieFace.Critical || die.Face == DieFace.HitCritical));
	}

	public static AttackResolution SpendHitOrAccuracyDie(AttackResolution attack)
	{
		return RemoveFirstMatchingDie(attack, (AttackDie die) => die.Face == DieFace.Hit || die.Face == DieFace.DoubleHit || die.Face == DieFace.HitCritical || die.Face == DieFace.Accuracy);
	}

	public static AttackResolution RerollCriticalIconDie(AttackResolution attack, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count == 0)
		{
			return attack;
		}
		List<AttackDie> list = new List<AttackDie>(attack.Dice);
		int num = list.FindIndex((AttackDie die) => die.Face == DieFace.Critical || die.Face == DieFace.HitCritical);
		if (num < 0)
		{
			return attack;
		}
		AttackDie attackDie = list[num];
		list[num] = new AttackDie(attackDie.Color, RollFace(new Random(seed), GetFaces(attackDie.Color)));
		return FromDice(attack, list);
	}

	public static AttackResolution RemoveDiceAtIndices(AttackResolution attack, IReadOnlyCollection<int> selectedIndices)
	{
		if (attack.Dice == null || attack.Dice.Count == 0 || selectedIndices == null || selectedIndices.Count == 0)
		{
			return attack;
		}
		List<AttackDie> list = new List<AttackDie>(attack.Dice);
		List<int> list2 = new List<int>(selectedIndices);
		list2.Sort((int first, int second) => second.CompareTo(first));
		foreach (int item in list2)
		{
			if (item >= 0 && item < list.Count)
			{
				list.RemoveAt(item);
			}
		}
		return FromDice(attack, list);
	}

	public static AttackResolution RerollDiceAtIndices(AttackResolution attack, IReadOnlyCollection<int> selectedIndices, int seed)
	{
		if (attack.Dice == null || attack.Dice.Count == 0 || selectedIndices == null || selectedIndices.Count == 0)
		{
			return attack;
		}
		List<AttackDie> list = new List<AttackDie>(attack.Dice);
		Random random = new Random(seed);
		foreach (int selectedIndex in selectedIndices)
		{
			if (selectedIndex >= 0 && selectedIndex < list.Count)
			{
				AttackDie attackDie = list[selectedIndex];
				list[selectedIndex] = new AttackDie(attackDie.Color, RollFace(random, GetFaces(attackDie.Color)));
			}
		}
		return FromDice(attack, list);
	}

	public static AttackResolution ChangeDieFaceAt(AttackResolution attack, int selectedIndex, DieFace face)
	{
		if (attack.Dice == null || selectedIndex < 0 || selectedIndex >= attack.Dice.Count)
		{
			return attack;
		}
		List<AttackDie> list = new List<AttackDie>(attack.Dice);
		list[selectedIndex] = new AttackDie(list[selectedIndex].Color, face);
		return FromDice(attack, list);
	}

	private static AttackResolution RemoveFirstMatchingDie(AttackResolution attack, Predicate<AttackDie> predicate)
	{
		if (attack.Dice == null || attack.Dice.Count == 0 || predicate == null)
		{
			return attack;
		}
		List<AttackDie> list = new List<AttackDie>(attack.Dice);
		int num = list.FindIndex(predicate);
		if (num < 0)
		{
			return attack;
		}
		list.RemoveAt(num);
		return FromDice(attack, list);
	}

	public static AttackResolution AddUnspentAccuracyDamage(AttackResolution attack, int accuracyIcons)
	{
		int num = Math.Min(Math.Max(0, accuracyIcons), attack.Accuracies);
		if (num <= 0)
		{
			return attack;
		}
		return new AttackResolution(attack.Range, attack.DiceRolled, attack.Hits + num, attack.Criticals, attack.Accuracies - num, attack.Blanks, attack.IsLegal, attack.CriticalDamagePerIcon, attack.Dice);
	}

	private static int DieUtilityScore(DieFace face)
	{
		return face switch
		{
			DieFace.Blank => 0,
			DieFace.Accuracy => 1,
			DieFace.Hit => 2,
			DieFace.Critical => 2,
			DieFace.DoubleHit => 3,
			DieFace.HitCritical => 3,
			_ => 0,
		};
	}

	private static int SquadronDamage(DieFace face)
	{
		return face switch
		{
			DieFace.DoubleHit => 2,
			DieFace.Hit => 1,
			DieFace.HitCritical => 1,
			_ => 0,
		};
	}

	private static int ShipDamage(DieFace face)
	{
		return face switch
		{
			DieFace.DoubleHit => 2,
			DieFace.HitCritical => 2,
			DieFace.Hit => 1,
			DieFace.Critical => 1,
			_ => 0,
		};
	}

	private static AttackResolution FromDice(AttackResolution original, IReadOnlyList<AttackDie> dice)
	{
		int num = 0;
		int num2 = 0;
		int num3 = 0;
		int num4 = 0;
		using (IEnumerator<AttackDie> enumerator = dice.GetEnumerator())
		{
			while (enumerator.MoveNext())
			{
				switch (enumerator.Current.Face)
				{
				case DieFace.Hit:
					num++;
					break;
				case DieFace.DoubleHit:
					num += 2;
					break;
				case DieFace.Critical:
					num2++;
					break;
				case DieFace.HitCritical:
					num++;
					num2++;
					break;
				case DieFace.Accuracy:
					num3++;
					break;
				default:
					num4++;
					break;
				}
			}
		}
		return new AttackResolution(original.Range, dice.Count, num, num2, num3, num4, original.IsLegal, original.CriticalDamagePerIcon, dice);
	}

	private static void RollDice(Random random, AttackDieColor color, int count, DieFace[] faces, List<DieRoll> rolls)
	{
		for (int i = 0; i < count; i++)
		{
			rolls.Add(new DieRoll(color, RollFace(random, faces)));
		}
	}

	private static DieFace RollFace(Random random, DieFace[] faces)
	{
		return faces[random.Next(0, faces.Length)];
	}

	private static DieFace[] GetFaces(AttackDieColor color)
	{
		return color switch
		{
			AttackDieColor.Red => RedFaces,
			AttackDieColor.Blue => BlueFaces,
			AttackDieColor.Black => BlackFaces,
			_ => RedFaces,
		};
	}
}
