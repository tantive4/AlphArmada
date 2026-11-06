#nullable disable
using System;
using System.Collections.Generic;
using Armada.Data;

namespace Armada.Rules;

public static class MovementResolver
{
	public static IReadOnlyList<ManeuverCourse> EnumerateOrdinaryLegalCourses(ShipDefinition definition, int speed)
	{
		int effectiveSpeed;
		int[] printedAllowances = GetPrintedAllowances(definition, speed, out effectiveSpeed);
		if (effectiveSpeed == 0)
		{
			return new ManeuverCourse[1] { ManeuverCourse.Standstill };
		}
		return ManeuverCourse.Enumerate(printedAllowances);
	}

	public static IReadOnlyList<ManeuverCourse> EnumerateLegalCourses(ShipDefinition definition, int speed, bool navigateCommand, int finalYawPenalty)
	{
		int effectiveSpeed;
		int[] printedAllowances = GetPrintedAllowances(definition, speed, out effectiveSpeed);
		if (effectiveSpeed == 0)
		{
			return new ManeuverCourse[1] { ManeuverCourse.Standstill };
		}
		int[] array = (int[])printedAllowances.Clone();
		ApplyFinalAdjustableJointPenalty(printedAllowances, array, finalYawPenalty);
		List<ManeuverCourse> list = new List<ManeuverCourse>();
		HashSet<string> seen = new HashSet<string>();
		AddUniqueCourses(ManeuverCourse.Enumerate(array), list, seen);
		if (navigateCommand)
		{
			for (int i = 0; i < array.Length; i++)
			{
				if (array[i] < 2)
				{
					int[] obj = (int[])array.Clone();
					obj[i]++;
					AddUniqueCourses(ManeuverCourse.Enumerate(obj), list, seen);
				}
			}
		}
		list.Sort(CompareCourses);
		return list;
	}

	public static IReadOnlyList<MovementSolution> EnumerateLegalSolutions(ShipDefinition definition, int speed, bool navigateCommand = false, int finalYawPenalty = 0)
	{
		IReadOnlyList<ManeuverCourse> readOnlyList = EnumerateLegalCourses(definition, speed, navigateCommand, finalYawPenalty);
		MovementSolution[] array = new MovementSolution[readOnlyList.Count];
		for (int i = 0; i < readOnlyList.Count; i++)
		{
			array[i] = Resolve(readOnlyList[i]);
		}
		return array;
	}

	public static MovementSolution Resolve(ManeuverCourse course)
	{
		return new MovementSolution(course);
	}

	public static int[] GetEffectiveAllowances(ShipDefinition definition, int speed, int finalYawPenalty, out int effectiveSpeed)
	{
		int[] printedAllowances = GetPrintedAllowances(definition, speed, out effectiveSpeed);
		if (effectiveSpeed == 0)
		{
			return Array.Empty<int>();
		}
		int[] array = (int[])printedAllowances.Clone();
		ApplyFinalAdjustableJointPenalty(printedAllowances, array, finalYawPenalty);
		return array;
	}

	public static MovementChoice Classify(ManeuverCourse course)
	{
		if (course == null || course.TotalClicks == 0)
		{
			return MovementChoice.Ahead;
		}
		if (course.TotalClicks <= -2)
		{
			return MovementChoice.HardPort;
		}
		if (course.TotalClicks < 0)
		{
			return MovementChoice.Port;
		}
		if (course.TotalClicks >= 2)
		{
			return MovementChoice.HardStarboard;
		}
		return MovementChoice.Starboard;
	}

	public static MovementSolution Resolve(MovementChoice choice, int speed, bool navigateCommand = false)
	{
		int num = Math.Clamp(speed, 0, 4);
		if (num == 0)
		{
			return new MovementSolution(0f, 0f);
		}
		float forwardDistance = 3.3f + (float)num * 1.7f;
		float num2 = 18f + (float)num * 5f;
		float num3 = num2 + (navigateCommand ? 24f : 0f);
		return choice switch
		{
			MovementChoice.HardPort => new MovementSolution(forwardDistance, 0f - num3),
			MovementChoice.Port => new MovementSolution(forwardDistance, 0f - num2),
			MovementChoice.Starboard => new MovementSolution(forwardDistance, num2),
			MovementChoice.HardStarboard => new MovementSolution(forwardDistance, num3),
			_ => new MovementSolution(forwardDistance, 0f),
		};
	}

	private static int[] GetPrintedAllowances(ShipDefinition definition, int speed, out int effectiveSpeed)
	{
		int max = ((definition != null) ? Math.Max(0, definition.ChartedMaximumSpeed) : Math.Max(0, speed));
		effectiveSpeed = Math.Clamp(speed, 0, max);
		if (effectiveSpeed == 0)
		{
			return Array.Empty<int>();
		}
		int[] jointYaw = ((definition != null) ? definition.GetManeuverRow(effectiveSpeed) : default(ManeuverSpeedRow)).JointYaw.ToArray();
		int[] array = new int[effectiveSpeed];
		for (int i = 0; i < array.Length; i++)
		{
			array[i] = ((i < jointYaw.Length) ? Math.Clamp(jointYaw[i], 0, 2) : 0);
		}
		return array;
	}

	private static void ApplyFinalAdjustableJointPenalty(IReadOnlyList<int> printed, int[] effective, int penalty)
	{
		if (penalty <= 0)
		{
			return;
		}
		int num = -1;
		for (int num2 = printed.Count - 1; num2 >= 0; num2--)
		{
			if (printed[num2] > 0)
			{
				num = num2;
				break;
			}
		}
		if (num >= 0)
		{
			effective[num] = Math.Max(0, effective[num] - penalty);
		}
	}

	private static void AddUniqueCourses(IReadOnlyList<ManeuverCourse> source, ICollection<ManeuverCourse> destination, ISet<string> seen)
	{
		for (int i = 0; i < source.Count; i++)
		{
			ManeuverCourse maneuverCourse = source[i];
			if (seen.Add(maneuverCourse.Id))
			{
				destination.Add(maneuverCourse);
			}
		}
	}

	private static int CompareCourses(ManeuverCourse left, ManeuverCourse right)
	{
		int num = left.TotalAbsoluteClicks.CompareTo(right.TotalAbsoluteClicks);
		if (num != 0)
		{
			return num;
		}
		num = Math.Abs(left.TotalClicks).CompareTo(Math.Abs(right.TotalClicks));
		if (num != 0)
		{
			return num;
		}
		num = left.TotalClicks.CompareTo(right.TotalClicks);
		if (num != 0)
		{
			return num;
		}
		return string.CompareOrdinal(left.Id, right.Id);
	}
}
