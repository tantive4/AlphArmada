#nullable disable
using System;
using System.Collections.Generic;
using System.Text;

namespace Armada.Rules;

public sealed class ManeuverCourse : IEquatable<ManeuverCourse>
{
	private readonly int[] jointClicks;

	public static ManeuverCourse Standstill { get; } = new ManeuverCourse(Array.Empty<int>());

	public int Speed => jointClicks.Length;

	public IReadOnlyList<int> JointClicks => jointClicks;

	public string Id { get; }

	public int TotalClicks { get; }

	public int TotalAbsoluteClicks { get; }

	public bool IsStraight => TotalAbsoluteClicks == 0;

	public ManeuverCourse(IReadOnlyList<int> clicks)
	{
		if (clicks == null || clicks.Count == 0)
		{
			jointClicks = Array.Empty<int>();
		}
		else
		{
			jointClicks = new int[clicks.Count];
			for (int i = 0; i < clicks.Count; i++)
			{
				jointClicks[i] = clicks[i];
			}
		}
		int num = 0;
		int num2 = 0;
		for (int j = 0; j < jointClicks.Length; j++)
		{
			num += jointClicks[j];
			num2 += Math.Abs(jointClicks[j]);
		}
		TotalClicks = num;
		TotalAbsoluteClicks = num2;
		Id = BuildId(jointClicks);
	}

	public int GetClick(int joint)
	{
		if (joint < 0 || joint >= jointClicks.Length)
		{
			return 0;
		}
		return jointClicks[joint];
	}

	public ManeuverCourse Truncate(int speed)
	{
		int num = Math.Clamp(speed, 0, Speed);
		if (num == Speed)
		{
			return this;
		}
		if (num == 0)
		{
			return Standstill;
		}
		int[] array = new int[num];
		Array.Copy(jointClicks, array, num);
		return new ManeuverCourse(array);
	}

	public ManeuverCourse Mirror()
	{
		if (Speed == 0)
		{
			return this;
		}
		int[] array = new int[Speed];
		for (int i = 0; i < Speed; i++)
		{
			array[i] = -jointClicks[i];
		}
		return new ManeuverCourse(array);
	}

	public static IReadOnlyList<ManeuverCourse> Enumerate(IReadOnlyList<int> maximumYawPerJoint)
	{
		if (maximumYawPerJoint == null || maximumYawPerJoint.Count == 0)
		{
			return new ManeuverCourse[1] { Standstill };
		}
		int[] working = new int[maximumYawPerJoint.Count];
		List<ManeuverCourse> list = new List<ManeuverCourse>();
		EnumerateJoint(maximumYawPerJoint, working, 0, list);
		list.Sort(CompareForPresentation);
		return list;
	}

	public bool Equals(ManeuverCourse other)
	{
		if (this == other)
		{
			return true;
		}
		if (other == null || Speed != other.Speed)
		{
			return false;
		}
		for (int i = 0; i < Speed; i++)
		{
			if (jointClicks[i] != other.jointClicks[i])
			{
				return false;
			}
		}
		return true;
	}

	public override bool Equals(object obj)
	{
		if (obj is ManeuverCourse other)
		{
			return Equals(other);
		}
		return false;
	}

	public override int GetHashCode()
	{
		int num = 17;
		for (int i = 0; i < jointClicks.Length; i++)
		{
			num = num * 31 + jointClicks[i];
		}
		return num;
	}

	public override string ToString()
	{
		return Id;
	}

	private static void EnumerateJoint(IReadOnlyList<int> maximumYawPerJoint, int[] working, int joint, ICollection<ManeuverCourse> results)
	{
		if (joint >= working.Length)
		{
			results.Add(new ManeuverCourse(working));
			return;
		}
		int num = Math.Clamp(maximumYawPerJoint[joint], 0, 2);
		for (int i = -num; i <= num; i++)
		{
			working[joint] = i;
			EnumerateJoint(maximumYawPerJoint, working, joint + 1, results);
		}
	}

	private static int CompareForPresentation(ManeuverCourse left, ManeuverCourse right)
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
		for (int i = 0; i < left.Speed; i++)
		{
			num = left.GetClick(i).CompareTo(right.GetClick(i));
			if (num != 0)
			{
				return num;
			}
		}
		return 0;
	}

	private static string BuildId(IReadOnlyList<int> clicks)
	{
		StringBuilder stringBuilder = new StringBuilder();
		stringBuilder.Append('S');
		stringBuilder.Append(clicks.Count);
		stringBuilder.Append('[');
		for (int i = 0; i < clicks.Count; i++)
		{
			if (i > 0)
			{
				stringBuilder.Append(',');
			}
			if (clicks[i] > 0)
			{
				stringBuilder.Append('+');
			}
			stringBuilder.Append(clicks[i]);
		}
		stringBuilder.Append(']');
		return stringBuilder.ToString();
	}
}
