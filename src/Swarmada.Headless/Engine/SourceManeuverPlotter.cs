#nullable disable
using System;
using System.Collections.Generic;
using Armada.Data;

namespace Armada.Rules;

public sealed class ManeuverPlotter
{
	private readonly int[] printedAllowances;

	private readonly int[] effectiveAllowances;

	private readonly List<int> committedClicks = new List<int>();

	public int JointIndex => committedClicks.Count;

	public int TotalJoints => effectiveAllowances.Length;

	public bool IsComplete => JointIndex >= TotalJoints;

	public bool IsStandstill => TotalJoints == 0;

	public bool NavigateAvailable { get; private set; }

	public bool NavigateSpent
	{
		get
		{
			if (HasNavigate)
			{
				return !NavigateAvailable;
			}
			return false;
		}
	}

	public bool HasNavigate { get; }

	public int NavigateJoint { get; private set; } = -1;

	public ManeuverPlotter(ShipDefinition definition, int speed, bool navigateCommand, int finalYawPenalty, int finalYawBonus = 0)
	{
		printedAllowances = MovementResolver.GetEffectiveAllowances(definition, speed, finalYawPenalty, out var _);
		effectiveAllowances = (int[])printedAllowances.Clone();
		if (effectiveAllowances.Length != 0 && finalYawBonus > 0)
		{
			int num = effectiveAllowances.Length - 1;
			effectiveAllowances[num] = Math.Min(2, effectiveAllowances[num] + finalYawBonus);
		}
		HasNavigate = navigateCommand;
		NavigateAvailable = navigateCommand;
	}

	public IReadOnlyList<int> LegalClicksForCurrentJoint()
	{
		List<int> list = new List<int>();
		if (IsComplete)
		{
			return list;
		}
		int num = effectiveAllowances[JointIndex];
		if (NavigateAvailable && num < 2)
		{
			num++;
		}
		for (int i = -num; i <= num; i++)
		{
			list.Add(i);
		}
		return list;
	}

	public bool ClickRequiresNavigate(int click)
	{
		if (IsComplete || !NavigateAvailable)
		{
			return false;
		}
		return Math.Abs(click) > effectiveAllowances[JointIndex];
	}

	public ManeuverCourse ProjectWithClick(int click)
	{
		int[] array = new int[TotalJoints];
		for (int i = 0; i < committedClicks.Count; i++)
		{
			array[i] = committedClicks[i];
		}
		if (!IsComplete)
		{
			array[JointIndex] = click;
		}
		return new ManeuverCourse(array);
	}

	public IReadOnlyList<ManeuverCourse> EnumerateCompletionsForCurrentClick(int click)
	{
		List<ManeuverCourse> list = new List<ManeuverCourse>();
		if (IsComplete)
		{
			return list;
		}
		bool flag = ClickRequiresNavigate(click);
		int num = effectiveAllowances[JointIndex] + (flag ? 1 : 0);
		if (Math.Abs(click) > num)
		{
			return list;
		}
		int[] array = new int[TotalJoints];
		for (int i = 0; i < committedClicks.Count; i++)
		{
			array[i] = committedClicks[i];
		}
		array[JointIndex] = click;
		EnumerateCompletions(JointIndex + 1, NavigateAvailable && !flag, array, list);
		return list;
	}

	private void EnumerateCompletions(int joint, bool navigateAvailable, int[] clicks, List<ManeuverCourse> courses)
	{
		if (joint >= TotalJoints)
		{
			courses.Add(new ManeuverCourse((int[])clicks.Clone()));
			return;
		}
		int num = effectiveAllowances[joint];
		int num2 = num;
		if (navigateAvailable && num2 < 2)
		{
			num2++;
		}
		for (int i = -num2; i <= num2; i++)
		{
			bool flag = navigateAvailable && Math.Abs(i) > num;
			clicks[joint] = i;
			EnumerateCompletions(joint + 1, navigateAvailable && !flag, clicks, courses);
		}
	}

	public ManeuverCourse CurrentCourse()
	{
		int[] array = new int[TotalJoints];
		for (int i = 0; i < committedClicks.Count; i++)
		{
			array[i] = committedClicks[i];
		}
		return new ManeuverCourse(array);
	}

	public bool Commit(int click)
	{
		if (IsComplete)
		{
			return false;
		}
		bool flag = ClickRequiresNavigate(click);
		int num = effectiveAllowances[JointIndex] + (flag ? 1 : 0);
		if (Math.Abs(click) > num)
		{
			return false;
		}
		if (flag)
		{
			NavigateAvailable = false;
			NavigateJoint = JointIndex;
		}
		committedClicks.Add(click);
		return true;
	}

	public bool Back()
	{
		if (committedClicks.Count == 0)
		{
			return false;
		}
		int num = committedClicks.Count - 1;
		committedClicks.RemoveAt(num);
		if (NavigateJoint == num)
		{
			NavigateJoint = -1;
			NavigateAvailable = HasNavigate;
		}
		return true;
	}

	public void Reset()
	{
		committedClicks.Clear();
		NavigateJoint = -1;
		NavigateAvailable = HasNavigate;
	}

	public int PrintedAllowanceAt(int joint)
	{
		if (joint < 0 || joint >= effectiveAllowances.Length)
		{
			return 0;
		}
		return effectiveAllowances[joint];
	}
}
