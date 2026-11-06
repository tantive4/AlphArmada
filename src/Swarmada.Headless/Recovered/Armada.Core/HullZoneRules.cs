#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Core;

public static class HullZoneRules
{
	private static readonly HullZone[] Standard = new HullZone[4]
	{
		HullZone.Front,
		HullZone.Left,
		HullZone.Right,
		HullZone.Rear
	};

	private static readonly HullZone[] Huge = new HullZone[6]
	{
		HullZone.Front,
		HullZone.Left,
		HullZone.Right,
		HullZone.Rear,
		HullZone.LeftAuxiliary,
		HullZone.RightAuxiliary
	};

	public static IReadOnlyList<HullZone> Zones(bool auxiliary)
	{
		if (!auxiliary)
		{
			return Standard;
		}
		return Huge;
	}

	public static HullZone[] Adjacent(HullZone zone, bool auxiliary = false)
	{
		switch (zone)
		{
		case HullZone.Front:
			return new HullZone[2]
			{
				HullZone.Left,
				HullZone.Right
			};
		case HullZone.Rear:
			if (auxiliary)
			{
				return new HullZone[2]
				{
					HullZone.LeftAuxiliary,
					HullZone.RightAuxiliary
				};
			}
			return new HullZone[2]
			{
				HullZone.Left,
				HullZone.Right
			};
		case HullZone.Left:
			return new HullZone[2]
			{
				HullZone.Front,
				auxiliary ? HullZone.LeftAuxiliary : HullZone.Rear
			};
		case HullZone.Right:
			return new HullZone[2]
			{
				HullZone.Front,
				auxiliary ? HullZone.RightAuxiliary : HullZone.Rear
			};
		case HullZone.LeftAuxiliary:
			return new HullZone[2]
			{
				HullZone.Left,
				HullZone.Rear
			};
		case HullZone.RightAuxiliary:
			return new HullZone[2]
			{
				HullZone.Right,
				HullZone.Rear
			};
		default:
			return Array.Empty<HullZone>();
		}
	}

	public static string Label(HullZone zone)
	{
		return zone switch
		{
			HullZone.LeftAuxiliary => "LEFT AUXILIARY",
			HullZone.RightAuxiliary => "RIGHT AUXILIARY",
			_ => zone.ToString().ToUpperInvariant(),
		};
	}
}
