#nullable disable
using Swarmada.Headless.Engine;
using System.Numerics;
using Swarmada.Headless.Geometry;
using Mathf = Swarmada.Headless.Engine.EngineMath;

namespace Armada.Rules;

public static class RaidRules
{
	public const string CaptainRex = "captain-rex-officer";

	private const float RexDistance = 8.227571f;

	public static int DiscardLimitFor(ShipView ship)
	{
		if (ship == null)
		{
			return int.MaxValue;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId != ship.OwnerId)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive("captain-rex-officer") && FootprintDistance(shipView, ship) <= 8.227571f)
				{
					return 1;
				}
			}
		}
		return int.MaxValue;
	}

	public static int BlackDiceFromRaidTokens(ShipView defender)
	{
		if (defender?.Commands == null)
		{
			return 0;
		}
		int raidTokenCount = defender.Commands.RaidTokenCount;
		if (raidTokenCount >= 2)
		{
			return 2;
		}
		return raidTokenCount;
	}

	public static float FootprintDistance(ShipView first, ShipView second)
	{
		if (first == null || second == null)
		{
			return float.PositiveInfinity;
		}
		if (!ReferenceEquals(first.Game, second.Game)) throw new ArgumentException("Unit range requires one game context.");
		return EntityGeometry.Distance(first.Game, first.Index, second.Index) / TabletopMeasurements.MillimetresPerWorldUnit;
	}

	private static Vector2[] FootprintCorners(ShipView ship)
	{
		GameplayFootprintView component = ship.GetComponent<GameplayFootprintView>();
		Vector2 obj = ((component != null) ? component.SourceSize : ship.Definition.GameplayFootprint);
		float num = obj.X * 0.5f;
		float num2 = obj.Y * 0.5f;
		UnitTransform transform = ship.transform;
		Vector3 position = transform.position;
		Vector3 normalized = EngineVectors.ProjectOnPlane(transform.right, Vector3.UnitY).NormalizeSafe();
		Vector3 normalized2 = EngineVectors.ProjectOnPlane(transform.forward, Vector3.UnitY).NormalizeSafe();
		Vector2[] array = new Vector2[4];
		(int, int)[] array2 = new(int, int)[4]
		{
			(1, 1),
			(1, -1),
			(-1, -1),
			(-1, 1)
		};
		for (int i = 0; i < 4; i++)
		{
			Vector3 vector = position + normalized * (num * (float)array2[i].Item1) + normalized2 * (num2 * (float)array2[i].Item2);
			array[i] = new Vector2(vector.X, vector.Z);
		}
		return array;
	}

	private static bool FootprintsOverlap(Vector2[] a, Vector2[] b)
	{
		for (int i = 0; i < 2; i++)
		{
			Vector2[] array = ((i == 0) ? a : b);
			for (int j = 0; j < 2; j++)
			{
				Vector2 vector = array[(j + 1) % 4] - array[j];
				Vector2 vector2 = new Vector2(0f - vector.Y, vector.X);
				if (!(vector2.LengthSquared() < 1E-06f))
				{
					vector2 = vector2.NormalizeSafe();
					Project(a, vector2, out var min, out var max);
					Project(b, vector2, out var min2, out var max2);
					if (max < min2 || max2 < min)
					{
						return false;
					}
				}
			}
		}
		return true;
	}

	private static void Project(Vector2[] poly, Vector2 axis, out float min, out float max)
	{
		min = float.PositiveInfinity;
		max = float.NegativeInfinity;
		for (int i = 0; i < 4; i++)
		{
			float num = Vector2.Dot(poly[i], axis);
			if (num < min)
			{
				min = num;
			}
			if (num > max)
			{
				max = num;
			}
		}
	}

	private static float SegmentDistance(Vector2 p1, Vector2 p2, Vector2 p3, Vector2 p4)
	{
		return Mathf.Min(Mathf.Min(PointSegmentDistance(p1, p3, p4), PointSegmentDistance(p2, p3, p4)), Mathf.Min(PointSegmentDistance(p3, p1, p2), PointSegmentDistance(p4, p1, p2)));
	}

	private static float PointSegmentDistance(Vector2 point, Vector2 start, Vector2 end)
	{
		Vector2 vector = end - start;
		float sqrMagnitude = vector.LengthSquared();
		if (sqrMagnitude < 1E-06f)
		{
			return Vector2.Distance(point, start);
		}
		float num = Mathf.Clamp01(Vector2.Dot(point - start, vector) / sqrMagnitude);
		return Vector2.Distance(point, start + vector * num);
	}
}
