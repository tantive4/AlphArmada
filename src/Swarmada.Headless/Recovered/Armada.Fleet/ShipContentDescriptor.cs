#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Fleet;

public sealed class ShipContentDescriptor
{
	public string Id { get; }

	public string DisplayName { get; }

	public string Faction { get; }

	public int Points { get; }

	public string ShipType { get; }

	public string Size { get; }

	public bool HasSizeMetadata { get; }

	public bool HasTraitMetadata { get; }

	public bool? IsFlotilla { get; }

	public IReadOnlyList<string> Traits { get; }

	public IReadOnlyList<string> UpgradeSlots { get; }

	public GameplayCoverage GameplayCoverage { get; }

	public ShipContentDescriptor(string id, string displayName, string faction, int points, string shipType, IEnumerable<string> upgradeSlots, string size = "", IEnumerable<string> traits = null, bool hasTraitMetadata = false, bool? isFlotilla = null, GameplayCoverage gameplayCoverage = GameplayCoverage.Unknown)
	{
		Id = FleetContentText.RequiredId(id, "id");
		DisplayName = FleetContentText.ValueOrId(displayName, Id);
		Faction = FleetContentText.NormalizeFaction(faction);
		Points = Math.Max(0, points);
		ShipType = FleetContentText.Normalize(shipType);
		Size = FleetContentText.Normalize(size);
		HasSizeMetadata = Size.Length > 0;
		HasTraitMetadata = hasTraitMetadata || traits != null;
		Traits = FleetContentText.CopyValues(traits);
		UpgradeSlots = FleetContentText.CopyValues(upgradeSlots, preserveDuplicates: true);
		IsFlotilla = isFlotilla ?? InferFlotilla(Size);
		GameplayCoverage = gameplayCoverage;
	}

	private static bool? InferFlotilla(string size)
	{
		if (string.IsNullOrEmpty(size))
		{
			return null;
		}
		if (string.Equals(size, "flotilla", StringComparison.OrdinalIgnoreCase))
		{
			return true;
		}
		if (string.Equals(size, "small", StringComparison.OrdinalIgnoreCase) || string.Equals(size, "medium", StringComparison.OrdinalIgnoreCase) || string.Equals(size, "large", StringComparison.OrdinalIgnoreCase) || string.Equals(size, "huge", StringComparison.OrdinalIgnoreCase))
		{
			return false;
		}
		return null;
	}
}
