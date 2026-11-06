#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Fleet;

public sealed class SquadronContentDescriptor
{
	public string Id { get; }

	public string DisplayName { get; }

	public string Faction { get; }

	public int Points { get; }

	public bool IsUnique { get; }

	public bool IsSingleCraft { get; }

	public bool HasDefenseTokenMetadata { get; }

	public int DefenseTokenCount { get; }

	public IReadOnlyList<string> UniqueClasses { get; }

	public IReadOnlyList<string> Keywords { get; }

	public GameplayCoverage GameplayCoverage { get; }

	public bool HasDefenseTokens
	{
		get
		{
			if (HasDefenseTokenMetadata)
			{
				return DefenseTokenCount > 0;
			}
			return false;
		}
	}

	public SquadronContentDescriptor(string id, string displayName, string faction, int points, bool isUnique, bool isSingleCraft, int defenseTokenCount, IEnumerable<string> uniqueClasses = null, bool hasDefenseTokenMetadata = true, GameplayCoverage gameplayCoverage = GameplayCoverage.Unknown, IEnumerable<string> keywords = null)
	{
		Id = FleetContentText.RequiredId(id, "id");
		DisplayName = FleetContentText.ValueOrId(displayName, Id);
		Faction = FleetContentText.NormalizeFaction(faction);
		Points = Math.Max(0, points);
		IsUnique = isUnique;
		IsSingleCraft = isSingleCraft;
		DefenseTokenCount = Math.Max(0, defenseTokenCount);
		HasDefenseTokenMetadata = hasDefenseTokenMetadata;
		UniqueClasses = FleetContentText.CopyValues(uniqueClasses);
		Keywords = FleetContentText.CopyValues(keywords);
		GameplayCoverage = gameplayCoverage;
	}
}
