#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Fleet;

public sealed class SquadronFleetMetadata
{
	public IReadOnlyList<string> UniqueClasses { get; }

	public int DefenseTokenCount { get; }

	public bool HasCompleteDefenseTokenMetadata { get; }

	public GameplayCoverage GameplayCoverage { get; }

	public SquadronFleetMetadata(IEnumerable<string> uniqueClasses, int defenseTokenCount, bool hasCompleteDefenseTokenMetadata, GameplayCoverage gameplayCoverage = GameplayCoverage.Unknown)
	{
		UniqueClasses = FleetContentText.CopyValues(uniqueClasses);
		DefenseTokenCount = Math.Max(0, defenseTokenCount);
		HasCompleteDefenseTokenMetadata = hasCompleteDefenseTokenMetadata;
		GameplayCoverage = gameplayCoverage;
	}
}
