#nullable disable
using System.Collections.Generic;

namespace Armada.Fleet;

public sealed class ShipFleetMetadata
{
	public string Size { get; }

	public IReadOnlyList<string> Traits { get; }

	public bool? IsFlotilla { get; }

	public GameplayCoverage GameplayCoverage { get; }

	public ShipFleetMetadata(string size, IEnumerable<string> traits, bool? isFlotilla = null, GameplayCoverage gameplayCoverage = GameplayCoverage.Unknown)
	{
		Size = FleetContentText.Normalize(size);
		Traits = FleetContentText.CopyValues(traits);
		IsFlotilla = isFlotilla;
		GameplayCoverage = gameplayCoverage;
	}
}
