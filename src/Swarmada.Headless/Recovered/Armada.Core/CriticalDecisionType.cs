#nullable disable
namespace Armada.Core;

public enum CriticalDecisionType
{
	None,
	Acknowledge,
	ChooseHullZone,
	ChooseUpToTwoHullZones,
	ChooseDefenseToken,
	OpponentSpeedOrCommand
}
