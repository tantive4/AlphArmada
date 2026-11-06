#nullable disable
namespace Armada.Session;

public readonly struct MatchUnitValidationRules
{
	public int MaximumSpeed { get; }

	public float FootprintWidth { get; }

	public float FootprintLength { get; }

	public bool IsHugeShip { get; }

	public MatchUnitValidationRules(int maximumSpeed, float footprintWidth, float footprintLength, bool isHugeShip = false)
	{
		MaximumSpeed = maximumSpeed;
		FootprintWidth = footprintWidth;
		FootprintLength = footprintLength;
		IsHugeShip = isHugeShip;
	}
}
