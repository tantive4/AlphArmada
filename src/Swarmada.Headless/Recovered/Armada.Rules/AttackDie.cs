#nullable disable
namespace Armada.Rules;

public readonly struct AttackDie
{
	public AttackDieColor Color { get; }

	public DieFace Face { get; }

	public RangeBand Band => Color switch
	{
		AttackDieColor.Red => RangeBand.Long,
		AttackDieColor.Blue => RangeBand.Medium,
		_ => RangeBand.Close,
	};

	[System.Text.Json.Serialization.JsonConstructor]
	public AttackDie(AttackDieColor color, DieFace face)
	{
		Color = color;
		Face = face;
	}
}
