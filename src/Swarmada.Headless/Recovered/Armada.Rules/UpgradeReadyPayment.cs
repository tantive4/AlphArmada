#nullable disable
namespace Armada.Rules;

public readonly struct UpgradeReadyPayment
{
	public ShipCommand Token { get; }

	public bool FromCard { get; }

	public int Amount { get; }

	public UpgradeReadyPayment(ShipCommand token, bool fromCard, int amount)
	{
		Token = token;
		FromCard = fromCard;
		Amount = amount;
	}
}
