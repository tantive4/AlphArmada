#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;

namespace Armada.Rules;

public static class UpgradeReadyRules
{
	public static bool RequiresPayment(UpgradeCardRuntimeState card)
	{
		return string.Equals((card?.Definition?.exhaustType ?? string.Empty).Replace("-", string.Empty).Trim(), "nonrecur", StringComparison.OrdinalIgnoreCase);
	}

	public static IReadOnlyList<UpgradeReadyPayment> GetPayments(UpgradeCardRuntimeState card, CommandStackState commands)
	{
		List<UpgradeReadyPayment> list = new List<UpgradeReadyPayment>();
		if (card == null || !card.Active || !card.Exhausted || !RequiresPayment(card) || card.Definition.readyAmount <= 0)
		{
			return list;
		}
		int readyAmount = card.Definition.readyAmount;
		string[] source = card.Definition.readyTokens ?? Array.Empty<string>();
		bool flag = (card.Definition.ability ?? string.Empty).IndexOf("tokens from this card to ready it", StringComparison.OrdinalIgnoreCase) >= 0;
		foreach (ShipCommand token in Enum.GetValues(typeof(ShipCommand)))
		{
			if (source.Any((string icon) => Matches(icon, token)))
			{
				if (commands != null && commands.TokenInstances.Count((ShipCommand value) => value == token) >= readyAmount)
				{
					list.Add(new UpgradeReadyPayment(token, fromCard: false, readyAmount));
				}
				if (flag && card.GetTypedCommandTokenCount(token) >= readyAmount)
				{
					list.Add(new UpgradeReadyPayment(token, fromCard: true, readyAmount));
				}
			}
		}
		return list;
	}

	public static bool TryReady(UpgradeCardRuntimeState card, CommandStackState commands, UpgradeReadyPayment payment)
	{
		if (!GetPayments(card, commands).Any((UpgradeReadyPayment option) => option.Token == payment.Token && option.FromCard == payment.FromCard && option.Amount == payment.Amount))
		{
			return false;
		}
		for (int num = 0; num < payment.Amount; num++)
		{
			if (payment.FromCard)
			{
				card.TrySpendTypedCommandToken(payment.Token);
			}
			else
			{
				commands.TrySpendToken(payment.Token);
			}
		}
		card.Ready();
		return true;
	}

	private static bool Matches(string icon, ShipCommand token)
	{
		switch ((icon ?? string.Empty).Trim().ToLowerInvariant())
		{
		case "any":
			return true;
		case "nav":
		case "navigate":
			return token == ShipCommand.Navigate;
		case "repair":
		case "engineering":
			return token == ShipCommand.Engineering;
		case "squadron":
		case "squad":
			return token == ShipCommand.Squadron;
		case "confire":
		case "concentrate-fire":
			return token == ShipCommand.ConcentrateFire;
		default:
			return false;
		}
	}
}
