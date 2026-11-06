#nullable disable
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Armada.Data;

namespace Armada.Fleet;

public static class OfficialContentPolicy
{
	public const int ExpectedRawUpgradeCount = 292;

	public const int ExpectedStrictUpgradeCount = 292;

	public const int ExpectedPendingUpgradeCount = 0;

	private static readonly HashSet<string> PendingUpgradeIds = new(StringComparer.OrdinalIgnoreCase);

	public static IReadOnlyCollection<string> ProvenancePendingUpgradeIds { get; } = new ReadOnlyCollection<string>(new List<string>(PendingUpgradeIds));

	public static OfficialContentDisposition ClassifyUpgrade(string id, string publisher)
	{
		if ((id ?? string.Empty).EndsWith("-fighter-group", StringComparison.OrdinalIgnoreCase))
		{
			return OfficialContentDisposition.Excluded;
		}
		if (string.Equals(publisher, "Fantasy Flight Games", StringComparison.OrdinalIgnoreCase) || string.Equals(publisher, "Atomic Mass Games", StringComparison.OrdinalIgnoreCase))
		{
			return OfficialContentDisposition.StrictOfficial;
		}
		return OfficialContentDisposition.Excluded;
	}

	public static bool IsStrictOfficial(UpgradeContentDescriptor upgrade)
	{
		if (upgrade != null)
		{
			return upgrade.OfficialDisposition == OfficialContentDisposition.StrictOfficial;
		}
		return false;
	}

	public static bool IsGameplayImplemented(string upgradeId, string publisher) =>
        !upgradeId.EndsWith("-fighter-group", StringComparison.OrdinalIgnoreCase) &&
        ClassifyUpgrade(upgradeId, publisher) == OfficialContentDisposition.StrictOfficial;

	public static UpgradeContentDescriptor Describe(UpgradeCardDefinition card)
	{
		if (card == null)
		{
			throw new ArgumentNullException("card");
		}
		return new UpgradeContentDescriptor(card.id, card.name, card.slot, card.factions, card.points, card.unique, card.uniqueClasses, card.modification, card.boundShipType, card.requiredTraits, card.requiredSizes, card.disallowedSlots, card.disabledSlots, card.enabledSlots, card.flagshipOnly, card.publisher, card.sourceTeam, card.sourceAlias, ClassifyUpgrade(card.id, card.publisher), (!IsGameplayImplemented(card.id, card.publisher)) ? GameplayCoverage.DisplayOnly : GameplayCoverage.Implemented);
	}

	public static IReadOnlyList<UpgradeContentDescriptor> DescribeAll(IEnumerable<UpgradeCardDefinition> cards)
	{
		List<UpgradeContentDescriptor> list = new List<UpgradeContentDescriptor>();
		if (cards != null)
		{
			foreach (UpgradeCardDefinition card in cards)
			{
				if (card != null)
				{
					list.Add(Describe(card));
				}
			}
		}
		return new ReadOnlyCollection<UpgradeContentDescriptor>(list);
	}

	public static OfficialUpgradePolicyAudit Audit(IEnumerable<UpgradeContentDescriptor> upgrades)
	{
		int num = 0;
		int num2 = 0;
		int num3 = 0;
		int num4 = 0;
		HashSet<string> hashSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
		if (upgrades != null)
		{
			foreach (UpgradeContentDescriptor upgrade in upgrades)
			{
				if (upgrade != null)
				{
					num++;
					switch (upgrade.OfficialDisposition)
					{
					case OfficialContentDisposition.StrictOfficial:
						num2++;
						break;
					case OfficialContentDisposition.ProvenancePending:
						num3++;
						hashSet.Add(upgrade.Id);
						break;
					default:
						num4++;
						break;
					}
				}
			}
		}
		bool flag = hashSet.SetEquals(PendingUpgradeIds);
		return new OfficialUpgradePolicyAudit(num, num2, num3, num4, num == 292 && num2 == 292 && num3 == 0 && num4 == 0 && flag);
	}
}
