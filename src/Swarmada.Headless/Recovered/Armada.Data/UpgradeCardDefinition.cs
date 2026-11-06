#nullable disable
using System;

namespace Armada.Data;

[Serializable]
public sealed class UpgradeCardDefinition
{
	public string id;

	public string sourceId;

	public string name;

	public string slot;

	public string[] factions;

	public int points;

	public string ability;

	public bool unique;

	public string[] uniqueClasses;

	public bool modification;

	public string boundShipType;

	public string[] requiredTraits;

	public string[] requiredSizes;

	public string[] disallowedSlots;

	public string[] disabledSlots;

	public string[] enabledSlots;

	public bool flagshipOnly;

	public string startCommandType;

	public string[] startCommandIcons;

	public int startCommandAmount;

	public string exhaustType;

	public string[] readyTokens;

	public int readyAmount;

	public string release;

	public string expansion;

	public string publisher;

	public string sourceTeam;

	public string sourceAlias;

	public string cardImageUrl;

	public UpgradeClarification[] clarifications;
}
