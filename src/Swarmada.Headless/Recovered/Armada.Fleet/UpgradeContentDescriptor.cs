#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Fleet;

public sealed class UpgradeContentDescriptor
{
	public string Id { get; }

	public string DisplayName { get; }

	public string Slot { get; }

	public IReadOnlyList<string> Factions { get; }

	public int Points { get; }

	public bool IsUnique { get; }

	public IReadOnlyList<string> UniqueClasses { get; }

	public bool IsModification { get; }

	public string BoundShipType { get; }

	public IReadOnlyList<string> RequiredTraits { get; }

	public IReadOnlyList<string> RequiredSizes { get; }

	public IReadOnlyList<string> DisallowedSlots { get; }

	public IReadOnlyList<string> DisabledSlots { get; }

	public IReadOnlyList<string> EnabledSlots { get; }

	public bool FlagshipOnly { get; }

	public string Publisher { get; }

	public string SourceTeam { get; }

	public string SourceAlias { get; }

	public OfficialContentDisposition OfficialDisposition { get; }

	public GameplayCoverage GameplayCoverage { get; }

	public UpgradeContentDescriptor(string id, string displayName, string slot, IEnumerable<string> factions, int points, bool isUnique = false, IEnumerable<string> uniqueClasses = null, bool isModification = false, string boundShipType = "", IEnumerable<string> requiredTraits = null, IEnumerable<string> requiredSizes = null, IEnumerable<string> disallowedSlots = null, IEnumerable<string> disabledSlots = null, IEnumerable<string> enabledSlots = null, bool flagshipOnly = false, string publisher = "", string sourceTeam = "", string sourceAlias = "", OfficialContentDisposition officialDisposition = OfficialContentDisposition.StrictOfficial, GameplayCoverage gameplayCoverage = GameplayCoverage.Unknown)
	{
		Id = FleetContentText.RequiredId(id, "id");
		DisplayName = FleetContentText.ValueOrId(displayName, Id);
		Slot = FleetContentText.Normalize(slot);
		Factions = FleetContentText.CopyValues(factions);
		Points = Math.Max(0, points);
		IsUnique = isUnique;
		UniqueClasses = FleetContentText.CopyValues(uniqueClasses);
		IsModification = isModification;
		BoundShipType = FleetContentText.Normalize(boundShipType);
		RequiredTraits = FleetContentText.CopyValues(requiredTraits);
		RequiredSizes = FleetContentText.CopyValues(requiredSizes);
		DisallowedSlots = FleetContentText.CopyValues(disallowedSlots);
		DisabledSlots = FleetContentText.CopyValues(disabledSlots);
		EnabledSlots = FleetContentText.CopyValues(enabledSlots);
		FlagshipOnly = flagshipOnly;
		Publisher = FleetContentText.Normalize(publisher);
		SourceTeam = FleetContentText.Normalize(sourceTeam);
		SourceAlias = FleetContentText.Normalize(sourceAlias);
		OfficialDisposition = officialDisposition;
		GameplayCoverage = gameplayCoverage;
	}
}
