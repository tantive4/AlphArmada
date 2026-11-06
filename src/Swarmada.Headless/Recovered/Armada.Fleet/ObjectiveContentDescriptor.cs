#nullable disable
namespace Armada.Fleet;

public sealed class ObjectiveContentDescriptor
{
	public string Id { get; }

	public string DisplayName { get; }

	public ObjectiveCategory Category { get; }

	public OfficialContentDisposition OfficialDisposition { get; }

	public GameplayCoverage GameplayCoverage { get; }

	public ObjectiveContentDescriptor(string id, string displayName, ObjectiveCategory category, OfficialContentDisposition officialDisposition = OfficialContentDisposition.StrictOfficial, GameplayCoverage gameplayCoverage = GameplayCoverage.Unknown)
	{
		Id = FleetContentText.RequiredId(id, "id");
		DisplayName = FleetContentText.ValueOrId(displayName, Id);
		Category = category;
		OfficialDisposition = officialDisposition;
		GameplayCoverage = gameplayCoverage;
	}
}
