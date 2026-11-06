#nullable disable
namespace Armada.Session;

public sealed class MapScenarioDefinition
{
	public string Id { get; }

	public string DisplayName { get; }

	public bool LocksLongApproach { get; }

	public ScenarioObjectiveShipRequirement ObjectiveShipRequirement { get; }

	public MapScenarioDefinition(string id, string displayName, bool locksLongApproach = false, ScenarioObjectiveShipRequirement objectiveShipRequirement = ScenarioObjectiveShipRequirement.None)
	{
		Id = id;
		DisplayName = displayName;
		LocksLongApproach = locksLongApproach;
		ObjectiveShipRequirement = objectiveShipRequirement;
	}
}
