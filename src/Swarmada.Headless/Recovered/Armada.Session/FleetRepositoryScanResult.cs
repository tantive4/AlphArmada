#nullable disable
using System.Collections.Generic;

namespace Armada.Session;

public sealed class FleetRepositoryScanResult
{
	private readonly List<FleetListDto> fleets = new List<FleetListDto>();

	private readonly List<string> errors = new List<string>();

	public IReadOnlyList<FleetListDto> Fleets => fleets;

	public IReadOnlyList<string> Errors => errors;

	internal void AddFleet(FleetListDto fleet)
	{
		fleets.Add(fleet);
	}

	internal void AddError(string error)
	{
		errors.Add(error);
	}
}
