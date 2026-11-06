#nullable disable
namespace Armada.Rules;

public readonly struct MovementSolution
{
	public float ForwardDistance { get; }

	public float YawDegrees { get; }

	public ManeuverCourse Course { get; }

	public bool HasArticulatedCourse => Course != null;

	public MovementSolution(float forwardDistance, float yawDegrees)
	{
		ForwardDistance = forwardDistance;
		YawDegrees = yawDegrees;
		Course = null;
	}

	public MovementSolution(ManeuverCourse course)
	{
		Course = course ?? ManeuverCourse.Standstill;
		ForwardDistance = (float)Course.Speed * 2.9196103f;
		YawDegrees = (float)Course.TotalClicks * 22.5f;
	}
}
