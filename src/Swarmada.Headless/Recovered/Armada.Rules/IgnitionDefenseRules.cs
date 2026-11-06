#nullable disable
namespace Armada.Rules;

public static class IgnitionDefenseRules
{
	public static bool AllowsSalvo(bool isIgnitionAttack)
	{
		return !isIgnitionAttack;
	}

	public static int ExtremeEvadeCancellationCount(bool foresightActive)
	{
		if (!foresightActive)
		{
			return 2;
		}
		return 3;
	}
}
