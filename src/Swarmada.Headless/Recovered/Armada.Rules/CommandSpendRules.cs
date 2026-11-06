#nullable disable
using System;

namespace Armada.Rules;

public static class CommandSpendRules
{
	public static bool UsesDial(CommandSpendMode mode)
	{
		if (mode != CommandSpendMode.Dial)
		{
			return mode == CommandSpendMode.DialAndToken;
		}
		return true;
	}

	public static bool UsesToken(CommandSpendMode mode)
	{
		if (mode != CommandSpendMode.Token)
		{
			return mode == CommandSpendMode.DialAndToken;
		}
		return true;
	}

	public static int NavigateSpeedLimit(CommandSpendMode mode)
	{
		if (mode != CommandSpendMode.DialAndToken)
		{
			return 1;
		}
		return 2;
	}

	public static bool GrantsNavigateYaw(CommandSpendMode mode)
	{
		return UsesDial(mode);
	}

	public static int EngineeringPoints(int printedEngineering, CommandSpendMode mode)
	{
		int num = (UsesDial(mode) ? Math.Max(0, printedEngineering) : 0);
		int num2 = (UsesToken(mode) ? ((Math.Max(0, printedEngineering) + 1) / 2) : 0);
		return num + num2;
	}

	public static int SquadronActivations(int printedSquadron, CommandSpendMode mode)
	{
		int num = (UsesDial(mode) ? Math.Max(0, printedSquadron) : 0);
		int num2 = (UsesToken(mode) ? 1 : 0);
		return num + num2;
	}

	public static bool GrantsConcentrateFireDie(CommandSpendMode mode)
	{
		return UsesDial(mode);
	}

	public static bool GrantsConcentrateFireReroll(CommandSpendMode mode)
	{
		return UsesToken(mode);
	}
}
