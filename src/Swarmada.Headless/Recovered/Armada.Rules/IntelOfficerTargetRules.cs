#nullable disable
using System;
using System.Text;
using Armada.Core;

namespace Armada.Rules;

public static class IntelOfficerTargetRules
{
	private static readonly UTF8Encoding StrictUtf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

	public static bool CanChooseTarget(IntelOfficerTimingPhase phase, bool diceModified)
	{
		if (phase == IntelOfficerTimingPhase.AfterInitialRoll)
		{
			return !diceModified;
		}
		return false;
	}

	public static bool CanTargetUnit(bool isInPlay, bool isDestroyed, bool isReserved)
	{
		if (isInPlay && !isDestroyed)
		{
			return !isReserved;
		}
		return false;
	}

	public static AttackRequest BuildInitialRollRequest(AttackRequest request)
	{
		return new AttackRequest(request.RedDice, request.BlueDice, request.BlackDice, request.Distance, request.Seed, 0, 0, request.CriticalDamagePerIcon, request.BlackDiceAtMedium, request.BlueDiceAtLong, request.DiceAlreadyGathered);
	}

	public static string BuildChoiceId(string prefix, IntelOfficerTokenTarget target)
	{
		return (prefix ?? string.Empty) + "target." + (int)target.UnitKind + "." + EncodeRuntimeId(target.UnitId.Value) + "." + (int)target.Token.Token + "." + target.Token.Slot + "." + (target.Token.Exhausted ? "1" : "0");
	}

	public static bool TryParseChoiceId(string choiceId, string prefix, out IntelOfficerTokenTarget target)
	{
		target = default(IntelOfficerTokenTarget);
		if (string.IsNullOrWhiteSpace(choiceId) || string.IsNullOrWhiteSpace(prefix) || !choiceId.StartsWith(prefix, StringComparison.Ordinal))
		{
			return false;
		}
		string[] array = choiceId.Substring(prefix.Length).Split('.');
		if (array.Length != 6 || array[0] != "target" || !int.TryParse(array[1], out var result) || !Enum.IsDefined(typeof(RuntimeUnitKind), result) || !TryDecodeRuntimeId(array[2], out var runtimeId) || !int.TryParse(array[3], out var result2) || !Enum.IsDefined(typeof(DefenseTokenType), result2) || !int.TryParse(array[4], out var result3) || result3 < 0 || (array[5] != "0" && array[5] != "1"))
		{
			return false;
		}
		RuntimeUnitKind runtimeUnitKind = (RuntimeUnitKind)result;
		if (runtimeUnitKind != RuntimeUnitKind.Ship && runtimeUnitKind != RuntimeUnitKind.Squadron)
		{
			return false;
		}
		target = new IntelOfficerTokenTarget(runtimeUnitKind, runtimeId, new DefenseTokenInstance((DefenseTokenType)result2, result3, array[5] == "1"));
		return true;
	}

	public static bool MatchesSpend(IntelOfficerTokenTarget chosen, RuntimeUnitKind unitKind, RuntimeUnitId unitId, DefenseTokenInstance spent)
	{
		if (chosen.UnitKind == unitKind && chosen.UnitId == unitId && chosen.Token.Token == spent.Token)
		{
			return chosen.Token.Slot == spent.Slot;
		}
		return false;
	}

	public static bool IsAvailable(IntelOfficerTokenTarget target, RuntimeUnitId unitId, ShipRuntimeState state)
	{
		if (target.UnitKind == RuntimeUnitKind.Ship && target.UnitId == unitId)
		{
			return ForcedDefenseTokenDiscardRules.IsAvailable(state, target.Token);
		}
		return false;
	}

	public static bool IsAvailable(IntelOfficerTokenTarget target, RuntimeUnitId unitId, SquadronRuntimeState state)
	{
		if (target.UnitKind == RuntimeUnitKind.Squadron && target.UnitId == unitId)
		{
			return ForcedDefenseTokenDiscardRules.IsAvailable(state, target.Token);
		}
		return false;
	}

	public static bool TryDiscardAfterSpend(ShipRuntimeState state, DefenseTokenInstance spent)
	{
		if (state == null)
		{
			return false;
		}
		return TryDiscardAfterSpend(state.IsDefenseTokenInstanceAvailable, (DefenseTokenInstance token) => ForcedDefenseTokenDiscardRules.TryDiscard(state, token), spent);
	}

	public static bool TryDiscardAfterSpend(SquadronRuntimeState state, DefenseTokenInstance spent)
	{
		if (state == null)
		{
			return false;
		}
		return TryDiscardAfterSpend(state.IsDefenseTokenInstanceAvailable, (DefenseTokenInstance token) => ForcedDefenseTokenDiscardRules.TryDiscard(state, token), spent);
	}

	private static bool TryDiscardAfterSpend(Func<DefenseTokenInstance, bool> isAvailable, Func<DefenseTokenInstance, bool> discard, DefenseTokenInstance spent)
	{
		if (isAvailable == null || discard == null)
		{
			return false;
		}
		if (spent.Exhausted)
		{
			return !isAvailable(spent);
		}
		DefenseTokenInstance arg = new DefenseTokenInstance(spent.Token, spent.Slot, exhausted: true);
		if (!discard(arg))
		{
			if (!isAvailable(spent))
			{
				return !isAvailable(arg);
			}
			return false;
		}
		return true;
	}

	private static string EncodeRuntimeId(string runtimeId)
	{
		return Convert.ToBase64String(StrictUtf8.GetBytes(runtimeId ?? string.Empty)).TrimEnd('=').Replace('+', '-')
			.Replace('/', '_');
	}

	private static bool TryDecodeRuntimeId(string encoded, out RuntimeUnitId runtimeId)
	{
		runtimeId = default(RuntimeUnitId);
		if (string.IsNullOrWhiteSpace(encoded))
		{
			return false;
		}
		string text = encoded.Replace('-', '+').Replace('_', '/');
		switch (text.Length % 4)
		{
		case 2:
			text += "==";
			break;
		case 3:
			text += "=";
			break;
		default:
			return false;
		case 0:
			break;
		}
		try
		{
			string text2 = StrictUtf8.GetString(Convert.FromBase64String(text));
			return EncodeRuntimeId(text2) == encoded && RuntimeUnitId.TryParse(text2, out runtimeId);
		}
		catch (FormatException)
		{
			return false;
		}
		catch (DecoderFallbackException)
		{
			return false;
		}
	}
}
