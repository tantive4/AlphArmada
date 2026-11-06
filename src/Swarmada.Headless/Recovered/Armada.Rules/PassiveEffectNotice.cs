#nullable disable
using Armada.Core;

namespace Armada.Rules;

public readonly struct PassiveEffectNotice
{
	public bool Applied { get; }

	public RuntimeUnitId? Unit { get; }

	public string LogLine { get; }

	public string Callout { get; }

	public PassiveNoticeAudience Audience { get; }

	public static PassiveEffectNotice None => default(PassiveEffectNotice);

	private PassiveEffectNotice(bool applied, RuntimeUnitId? unit, string logLine, string callout, PassiveNoticeAudience audience)
	{
		Applied = applied;
		Unit = unit;
		LogLine = logLine;
		Callout = callout;
		Audience = audience;
	}

	public static PassiveEffectNotice Battlefield(RuntimeUnitId unit, string logLine, string callout)
	{
		return new PassiveEffectNotice(applied: true, unit, logLine, callout, PassiveNoticeAudience.Battlefield);
	}

	public static PassiveEffectNotice LogOnly(string logLine)
	{
		return new PassiveEffectNotice(applied: true, null, logLine, null, PassiveNoticeAudience.LogOnly);
	}
}
