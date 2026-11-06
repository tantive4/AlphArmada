#nullable disable
using System;

namespace Armada.Core;

public readonly struct DamageReport
{
	public HullZone Zone { get; }

	public int IncomingDamage { get; }

	public int ShieldDamage { get; }

	public int HullDamage { get; }

	public bool Destroyed { get; }

	public string FaceupDamageCard { get; }

	public int AdditionalCriticalDamage { get; }

	public CriticalDamageCard? CriticalCard { get; }

	public int TotalApplied => ShieldDamage + HullDamage + AdditionalCriticalDamage;

	public DamageReport(HullZone zone, int incomingDamage, int shieldDamage, int hullDamage, bool destroyed, string faceupDamageCard = "", int additionalCriticalDamage = 0, CriticalDamageCard? criticalCard = null)
	{
		Zone = zone;
		IncomingDamage = Math.Max(0, incomingDamage);
		ShieldDamage = Math.Max(0, shieldDamage);
		HullDamage = Math.Max(0, hullDamage);
		Destroyed = destroyed;
		FaceupDamageCard = faceupDamageCard ?? string.Empty;
		AdditionalCriticalDamage = Math.Max(0, additionalCriticalDamage);
		CriticalCard = criticalCard;
	}

	public static DamageReport Combine(HullZone primaryZone, int incomingDamage, DamageReport first, DamageReport second)
	{
		return new DamageReport(primaryZone, incomingDamage, first.ShieldDamage + second.ShieldDamage, first.HullDamage + second.HullDamage, first.Destroyed || second.Destroyed, (!string.IsNullOrEmpty(first.FaceupDamageCard)) ? first.FaceupDamageCard : second.FaceupDamageCard, first.AdditionalCriticalDamage + second.AdditionalCriticalDamage, first.CriticalCard ?? second.CriticalCard);
	}
}
