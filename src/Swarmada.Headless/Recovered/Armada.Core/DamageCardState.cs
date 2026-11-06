#nullable disable
using System;

namespace Armada.Core;

[Serializable]
public readonly struct DamageCardState
{
	public bool FaceUp { get; }

	public CriticalDamageCard? CriticalCard { get; }

	public DamageCardState(bool faceUp, CriticalDamageCard? criticalCard)
	{
		FaceUp = faceUp;
		CriticalCard = criticalCard;
	}
}
