#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Core;

public static class CriticalEffectRegistry
{
	private static readonly Dictionary<CriticalDamageCard, CriticalEffectDefinition> Definitions = new Dictionary<CriticalDamageCard, CriticalEffectDefinition>
	{
		{
			CriticalDamageCard.BlindedGunners,
			Persistent(CriticalDamageCard.BlindedGunners, "BLINDED GUNNERS", "While attacking, you cannot spend accuracy icons.")
		},
		{
			CriticalDamageCard.CapacitorFailure,
			Persistent(CriticalDamageCard.CapacitorFailure, "CAPACITOR FAILURE", "A hull zone with no shields cannot recover or receive shields. If it is defending, Redirect cannot be spent.")
		},
		{
			CriticalDamageCard.CommNoise,
			Immediate(CriticalDamageCard.CommNoise, "COMM NOISE", "Your opponent may reduce your speed by 1 or choose a new command on your top command dial. Then flip this card facedown.", CriticalDecisionType.OpponentSpeedOrCommand, CriticalDecisionOwner.Opponent)
		},
		{
			CriticalDamageCard.CompartmentFire,
			Persistent(CriticalDamageCard.CompartmentFire, "COMPARTMENT FIRE", "You cannot ready your defense tokens.")
		},
		{
			CriticalDamageCard.CoolantDischarge,
			Persistent(CriticalDamageCard.CoolantDischarge, "COOLANT DISCHARGE", "Only 1 attack you perform each round can target a ship.")
		},
		{
			CriticalDamageCard.CrewPanic,
			Triggered(CriticalDamageCard.CrewPanic, "CREW PANIC", "Before revealing a command dial, suffer 1 damage or discard that dial. A discarded dial is not revealed this round.")
		},
		{
			CriticalDamageCard.DamagedControls,
			Triggered(CriticalDamageCard.DamagedControls, "DAMAGED CONTROLS", "When you overlap a ship or obstacle, deal 1 additional facedown damage card to your ship.")
		},
		{
			CriticalDamageCard.DamagedMunitions,
			Triggered(CriticalDamageCard.DamagedMunitions, "DAMAGED MUNITIONS", "While attacking a ship, before rolling, remove 1 die of your choice.")
		},
		{
			CriticalDamageCard.DepoweredArmament,
			Persistent(CriticalDamageCard.DepoweredArmament, "DEPOWERED ARMAMENT", "You cannot attack at long range.")
		},
		{
			CriticalDamageCard.DisengagedFireControl,
			Persistent(CriticalDamageCard.DisengagedFireControl, "DISENGAGED FIRE CONTROL", "You cannot declare an obstructed target.")
		},
		{
			CriticalDamageCard.FaultyCountermeasures,
			Persistent(CriticalDamageCard.FaultyCountermeasures, "FAULTY COUNTERMEASURES", "You cannot spend exhausted defense tokens.")
		},
		{
			CriticalDamageCard.InjuredCrew,
			Immediate(CriticalDamageCard.InjuredCrew, "INJURED CREW", "Choose and discard 1 defense token. Then flip this card facedown.", CriticalDecisionType.ChooseDefenseToken)
		},
		{
			CriticalDamageCard.LifeSupportFailure,
			Persistent(CriticalDamageCard.LifeSupportFailure, "LIFE SUPPORT FAILURE", "Discard all command tokens. You cannot have command tokens.")
		},
		{
			CriticalDamageCard.PointDefenseFailure,
			Triggered(CriticalDamageCard.PointDefenseFailure, "POINT DEFENSE FAILURE", "While attacking a squadron, before rolling, remove 1 die of your choice.")
		},
		{
			CriticalDamageCard.PowerFailure,
			Persistent(CriticalDamageCard.PowerFailure, "POWER FAILURE", "Your engineering value is halved, rounded down.")
		},
		{
			CriticalDamageCard.ProjectorMisaligned,
			Immediate(CriticalDamageCard.ProjectorMisaligned, "PROJECTOR MISALIGNED", "The hull zone with the most remaining shields loses all shields. If tied, choose a tied zone. Then flip this card facedown.", CriticalDecisionType.ChooseHullZone)
		},
		{
			CriticalDamageCard.RupturedEngine,
			Triggered(CriticalDamageCard.RupturedEngine, "RUPTURED ENGINE", "After executing a maneuver at speed greater than 1, suffer 1 damage.")
		},
		{
			CriticalDamageCard.ShieldFailure,
			Immediate(CriticalDamageCard.ShieldFailure, "SHIELD FAILURE", "Your opponent may choose up to 2 hull zones. Each chosen zone loses 1 shield. Then flip this card facedown.", CriticalDecisionType.ChooseUpToTwoHullZones, CriticalDecisionOwner.Opponent)
		},
		{
			CriticalDamageCard.StructuralDamage,
			Immediate(CriticalDamageCard.StructuralDamage, "STRUCTURAL DAMAGE", "Deal 1 additional damage to your ship as 1 facedown damage card. Then flip this card facedown.", CriticalDecisionType.Acknowledge)
		},
		{
			CriticalDamageCard.TargeterDisruption,
			Persistent(CriticalDamageCard.TargeterDisruption, "TARGETER DISRUPTION", "While attacking, you cannot resolve critical effects.")
		},
		{
			CriticalDamageCard.ThrustControlMalfunction,
			Persistent(CriticalDamageCard.ThrustControlMalfunction, "THRUST CONTROL MALFUNCTION", "The yaw value of the last adjustable joint at your current speed is reduced by 1.")
		},
		{
			CriticalDamageCard.ThrusterFissure,
			Triggered(CriticalDamageCard.ThrusterFissure, "THRUSTER FISSURE", "When you change speed by 1 or more, suffer 1 damage.")
		}
	};

	public static CriticalEffectDefinition Get(CriticalDamageCard card)
	{
		if (Definitions.TryGetValue(card, out var value))
		{
			return value;
		}
		throw new ArgumentOutOfRangeException("card", card, null);
	}

	private static CriticalEffectDefinition Persistent(CriticalDamageCard card, string displayName, string rulesText)
	{
		return new CriticalEffectDefinition(card, displayName, rulesText, CriticalEffectTiming.Persistent);
	}

	private static CriticalEffectDefinition Triggered(CriticalDamageCard card, string displayName, string rulesText)
	{
		return new CriticalEffectDefinition(card, displayName, rulesText, CriticalEffectTiming.Triggered);
	}

	private static CriticalEffectDefinition Immediate(CriticalDamageCard card, string displayName, string rulesText, CriticalDecisionType decisionType, CriticalDecisionOwner decisionOwner = CriticalDecisionOwner.AffectedPlayer)
	{
		return new CriticalEffectDefinition(card, displayName, rulesText, CriticalEffectTiming.ImmediateAndFlip, decisionType, decisionOwner);
	}
}
