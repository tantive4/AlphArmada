#nullable disable
using System.Collections.Generic;

namespace Armada.Rules;

public static class UpgradeCriticalRules
{
	public const string HeavyIonEmplacements = "heavy-ion-emplacements";

	public const string IonCannonBatteries = "ion-cannon-batteries";

	public const string Ms1IonCannons = "ms-1-ion-cannons";

	public const string Nk7IonCannons = "nk-7-ion-cannons";

	public const string OverloadPulse = "overload-pulse";

	public const string AssaultConcussionMissiles = "assault-concussion-missiles";

	public const string AssaultProtonTorpedoes = "assault-proton-torpedoes";

	public const string Xx9Turbolasers = "xx-9-turbolasers";

	public const string FireControlTeam = "fire-control-team";

	public const string DodonnasPride = "dodonnas-pride-title";

	public const string Sunder = "sunder-title";

	public const string WideAreaBarrage = "wide-area-barrage";

	public const string OrbitalBombardmentParticleCannons = "orbital-bombardment-particle-cannons";

	public const string SuperheavyCompositeBeamTurbolasers = "superheavy-composite-beam-turbolasers";

	public const string KyrstaAgateOfficer = "kyrsta-agate-officer";

	public static bool NeedsBlueCritical(string cardId)
	{
		switch (cardId)
		{
		default:
			return cardId == "dodonnas-pride-title";
		case "heavy-ion-emplacements":
		case "ion-cannon-batteries":
		case "ms-1-ion-cannons":
		case "nk-7-ion-cannons":
		case "overload-pulse":
			return true;
		}
	}

	public static bool NeedsBlackCritical(string cardId)
	{
		if (!(cardId == "assault-concussion-missiles") && !(cardId == "assault-proton-torpedoes"))
		{
			return cardId == "wide-area-barrage";
		}
		return true;
	}

	public static bool NeedsBlueOrBlackCritical(string cardId)
	{
		return cardId == "sunder-title";
	}

	public static bool NeedsRedOrBlueCritical(string cardId)
	{
		return cardId == "kyrsta-agate-officer";
	}

	public static bool IsIgnitionCritical(string cardId)
	{
		if (!(cardId == "orbital-bombardment-particle-cannons"))
		{
			return cardId == "superheavy-composite-beam-turbolasers";
		}
		return true;
	}

	public static bool HasRequiredIgnitionCritical(string cardId, AttackResolution attack, bool isIgnitionAttack)
	{
		if (IsIgnitionCritical(cardId))
		{
			if (isIgnitionAttack)
			{
				return AttackResolver.HasCriticalIcon(attack, AttackDieColor.Red);
			}
			return false;
		}
		return true;
	}

	public static int CriticalEffectLimit(bool fireControlTeamActive, bool fireControlTeamExhausted)
	{
		if (!fireControlTeamActive || fireControlTeamExhausted)
		{
			return 1;
		}
		return 2;
	}

	public static bool SunderCanDiscardSlot(string slot)
	{
		switch (slot)
		{
		default:
			return slot == "turbolaser";
		case "offensive-retro":
		case "defensive-retro":
		case "ordnance":
		case "ion-cannon":
			return true;
		}
	}

	public static int CountCriticalIcons(AttackResolution attack, params AttackDieColor[] colors)
	{
		HashSet<AttackDieColor> hashSet = new HashSet<AttackDieColor>(colors);
		int num = 0;
		foreach (AttackDie die in attack.Dice)
		{
			if (hashSet.Contains(die.Color) && (die.Face == DieFace.Critical || die.Face == DieFace.HitCritical))
			{
				num++;
			}
		}
		return num;
	}
}
