#nullable disable
using System;
using System.Collections.Generic;
using Armada.Rules;

namespace Armada.Core;

[Serializable]
public sealed partial class ShipRuntimeState
{
	private struct DefenseTokenSlotState
	{
		public bool Exhausted;

		public bool Discarded;
	}

	private Dictionary<HullZone, int> shields;

	private Dictionary<HullZone, int> maximumShields;

	private Dictionary<DefenseTokenType, int> defenseTokenCounts = new Dictionary<DefenseTokenType, int>();

	private Dictionary<DefenseTokenType, int> spentDefenseTokens = new Dictionary<DefenseTokenType, int>();

	private Dictionary<DefenseTokenType, int> discardedDefenseTokens = new Dictionary<DefenseTokenType, int>();

	private Dictionary<DefenseTokenType, List<DefenseTokenSlotState>> defenseTokenSlots = new Dictionary<DefenseTokenType, List<DefenseTokenSlotState>>();

	private List<DamageCardState> damageCards = new List<DamageCardState>();

	public bool IsInitialized
	{
		get
		{
			if (shields != null)
			{
				return maximumShields != null;
			}
			return false;
		}
	}

	public string ShipId { get; }

	public int MaximumHull { get; private set; }

	public int CurrentHull { get; private set; }

	public int CurrentSpeed { get; private set; }

	public bool Destroyed => CurrentHull <= 0;

	public IReadOnlyList<DamageCardState> DamageCards
	{
		get
		{
			EnsureTokenMaps();
			return damageCards;
		}
	}

	public int FaceupDamageCardCount
	{
		get
		{
			EnsureTokenMaps();
			int count = 0;
			for (int i = 0; i < damageCards.Count; i++) if (damageCards[i].FaceUp) count++;
			return count;
		}
	}

	public int FacedownDamageCardCount
	{
		get
		{
			EnsureTokenMaps();
			int count = 0;
			for (int i = 0; i < damageCards.Count; i++) if (!damageCards[i].FaceUp) count++;
			return count;
		}
	}

	public int FrontShields => GetShields(HullZone.Front);

	public int LeftShields => GetShields(HullZone.Left);

	public int RightShields => GetShields(HullZone.Right);

	public int RearShields => GetShields(HullZone.Rear);

	public bool HasAuxiliaryHullZones => maximumShields?.ContainsKey(HullZone.LeftAuxiliary) ?? false;

	public IReadOnlyList<HullZone> HullZones => HullZoneRules.Zones(HasAuxiliaryHullZones);

	public bool CanSpendAccuracyIcons => !HasFaceupCritical(CriticalDamageCard.BlindedGunners);

	public bool CanResolveCriticalEffects => !HasFaceupCritical(CriticalDamageCard.TargeterDisruption);

	public bool CanPerformSecondAttackAgainstShip => !HasFaceupCritical(CriticalDamageCard.CoolantDischarge);

	public bool MustResolveCrewPanicBeforeCommand => HasFaceupCritical(CriticalDamageCard.CrewPanic);

	public bool DealsExtraDamageWhenOverlapping => HasFaceupCritical(CriticalDamageCard.DamagedControls);

	public bool MustRemoveDieWhenAttackingShip => HasFaceupCritical(CriticalDamageCard.DamagedMunitions);

	public bool CanChooseObstructedTarget => !HasFaceupCritical(CriticalDamageCard.DisengagedFireControl);

	public bool CanSpendExhaustedDefenseTokens => !HasFaceupCritical(CriticalDamageCard.FaultyCountermeasures);

	public bool CanHaveCommandTokens => !HasFaceupCritical(CriticalDamageCard.LifeSupportFailure);

	public bool MustRemoveDieWhenAttackingSquadron => HasFaceupCritical(CriticalDamageCard.PointDefenseFailure);

	public int FinalYawPenalty => HasFaceupCritical(CriticalDamageCard.ThrustControlMalfunction) ? 1 : 0;

	public int DamageAfterManeuver
	{
		get
		{
			if (!HasFaceupCritical(CriticalDamageCard.RupturedEngine) || CurrentSpeed <= 1)
			{
				return 0;
			}
			return 1;
		}
	}

	private void EnsureShieldMaps()
	{
		if (shields == null)
		{
			shields = new Dictionary<HullZone, int>();
		}
		if (maximumShields == null)
		{
			maximumShields = new Dictionary<HullZone, int>();
		}
	}

	private void EnsureTokenMaps()
	{
		if (defenseTokenCounts == null)
		{
			defenseTokenCounts = new Dictionary<DefenseTokenType, int>();
		}
		if (spentDefenseTokens == null)
		{
			spentDefenseTokens = new Dictionary<DefenseTokenType, int>();
		}
		if (discardedDefenseTokens == null)
		{
			discardedDefenseTokens = new Dictionary<DefenseTokenType, int>();
		}
		if (defenseTokenSlots == null)
		{
			defenseTokenSlots = new Dictionary<DefenseTokenType, List<DefenseTokenSlotState>>();
		}
		if (damageCards == null)
		{
			damageCards = new List<DamageCardState>();
		}
		if (defenseTokenSlots.Count == 0 && defenseTokenCounts.Count > 0)
		{
			RebuildDefenseTokenSlotsFromCounts();
		}
	}

	private void RebuildDefenseTokenSlotsFromCounts()
	{
		defenseTokenSlots.Clear();
		foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
		{
			int num = Math.Max(0, GetCount(defenseTokenCounts, value));
			int num2 = Math.Clamp(GetCount(discardedDefenseTokens, value), 0, num);
			int num3 = num - num2;
			int num4 = Math.Clamp(GetCount(spentDefenseTokens, value), 0, num3);
			List<DefenseTokenSlotState> list = new List<DefenseTokenSlotState>(num);
			for (int i = 0; i < num; i++)
			{
				list.Add(new DefenseTokenSlotState
				{
					Exhausted = (i < num4),
					Discarded = (i >= num3)
				});
			}
			defenseTokenSlots[value] = list;
		}
	}

	private List<DefenseTokenSlotState> GetDefenseTokenSlots(DefenseTokenType token)
	{
		EnsureTokenMaps();
		if (!defenseTokenSlots.TryGetValue(token, out var value))
		{
			value = new List<DefenseTokenSlotState>();
			defenseTokenSlots[token] = value;
		}
		return value;
	}

	private void SyncDefenseTokenCounts(DefenseTokenType token)
	{
		List<DefenseTokenSlotState> list = GetDefenseTokenSlots(token);
		defenseTokenCounts[token] = list.Count;
		int num = 0;
		int num2 = 0;
		foreach (DefenseTokenSlotState item in list)
		{
			if (item.Discarded)
			{
				num2++;
			}
			else if (item.Exhausted)
			{
				num++;
			}
		}
		spentDefenseTokens[token] = num;
		discardedDefenseTokens[token] = num2;
	}

	public void IncreaseMaximumHull(int amount)
	{
		if (amount > 0)
		{
			MaximumHull += amount;
			CurrentHull += amount;
		}
	}

	public ShipRuntimeState(string shipId, int hull, int frontShields, int leftShields, int rightShields, int rearShields, int startingSpeed, int braceTokens = 1, int redirectTokens = 1, int evadeTokens = 1, int containTokens = 0, int salvoTokens = 0, int scatterTokens = 0, int leftAuxiliaryShields = -1, int rightAuxiliaryShields = -1)
	{
		ShipId = shipId;
		MaximumHull = Math.Max(1, hull);
		CurrentHull = MaximumHull;
		CurrentSpeed = Math.Max(0, startingSpeed);
		maximumShields = new Dictionary<HullZone, int>
		{
			{
				HullZone.Front,
				Math.Max(0, frontShields)
			},
			{
				HullZone.Left,
				Math.Max(0, leftShields)
			},
			{
				HullZone.Right,
				Math.Max(0, rightShields)
			},
			{
				HullZone.Rear,
				Math.Max(0, rearShields)
			}
		};
		if (leftAuxiliaryShields >= 0 && rightAuxiliaryShields >= 0)
		{
			maximumShields[HullZone.LeftAuxiliary] = leftAuxiliaryShields;
			maximumShields[HullZone.RightAuxiliary] = rightAuxiliaryShields;
		}
		shields = new Dictionary<HullZone, int>(maximumShields);
		defenseTokenCounts[DefenseTokenType.Brace] = Math.Max(0, braceTokens);
		defenseTokenCounts[DefenseTokenType.Redirect] = Math.Max(0, redirectTokens);
		defenseTokenCounts[DefenseTokenType.Evade] = Math.Max(0, evadeTokens);
		defenseTokenCounts[DefenseTokenType.Contain] = Math.Max(0, containTokens);
		defenseTokenCounts[DefenseTokenType.Salvo] = Math.Max(0, salvoTokens);
		defenseTokenCounts[DefenseTokenType.Scatter] = Math.Max(0, scatterTokens);
		RebuildDefenseTokenSlotsFromCounts();
	}

	public void SetSpeed(int speed, int maximumSpeed)
	{
		CurrentSpeed = Math.Clamp(speed, 0, maximumSpeed);
	}

	public int GetShields(HullZone zone)
	{
		EnsureShieldMaps();
		if (!shields.TryGetValue(zone, out var value))
		{
			return 0;
		}
		return value;
	}

	public int GetMaximumShields(HullZone zone)
	{
		EnsureShieldMaps();
		if (!maximumShields.TryGetValue(zone, out var value))
		{
			return 0;
		}
		return value;
	}

	public bool IsDefenseTokenReady(DefenseTokenType token)
	{
		return GetReadyDefenseTokenCount(token) > 0;
	}

	public bool HasDefenseToken(DefenseTokenType token)
	{
		return GetAvailableDefenseTokenCount(token) > 0;
	}

	public bool SpendDefenseToken(DefenseTokenType token)
	{
		foreach (DefenseTokenInstance defenseTokenInstance in GetDefenseTokenInstances(token))
		{
			if (!defenseTokenInstance.Exhausted)
			{
				return SpendDefenseToken(defenseTokenInstance);
			}
		}
		return false;
	}

	public bool SpendDefenseToken(DefenseTokenInstance token)
	{
		if (token.Exhausted || !IsDefenseTokenInstanceAvailable(token))
		{
			return false;
		}
		List<DefenseTokenSlotState> list = GetDefenseTokenSlots(token.Token);
		DefenseTokenSlotState value = list[token.Slot];
		value.Exhausted = true;
		list[token.Slot] = value;
		SyncDefenseTokenCounts(token.Token);
		return true;
	}

	public bool CanResolveDefenseToken(DefenseTokenType token)
	{
		if (!IsDefenseTokenReady(token))
		{
			if (CanSpendExhaustedDefenseTokens)
			{
				return HasDefenseToken(token);
			}
			return false;
		}
		return true;
	}

	public bool TryResolveDefenseToken(DefenseTokenType token, out bool discarded)
	{
		DefenseTokenInstance spentToken;
		return TryResolveDefenseToken(token, out discarded, out spentToken);
	}

	public bool TryResolveDefenseToken(DefenseTokenType token, out bool discarded, out DefenseTokenInstance spentToken)
	{
		discarded = false;
		spentToken = default(DefenseTokenInstance);
		foreach (DefenseTokenInstance defenseTokenInstance in GetDefenseTokenInstances(token))
		{
			if (!defenseTokenInstance.Exhausted)
			{
				spentToken = defenseTokenInstance;
				return SpendDefenseToken(defenseTokenInstance);
			}
		}
		if (!CanSpendExhaustedDefenseTokens || !HasDefenseToken(token))
		{
			return false;
		}
		foreach (DefenseTokenInstance defenseTokenInstance2 in GetDefenseTokenInstances(token))
		{
			if (defenseTokenInstance2.Exhausted)
			{
				spentToken = defenseTokenInstance2;
				discarded = DiscardDefenseToken(defenseTokenInstance2);
				return discarded;
			}
		}
		return false;
	}

	public bool TryResolveDefenseToken(DefenseTokenInstance token, out bool discarded)
	{
		discarded = false;
		if (!IsDefenseTokenInstanceAvailable(token))
		{
			return false;
		}
		if (!token.Exhausted)
		{
			return SpendDefenseToken(token);
		}
		if (!CanSpendExhaustedDefenseTokens)
		{
			return false;
		}
		discarded = DiscardDefenseToken(token);
		return discarded;
	}

	public bool DiscardDefenseToken(DefenseTokenType token)
	{
		foreach (DefenseTokenInstance defenseTokenInstance in GetDefenseTokenInstances(token))
		{
			if (defenseTokenInstance.Exhausted)
			{
				return DiscardDefenseToken(defenseTokenInstance);
			}
		}
		foreach (DefenseTokenInstance defenseTokenInstance2 in GetDefenseTokenInstances(token))
		{
			if (!defenseTokenInstance2.Exhausted)
			{
				return DiscardDefenseToken(defenseTokenInstance2);
			}
		}
		return false;
	}

	public bool DiscardDefenseToken(DefenseTokenType token, bool exhausted)
	{
		foreach (DefenseTokenInstance defenseTokenInstance in GetDefenseTokenInstances(token))
		{
			if (defenseTokenInstance.Exhausted == exhausted)
			{
				return DiscardDefenseToken(defenseTokenInstance);
			}
		}
		return false;
	}

	public bool DiscardDefenseToken(DefenseTokenInstance token)
	{
		if (!IsDefenseTokenInstanceAvailable(token))
		{
			return false;
		}
		List<DefenseTokenSlotState> list = GetDefenseTokenSlots(token.Token);
		DefenseTokenSlotState value = list[token.Slot];
		value.Discarded = true;
		list[token.Slot] = value;
		SyncDefenseTokenCounts(token.Token);
		return true;
	}

	public bool IsDefenseTokenInstanceAvailable(DefenseTokenInstance token)
	{
		List<DefenseTokenSlotState> list = GetDefenseTokenSlots(token.Token);
		if (token.Slot < 0 || token.Slot >= list.Count)
		{
			return false;
		}
		DefenseTokenSlotState defenseTokenSlotState = list[token.Slot];
		if (!defenseTokenSlotState.Discarded)
		{
			return token.Exhausted == defenseTokenSlotState.Exhausted;
		}
		return false;
	}

	public IReadOnlyList<DefenseTokenInstance> GetDefenseTokenInstances(DefenseTokenType token)
	{
		List<DefenseTokenSlotState> list = GetDefenseTokenSlots(token);
		List<DefenseTokenInstance> list2 = new List<DefenseTokenInstance>();
		for (int i = 0; i < list.Count; i++)
		{
			if (!list[i].Discarded)
			{
				list2.Add(new DefenseTokenInstance(token, i, list[i].Exhausted));
			}
		}
		return list2;
	}

	public void AddDefenseToken(DefenseTokenType token, int count = 1)
	{
		if (count > 0)
		{
			List<DefenseTokenSlotState> list = GetDefenseTokenSlots(token);
			for (int i = 0; i < count; i++)
			{
				list.Add(default(DefenseTokenSlotState));
			}
			SyncDefenseTokenCounts(token);
		}
	}

	public bool TryReadyDefenseToken(DefenseTokenType token)
	{
		List<DefenseTokenSlotState> list = GetDefenseTokenSlots(token);
		for (int i = 0; i < list.Count; i++)
		{
			DefenseTokenSlotState value = list[i];
			if (!value.Discarded && value.Exhausted)
			{
				value.Exhausted = false;
				list[i] = value;
				SyncDefenseTokenCounts(token);
				return true;
			}
		}
		return false;
	}

	public bool TryReplaceDefenseToken(DefenseTokenType remove, DefenseTokenType add)
	{
		if (!HasDefenseToken(remove))
		{
			return false;
		}
		List<DefenseTokenSlotState> list = GetDefenseTokenSlots(remove);
		int num = list.FindLastIndex((DefenseTokenSlotState slot) => !slot.Discarded && !slot.Exhausted);
		if (num < 0)
		{
			num = list.FindLastIndex((DefenseTokenSlotState slot) => !slot.Discarded);
		}
		if (num < 0)
		{
			return false;
		}
		list.RemoveAt(num);
		SyncDefenseTokenCounts(remove);
		AddDefenseToken(add);
		return true;
	}

	public int GetDiscardedDefenseTokenCount(DefenseTokenType token)
	{
		int num = 0;
		foreach (DefenseTokenSlotState defenseTokenSlot in GetDefenseTokenSlots(token))
		{
			if (defenseTokenSlot.Discarded)
			{
				num++;
			}
		}
		return num;
	}

	public bool TryRecoverDiscardedDefenseToken(DefenseTokenType token)
	{
		List<DefenseTokenSlotState> list = GetDefenseTokenSlots(token);
		for (int i = 0; i < list.Count; i++)
		{
			DefenseTokenSlotState value = list[i];
			if (value.Discarded)
			{
				value.Discarded = false;
				value.Exhausted = false;
				list[i] = value;
				SyncDefenseTokenCounts(token);
				return true;
			}
		}
		return false;
	}

	public bool TryRecoverAnyDiscardedDefenseToken(out DefenseTokenType recovered)
	{
		foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
		{
			if (TryRecoverDiscardedDefenseToken(value))
			{
				recovered = value;
				return true;
			}
		}
		recovered = DefenseTokenType.Brace;
		return false;
	}

	public void RefreshDefenseTokens()
	{
		if (HasFaceupCritical(CriticalDamageCard.CompartmentFire))
		{
			return;
		}
		foreach (DefenseTokenType value2 in Enum.GetValues(typeof(DefenseTokenType)))
		{
			List<DefenseTokenSlotState> list = GetDefenseTokenSlots(value2);
			for (int i = 0; i < list.Count; i++)
			{
				DefenseTokenSlotState value = list[i];
				if (!value.Discarded)
				{
					value.Exhausted = false;
					list[i] = value;
				}
			}
			SyncDefenseTokenCounts(value2);
		}
	}

	public int GetPrintedDefenseTokenCount(DefenseTokenType token)
	{
		return GetDefenseTokenSlots(token).Count;
	}

	public int GetAvailableDefenseTokenCount(DefenseTokenType token)
	{
		int num = 0;
		foreach (DefenseTokenSlotState defenseTokenSlot in GetDefenseTokenSlots(token))
		{
			if (!defenseTokenSlot.Discarded)
			{
				num++;
			}
		}
		return num;
	}

	public int GetReadyDefenseTokenCount(DefenseTokenType token)
	{
		int num = 0;
		foreach (DefenseTokenSlotState defenseTokenSlot in GetDefenseTokenSlots(token))
		{
			if (!defenseTokenSlot.Discarded && !defenseTokenSlot.Exhausted)
			{
				num++;
			}
		}
		return num;
	}

	private static int GetCount(Dictionary<DefenseTokenType, int> values, DefenseTokenType token)
	{
		if (values == null)
		{
			return 0;
		}
		if (!values.TryGetValue(token, out var value))
		{
			return 0;
		}
		return value;
	}

	public bool CanAttackAtRange(RangeBand range)
	{
		if (range == RangeBand.Long)
		{
			return !HasFaceupCritical(CriticalDamageCard.DepoweredArmament);
		}
		return true;
	}

	public int DamageWhenChangingSpeed(int previousSpeed)
	{
		if (!HasFaceupCritical(CriticalDamageCard.ThrusterFissure) || previousSpeed == CurrentSpeed)
		{
			return 0;
		}
		return 1;
	}

	public int GetEngineeringValue(int printedValue)
	{
		int num = Math.Max(0, printedValue);
		if (!HasFaceupCritical(CriticalDamageCard.PowerFailure))
		{
			return num;
		}
		return num / 2;
	}

	public DamageReport ApplyDamage(HullZone zone, int incomingDamage, bool firstHullCardFaceUp = false, int criticalSeed = 0, bool ignoreShields = false, IReadOnlyList<CriticalDamageCard> dealtCards = null, int faceupDamageCardCount = 0)
	{
		EnsureTokenMaps();
		int num = Math.Max(0, incomingDamage);
		int num2 = ((!ignoreShields) ? Math.Min(GetShields(zone), num) : 0);
		EnsureShieldMaps();
		shields[zone] = GetShields(zone) - num2;
		num -= num2;
		int num3 = Math.Min(CurrentHull, num);
		CurrentHull -= num3;
		string faceupDamageCard = string.Empty;
		int additionalCriticalDamage = 0;
		CriticalDamageCard? criticalCard = null;
		for (int i = 0; i < num3; i++)
		{
			CriticalDamageCard criticalDamageCard = ((dealtCards != null && i < dealtCards.Count) ? dealtCards[i] : DrawCriticalCard(criticalSeed + i * 104729));
			bool flag = i < Math.Max(faceupDamageCardCount, firstHullCardFaceUp ? 1 : 0);
			damageCards.Add(new DamageCardState(flag, criticalDamageCard));
			if (flag)
			{
				criticalCard = criticalDamageCard;
				faceupDamageCard = CriticalEffectRegistry.Get(criticalDamageCard).DisplayName;
			}
		}
		return new DamageReport(zone, incomingDamage, num2, num3, CurrentHull == 0, faceupDamageCard, additionalCriticalDamage, criticalCard);
	}

	public int DealFacedownDamageCard(CriticalDamageCard? criticalCard = null)
	{
		EnsureTokenMaps();
		damageCards.Add(new DamageCardState(faceUp: false, criticalCard ?? DrawCriticalCard(damageCards.Count * 104729 + 17)));
		if (CurrentHull <= 0)
		{
			return 0;
		}
		CurrentHull--;
		return 1;
	}

	public int DealFaceupDamageCard(CriticalDamageCard criticalCard)
	{
		EnsureTokenMaps();
		damageCards.Add(new DamageCardState(faceUp: true, criticalCard));
		if (CurrentHull <= 0)
		{
			return 0;
		}
		CurrentHull--;
		return 1;
	}

	public int DealRandomFaceupDamageCard(int seed)
	{
		return DealFaceupDamageCard(DrawCriticalCard(seed));
	}

	public int LoseAllShields(HullZone zone)
	{
		int result = GetShields(zone);
		EnsureShieldMaps();
		shields[zone] = 0;
		return result;
	}

	public int LoseShields(HullZone zone, int amount)
	{
		int num = GetShields(zone);
		EnsureShieldMaps();
		shields[zone] = Math.Max(0, num - Math.Max(0, amount));
		return num - shields[zone];
	}

	public bool FlipCriticalFacedown(CriticalDamageCard criticalCard)
	{
		EnsureTokenMaps();
		for (int i = 0; i < damageCards.Count; i++)
		{
			DamageCardState damageCardState = damageCards[i];
			if (damageCardState.FaceUp && damageCardState.CriticalCard == criticalCard)
			{
				damageCards[i] = new DamageCardState(faceUp: false, criticalCard);
				return true;
			}
		}
		return false;
	}

	public int FlipCrewDamageCardsFacedown()
	{
		EnsureTokenMaps();
		int num = 0;
		for (int i = 0; i < damageCards.Count; i++)
		{
			DamageCardState damageCardState = damageCards[i];
			if (damageCardState.FaceUp && damageCardState.CriticalCard.HasValue && IsCrewDamageCard(damageCardState.CriticalCard.Value))
			{
				damageCards[i] = new DamageCardState(faceUp: false, damageCardState.CriticalCard);
				num++;
			}
		}
		return num;
	}

	public int FlipFacedownDamageCardsFaceup(int maximum, int seed)
	{
		EnsureTokenMaps();
		int num = 0;
		for (int i = 0; i < damageCards.Count; i++)
		{
			if (num >= Math.Max(0, maximum))
			{
				break;
			}
			DamageCardState damageCardState = damageCards[i];
			if (!damageCardState.FaceUp)
			{
				CriticalDamageCard value = damageCardState.CriticalCard ?? DrawCriticalCard(seed + i * 7919);
				damageCards[i] = new DamageCardState(faceUp: true, value);
				num++;
			}
		}
		return num;
	}

	public bool FlipFacedownDamageCardFaceupAt(int index, out CriticalDamageCard criticalCard)
	{
		EnsureTokenMaps();
		criticalCard = CriticalDamageCard.StructuralDamage;
		if (index < 0 || index >= damageCards.Count)
		{
			return false;
		}
		DamageCardState damageCardState = damageCards[index];
		if (damageCardState.FaceUp)
		{
			return false;
		}
		criticalCard = damageCardState.CriticalCard ?? DrawCriticalCard(index * 104729 + 31);
		damageCards[index] = new DamageCardState(faceUp: true, criticalCard);
		return true;
	}

	public static bool IsCrewDamageCard(CriticalDamageCard card)
	{
		if (card != CriticalDamageCard.BlindedGunners && card != CriticalDamageCard.CommNoise && card != CriticalDamageCard.CompartmentFire && card != CriticalDamageCard.CrewPanic && card != CriticalDamageCard.InjuredCrew)
		{
			return card == CriticalDamageCard.LifeSupportFailure;
		}
		return true;
	}

	public int RepairHull(int amount)
	{
		EnsureTokenMaps();
		int currentHull = CurrentHull;
		int num = Math.Min(Math.Max(0, amount), Math.Min(damageCards.Count, MaximumHull - CurrentHull));
		for (int i = 0; i < num; i++)
		{
			damageCards.RemoveAt(damageCards.Count - 1);
		}
		CurrentHull = Math.Min(MaximumHull, CurrentHull + num);
		return CurrentHull - currentHull;
	}

	public int RepairFacedownDamage(int amount)
	{
		EnsureTokenMaps();
		int num = Math.Max(0, amount);
		int num2 = 0;
		int num3 = damageCards.Count - 1;
		while (num3 >= 0 && num2 < num)
		{
			if (!damageCards[num3].FaceUp)
			{
				damageCards.RemoveAt(num3);
				num2++;
			}
			num3--;
		}
		CurrentHull = Math.Min(MaximumHull, CurrentHull + num2);
		return num2;
	}

	public bool TryRepairFacedownDamage(out CriticalDamageCard repairedCard)
	{
		EnsureTokenMaps();
		for (int num = damageCards.Count - 1; num >= 0; num--)
		{
			DamageCardState damageCardState = damageCards[num];
			if (!damageCardState.FaceUp)
			{
				repairedCard = damageCardState.CriticalCard ?? CriticalDamageCard.StructuralDamage;
				damageCards.RemoveAt(num);
				CurrentHull = Math.Min(MaximumHull, CurrentHull + 1);
				return true;
			}
		}
		repairedCard = CriticalDamageCard.StructuralDamage;
		return false;
	}

	public int RepairFaceupDamage(CriticalDamageCard card)
	{
		EnsureTokenMaps();
		for (int i = 0; i < damageCards.Count; i++)
		{
			DamageCardState damageCardState = damageCards[i];
			if (damageCardState.FaceUp && damageCardState.CriticalCard == card)
			{
				damageCards.RemoveAt(i);
				CurrentHull = Math.Min(MaximumHull, CurrentHull + 1);
				return 1;
			}
		}
		return 0;
	}

	public int RestoreShield(HullZone zone, int amount, bool increaseSideMaximum = false)
	{
		if (GetShields(zone) == 0 && HasFaceupCritical(CriticalDamageCard.CapacitorFailure))
		{
			return 0;
		}
		int num = GetShields(zone);
		int num2 = GetMaximumShields(zone);
		if (increaseSideMaximum && (zone == HullZone.Left || zone == HullZone.Right))
		{
			num2 = Math.Min(4, num2 + 1);
		}
		EnsureShieldMaps();
		shields[zone] = Math.Min(num2, num + Math.Max(0, amount));
		return shields[zone] - num;
	}

	public int TransferShield(HullZone source, HullZone destination, bool increaseSideMaximum = false, int maximumOverride = -1)
	{
		if (source == destination || GetShields(source) <= 0)
		{
			return 0;
		}
		if (GetShields(destination) == 0 && HasFaceupCritical(CriticalDamageCard.CapacitorFailure))
		{
			return 0;
		}
		int num = GetMaximumShields(destination);
		if (maximumOverride >= 0)
		{
			num = Math.Max(num, maximumOverride);
		}
		if (increaseSideMaximum && (destination == HullZone.Left || destination == HullZone.Right))
		{
			num = Math.Min(4, num + 1);
		}
		if (GetShields(destination) >= num)
		{
			return 0;
		}
		EnsureShieldMaps();
		shields[source] = GetShields(source) - 1;
		shields[destination] = GetShields(destination) + 1;
		return 1;
	}

	public HullZone FindMostDamagedShieldZone()
	{
		HullZone result = HullZone.Front;
		int num = -1;
		foreach (HullZone value in Enum.GetValues(typeof(HullZone)))
		{
			int num2 = GetMaximumShields(value) - GetShields(value);
			if (num2 > num)
			{
				result = value;
				num = num2;
			}
		}
		return result;
	}

	public HullZone FindBestRedirectZone(HullZone attackedZone)
	{
		HullZone hullZone = ((attackedZone == HullZone.Front || attackedZone == HullZone.Rear) ? HullZone.Left : HullZone.Front);
		HullZone hullZone2 = ((attackedZone == HullZone.Front || attackedZone == HullZone.Rear) ? HullZone.Right : HullZone.Rear);
		if (GetShields(hullZone) < GetShields(hullZone2))
		{
			return hullZone2;
		}
		return hullZone;
	}

	public bool HasFaceupCritical(CriticalDamageCard card)
	{
		EnsureTokenMaps();
		for (int i = 0; i < damageCards.Count; i++)
			if (damageCards[i].FaceUp && damageCards[i].CriticalCard == card) return true;
		return false;
	}

	public string GetCriticalSummary()
	{
		EnsureTokenMaps();
		List<string> list = new List<string>();
		foreach (DamageCardState damageCard in damageCards)
		{
			if (damageCard.FaceUp && damageCard.CriticalCard.HasValue)
			{
				list.Add(CriticalEffectRegistry.Get(damageCard.CriticalCard.Value).DisplayName);
			}
		}
		if (list.Count != 0)
		{
			return string.Join(", ", list);
		}
		return "NONE";
	}

	public IReadOnlyList<CriticalDamageCard> GetFaceupCriticalCards()
	{
		EnsureTokenMaps();
		List<CriticalDamageCard> list = new List<CriticalDamageCard>();
		foreach (DamageCardState damageCard in damageCards)
		{
			if (damageCard.FaceUp && damageCard.CriticalCard.HasValue)
			{
				list.Add(damageCard.CriticalCard.Value);
			}
		}
		return list;
	}

	private CriticalDamageCard DrawCriticalCard(int seed)
	{
		CriticalDamageCard[] array = (CriticalDamageCard[])Enum.GetValues(typeof(CriticalDamageCard));
		int num = (seed & 0x7FFFFFFF) % array.Length;
		return array[num];
	}
}
