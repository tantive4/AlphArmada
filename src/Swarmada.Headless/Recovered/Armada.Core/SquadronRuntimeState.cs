#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Core;

[Serializable]
public sealed partial class SquadronRuntimeState
{
	private struct DefenseTokenSlotState
	{
		public bool Exhausted;

		public bool Discarded;
	}

	private bool temporaryAttackOnly;

	private bool movementPreventedForShipActivation;

	private Dictionary<DefenseTokenType, int> defenseTokenCounts = new Dictionary<DefenseTokenType, int>();

	private Dictionary<DefenseTokenType, int> spentDefenseTokens = new Dictionary<DefenseTokenType, int>();

	private Dictionary<DefenseTokenType, int> discardedDefenseTokens = new Dictionary<DefenseTokenType, int>();

	private Dictionary<DefenseTokenType, List<DefenseTokenSlotState>> defenseTokenSlots = new Dictionary<DefenseTokenType, List<DefenseTokenSlotState>>();

	public int CurrentHull { get; private set; }

	public bool Activated { get; private set; }

	public bool Commanded { get; private set; }

	public string CommandingShipRuntimeId { get; private set; } = string.Empty;

	public bool TemporaryGrit { get; private set; }

	public int TemporarySnipeValue { get; private set; }

	public bool TemporaryAssault { get; private set; }

	public bool TemporarySwarm { get; private set; }

	public int TemporarySpeedOverride { get; private set; } = -1;

	public bool TemporaryCannotAttack { get; private set; }

	public bool TemporaryAttackOnly
	{
		get
		{
			if (!temporaryAttackOnly)
			{
				return movementPreventedForShipActivation;
			}
			return true;
		}
		private set
		{
			temporaryAttackOnly = value;
		}
	}

	public bool TemporaryYavarisBonusDie { get; private set; }

	public bool TemporaryCloak { get; private set; }

	public bool TemporaryRogue { get; private set; }

	public bool TemporaryIgnoresEngagement { get; private set; }

	public bool TemporaryCommanded { get; private set; }

	public bool DestructionDeferred { get; private set; }

	public bool Destroyed
	{
		get
		{
			if (CurrentHull <= 0)
			{
				return !DestructionDeferred;
			}
			return false;
		}
	}

	public bool LethallyDamaged => CurrentHull <= 0;

	public void Initialize(int hull, int braceTokens = 0, int scatterTokens = 0, int evadeTokens = 0)
	{
		CurrentHull = Math.Max(1, hull);
		Activated = false;
		Commanded = false;
		CommandingShipRuntimeId = string.Empty;
		TemporaryGrit = false;
		TemporaryCloak = false;
		TemporaryRogue = false;
		DestructionDeferred = false;
		ClearActivationModifiers();
		EnsureTokenMaps();
		defenseTokenCounts.Clear();
		spentDefenseTokens.Clear();
		discardedDefenseTokens.Clear();
		defenseTokenCounts[DefenseTokenType.Brace] = Math.Max(0, braceTokens);
		defenseTokenCounts[DefenseTokenType.Scatter] = Math.Max(0, scatterTokens);
		defenseTokenCounts[DefenseTokenType.Evade] = Math.Max(0, evadeTokens);
		RebuildDefenseTokenSlotsFromCounts();
	}

	public bool IsDefenseTokenReady(DefenseTokenType token)
	{
		return GetReadyDefenseTokenCount(token) > 0;
	}

	public bool SpendDefenseToken(DefenseTokenType token)
	{
		DefenseTokenInstance spentToken;
		return SpendDefenseToken(token, out spentToken);
	}

	public bool SpendDefenseToken(DefenseTokenType token, out DefenseTokenInstance spentToken)
	{
		spentToken = default(DefenseTokenInstance);
		foreach (DefenseTokenInstance defenseTokenInstance in GetDefenseTokenInstances(token))
		{
			if (!defenseTokenInstance.Exhausted)
			{
				spentToken = defenseTokenInstance;
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

	public void GrantDefenseToken(DefenseTokenType token)
	{
		GetDefenseTokenSlots(token).Add(default(DefenseTokenSlotState));
		SyncDefenseTokenCounts(token);
	}

	public int ApplyDamage(int amount)
	{
		int currentHull = CurrentHull;
		CurrentHull = Math.Max(0, CurrentHull - Math.Max(0, amount));
		return currentHull - CurrentHull;
	}

	public int RecoverHull(int amount, int maximumHull)
	{
		if (CurrentHull <= 0 || amount <= 0 || maximumHull <= 0 || CurrentHull >= maximumHull)
		{
			return 0;
		}
		int currentHull = CurrentHull;
		CurrentHull = Math.Min(maximumHull, CurrentHull + Math.Max(0, amount));
		return CurrentHull - currentHull;
	}

	public bool Revive(int hull, bool activated = true)
	{
		if (!Destroyed || hull <= 0)
		{
			return false;
		}
		CurrentHull = hull;
		Activated = activated;
		Commanded = false;
		CommandingShipRuntimeId = string.Empty;
		DestructionDeferred = false;
		return true;
	}

	public bool DeferDestruction()
	{
		if (CurrentHull > 0 || DestructionDeferred)
		{
			return false;
		}
		DestructionDeferred = true;
		return true;
	}

	public bool FinalizeDeferredDestruction()
	{
		if (!DestructionDeferred)
		{
			return false;
		}
		CurrentHull = 0;
		DestructionDeferred = false;
		return true;
	}

	public void MarkActivated(bool commanded)
	{
		Activated = true;
		Commanded = commanded;
		if (!commanded)
		{
			CommandingShipRuntimeId = string.Empty;
		}
	}

	public void MarkCommandedBy(string shipRuntimeId)
	{
		Activated = true;
		Commanded = true;
		CommandingShipRuntimeId = shipRuntimeId ?? string.Empty;
	}

	public bool ReadyActivation()
	{
		if (Destroyed || !Activated)
		{
			return false;
		}
		Activated = false;
		Commanded = false;
		CommandingShipRuntimeId = string.Empty;
		return true;
	}

	public void GrantTemporaryGrit()
	{
		if (!Destroyed)
		{
			TemporaryGrit = true;
		}
	}

	public void GrantTemporarySnipe(int value)
	{
		if (!Destroyed)
		{
			TemporarySnipeValue = Math.Max(TemporarySnipeValue, Math.Max(0, value));
		}
	}

	public void GrantTemporaryAssault()
	{
		if (!Destroyed)
		{
			TemporaryAssault = true;
		}
	}

	public void GrantTemporarySwarm()
	{
		if (!Destroyed)
		{
			TemporarySwarm = true;
		}
	}

	public void SetTemporarySpeedOverride(int speed, bool cannotAttack)
	{
		if (!Destroyed)
		{
			TemporarySpeedOverride = Math.Max(0, speed);
			TemporaryCannotAttack = cannotAttack;
		}
	}

	public void ChooseTemporaryAttackOnly(bool yavarisBonusDie)
	{
		if (!Destroyed)
		{
			TemporaryAttackOnly = true;
			TemporaryYavarisBonusDie = yavarisBonusDie;
		}
	}

	public void PreventMovementForShipActivation()
	{
		if (!Destroyed)
		{
			movementPreventedForShipActivation = true;
		}
	}

	public void ClearShipActivationMovementRestriction()
	{
		movementPreventedForShipActivation = false;
	}

	public void GrantTemporaryCloak()
	{
		if (!Destroyed)
		{
			TemporaryCloak = true;
		}
	}

	public void GrantTemporaryRogue()
	{
		if (!Destroyed)
		{
			TemporaryRogue = true;
		}
	}

	public void GrantTemporaryIgnoreEngagement()
	{
		if (!Destroyed)
		{
			TemporaryIgnoresEngagement = true;
		}
	}

	public void GrantTemporaryCommanded()
	{
		if (!Destroyed)
		{
			TemporaryCommanded = true;
		}
	}

	public void ClearActivationModifiers()
	{
		TemporarySnipeValue = 0;
		TemporaryAssault = false;
		TemporarySwarm = false;
		TemporarySpeedOverride = -1;
		TemporaryCannotAttack = false;
		TemporaryAttackOnly = false;
		TemporaryYavarisBonusDie = false;
		TemporaryIgnoresEngagement = false;
		TemporaryCommanded = false;
	}

	public void Refresh()
	{
		Activated = false;
		Commanded = false;
		CommandingShipRuntimeId = string.Empty;
		TemporaryGrit = false;
		TemporaryCloak = false;
		TemporaryRogue = false;
		ClearShipActivationMovementRestriction();
		ClearActivationModifiers();
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

	private static int GetCount(Dictionary<DefenseTokenType, int> values, DefenseTokenType token)
	{
		if (values == null || !values.TryGetValue(token, out var value))
		{
			return 0;
		}
		return value;
	}
}
