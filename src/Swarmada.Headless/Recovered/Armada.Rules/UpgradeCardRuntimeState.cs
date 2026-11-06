#nullable disable
using System;
using System.Collections.Generic;
using Armada.Core;
using Armada.Data;

namespace Armada.Rules;

[Serializable]
public sealed partial class UpgradeCardRuntimeState
{
	private readonly Dictionary<ShipCommand, int> typedCommandTokens = new Dictionary<ShipCommand, int>();

	private readonly List<string> storedUnitIds = new List<string>();

	private readonly List<int> scheduledRounds = new List<int>();

	public UpgradeCardDefinition Definition { get; }

	public string EffectiveDefinitionId { get; private set; }

	public UpgradeCardDefinition EffectiveDefinition
	{
		get
		{
			if (!UpgradeCatalog.TryGet(EffectiveDefinitionId, out var definition))
			{
				return Definition;
			}
			return definition;
		}
	}

	public bool Exhausted { get; private set; }

	public bool Discarded { get; private set; }

	public int CommandTokens { get; private set; }

	public ShipCommand? AssignedCommandToken { get; private set; }

	public int UsesThisActivation { get; private set; }

	public int UsesThisAttack { get; private set; }

	public int LastUsedRound { get; private set; }

	public HullZone? FocusZone { get; private set; }

	public bool SetupInitialized { get; private set; }

	public AttackDieColor StoredDieColor { get; private set; }

	public DieFace StoredDieFace { get; private set; }

	public bool HasStoredDie => StoredDieColor != AttackDieColor.None;

	public IReadOnlyDictionary<ShipCommand, int> TypedCommandTokens => typedCommandTokens;

	public IReadOnlyList<string> StoredUnitIds => storedUnitIds;

	public bool Active => !Discarded;

	public IReadOnlyList<int> ScheduledRounds => scheduledRounds;

	public UpgradeCardRuntimeState(UpgradeCardDefinition definition)
	{
		Definition = definition ?? throw new ArgumentNullException("definition");
		EffectiveDefinitionId = definition.id;
		CommandTokens = 0;
		if (definition.startCommandIcons != null && definition.startCommandIcons.Length == 1 && TryParseCommandIcon(definition.startCommandIcons[0], out var command))
		{
			AssignedCommandToken = command;
		}
		if (definition.startCommandIcons != null && definition.startCommandIcons.Length != 0 && definition.startCommandAmount > 0)
		{
			List<ShipCommand> list = new List<ShipCommand>();
			bool flag = false;
			string[] startCommandIcons = definition.startCommandIcons;
			foreach (string text in startCommandIcons)
			{
				ShipCommand command2;
				if (string.Equals(text, "any", StringComparison.OrdinalIgnoreCase))
				{
					flag = true;
				}
				else if (TryParseCommandIcon(text, out command2))
				{
					list.Add(command2);
				}
			}
			for (int j = 0; j < definition.startCommandAmount; j++)
			{
				if (list.Count <= 0)
				{
					break;
				}
				if (flag)
				{
					break;
				}
				ShipCommand command3 = list[j % list.Count];
				AddTypedCommandToken(command3);
			}
		}
		else
		{
			CommandTokens = Math.Max(0, definition.startCommandAmount);
		}
	}

	public bool Exhaust()
	{
		if (!Active || Exhausted)
		{
			return false;
		}
		Exhausted = true;
		return true;
	}

	public bool Discard()
	{
		if (!Active)
		{
			return false;
		}
		Discarded = true;
		Exhausted = false;
		CommandTokens = 0;
		AssignedCommandToken = null;
		UsesThisActivation = 0;
		UsesThisAttack = 0;
		LastUsedRound = 0;
		SetupInitialized = false;
		StoredDieColor = AttackDieColor.None;
		StoredDieFace = DieFace.Blank;
		typedCommandTokens.Clear();
		storedUnitIds.Clear();
		return true;
	}

	public bool TryReplaceEffectiveDefinition(string cardId)
	{
		if (!Active || !UpgradeCatalog.TryGet(cardId, out var definition) || !string.Equals(definition.slot, "fleet-command", StringComparison.OrdinalIgnoreCase))
		{
			return false;
		}
		EffectiveDefinitionId = definition.id;
		Exhausted = false;
		return true;
	}

	public void Ready()
	{
		if (Active)
		{
			Exhausted = false;
			FocusZone = null;
		}
	}

	public bool SetFocusZone(HullZone zone)
	{
		if (!Active)
		{
			return false;
		}
		FocusZone = zone;
		return true;
	}

	public void SetCommandTokens(int value)
	{
		CommandTokens = Math.Max(0, value);
	}

	public bool TrySpendStoredCommandToken(int amount = 1)
	{
		int num = Math.Max(1, amount);
		if (!Active || CommandTokens < num)
		{
			return false;
		}
		CommandTokens -= num;
		return true;
	}

	public void AddStoredCommandToken(int amount = 1)
	{
		if (Active)
		{
			CommandTokens += Math.Max(0, amount);
		}
	}

	public int GetTypedCommandTokenCount(ShipCommand command)
	{
		if (!typedCommandTokens.TryGetValue(command, out var value))
		{
			return 0;
		}
		return value;
	}

	public void AddTypedCommandToken(ShipCommand command, int amount = 1)
	{
		int num = Math.Max(0, amount);
		if (Active && num > 0)
		{
			typedCommandTokens[command] = GetTypedCommandTokenCount(command) + num;
			CommandTokens += num;
		}
	}

	public bool TrySpendTypedCommandToken(ShipCommand command)
	{
		int typedCommandTokenCount = GetTypedCommandTokenCount(command);
		if (!Active || typedCommandTokenCount <= 0)
		{
			return false;
		}
		if (typedCommandTokenCount == 1)
		{
			typedCommandTokens.Remove(command);
		}
		else
		{
			typedCommandTokens[command] = typedCommandTokenCount - 1;
		}
		CommandTokens = Math.Max(0, CommandTokens - 1);
		return true;
	}

	public void ClearTypedCommandTokens()
	{
		typedCommandTokens.Clear();
		CommandTokens = 0;
	}

	public bool StoreUnit(string runtimeId)
	{
		if (!Active || string.IsNullOrWhiteSpace(runtimeId) || storedUnitIds.Contains(runtimeId))
		{
			return false;
		}
		storedUnitIds.Add(runtimeId);
		return true;
	}

	public void MarkSetupInitialized()
	{
		if (Active)
		{
			SetupInitialized = true;
		}
	}

	public void StoreDie(AttackDieColor color, DieFace face)
	{
		if (Active && color != AttackDieColor.None)
		{
			StoredDieColor = color;
			StoredDieFace = face;
		}
	}

	public bool RemoveStoredUnit(string runtimeId)
	{
		if (Active && !string.IsNullOrWhiteSpace(runtimeId))
		{
			return storedUnitIds.Remove(runtimeId);
		}
		return false;
	}

	private static bool TryParseCommandIcon(string icon, out ShipCommand command)
	{
		switch ((icon ?? string.Empty).Trim().ToLowerInvariant())
		{
		case "navigate":
		case "nav":
			command = ShipCommand.Navigate;
			return true;
		case "repair":
		case "engineering":
			command = ShipCommand.Engineering;
			return true;
		case "squadron":
		case "squad":
			command = ShipCommand.Squadron;
			return true;
		case "confire":
			command = ShipCommand.ConcentrateFire;
			return true;
		default:
			command = ShipCommand.Navigate;
			return false;
		}
	}

	public void SetAssignedCommandToken(ShipCommand command)
	{
		AssignedCommandToken = command;
	}

	public bool TryUseThisActivation(int limit = 1)
	{
		if (!Active || UsesThisActivation >= Math.Max(1, limit))
		{
			return false;
		}
		UsesThisActivation++;
		return true;
	}

	public void ResetActivationUsage()
	{
		UsesThisActivation = 0;
		UsesThisAttack = 0;
	}

	public bool TryUseThisAttack(int limit = 1)
	{
		if (!Active || UsesThisAttack >= Math.Max(1, limit))
		{
			return false;
		}
		UsesThisAttack++;
		return true;
	}

	public void ResetAttackUsage()
	{
		UsesThisAttack = 0;
	}

	public bool TryUseThisRound(int round)
	{
		if (!Active || round <= 0 || LastUsedRound == round)
		{
			return false;
		}
		LastUsedRound = round;
		return true;
	}

	public void ScheduleRounds(params int[] rounds)
	{
		scheduledRounds.Clear();
		if (rounds == null)
		{
			return;
		}
		foreach (int num in rounds)
		{
			if (num > 0 && !scheduledRounds.Contains(num))
			{
				scheduledRounds.Add(num);
			}
		}
		scheduledRounds.Sort();
	}

	public bool IsScheduledFor(int round)
	{
		return scheduledRounds.Contains(round);
	}
}
