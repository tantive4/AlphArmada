#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Rules;

[Serializable]
public sealed partial class CommandStackState
{
	private readonly List<ShipCommand> commandStack = new List<ShipCommand>();

	private readonly Dictionary<ShipCommand, int> commandTokens = new Dictionary<ShipCommand, int>();

	private readonly List<ShipCommand> raidTokens = new List<ShipCommand>();

	public int Capacity { get; private set; }

	public int TokenCapacity { get; private set; }

	public bool AllowsDuplicateTokens { get; private set; }

	public int Count => commandStack.Count;

	public int CommandsNeeded => Math.Max(0, Capacity - Count);

	public int TokenCount
	{
		get
		{
			int num = 0;
			foreach (KeyValuePair<ShipCommand, int> commandToken in commandTokens)
			{
				num += commandToken.Value;
			}
			return num;
		}
	}

	public IReadOnlyList<ShipCommand> Stack => commandStack;

	public IReadOnlyCollection<ShipCommand> Tokens => commandTokens.Keys;

	public IEnumerable<ShipCommand> TokenInstances
	{
		get
		{
			foreach (KeyValuePair<ShipCommand, int> pair in commandTokens)
			{
				for (int index = 0; index < pair.Value; index++)
				{
					yield return pair.Key;
				}
			}
		}
	}

	public IReadOnlyList<ShipCommand> RaidTokens => raidTokens;

	public int RaidTokenCount => raidTokens.Count;

	public int RaidTokenCountOf(ShipCommand command)
	{
		int num = 0;
		foreach (ShipCommand raidToken in raidTokens)
		{
			if (raidToken == command)
			{
				num++;
			}
		}
		return num;
	}

	public bool CanGainRaidToken(ShipCommand command)
	{
		return !raidTokens.Contains(command);
	}

	public bool TryGainRaidToken(ShipCommand command)
	{
		if (!CanGainRaidToken(command))
		{
			return false;
		}
		raidTokens.Add(command);
		return true;
	}

	public void GainRaidToken(ShipCommand command)
	{
		TryGainRaidToken(command);
	}

	public bool DiscardRaidToken(ShipCommand command)
	{
		return raidTokens.Remove(command);
	}

	public List<ShipCommand> DiscardRaidTokensForDial(int limit = int.MaxValue)
	{
		List<ShipCommand> list = new List<ShipCommand>();
		if (limit <= 0)
		{
			return list;
		}
		foreach (ShipCommand item in new HashSet<ShipCommand>(raidTokens))
		{
			if (list.Count >= limit)
			{
				break;
			}
			if (raidTokens.Remove(item))
			{
				list.Add(item);
			}
		}
		return list;
	}

	public CommandStackState(int commandValue)
	{
		Capacity = Math.Clamp(commandValue, 1, 4);
		TokenCapacity = Capacity;
	}

	public void ConfigureTokenStorage(int capacity, bool allowDuplicates = false)
	{
		TokenCapacity = Math.Max(Capacity, capacity);
		AllowsDuplicateTokens = allowDuplicates;
	}

	public void ConfigureCommandCapacity(int capacity)
	{
		Capacity = Math.Clamp(capacity, 1, 4);
		TokenCapacity = Math.Max(TokenCapacity, Capacity);
	}

	public bool QueueAtBottom(ShipCommand command)
	{
		if (commandStack.Count >= Capacity)
		{
			return false;
		}
		commandStack.Add(command);
		return true;
	}

	public bool TryPeekTop(out ShipCommand command)
	{
		if (commandStack.Count == 0)
		{
			command = ShipCommand.Navigate;
			return false;
		}
		command = commandStack[0];
		return true;
	}

	public bool TryRevealTop(out ShipCommand command)
	{
		if (!TryPeekTop(out command))
		{
			return false;
		}
		commandStack.RemoveAt(0);
		return true;
	}

	public bool TryDiscardTop(out ShipCommand command)
	{
		return TryRevealTop(out command);
	}

	public int DiscardAllCommands()
	{
		int count = commandStack.Count;
		commandStack.Clear();
		return count;
	}

	public bool ReplaceTop(ShipCommand command)
	{
		if (commandStack.Count == 0)
		{
			return false;
		}
		commandStack[0] = command;
		return true;
	}

	public bool ReplaceAt(int index, ShipCommand command)
	{
		if (index < 0 || index >= commandStack.Count)
		{
			return false;
		}
		commandStack[index] = command;
		return true;
	}

	public int ReplaceAll(ShipCommand command)
	{
		for (int i = 0; i < commandStack.Count; i++)
		{
			commandStack[i] = command;
		}
		return commandStack.Count;
	}

	public int ReplaceAllWith(IReadOnlyList<ShipCommand> commands)
	{
		if (commands == null || commands.Count == 0)
		{
			return 0;
		}
		int num = 0;
		for (int i = 0; i < commandStack.Count; i++)
		{
			commandStack[i] = commands[i % commands.Count];
			num++;
		}
		return num;
	}

	public bool HasToken(ShipCommand command)
	{
		if (commandTokens.TryGetValue(command, out var value))
		{
			return value > 0;
		}
		return false;
	}

	public bool CanStoreToken(ShipCommand command)
	{
		if (TokenCount < TokenCapacity)
		{
			if (!AllowsDuplicateTokens)
			{
				return !HasToken(command);
			}
			return true;
		}
		return false;
	}

	public bool TryStoreToken(ShipCommand command)
	{
		if (!CanStoreToken(command))
		{
			return false;
		}
		commandTokens[command] = ((!commandTokens.TryGetValue(command, out var value)) ? 1 : (value + 1));
		return true;
	}

	public bool TrySpendToken(ShipCommand command)
	{
		if (!commandTokens.TryGetValue(command, out var value) || value <= 0)
		{
			return false;
		}
		if (value == 1)
		{
			commandTokens.Remove(command);
		}
		else
		{
			commandTokens[command] = value - 1;
		}
		return true;
	}
}
