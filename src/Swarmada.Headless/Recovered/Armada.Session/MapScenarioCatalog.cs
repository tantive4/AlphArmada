#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Session;

public static class MapScenarioCatalog
{
	public const string DefaultId = "default";

	public const string EscortVipId = "escort-the-vip";

	public const string SeizeStationId = "seize-the-station";

	public const string IntelRetrievalId = "intel-retrieval";

	private static readonly MapScenarioDefinition[] Definitions = new MapScenarioDefinition[4]
	{
		new MapScenarioDefinition("default", "Default"),
		new MapScenarioDefinition("escort-the-vip", "Escort the VIP", locksLongApproach: true),
		new MapScenarioDefinition("seize-the-station", "Seize the Station"),
		new MapScenarioDefinition("intel-retrieval", "Intel Retrieval", locksLongApproach: false, ScenarioObjectiveShipRequirement.LightOrMedium)
	};

	public static IReadOnlyList<MapScenarioDefinition> All => Definitions;

	public static string Normalize(string scenarioId)
	{
		if (string.IsNullOrWhiteSpace(scenarioId))
		{
			return "default";
		}
		MapScenarioDefinition[] definitions = Definitions;
		foreach (MapScenarioDefinition mapScenarioDefinition in definitions)
		{
			if (string.Equals(mapScenarioDefinition.Id, scenarioId, StringComparison.OrdinalIgnoreCase))
			{
				return mapScenarioDefinition.Id;
			}
		}
		return "default";
	}

	public static bool IsSupported(string scenarioId)
	{
		if (string.IsNullOrWhiteSpace(scenarioId))
		{
			return true;
		}
		MapScenarioDefinition[] definitions = Definitions;
		for (int i = 0; i < definitions.Length; i++)
		{
			if (string.Equals(definitions[i].Id, scenarioId, StringComparison.OrdinalIgnoreCase))
			{
				return true;
			}
		}
		return false;
	}

	public static MapScenarioDefinition Get(string scenarioId)
	{
		string b = Normalize(scenarioId);
		MapScenarioDefinition[] definitions = Definitions;
		foreach (MapScenarioDefinition mapScenarioDefinition in definitions)
		{
			if (string.Equals(mapScenarioDefinition.Id, b, StringComparison.Ordinal))
			{
				return mapScenarioDefinition;
			}
		}
		return Definitions[0];
	}

	public static string DisplayName(string scenarioId)
	{
		return Get(scenarioId).DisplayName;
	}

	public static bool LocksLongApproach(string scenarioId)
	{
		return Get(scenarioId).LocksLongApproach;
	}

	public static ScenarioObjectiveShipRequirement ObjectiveShipRequirement(string scenarioId)
	{
		return Get(scenarioId).ObjectiveShipRequirement;
	}

	public static int NormalizeEscortPlayerId(int playerId)
	{
		if (playerId != 1)
		{
			return 2;
		}
		return 1;
	}

	public static int InterceptorPlayerId(int escortPlayerId)
	{
		if (NormalizeEscortPlayerId(escortPlayerId) != 1)
		{
			return 1;
		}
		return 2;
	}

	public static BattleStartingMode ResolveStartingMode(string scenarioId, BattleStartingMode requested)
	{
		if (!LocksLongApproach(scenarioId))
		{
			return requested;
		}
		return BattleStartingMode.LongApproach;
	}

	public static BattleStartingMode DefaultStartingMode(string scenarioId)
	{
		if (!LocksLongApproach(scenarioId))
		{
			return BattleStartingMode.StandardFront;
		}
		return BattleStartingMode.LongApproach;
	}

	public static string Step(string scenarioId, int delta)
	{
		string b = Normalize(scenarioId);
		int num = 0;
		for (int i = 0; i < Definitions.Length; i++)
		{
			if (string.Equals(Definitions[i].Id, b, StringComparison.Ordinal))
			{
				num = i;
				break;
			}
		}
		int num2 = ((delta >= 0) ? 1 : (-1));
		int num3 = (num + num2) % Definitions.Length;
		if (num3 < 0)
		{
			num3 += Definitions.Length;
		}
		return Definitions[num3].Id;
	}
}
