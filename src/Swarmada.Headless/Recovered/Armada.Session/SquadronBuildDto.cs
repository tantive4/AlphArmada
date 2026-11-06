#nullable disable
using System;

namespace Armada.Session;

[Serializable]
public sealed class SquadronBuildDto
{
	public int schemaVersion = 1;

	public string instanceId = string.Empty;

	public string squadronDefinitionId = string.Empty;

	public string leaderUpgradeId = string.Empty;
}
