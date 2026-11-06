#nullable disable
using System;

namespace Armada.Session;

[Serializable]
public sealed class ShipBuildDto
{
	public int schemaVersion = 1;

	public string instanceId = string.Empty;

	public string shipDefinitionId = string.Empty;

	public string[] upgradeIds = Array.Empty<string>();
}
