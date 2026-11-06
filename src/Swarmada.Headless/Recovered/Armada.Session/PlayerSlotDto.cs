#nullable disable
using System;

namespace Armada.Session;

[Serializable]
public sealed class PlayerSlotDto
{
	public int schemaVersion = 1;

	public int playerId;

	public string displayName = string.Empty;

	public bool ready;

	public PlayerControlKind controlKind;

	public BattleAiDifficulty aiDifficulty;

	public FleetListDto fleet;
}
