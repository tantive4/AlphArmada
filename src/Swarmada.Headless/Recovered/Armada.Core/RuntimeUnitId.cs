#nullable disable
using System;

namespace Armada.Core;

[Serializable]
public readonly struct RuntimeUnitId : IEquatable<RuntimeUnitId>, IComparable<RuntimeUnitId>
{
	private readonly string value;

	public string Value => value ?? string.Empty;

	public bool IsValid => !string.IsNullOrWhiteSpace(value);

	public RuntimeUnitId(string value)
	{
		if (string.IsNullOrWhiteSpace(value))
		{
			throw new ArgumentException("A runtime unit ID cannot be empty.", "value");
		}
		this.value = value.Trim();
	}

	public static bool TryParse(string candidate, out RuntimeUnitId runtimeUnitId)
	{
		if (string.IsNullOrWhiteSpace(candidate))
		{
			runtimeUnitId = default(RuntimeUnitId);
			return false;
		}
		runtimeUnitId = new RuntimeUnitId(candidate);
		return true;
	}

	public static RuntimeUnitId ForMatchEntry(string matchId, int playerId, RuntimeUnitKind kind, string fleetEntryInstanceId)
	{
		if (string.IsNullOrWhiteSpace(matchId))
		{
			throw new ArgumentException("Match ID is required.", "matchId");
		}
		if (playerId <= 0)
		{
			throw new ArgumentOutOfRangeException("playerId");
		}
		if (string.IsNullOrWhiteSpace(fleetEntryInstanceId))
		{
			throw new ArgumentException("Fleet entry instance ID is required.", "fleetEntryInstanceId");
		}
		if (kind != RuntimeUnitKind.Ship && kind != RuntimeUnitKind.Squadron)
		{
			throw new ArgumentOutOfRangeException("kind");
		}
		string arg = ((kind == RuntimeUnitKind.Ship) ? "ship" : "squadron");
		return new RuntimeUnitId($"{matchId.Trim()}|p{playerId}|{arg}|" + fleetEntryInstanceId.Trim());
	}

	public bool Equals(RuntimeUnitId other)
	{
		return string.Equals(Value, other.Value, StringComparison.Ordinal);
	}

	public override bool Equals(object obj)
	{
		if (obj is RuntimeUnitId other)
		{
			return Equals(other);
		}
		return false;
	}

	public override int GetHashCode()
	{
		return StringComparer.Ordinal.GetHashCode(Value);
	}

	public int CompareTo(RuntimeUnitId other)
	{
		return string.Compare(Value, other.Value, StringComparison.Ordinal);
	}

	public override string ToString()
	{
		return Value;
	}

	public static bool operator ==(RuntimeUnitId left, RuntimeUnitId right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(RuntimeUnitId left, RuntimeUnitId right)
	{
		return !left.Equals(right);
	}
}
