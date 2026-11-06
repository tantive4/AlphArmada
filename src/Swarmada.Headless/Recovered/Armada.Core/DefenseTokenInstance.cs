#nullable disable
using System;

namespace Armada.Core;

[Serializable]
public readonly struct DefenseTokenInstance : IEquatable<DefenseTokenInstance>
{
	public DefenseTokenType Token { get; }

	public int Slot { get; }

	public bool Exhausted { get; }

	[System.Text.Json.Serialization.JsonConstructor]
	public DefenseTokenInstance(DefenseTokenType token, int slot, bool exhausted)
	{
		Token = token;
		Slot = Math.Max(0, slot);
		Exhausted = exhausted;
	}

	public bool Equals(DefenseTokenInstance other)
	{
		if (Token == other.Token && Slot == other.Slot)
		{
			return Exhausted == other.Exhausted;
		}
		return false;
	}

	public override bool Equals(object obj)
	{
		if (obj is DefenseTokenInstance other)
		{
			return Equals(other);
		}
		return false;
	}

	public override int GetHashCode()
	{
		return HashCode.Combine(Token, Slot, Exhausted);
	}

	public static bool operator ==(DefenseTokenInstance left, DefenseTokenInstance right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(DefenseTokenInstance left, DefenseTokenInstance right)
	{
		return !left.Equals(right);
	}
}
