#nullable disable
using System;
using Armada.Core;

namespace Armada.Rules;

public readonly struct IntelOfficerTokenTarget : IEquatable<IntelOfficerTokenTarget>
{
	public RuntimeUnitKind UnitKind { get; }

	public RuntimeUnitId UnitId { get; }

	public DefenseTokenInstance Token { get; }

	public IntelOfficerTokenTarget(RuntimeUnitKind unitKind, RuntimeUnitId unitId, DefenseTokenInstance token)
	{
		if (unitKind != RuntimeUnitKind.Ship && unitKind != RuntimeUnitKind.Squadron)
		{
			throw new ArgumentOutOfRangeException("unitKind");
		}
		if (!unitId.IsValid)
		{
			throw new ArgumentException("A stable unit ID is required.", "unitId");
		}
		UnitKind = unitKind;
		UnitId = unitId;
		Token = token;
	}

	public bool Equals(IntelOfficerTokenTarget other)
	{
		if (UnitKind == other.UnitKind && UnitId == other.UnitId)
		{
			return Token == other.Token;
		}
		return false;
	}

	public override bool Equals(object obj)
	{
		if (obj is IntelOfficerTokenTarget other)
		{
			return Equals(other);
		}
		return false;
	}

	public override int GetHashCode()
	{
		return HashCode.Combine(UnitKind, UnitId, Token);
	}

	public static bool operator ==(IntelOfficerTokenTarget left, IntelOfficerTokenTarget right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(IntelOfficerTokenTarget left, IntelOfficerTokenTarget right)
	{
		return !left.Equals(right);
	}
}
