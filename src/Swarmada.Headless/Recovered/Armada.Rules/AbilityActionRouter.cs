#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Rules;

public sealed class AbilityActionRouter
{
	private readonly struct PrefixRoute
	{
		public string Prefix { get; }

		public Func<AbilityAction, bool> Handler { get; }

		public PrefixRoute(string prefix, Func<AbilityAction, bool> handler)
		{
			Prefix = prefix;
			Handler = handler;
		}
	}

	private readonly Dictionary<string, Func<AbilityAction, bool>> exactRoutes = new Dictionary<string, Func<AbilityAction, bool>>(StringComparer.Ordinal);

	private readonly List<PrefixRoute> prefixRoutes = new List<PrefixRoute>();

	public AbilityActionRouter Register(string abilityId, Func<AbilityAction, bool> handler)
	{
		if (string.IsNullOrWhiteSpace(abilityId))
		{
			throw new ArgumentException("Ability ID is required.", "abilityId");
		}
		if (handler == null)
		{
			throw new ArgumentNullException("handler");
		}
		if (exactRoutes.ContainsKey(abilityId))
		{
			throw new InvalidOperationException("Ability '" + abilityId + "' is already registered.");
		}
		exactRoutes.Add(abilityId, handler);
		return this;
	}

	public AbilityActionRouter RegisterPrefix(string prefix, Func<AbilityAction, bool> handler)
	{
		if (string.IsNullOrWhiteSpace(prefix))
		{
			throw new ArgumentException("Ability prefix is required.", "prefix");
		}
		if (handler == null)
		{
			throw new ArgumentNullException("handler");
		}
		foreach (PrefixRoute prefixRoute in prefixRoutes)
		{
			if (string.Equals(prefixRoute.Prefix, prefix, StringComparison.Ordinal))
			{
				throw new InvalidOperationException("Ability prefix '" + prefix + "' is already registered.");
			}
		}
		prefixRoutes.Add(new PrefixRoute(prefix, handler));
		prefixRoutes.Sort((PrefixRoute left, PrefixRoute right) => right.Prefix.Length.CompareTo(left.Prefix.Length));
		return this;
	}

	public bool TryDispatch(AbilityAction action)
	{
		if (string.IsNullOrWhiteSpace(action.AbilityId))
		{
			return false;
		}
		if (exactRoutes.TryGetValue(action.AbilityId, out var value))
		{
			return value(action);
		}
		foreach (PrefixRoute prefixRoute in prefixRoutes)
		{
			if (action.AbilityId.StartsWith(prefixRoute.Prefix, StringComparison.Ordinal))
			{
				return prefixRoute.Handler(action);
			}
		}
		return false;
	}
}
