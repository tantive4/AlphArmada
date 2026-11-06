#nullable disable
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Armada.Data;

namespace Armada.Fleet;

internal static class FleetContentText
{
	public static string RequiredId(string value, string parameter)
	{
		string text = Normalize(value);
		if (text.Length == 0)
		{
			throw new ArgumentException("A stable content ID is required.", parameter);
		}
		return text;
	}

	public static string ValueOrId(string value, string id)
	{
		string text = value?.Trim() ?? string.Empty;
		if (text.Length <= 0)
		{
			return id;
		}
		return text;
	}

	public static string Normalize(string value)
	{
		return value?.Trim() ?? string.Empty;
	}

	public static string NormalizeFaction(string value)
	{
		return ShipFactionCatalog.NormalizeId(value);
	}

	public static IReadOnlyList<string> CopyValues(IEnumerable<string> source, bool preserveDuplicates = false)
	{
		List<string> list = new List<string>();
		HashSet<string> hashSet = (preserveDuplicates ? null : new HashSet<string>(StringComparer.OrdinalIgnoreCase));
		if (source != null)
		{
			foreach (string item in source)
			{
				string text = Normalize(item);
				if (text.Length != 0 && (hashSet == null || hashSet.Add(text)))
				{
					list.Add(text);
				}
			}
		}
		return new ReadOnlyCollection<string>(list);
	}
}
