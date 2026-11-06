#nullable disable
using System;
using System.Collections.Generic;

namespace Armada.Rules;

public static class RulesTextPresenter
{
	private static readonly KeyValuePair<string, string>[] Replacements = new KeyValuePair<string, string>[78]
	{
		Pair(":crit:", "critical result"),
		Pair(":accuracy:", "accuracy result"),
		Pair(":hit:", "hit result"),
		Pair(":squadron:", "Squadron command"),
		Pair(":squad:", "Squadron command"),
		Pair(":concentrate fire:", "Concentrate Fire command"),
		Pair(":confire:", "Concentrate Fire command"),
		Pair(":navigate:", "Navigate command"),
		Pair(":nav:", "Navigate command"),
		Pair(":repair:", "Engineering command"),
		Pair(":engineering:", "Engineering command"),
		Pair(":brace:", "Brace token"),
		Pair(":redirect:", "Redirect token"),
		Pair(":evade:", "Evade token"),
		Pair(":contain:", "Contain token"),
		Pair(":salvo:", "Salvo token"),
		Pair(":scatter:", "Scatter token"),
		Pair("long-range weapon dice", "long-range shots"),
		Pair("long-range weapon die", "long-range shot"),
		Pair("midrange weapon dice", "mid-range shots"),
		Pair("midrange weapon die", "mid-range shot"),
		Pair("mid-range weapon dice", "mid-range shots"),
		Pair("mid-range weapon die", "mid-range shot"),
		Pair("close-range weapon dice", "close-range shots"),
		Pair("close-range weapon die", "close-range shot"),
		Pair("long-range dice", "long-range shots"),
		Pair("long-range die", "long-range shot"),
		Pair("midrange dice", "mid-range shots"),
		Pair("midrange die", "mid-range shot"),
		Pair("mid-range dice", "mid-range shots"),
		Pair("mid-range die", "mid-range shot"),
		Pair("close-range dice", "close-range shots"),
		Pair("close-range die", "close-range shot"),
		Pair("reroll red", "reroll long-range shot"),
		Pair("red reroll", "long-range shot reroll"),
		Pair("reroll blue", "reroll mid-range shot"),
		Pair("blue reroll", "mid-range shot reroll"),
		Pair("reroll black", "reroll close-range shot"),
		Pair("black reroll", "close-range shot reroll"),
		Pair("red critical-icon dice", "long-range critical-icon shots"),
		Pair("red critical-icon die", "long-range critical-icon shot"),
		Pair("blue critical-icon dice", "mid-range critical-icon shots"),
		Pair("blue critical-icon die", "mid-range critical-icon shot"),
		Pair("black critical-icon dice", "close-range critical-icon shots"),
		Pair("black critical-icon die", "close-range critical-icon shot"),
		Pair("red accuracy dice", "long-range accuracy shots"),
		Pair("red accuracy die", "long-range accuracy shot"),
		Pair("blue accuracy dice", "mid-range accuracy shots"),
		Pair("blue accuracy die", "mid-range accuracy shot"),
		Pair("black accuracy dice", "close-range accuracy shots"),
		Pair("black accuracy die", "close-range accuracy shot"),
		Pair("red attack dice", "long-range shots"),
		Pair("red attack die", "long-range shot"),
		Pair("blue attack dice", "mid-range shots"),
		Pair("blue attack die", "mid-range shot"),
		Pair("black attack dice", "close-range shots"),
		Pair("black attack die", "close-range shot"),
		Pair("red battery dice", "long-range shots"),
		Pair("red battery die", "long-range shot"),
		Pair("blue battery dice", "mid-range shots"),
		Pair("blue battery die", "mid-range shot"),
		Pair("black battery dice", "close-range shots"),
		Pair("black battery die", "close-range shot"),
		Pair("red-dice", "long-range shots"),
		Pair("red-die", "long-range shot"),
		Pair("blue-dice", "mid-range shots"),
		Pair("blue-die", "mid-range shot"),
		Pair("black-dice", "close-range shots"),
		Pair("black-die", "close-range shot"),
		Pair("red dice", "long-range shots"),
		Pair("red die", "long-range shot"),
		Pair("blue dice", "mid-range shots"),
		Pair("blue die", "mid-range shot"),
		Pair("black dice", "close-range shots"),
		Pair("black die", "close-range shot"),
		Pair("red critical result", "long-range critical result"),
		Pair("blue critical result", "mid-range critical result"),
		Pair("black critical result", "close-range critical result")
	};

	public static string ToVideoGameTerms(string printedText)
	{
		if (string.IsNullOrWhiteSpace(printedText))
		{
			return string.Empty;
		}
		string text = printedText.Trim();
		KeyValuePair<string, string>[] replacements = Replacements;
		for (int i = 0; i < replacements.Length; i++)
		{
			KeyValuePair<string, string> keyValuePair = replacements[i];
			text = ReplaceIgnoreCase(text, keyValuePair.Key, keyValuePair.Value);
		}
		text = text.Replace("\r\n", " ").Replace('\n', ' ').Replace("  ", " ");
		if (text.IndexOf("discard 1 command token from this card", StringComparison.OrdinalIgnoreCase) >= 0 && text.IndexOf("each friendly ship gains 1 matching command token", StringComparison.OrdinalIgnoreCase) >= 0 && text.IndexOf("six command tokens", StringComparison.OrdinalIgnoreCase) < 0)
		{
			text = "At the beginning of the game, choose six command tokens to store on this card. " + text;
		}
		return text.Trim();
	}

	private static KeyValuePair<string, string> Pair(string printed, string presented)
	{
		return new KeyValuePair<string, string>(printed, presented);
	}

	private static string ReplaceIgnoreCase(string source, string oldValue, string newValue)
	{
		int startIndex = 0;
		while (true)
		{
			int num = source.IndexOf(oldValue, startIndex, StringComparison.OrdinalIgnoreCase);
			if (num < 0)
			{
				break;
			}
			string text = MatchCase(source.Substring(num, oldValue.Length), newValue);
			source = source.Substring(0, num) + text + source.Substring(num + oldValue.Length);
			startIndex = num + text.Length;
		}
		return source;
	}

	private static string MatchCase(string printed, string presented)
	{
		if (string.IsNullOrEmpty(printed) || string.IsNullOrEmpty(presented))
		{
			return presented;
		}
		if (string.Equals(printed, printed.ToUpperInvariant(), StringComparison.Ordinal))
		{
			return presented.ToUpperInvariant();
		}
		if (char.IsUpper(printed[0]))
		{
			return char.ToUpperInvariant(presented[0]) + presented.Substring(1);
		}
		return presented;
	}
}
