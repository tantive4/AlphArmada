#nullable disable
namespace Armada.Data;

public static class ShipFactionCatalog
{
	public static string Id(ShipFaction faction)
	{
		return faction switch
		{
			ShipFaction.Rebel => "rebel",
			ShipFaction.Empire => "empire",
			ShipFaction.Republic => "republic",
			ShipFaction.Separatist => "separatist",
			_ => string.Empty,
		};
	}

	public static string Label(ShipFaction faction)
	{
		return faction switch
		{
			ShipFaction.Rebel => "REBEL ALLIANCE",
			ShipFaction.Empire => "GALACTIC EMPIRE",
			ShipFaction.Republic => "GALACTIC REPUBLIC",
			ShipFaction.Separatist => "SEPARATIST ALLIANCE",
			_ => "UNKNOWN FACTION",
		};
	}

	public static string FleetNamePrefix(ShipFaction faction)
	{
		return faction switch
		{
			ShipFaction.Rebel => "Rebel",
			ShipFaction.Empire => "Imperial",
			ShipFaction.Republic => "Republic",
			ShipFaction.Separatist => "Separatist",
			_ => "Unknown",
		};
	}

	public static bool TryParse(string value, out ShipFaction faction)
	{
		switch (NormalizeId(value))
		{
		case "rebel":
			faction = ShipFaction.Rebel;
			return true;
		case "empire":
			faction = ShipFaction.Empire;
			return true;
		case "republic":
			faction = ShipFaction.Republic;
			return true;
		case "separatist":
			faction = ShipFaction.Separatist;
			return true;
		default:
			faction = ShipFaction.Rebel;
			return false;
		}
	}

	public static string NormalizeId(string value)
	{
		string text = value?.Trim().ToLowerInvariant() ?? string.Empty;
		switch (text)
		{
		case "rebels":
		case "rebel alliance":
			return "rebel";
		case "imperial":
		case "imperials":
		case "galactic empire":
			return "empire";
		case "gar":
		case "galactic republic":
			return "republic";
		case "cis":
		case "confederacy of independent systems":
		case "separatists":
		case "separatist alliance":
			return "separatist";
		default:
			return text;
		}
	}

	public static bool UsesRebelPresentation(ShipFaction faction)
	{
		if (faction != ShipFaction.Rebel)
		{
			return faction == ShipFaction.Republic;
		}
		return true;
	}

	public static bool UsesRebelAudio(ShipFaction faction)
	{
		return faction != ShipFaction.Empire;
	}
}
