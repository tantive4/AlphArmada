#nullable disable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using Armada.Data;
using Armada.Session;

namespace Armada.Fleet;

public sealed class FleetTextImporter
{
	private sealed class RawUpgrade
	{
		public string Name = string.Empty;

		public int Points;
	}

	private sealed class RawShip
	{
		public string Name = string.Empty;

		public int Points;

		public readonly List<RawUpgrade> Upgrades = new List<RawUpgrade>();
	}

	private sealed class RawSquadron
	{
		public string Name = string.Empty;

		public int Count = 1;

		public int TotalPoints;
	}

	private sealed class ParsedFleet
	{
		public string Name = string.Empty;

		public string Faction = string.Empty;

		public int? DeclaredPoints;

		public RawUpgrade HeaderCommander;

		public readonly List<RawShip> Ships = new List<RawShip>();

		public readonly List<RawSquadron> Squadrons = new List<RawSquadron>();
	}

	private static readonly Regex CardPointsPattern = new Regex("^(?<name>.+?)\\s*\\(\\s*(?<points>\\d+)\\s*(?:points?|pts?)?\\s*\\)\\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

	private static readonly Regex SquadronCountPattern = new Regex("^(?<count>\\d+)\\s*[x\\u00d7]\\s*(?<name>.+)$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

	private static readonly Regex SubtotalPattern = new Regex("^=\\s*\\(?\\s*\\d+\\s*(?:points?|pts?)\\s*\\)?\\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

	private static readonly Regex SectionPointsPattern = new Regex("^\\(?\\s*\\d+\\s*(?:points?|pts?)?\\s*\\)?$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

	private static readonly Regex ShipGroupHeaderPattern = new Regex("^(?:Flagship|Ship\\s+\\d+)\\s*:\\s*(?:\\(\\s*\\d+\\s*(?:points?|pts?)?\\s*\\))?\\s*$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

	private static readonly Regex ObjectivePattern = new Regex("^(?<kind>Assault|Defense|Navigation|Campaign|Skirmish)(?:\\s+Objective)?\\s*:\\s*(?<name>.+)$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

	private static readonly Regex ContentTagPattern = new Regex("\\s*(?:\\[[^\\]]+\\]|\\{[^}]+\\})\\s*", RegexOptions.CultureInvariant);

	private static readonly Regex HtmlBreakPattern = new Regex("<br\\s*/?>", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

	private readonly IFleetContentCatalog content;

	private readonly FleetLegalityValidator upgradeProbe;

	public FleetTextImporter(IFleetContentCatalog content)
	{
		this.content = content ?? throw new ArgumentNullException("content");
		upgradeProbe = new FleetLegalityValidator(content, new FleetRuleset("fleet-import-upgrade-probe", 9999, 999, 1, requireStandardObjectives: false, requireStrictOfficialContent: false), new FleetValidationOptions(reportGameplayCoverage: false));
	}

	public FleetImportResult Import(string exportedText, string saveAsName = null)
	{
		FleetImportResult fleetImportResult = new FleetImportResult();
		if (string.IsNullOrWhiteSpace(exportedText))
		{
			fleetImportResult.Error = "Paste a fleet text export first.";
			return fleetImportResult;
		}
		ParsedFleet parsedFleet = Parse(exportedText, fleetImportResult);
		fleetImportResult.SourceDeclaredPoints = parsedFleet.DeclaredPoints;
		string faction = NormalizeSupportedFaction(parsedFleet.Faction);
		if (!string.IsNullOrWhiteSpace(parsedFleet.Faction) && faction.Length == 0)
		{
			fleetImportResult.Error = "Only Rebel Alliance, Galactic Empire, Galactic Republic, and Separatist Alliance fleets can be imported in this version.";
			return fleetImportResult;
		}
		FleetDraft fleetDraft = FleetDraft.Create((!string.IsNullOrWhiteSpace(saveAsName)) ? saveAsName.Trim() : (string.IsNullOrWhiteSpace(parsedFleet.Name) ? "Imported Fleet" : parsedFleet.Name.Trim()), faction, 9999, "custom");
		AddShips(parsedFleet, fleetDraft, fleetImportResult, ref faction);
		AddSquadrons(parsedFleet, fleetDraft, fleetImportResult, ref faction);
		if (faction.Length == 0)
		{
			fleetImportResult.Error = "The fleet faction could not be determined from its header or supported units.";
			return fleetImportResult;
		}
		fleetDraft.Document.faction = faction;
		fleetDraft.Document.pointLimit = FleetRuleset.OfficialStandard400.MaximumFleetPoints;
		if (fleetDraft.Document.ships.Length == 0)
		{
			fleetImportResult.Error = "No supported capital ship was found, so the fleet was not saved.";
			return fleetImportResult;
		}
		FleetListDto fleet = (fleetImportResult.Fleet = fleetDraft.Snapshot());
		fleetImportResult.RecalculatedPoints = new FleetLegalityValidator(content, FleetRuleset.Custom, new FleetValidationOptions(reportGameplayCoverage: false)).Validate(fleet).TotalPoints;
		return fleetImportResult;
	}

	private static ParsedFleet Parse(string exportedText, FleetImportResult result)
	{
		ParsedFleet parsed = new ParsedFleet();
		string text = Regex.Replace(HtmlBreakPattern.Replace(WebUtility.HtmlDecode(exportedText), "\n"), "<[^>]*>", string.Empty).Replace("\r\n", "\n").Replace('\r', '\n');
		bool flag = false;
		RawShip rawShip = null;
		string[] array = text.Split('\n');
		for (int i = 0; i < array.Length; i++)
		{
			string line = array[i].Trim();
			if (line.Length == 0)
			{
				continue;
			}
			if (TryReadHeader(line, "Name", out var value))
			{
				parsed.Name = value;
			}
			else if (TryReadHeader(line, "Faction", out value))
			{
				parsed.Faction = value;
			}
			else if (TryReadHeader(line, "Commander", out value))
			{
				if (TryParsePointedCard(value, out var name, out var points))
				{
					parsed.HeaderCommander = new RawUpgrade
					{
						Name = name,
						Points = points
					};
				}
				else if (!string.IsNullOrWhiteSpace(value))
				{
					parsed.HeaderCommander = new RawUpgrade
					{
						Name = RemoveSlotPrefix(value),
						Points = 0
					};
				}
			}
			else if (TryReadHeader(line, "Total Points", out value) || TryReadHeader(line, "Total Fleet Points", out value) || TryReadHeader(line, "Points", out value))
			{
				if (TryReadFirstInteger(value, out var number))
				{
					parsed.DeclaredPoints = number;
				}
			}
			else
			{
				if (TryReadHeader(line, "Gamemode", out var _))
				{
					continue;
				}
				Match match = ObjectivePattern.Match(line);
				if (match.Success)
				{
					result.Ignore("OBJECTIVE", match.Groups["name"].Value.Trim(), "Objectives belong to the selected game scenario and are not stored in fleet rosters.");
				}
				else if (IsAggregateSectionHeader(line, "Squadrons"))
				{
					flag = true;
					rawShip = null;
				}
				else if (IsAggregateSectionHeader(line, "Ships"))
				{
					flag = false;
					rawShip = null;
				}
				else if (ShipGroupHeaderPattern.IsMatch(line))
				{
					flag = false;
					rawShip = null;
				}
				else
				{
					if (SubtotalPattern.IsMatch(line) || line.StartsWith("Outdated printings owned:", StringComparison.OrdinalIgnoreCase))
					{
						continue;
					}
					bool flag2 = StripBullet(ref line);
					if (!TryParsePointedCard(line, out var name2, out var points2))
					{
						continue;
					}
					if (flag)
					{
						int count = 1;
						Match match2 = SquadronCountPattern.Match(name2);
						if (match2.Success)
						{
							count = Math.Max(1, int.Parse(match2.Groups["count"].Value, CultureInfo.InvariantCulture));
							name2 = match2.Groups["name"].Value.Trim();
						}
						parsed.Squadrons.Add(new RawSquadron
						{
							Name = name2,
							Count = count,
							TotalPoints = points2
						});
					}
					else if (flag2 && rawShip != null)
					{
						rawShip.Upgrades.Add(new RawUpgrade
						{
							Name = RemoveSlotPrefix(name2),
							Points = points2
						});
					}
					else
					{
						rawShip = new RawShip
						{
							Name = name2,
							Points = points2
						};
						parsed.Ships.Add(rawShip);
					}
				}
			}
		}
		if (parsed.HeaderCommander != null && parsed.Ships.Count > 0 && !parsed.Ships.Any((RawShip ship) => ship.Upgrades.Any((RawUpgrade upgrade) => SameCardName(upgrade.Name, parsed.HeaderCommander.Name))))
		{
			parsed.Ships[0].Upgrades.Insert(0, parsed.HeaderCommander);
		}
		return parsed;
	}

	private void AddShips(ParsedFleet parsed, FleetDraft draft, FleetImportResult result, ref string faction)
	{
		foreach (RawShip ship in parsed.Ships)
		{
			bool ambiguity;
			ShipContentDescriptor shipContentDescriptor = FindShip(ship.Name, ship.Points, out ambiguity);
			if (shipContentDescriptor == null)
			{
				string reason = (ambiguity ? "The name matches multiple in-game ships and the exported cost did not identify one safely." : "The ship is not available in this game's current official faction catalog.");
				result.Ignore("SHIP", ship.Name, reason);
				IgnoreChildUpgrades(ship, result);
				continue;
			}
			if (faction.Length == 0)
			{
				faction = NormalizeSupportedFaction(shipContentDescriptor.Faction);
			}
			if (!SameFaction(faction, shipContentDescriptor.Faction))
			{
				result.Ignore("SHIP", ship.Name, "The ship belongs to " + shipContentDescriptor.Faction + ", not the imported " + faction + " fleet.");
				IgnoreChildUpgrades(ship, result);
				continue;
			}
			draft.Document.faction = faction;
			ShipBuildDto build = draft.AddShip(shipContentDescriptor.Id);
			result.ImportedShipCount++;
			foreach (RawUpgrade upgrade in ship.Upgrades)
			{
				AddUpgrade(draft, build, upgrade, faction, result);
			}
		}
	}

	private void AddUpgrade(FleetDraft draft, ShipBuildDto build, RawUpgrade raw, string faction, FleetImportResult result)
	{
		bool ambiguity;
		bool factionMismatch;
		UpgradeContentDescriptor upgradeContentDescriptor = FindUpgrade(raw.Name, raw.Points, faction, out ambiguity, out factionMismatch);
		if (upgradeContentDescriptor == null)
		{
			result.Ignore("UPGRADE", raw.Name, factionMismatch ? "The card is not legal for the imported fleet's faction." : (ambiguity ? "The name matches multiple in-game upgrades and could not be resolved safely." : "The upgrade is not available in the current in-game catalog."));
			return;
		}
		if (upgradeContentDescriptor.OfficialDisposition != OfficialContentDisposition.StrictOfficial)
		{
			result.Ignore("UPGRADE", upgradeContentDescriptor.DisplayName, "This card is not in the strict official content set.");
			return;
		}
		if (upgradeContentDescriptor.GameplayCoverage != GameplayCoverage.Implemented)
		{
			result.Ignore("UPGRADE", upgradeContentDescriptor.DisplayName, "Its printed gameplay effect is not implemented yet.");
			return;
		}
		FleetLegalityReport baselineReport = upgradeProbe.Validate(draft.Document);
		if (!FleetUpgradeTrialValidator.TryExplainAddUpgrade(draft.Document, build.instanceId, upgradeContentDescriptor.Id, upgradeProbe, baselineReport, out var reason))
		{
			result.Ignore("UPGRADE", upgradeContentDescriptor.DisplayName, string.IsNullOrWhiteSpace(reason) ? "The card is not legal on this ship." : reason);
			return;
		}
		draft.ToggleUpgrade(build.instanceId, upgradeContentDescriptor.Id);
		result.ImportedUpgradeCount++;
	}

	private void AddSquadrons(ParsedFleet parsed, FleetDraft draft, FleetImportResult result, ref string faction)
	{
		foreach (RawSquadron squadron in parsed.Squadrons)
		{
			int importedPoints = ((squadron.Count > 0 && squadron.TotalPoints % squadron.Count == 0) ? (squadron.TotalPoints / squadron.Count) : squadron.TotalPoints);
			bool ambiguity;
			SquadronContentDescriptor squadronContentDescriptor = FindSquadron(squadron.Name, importedPoints, out ambiguity);
			if (squadronContentDescriptor == null)
			{
				result.Ignore("SQUADRON", QuantityLabel(squadron.Count, squadron.Name), ambiguity ? "The ace name matches multiple in-game squadrons and the exported cost did not identify one safely." : "The squadron is not available in this game's current official faction catalog.");
				continue;
			}
			if (faction.Length == 0)
			{
				faction = NormalizeSupportedFaction(squadronContentDescriptor.Faction);
			}
			if (!SameFaction(faction, squadronContentDescriptor.Faction))
			{
				result.Ignore("SQUADRON", QuantityLabel(squadron.Count, squadron.Name), "The squadron belongs to " + squadronContentDescriptor.Faction + ", not the imported " + faction + " fleet.");
				continue;
			}
			draft.Document.faction = faction;
			for (int i = 0; i < squadron.Count; i++)
			{
				if (squadronContentDescriptor.IsUnique && i > 0)
				{
					result.Ignore("SQUADRON", squadronContentDescriptor.DisplayName, "Extra copies of a unique ace are not legal.");
					continue;
				}
				draft.AddSquadron(squadronContentDescriptor.Id);
				result.ImportedSquadronCount++;
			}
		}
	}

	private ShipContentDescriptor FindShip(string rawName, int importedPoints, out bool ambiguity)
	{
		HashSet<string> candidates = CandidateNames(rawName);
		return SelectByPoints((from value in content.Ships.Where((ShipContentDescriptor value) => MatchesShipDescriptor(candidates, value)).GroupBy((ShipContentDescriptor value) => value.Id, StringComparer.OrdinalIgnoreCase)
			select value.First()).ToList(), importedPoints, (ShipContentDescriptor value) => value.Points, out ambiguity);
	}

	private SquadronContentDescriptor FindSquadron(string rawName, int importedPoints, out bool ambiguity)
	{
		HashSet<string> candidates = CandidateNames(rawName);
		return SelectByPoints((from value in content.Squadrons.Where((SquadronContentDescriptor value) => MatchesDescriptor(candidates, value.Id, value.DisplayName, FleetContentKind.Squadron)).GroupBy((SquadronContentDescriptor value) => value.Id, StringComparer.OrdinalIgnoreCase)
			select value.First()).ToList(), importedPoints, (SquadronContentDescriptor value) => value.Points, out ambiguity);
	}

	private UpgradeContentDescriptor FindUpgrade(string rawName, int importedPoints, string faction, out bool ambiguity, out bool factionMismatch)
	{
		HashSet<string> candidates = CandidateNames(rawName);
		List<UpgradeContentDescriptor> list = (from value in content.Upgrades.Where((UpgradeContentDescriptor value) => MatchesDescriptor(candidates, value.Id, value.DisplayName, FleetContentKind.Upgrade)).GroupBy((UpgradeContentDescriptor value) => value.Id, StringComparer.OrdinalIgnoreCase)
			select value.First()).ToList();
		List<UpgradeContentDescriptor> list2 = list.Where((UpgradeContentDescriptor value) => IsFactionAllowed(value.Factions, faction)).ToList();
		factionMismatch = list.Count > 0 && list2.Count == 0;
		return SelectByPoints(list2, importedPoints, (UpgradeContentDescriptor value) => value.Points, out ambiguity);
	}

	private bool MatchesDescriptor(HashSet<string> candidates, string id, string displayName, FleetContentKind kind)
	{
		if (candidates.Contains(NormalizeCardName(id)) || candidates.Contains(NormalizeCardName(displayName)))
		{
			return true;
		}
		foreach (FleetContentAlias alias in content.Aliases)
		{
			if (alias.Kind == kind && string.Equals(alias.CanonicalId, id, StringComparison.OrdinalIgnoreCase) && candidates.Contains(NormalizeCardName(alias.AliasId)))
			{
				return true;
			}
		}
		return false;
	}

	private bool MatchesShipDescriptor(HashSet<string> candidates, ShipContentDescriptor ship)
	{
		if (MatchesDescriptor(candidates, ship.Id, ship.DisplayName, FleetContentKind.Ship))
		{
			return true;
		}
		string relaxedId = RelaxShipName(NormalizeCardName(ship.Id));
		string relaxedDisplay = RelaxShipName(NormalizeCardName(ship.DisplayName));
		return candidates.Any(delegate(string candidate)
		{
			string text = RelaxShipName(candidate);
			return text.Length > 2 && (text == relaxedId || text == relaxedDisplay);
		});
	}

	private static T SelectByPoints<T>(List<T> matches, int importedPoints, Func<T, int> points, out bool ambiguity) where T : class
	{
		ambiguity = false;
		if (matches.Count == 0)
		{
			return null;
		}
		if (matches.Count == 1)
		{
			return matches[0];
		}
		List<T> list = matches.Where((T value) => points(value) == importedPoints).ToList();
		if (list.Count == 1)
		{
			return list[0];
		}
		ambiguity = true;
		return null;
	}

	private static HashSet<string> CandidateNames(string rawName)
	{
		HashSet<string> hashSet = new HashSet<string>(StringComparer.Ordinal);
		AddCandidate(hashSet, rawName);
		string text = StripContentTags(rawName);
		AddCandidate(hashSet, Regex.Replace(text, "\\bSquadrons\\b", "Squadron", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant));
		AddCandidate(hashSet, Regex.Replace(text, "\\bShuttles\\b", "Shuttle", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant));
		int num = text.IndexOf(':');
		if (num >= 0 && num + 1 < text.Length)
		{
			AddCandidate(hashSet, text.Substring(num + 1));
		}
		string[] array = Regex.Split(text, "\\s+[\\-\\u2013\\u2014]\\s+");
		if (array.Length > 1)
		{
			string[] array2 = array;
			foreach (string value in array2)
			{
				AddCandidate(hashSet, value);
			}
		}
		Match match = Regex.Match(text, "^(?<name>.+?)\\s*\\((?<type>[^)]+)\\)\\s*$");
		if (match.Success)
		{
			AddCandidate(hashSet, match.Groups["name"].Value);
			AddCandidate(hashSet, match.Groups["type"].Value);
		}
		return hashSet;
	}

	private static void AddCandidate(HashSet<string> candidates, string value)
	{
		string text = NormalizeCardName(value);
		if (text.Length > 0)
		{
			candidates.Add(text);
		}
	}

	private static string NormalizeCardName(string value)
	{
		string text = StripContentTags(value).Replace('‘', '\'').Replace('’', '\'').Replace('“', '"')
			.Replace('”', '"')
			.Normalize(NormalizationForm.FormD);
		StringBuilder stringBuilder = new StringBuilder(text.Length);
		bool flag = true;
		string text2 = text;
		foreach (char c in text2)
		{
			if (CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark)
			{
				if (char.IsLetterOrDigit(c))
				{
					stringBuilder.Append(char.ToLowerInvariant(c));
					flag = false;
				}
				else if (!flag)
				{
					stringBuilder.Append(' ');
					flag = true;
				}
			}
		}
		string text3 = stringBuilder.ToString().Trim();
		if (!(text3 == "ray antilles"))
		{
			return text3;
		}
		return "raymus antilles";
	}

	private static string RelaxShipName(string normalized)
	{
		List<string> list = (from value in (normalized ?? string.Empty).Split(new char[1] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(delegate(string value)
			{
				switch (value)
				{
				case "i":
					return "1";
				case "ii":
				case "mk2":
					return "2";
				case "iii":
				case "mk3":
					return "3";
				default:
					return value;
				}
			})
			where value != "class" && value != "mk" && value != "mark" && value != "star" && value != "destroyer" && value != "corvette" && value != "cruiser" && value != "carrier"
			select value).ToList();
		if (list.Count > 2 && list[0] == "imperial" && list.Contains("refit"))
		{
			list.RemoveAt(0);
		}
		return string.Join(" ", list);
	}

	private static bool TryReadHeader(string line, string header, out string value)
	{
		string text = header + ":";
		if (line.StartsWith(text, StringComparison.OrdinalIgnoreCase))
		{
			value = line.Substring(text.Length).Trim();
			return true;
		}
		value = string.Empty;
		return false;
	}

	private static bool IsAggregateSectionHeader(string line, string header)
	{
		if (string.Equals(line, header, StringComparison.OrdinalIgnoreCase) || string.Equals(line, header + ":", StringComparison.OrdinalIgnoreCase))
		{
			return true;
		}
		if (TryReadHeader(line, header, out var value))
		{
			return SectionPointsPattern.IsMatch(value);
		}
		return false;
	}

	private static bool TryParsePointedCard(string value, out string name, out int points)
	{
		Match match = CardPointsPattern.Match(StripContentTags(value));
		if (!match.Success)
		{
			name = string.Empty;
			points = 0;
			return false;
		}
		name = match.Groups["name"].Value.Trim();
		points = int.Parse(match.Groups["points"].Value, CultureInfo.InvariantCulture);
		return name.Length > 0;
	}

	private static bool TryReadFirstInteger(string value, out int number)
	{
		Match match = Regex.Match(value ?? string.Empty, "\\d+");
		return int.TryParse(match.Success ? match.Value : string.Empty, NumberStyles.None, CultureInfo.InvariantCulture, out number);
	}

	private static bool StripBullet(ref string line)
	{
		if (line.StartsWith("â€¢", StringComparison.Ordinal))
		{
			line = line.Substring(3).TrimStart();
			return true;
		}
		if (line.Length == 0 || (line[0] != '•' && line[0] != '·' && line[0] != '-' && line[0] != '='))
		{
			return false;
		}
		line = line.Substring(1).TrimStart();
		return true;
	}

	private static string RemoveSlotPrefix(string value)
	{
		int num = value.IndexOf(':');
		if (num < 0 || num + 1 >= value.Length)
		{
			return value.Trim();
		}
		return value.Substring(num + 1).Trim();
	}

	private static string StripContentTags(string value)
	{
		return ContentTagPattern.Replace(value?.Trim() ?? string.Empty, " ").Trim();
	}

	private static bool SameCardName(string left, string right)
	{
		return string.Equals(NormalizeCardName(left), NormalizeCardName(right), StringComparison.Ordinal);
	}

	private static string NormalizeSupportedFaction(string value)
	{
		string text = ShipFactionCatalog.NormalizeId(value);
		if (!ShipFactionCatalog.TryParse(text, out var _))
		{
			return string.Empty;
		}
		return text;
	}

	private static bool SameFaction(string left, string right)
	{
		return string.Equals(NormalizeSupportedFaction(left), NormalizeSupportedFaction(right), StringComparison.OrdinalIgnoreCase);
	}

	private static bool IsFactionAllowed(IReadOnlyList<string> allowed, string faction)
	{
		if (allowed == null || allowed.Count == 0)
		{
			return true;
		}
		return allowed.Any((string value) => string.IsNullOrWhiteSpace(value) || SameFaction(value, faction));
	}

	private static string QuantityLabel(int count, string name)
	{
		if (count <= 1)
		{
			return name;
		}
		return $"{count} x {name}";
	}

	private static void IgnoreChildUpgrades(RawShip ship, FleetImportResult result)
	{
		foreach (RawUpgrade upgrade in ship.Upgrades)
		{
			result.Ignore("UPGRADE", upgrade.Name, "Its parent ship was not imported.");
		}
	}
}
