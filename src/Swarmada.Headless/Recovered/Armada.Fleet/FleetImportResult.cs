#nullable disable
using System.Collections.Generic;
using Armada.Session;

namespace Armada.Fleet;

public sealed class FleetImportResult
{
	private readonly List<FleetImportNotice> ignored = new List<FleetImportNotice>();

	public FleetListDto Fleet { get; internal set; }

	public string Error { get; internal set; } = string.Empty;

	public int? SourceDeclaredPoints { get; internal set; }

	public int RecalculatedPoints { get; internal set; }

	public int ImportedShipCount { get; internal set; }

	public int ImportedSquadronCount { get; internal set; }

	public int ImportedUpgradeCount { get; internal set; }

	public IReadOnlyList<FleetImportNotice> Ignored => ignored;

	public bool Success
	{
		get
		{
			if (Fleet != null)
			{
				return string.IsNullOrWhiteSpace(Error);
			}
			return false;
		}
	}

	internal void Ignore(string kind, string name, string reason)
	{
		ignored.Add(new FleetImportNotice(kind, name, reason));
	}
}
