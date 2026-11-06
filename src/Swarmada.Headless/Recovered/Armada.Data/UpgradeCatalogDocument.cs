#nullable disable
#pragma warning disable CS0649 // Populated by System.Text.Json IncludeFields.
using System;

namespace Armada.Data;

[Serializable]
internal sealed class UpgradeCatalogDocument
{
	public int schemaVersion;

	public string ruleset;

	public string source;

	public string sourceSha256;

	public UpgradeCardDefinition[] cards;
}
