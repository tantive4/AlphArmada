#nullable disable
using System;

namespace Armada.Data;

[Serializable]
public sealed class UpgradeClarification
{
	public string type;

	public string source;

	public string date;

	public string version;

	public string text;
}
