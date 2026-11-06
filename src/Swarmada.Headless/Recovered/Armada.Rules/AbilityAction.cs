#nullable disable
using Armada.Core;

namespace Armada.Rules;

public readonly struct AbilityAction
{
	public string AbilityId { get; }

	public string SourceId { get; }

	public string SourceName { get; }

	public AbilitySourceKind SourceKind { get; }

	public string Label { get; }

	public string Prompt { get; }

	public string RulesText { get; }

	public AbilityTiming Timing { get; }

	public bool Optional { get; }

	public bool Selected { get; }

	public RuntimeUnitId PreviewUnitId { get; }

	public AbilityAction(string abilityId, string sourceId, string sourceName, AbilitySourceKind sourceKind, string label, string prompt, string rulesText, AbilityTiming timing, bool optional = true, bool selected = false, RuntimeUnitId previewUnitId = default(RuntimeUnitId))
	{
		AbilityId = abilityId ?? string.Empty;
		SourceId = sourceId ?? string.Empty;
		SourceName = sourceName ?? string.Empty;
		SourceKind = sourceKind;
		Label = label ?? string.Empty;
		Prompt = prompt ?? string.Empty;
		RulesText = rulesText ?? string.Empty;
		Timing = timing;
		Optional = optional;
		Selected = selected;
		PreviewUnitId = previewUnitId;
	}
}
