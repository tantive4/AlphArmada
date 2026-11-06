#nullable disable
namespace Armada.Session;

public sealed class MapPlacementTemplate
{
	public string ContentId { get; }

	public string DisplayName { get; }

	public MapPlacementKind Kind { get; }

	public float WidthWorldUnits { get; }

	public float LengthWorldUnits { get; }

	public float InteractionRangeWorldUnits { get; }

	public string PreviewResourcePath { get; }

	public MapPlacementTemplate(string contentId, string displayName, MapPlacementKind kind, float widthWorldUnits, float lengthWorldUnits, float interactionRangeWorldUnits = 0f, string previewResourcePath = "")
	{
		ContentId = contentId;
		DisplayName = displayName;
		Kind = kind;
		WidthWorldUnits = widthWorldUnits;
		LengthWorldUnits = lengthWorldUnits;
		InteractionRangeWorldUnits = interactionRangeWorldUnits;
		PreviewResourcePath = previewResourcePath ?? string.Empty;
	}
}
