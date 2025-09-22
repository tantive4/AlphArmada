# visualizer.py

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

def draw_polygon_to_file(polygon: Polygon, file_path: str = "polygon.png", board_size: tuple = (900, 900)):
    """
    Draws a single shapely Polygon to a PNG file with a transparent background.

    Args:
        polygon: The shapely Polygon to draw.
        file_path: The path to save the output image.
        board_size: A tuple representing the (width, height) of the board.
    """
    if polygon.is_empty:
        print("Warning: Attempted to draw an empty polygon.")
        return

    # Create a new image with a transparent background (RGBA)
    image = Image.new("RGBA", board_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Get the polygon's exterior coordinates
    # Pillow's polygon drawer requires a flat list of [(x1, y1), (x2, y2), ...]
    coords = polygon.exterior.coords

    # Draw the polygon onto the image
    draw.polygon(coords, fill="red")

    # Save the image
    image.save(file_path)
    print(f"Visualization saved to {file_path}")