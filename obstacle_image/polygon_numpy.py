import cv2
import numpy as np

def get_polygon_from_png(image_path: str, simplification_factor: float = 0.01) -> np.ndarray:
    """
    Extracts polygon coordinates from a transparent PNG and
    simplifies the resulting contour.

    Args:
        image_path: The file path to the PNG image.
        simplification_factor: A factor to control the "resolution".
            0.0 = high-res, 0.05 = very low-res.
            It's a percentage of the polygon's perimeter.

    Returns:
        A (simplified) NumPy array of shape (n, 2) as floats.
    """
    
    # 1. Load the image with alpha
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")
    if image.shape[2] < 4:
        raise ValueError("Image does not have an alpha channel.")

    # 2. Create binary mask from alpha
    alpha_channel = image[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)

    # 3. Find contours
    # We must use CHAIN_APPROX_SIMPLE here for approxPolyDP to work well
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No polygon found in the image.")
    
    # Get the single largest contour
    polygon_contour = contours[0]

    # --- 🚀 NEW SIMPLIFICATION STEP ---
    if simplification_factor > 0:
        # 4. Calculate the perimeter
        # True = it's a closed contour
        perimeter = cv2.arcLength(polygon_contour, True)
        
        # 5. Calculate epsilon (the "max distance")
        epsilon = simplification_factor * perimeter
        
        # 6. Get the simplified contour
        simplified_contour = cv2.approxPolyDP(polygon_contour, epsilon, True)
    else:
        # Skip simplification if factor is 0
        simplified_contour = polygon_contour
    # --- End of new step ---

    # 7. Squeeze, convert, and return the *simplified* contour
    polygon_array = simplified_contour.squeeze(axis=1).astype(float)
    
    if polygon_array.ndim == 1:
        polygon_array = polygon_array.reshape(1, 2)
        
    return polygon_array
def save_polygon_visualization(coordinates: np.ndarray, output_filename: str):
    """Draws the polygon coordinates onto a blank image and saves it."""
    
    # 1. Create a new black image (100x100, 3-channel for color)
    vis_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 2. Convert float coordinates to int32 for drawing
    # cv2.polylines expects a list of contours
    int_coords = [coordinates.astype(np.int32)]

    # 3. Draw the polygon
    # - vis_image: The canvas
    # - int_coords: Your polygon points
    # - isClosed=True: Connects the last point to the first
    # - color=(0, 255, 0): Green color (BGR format)
    # - thickness=1: 1-pixel line
    cv2.polylines(
        img=vis_image,
        pts=int_coords,
        isClosed=True,
        color=(0, 255, 0), 
        thickness=1
    )

    # 4. Save the image
    cv2.imwrite(output_filename, vis_image)
    print(f"✅ Visualization saved to '{output_filename}'")


# --- Main execution ---
if __name__ == "__main__":
    
    try:
        # 1. Get your coordinates
        coords = get_polygon_from_png("obstacle/Asteroid3.png")
        print(f"Loaded {coords.shape[0]} coordinates.")
        print(coords)
        # 2. Save the visualization
        save_polygon_visualization(coords, "obstacle/polygon_check.png")

    except (FileNotFoundError, ValueError) as e:
        print(f"❌ An error occurred: {e}")