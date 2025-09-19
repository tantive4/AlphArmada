import numpy as np

def calculate_final_position(length: float, orientation: float) -> np.ndarray:
    """
    Calculates the final position after two jumps.

    Args:
        length: The magnitude of the jump.
        orientation: The new orientation in radians after the first jump.

    Returns:
        A NumPy array containing the final [x, y] coordinates.
    """
    # 1. The first jump is from (0,0) to a point on the x-axis.
    position_after_first_jump = np.array([length, 0])

    # 2. Calculate the displacement vector for the second "left" jump.
    # Based on your example, the orientation angle is measured clockwise from the +y axis.
    # A "left" jump is perpendicular to that orientation.
    second_jump_displacement = np.array([
        -length * np.cos(orientation),
        length * np.sin(orientation)
    ])

    # 3. Add the first position and the second displacement to get the final position.
    final_position = position_after_first_jump + second_jump_displacement
    
    return final_position

# --- Your Example ---
LENGTH = 1.0
ORIENTATION = np.pi / 4  # 90 degrees

# Calculate and print the final position
final_coords = calculate_final_position(LENGTH, ORIENTATION)

print(f"For LENGTH = {LENGTH} and ORIENTATION = {np.rad2deg(ORIENTATION)} degrees:")
print(f"Final Position: {final_coords}") 
# Expected output for your example: [1. 1.]