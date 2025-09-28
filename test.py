import numpy as np
from typing import Tuple, List, Dict

# This is a placeholder for your existing function. It needs to be
# defined for the code to be runnable as a standalone example. In a real
# scenario, this would calculate the ship's geometry based on its state.
def _cached_coordinate(ship_state: Tuple[str, float, float, float]) -> Dict[str, List[List[float]]]:
    """
    Placeholder function to generate rectangle corners based on a ship's state.

    Args:
        ship_state: A tuple containing the ship's name, x-y coordinates, and rotation.

    Returns:
        A dictionary with the key 'token_corners' mapping to a list of corner points.
    """
    _, x, y, theta = ship_state
    center = np.array([x, y])
    # Example rectangle dimensions
    width = 4.0
    height = 2.0
    
    # Create rotation matrix from the angle (theta)
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    
    # Define the corners in local space (centered at origin)
    half_w, half_h = width / 2, height / 2
    local_corners = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h]
    ])
    
    # Transform corners to world space by rotating and translating them
    world_corners = (rot_matrix @ local_corners.T).T + center
    
    return {'token_corners': world_corners.tolist()}

def _cached_obstruction(
    targeting_point: Tuple[Tuple[float, float], Tuple[float, float]],
    ship_state: Tuple[str, float, float, float]
) -> bool:
    """
    Checks if a line segment intersects with a rotated rectangle using the
    Separating Axis Theorem (SAT). This function does not use the `shapely` library.

    Note: This implementation detects any kind of overlap (crossing, touching,
    or full containment). This is a common interpretation of "crossing" but differs
    from the strict topological definition provided by `shapely.crosses`, which
    can exclude simple touching or full containment.

    Args:
        targeting_point: A tuple containing the start and end points of the
                         line segment, e.g., ((x1, y1), (x2, y2)).
        ship_state: A tuple describing the ship's state, used by
                    _cached_coordinate to get the rectangle's corners.

    Returns:
        True if the line segment intersects the rectangle in any way, False otherwise.
    """
    # --- 1. Get geometry from inputs and convert to numpy arrays ---
    line_seg_vertices = np.array(targeting_point, dtype=float)
    try:
        # The rectangle's corners should be provided in a consistent order (e.g., clockwise)
        rect_vertices = np.array(_cached_coordinate(ship_state)['token_corners'], dtype=float)
    except (KeyError, TypeError):
        # Handle cases where _cached_coordinate might return invalid data
        return False

    # --- 2. Helper functions for the SAT algorithm ---

    def get_axes(vertices: np.ndarray) -> List[np.ndarray]:
        """Calculates the unique, normalized perpendicular axes for each edge."""
        axes = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            edge = p2 - p1
            
            # The normal is the perpendicular vector to the edge
            normal = np.array([-edge[1], edge[0]])
            
            # Normalize the axis vector, but skip if it's a zero vector (e.g., point)
            norm_val = np.linalg.norm(normal)
            if norm_val > 1e-6: # Use a small epsilon for floating point safety
                axes.append(normal / norm_val)
        return axes

    def project(vertices: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
        """Projects a shape's vertices onto an axis and returns the min/max projection."""
        # The dot product of each vertex with the axis gives the projection's position
        projections = vertices @ axis
        return np.min(projections), np.max(projections)

    # --- 3. Perform the Separating Axis Test ---

    # The potential separating axes are the normals of the edges of BOTH shapes.
    # A line segment is treated as a degenerate polygon with 2 vertices and 1 unique edge normal.
    # A rectangle has 4 edges but only 2 unique edge normals.
    axes = get_axes(rect_vertices) + get_axes(line_seg_vertices)
    if not axes:
        # This can happen if the line segment has zero length (start point == end point).
        # A single point cannot "cross" a polygon, so we can return False.
        return False

    for axis in axes:
        # Project both the rectangle and the line segment onto the current axis.
        min_rect, max_rect = project(rect_vertices, axis)
        min_line, max_line = project(line_seg_vertices, axis)

        # Check for a gap between the two projected ranges.
        # If there is a gap, we have found a separating axis, which means the shapes
        # do not intersect. We can exit early with a False result.
        if max_rect < min_line or max_line < min_rect:
            return False

    # If the loop completes without finding any separating axis, it means
    # the projections overlap on all axes. Therefore, the shapes must intersect.
    return True

# --- Example Usage for Verification ---
if __name__ == '__main__':
    # Define a ship (rectangle) centered at (10, 10), rotated by 45 degrees
    ship = ('ship_1', 10.0, 10.0, np.pi / 4)

    # Case 1: Line clearly crosses the rectangle
    crossing_line = ((0, 10), (20, 10))
    print(f"Line {crossing_line} intersects? {_cached_obstruction(crossing_line, ship)}")

    # Case 2: Line is completely outside, far away
    outside_line = ((0, 0), (5, 5))
    print(f"Line {outside_line} intersects? {_cached_obstruction(outside_line, ship)}")

    # Case 3: Line is completely inside the rectangle
    inside_line = ((9.5, 9.5), (10.5, 10.5))
    print(f"Line {inside_line} intersects? {_cached_obstruction(inside_line, ship)}")

    # Case 4: Line starts inside and ends outside
    touching_line = ((10, 10), (15, 15))
    print(f"Line {touching_line} intersects? {_cached_obstruction(touching_line, ship)}")

    # Case 5: Line is outside but its projection would overlap on some axes
    near_miss_line = ((5, 12), (8, 15))
    print(f"Line {near_miss_line} intersects? {_cached_obstruction(near_miss_line, ship)}")
    
    # Case 6: Line with zero length (a point) outside the rectangle
    point_line_outside = ((0, 0), (0, 0))
    print(f"Line {point_line_outside} intersects? {_cached_obstruction(point_line_outside, ship)}")
