import numpy as np
from enum_class import SizeClass


    
CLOSE_RANGE = 123.3
MEDIUM_RANGE = 186.5
LONG_RANGE = 304.8

DISTANCE = {i : distance for i, distance in enumerate((0, 77, 125, 185, 245, 304.8))} # check


SHIP_BASE_SIZE : dict[SizeClass, tuple]= {SizeClass.SMALL : (43, 71), SizeClass.MEDIUM : (63, 102), SizeClass.LARGE : (77.5, 129)}
SHIP_TOKEN_SIZE :  dict[SizeClass, tuple] = {SizeClass.SMALL : (38.5, 70.25), SizeClass.MEDIUM : (58.5, 101.5)}
TOOL_WIDTH_HALF : float= 15.25 / 2
TOOL_LENGTH : float= 46.13 
TOOL_PART_LENGTH : float = 22.27

SQUAD_BASE_RADIUS : float = 16.875 # check
SQUAD_TOKEN_RADIUS : float = 16 # check
SQUAD_RANGE = DISTANCE[1] + SQUAD_TOKEN_RADIUS * 2


def SAT_overlapping_check(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """
    Determines if two convex polygons or line segments intersect using the 
    Separating Axis Theorem (SAT).

    Args:
        poly1 (np.ndarray): A NumPy array of shape (N, 2) containing the 
                            (x, y) coordinates of the vertices of the first polygon.
        poly2 (np.ndarray): A NumPy array of shape (M, 2) containing the 
                            (x, y) coordinates of the vertices of the second polygon.

    Returns:
        bool: True if the shapes are colliding, False otherwise.
    """

    def get_axes(polygon: np.ndarray) -> np.ndarray:
        """
        Calculates the perpendicular axes (normals) for each edge of a polygon.
        Handles shapes with 2 or more vertices (lines or polygons).
        """
        axes = []
        # Loop through the vertices to form edges
        for i in range(len(polygon)):
            p1 = polygon[i]
            # The next vertex, wrapping around from the last to the first
            p2 = polygon[(i + 1) % len(polygon)]
            
            edge = p2 - p1
            
            # The perpendicular vector (normal) to the edge
            # For an edge (x, y), the normal is (-y, x)
            normal = np.array([-edge[1], edge[0]])
            
            # We only need to store the axis if it's not a zero vector
            # This handles cases of duplicate vertices.
            norm_magnitude = np.linalg.norm(normal)
            if norm_magnitude != 0:
                axes.append(normal / norm_magnitude)
        
        # For a line segment (2 vertices), the two normals will be opposites.
        # We can remove duplicates to be more efficient, though it's not required.
        # Using a trick with rounding to handle floating point inaccuracies.
        unique_axes = np.unique(np.round(axes, 8), axis=0)
        return unique_axes

    def project(polygon: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
        """
        Projects a polygon's vertices onto a given axis and returns the
        minimum and maximum projection values.
        """
        # The dot product of each vertex with the axis gives the projection
        projections = polygon.dot(axis)
        return np.min(projections), np.max(projections)

    # --- Main function logic ---
    
    # 1. Get all the unique axes to test from both polygons
    axes1 = get_axes(poly1)
    axes2 = get_axes(poly2)
    # If both inputs are single points, check if they are the same
    if len(axes1) == 0 and len(axes2) == 0:
        return np.array_equal(poly1, poly2)
        
    all_axes = np.concatenate((axes1, axes2))

    # 2. Loop through each axis
    for axis in all_axes:
        # 3. Project both polygons onto the current axis
        min1, max1 = project(poly1, axis)
        min2, max2 = project(poly2, axis)

        # 4. Check for a gap between the two projections
        # If max of projection 1 is less than min of projection 2, there's a gap.
        # If max of projection 2 is less than min of projection 1, there's a gap.
        if max1 < min2 or max2 < min1:
            # A separating axis is found, so the polygons do not collide.
            return False

    # 5. If no separating axis was found after checking all axes, the polygons must be colliding.
    return True

# --- Example Usage ---
if __name__ == '__main__':
    print("--- SAT Collision Detection Examples ---")

    # Example 1: Two squares colliding
    square1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    square2 = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
    print(f"\nColliding squares: {SAT_overlapping_check(square1, square2)}") # Expected: True

    # Example 2: Two squares not colliding
    square3 = np.array([[2, 2], [3, 2], [3, 3], [2, 3]])
    print(f"Non-colliding squares: {SAT_overlapping_check(square1, square3)}") # Expected: False

    # Example 3: A triangle and a square colliding
    triangle = np.array([[0.8, 0.8], [2, 1.5], [1.2, 2.5]])
    print(f"Colliding triangle and square: {SAT_overlapping_check(square2, triangle)}") # Expected: True

    # Example 4: A square and a line segment colliding
    line1 = np.array([[-0.5, 0.5], [0.5, -0.5]])
    print(f"Colliding square and line: {SAT_overlapping_check(square1, line1)}") # Expected: True
    
    # Example 5: A square and a line segment NOT colliding
    line2 = np.array([[-0.5, -0.5], [-0.1, -0.1]])
    print(f"Non-colliding square and line: {SAT_overlapping_check(square1, line2)}") # Expected: False
    
    # Example 6: Two line segments crossing
    line3 = np.array([[2, 0], [0, 2]])
    line4 = np.array([[0, 0], [2, 2]])
    print(f"Crossing line segments: {SAT_overlapping_check(line3, line4)}") # Expected: True