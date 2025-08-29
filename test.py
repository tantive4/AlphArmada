import numpy as np
from shapely.geometry import Polygon, LineString

def _get_axes(vertices):
    """
    Helper function to get all unique perpendicular axes from a shape's edges.
    This works for both convex polygons and single line segments.
    """
    axes = []
    
    # CORRECTED LOGIC: Handle 2-point line segments differently from closed polygons.
    if len(vertices) == 2:
        # For a single line segment, there is only one edge and one perpendicular axis.
        edge = vertices[1] - vertices[0]
        normal = np.array([-edge[1], edge[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-6:
            axes.append(normal / norm_len)
    else:
        # For polygons, iterate through all vertices to form edges
        for i in range(len(vertices)):
            p1 = vertices[i]
            # The next vertex is either the next in the list or the first one to close the loop
            p2 = vertices[(i + 1) % len(vertices)]
            
            edge = p2 - p1
            
            # The normal is a 90-degree rotation of the edge vector
            normal = np.array([-edge[1], edge[0]])
            
            # Normalize the axis vector
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-6:  # Avoid division by zero for zero-length edges
                axes.append(normal / norm_len)
            
    return np.array(axes)

def _project_shape_onto_axis(axis, vertices):
    """Projects all vertices of a shape onto a given axis and finds the min/max."""
    # Using dot product to project
    projections = np.dot(vertices, axis)
    return np.min(projections), np.max(projections)

def _sat_check(vertices1, vertices2):
    """
    Performs the core Separating Axis Theorem check between two convex shapes.
    """
    # Get the perpendicular axes from both shapes
    axes1 = _get_axes(vertices1)
    axes2 = _get_axes(vertices2)
    
    # If one of the shapes has no valid axes (e.g., it's a point), we can't check.
    if len(axes1) == 0 or len(axes2) == 0:
        return False # Or handle as a special case if points should collide.
        
    # Combine all axes
    all_axes = np.vstack((axes1, axes2))
    
    for axis in all_axes:
        # Project both shapes onto the current axis
        min1, max1 = _project_shape_onto_axis(axis, vertices1)
        min2, max2 = _project_shape_onto_axis(axis, vertices2)
        
        # Check for separation. If the "shadows" do not overlap, we've found
        # a separating axis, and the shapes do not intersect.
        if max1 < min2 or max2 < min1:
            return False  # They do not overlap

    # If we get through all axes without finding a separation, they must overlap.
    return True

def shapes_overlap(shape1_coords, shape2_coords):
    """
    Calculates if two convex shapes (Polygon or LineString) overlap.

    Accepts coordinates in formats like NumPy arrays or Shapely's `polygon.exterior.coords`.

    Args:
        shape1_coords: A list, NumPy array, or coordinate sequence for the first shape.
        shape2_coords: A list, NumPy array, or coordinate sequence for the second shape.

    Returns:
        bool: True if the shapes overlap, False otherwise.
    """
    # 1. Standardize inputs to NumPy arrays for consistent processing
    vertices1 = np.asarray(shape1_coords, dtype=np.float64)
    vertices2 = np.asarray(shape2_coords, dtype=np.float64)

    # Handle multi-segment LineStrings by checking each segment individually.
    # A shape with more than 2 vertices is assumed to be a convex polygon.
    is_multisegment_linestring1 = len(vertices1.shape) == 2 and np.array_equal(vertices1[0], vertices1[-1]) == False and len(vertices1) > 2
    is_multisegment_linestring2 = len(vertices2.shape) == 2 and np.array_equal(vertices2[0], vertices2[-1]) == False and len(vertices2) > 2

    if is_multisegment_linestring1:
        for i in range(len(vertices1) - 1):
            segment = vertices1[i:i+2]
            if shapes_overlap(segment, vertices2): # Recursive call
                return True
        return False
        
    if is_multisegment_linestring2:
        for i in range(len(vertices2) - 1):
            segment = vertices2[i:i+2]
            if shapes_overlap(vertices1, segment): # Recursive call
                return True
        return False

    # For convex polygons and single line segments, perform the SAT check.
    return _sat_check(vertices1, vertices2)

# --- Example Usage ---
if __name__ == '__main__':
    # --- Define Shapes for Testing ---
    # A square polygon
    poly1_verts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    
    # A triangular polygon that overlaps with the square
    poly2_verts = np.array([[5, 5], [15, 5], [15, 15]])
    
    # A polygon that does NOT overlap
    poly3_verts = np.array([[20, 20], [30, 20], [30, 30], [20, 30]])
    
    # A line segment that intersects the square
    line1_verts = np.array([[-5, 5], [5, 5]])
    
    # A line segment that does NOT intersect
    line2_verts = np.array([[-5, -5], [5, -5]])
    
    # A multi-segment LineString that crosses the square
    line3_verts = np.array([[1, -2], [1, 12], [8, 12]])

    print("--- Polygon vs. Polygon ---")
    print(f"Overlapping Polygons: {shapes_overlap(poly1_verts, poly2_verts)}")  # Expected: True
    print(f"Separate Polygons:    {shapes_overlap(poly1_verts, poly3_verts)}")  # Expected: False

    print("\n--- LineString vs. Polygon ---")
    print(f"Intersecting Line:    {shapes_overlap(poly1_verts, line1_verts)}")  # Expected: True
    print(f"Separate Line:        {shapes_overlap(poly1_verts, line2_verts)}")  # Expected: False
    print(f"Multi-segment Line:   {shapes_overlap(poly1_verts, line3_verts)}")  # Expected: True

    print("\n--- Using Shapely Coordinate Input ---")
    # Create Shapely objects to demonstrate input flexibility
    shapely_poly = Polygon(poly1_verts)
    shapely_line = LineString(line1_verts)
    
    # Pass the coordinate sequence directly from the Shapely object
    print(f"Shapely Poly vs Line: {shapes_overlap(shapely_poly.exterior.coords, shapely_line.coords)}") # Expected: True
