from shapely.geometry import Point, Polygon
import math

def is_polygon_in_extended_area(line1_pts, line2_pts, polygon_pts, extension_factor=1e4):
    """
    Checks if a polygon intersects with the area created by extending two line segments.

    Args:
        line1_pts (tuple): A tuple of two points defining the first line segment, e.g., ((x1, y1), (x2, y2)).
        line2_pts (tuple): A tuple of two points defining the second line segment.
        polygon_pts (list): A list of points defining the polygon vertices.
        extension_factor (float): A large number to extend the vectors for the quad.

    Returns:
        bool: True if the polygon intersects the extended area, False otherwise.
    """
    # Define the points from the input tuples
    a1 = Point(line1_pts[0])
    a2 = Point(line1_pts[1])
    b1 = Point(line2_pts[0])
    b2 = Point(line2_pts[1])

    # --- Vector Extension ---
    # Create vectors from a1 to a2 and b1 to b2
    vec_a = (a2.x - a1.x, a2.y - a1.y)
    vec_b = (b2.x - b1.x, b2.y - b1.y)

    # Calculate the extended points by scaling the vectors
    # This creates two new points very far away along the direction of the vectors
    extended_a2 = Point(a1.x + vec_a[0] * extension_factor, a1.y + vec_a[1] * extension_factor)
    extended_b2 = Point(b1.x + vec_b[0] * extension_factor, b1.y + vec_b[1] * extension_factor)

    # --- Create Geometries ---
    # Create the polygon to be checked
    polygon = Polygon(polygon_pts)

    # Create the large quadrilateral (quad) that represents the extended area.
    # The order of points is important to create a valid, non-self-intersecting polygon.
    # We use the original start points and the new, far-away extended points.
    large_quad = Polygon([a1, b1, extended_b2, extended_a2])

    # --- Intersection Check ---
    # Check if the polygon and the large quad intersect.
    # The .intersects() method returns True if any part of the geometries overlap.
    does_intersect = polygon.intersects(large_quad)

    return does_intersect
    


def in_arc(from_ship, from_hull, to_ship, to_hull) :
    if from_hull == 0 or from_hull == 1 :
        arc1 = (from_ship.get_coordination((0, - from_ship.front_arc[0])), from_ship.get_coordination((from_ship.size_dimension[0] / 2, - from_ship.front_arc[1])))
    else : 
        arc1 = (from_ship.get_coordination((0, - from_ship.rear_arc[0])), from_ship.get_coordination((- from_ship.size_dimension[0] / 2, - from_ship.rear_arc[1])))
    
    if from_hull == 1 or from_hull == 2 :
        arc2 = (from_ship.get_coordination((0, - from_ship.rear_arc[0])), from_ship.get_coordination((from_ship.size_dimension[0] / 2, - from_ship.rear_arc[1])))
    else :
        arc2 = (from_ship.get_coordination((0, - from_ship.front_arc[0])), from_ship.get_coordination((- from_ship.size_dimension[0] / 2, - from_ship.front_arc[1])))

    if to_hull == 0 :
        to_hull_polygon = (
            to_ship.get_coordination((to_ship.size_dimension[0] / 2, 0)),
            to_ship.get_coordination((to_ship.size_dimension[0] / 2, - to_ship.front_arc[1])),
            to_ship.get_coordination((0, - to_ship.front_arc[0])),
            to_ship.get_coordination((- to_ship.size_dimension[0] / 2, - to_ship.front_arc[1])),
            to_ship.get_coordination((-to_ship.size_dimension[0] / 2, 0)),
            )
    elif to_hull ==  1:
        to_hull_polygon = (
            to_ship.get_coordination((to_ship.size_dimension[0] / 2, - to_ship.front_arc[1])),
            to_ship.get_coordination((to_ship.size_dimension[0] / 2, - to_ship.rear_arc[1])),
            to_ship.get_coordination((0, - to_ship.rear_arc[0])),
            to_ship.get_coordination((0, - to_ship.front_arc[0]))
        )
    elif to_hull == 2:
        to_hull_polygon = (
            to_ship.get_coordination((to_ship.size_dimension[0] / 2, - to_ship.rear_arc[1])),
            to_ship.get_coordination((to_ship.size_dimension[0] / 2, - to_ship.size_dimension[1])),
            to_ship.get_coordination((- to_ship.size_dimension[0] / 2, - to_ship.size_dimension[1])),
            to_ship.get_coordination((- to_ship.size_dimension[0] / 2, - to_ship.rear_arc[1])),
            to_ship.get_coordination((0, - to_ship.rear_arc[0]))
        )
    elif to_hull == 3 :
        to_hull_polygon = (
            to_ship.get_coordination((- to_ship.size_dimension[0] / 2, - to_ship.front_arc[1])),
            to_ship.get_coordination((- to_ship.size_dimension[0] / 2, - to_ship.rear_arc[1])),
            to_ship.get_coordination((0, - to_ship.rear_arc[0])),
            to_ship.get_coordination((0, - to_ship.front_arc[0]))
        )
    
    return is_polygon_in_extended_area(arc1, arc2, to_hull_polygon)

def range(from_ship, from_hull, to_ship, to_hull):
    return 1 