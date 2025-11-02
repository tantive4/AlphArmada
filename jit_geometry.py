from __future__ import annotations
from typing import TYPE_CHECKING
import time

import numpy as np
import numba

from enum_class import *
from measurement import *
if TYPE_CHECKING:
    from armada import Armada

def pre_compile_jit_geometry(dummy_game: Armada):
    """Pre-compiles the Numba-jitted functions to avoid runtime overhead."""
    print("Pre-compiling JIT geometry functions...")
    start = time.time()
    dummy_game.rollout()
    end = time.time()
    print(f"Compiling Complete in {end - start:.4f} seconds.")

@numba.njit(cache=True)
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
        n = len(polygon)
        axes = np.empty((n, 2), dtype=np.float32)
        count = 0
        # Loop through the vertices to form edges
        for i in range(n):
            p1 = polygon[i]
            # The next vertex, wrapping around from the last to the first
            p2 = polygon[(i + 1) % n]

            edge = p2 - p1

            # The perpendicular vector (normal) to the edge
            # For an edge (x, y), the normal is (-y, x)
            normal = np.array([-edge[1], edge[0]], dtype=np.float32)

            # Store axis only if it's not a zero vector (handles duplicate vertices)
            norm_magnitude = np.linalg.norm(normal)
            if norm_magnitude != 0.0:
                axes[count, 0] = normal[0] / norm_magnitude
                axes[count, 1] = normal[1] / norm_magnitude
                count += 1

        # Return only the filled portion; duplicates are acceptable for SAT
        return axes[:count]

    def project(polygon: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
        """
        Projects a polygon's vertices onto a given axis and returns the
        minimum and maximum projection values.
        """
        # The dot product of each vertex with the axis gives the projection
        projections = polygon.dot(axis)
        return np.min(projections), np.max(projections)

    # --- Main function logic ---

    # Ensure inputs are float32 for Numba-supported linalg operations
    poly1 = poly1.astype(np.float32)
    poly2 = poly2.astype(np.float32)

    # 1. Get all the axes to test from both polygons
    axes1 = get_axes(poly1)
    axes2 = get_axes(poly2)
    # If both inputs are single points, check if they are the same
    if len(axes1) == 0 and len(axes2) == 0:
        return np.array_equal(poly1, poly2)

    all_axes = np.concatenate((axes1, axes2))

    # 2. Loop through each axis
    for axis in all_axes:
        # Ensure axis is contiguous for faster dot product under Numba
        axis = np.ascontiguousarray(axis)
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

@numba.njit(cache=True)
def segments_intersect(p1, q1, p2, q2):
    """Checks if line segment 'p1q1' and 'p2q2' intersect."""
    def _orientation(p, q, r):
        """Finds orientation of ordered triplet (p, q, r)."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - \
            (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise
    
    def _on_segment(p, q, r):
        """Given three collinear points p, q, r, checks if point q lies on segment pr."""
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    # Special Cases for collinear points
    if o1 == 0 and _on_segment(p1, p2, q1): return True
    if o2 == 0 and _on_segment(p1, q2, q1): return True
    if o3 == 0 and _on_segment(p2, p1, q2): return True
    if o4 == 0 and _on_segment(p2, q1, q2): return True

    return False

@numba.njit(cache=True)
def polygon_polygon_nearest_points(poly1:np.ndarray, poly2:np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Finds the minimum distance and nearest points between two polygons."""

    def _point_segment_distance_sq(p, a, b):
        """Calculates the minimum distance from a point p to a line segment ab."""
        ap = p - a
        ab = b - a
        ab_len_sq = ab[0]**2 + ab[1]**2
        if ab_len_sq == 0.0:
            return ap[0]**2 + ap[1]**2, a

        t = np.dot(ap, ab) / ab_len_sq
        if t < 0.0:
            closest_point = a
        elif t > 1.0:
            closest_point = b
        else:
            closest_point = a + t * ab
        distance = p - closest_point
        return distance[0]**2 + distance[1]**2, closest_point


    min_dist_sq = np.inf
    p1_nearest = poly1[0]
    p2_nearest = poly2[0]

    # Check all vertices of poly1 against all edges of poly2
    for i in range(poly1.shape[0]):
        p = poly1[i]
        for j in range(poly2.shape[0]):
            a = poly2[j]
            b = poly2[(j + 1) % poly2.shape[0]]
            dist_sq, closest_point = _point_segment_distance_sq(p, a, b)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                p1_nearest = p
                p2_nearest = closest_point

    # Check all vertices of poly2 against all edges of poly1
    for i in range(poly2.shape[0]):
        p = poly2[i]
        for j in range(poly1.shape[0]):
            a = poly1[j]
            b = poly1[(j + 1) % poly1.shape[0]]
            dist_sq, closest_point = _point_segment_distance_sq(p, a, b)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                p1_nearest = closest_point
                p2_nearest = p

    return np.sqrt(min_dist_sq), p1_nearest, p2_nearest

@numba.njit(cache=True)
def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    """Clips a polygon using the Sutherland-Hodgman algorithm."""
    # Pre-allocate a large buffer.
    max_size = len(subject_polygon) + len(clip_polygon)
    output_list = np.zeros((max_size, 2), dtype=np.float32) 
    
    # YOUR FIX: Pre-allocate the input buffer to be the same large size.
    input_list = np.zeros((max_size, 2), dtype=np.float32)
    input_len = len(subject_polygon)
    input_list[:input_len] = subject_polygon

    for i in range(len(clip_polygon)):
        clip_edge_p1 = clip_polygon[i]
        clip_edge_p2 = clip_polygon[(i + 1) % len(clip_polygon)]
        
        output_len = 0 # Reset output list for this clipping edge
        
        for j in range(input_len):
            current_point = input_list[j]
            prev_point = input_list[(j - 1 + input_len) % input_len]

            # ... (rest of the clipping logic is the same) ...
            clip_edge_vec = clip_edge_p2 - clip_edge_p1
            
            is_current_inside = (clip_edge_vec[0] * (current_point[1] - clip_edge_p1[1]) -
                                 clip_edge_vec[1] * (current_point[0] - clip_edge_p1[0])) >= 0
            
            is_prev_inside = (clip_edge_vec[0] * (prev_point[1] - clip_edge_p1[1]) -
                              clip_edge_vec[1] * (prev_point[0] - clip_edge_p1[0])) >= 0

            if is_current_inside != is_prev_inside:
                line1_vec = current_point - prev_point
                t_num = (clip_edge_p1[0] - prev_point[0]) * clip_edge_vec[1] - \
                        (clip_edge_p1[1] - prev_point[1]) * clip_edge_vec[0]
                t_den = line1_vec[0] * clip_edge_vec[1] - line1_vec[1] * clip_edge_vec[0]
                
                if t_den != 0:
                    t = t_num / t_den
                    intersection_point = prev_point + t * line1_vec
                    output_list[output_len] = intersection_point
                    output_len += 1

            if is_current_inside:
                output_list[output_len] = current_point
                output_len += 1
        
        if output_len == 0:
            return np.empty((0, 2), dtype=np.float32)

        # The result of this clip becomes the input for the next one.
        # This is now safe because both buffers are the same size.
        input_len = output_len
        input_list[:output_len] = output_list[:output_len]

    return output_list[:output_len]

@numba.njit(cache=True)
def create_hull_arrays(arc_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates NumPy arrays for each hull section."""
    # HullSection enum must be 0, 1, 2, 3
    front = np.stack((arc_coords[0], arc_coords[2], arc_coords[7], arc_coords[6], arc_coords[1]))
    right = np.stack((arc_coords[0], arc_coords[2], arc_coords[5], arc_coords[3]))
    rear = np.stack((arc_coords[3], arc_coords[5], arc_coords[8], arc_coords[9], arc_coords[4]))
    left = np.stack((arc_coords[0], arc_coords[1], arc_coords[4], arc_coords[3]))
    return front, right, rear, left

@numba.njit(cache=True)
def attack_range_s2s_numba(
    attacker_arc_pts, attacker_tgt_pts, attacker_orientation_vec,
    defender_arc_pts, defender_tgt_pts,
    attacker_hulls, defender_hulls,
    extension_factor,
):
    """
    Core Numba-jitted calculation loop. Operates only on NumPy arrays and primitives.
    """
    target_results = np.zeros(4, dtype=np.bool_)
    measure_results = np.full((4, 4), -1, dtype=np.int8)  # Using ints for AttackRange enum

    for from_hull_idx in range(4):
        for to_hull_idx in range(4):
            # 1. Attack hull orientation check (same as your original)
            attack_orientation_vector = ROTATION_MATRICES[from_hull_idx] @ attacker_orientation_vec
            hull_target_vector = defender_tgt_pts[to_hull_idx] - attacker_tgt_pts[from_hull_idx]
            if np.dot(attack_orientation_vector, hull_target_vector) < 0:
                continue

            # 2. Line of Sight blocked (targeting points)
            is_blocked = False
            p1 = attacker_tgt_pts[from_hull_idx]
            q1 = defender_tgt_pts[to_hull_idx]
            for hull_idx in range(4):
                if hull_idx != to_hull_idx:
                    defender_poly = defender_hulls[hull_idx]
                    for i in range(len(defender_poly)):
                        p2 = defender_poly[i]
                        q2 = defender_poly[(i + 1) % len(defender_poly)]
                        if segments_intersect(p1, q1, p2, q2):
                            is_blocked = True
                            break
                    if is_blocked: break
            if is_blocked: 
                continue

            # 3. Build the arc polygon
            if from_hull_idx == 0: # FRONT
                arc_p1_center = attacker_arc_pts[0]
                arc_p1 = attacker_arc_pts[1] # front_left
                arc_p2_center = attacker_arc_pts[0]
                arc_p2 = attacker_arc_pts[2] # front_right
            elif from_hull_idx == 1: # RIGHT
                arc_p1_center = attacker_arc_pts[0]
                arc_p1 = attacker_arc_pts[2] # front_right
                arc_p2_center = attacker_arc_pts[3]
                arc_p2 = attacker_arc_pts[5] # rear_right
            elif from_hull_idx == 2: # REAR
                arc_p1_center = attacker_arc_pts[3]
                arc_p1 = attacker_arc_pts[5] # rear_right
                arc_p2_center = attacker_arc_pts[3]
                arc_p2 = attacker_arc_pts[4] # rear_left
            else: # LEFT (from_hull_idx == 3)
                arc_p1_center = attacker_arc_pts[3]
                arc_p1 = attacker_arc_pts[4] # rear_left
                arc_p2_center = attacker_arc_pts[0]
                arc_p2 = attacker_arc_pts[1] # front_left

            vec1 = arc_p1 - arc_p1_center
            vec2 = arc_p2 - arc_p2_center
            
            # Construct polygon with a consistent counter-clockwise winding order
            p1 = arc_p1
            p2 = arc_p2
            p3 = arc_p2 + vec2 * extension_factor
            p4 = arc_p1 + vec1 * extension_factor
            
            # Use np.stack to combine the arrays into a single (4, 2) array
            arc_polygon = np.stack((p1, p2, p3, p4))
            
            # 4. Check if defender hull is in the arc by clipping
            to_hull_poly = defender_hulls[to_hull_idx]
            to_hull_in_arc = sutherland_hodgman_clip(to_hull_poly, arc_polygon)
            if len(to_hull_in_arc) < 3:
                continue # not in arc (result is not a valid polygon)
            
            # 5. Range measurement
            from_hull_poly = attacker_hulls[from_hull_idx]
            distance, p_attacker, p_defender = polygon_polygon_nearest_points(from_hull_poly, to_hull_in_arc)

            # 6. Final Line of Sight check for range measurement line
            is_blocked = False
            for hull_idx in range(4):
                if hull_idx == to_hull_idx:
                    continue
                obstructing_poly = defender_hulls[hull_idx]
                for i in range(len(obstructing_poly)):
                    p2 = obstructing_poly[i]
                    q2 = obstructing_poly[(i + 1) % len(obstructing_poly)]

                    # Check if the LoS endpoint is one of the vertices of the obstructing edge.
                    # If so, this isn't a "crossing", it's a "touch". Skip this edge.
                    dist_sq_p2 = (p_defender[0] - p2[0])**2 + (p_defender[1] - p2[1])**2
                    dist_sq_q2 = (p_defender[0] - q2[0])**2 + (p_defender[1] - q2[1])**2
                    if dist_sq_p2 < ERROR_EPSILON or dist_sq_q2 < ERROR_EPSILON:
                        continue
                    
                    if segments_intersect(p_attacker, p_defender, p2, q2):
                        is_blocked = True
                        break
                if is_blocked: break
            if is_blocked: 
                continue

            # 7. Set results based on distance
            attack_range = distance_to_range(distance)
            measure_results[from_hull_idx][to_hull_idx] = attack_range
            if attack_range < 3:
                target_results[from_hull_idx] = True

    return target_results, measure_results

@numba.njit(cache=True)
def attack_range_s2q_numba(
    attacker_arc_pts, attacker_tgt_pts, attacker_orientation_vec,
    squad_center,
    attacker_hulls, 
    extension_factor,
):
    measure_results = np.full(4, -1, dtype=np.int32)

    for from_hull_idx in range(4):
        # 1. Attack hull orientation check (same as your original)
        attack_orientation_vector = ROTATION_MATRICES[from_hull_idx] @ attacker_orientation_vec
        hull_target_vector = squad_center - attacker_tgt_pts[from_hull_idx]
        if np.dot(attack_orientation_vector, hull_target_vector) < 0:
            continue

        # 2. Build the arc polygon
        if from_hull_idx == 0: # FRONT
            arc_p1_center = attacker_arc_pts[0]
            arc_p1 = attacker_arc_pts[1] # front_left
            arc_p2_center = attacker_arc_pts[0]
            arc_p2 = attacker_arc_pts[2] # front_right
        elif from_hull_idx == 1: # RIGHT
            arc_p1_center = attacker_arc_pts[0]
            arc_p1 = attacker_arc_pts[2] # front_right
            arc_p2_center = attacker_arc_pts[3]
            arc_p2 = attacker_arc_pts[5] # rear_right
        elif from_hull_idx == 2: # REAR
            arc_p1_center = attacker_arc_pts[3]
            arc_p1 = attacker_arc_pts[5] # rear_right
            arc_p2_center = attacker_arc_pts[3]
            arc_p2 = attacker_arc_pts[4] # rear_left
        else: # LEFT (from_hull_idx == 3)
            arc_p1_center = attacker_arc_pts[3]
            arc_p1 = attacker_arc_pts[4] # rear_left
            arc_p2_center = attacker_arc_pts[0]
            arc_p2 = attacker_arc_pts[1] # front_left

        vec1 = arc_p1 - arc_p1_center
        vec2 = arc_p2 - arc_p2_center
        
        # Construct polygon with a consistent counter-clockwise winding order
        p1 = arc_p1
        p2 = arc_p2
        p3 = arc_p2 + vec2 * extension_factor
        p4 = arc_p1 + vec1 * extension_factor
        
        # Use np.stack to combine the arrays into a single (4, 2) array
        arc_polygon = np.stack((p1, p2, p3, p4))
        
        # 3. Check if defender hull is in the arc by clipping
        to_hull_poly = SQUAD_TOKEN_POLY + squad_center
        to_hull_in_arc = sutherland_hodgman_clip(to_hull_poly, arc_polygon)
        if len(to_hull_in_arc) < 3:
            continue # not in arc (result is not a valid polygon)
        
        # 4. Range measurement
        from_hull_poly = attacker_hulls[from_hull_idx]
        distance = polygon_polygon_nearest_points(from_hull_poly, to_hull_in_arc)[0]

        # 5. Set results based on distance
        attack_range = distance_to_range(distance)
        measure_results[from_hull_idx] = attack_range

    return measure_results

@numba.njit(cache=True)
def attack_range_q2s_numba(
    squad_center:np.ndarray,
    defender_tgt_pts:np.ndarray, defender_hulls:tuple[np.ndarray,...]
):
    squad_poly = squad_center.reshape(1, 2)
    target_results = np.zeros(4, dtype=np.bool_)
    for to_hull_idx in range(4):

        # 1. Line of Sight blocked (targeting points)
        p1 = squad_center
        q1 = defender_tgt_pts[to_hull_idx]

        is_blocked = False
        for hull_idx in range(4):
            if hull_idx == to_hull_idx: continue
            defender_poly = defender_hulls[hull_idx]
            for i in range(len(defender_poly)):
                p2 = defender_poly[i]
                q2 = defender_poly[(i + 1) % len(defender_poly)]
                if segments_intersect(p1, q1, p2, q2):
                    is_blocked = True
                    break
            if is_blocked: break
        if is_blocked: 
            continue

        # 2. Range measurement
        attack_range, p_attacker, p_defender = polygon_polygon_nearest_points(squad_poly, defender_hulls[to_hull_idx])
        is_blocked = False
        for hull_idx in range(4):
            if hull_idx == to_hull_idx:
                continue
            obstructing_poly = defender_hulls[hull_idx]
            for i in range(len(obstructing_poly)):
                p2 = obstructing_poly[i]
                q2 = obstructing_poly[(i + 1) % len(obstructing_poly)]

                # Check if the LoS endpoint is one of the vertices of the obstructing edge.
                # If so, this isn't a "crossing", it's a "touch". Skip this edge.
                dist_sq_p2 = (p_defender[0] - p2[0])**2 + (p_defender[1] - p2[1])**2
                dist_sq_q2 = (p_defender[0] - q2[0])**2 + (p_defender[1] - q2[1])**2
                if dist_sq_p2 < ERROR_EPSILON or dist_sq_q2 < ERROR_EPSILON:
                    continue
                
                if segments_intersect(p_attacker, p_defender, p2, q2):
                    is_blocked = True
                    break
            if is_blocked: break
        if is_blocked: 
            continue

        if attack_range < Q2S_RANGE :
            target_results[to_hull_idx] = True
    return target_results

@numba.njit(cache=True)
def maneuver_tool_numba(base_size : np.ndarray, token_size:np.ndarray, course:tuple[int,...], placement:int) -> tuple[np.ndarray, float]:
    if not course :
        return np.array([0.0, 0.0], dtype=np.float32), 0.0

    yaw_changes = np.array([0] + list(course), dtype=np.float32) * (np.pi / 8)
    joint_orientations = np.cumsum(yaw_changes)

    # --- Step 2: Get the final orientation directly ---
    # The final orientation is simply the last element of the cumulative sum.
    final_orientation = joint_orientations[-1]

    # --- Step 3: Calculate the total displacement vector ---
    # Get the orientations for the long and short segments of the path.
    long_segment_orientations = joint_orientations[:-1]
    short_segment_orientations = joint_orientations[1:]

    # Sum the x and y components of all segment vectors without storing them.
    # total_displacement_vector = sum(length * [sin(angle), cos(angle)])
    total_dx = np.sum(TOOL_LENGTH * np.sin(long_segment_orientations)) + \
               np.sum(TOOL_PART_LENGTH * np.sin(short_segment_orientations))
               
    total_dy = np.sum(TOOL_LENGTH * np.cos(long_segment_orientations)) + \
               np.sum(TOOL_PART_LENGTH * np.cos(short_segment_orientations))


    tool_offset = placement * (base_size[0]/2 + TOOL_WIDTH_HALF)
    ship_to_tool = np.array([tool_offset, (base_size[1] - token_size[1])/2], dtype=np.float32)
    c, s = np.cos(-final_orientation), np.sin(-final_orientation)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    tool_to_ship = rotation @ -ship_to_tool


    # --- Step 4: Calculate final position ---
    final_position = ship_to_tool + np.array([total_dx, total_dy], dtype=np.float32) + tool_to_ship
    return final_position, final_orientation

@numba.njit(cache=True)
def distance_to_range(distance: float) -> int:
    """Converts a distance to an AttackRange enum value."""
    if distance <= CLOSE_RANGE:
        return 0  # AttackRange.CLOSE
    elif distance <= MEDIUM_RANGE:
        return 1  # AttackRange.MEDIUM
    elif distance <= LONG_RANGE:
        return 2  # AttackRange.LONG
    else:
        return 3  # AttackRange.EXTREME
