from functools import lru_cache

import numpy as np
from shapely.geometry import Polygon
import numpy as np
from skimage.draw import polygon as draw_polygon
from skimage.measure import block_reduce

from enum_class import *
from measurement import *
import jit_geometry as jit

def delete_cache():
    """
    Clear the cache for all functions in this module.
    This is useful for testing purposes to ensure fresh calculations.
    """
    _ship_coordinate.cache_clear()
    attack_range_s2s.cache_clear()
    attack_range_s2q.cache_clear()
    attack_range_q2s.cache_clear()
    is_obstruct.cache_clear()
    is_overlap_s2s.cache_clear()
    is_overlap_s2q.cache_clear()
    distance_s2s.cache_clear()
    range_s2q.cache_clear()
    maneuver_tool.cache_clear()
    _ship_presence_plane.cache_clear()
    _squad_presence_plane.cache_clear()
    _threat_plane.cache_clear()

@lru_cache(maxsize=64000)
def _ship_coordinate(ship_state : tuple[str, float, float, float]) -> dict[str,np.ndarray] :

    ship_dict = SHIP_DATA[ship_state[0]]

    front_arc : tuple[float, float] = (ship_dict['front_arc_center'], ship_dict['front_arc_end']) 
    rear_arc : tuple[float, float] = (ship_dict['rear_arc_center'], ship_dict['rear_arc_end'])
    token_size : tuple[float, float] = SHIP_TOKEN_SIZE[SizeClass[ship_dict['size_class']]]
    token_half_w = token_size[0] / 2
    base_size : tuple[float, float] = SHIP_BASE_SIZE[SizeClass[ship_dict['size_class']]]
    base_half_w = base_size[0] / 2
    base_token_offset = (base_size[1]-token_size[1])/2

    squad_on_side = base_half_w + SQUAD_BASE_RADIUS
    squad_on_front = base_token_offset + SQUAD_BASE_RADIUS
    squad_on_rear = base_token_offset - base_size[1] - SQUAD_BASE_RADIUS

    template_vertices = np.array([
    [0, -front_arc[0]],                 # front_arc_center_pt 0
    [-token_half_w, -front_arc[1]],     # front_left_arc_pt 1
    [token_half_w, -front_arc[1]],      # front_right_arc_pt 2
    [0, -rear_arc[0]],                  # rear_arc_center_pt 3
    [-token_half_w, -rear_arc[1]],      # rear_left_arc_pt 4
    [token_half_w, -rear_arc[1]],       # rear_right_arc_pt 5
    [-token_half_w, 0],                 # front_left_token_pt 6
    [token_half_w, 0],                  # front_right_token_pt 7
    [token_half_w, -token_size[1]],     # rear_right_token_pt 8
    [-token_half_w, -token_size[1]],    # rear_left_token_pt 9

    [0, -ship_dict['front_targeting_point']],
    [ship_dict['side_targeting_point'][0], -ship_dict['side_targeting_point'][1]],
    [0, -ship_dict['rear_targeting_point']],
    [- ship_dict['side_targeting_point'][0], -ship_dict['side_targeting_point'][1]],

    [0, -token_size[1]/2],              # center point 14

    [-base_half_w, base_token_offset],                 # left front base 15
    [base_half_w, base_token_offset],                  # right front base 16
    [base_half_w, base_token_offset-base_size[1]],      # right rear base 17
    [-base_half_w, base_token_offset-base_size[1]],    # left rear base 18

    [ (base_half_w + TOOL_WIDTH_HALF), base_token_offset],  # right tool insert 19
    [-(base_half_w + TOOL_WIDTH_HALF), base_token_offset],   # left tool insert 20

    # overlapped squad placing position 21 ~ 42
    [-base_half_w,     squad_on_front], 
    [-base_half_w*0.5, squad_on_front],
    [ 0,               squad_on_front],
    [ base_half_w*0.5, squad_on_front],
    [ base_half_w,     squad_on_front],

    [ squad_on_side, base_token_offset],
    [ squad_on_side, base_token_offset - base_size[1]*0.2],
    [ squad_on_side, base_token_offset - base_size[1]*0.4],
    [ squad_on_side, base_token_offset - base_size[1]*0.6],
    [ squad_on_side, base_token_offset - base_size[1]*0.8],
    [ squad_on_side, base_token_offset - base_size[1]],

    [-base_half_w,     squad_on_rear], 
    [-base_half_w*0.5, squad_on_rear],
    [ 0,               squad_on_rear],
    [ base_half_w*0.5, squad_on_rear],
    [ base_half_w,     squad_on_rear],

    [-squad_on_side, base_token_offset],
    [-squad_on_side, base_token_offset - base_size[1]*0.2],
    [-squad_on_side, base_token_offset - base_size[1]*0.4],
    [-squad_on_side, base_token_offset - base_size[1]*0.6],
    [-squad_on_side, base_token_offset - base_size[1]*0.8],
    [-squad_on_side, base_token_offset - base_size[1]],

    ], dtype=np.float32)

    # CW rotation matrix
    c, s = np.cos(-ship_state[3]), np.sin(-ship_state[3])
    rotation_matrix = np.array([[c, -s], [s, c]], dtype=np.float32)
    translation_vector = np.array([ship_state[1], ship_state[2]], dtype=np.float32)

    # Rotate each vertex by applying the transpose of the rotation matrix
    # to the (N,2) vertices array, then translate by (2,) vector.
    current_vertices =  template_vertices @ rotation_matrix.T + translation_vector
    return {
        'arc_points' : current_vertices[0:10],
        'targeting_points' : current_vertices[10:14],
        'center_point' : current_vertices[14],
        'token_corners' : current_vertices[6:10],
        'base_corners' : current_vertices[15:19],
        'tool_insert_points' : current_vertices[19:21],
        'squad_placement_points' : current_vertices[21:43], 
    }

@lru_cache(maxsize=64000)
def attack_range_s2s(attacker_state, defender_state, extension_factor=500):
    """
    High-performance version of attack_range_s2s using Numba.
    This function prepares data and calls the jitted core function.
    """
    # 1. Prepare data in Python (unchanged)
    attacker_coords = _ship_coordinate(attacker_state)
    defender_coords = _ship_coordinate(defender_state)

    attacker_orientation_vector = np.array([np.sin(attacker_state[3]), np.cos(attacker_state[3])], dtype=np.float32) 

    # 2. Create NumPy representations of hulls
    attacker_hulls = jit.create_hull_arrays(attacker_coords['arc_points'])
    defender_hulls = jit.create_hull_arrays(defender_coords['arc_points'])

    # 3. Call the Numba JIT function with primitive types
    target_results, measure_results = jit.attack_range_s2s_numba(
        attacker_coords['arc_points'], attacker_coords['targeting_points'], attacker_orientation_vector,
        defender_coords['arc_points'], defender_coords['targeting_points'],
        attacker_hulls, defender_hulls,
        extension_factor
    )

    # 4. Convert results back to Python dictionaries and enums
    target_dict = {HullSection(i): bool(res) for i, res in enumerate(target_results)}
    measure_dict = {
        HullSection(r): {
            HullSection(c): AttackRange(val) for c, val in enumerate(row)
        } for r, row in enumerate(measure_results)
    }
    
    return target_dict, measure_dict


@lru_cache(maxsize=64000)
def attack_range_s2q(ship_state : tuple[str, float, float, float], squad_state : tuple[float, float], extension_factor=500) -> dict[HullSection, AttackRange]:
    """
    return:
        attack_range (AttackRange) for each hull section
    """

    ship_coords = _ship_coordinate(ship_state)
    squad_coords = np.array(squad_state, dtype=np.float32)
    attacker_orientation_vector : np.ndarray = np.array([np.sin(ship_state[3]), np.cos(ship_state[3])], dtype=np.float32)
    attacker_hulls = jit.create_hull_arrays(ship_coords['arc_points'])

    measure_results = jit.attack_range_s2q_numba(
        ship_coords['arc_points'], ship_coords['targeting_points'], attacker_orientation_vector,
        squad_coords,
        attacker_hulls,
        extension_factor
    )

    measure_dict = {
        HullSection(r): AttackRange(val) for r, val in enumerate(measure_results)
    }

    return measure_dict

@lru_cache(maxsize=64000)
def attack_range_q2s(squad_state : tuple[float, float], ship_state : tuple[str, float, float, float]) -> dict[HullSection, bool]:
    """
    return:
        in_range (bool) for each hull section
    """

    ship_coords = _ship_coordinate(ship_state)
    ship_poly : tuple[np.ndarray,...] = jit.create_hull_arrays(ship_coords['arc_points'])
    squad_coords : np.ndarray = np.array(squad_state, dtype=np.float32)

    measure_results = jit.attack_range_q2s_numba(squad_coords, ship_coords['targeting_points'], ship_poly)
    
    in_range_dict = {hull : bool(measure_results[hull]) for hull in HullSection}

    return in_range_dict


@lru_cache(maxsize=64000)
def is_obstruct(targeting_point : tuple[tuple[float, float], tuple[float, float]], ship_state : tuple[str, float, float, float]) -> bool :
    line_of_sight : np.ndarray = np.array(targeting_point, dtype=np.float32)

    ship_token : np.ndarray = np.array(_ship_coordinate(ship_state)['token_corners'], dtype=np.float32)

    return jit.SAT_overlapping_check(line_of_sight, ship_token)

@lru_cache(maxsize=64000)
def is_overlap_s2s(self_state : tuple[str, float, float, float], ship_state : tuple[str, float, float, float]) -> bool :
    self_coordinate = _ship_coordinate(self_state)['base_corners']
    other_coordinate = _ship_coordinate(ship_state)['base_corners']
    return jit.SAT_overlapping_check(self_coordinate, other_coordinate)

@lru_cache(maxsize=64000)
def is_overlap_s2q(ship_state : tuple[str, float, float, float], squad_state : tuple[float, float]) -> bool :
    ship_base : np.ndarray = _ship_coordinate(ship_state)['base_corners']
    squad_center: np.ndarray = np.array([squad_state], dtype=np.float32)
    if jit.SAT_overlapping_check(ship_base, squad_center) :
        return True

    return jit.polygon_polygon_nearest_points(ship_base, squad_center)[0] <= SQUAD_BASE_RADIUS

@lru_cache(maxsize=64000)
def distance_s2s(self_state : tuple[str, float, float, float], ship_state : tuple[str, float, float, float]) -> float :
    self_poly : np.ndarray = _ship_coordinate(self_state)['base_corners']
    ship_poly : np.ndarray = _ship_coordinate(ship_state)['base_corners']
    return jit.polygon_polygon_nearest_points(self_poly, ship_poly)[0]

@lru_cache(maxsize=64000)
def range_s2q(ship_state : tuple[str, float, float, float], squad_state : tuple[float, float]) -> AttackRange:
    """
    Returns:
        AttackRange : The best range from the ship to the squad, used for squad command activation
    """
    ship_token :np.ndarray = _ship_coordinate(ship_state)['token_corners']
    squad_center :np.ndarray = np.array([squad_state], dtype=np.float32)
    distance = jit.polygon_polygon_nearest_points(ship_token,squad_center)[0] - SQUAD_TOKEN_RADIUS

    return AttackRange(jit.distance_to_range(distance))                                                                                                                                                                                                                                                                        

@lru_cache(maxsize=64000)
def maneuver_tool(size_class :SizeClass, course : tuple[int, ...], placement : int) -> tuple[np.ndarray, float]:
    """
    translate maneuver tool coordination to ship coordination
    from ship.(x,y) to (x,y) after maneuver
    """
    base_size = np.array(SHIP_BASE_SIZE[size_class], dtype=np.float32)
    token_size = np.array(SHIP_TOKEN_SIZE[size_class], dtype=np.float32)
    return jit.maneuver_tool_numba(base_size, token_size, course, placement)

@lru_cache(maxsize=64000)
def _ship_presence_plane(
    ship_state: tuple[str, float, float, float],
    value: float,
    width_step: float,
    height_step: float,
    width_res: int,
    height_res: int,
) -> np.ndarray:
    """
    Encodes the ship's position and orientation into a fixed-size NumPy array.
    This encoding is suitable for use as input to machine learning models.
    """

    presence_plane = np.zeros((height_res, width_res), dtype=np.float32)
    scaled_vertices = np.array(_ship_coordinate(ship_state)['base_corners'], dtype=np.float32) / [width_step, height_step]
    rr, cc = draw_polygon(scaled_vertices[:, 1], scaled_vertices[:, 0], shape=presence_plane.shape)
    presence_plane[rr, cc] = value

    return presence_plane

@lru_cache(maxsize=64000)
def _squad_presence_plane(
    squad_state: tuple[float, float],
    value: float,
    width_step: float,
    height_step: float,
    width_res: int,
    height_res: int,
) -> np.ndarray:
    """
    Encodes the squad's position into a fixed-size NumPy array.
    This encoding is suitable for use as input to machine learning models.
    """
    # Create empty plane
    presence_plane = np.zeros((height_res, width_res), dtype=np.float32)

    # Convert world coordinates (x, y) to grid indices (col, row)
    scaled = np.array(squad_state, dtype=np.float32) / [width_step, height_step]
    col = int(scaled[0])
    row = int(scaled[1])

    # Clip to bounds and set a single pixel
    if 0 <= row < height_res and 0 <= col < width_res:
        presence_plane[row, col] = value
    else : raise ValueError(f"Squad position out of bounds\n{squad_state}, scaled: {scaled}, grid: ({col}, {row}), grid size: ({width_res}, {height_res})")

    return presence_plane

@lru_cache(maxsize=64000)
def _threat_plane(
    ship_state: tuple[str, float, float, float],
    width_step: float,
    height_step: float,
    width_res: int,
    height_res: int,
    extension_factor=500,
) -> np.ndarray:
    """
    Encodes the ship's threat area into two fixed-size NumPy arrays.
    This encoding is suitable for use as input to machine learning models.
    """
    # --- 1. Set up High-Resolution Grid ---
    # We will draw on a grid 4x the size (2x in each dimension)
    high_res_w = width_res * 2
    high_res_h = height_res * 2
    threat_planes_hr = np.zeros((12, high_res_h, high_res_w), dtype=np.float16)
    
    # Calculate the step size for the finer grid
    width_step_hr = width_step / 2
    height_step_hr = height_step / 2

    ship_token : Polygon = Polygon(_ship_coordinate(ship_state)['token_corners'])

    arc_coords = _ship_coordinate(ship_state)['arc_points']
    long_zone : Polygon = ship_token.buffer(LONG_RANGE)
    medium_zone : Polygon = ship_token.buffer(MEDIUM_RANGE)
    close_zone : Polygon = ship_token.buffer(CLOSE_RANGE)

    for hull in HullSection:
        # Get the correct arc coordinates for each hull section
        if hull in (HullSection.FRONT, HullSection.RIGHT):
            arc1_center, arc1_end = arc_coords[0], arc_coords[2]
        else:
            arc1_center, arc1_end = arc_coords[3], arc_coords[4]

        if hull in (HullSection.FRONT, HullSection.LEFT):
            arc2_center, arc2_end = arc_coords[0], arc_coords[1]
        else:
            arc2_center, arc2_end = arc_coords[3], arc_coords[5]

        # Build the arc polygon for this hull section
        vec1 = np.array(arc1_end, dtype=np.float32) - np.array(arc1_center, dtype=np.float32)
        vec2 = np.array(arc2_end, dtype=np.float32) - np.array(arc2_center, dtype=np.float32)
        arc_polygon = Polygon([
            arc1_end,
            np.array(arc1_end, dtype=np.float32) + vec1 * extension_factor,
            np.array(arc2_end, dtype=np.float32) + vec2 * extension_factor,
            arc2_end
        ])
        
        # Get threat zones for this hull section
        long_threat_zone = long_zone.intersection(arc_polygon)
        medium_threat_zone = medium_zone.intersection(arc_polygon)
        close_threat_zone = close_zone.intersection(arc_polygon)

        # Get battery data for this hull section
        battery_dict = SHIP_DATA[ship_state[0]]['battery']
        if hull == HullSection.LEFT:
            battery = battery_dict[1]  # Left uses right battery data
        else:
            battery = battery_dict[hull]

        # Extract dice counts for each range
        long_dice = battery[AttackRange.LONG] 
        medium_dice = battery[AttackRange.MEDIUM] 
        close_dice = battery[AttackRange.CLOSE] 

        # Helper function to safely draw polygon on threat plane
        def draw_threat_zone(threat_zone, dice_count, plane_layer):
            if dice_count > 0 and not threat_zone.is_empty:
                coords = np.array(threat_zone.exterior.coords, dtype=np.float32)
                scaled_verts = coords / [width_step, height_step]
                # Clip to valid bounds
                # --- 2. Scale vertices to the High-Resolution Grid ---
                scaled_verts = np.array(threat_zone.exterior.coords, dtype=np.float32) / [width_step_hr, height_step_hr]
                rr, cc = draw_polygon(scaled_verts[:, 1], scaled_verts[:, 0], shape=(high_res_h, high_res_w))
                threat_planes_hr[plane_layer, rr, cc] = dice_count


        # Draw threat zones for each range
        draw_threat_zone(long_threat_zone, long_dice, hull * 3 + AttackRange.LONG)
        draw_threat_zone(medium_threat_zone, medium_dice, hull * 3 + AttackRange.MEDIUM)
        draw_threat_zone(close_threat_zone, close_dice, hull * 3 + AttackRange.CLOSE)

    # --- 3. NEW LOGIC: Reduce and Combine ---
    # Create an empty array to hold the final threat map for each of the 4 hulls
    hull_threat_maps = np.zeros((4, height_res, width_res), dtype=np.float16)

    for hull in HullSection:
        # Select the 3 high-res range layers for the current hull
        start_index = hull * 3
        hull_range_layers_hr = threat_planes_hr[start_index : start_index + 3]

        # First, find the maximum threat across the 3 range bands for this hull
        max_threat_per_hull_hr = np.max(hull_range_layers_hr, axis=0)

    # Now, downsample this max-threat map to the final HxW resolution
    # We use np.max here again to ensure the strongest threat in any 2x2 block is preserved
    hull_threat_maps[hull] = block_reduce(max_threat_per_hull_hr, block_size=2, func=np.max)

    # --- 4. Final Summation ---
    # Sum the 4 final hull maps to get the total threat, correctly capturing double-arcing
    final_threat_map = np.sum(hull_threat_maps, axis=0) / 8  # Normalize to [0, 1] range

    return final_threat_map