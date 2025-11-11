from functools import lru_cache

import numpy as np
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
def _ship_coordinate(ship_state : tuple[str, int, int, int]) -> dict[str|tuple[int, int], np.ndarray] :

    template_vertices : np.ndarray = SHIP_TEMPLATE_POLY[ship_state[0]]
    threat_zone_vertices, split_index = SHIP_THREAT_ZONES[ship_state[0]]
    # CW rotation matrix
    orientation = ship_state[3]*HASH_PRECISION_INV
    c, s = np.cos(-orientation), np.sin(-orientation)
    rotation_matrix = np.array([[c, -s], [s, c]], dtype=np.float32)
    translation_vector = np.array([ship_state[1], ship_state[2]], dtype=np.float32)*HASH_PRECISION_INV

    # Rotate each vertex by applying the transpose of the rotation matrix
    # to the (N,2) vertices array, then translate by (2,) vector.
    current_vertices =  template_vertices @ rotation_matrix.T + translation_vector
    
    coord_dict : dict[str|tuple[int, int], np.ndarray]= {
        'arc_points' : current_vertices[0:10],
        'targeting_points' : current_vertices[10:14],
        'center_point' : current_vertices[14],
        'token_corners' : current_vertices[6:10],
        'base_corners' : current_vertices[15:19],
        'tool_insert_points' : current_vertices[19:21],
        'squad_placement_points' : current_vertices[21:43], 
        
    }
    current_threat_vertices = np.split(threat_zone_vertices @ rotation_matrix.T + translation_vector, split_index)
    for hull in HULL_SECTIONS:
        for attack_range in ATTACK_RANGES:
            coord_dict[(hull, attack_range)] = current_threat_vertices[hull*3 + attack_range]

    return coord_dict

@lru_cache(maxsize=64000)
def attack_range_s2s(attacker_state, defender_state, extension_factor=500) -> tuple[list[bool], list[list[int]]]:
    """
    High-performance version of attack_range_s2s using Numba.
    This function prepares data and calls the jitted core function.
    """
    # 1. Prepare data in Python (unchanged)
    attacker_coords = _ship_coordinate(attacker_state)
    defender_coords = _ship_coordinate(defender_state)

    attacker_orientation_vector = np.array([np.sin(attacker_state[3]*HASH_PRECISION_INV), np.cos(attacker_state[3]*HASH_PRECISION_INV)], dtype=np.float32)

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

    # 4. Convert results back to Python list
    target_list = target_results.tolist()
    measure_list = measure_results.tolist()
    
    return target_list, measure_list


@lru_cache(maxsize=64000)
def attack_range_s2q(ship_state : tuple[str, int, int, int], squad_state : tuple[int, int], extension_factor=500) -> list[int]:
    """
    return:
        attack_range (AttackRange) for each hull section
    """

    ship_coords = _ship_coordinate(ship_state)
    squad_coords = np.array(squad_state, dtype=np.float32)*HASH_PRECISION_INV
    attacker_orientation_vector : np.ndarray = np.array([np.sin(ship_state[3]*HASH_PRECISION_INV), np.cos(ship_state[3]*HASH_PRECISION_INV)], dtype=np.float32)
    attacker_hulls = jit.create_hull_arrays(ship_coords['arc_points'])

    measure_results = jit.attack_range_s2q_numba(
        ship_coords['arc_points'], ship_coords['targeting_points'], attacker_orientation_vector,
        squad_coords,
        attacker_hulls,
        extension_factor
    )

    measure_list = measure_results.tolist()

    return measure_list

@lru_cache(maxsize=64000)
def attack_range_q2s(squad_state : tuple[int, int], ship_state : tuple[str, int, int, int]) -> list[bool]:
    """
    return:
        in_range (bool) for each hull section
    """

    ship_coords = _ship_coordinate(ship_state)
    ship_poly : tuple[np.ndarray,...] = jit.create_hull_arrays(ship_coords['arc_points'])
    squad_coords : np.ndarray = np.array(squad_state, dtype=np.float32)*HASH_PRECISION_INV

    measure_results = jit.attack_range_q2s_numba(squad_coords, ship_coords['targeting_points'], ship_poly)
    
    in_range_list = measure_results.tolist()

    return in_range_list


@lru_cache(maxsize=64000)
def is_obstruct(targeting_point : tuple[tuple[float, float], tuple[float, float]], ship_state : tuple[str, int, int, int]) -> bool :
    line_of_sight : np.ndarray = np.array(targeting_point, dtype=np.float32)

    ship_token : np.ndarray = np.array(_ship_coordinate(ship_state)['token_corners'], dtype=np.float32)

    return jit.SAT_overlapping_check(line_of_sight, ship_token)

@lru_cache(maxsize=64000)
def is_overlap_s2s(self_state : tuple[str, int, int, int], ship_state : tuple[str, int, int, int]) -> bool :
    self_coordinate = _ship_coordinate(self_state)['base_corners']
    other_coordinate = _ship_coordinate(ship_state)['base_corners']
    return jit.SAT_overlapping_check(self_coordinate, other_coordinate)

@lru_cache(maxsize=64000)
def is_overlap_s2q(ship_state : tuple[str, int, int, int], squad_state : tuple[int, int]) -> bool :
    ship_base : np.ndarray = _ship_coordinate(ship_state)['base_corners']
    squad_center: np.ndarray = np.array([squad_state], dtype=np.float32)*HASH_PRECISION_INV
    if jit.SAT_overlapping_check(ship_base, squad_center) :
        return True

    return jit.polygon_polygon_nearest_points(ship_base, squad_center)[0] <= SQUAD_BASE_RADIUS

@lru_cache(maxsize=64000)
def distance_s2s(self_state : tuple[str, int, int, int], ship_state : tuple[str, int, int, int]) -> float :
    self_poly : np.ndarray = _ship_coordinate(self_state)['base_corners']
    ship_poly : np.ndarray = _ship_coordinate(ship_state)['base_corners']
    return jit.polygon_polygon_nearest_points(self_poly, ship_poly)[0]

@lru_cache(maxsize=64000)
def range_s2q(ship_state : tuple[str, int, int, int], squad_state : tuple[int, int]) -> int:
    """
    Returns:
        AttackRange : The best range from the ship to the squad, used for squad command activation
    """
    ship_token :np.ndarray = _ship_coordinate(ship_state)['token_corners']
    squad_center :np.ndarray = np.array([squad_state], dtype=np.float32)*HASH_PRECISION_INV
    distance = jit.polygon_polygon_nearest_points(ship_token,squad_center)[0] - SQUAD_TOKEN_RADIUS

    return jit.distance_to_range(distance)                                                                                                                                                                                                                                                        

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
    ship_state: tuple[str, int, int, int],
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
    squad_state: tuple[int, int],
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
    scaled = np.array(squad_state, dtype=np.float32)*HASH_PRECISION_INV / [width_step, height_step]
    col = int(scaled[0])
    row = int(scaled[1])

    # Clip to bounds and set a single pixel
    if 0 <= row < height_res and 0 <= col < width_res:
        presence_plane[row, col] = value
    else : raise ValueError(f"Squad position out of bounds\n{squad_state}, scaled: {scaled}, grid: ({col}, {row}), grid size: ({width_res}, {height_res})")

    return presence_plane
    

@lru_cache(maxsize=64000)
def _threat_plane(
    ship_state: tuple[str, int, int, int],
    width_step: float,
    height_step: float,
    width_res: int,
    height_res: int,
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

    threat_zone_coords = _ship_coordinate(ship_state)
    for hull in HULL_SECTIONS:
        close_threat_zone = threat_zone_coords[(hull, AttackRange.CLOSE)]
        medium_threat_zone = threat_zone_coords[(hull, AttackRange.MEDIUM)]
        long_threat_zone = threat_zone_coords[(hull, AttackRange.LONG)]

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
            if dice_count > 0:
                # Clip to valid bounds
                # --- 2. Scale vertices to the High-Resolution Grid ---
                scaled_verts = threat_zone / [width_step_hr, height_step_hr]
                rr, cc = draw_polygon(scaled_verts[:, 1], scaled_verts[:, 0], shape=(high_res_h, high_res_w))
                threat_planes_hr[plane_layer, rr, cc] = dice_count


        # Draw threat zones for each range
        draw_threat_zone(long_threat_zone, long_dice, hull * 3 + AttackRange.LONG)
        draw_threat_zone(medium_threat_zone, medium_dice, hull * 3 + AttackRange.MEDIUM)
        draw_threat_zone(close_threat_zone, close_dice, hull * 3 + AttackRange.CLOSE)

    # --- 3. NEW LOGIC: Reduce and Combine ---
    # Create an empty array to hold the final threat map for each of the 4 hulls
    hull_threat_maps = np.zeros((4, height_res, width_res), dtype=np.float16)

    for hull in HULL_SECTIONS:
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