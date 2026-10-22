import warnings

import numpy as np
from shapely.geometry import Polygon

from armada_game.helpers.enum_class import *

HASH_PRECISION :int = 1024
HASH_PRECISION_INV : float = 1/HASH_PRECISION

CLOSE_RANGE = 123.3
MEDIUM_RANGE = 186.5
LONG_RANGE = 304.8

DISTANCE = {i : distance for i, distance in enumerate((0, 76.5, 124.5, 185.5, 245.5, 304.8))}

SHIP_BASE_SIZE : dict[SizeClass, tuple]= {SizeClass.SMALL : (43, 71), SizeClass.MEDIUM : (63, 102), SizeClass.LARGE : (77.5, 129)}
SHIP_TOKEN_SIZE : dict[SizeClass, tuple] = {SizeClass.SMALL : (38.5, 70.25), SizeClass.MEDIUM : (58.5, 101.5)}
TOOL_WIDTH_HALF : float= 15.25 / 2
TOOL_LENGTH : float= 46.13 
TOOL_PART_LENGTH : float = 22.27

SQUAD_BASE_RADIUS : float = 17
# The cardboard disk.  Rendering only: every rules measurement uses the base,
# because range and distance are measured to the closest point of the
# squadron's base and only the activation slider is ignored (RRG 693, 701, 705).
SQUAD_TOKEN_RADIUS : float = 15.5
# Distance 1 from a squadron, as a centre radius.  Closest-point measurement
# against a disk collapses exactly onto its centre plus the radius (TODO G2/G3).
Q2S_RANGE = DISTANCE[1] + SQUAD_BASE_RADIUS
Q2Q_RANGE = DISTANCE[1] + SQUAD_BASE_RADIUS * 2
# Touching is not overlapping (RRG FAQ: a ship exactly touching a squadron did
# not overlap it), so overlap tests are strict, with this tolerance absorbing
# float32 rounding in the cached ship geometry (measured at ~5e-5 mm).
SQUAD_TOUCH_EPS : float = 1e-3

# Squadron move grid (TODO D2): a 33x33 grid of absolute offsets centred on the
# squadron, spanning the largest speed's reach.  Odd, so the centre cell is
# "remain in its current position" (RRG Squadron Movement).
SQUAD_MOVE_GRID : int = 33
SQUAD_MOVE_CENTER : int = SQUAD_MOVE_GRID // 2
SQUAD_MOVE_PITCH : float = 2 * DISTANCE[5] / SQUAD_MOVE_GRID
# Touching slots around a ship for overlapped squadrons (RRG Overlapping).
SQUAD_PLACEMENT_SLOTS : int = 28

ERROR_EPSILON = 1e-9
ROTATION_MATRICES = np.stack([
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),         # 0 deg
    np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),        # 90 deg CW
    np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),       # 180 deg CW
    np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)         # 270 deg CW
])
SQUAD_TOKEN_POLY = np.array([
    [np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, num=60, endpoint=False, dtype=np.float32)
], dtype=np.float32) * SQUAD_TOKEN_RADIUS
SQUAD_BASE_POLY = np.array([
    [np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, num=60, endpoint=False, dtype=np.float32)
], dtype=np.float32) * SQUAD_BASE_RADIUS

STATION = np.array([
    [23,  9],
    [ 8, 19],
    [ 6, 23],
    [ 6, 38],
    [11, 56],
    [19, 67],
    [26, 85],
    [36, 92],
    [43, 92],
    [69, 79],
    [78, 72],
    [78, 62],
    [84, 47],
    [94, 38],
    [94, 32],
    [81, 25],
    [68, 13],
    [59,  9]], dtype=np.float32) - np.array([50, 50], dtype=np.float32)
DEBRIS1 = np.array([
    [35, 23],
    [27, 34],
    [27, 51],
    [20, 63],
    [22, 73],
    [35, 81],
    [77, 75],
    [80, 63],
    [72, 54],
    [69, 34],
    [65, 27],
    [56, 19]], dtype=np.float32) - np.array([50, 50], dtype=np.float32)
DEBRIS2 = np.array([
    [18, 11],
    [13, 17],
    [14, 26],
    [50, 58],
    [50, 65],
    [38, 84],
    [38, 90],
    [45, 91],
    [86, 56],
    [87, 44],
    [71, 16],
    [62, 10]], dtype=np.float32) - np.array([50, 50], dtype=np.float32)
ASTEROID1 = np.array([
    [50, 19],
    [38, 36],
    [27, 45],
    [21, 64],
    [35, 77],
    [58, 80],
    [71, 73],
    [78, 62],
    [78, 56],
    [67, 44],
    [59, 22]], dtype=np.float32) - np.array([50, 50], dtype=np.float32)
ASTEROID2 = np.array([
    [56, 19],
    [51, 19],
    [44, 25],
    [31, 53],
    [32, 65],
    [40, 82],
    [62, 76],
    [68, 66],
    [62, 41],
    [57, 36]], dtype=np.float32) - np.array([50, 50], dtype=np.float32)
ASTEROID3 = np.array([
    [59, 19],
    [50, 20],
    [40, 28],
    [38, 33],
    [37, 73],
    [45, 81],
    [49, 81],
    [54, 77],
    [60, 59],
    [62, 41]], dtype=np.float32) - np.array([50, 50], dtype=np.float32)
OBSTACLE = (STATION, DEBRIS1, DEBRIS2, ASTEROID1, ASTEROID2, ASTEROID3)


def create_template_polygons(ship_dict):
    front_arc : tuple[float, float] = (ship_dict['front_arc_center'], ship_dict['front_arc_end']) 
    rear_arc : tuple[float, float] = (ship_dict['rear_arc_center'], ship_dict['rear_arc_end'])
    token_size : tuple[float, float] = SHIP_TOKEN_SIZE[SizeClass[ship_dict['size_class']]]
    token_half_w = token_size[0] / 2
    base_size : tuple[float, float] = SHIP_BASE_SIZE[SizeClass[ship_dict['size_class']]]
    base_half_w = base_size[0] / 2
    base_token_offset = (base_size[1]-token_size[1])/2
    base_rear = base_token_offset - base_size[1]

    # Overlapped-squadron placement slots, each centre exactly SQUAD_BASE_RADIUS
    # from the base so the squadron touches it (RRG Overlapping).  Clockwise
    # from the front-left: front edge x5, front-right corner, right edge x7,
    # rear-right corner, rear edge x5, rear-left corner, left edge x7,
    # front-left corner.  A fixed count per edge keeps each slot index on the
    # same relative spot of every ship; corner slots sit on the rounded part of
    # the touching locus, at 45 degrees from the base corner.
    r = SQUAD_BASE_RADIUS
    d = SQUAD_BASE_RADIUS / np.sqrt(2)
    placement_slots = (
        [[-base_half_w + k * base_half_w / 2, base_token_offset + r] for k in range(5)]
        + [[base_half_w + d, base_token_offset + d]]
        + [[base_half_w + r, base_token_offset - k * base_size[1] / 6] for k in range(7)]
        + [[base_half_w + d, base_rear - d]]
        + [[base_half_w - k * base_half_w / 2, base_rear - r] for k in range(5)]
        + [[-base_half_w - d, base_rear - d]]
        + [[-base_half_w - r, base_rear + k * base_size[1] / 6] for k in range(7)]
        + [[-base_half_w - d, base_token_offset + d]]
    )
    assert len(placement_slots) == SQUAD_PLACEMENT_SLOTS

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

    [0, -token_size[1]/2],                             # center point 14

    [-base_half_w, base_token_offset],                 # left front base 15
    [base_half_w, base_token_offset],                  # right front base 16
    [base_half_w, base_token_offset-base_size[1]],     # right rear base 17
    [-base_half_w, base_token_offset-base_size[1]],    # left rear base 18

    [ (base_half_w + TOOL_WIDTH_HALF), base_token_offset],   # right tool insert 19
    [-(base_half_w + TOOL_WIDTH_HALF), base_token_offset],   # left tool insert 20

    # 21 ~ 48 : overlapped squadron placement slots (see placement_slots)
    *placement_slots,

    ], dtype=np.float32)
    template_vertices += np.array([0, token_size[1]/2], dtype=np.float32)
    return template_vertices
SHIP_TEMPLATE_POLY :dict[str, np.ndarray]= {name : create_template_polygons(ship_dict) for name, ship_dict in SHIP_DATA.items()}


def create_threat_zones(name) -> tuple[np.ndarray, list[int]]:
    template_vertices = SHIP_TEMPLATE_POLY[name]
    ship_token : Polygon = Polygon(template_vertices[6:10])
    arc_coords = template_vertices[0:10]
    close_zone : Polygon = ship_token.buffer(CLOSE_RANGE)
    medium_zone : Polygon = ship_token.buffer(MEDIUM_RANGE)
    long_zone : Polygon = ship_token.buffer(LONG_RANGE)
    threat_zones_list :list[np.ndarray] = []
    index_list :list[int]= []
    current_total_index = 0

    for hull in HULL_SECTIONS:
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
            np.array(arc1_end, dtype=np.float32) + vec1 * 500,
            np.array(arc2_end, dtype=np.float32) + vec2 * 500,
            arc2_end
        ])
        
        # Get threat zones for this hull section
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in intersection")
            # shapely bug error message is ignored
            
            close_threat_zone : Polygon = close_zone.intersection(arc_polygon) #type: ignore
            medium_threat_zone : Polygon = medium_zone.intersection(arc_polygon) #type: ignore
            long_threat_zone : Polygon = long_zone.intersection(arc_polygon) #type: ignore

        close_coords = np.array(close_threat_zone.exterior.coords, dtype=np.float32)
        medium_coords = np.array(medium_threat_zone.exterior.coords, dtype=np.float32)
        long_coords = np.array(long_threat_zone.exterior.coords, dtype=np.float32)

        threat_zones_list.append(np.concatenate([
            close_coords,
            medium_coords,
            long_coords,
        ]))
        for length in (len(close_coords), len(medium_coords), len(long_coords)):
            current_total_index += length
            index_list.append(current_total_index)

    threat_coords = np.concatenate(threat_zones_list, axis=0)
    return threat_coords, index_list[:-1]
SHIP_THREAT_ZONES : dict[str, tuple[np.ndarray, list[int]]] = {name : create_threat_zones(name) for name in SHIP_DATA.keys()}

if __name__ == "__main__":
    name = "VSD1"

    threat_zone, split_index = SHIP_THREAT_ZONES[name]
    threat_zone = np.split(threat_zone, split_index)
    print(threat_zone)
    print(f"{name} threat zones:")
    for i, zone in enumerate(threat_zone):
        print(f"  Zone {i+1}: {zone.shape[0]} points")