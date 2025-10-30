import numpy as np
from shapely.geometry import Polygon

from enum_class import *

HASH_PRECISION :int = 1024
HASH_PRECISION_INV : float = 1/HASH_PRECISION

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
Q2S_RANGE = DISTANCE[1] + SQUAD_TOKEN_RADIUS
Q2Q_RANGE = DISTANCE[1] + SQUAD_TOKEN_RADIUS * 2

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


def create_template_polygons(ship_dict):
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

    [0, -token_size[1]/2],                             # center point 14

    [-base_half_w, base_token_offset],                 # left front base 15
    [base_half_w, base_token_offset],                  # right front base 16
    [base_half_w, base_token_offset-base_size[1]],     # right rear base 17
    [-base_half_w, base_token_offset-base_size[1]],    # left rear base 18

    [ (base_half_w + TOOL_WIDTH_HALF), base_token_offset],   # right tool insert 19
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

    # 43 ~ 45 : FRONT CLOSE, MEDIUM, LONG RANGE THREAT ZONES
    # 46 ~ 48 : RIGHT CLOSE, MEDIUM, LONG RANGE THREAT ZONES
    # 49 ~ 51 : REAR CLOSE, MEDIUM, LONG RANGE THREAT ZONES
    # 52 ~ 54 : LEFT CLOSE, MEDIUM, LONG RANGE THREAT ZONES

    ], dtype=np.float32)

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
        if name == "VSD1":
            print(len(close_coords), len(medium_coords), len(long_coords))
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