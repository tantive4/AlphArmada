import math
import random
import json
import os
import shutil
import warnings

import numpy as np
from shapely.geometry import Polygon, box

from armada import Armada
from ship import Ship
from squad import Squad
from obstacle import Obstacle
from armada_game.helpers.enum_class import *
from armada_game.helpers import measurement
from armada_game.helpers import cache_function as cache
from armada_game.helpers.paths import data_path

# --- Data Loading ---
def load_json_data(filename):
    with data_path(filename).open("r", encoding="utf-8") as f:
        return json.load(f)

# Load data into global variables
try:
    SHIP_DATA = load_json_data('ship_dict.json')
    SQUAD_DATA = load_json_data('squad_dict.json')
except FileNotFoundError:
    print("Warning: dictionary files not found under armada_game/data.")
    SHIP_DATA = {}
    SQUAD_DATA = {}


# --- Constants ---
# Board dimensions (approx 3ft x 3ft in mm, based on coordinates in original file)
BOARD_WIDTH = measurement.LONG_RANGE * 3
BOARD_HEIGHT = measurement.LONG_RANGE * 3

DIST_1 = measurement.DISTANCE[1] # 76.5
DIST_2 = measurement.DISTANCE[2] # 124.5
DIST_3 = measurement.DISTANCE[3] # 185.5

# --- Helper Functions ---

def get_random_fleet(faction: Faction, max_points=200, max_squad_points=67):
    """
    Generates a random fleet by iteratively adding any valid unit that fits 
    within the point constraints until no more units can be added.
    """
    target_faction_str = faction.name  # "REBEL" or "EMPIRE"
    
    # 1. Make a full list of ships and squads of given faction
    faction_ships = [name for name, data in SHIP_DATA.items() if data.get('faction') == target_faction_str]
    faction_squads = [name for name, data in SQUAD_DATA.items() if data.get('faction') == target_faction_str]
    
    selected_ships = []
    selected_squads = []
    current_points = 0
    current_squad_points = 0
    
    while True:
        candidates = []

        # Check which Ships can be added (Mark True/False)
        for name in faction_ships:
            cost = SHIP_DATA[name]['point']
            # Condition: Max total points <= 200
            if current_points + cost <= max_points:
                candidates.append((name, 'ship', cost))
        
        # Check which Squads can be added (Mark True/False)
        for name in faction_squads:
            cost = SQUAD_DATA[name]['point']
            # Condition: Max total points <= 200 AND Max squad points <= 67
            if (current_points + cost <= max_points) and (current_squad_points + cost <= max_squad_points):
                candidates.append((name, 'squad', cost))
        
        # If no units can be added, stop
        if not candidates:
            break
            
        # Pick one unit from available units
        pick_name, pick_type, pick_cost = random.choice(candidates)
        
        # Add to list and update points
        if pick_type == 'ship':
            selected_ships.append(pick_name)
            current_points += pick_cost
        else:
            selected_squads.append(pick_name)
            current_points += pick_cost
            current_squad_points += pick_cost

    # Sorting
    selected_squads.sort()
            
    return selected_ships, selected_squads, current_points

def random_coord_in_rect(rect):
    """rect: (min_x, min_y, max_x, max_y)"""
    x = random.uniform(rect[0], rect[2])
    y = random.uniform(rect[1], rect[3])
    return x, y

def get_ship_polygon(ship, x, y, orientation):
    """Returns the ship base polygon transformed to x, y, orientation."""
    ship_state = (ship.name, int(x*measurement.HASH_PRECISION), int(y*measurement.HASH_PRECISION), int(orientation*measurement.HASH_PRECISION))
    return Polygon(cache._ship_coordinate(ship_state)['base_corners'])

def get_obstacle_polygon(obs_obj, x, y, orientation, flip=False):
    """Returns obstacle polygon."""
    return Polygon(cache._obstacle_coordinate((obs_obj.index, x, y, orientation, flip)))


def get_random_obstacle_layout(obstacles, *, max_layout_attempts=100, max_obstacle_attempts=1000):
    """Return a complete random legal six-obstacle layout.

    Rule Reference, Setup / Place Obstacles: ordinary obstacles must be in the
    setup area, beyond distance 3 of its edges, and beyond distance 1 of one
    another.  A complete layout is generated before mutating the game so a
    failed rejection-sampling attempt can be restarted cleanly.
    """
    # TODO(6x3-map): Keep this distance-3 inset for the current 3' x 3' board.
    # When BOARD_WIDTH becomes 6 * LONG_RANGE, replace the horizontal bounds
    # with the central setup area's distance-5 short-edge inset.  See TODO.md.
    obstacle_zone = box(
        DIST_3,
        DIST_3,
        BOARD_WIDTH - DIST_3,
        BOARD_HEIGHT - DIST_3,
    )
    obstacle_rect = obstacle_zone.bounds

    for _ in range(max_layout_attempts):
        placement_order = list(obstacles)
        random.shuffle(placement_order)
        placed_polygons = []
        placements = []

        for obstacle in placement_order:
            for _ in range(max_obstacle_attempts):
                x, y = random_coord_in_rect(obstacle_rect)
                orientation = random.uniform(0, 2 * math.pi)
                flip = bool(random.getrandbits(1))
                polygon = get_obstacle_polygon(obstacle, x, y, orientation, flip)

                if not obstacle_zone.contains(polygon):
                    continue
                if any(polygon.distance(other) <= DIST_1 for other in placed_polygons):
                    continue

                placed_polygons.append(polygon)
                placements.append((obstacle, x, y, orientation, flip, polygon))
                break
            else:
                break

        if len(placements) == len(obstacles):
            return placements

    raise RuntimeError(
        f"Could not generate a legal layout for all {len(obstacles)} obstacles "
        f"after {max_layout_attempts} full-layout attempts."
    )

def setup_game(*, debuging_visual:bool=False, para_index:int=0) -> Armada: 
    players = [Faction.REBEL, Faction.EMPIRE]
    # 1. Generate Fleets
    first_faction, second_faction = random.choices(players, k=2)
    first_ship_names, first_squad_names, first_points = get_random_fleet(first_faction, max_squad_points=0) # simplified
    second_ship_names, second_squad_names, second_points = get_random_fleet(second_faction, max_squad_points=0) # simplified

    game = Armada(faction=(first_faction, second_faction), para_index=para_index)
    game.debuging_visual = debuging_visual

    # Create Objects
    first_ships = [Ship(SHIP_DATA[name], 1) for name in first_ship_names]
    first_squads = [Squad(SQUAD_DATA[name], 1) for name in first_squad_names]
    second_ships = [Ship(SHIP_DATA[name], -1) for name in second_ship_names]
    second_squads = [Squad(SQUAD_DATA[name], -1) for name in second_squad_names]

    obstacles = [
        Obstacle(ObstacleType.ASTEROID, 1),
        Obstacle(ObstacleType.ASTEROID, 2),
        Obstacle(ObstacleType.ASTEROID, 3),
        Obstacle(ObstacleType.DEBRIS, 1),
        Obstacle(ObstacleType.DEBRIS, 2),
        Obstacle(ObstacleType.STATION, 1),
    ]

    # 2. Place Obstacles.  Random order substitutes for alternating player
    # choices while retaining the fixed standard mix and every geometry rule.
    obstacle_layout = get_random_obstacle_layout(obstacles)
    placed_polygons = []
    for obstacle, x, y, orientation, flip, polygon in obstacle_layout:
        game.place_obstacle(obstacle, x, y, orientation, flip)
        placed_polygons.append(polygon)



    first_ship_rect = (DIST_3, 0, BOARD_WIDTH - DIST_3, DIST_3)
    first_ship_zone = box(*first_ship_rect)
    second_ship_rect = (DIST_3, BOARD_HEIGHT - DIST_3, BOARD_WIDTH - DIST_3, BOARD_HEIGHT)
    second_ship_zone = box(*second_ship_rect)
    # 3. Place Ships
    ships = first_ships + second_ships
    for ship in ships:
        is_first = ship in first_ships
        speed = random.choice(list(ship.nav_chart.keys()))
        placed = False
        for _ in range(200): # Retry limit
            if is_first:
                x, y = random_coord_in_rect(first_ship_rect)
                deployment_zone = first_ship_zone
            else:
                x, y = random_coord_in_rect(second_ship_rect)
                deployment_zone = second_ship_zone

            orientation = random.uniform(-math.pi/6, math.pi/6) if is_first else random.uniform(math.pi*5/6, math.pi*7/6)
            poly = get_ship_polygon(ship, x, y, orientation)

            valid = True

            # Check bounds
            if not deployment_zone.contains(poly):
                valid = False

            # Check overlap/distance with existing polygons
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for existing_poly in placed_polygons:
                    if poly.intersects(existing_poly):
                        valid = False
                        break
            
            if valid:
                game.deploy_ship(ship, x, y, orientation, speed)
                placed_polygons.append(poly)
                placed = True
                break
        
        if not placed:
            print(f"Warning: Could not place ship {ship.name} validly.")
    
    # simplified
    # first_squad_rect = (DIST_5, DIST_3, BOARD_WIDTH - DIST_5, DIST_5)
    # first_squad_zone = box(*first_squad_rect)
    # second_squad_rect = (DIST_5, BOARD_HEIGHT - DIST_5, BOARD_WIDTH - DIST_5, BOARD_HEIGHT - DIST_3)
    # second_squad_zone = box(*second_squad_rect)
    # # 4. Place Squads
    # squads = first_squads + second_squads
    # for squad in squads:
    #     is_first = squad in first_squads
    #     placed = False
    #     for _ in range(200): # Retry limit
    #         if is_first:
    #             x, y = random_coord_in_rect(first_squad_rect)
    #             deployment_zone = first_squad_zone
    #         else:
    #             x, y = random_coord_in_rect(second_squad_rect)
    #             deployment_zone = second_squad_zone

    #         poly = Polygon(measurement.SQUAD_BASE_POLY + np.array([[x, y]]))

    #         valid = True

    #         # Check bounds
    #         if not deployment_zone.contains(poly):
    #             valid = False

    #         friendly_ships = first_ship if is_first else second_ship
    #         # Check distance from friendly ships
    #         min_dist = min([poly.distance(get_ship_polygon(ship, ship.x, ship.y, ship.orientation)) for ship in friendly_ships])
    #         if min_dist > DIST_2:
    #             valid = False

    #         # Check overlap/distance with existing polygons
    #         for existing_poly in placed_polygons:
    #             if poly.intersects(existing_poly):
    #                 valid = False
    #                 break
            
    #         if valid:
    #             game.deploy_squad(squad, x, y)
    #             placed_polygons.append(poly)
    #             placed = True
    #             break
        
    #     if not placed:
    #         print(f"Warning: Could not place squad {squad.name} validly.")


    return game

if __name__ == "__main__":
    if os.path.exists("game_visuals"):
        shutil.rmtree("game_visuals")
    game = setup_game(debuging_visual=True, para_index=0)
    print("Game setup complete.")
