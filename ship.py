from __future__ import annotations
import math
from enum import IntEnum, Enum
from typing import TYPE_CHECKING
from collections import Counter
import itertools
from functools import lru_cache
import json

from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
import shapely.ops
import numpy as np

from dice import Dice, Critical
from defense_token import DefenseToken, TokenType
from measurement import AttackRange, CLOSE_RANGE, MEDIUM_RANGE, LONG_RANGE
if TYPE_CHECKING:
    from armada import Armada

with open('ship_info.json', 'r') as f:
    SHIP_DATA: dict = json.load(f)

class HullSection(IntEnum):
    FRONT = 0
    RIGHT = 1
    REAR = 2
    LEFT = 3
    def __str__(self):
        return self.name
    __repr__ = __str__

class SizeClass(IntEnum) :
    SMALL = 1
    MEDIUM = 2
    LARGE = 3
    HUGE = 4

class Command(IntEnum) :
    NAV = 0
    REPAIR = 1
    CONFIRE = 2
    # SQUAD = 3
    def __str__(self):
        return self.name
    __repr__ = __str__


SHIP_BASE_SIZE : dict[SizeClass, tuple]= {SizeClass.SMALL : (43, 71), SizeClass.MEDIUM : (63, 102), SizeClass.LARGE : (77.5, 129)}
SHIP_TOKEN_SIZE :  dict[SizeClass, tuple] = {SizeClass.SMALL : (38.5, 70.25), SizeClass.MEDIUM : (58.5, 101.5)}
TOOL_WIDTH_HALF : float= 15.25 / 2
TOOL_LENGTH : float= 46.13 
TOOL_PART_LENGTH : float = 22.27
ROTATION_MATRIX_90_CW = np.array([[0, -1], [1, 0]])
ROTATION_MATRICES = [np.linalg.matrix_power(ROTATION_MATRIX_90_CW, i) for i in range(4)]

class Ship:
    def __init__(self, ship_dict : dict, player : int) -> None:
        self.player : int = player
        self.name : str = ship_dict['name']

        self.max_hull : int = ship_dict['hull']
        self.size_class : SizeClass = SizeClass[ship_dict['size_class']]
        self.token_size : tuple [float, float] = SHIP_TOKEN_SIZE[self.size_class]
        self.base_size : tuple [float, float] = SHIP_BASE_SIZE[self.size_class]

        self.battery :  dict[HullSection, list[int]] = {HullSection.FRONT : ship_dict['battery'][0], 
                                                        HullSection.RIGHT : ship_dict['battery'][1], 
                                                        HullSection.REAR : ship_dict['battery'][2], 
                                                        HullSection.LEFT : ship_dict['battery'][1]}
        self.battery_range : dict[HullSection, dict[AttackRange, tuple[int, ...]]] = {
            hull: {
                attack_range : tuple(self.battery[hull][dice_type.value] if dice_type.value >= attack_range.value else 0 for dice_type in Dice)
                for attack_range in AttackRange if attack_range != AttackRange.INVALID
            } for hull in HullSection
        }
        
        self.defense_tokens: dict[int, DefenseToken] = {}
        token_counts = Counter()
        # Iterate through the list of token strings from the JSON
        for token_type_str in ship_dict['defense_token']:
            token_enum = TokenType[token_type_str.upper()]
            # token_counts[token_enum] will be 0 for the first, 1 for the second, etc.
            key = token_enum.value * 2 + token_counts[token_enum]
            # Add the token to the dictionary and increment the count for that type
            self.defense_tokens[key] = DefenseToken(token_type_str)
            token_counts[token_enum] += 1

        self.nav_chart : dict[int, list[int]] = {int(k) : v for k, v in ship_dict['navchart'].items()}
        self.max_shield : dict[HullSection, int] = {HullSection.FRONT : ship_dict['shield'][0], 
                                                    HullSection.RIGHT : ship_dict['shield'][1], 
                                                    HullSection.REAR : ship_dict['shield'][2], 
                                                    HullSection.LEFT : ship_dict['shield'][1]}
        self.point : int = ship_dict['point']
        self.command_value : int = ship_dict['command']
        self.engineer_value : int = ship_dict['engineering']
        self.destroyed = False

        self.front_arc : tuple[float, float] = (ship_dict['front_arc_center'], ship_dict['front_arc_end']) 
        self.rear_arc : tuple[float, float] = (ship_dict['rear_arc_center'], ship_dict['rear_arc_end'])
        
        self._create_template_geometries(ship_dict)
        self._course_cache : dict[tuple[int, bool], list[tuple[int, ...]]]= {}
        
    def __str__(self):
        return self.name
    __repr__ = __str__



    def deploy(self, game : Armada , x : float, y : float, orientation : float, speed : int, ship_id : int) -> None:
        """
        deploy the ship to the game board

        Args:
            game (Armada) : the game
            (x, y) (float) : coordination of front center of the **ship token**
            orientation (float) : ship orientation
            speed (int) : ship speed
            ship_id (int) : ship id
        """
        self.game : Armada = game
        self.x: float = x
        self.y: float = y
        self.orientation: float = orientation
        self.speed: int = speed

        self.hull: int = self.max_hull
        self.shield: tuple[int, int, int, int] = tuple(self.max_shield[hull] for hull in HullSection)
        self.ship_id: int = ship_id
        self.command_stack: tuple[Command, ...] = ()
        self.command_dial : tuple[Command, ...] = ()
        self.command_token : tuple[Command, ...] = ()
        self.resolved_command : tuple[Command, ...] = ()
        self.engineer_point : int = 0
        self.attack_count : int = 0
        self.attack_impossible_hull : tuple[HullSection, ...] = ()
        self.repaired_hull : tuple[HullSection, ...] = ()
        self.refresh()
    
    def asign_command(self, Command) -> None :
        if len(self.command_stack) >= self.command_value : raise ValueError("Cannot asigne more command then Command Value")
        self.command_stack += (Command,)


    def destroy(self) -> None:
        self.destroyed = True
        self.hull = 0
        self.shield = (0, 0, 0, 0)
        for token in self.defense_tokens.values() :
            if not token.discarded : token.discard()
        self.game.visualize(f'{self} is destroyed!')

    def refresh(self) -> None:
        self.activated = False
        for token in self.defense_tokens.values():
            if not token.discarded:
                token.ready()

    def end_activation(self) -> None :
        self.activated = True
        self.attack_impossible_hull = ()
        self.attack_count = 0
        self.command_dial = ()
        self.resolved_command = ()

    def execute_maneuver(self, course : tuple[int, ...], placement : int) -> None:
        overlap_ships = self.move_ship(course, placement, set())
        self.overlap(overlap_ships)
        if self.out_of_board() :
            self.game.visualize(f'{self} is out of board!')
            self.destroy()

    def move_ship(self, course : tuple[int, ...], placement : int, overlap_ships : set[int]) -> set[int]:
        if not course :
            self.game.visualize(f'{self} executes speed 0 maneuver.')
            return overlap_ships
        # tool_coord = self._tool_coordination(course, placement)[0]
        
        original_position, original_orientation = np.array([self.x, self.y]), self.orientation

        tool_translation, tool_rotation = _cached_maneuver_tool(self.size_class, course, placement)
        # change translation vector according to the current orientation
        # rotation matrix use CW orientation    
        c = np.cos(-original_orientation)
        s = np.sin(-original_orientation)
        rotation_matrix = np.array([[c, -s],
                                    [s,  c]])
        tool_translation = rotation_matrix @ tool_translation
        self.x, self.y = original_position + tool_translation
        self.orientation += tool_rotation
        # self.game.visualize(f'{self} executes speed {len(course)} maneuver.',tool_coord)
        current_overlap = self.is_overlap()
        
        if current_overlap:
            # self.game.visualize(f'{self} overlaps ships at speed {len(course)} maneuver.',tool_coord)
            overlap_ships = overlap_ships.union(current_overlap)
            self.x, self.y, self.orientation = *original_position, original_orientation
            new_overlap = self.move_ship(course[:-1], placement, overlap_ships)
            overlap_ships = overlap_ships.union(new_overlap)
        return overlap_ships






# sub method for ship dimension

    def _create_template_geometries(self, ship_dict : dict) -> None:
        """
        Creates template vertices (points) for all ship shapes, relative to a single
        (0,0) pivot at the token's front-center.
        """
        token_half_w = self.token_size[0] / 2

        base_half_w = self.base_size[0] / 2
        base_front_y = (self.base_size[1] - self.token_size[1]) / 2
        base_rear_y = base_front_y - self.base_size[1]
        self.template_base_vertices = np.array([
            [base_half_w, base_front_y], [-base_half_w, base_front_y],
            [-base_half_w, base_rear_y], [base_half_w, base_rear_y]
        ])

        front_arc_center_pt = (0, -self.front_arc[0])
        front_left_arc_pt = (-token_half_w, -self.front_arc[1])
        front_right_arc_pt = (token_half_w, -self.front_arc[1])
        rear_arc_center_pt = (0, -self.rear_arc[0])
        rear_left_arc_pt = (-token_half_w, -self.rear_arc[1])
        rear_right_arc_pt = (token_half_w, -self.rear_arc[1])
        front_left_token_pt = (-token_half_w, 0)
        front_right_token_pt = (token_half_w, 0)
        rear_left_token_pt = (-token_half_w, -self.token_size[1])
        rear_right_token_pt = (token_half_w, -self.token_size[1])

        self.template_token_vertices = np.array([
            front_right_token_pt, front_left_token_pt,
            rear_left_token_pt, rear_right_token_pt
        ])

        self.template_hull_vertices = {
            HullSection.FRONT: np.array([
                front_arc_center_pt, front_right_arc_pt, front_right_token_pt, 
                front_left_token_pt, front_left_arc_pt
            ]),
            HullSection.RIGHT: np.array([
                front_arc_center_pt, front_right_arc_pt, 
                rear_right_arc_pt, rear_arc_center_pt
            ]),
            HullSection.REAR: np.array([
                rear_arc_center_pt, rear_right_arc_pt, rear_right_token_pt,
                rear_left_token_pt, rear_left_arc_pt
            ]),
            HullSection.LEFT: np.array([
                front_arc_center_pt, front_left_arc_pt,
                rear_left_arc_pt, rear_arc_center_pt
            ])
        }
        
        # 0~3 : targeting points
        # 4,5 : Right/Left maneuver tool insert point
        # 6 : ship token center
        self.template_targeting_points_and_maneuver_tool_insert = np.array([
            [0, -ship_dict['front_targeting_point']],
            [ship_dict['side_targeting_point'][0], -ship_dict['side_targeting_point'][1]],
            [0, -ship_dict['rear_targeting_point']],
            [- ship_dict['side_targeting_point'][0], -ship_dict['side_targeting_point'][1]],
            [ (self.base_size[0] + TOOL_WIDTH_HALF)/2, (self.base_size[1]-self.token_size[1])/2],
            [-(self.base_size[0] + TOOL_WIDTH_HALF)/2, (self.base_size[1]-self.token_size[1])/2],
            [0, -self.token_size[1]/2]
        ])





# sub method for attack
    def measure_line_of_sight(self, from_hull : HullSection, to_ship : Ship, to_hull : HullSection) -> bool :
        """
        Checks if the line of sight between two ships is obstructed.

        Args:
            from_hull (HullSection): The firing hull section of the attacking ship.
            to_ship (Ship): The target ship.
            to_hull (HullSection): The targeted hull section of the defending ship.

        Returns:
            obstructed (bool)
        """
        line_of_sight : tuple[tuple[float, float], ...] = (tuple(_cached_coordinate(self.get_ship_hash_state())['targeting_points'][from_hull]), tuple(_cached_coordinate(to_ship.get_ship_hash_state())['targeting_points'][to_hull]))

        for ship in self.game.ships:

            if ship.ship_id == self.ship_id or ship.ship_id == to_ship.ship_id:
                continue
            if ship.destroyed:
                continue

            if _cached_obstruction(line_of_sight, ship.get_ship_hash_state()):
                return True

        return False

    def gather_dice(self, attack_hull : HullSection, attack_range : AttackRange) -> tuple[int, ...] :
        if attack_range in (AttackRange.INVALID, AttackRange.EXTREME): return (0, 0, 0)
        attack_pool = self.battery_range[attack_hull][attack_range]
        return attack_pool

    def defend(self, defend_hull : HullSection, total_damage : int, critical: Critical | None) -> None:
        # Absorb damage with shields first
        shield_damage = min(total_damage, self.shield[defend_hull])

        shield_list = list(self.shield)
        shield_list[defend_hull] -= shield_damage
        self.shield = tuple(shield_list)

        total_damage -= shield_damage

        # Apply remaining damage to the hull
        if total_damage > 0:
            self.hull -= total_damage
            if critical == Critical.STANDARD : self.hull -= 1 # Structural Damage

        if self.hull <= 0 : self.destroy()

    def get_valid_target(self, attack_hull : HullSection) -> list[tuple[Ship, HullSection]] :
        valid_targets : list[tuple[Ship, HullSection]]= []

        for ship in self.game.ships:
            if ship.ship_id == self.ship_id or ship.destroyed or ship.player == self.player: continue

            target_dict, range_dict =  _cached_range(self.get_ship_hash_state(), ship.get_ship_hash_state())

            if not target_dict[attack_hull] : continue

            for target_hull in HullSection :
                attack_range : AttackRange = range_dict[attack_hull][target_hull] 
                
                if attack_range in (AttackRange.INVALID, AttackRange.EXTREME): continue

                dice_count = sum(self.gather_dice(attack_hull, attack_range))
                if dice_count == 0 : continue
                elif dice_count == 1:
                    if self.measure_line_of_sight(attack_hull, ship, target_hull) : continue
                valid_targets.append((ship, target_hull))


        return valid_targets

    def get_valid_attack_hull(self) -> list[HullSection]:
        """
        Get a list of valid attacking hull sections for the ship.

        Returns:
            valid_attacker (list[HullSection]): A list of valid attacking hull sections.
        """
        valid_attacker = []
        for hull in HullSection:
            if hull in self.attack_impossible_hull: continue

            if self.get_valid_target(hull):
                valid_attacker.append(hull)

        return valid_attacker

    def get_critical_effect(self, black_crit : bool, blue_crit : bool, red_crit : bool) -> list[Critical] :
        critical_list : list[Critical] = []
        if black_crit or blue_crit or red_crit :
            critical_list.append(Critical.STANDARD)
        return critical_list




# sub method for execute maneuver
    def _tool_coordination(self, course : tuple[int, ...], placement : int) -> tuple[list[tuple[float, float]], list[float]]:
        """
        Calculates the coordinates and orientations along a maneuver tool's path using NumPy.
        """

        tool_coords = _cached_coordinate(self.get_ship_hash_state())['tool_corners']
        tool_coord : tuple[float, float] = tool_coords[0] if placement == 1 else tool_coords[1]
        
        if not course:
            # For speed 0, return the starting point and orientation
            return [tool_coord], [self.orientation]
        
        # --- Step 1: Set up initial conditions ---
        initial_position = np.array(tool_coord)
        initial_orientation = self.orientation
        speed = len(course)

        # --- Step 2: Calculate all joint orientations at once ---
        # This array will have shape (speed + 1) and includes the initial orientation
        yaw_changes = np.array([0] + list(course)) * (math.pi / 8)
        joint_orientations = initial_orientation + np.cumsum(yaw_changes)

        # --- Step 3: Calculate the direction vectors for each segment ---
        long_segment_orientations = joint_orientations[:-1]
        short_segment_orientations = joint_orientations[1:]
        
        segment_orientations = np.empty(2 * speed)
        segment_orientations[0::2] = long_segment_orientations
        segment_orientations[1::2] = short_segment_orientations
        
        segment_lengths = np.tile([TOOL_LENGTH, TOOL_PART_LENGTH], speed)
        direction_vectors = np.array([np.sin(segment_orientations), np.cos(segment_orientations)]).T

        # --- Step 4: Calculate all position vectors ---
        position_vectors = direction_vectors * segment_lengths[:, np.newaxis]
        
        # This creates an array of all points along the path, including the start point.
        # The shape will be (2 * speed + 1, 2)
        all_points = np.vstack([initial_position, np.cumsum(position_vectors, axis=0) + initial_position])

        # --- Step 5: Convert back to the required list format (Corrected) ---
        # Return the full list of orientations to match the original function's output.
        # The shape will now be (speed + 1)
        return all_points.tolist(), joint_orientations.tolist()
        
    def is_overlap(self) -> set[int] :        
        """
        determines which ship overlaps to this ship at current location
        
        Returns:
            overlap_list (list[bool]): A list indicating which ships were overlapped.
        """
        overlap_list = set()
        for ship in self.game.ships:
            if self.ship_id == ship.ship_id or ship.destroyed :
                continue

            if _cached_overlapping(self.get_ship_hash_state(), ship.get_ship_hash_state()):
                overlap_list.add(ship.ship_id)
        return overlap_list

    def overlap(self, overlap_list : set[int]) -> None:
        """
        Determines which of the overlapping ships is closest and handles the collision.

        Args:
            overlap_list (set[int]): A set indicating which ships were overlapped.
        """
        closest_ship = None
        min_distance = float('inf')

        for ship_id in overlap_list:
            other_ship = self.game.ships[ship_id]

            # Calculate the distance between the two ship bases.
            distance = _cached_distance(self.get_ship_hash_state(), other_ship.get_ship_hash_state())
            if distance < min_distance:
                min_distance = distance
                closest_ship = other_ship

        if closest_ship:
            self.hull -= 1
            closest_ship.hull -= 1
            self.game.visualize(f"\n{self} overlaps to {closest_ship}.")
            if self.hull <= 0 : self.destroy()
            if closest_ship.hull <= 0 :closest_ship.destroy()

    def out_of_board(self) -> bool:
        """
        Checks if the ship's base is completely within the game board.

        Returns:
            bool: True if the ship is out of the board, False otherwise.
        """
        coords = _cached_coordinate(self.get_ship_hash_state())
        base_corners = coords['base_corners']
        min_x, min_y = base_corners.min(axis=0)
        max_x, max_y = base_corners.max(axis=0)

        if min_x >= 0 and max_x <= self.game.player_edge and min_y >= 0 and max_y <= self.game.short_edge:
            return False
        
        return True
    
    def get_valid_speed(self) -> list[int]:
        """
        Get a list of valid speeds for the ship based on its navchart.

        Returns:
            list[int]: A list of valid speeds.
        """
        valid_speed = []
        speed_change : int = int(Command.NAV in self.command_dial) + int(Command.NAV in self.command_token)

        for speed in range(5):
            if abs(speed - self.speed) > speed_change:
                continue
            if self.nav_chart.get(speed) is not None or speed == 0:
                valid_speed.append(speed)

        return valid_speed
    
    def get_valid_yaw(self, speed : int, joint : int) -> list[int]:
        """
        Get a list of valid yaw adjustments for the ship based on its navchart.

        Args:
            speed (int): The current speed of the ship.
            joint (int): The joint index for which to get valid yaw adjustments.

        Returns:
            list[int]: A list of valid yaw adjustments. -2 ~ 2
        """
        valid_yaw = []
        
        for yaw in range(5):
            if abs(yaw - 2) > self.nav_chart[speed][joint]:
                continue
            valid_yaw.append(yaw -2)

        return valid_yaw
    
    def nav_command_used(self, course: tuple[int, ...]) -> tuple[bool, bool] :
            nav_dial_used = False
            nav_token_used = False
            new_speed = len(course)

            # 1. SPEED CHANGE VALIDATION
            speed_change = abs(new_speed - self.speed)

            if speed_change > 2:
                raise ValueError(f"Invalid speed change of {speed_change}. Maximum is 2.")

            if speed_change == 1:
                # Must use one NAVIGATE command from either dial or token
                if Command.NAV in self.command_dial:
                    nav_dial_used = True
                elif Command.NAV in self.command_token:
                    nav_token_used = True
                else:
                    raise ValueError("Speed change of 1 requires a NAVIGATE command from a dial or token.")

            if speed_change == 2:
                # Must use two NAVIGATE commands, one from each source
                if Command.NAV in self.command_dial and Command.NAV in self.command_token:
                    nav_dial_used = True
                    nav_token_used = True
                else:
                    raise ValueError("Speed change of 2 requires NAVIGATE commands from BOTH a dial and a token.")

            # 2. COURSE VALIDITY CHECK (for extra clicks)
            is_standard = self.is_standard_course(course)
            
            if not is_standard:
                # This is a special course that requires an extra click from the command dial.
                if Command.NAV in self.command_dial:
                    nav_dial_used = True
                else:
                    raise ValueError("This course requires an extra click, which needs a NAVIGATE command from the dial.")
            return nav_dial_used, nav_token_used
    
    def is_standard_course(self, course:tuple[int, ...]) -> bool :
        """
        Checks if a given course is a standard maneuver for a given speed,
        without using any special abilities like adding a click.
        """
        speed = len(course)
        if speed == 0 : return True

        for joint, yaw in enumerate(course) :
            if abs(yaw) > self.nav_chart[speed][joint] : return False
        return True

    def get_all_possible_courses(self, speed: int) -> list[tuple[int, ...]]:
        """
        Gets all possible maneuver courses for a given speed.

        If can_add_click is True, it will also generate additional courses by modifying
        a single joint of any originally valid course by one click (yaw).

        Args:
            speed (int): The maneuver speed.

        Returns:
            A list of all unique, valid course lists.
        """
        has_nav_dial = Command.NAV in self.command_dial
        cache_key = (speed, has_nav_dial)

        if cache_key in self._course_cache :
            return self._course_cache[cache_key]
        
        if speed == 0:
            return [()] # A speed 0 maneuver has an empty course

        # 1. Get the base valid yaw options for each joint from the nav chart.
        original_yaw_options = [self.get_valid_yaw(speed, joint) for joint in range(speed)]

        # 2. Generate all standard courses using the Cartesian product.
        # Using a set handles duplicates automatically.
        all_courses = set(itertools.product(*original_yaw_options))

        # 3. If the special condition isn't met, return the standard courses.
        if not Command.NAV in self.command_dial:
            return [course for course in all_courses]

        # 4. If the condition is met, generate new courses from the standard ones.
        # We start with a copy of the original courses to iterate over.
        standard_courses = all_courses.copy()
        for course_tuple in standard_courses:
            course_list = list(course_tuple)
            # Iterate through each joint in the current standard course.
            for i in range(len(course_list)):
                original_yaw = course_list[i]

                # Option A: Add one click to the current joint
                modified_yaw_add = original_yaw + 1
                if abs(modified_yaw_add) <= 2:
                    new_course = list(course_list) # Create a copy
                    new_course[i] = modified_yaw_add
                    all_courses.add(tuple(new_course))

                # Option B: Subtract one click from the current joint
                modified_yaw_sub = original_yaw - 1
                if abs(modified_yaw_sub) <= 2:
                    new_course = list(course_list) # Create a copy
                    new_course[i] = modified_yaw_sub
                    all_courses.add(tuple(new_course))


        final_result = sorted(list(all_courses))
        self._course_cache[cache_key] = final_result
        return final_result
    
    def get_valid_placement(self, course : tuple[int, ...]) -> list[int]:
        """
        Get a list of valid placements for the ship based on its navchart.

        Args:
            course (list[int]): The course of the ship.

        Returns:
            list[int]: A list of valid placements. 
                1 for right, -1 for left
        """
        speed = len(course)
        if speed == 0 : return [0]
        
        valid_placement = [-1, 1]
        if speed > 0 :
            if course[-1] > 0 : valid_placement.remove(-1)
            elif course[-1] < 0 : valid_placement.remove(1)
            else : 
                if speed >= 2 and self.size_class > SizeClass.SMALL :
                    if course[-2] > 0 : valid_placement.remove(-1)
                    elif course[-2] < 0 : valid_placement.remove(1)
        return valid_placement

    def get_snapshot(self) -> dict :
        return {
            "x": self.x,
            "y": self.y,
            "orientation": self.orientation,
            "speed": self.speed,
            "hull": self.hull,
            "shield": self.shield,
            "destroyed": self.destroyed,
            "activated": self.activated,
            "command_stack": self.command_stack,
            "command_dial": self.command_dial,
            "command_token": self.command_token,
            "resolved_command": self.resolved_command,
            "attack_count": self.attack_count,
            "attack_impossible_hull": self.attack_impossible_hull,
            "engineer_point" : self.engineer_point,
            "repaired_hull" : self.repaired_hull,
            "defense_tokens": {
                key: (dt.readied, dt.discarded, dt.accuracy) 
                for key, dt in self.defense_tokens.items()
            }
        }
    
    def revert_snapshot(self, snapshot: dict) -> None:
        """Restores the ship's state from a snapshot."""
        self.x = snapshot["x"]
        self.y = snapshot["y"]
        self.orientation = snapshot["orientation"]
        self.speed = snapshot["speed"]
        self.hull = snapshot["hull"]
        self.shield = snapshot["shield"]
        self.destroyed = snapshot["destroyed"]
        self.activated = snapshot["activated"]
        self.command_stack = snapshot["command_stack"]
        self.command_dial = snapshot["command_dial"]
        self.command_token = snapshot["command_token"]
        self.resolved_command = snapshot["resolved_command"]
        self.attack_count = snapshot["attack_count"]
        self.engineer_point = snapshot["engineer_point"]
        self.repaired_hull = snapshot["repaired_hull"]
        self.attack_impossible_hull = snapshot["attack_impossible_hull"]

        for key, token_state in snapshot["defense_tokens"].items():
            self.defense_tokens[key].readied, self.defense_tokens[key].discarded, self.defense_tokens[key].accuracy = token_state


    def get_ship_hash_state(self) -> tuple[str, float, float, float]:
        """Returns a hashable tuple representing the ship's state."""
        return (self.name, self.x, self.y, self.orientation)
    

def _cached_coordinate(ship_state : tuple[str, float, float, float]) -> dict[str,np.ndarray] :

    ship_dict = SHIP_DATA[ship_state[0]]

    front_arc : tuple[float, float] = (ship_dict['front_arc_center'], ship_dict['front_arc_end']) 
    rear_arc : tuple[float, float] = (ship_dict['rear_arc_center'], ship_dict['rear_arc_end'])
    token_size : tuple[float, float] = SHIP_TOKEN_SIZE[SizeClass[ship_dict['size_class']]]
    token_half_w = token_size[0] / 2
    base_size : tuple[float, float] = SHIP_BASE_SIZE[SizeClass[ship_dict['size_class']]]
    base_half_w = base_size[0] / 2

    template_vertices = np.array([
    [0, -front_arc[0]],                 # front_arc_center_pt 0
    [-token_half_w, -front_arc[1]],     # front_left_arc_pt 1
    [token_half_w, -front_arc[1]],      # front_right_arc_pt 2
    [0, -rear_arc[0]],                  # rear_arc_center_pt 3
    [-token_half_w, -rear_arc[1]],      # rear_left_arc_pt 4
    [token_half_w, -rear_arc[1]],       # rear_right_arc_pt 5
    [-token_half_w, 0],                 # front_left_token_pt 6
    [token_half_w, 0],                  # front_right_token_pt 7
    [-token_half_w, -token_size[1]],    # rear_left_token_pt 8
    [token_half_w, -token_size[1]],     # rear_right_token_pt 9
    [0, -ship_dict['front_targeting_point']],
    [ship_dict['side_targeting_point'][0], -ship_dict['side_targeting_point'][1]],
    [0, -ship_dict['rear_targeting_point']],
    [- ship_dict['side_targeting_point'][0], -ship_dict['side_targeting_point'][1]],
    [0, -token_size[1]/2],              # center point 14
    [-base_half_w, 0],                 # left front base 15
    [base_half_w, 0],                  # right front base 16
    [base_half_w, -base_size[1]],      # right rear base 17
    [-base_half_w, -base_size[1]],    # left rear base 18
    [ (base_half_w + TOOL_WIDTH_HALF), (base_size[1]-token_size[1])/2],  # right tool insert 19
    [-(base_half_w + TOOL_WIDTH_HALF), (base_size[1]-token_size[1])/2]   # left tool insert 20
    ])

    # CW rotation matrix
    c, s = math.cos(-ship_state[3]), math.sin(-ship_state[3])
    rotation_matrix = np.array([[c, -s], [s, c]])
    translation_vector = np.array([ship_state[1], ship_state[2]])

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
    }

@lru_cache(maxsize=None)
def _cached_polygons(ship_state: tuple[str, float, float, float]) -> dict[HullSection | str, Polygon]:
    """
    Creates and caches all hull polygons for a given ship state.
    """
    coords = _cached_coordinate(ship_state)
    arc_coords = coords['arc_points']
    base_coords = coords['base_corners']
    token_coords = coords['token_corners']
    
    return {
        HullSection.FRONT: Polygon([arc_coords[0], arc_coords[2], arc_coords[7], arc_coords[6], arc_coords[1]]),
        HullSection.RIGHT: Polygon([arc_coords[0], arc_coords[2], arc_coords[5], arc_coords[3]]),
        HullSection.REAR: Polygon([arc_coords[3], arc_coords[5], arc_coords[9], arc_coords[8], arc_coords[4]]),
        HullSection.LEFT: Polygon([arc_coords[0], arc_coords[1], arc_coords[4], arc_coords[3]]),
        'token' : Polygon(token_coords),
        'base' : Polygon(base_coords)
    }

@lru_cache(maxsize=None)
def _cached_point_range(ship_state: tuple[str, float, float, float], point: tuple[float, float]) -> dict[HullSection, AttackRange]:
    """
    Checks if a point is within a specific firing arc and within max_range.
    
    Args:
        attack_hull: The hull section whose arc is being checked.
        point: The (x, y) coordinate of the point to check.
        max_range: The maximum distance the arc extends to.

    Returns:
        True if the point is within the arc and range, False otherwise.
    """
    measure_dict : dict[HullSection, AttackRange] = {from_hull : AttackRange.INVALID for from_hull in HullSection}
    
    ship_coords = _cached_coordinate(ship_state)
    ship_center : np.ndarray = ship_coords['center_point']

    target_vector : np.ndarray = np.array(point) - ship_center

    arc_vector_tuple : tuple[np.ndarray, ...] = (
        ship_coords['arc_points'][2] - ship_center, # front right
        ship_coords['arc_points'][5] - ship_center, # rear right
        ship_coords['arc_points'][4] - ship_center, # rear left
        ship_coords['arc_points'][1] - ship_center, # front left
    )

    def is_point_in_arc(target_vector: np.ndarray, 
                                arc1_start: np.ndarray, 
                                arc2_end: np.ndarray) -> bool:
        """
        Checks if a target vector lies within a clockwise arc.
        This version uses manual calculation to avoid NumPy 2.0 deprecation warnings.
        """
        # Manual 2D cross product: a[0]*b[1] - a[1]*b[0]
        
        # Is the target clockwise relative to the start? (cross product <= 0)
        cross_product_start = arc1_start[0] * target_vector[1] - arc1_start[1] * target_vector[0]
        is_clockwise_from_start = cross_product_start <= 0

        # Is the target counter-clockwise relative to the end? (cross product >= 0)
        cross_product_end = arc2_end[0] * target_vector[1] - arc2_end[1] * target_vector[0]
        is_counter_clockwise_from_end = cross_product_end >= 0

        return is_clockwise_from_start and is_counter_clockwise_from_end
        
    for from_hull in HullSection :
        arc1 = arc_vector_tuple[from_hull.value]
        arc2 = arc_vector_tuple[(from_hull.value + 1) % 4]

        if not is_point_in_arc(target_vector, arc1, arc2) :
            continue

        targeting_pt = ship_coords['targeting_points'][from_hull]
        distance = np.linalg.norm(targeting_pt - np.array(point))

        if distance <= CLOSE_RANGE : 
            measure_dict[from_hull] = AttackRange.CLOSE # close range
        elif distance <= MEDIUM_RANGE : 
            measure_dict[from_hull] = AttackRange.MEDIUM # medium range
        elif distance <= LONG_RANGE : 
            measure_dict[from_hull] = AttackRange.LONG # long range
        else : measure_dict[from_hull] = AttackRange.EXTREME # invalid range (extreme)
    return measure_dict


@lru_cache(maxsize=None)
def _cached_range(attacker_state : tuple[str, float, float, float], defender_state : tuple[str, float, float, float], extension_factor=500) -> tuple[dict[HullSection, bool],dict[HullSection, dict[HullSection, AttackRange]]]:
    """
    return:
        attack_range (AttackRange)
    """
    target_dict : dict[HullSection, bool] = {from_hull : False for from_hull in HullSection}
    measure_dict : dict[HullSection, dict[HullSection, AttackRange]] = {from_hull : {to_hull : AttackRange.INVALID for to_hull in HullSection} for from_hull in HullSection}
    
    attacker_coords = _cached_coordinate(attacker_state)
    defender_coords = _cached_coordinate(defender_state)

    attacker_center : np.ndarray = attacker_coords['center_point']
    defender_center : np.ndarray = defender_coords['center_point']
    # orientation vector points to the front of the ship
    attacker_orientation_vector : np.ndarray = np.array([math.sin(-attacker_state[3]), math.cos(-attacker_state[3])])
    

    # # distance check
    target_vector : np.ndarray = attacker_center - defender_center
    # distance = np.linalg.norm(target_vector)
    # if distance > 2 * LONG_RANGE :
    #     return target_dict, measure_dict
    
    attacker_poly : dict[HullSection | str, Polygon] = _cached_polygons(attacker_state)
    defender_poly : dict[HullSection | str, Polygon] = _cached_polygons(defender_state)


    for from_hull in HullSection :
        for to_hull in HullSection :
            from_hull_targeting_pt = attacker_coords['targeting_points'][from_hull]
            to_hull_targeting_pt = defender_coords['targeting_points'][to_hull]

            # attack hull orientation check
            attack_orientation_vector = ROTATION_MATRICES[from_hull] @ attacker_orientation_vector
            hull_target_vector = to_hull_targeting_pt - from_hull_targeting_pt
            dot_product = np.dot(attack_orientation_vector, hull_target_vector)
            if dot_product < 0 :
                continue

            # Line of Sight blocked
            is_blocked : bool = False
            line_of_sight = LineString([from_hull_targeting_pt, to_hull_targeting_pt])
            for hull in HullSection:
                if hull != to_hull and line_of_sight.crosses(defender_poly[hull].exterior):
                    is_blocked = True
                    continue
            if is_blocked : continue

            # Range
            from_hull_poly = attacker_poly[from_hull]
            to_hull_poly = defender_poly[to_hull]

            if from_hull in (HullSection.FRONT, HullSection.RIGHT) :
                arc1_center, arc1_end = attacker_coords['arc_points'][0], attacker_coords['arc_points'][2]
            else :
                arc1_center, arc1_end = attacker_coords['arc_points'][3], attacker_coords['arc_points'][4]

            if from_hull in (HullSection.FRONT, HullSection.LEFT) :
                arc2_center, arc2_end = attacker_coords['arc_points'][0], attacker_coords['arc_points'][1]
            else :
                arc2_center, arc2_end = attacker_coords['arc_points'][3], attacker_coords['arc_points'][5]

            # Build the arc polygon. This logic works for ALL hull sections.
            vec1 = np.array(arc1_end) - np.array(arc1_center)
            vec2 = np.array(arc2_end) - np.array(arc2_center) # Note: vector points away from the ship
            arc_polygon = Polygon([
                arc1_end,
                np.array(arc1_end) + vec1 * extension_factor,
                np.array(arc2_end) + vec2 * extension_factor,
                arc2_end
            ])

            target_hull = to_hull_poly.exterior
            to_hull_in_arc = target_hull.intersection(arc_polygon)

            if to_hull_in_arc.is_empty :
                continue # not in arc

            range_measure = LineString(shapely.ops.nearest_points(from_hull_poly.exterior, to_hull_in_arc))

            for hull in HullSection :
                if hull != to_hull and range_measure.crosses(defender_poly[hull].exterior) :
                    is_blocked = True
                    break
            if is_blocked : continue

            distance = range_measure.length

            if distance <= CLOSE_RANGE : 
                measure_dict[from_hull][to_hull] = AttackRange.CLOSE # close range
                target_dict[from_hull] = True
            elif distance <= MEDIUM_RANGE : 
                measure_dict[from_hull][to_hull] = AttackRange.MEDIUM # medium range
                target_dict[from_hull] = True
            elif distance <= LONG_RANGE : 
                measure_dict[from_hull][to_hull] = AttackRange.LONG # long range
                target_dict[from_hull] = True
            else : measure_dict[from_hull][to_hull] = AttackRange.EXTREME # invalid range (extreme)

    return target_dict, measure_dict

@lru_cache(maxsize=None)
def _cached_obstruction(targeting_point : tuple[tuple[float, float], tuple[float, float]], ship_state : tuple[str, float, float, float]) -> bool :
    line_of_sight : LineString = LineString(targeting_point)
    token_poly : Polygon = _cached_polygons(ship_state)['token']

    return line_of_sight.crosses(token_poly.exterior)

@lru_cache(maxsize=None)
def _cached_overlapping(self_state : tuple[str, float, float, float], ship_state : tuple[str, float, float, float]) -> bool :
    self_coordinate = _cached_coordinate(self_state)['base_corners']
    other_coordinate = _cached_coordinate(ship_state)['base_corners']

    def get_axes(corners):
        """
        Gets the two unique perpendicular axes from a rectangle's corners.
        An axis is a normalized vector perpendicular to an edge.
        """
        axes = np.zeros((2, 2))
        
        # Axis for the first edge (e.g., from left-front to right-front)
        edge1 = corners[1] - corners[0]
        # The normal is (-edge.y, edge.x)
        normal1 = np.array([-edge1[1], edge1[0]])
        norm1_len = np.sqrt(normal1[0]**2 + normal1[1]**2)
        if norm1_len > 0:
            axes[0] = normal1 / norm1_len # Normalize the axis
        
        # Axis for the second edge (e.g., from right-front to right-rear)
        edge2 = corners[2] - corners[1]
        normal2 = np.array([-edge2[1], edge2[0]])
        norm2_len = np.sqrt(normal2[0]**2 + normal2[1]**2)
        if norm2_len > 0:
            axes[1] = normal2 / norm2_len # Normalize the axis
            
        return axes

    def check_overlap_numpy(corners1, corners2):
        """
        Checks if two rotated rectangles overlap using an AABB pre-check
        followed by the Separating Axis Theorem (SAT).

        Args:
            corners1 (np.ndarray): A 4x2 NumPy array of the first rectangle's corners.
            corners2 (np.ndarray): A 4x2 NumPy array of the second rectangle's corners.

        Returns:
            bool: True if they overlap, False otherwise.
        """
        # --- Step 1: Fast Axis-Aligned Bounding Box (AABB) Pre-Check ---
        min1_x, min1_y = np.min(corners1, axis=0)
        max1_x, max1_y = np.max(corners1, axis=0)
        min2_x, min2_y = np.min(corners2, axis=0)
        max2_x, max2_y = np.max(corners2, axis=0)
        
        if max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y:
            return False

        # --- Step 2: Separating Axis Theorem (SAT) ---
        axes1 = get_axes(corners1)
        axes2 = get_axes(corners2)
        
        all_axes = np.vstack((axes1, axes2))
        
        for axis in all_axes:
            # Project all 8 corners onto the current axis
            proj1 = np.dot(corners1, axis)
            proj2 = np.dot(corners2, axis)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation.
            if max1 < min2 or max2 < min1:
                return False

        return True
    
    return check_overlap_numpy(self_coordinate, other_coordinate)

@lru_cache(maxsize=None)
def _cached_distance(self_state : tuple[str, float, float, float], ship_state : tuple[str, float, float, float]) -> float :
    self_poly : Polygon = _cached_polygons(self_state)['base']
    ship_poly : Polygon = _cached_polygons(ship_state)['base']

    return self_poly.distance(ship_poly)

@lru_cache(maxsize=None)
def _cached_maneuver_tool(size_class :SizeClass, course : tuple[int, ...], placement : int) -> tuple[np.ndarray, float]:
    """
    translate maneuver tool coordination to ship coordination
    from ship.(x,y) to (x,y) after maneuver
    """
    if not course :
        return np.array([0.0, 0.0]), 0.0

    yaw_changes = np.array([0] + list(course)) * (math.pi / 8)
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


    tool_offset = placement * (SHIP_BASE_SIZE[size_class][0]/2 + TOOL_WIDTH_HALF)
    ship_to_tool = np.array([tool_offset, (SHIP_BASE_SIZE[size_class][1] - SHIP_TOKEN_SIZE[size_class][1])/2])
    c, s = np.cos(-final_orientation), np.sin(-final_orientation)
    rotation = np.array([[c, -s], [s, c]])
    tool_to_ship = rotation @ -ship_to_tool


    # --- Step 4: Calculate final position ---
    final_position = ship_to_tool + np.array([total_dx, total_dy]) + tool_to_ship
    return final_position, final_orientation