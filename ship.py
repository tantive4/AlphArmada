from __future__ import annotations
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
import shapely.ops
import numpy as np
import math
from enum import IntEnum, Enum
from typing import TYPE_CHECKING
from dice import Dice, Critical
from defense_token import DefenseToken
from measurement import AttackRange, CLOSE_RANGE, MEDIUM_RANGE, LONG_RANGE
import itertools
from functools import lru_cache
if TYPE_CHECKING:
    from armada import Armada
import json
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

class Command(Enum) :
    NAVIGATION = 'NAV'
    # SQUADRON = 'SQAD'
    # ENGINEERING = 'ENGINEER'
    CONCENTRATE_FIRE = 'CONFIRE'
    def __str__(self):
        return self.value
    __repr__ = __str__


SHIP_BASE_SIZE : dict[SizeClass, tuple]= {SizeClass.SMALL : (43, 71), SizeClass.MEDIUM : (63, 102), SizeClass.LARGE : (77.5, 129)}
SHIP_TOKEN_SIZE :  dict[SizeClass, tuple] = {SizeClass.SMALL : (38.5, 70.25), SizeClass.MEDIUM : (58.5, 101.5)}
TOOL_WIDTH : float= 15.25
TOOL_LENGTH : float= 46.13 
TOOL_PART_LENGTH : float = 22.27
ROTATION_MATRIX_90_CCW = np.array([[0, -1], [1, 0]])
ROTATION_MATRICES = [np.linalg.matrix_power(ROTATION_MATRIX_90_CCW, i) for i in range(4)]

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
        self.defense_tokens : list[DefenseToken] = [DefenseToken(token_type, token_index) for token_index, token_type in enumerate(ship_dict['defense_token'])]
        self.navchart : dict[str, list[int]] = ship_dict['navchart']
        self.max_shield : list[int] = ship_dict['shield']
        self.point : int = ship_dict['point']
        self.command_value : int = ship_dict['command']
        self.destroyed = False

        self.front_arc : tuple[float, float] = (ship_dict['front_arc_center'], ship_dict['front_arc_end']) 
        self.rear_arc : tuple[float, float] = (ship_dict['rear_arc_center'], ship_dict['rear_arc_end'])
        
        self._create_template_geometries(ship_dict)
        self._course_cache : dict[tuple[int, bool], list[list[int]]]= {}
        
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
        self.x = x
        self.y = y
        self.orientation = orientation
        
        self.speed = speed
        self.hull = self.max_hull
        self.shield = {HullSection.FRONT : self.max_shield[0], 
                       HullSection.RIGHT : self.max_shield[1], 
                       HullSection.REAR : self.max_shield[2], 
                       HullSection.LEFT : self.max_shield[1]} # [Front, Right, Rear, Left]
        self.ship_id = ship_id
        self.command_stack : list[Command] = []
        self.command_dial : list[Command] = []
        self.command_token : list[Command] = []
        self.resolved_command : list[Command] = []
        self.attack_count : int = 0
        self.attack_possible_hull = [True, True, True, True]
        self.target_exist_hull = [True, True, True, True]
        self._set_coordination()
        self.refresh()
    
    def asign_command(self, Command) -> None :
        if len(self.command_stack) >= self.command_value : raise ValueError("Cannot asigne more command then Command Value")
        self.command_stack.append(Command)


    def destroy(self) -> None:
        self.destroyed = True
        self.hull = 0
        self.shield = {hull : 0 for hull in HullSection}
        for token in self.defense_tokens :
            if not token.discarded : token.discard()
        self.game.visualize(f'{self} is destroyed!')

    def refresh(self) -> None:
        self.activated = False
        for token in self.defense_tokens:
            if not token.discarded:
                token.ready()

    def end_activation(self) -> None :
        self.activated = True
        self.attack_possible_hull = [True, True, True, True]
        self.target_exist_hull = [True, True, True, True]
        self.attack_count = 0
        self.command_dial = []
        self.resolved_command = []

    def move_ship(self, course : list[int], placement : int) -> None:
        # speed 0 maneuver
        if not course :
            self.game.visualize(f'{self} executes speed 0 maneuver.')
            return

        original_x, original_y, original_orientaion = self.x, self.y, self.orientation

        joint_coordination, joint_orientaion = self._tool_coordination(course, placement)

        overlap_list = [False for _ in self.game.ships]

        while True :
            self._maneuver_to_coordination(placement, joint_coordination[-1], joint_orientaion[-1])
            self.game.visualize(f'{self} executes speed {len(joint_orientaion) - 1} maneuver.', joint_coordination)

            current_overlap = self.is_overlap()

            if not any(current_overlap):
                break
            self.game.visualize(f'{self} overlaps ships at speed {len(joint_orientaion) - 1} maneuver.')

            overlap_list = [overlap_list[i] or current_overlap[i] for i in range(len(overlap_list))]

            self.x, self.y, self.orientation = original_x, original_y, original_orientaion
            self._set_coordination()
            del joint_coordination[-2 :]
            del joint_orientaion[-1] 

            if len(joint_coordination) == 1 : break

        if self.out_of_board() :
            self.game.visualize(f'{self} is out of board!')
            self.destroy()

        self.overlap(overlap_list)








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
            [ (self.base_size[0] + TOOL_WIDTH)/2, (self.base_size[1]-self.token_size[1])/2],
            [-(self.base_size[0] + TOOL_WIDTH)/2, (self.base_size[1]-self.token_size[1])/2],
            [0, -self.token_size[1]/2]
        ])

    
    def _set_coordination(self) -> None:
        """
        Transforms template vertices using NumPy and creates final Polygon objects.
        This is the fastest and safest method, avoiding both repeated calculations and the need for .buffer(0).
        """
        c, s = math.cos(-self.orientation), math.sin(-self.orientation)
        rotation_matrix = np.array([[c, -s], [s, c]])
        translation_vector = np.array([self.x, self.y])

        # Transform vertices and create final Polygon objects
        self.ship_base = Polygon((self.template_base_vertices @ rotation_matrix.T) + translation_vector)
        self.ship_token = Polygon((self.template_token_vertices @ rotation_matrix.T) + translation_vector)

        self.hull_polygon : dict[HullSection, Polygon]= {}
        for section, vertices in self.template_hull_vertices.items():
            self.hull_polygon[section] = Polygon((vertices @ rotation_matrix.T) + translation_vector)

        final_targeting_points = (self.template_targeting_points_and_maneuver_tool_insert @ rotation_matrix.T) + translation_vector
        self.targeting_point : dict[HullSection, tuple[int, int]] = {hull : tuple(final_targeting_points[hull.value]) for hull in HullSection}
        self.tool_coords : dict[int, tuple[float, float]]  = {1 : tuple(final_targeting_points[4]),
                            -1: tuple(final_targeting_points[5])}
        self.center_point = final_targeting_points[6]
        self.orientation_vector = np.array([-s, c])


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
        line_of_sight = LineString([self.targeting_point[from_hull], to_ship.targeting_point[to_hull]])

        obstructed_line_of_sight = False
        for ship in self.game.ships:

            if ship.ship_id == self.ship_id or ship.ship_id == to_ship.ship_id:
                continue
            if ship.destroyed:
                continue

            if line_of_sight.crosses(ship.ship_token.exterior):
                obstructed_line_of_sight = True
                break

        return obstructed_line_of_sight

    def gather_dice(self, attack_hull : HullSection, attack_range : AttackRange) -> dict[Dice, int] :
        attack_pool = self.battery[attack_hull].copy()
        attack_pool = {dice_type : self.battery[attack_hull][dice_type.value] for dice_type in Dice}
        for dice_type in Dice :
            if dice_type.value < attack_range.value: attack_pool[dice_type] = 0
        return attack_pool

    def defend(self, defend_hull : HullSection, total_damage : int, critical: Critical | None) -> None:
        # Absorb damage with shields first
        shield_damage = min(total_damage, self.shield[defend_hull])
        self.shield[defend_hull] -= shield_damage
        total_damage -= shield_damage

        # Apply remaining damage to the hull
        if total_damage > 0:
            self.hull -= total_damage
            if critical == Critical.STANDARD : self.hull -= 1 # Structural Damage

        if self.hull <= 0 : self.destroy()



    def get_valid_target_hull(self, attack_hull : HullSection, defend_ship : Ship) -> list[ HullSection]:
        """
        Get a list of valid targets for the given attacking hull section.

        Args:
            attack_hull (HullSection): The attacking hull section.
            defend_ship (Ship): The defending ship.

        Returns:
            valid_targets (list[HullSection]): A list of tuples containing the target ship and the targeted hull section.
        """
        
        valid_targets = []
        vector_to_defender = np.array([defend_ship.center_point[0] - self.center_point[0],
                                    defend_ship.center_point[1] - self.center_point[1]])


        for target_hull in HullSection:

            defend_orientation_vector = defend_ship.orientation_vector @ ROTATION_MATRICES[target_hull.value]
            dot_product = np.dot(vector_to_defender, defend_orientation_vector)
            if dot_product / np.linalg.norm(vector_to_defender) >= 0.25 : continue

            attack_range = _cached_measurements(self.get_ship_hash_state(), defend_ship.get_ship_hash_state(), attack_hull, target_hull)

            # arc and range
            if attack_range in (AttackRange.INVALID, AttackRange.EXTREME) : continue

            # gather attack dice
            dice_count = sum(self.gather_dice(attack_hull, attack_range).values())
            if dice_count == 0 : continue
            elif dice_count == 1:
                if self.measure_line_of_sight(attack_hull, defend_ship, target_hull) : continue
            else : valid_targets.append(target_hull)

        return valid_targets

    def get_valid_target_ship(self, attack_hull : HullSection) -> list[Ship]:
        """
        Get a list of valid target ships for the given attacking hull section.

        Args:
            attack_hull (HullSection): The attacking hull section.

        Returns:
            valid_targets (list[Ship]): A list of valid target ships.
        """
        valid_targets = []
        attack_orientation_vector = self.orientation_vector @ ROTATION_MATRICES[attack_hull.value]

        for ship in self.game.ships:
            if ship.ship_id == self.ship_id or ship.destroyed or ship.player == self.player: continue

            distance_to_target = math.dist(self.center_point, ship.center_point)
            if distance_to_target > LONG_RANGE * 2 : continue

            
            vector_to_target = np.array([ship.center_point[0] - self.center_point[0], 
                                        ship.center_point[1] - self.center_point[1]])
            
            dot_product = np.dot(attack_orientation_vector, vector_to_target)
            if dot_product / distance_to_target < -0.25 : continue

            if self.get_valid_target_hull(attack_hull, ship):
                valid_targets.append(ship)

        return valid_targets
    

    def get_valid_attack_hull(self) -> list[HullSection]:
        """
        Get a list of valid attacking hull sections for the ship.

        Returns:
            valid_attacker (list[HullSection]): A list of valid attacking hull sections.
        """
        valid_attacker = []
        for hull in HullSection:
            if not self.attack_possible_hull[hull.value] : continue

            if self.target_exist_hull[hull.value] and self.get_valid_target_ship(hull):
                valid_attacker.append(hull)
            else :
                self.target_exist_hull[hull.value] = False 
        return valid_attacker

    def get_critical_effect(self, black_crit : bool, blue_crit : bool, red_crit : bool) -> list[Critical] :
        critical_list : list[Critical] = []
        if black_crit or blue_crit or red_crit :
            critical_list.append(Critical.STANDARD)
        return critical_list


        


# sub method for execute maneuver

    def _tool_coordination(self, course : list[int], placement : int) -> tuple[list[tuple[float, float]], list[float]]:
        """
        Calculates the coordinates and orientations along a maneuver tool's path using NumPy.
        """
        if not course:
            # For speed 0, return the starting point and orientation
            return [self.tool_coords[placement]], [self.orientation]

        # --- Step 1: Set up initial conditions ---
        initial_position = np.array(self.tool_coords[placement])
        initial_orientation = self.orientation
        speed = len(course)

        # --- Step 2: Calculate all joint orientations at once ---
        # This array will have shape (speed + 1) and includes the initial orientation
        yaw_changes = np.array([0] + course) * (math.pi / 8)
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
    
    def _maneuver_to_coordination(self, placement : int, tool_coordination : tuple[float, float], tool_orientaion : float) -> None :
        """
        move the ship to given toolool coordination

        Args:
            placement (int): 1 for right, -1 for left
            tool_coordination (tuple) : endpoint of maneuver tool
            tool_orientaion (float) : orientation of maneuver tool
        """
        self.x = tool_coordination[0] - placement * math.cos(tool_orientaion) * (self.base_size[0] + TOOL_WIDTH) / 2
        self.y = tool_coordination[1] + placement * math.sin(tool_orientaion) * (self.base_size[0] + TOOL_WIDTH) / 2
        self.orientation = tool_orientaion
        self._set_coordination()
        
    def is_overlap(self) -> list[bool] :
        """
        determines which ship overlaps to this ship at current location
        
        Returns:
            overlap_list (list[bool]): A list indicating which ships were overlapped.
        """
        overlap_list = []
        for ship in self.game.ships:
            if self.ship_id == ship.ship_id or ship.destroyed :
                overlap_list.append(False)
                continue
            
            if self.ship_base.intersects(ship.ship_base) and not self.ship_base.touches(ship.ship_base):
                overlap_list.append(True)
            else:
                overlap_list.append(False)
        return overlap_list

    def overlap(self, overlap_list : list[bool]) -> None:
        """
        Determines which of the overlapping ships is closest and handles the collision.

        Args:
            overlap_list (list[bool]): A list indicating which ships were overlapped.
        """
        closest_ship = None
        min_distance = float('inf')

        for i, is_overlapped in enumerate(overlap_list):
            if is_overlapped:
                other_ship = self.game.ships[i]
                
                # Calculate the distance between the two ship bases.
                distance = self.ship_base.distance(other_ship.ship_base)
                
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
        return not self.ship_base.within(self.game.game_board)
    
    def get_valid_speed(self) -> list[int]:
        """
        Get a list of valid speeds for the ship based on its navchart.

        Returns:
            list[int]: A list of valid speeds.
        """
        valid_speed = []
        speed_change : int = int(Command.NAVIGATION in self.command_dial) + int(Command.NAVIGATION in self.command_token)

        for speed in range(5):
            if abs(speed - self.speed) > speed_change:
                continue
            if self.navchart.get(str(speed)) is not None or speed == 0:
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
            if abs(yaw - 2) > self.navchart[str(speed)][joint]:
                continue
            valid_yaw.append(yaw -2)

        return valid_yaw
    
    def nav_command_used(self, course: list[int]) -> tuple[bool, bool] :
            nav_dial_used = False
            nav_token_used = False
            new_speed = len(course)

            # 1. SPEED CHANGE VALIDATION
            speed_change = abs(new_speed - self.speed)

            if speed_change > 2:
                raise ValueError(f"Invalid speed change of {speed_change}. Maximum is 2.")

            if speed_change == 1:
                # Must use one NAVIGATE command from either dial or token
                if Command.NAVIGATION in self.command_dial:
                    nav_dial_used = True
                elif Command.NAVIGATION in self.command_token:
                    nav_token_used = True
                else:
                    raise ValueError("Speed change of 1 requires a NAVIGATE command from a dial or token.")

            if speed_change == 2:
                # Must use two NAVIGATE commands, one from each source
                if Command.NAVIGATION in self.command_dial and Command.NAVIGATION in self.command_token:
                    nav_dial_used = True
                    nav_token_used = True
                else:
                    raise ValueError("Speed change of 2 requires NAVIGATE commands from BOTH a dial and a token.")

            # 2. COURSE VALIDITY CHECK (for extra clicks)
            is_standard = self.is_standard_course(course)
            
            if not is_standard:
                # This is a special course that requires an extra click from the command dial.
                if Command.NAVIGATION in self.command_dial:
                    nav_dial_used = True
                else:
                    raise ValueError("This course requires an extra click, which needs a NAVIGATE command from the dial.")
            return nav_dial_used, nav_token_used
    
    def is_standard_course(self, course:list[int]) -> bool :
        """
        Checks if a given course is a standard maneuver for a given speed,
        without using any special abilities like adding a click.
        """
        speed = len(course)
        if speed == 0 : return True

        for joint, yaw in enumerate(course) :
            if abs(yaw) > self.navchart[str(speed)][joint] : return False
        return True

    def get_all_possible_courses(self, speed: int) -> list[list[int]]:
        """
        Gets all possible maneuver courses for a given speed.

        If can_add_click is True, it will also generate additional courses by modifying
        a single joint of any originally valid course by one click (yaw).

        Args:
            speed (int): The maneuver speed.

        Returns:
            A list of all unique, valid course lists.
        """
        has_nav_dial = Command.NAVIGATION in self.command_dial
        cache_key = (speed, has_nav_dial)

        if cache_key in self._course_cache :
            return self._course_cache[cache_key]
        
        if speed == 0:
            return [[]] # A speed 0 maneuver has an empty course

        # 1. Get the base valid yaw options for each joint from the nav chart.
        original_yaw_options = [self.get_valid_yaw(speed, joint) for joint in range(speed)]

        # 2. Generate all standard courses using the Cartesian product.
        # Using a set handles duplicates automatically.
        all_courses = set(itertools.product(*original_yaw_options))

        # 3. If the special condition isn't met, return the standard courses.
        if not Command.NAVIGATION in self.command_dial:
            return [list(course) for course in all_courses]

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


        final_result = [list(course) for course in all_courses]
        self._course_cache[cache_key] = final_result
        return final_result
    
    def get_valid_placement(self, course : list[int]) -> list[int]:
        """
        Get a list of valid placements for the ship based on its navchart.

        Args:
            course (list[int]): The course of the ship.

        Returns:
            list[int]: A list of valid placements. 
                1 for right, -1 for left
        """
        speed = len(course)
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
            "shield": self.shield.copy(),
            "destroyed": self.destroyed,
            "activated": self.activated,
            "command_stack": list(self.command_stack),
            "command_dial": list(self.command_dial),
            "command_token": list(self.command_token),
            "resolved_command": list(self.resolved_command),
            "attack_count": self.attack_count,
            "attack_possible_hull": list(self.attack_possible_hull),
            "target_exist_hull": list(self.target_exist_hull),
            "defense_tokens": [(dt.readied, dt.discarded, dt.accuracy) for dt in self.defense_tokens]
        }
    
    def revert_snapshot(self, snapshot: dict) -> None:
        """Restores the ship's state from a snapshot."""
        self.x = snapshot["x"]
        self.y = snapshot["y"]
        self.orientation = snapshot["orientation"]
        self.speed = snapshot["speed"]
        self.hull = snapshot["hull"]
        self.shield = snapshot["shield"].copy()
        self.destroyed = snapshot["destroyed"]
        self.activated = snapshot["activated"]
        self.command_stack = snapshot["command_stack"].copy()
        self.command_dial = snapshot["command_dial"].copy()
        self.command_token = snapshot["command_token"].copy()
        self.resolved_command = snapshot["resolved_command"].copy()
        self.attack_count = snapshot["attack_count"]
        self.attack_possible_hull = snapshot["attack_possible_hull"].copy()
        self.target_exist_hull = snapshot["target_exist_hull"].copy()

        for i, token_state in enumerate(snapshot["defense_tokens"]):
            self.defense_tokens[i].readied, self.defense_tokens[i].discarded, self.defense_tokens[i].accuracy = token_state

        # Crucially, update the geometry after restoring coordinates
        self._set_coordination()

    def get_ship_hash_state(self) -> tuple[str, float, float, float]:
        """Returns a hashable tuple representing the ship's state."""
        return (self.name, self.x, self.y, self.orientation)
    

@lru_cache(maxsize=None)
def _cached_coordinate(ship_state : tuple[str, float, float, float]) -> np.ndarray :

    ship_dict = SHIP_DATA[ship_state[0]]

    front_arc : tuple[float, float] = (ship_dict['front_arc_center'], ship_dict['front_arc_end']) 
    rear_arc : tuple[float, float] = (ship_dict['rear_arc_center'], ship_dict['rear_arc_end'])
    token_size : tuple[float, float] = SHIP_TOKEN_SIZE[SizeClass[ship_dict['size_class']]]
    token_half_w = token_size[0] / 2

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
    [- ship_dict['side_targeting_point'][0], -ship_dict['side_targeting_point'][1]]
    ])

    c, s = math.cos(-ship_state[3]), math.sin(-ship_state[3])
    rotation_matrix = np.array([[c, -s], [s, c]])
    translation_vector = np.array([ship_state[1], ship_state[2]])

    return template_vertices @ rotation_matrix.T + translation_vector


@lru_cache(maxsize=20000)
def _cached_hull_polygons(ship_state: tuple[str, float, float, float]) -> dict[HullSection, Polygon]:
    """
    Creates and caches all hull polygons for a given ship state.
    """
    coords = _cached_coordinate(ship_state)
    return {
        HullSection.FRONT: Polygon([coords[0], coords[2], coords[7], coords[6], coords[1]]),
        HullSection.RIGHT: Polygon([coords[0], coords[2], coords[5], coords[3]]),
        HullSection.REAR: Polygon([coords[3], coords[5], coords[9], coords[8], coords[4]]),
        HullSection.LEFT: Polygon([coords[0], coords[1], coords[4], coords[3]]),
    }


@lru_cache(maxsize=None)
def _cached_measurements(attacker_state : tuple[str, float, float, float], defender_state : tuple[str, float, float, float], from_hull : HullSection, to_hull : HullSection, extension_factor=500):
    """
    return:
        attack_range (AttackRange)
    """

    attacker_coords = _cached_coordinate(attacker_state)
    defender_coords = _cached_coordinate(defender_state)
    
    from_hull_poly = _cached_hull_polygons(attacker_state)[from_hull]
    to_hull_poly_dict = _cached_hull_polygons(defender_state)

    from_hull_targeting_pt = attacker_coords[from_hull.value + 10]
    to_hull_targeting_pt = defender_coords[to_hull.value + 10]


    # Line of Sight blocked
    line_of_sight = LineString([from_hull_targeting_pt, to_hull_targeting_pt])


    for hull in HullSection:
        if hull == to_hull:
            continue
        if line_of_sight.crosses(to_hull_poly_dict[hull].exterior):
            return AttackRange.INVALID

    # Range
    hull_coords = from_hull_poly.exterior.coords
    if from_hull in (HullSection.FRONT, HullSection.REAR) :
        arc1_center, arc1_end = hull_coords[0], hull_coords[1]
        arc2_center, arc2_end = hull_coords[-2], hull_coords[-3] 
    else :
        arc1_center, arc1_end = hull_coords[0], hull_coords[1]
        arc2_center, arc2_end = hull_coords[-1], hull_coords[-2] 

    # Build the arc polygon. This logic works for ALL hull sections.
    vec1 = np.array(arc1_end) - np.array(arc1_center)
    vec2 = np.array(arc2_end) - np.array(arc2_center) # Note: vector points away from the ship
    arc_polygon = Polygon([
        arc1_end,
        np.array(arc1_end) + vec1 * extension_factor,
        np.array(arc2_end) + vec2 * extension_factor,
        arc2_end
    ])

    target_hull = to_hull_poly_dict[to_hull].exterior
    to_hull_in_arc = target_hull.intersection(arc_polygon)

    if to_hull_in_arc.is_empty :
        return AttackRange.INVALID # not in arc

    range_measure = LineString(shapely.ops.nearest_points(from_hull_poly.exterior, to_hull_in_arc))

    for hull in HullSection :
        if hull != to_hull and range_measure.crosses(to_hull_poly_dict[hull].exterior) :
            return AttackRange.INVALID # range not valid
    distance = range_measure.length

    if distance <= CLOSE_RANGE : return AttackRange.CLOSE # close range
    elif distance <= MEDIUM_RANGE : return AttackRange.MEDIUM # medium range
    elif distance <= LONG_RANGE : return AttackRange.LONG # long range
    else : return AttackRange.EXTREME # extreme range
