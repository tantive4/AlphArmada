from __future__ import annotations
from shapely.geometry import Polygon, LineString
import shapely.ops
import numpy as np
import math
from enum import IntEnum, Enum
from typing import TYPE_CHECKING
from dice import Dice, Critical
from defense_token import DefenseToken
from measurement import AttackRange, CLOSE_RANGE, MEDIUM_RANGE, LONG_RANGE
import itertools
if TYPE_CHECKING:
    from armada import Armada


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

class Ship:
    def __init__(self, ship_dict : dict, player : int) -> None:
        self.player : int = player
        self.name : str = ship_dict['name']

        self.max_hull : int = ship_dict['hull']
        self.size_class : SizeClass = SizeClass[ship_dict['size'].upper()]
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

        self.front_arc : tuple[float, float] = (ship_dict['front_arc_center'], ship_dict['front_arc_end']) 
        self.rear_arc : tuple[float, float] = (ship_dict['rear_arc_center'], ship_dict['rear_arc_end'])

        self.targeting_point_dict : dict[str, tuple[float, float]] = {
            'front' : (0, ship_dict['front_targeting_point']),
            'right' : (ship_dict['side_targeting_point'][0], ship_dict['side_targeting_point'][1]),
            'rear' : (0, ship_dict['rear_targeting_point']),
            'left' : (- ship_dict['side_targeting_point'][0], ship_dict['side_targeting_point'][1])}
        
    def __str__(self):
        return self.name
    __repr__ = __str__

    def deploy(self, game : Armada , x : float, y : float, orientation : float, speed : int, ship_id : int) -> None:
        """
        deploy the ship to the game board

        Args:
            game (Armada) : the game
            (x, y) (float) : coordination of front center of the ship token
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
        self.destroyed = False
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
        self.shield = [0,0,0,0]
        self.battery = {hull : [0,0,0] for hull in HullSection}
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

    def _get_coordination(self, add_x : float, add_y : float) -> tuple[float, float] :
        """
        Args:
            vector (tuple): (x, y) transition tuple
        Returns:
            (x, y) point after transition from top center of the ship, considering current orientation
        """
        x, y = self.x, self.y
        rotated_add_x = add_x * math.cos(self.orientation) + add_y * math.sin(self.orientation)
        rotated_add_y = -add_x * math.sin(self.orientation) + add_y * math.cos(self.orientation)
        return (x + rotated_add_x, y + rotated_add_y)
    
    def _set_coordination(self) -> None: 
        """
        define coordination for ship measuring points

        update when ship is moved
        """
        self.front_right_token = self._get_coordination(self.token_size[0] / 2,0)
        self.front_left_token = self._get_coordination(- self.token_size[0] / 2,0)
        self.rear_right_token = self._get_coordination(self.token_size[0] / 2, - self.token_size[1])
        self.rear_left_token = self._get_coordination(- self.token_size[0] / 2, - self.token_size[1])

        self.front_right_arc = self._get_coordination(self.token_size[0]/2, - self.front_arc[1])
        self.front_left_arc = self._get_coordination(-self.token_size[0]/2, - self.front_arc[1])
        self.rear_right_arc = self._get_coordination(self.token_size[0]/2, - self.rear_arc[1])
        self.rear_left_arc = self._get_coordination(-self.token_size[0]/2, - self.rear_arc[1])

        self.front_arc_center = self._get_coordination(0, -self.front_arc[0])
        self.rear_arc_center = self._get_coordination(0, -self.rear_arc[0])

        self.front_right_base = self._get_coordination(self.base_size[0] / 2, (self.base_size[1] - self.token_size[1]) / 2)
        self.front_left_base = self._get_coordination(- self.base_size[0] / 2,(self.base_size[1] - self.token_size[1]) / 2)
        self.rear_right_base = self._get_coordination(self.base_size[0] / 2, (self.base_size[1] - self.token_size[1]) / 2 - self.base_size[1])
        self.rear_left_base = self._get_coordination(- self.base_size[0] / 2, (self.base_size[1] - self.token_size[1]) / 2 - self.base_size[1])
        
        self._make_polygon()

        self.targeting_point : dict[HullSection, tuple[float, float]]= {
            HullSection.FRONT : self._get_coordination(self.targeting_point_dict['front'][0], -self.targeting_point_dict['front'][1]),
            HullSection.RIGHT : self._get_coordination(self.targeting_point_dict['right'][0], -self.targeting_point_dict['right'][1]),
            HullSection.REAR : self._get_coordination(self.targeting_point_dict['rear'][0], -self.targeting_point_dict['rear'][1]),
            HullSection.LEFT : self._get_coordination(self.targeting_point_dict['left'][0], -self.targeting_point_dict['left'][1])}

    def _make_polygon(self) -> None:
        self.ship_token = Polygon([
            self.front_right_token,
            self.front_left_token,
            self.rear_left_token,
            self.rear_right_token
        ]).buffer(0)

        self.ship_base = Polygon([
            self.front_right_base,
            self.front_left_base,
            self.rear_left_base,
            self.rear_right_base
        ]).buffer(0)

        front_hull = Polygon([
            self.front_right_token,
            self.front_right_arc,
            self.front_arc_center,
            self.front_left_arc,
            self.front_left_token
                ]).buffer(0)
        right_hull = Polygon([
            self.front_arc_center,
            self.front_right_arc,
            self.rear_right_arc,
            self.rear_arc_center
                ]).buffer(0)
        rear_hull = Polygon([
            self.rear_right_arc,
            self.rear_right_token,
            self.rear_left_token,
            self.rear_left_arc,
            self.rear_arc_center
                ]).buffer(0)
        left_hull = Polygon([
            self.front_arc_center,
            self.front_left_arc,
            self.rear_left_arc,
            self.rear_arc_center
                ]).buffer(0)

        self.hull_polygon : dict[HullSection, Polygon]= {HullSection.FRONT : front_hull, 
                                                         HullSection.RIGHT : right_hull, 
                                                         HullSection.REAR : rear_hull, 
                                                         HullSection.LEFT : left_hull}


# sub method for attack

    def measure_arc_and_range(self, from_hull : HullSection, to_ship : "Ship", to_hull : HullSection, extension_factor=1e4) -> AttackRange:
        """Measures the range and validity of a firing arc to a target.

        This method checks if a target hull is within the firing ship's
        specified arc and calculates the range if the line of sight is clear.

        Args:
            from_hull (HullSection): The index of the firing hull section (0-3).
            to_ship (Ship): The target Ship object.
            to_hull (HullSection): The index of the target hull section (0-3).

        Returns:
            AttackRange: An integer code representing the result.
                -1: Not in arc or invalid range.
                 0: Close range.
                 1: Medium range.
                 2: Long range.
                 3: Extreme range.
        """
        if to_ship.destroyed : return AttackRange.INVALID

        if from_hull == HullSection.FRONT or from_hull == HullSection.RIGHT :
            arc1 = (self.front_arc_center, self.front_right_arc)
        else : 
            arc1 = (self.rear_arc_center, self.rear_left_arc)
        
        if from_hull == HullSection.RIGHT or from_hull == HullSection.REAR :
            arc2 = (self.rear_arc_center, self.rear_right_arc)
        else :
            arc2 = (self.front_arc_center, self.front_left_arc)

        arc1_vector = np.array(arc1[1]) - np.array(arc1[0])
        arc2_vector = np.array(arc2[1]) - np.array(arc2[0])

        arc_polygon = Polygon([arc1[0], arc1[1] + arc1_vector * extension_factor,
                            arc2[1] + arc2_vector * extension_factor, arc2[0]]).buffer(0)
        to_hull_in_arc = to_ship.hull_polygon[to_hull].intersection(arc_polygon)

        if to_hull_in_arc.is_empty or not isinstance(to_hull_in_arc, Polygon):
            return AttackRange.INVALID # not in arc

        range_measure = LineString(shapely.ops.nearest_points(self.hull_polygon[from_hull], to_hull_in_arc))

        for hull in HullSection :
            if hull != to_hull and range_measure.crosses(to_ship.hull_polygon[hull]) :
                return AttackRange.INVALID # range not valid
        distance = range_measure.length

        if distance <= CLOSE_RANGE : return AttackRange.CLOSE # close range
        elif distance <= MEDIUM_RANGE : return AttackRange.MEDIUM # medium range
        elif distance <= LONG_RANGE : return AttackRange.LONG # long range
        else : return AttackRange.EXTREME # extreme range

    def measure_line_of_sight(self, from_hull : HullSection, to_ship : Ship, to_hull : HullSection) -> tuple[bool, bool] :
        """
        Checks if the line of sight between two ships is obstructed.

        Args:
            from_hull (HullSection): The firing hull section of the attacking ship.
            to_ship (Ship): The target ship.
            to_hull (HullSection): The targeted hull section of the defending ship.

        Returns:
            (bool, bool): A tuple where:
                - The first boolean is True if blocked by another hull of the target ship.
                - The second boolean is True if obstructed by another ship.
        """
        line_of_sight = LineString([self.targeting_point[from_hull], to_ship.targeting_point[to_hull]])

        blocked_line_of_sight = False

        for hull in HullSection:
            if hull == to_hull:
                continue

            if line_of_sight.crosses(to_ship.hull_polygon[hull]):
                blocked_line_of_sight = True
                break 

        obstructed_line_of_sight = False
        for ship in self.game.ships:

            if ship.ship_id == self.ship_id or ship.ship_id == to_ship.ship_id:
                continue
            if ship.destroyed:
                continue

            if line_of_sight.crosses(ship.ship_token):
                obstructed_line_of_sight = True
                break

        return blocked_line_of_sight, obstructed_line_of_sight

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



    def get_valid_target_hull(self, attack_hull : HullSection, defend_ship : "Ship") -> list[ HullSection]:
        """
        Get a list of valid targets for the given attacking hull section.

        Args:
            attack_hull (HullSection): The attacking hull section.
            defend_ship (Ship): The defending ship.

        Returns:
            valid_targets (list[HullSection]): A list of tuples containing the target ship and the targeted hull section.
        """
        valid_targets = []


        for target_hull in HullSection:

            # line of sight
            blocked, obstructed = self.measure_line_of_sight(attack_hull, defend_ship, target_hull)
            if blocked : continue

            # arc and range
            attack_range = self.measure_arc_and_range(attack_hull, defend_ship, target_hull)
            if attack_range == AttackRange.INVALID: continue  

            # gather attack dice
            if sum(self.gather_dice(attack_hull, attack_range).values()) <= int(obstructed): continue 

            valid_targets.append(target_hull)
        return valid_targets

    def get_valid_target_ship(self, attack_hull : HullSection) -> list["Ship"]:
        """
        Get a list of valid target ships for the given attacking hull section.

        Args:
            attack_hull (HullSection): The attacking hull section.

        Returns:
            valid_targets (list[Ship]): A list of valid target ships.
        """
        valid_targets = []
        for ship in self.game.ships:
            if ship.ship_id == self.ship_id or ship.destroyed or ship.player == self.player: continue
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

    def _tool_coordination(self, course : list[int], placement : int) -> tuple[list[tuple[float, float]], list[float]] :
        """Calculates the coordinates and orientations along a maneuver tool's path.

        This method simulates the placement of the maneuver tool against the
        ship's base and calculates the series of points and angles that define
        the maneuver course.

        Args:
            course (list[int]): A list of yaw adjustments for each joint of the maneuver tool. The length corresponds to the maneuver speed.
            placement (int): The side of the ship to place the tool. `1` for the right side, `-1` for the left side.

        Returns:
            A tuple containing two lists:
            - A list of (x, y) tuples representing the coordinates of each joint along the tool's path.
            - A list of orientations in radians at each joint along the tool's path.
        """
        (tool_x, tool_y) = self._get_coordination(placement * (self.base_size[0] + TOOL_WIDTH) / 2,0)
        tool_orientaion = self.orientation

        joint_coordination = [(tool_x, tool_y)]
        joint_orientation = [tool_orientaion]


        for joint in course :
            tool_x += math.sin(tool_orientaion) * TOOL_LENGTH
            tool_y += math.cos(tool_orientaion) * TOOL_LENGTH
            joint_coordination.append((tool_x, tool_y))

            tool_orientaion += joint * math.pi / 8

            tool_x += math.sin(tool_orientaion) * TOOL_PART_LENGTH
            tool_y += math.cos(tool_orientaion) * TOOL_PART_LENGTH
            joint_coordination.append((tool_x, tool_y))
            joint_orientation.append(tool_orientaion)
        

        return joint_coordination, joint_orientation
    
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
        
        return [list(course) for course in all_courses]
    
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