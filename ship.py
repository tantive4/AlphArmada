from shapely.geometry import Polygon, LineString
import shapely.ops
import numpy as np
import math
from dice import *
from enum import IntEnum
from typing import TYPE_CHECKING

# Conditionally import Armada only for type checking
if TYPE_CHECKING:
    from armada import Armada


class HullSection(IntEnum):
    FRONT = 0
    RIGHT = 1
    REAR = 2
    LEFT = 3

class Critical(IntEnum) :
    STANDARD = 0

SHIP_BASE_SIZE : dict[str, tuple]= {'small' : (43, 71), 'medium' :(63, 102), 'large' : (77.5, 129)}
SHIP_TOKEN_SIZE :  dict[str, tuple] = {'small' : (38.5, 70.25), 'medium' : (58.5, 101.5)}
TOOL_WIDTH : float= 15.25
TOOL_LENGTH : float= 46.13 
TOOL_PART_LENGTH : float = 22.27

class Ship:
    def __init__(self, ship_dict : dict, player : int) -> None:
        self.player : int = player
        self.name : str = ship_dict['name']

        self.max_hull : int = ship_dict['hull']
        self.size_class : str = ship_dict['size']
        self.token_size : tuple [float, float] = SHIP_TOKEN_SIZE[self.size_class]
        self.base_size : tuple [float, float] = SHIP_BASE_SIZE[self.size_class]

        self.battery :  tuple[list[int], list[int], list[int], list[int]] = (ship_dict['battery'][0], ship_dict['battery'][1], ship_dict['battery'][2], ship_dict['battery'][1]) # (Front, Right, Rear, Left)
        self.navchart : dict[str, list[int]] = ship_dict['navchart']
        self.max_shield : list[int] = ship_dict['shield']
        self.point : int = ship_dict['point']

        self.front_arc : tuple[float, float] = (ship_dict['front_arc_center'], ship_dict['front_arc_end']) 
        self.rear_arc : tuple[float, float] = (ship_dict['rear_arc_center'], ship_dict['rear_arc_end'])

        self.targeting_point_dict : dict[str, tuple[float, float]] = {
            'front' : (0, ship_dict['front_targeting_point']),
            'right' : (ship_dict['side_targeting_point'][0], ship_dict['side_targeting_point'][1]),
            'rear' : (0, ship_dict['rear_targeting_point']),
            'left' : (- ship_dict['side_targeting_point'][0], ship_dict['side_targeting_point'][1])}
    

    def deploy(self, game : "Armada" , x : float, y : float, orientation : float, speed : int, ship_id : int) -> None:
        """
        deploy the ship to the game board

        Args:
            game (Armada) : the game
            (x, y) (float) : coordination of front center of the ship token
            orientation (float) : ship orientation
            speed (int) : ship speed
            ship_id (int) : ship id
        """
        self.game : "Armada" = game
        self.x = x 
        self.y = y
        self.orientation = orientation
        
        self.speed = speed
        self.hull = self.max_hull
        self.shield = [self.max_shield[0], self.max_shield[1], self.max_shield[2], self.max_shield[1]] # [Front, Right, Rear, Left]
        self.destroyed = False
        self.ship_id = ship_id
        self._set_coordination()
        self.refresh()
        self.game.visualize(f'{self.name} is deployed.')

    def destroy(self) -> None:
        self.destroyed = True
        self.hull = 0
        self.shield = [0,0,0,0]
        self.battery = ([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
        self.game.visualize(f'{self.name} is destroyed!')

    def refresh(self) -> None:
        self.activated = False
        self.attack_possible_hull = [True, True, True, True]
        self.target_exist_hull = [True, True, True, True]
        self.attack_count = 0




# core activation method

    # def activate(self) -> None:



# ship activation sequence

    def reveal_command(self) :
        """pass"""
        pass

    # def attack(self) -> None:

    # def execute_maneuver(self) -> None:





# attack sequence

    # def declare_target(self) -> tuple[HullSection, "Ship", HullSection] | None :

    def roll_attack_dice(self, attack_hull : HullSection, defend_ship : "Ship", defend_hull : HullSection) -> list[list[int]] | None :
        self.attack_count += 1
        self.attack_possible_hull[attack_hull.value] = False
        
        # gathering dice
        attack_range = self.measure_arc_and_range(attack_hull, defend_ship, defend_hull)
        if attack_range == -1 : return # attack is canceled

        self.game.visualize(f'{self.name} attacks {defend_ship.name}! {attack_hull.name} to {defend_hull.name}! Range : {["close", "medium", "long"][attack_range]}')

        attack_pool = self.gather_dice(attack_hull, attack_range)

        if sum(attack_pool) == 0 : return # cannot gather attack dice

        # rolling dice
        attack_pool = roll_dice(attack_pool)
        self.game.visualize((f'''
          Dice Rolled!
          Black [Blank, Hit, Double] : {attack_pool[0]}
          Blue [Hit, Critical, Accuracy] : {attack_pool[1]}
          Red [Blank, Hit, Critical, Double, Accuracy] : {attack_pool[2]}
        '''))
        return attack_pool

    def resolve_attack_effect(self):
        """pass"""
        pass

    def spend_defense_token(self):
        """pass"""
        pass

    def resolve_damage(self, defend_ship : "Ship", defend_hull : HullSection, attack_pool : list[list[int]]) -> None :
        """
        Resolve the damage dealt to the defending ship.
        Args:
            defend_ship (Ship): The defending ship.
            defend_hull (HullSection): The hull section being attacked.
            attack_pool (list[list[int]]): The rolled attack dice.
        """
        black_critical = bool(attack_pool[CRIT_INDICES[0]])
        blue_critical = bool(attack_pool[CRIT_INDICES[1]])
        red_critical = bool(attack_pool[CRIT_INDICES[2]])
        total_damage = sum(
            sum(damage * dice for damage, dice in zip(damage_values, dice_counts)) for damage_values, dice_counts in zip(DAMAGE_INDICES, attack_pool)
            )
        
        if black_critical or blue_critical or red_critical :
            critical = Critical.STANDARD
        defend_ship.defend(defend_hull, total_damage, critical)


# execute maneuver sequence

    # def determine_course(self) -> tuple[list, int]:

    def move_ship(self, course : list[int], placement : int) -> None:
        original_x, original_y, original_orientaion = self.x, self.y, self.orientation

        joint_coordination, joint_orientaion = self._tool_coordination(course, placement)

        overlap_list = [False for _ in self.game.ships]

        while True :
            self._maneuver_to_coordination(placement, joint_coordination[-1], joint_orientaion[-1])
            self.game.visualize(f'{self.name} executes speed {len(joint_orientaion) - 1} maneuver.', joint_coordination)

            current_overlap = self.is_overlap()

            if not any(current_overlap):
                break
            self.game.visualize(f'{self.name} overlaps ships at speed {len(joint_orientaion) - 1} maneuver.')

            overlap_list = [overlap_list[i] or current_overlap[i] for i in range(len(overlap_list))]

            self.x, self.y, self.orientation = original_x, original_y, original_orientaion
            self._set_coordination()
            del joint_coordination[-2 :]
            del joint_orientaion[-1] 

            if len(joint_coordination) == 1 : break

        if self.out_of_board() :
            self.game.visualize(f'{self.name} is out of board!')
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

        self.targeting_point = [
            self._get_coordination(self.targeting_point_dict['front'][0], -self.targeting_point_dict['front'][1]),
            self._get_coordination(self.targeting_point_dict['right'][0], -self.targeting_point_dict['right'][1]),
            self._get_coordination(self.targeting_point_dict['rear'][0], -self.targeting_point_dict['rear'][1]),
            self._get_coordination(self.targeting_point_dict['left'][0], -self.targeting_point_dict['left'][1])]

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

        self.hull_polygon = [front_hull, right_hull, rear_hull, left_hull]


# sub method for attack

    def measure_arc_and_range(self, from_hull : HullSection, to_ship : "Ship", to_hull : HullSection, extension_factor=1e4) -> int:
        """Measures the range and validity of a firing arc to a target.

        This method checks if a target hull is within the firing ship's
        specified arc and calculates the range if the line of sight is clear.

        Args:
            from_hull (HullSection): The index of the firing hull section (0-3).
            to_ship (Ship): The target Ship object.
            to_hull (HullSection): The index of the target hull section (0-3).

        Returns:
            int: An integer code representing the result.
                -1: Not in arc or invalid range.
                 0: Close range.
                 1: Medium range.
                 2: Long range.
                 3: Extreme range.
        """
        if to_ship.destroyed : return -1

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
        to_hull_in_arc = to_ship.hull_polygon[to_hull.value].intersection(arc_polygon)

        if to_hull_in_arc.is_empty or not isinstance(to_hull_in_arc, Polygon):
            return -1 # not in arc
        else :
            range_measure = LineString(shapely.ops.nearest_points(self.hull_polygon[from_hull.value], to_hull_in_arc))

            for hull_index in range(4) :
                if hull_index != to_hull.value and  range_measure.crosses(to_ship.hull_polygon[hull_index]) :
                    return -1 # range not valid
            distance = range_measure.length

            if distance <= 123.3 : return 0 # close range
            elif distance <= 186.5 : return 1 # medium range
            elif distance <= 304.8 : return 2 # long range
            else : return 3 # extreme range

    def measure_line_of_sight(self, from_hull : HullSection, to_ship : "Ship", to_hull : HullSection) -> tuple[bool, bool] :
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
        line_of_sight = LineString([self.targeting_point[from_hull.value], to_ship.targeting_point[to_hull.value]])

        blocked_line_of_sight = False

        for i, hull_polygon in enumerate(to_ship.hull_polygon):
            if i == to_hull.value:
                continue

            if line_of_sight.crosses(hull_polygon):
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

    def gather_dice(self, attack_hull : HullSection, attack_range : int) -> list[int] :
        attack_pool = self.battery[attack_hull].copy()
        for i in range(3) :
            if i < attack_range: attack_pool[i] = 0
        return attack_pool

    def defend(self, defend_hull : HullSection, total_damage : int, critical: Critical | None = None ) -> None:
        
        self.game.visualize(f'{self.name} is defending. Total Damge is {total_damage}')

        # Absorb damage with shields first
        shield_damage = min(total_damage, self.shield[defend_hull.value])
        self.shield[defend_hull.value] -= shield_damage
        total_damage -= shield_damage

        # Apply remaining damage to the hull
        if total_damage > 0:
            self.hull -= total_damage
            if critical == Critical.STANDARD : self.hull -= 1 # Structural Damage

        self.game.visualize(f'{self.name} is defending. Remaining Hull : {max(0, self.hull)}, Remaining Sheid : {self.shield}')

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
            if attack_range == -1: continue  

            # gather attack dice
            if sum(self.gather_dice(attack_hull, attack_range)) <= int(obstructed): continue 

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
            self.game.visualize(f"{self.name} overlaps to {closest_ship.name}.")
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

        for speed in range(5):
            if abs(speed - self.speed) > 1:
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
                if speed >= 2 and self.size_class != 'small' :
                    if course[-2] > 0 : valid_placement.remove(-1)
                    elif course[-2] < 0 : valid_placement.remove(1)
        return valid_placement