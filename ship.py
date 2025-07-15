from shapely.geometry import Polygon, LineString
import shapely.ops
import numpy as np
import math
from dice import *
import model
from enum import IntEnum

class HullSection(IntEnum):
    FRONT = 0
    RIGHT = 1
    REAR = 2
    LEFT = 3

SHIP_BASE_SIZE = {'small' : (43, 71), 'medium' :(63, 102), 'large' : (77.5, 129)}
SHIP_TOKEN_SIZE = {'small' : (38.5, 70.25), 'medium' : (58.5, 101.5)}
TOOL_WIDTH = 15.25
TOOL_LENGTH = 48.4 # not accurate
TOOL_PART_LENGTH = 20 # not accurate

class Ship:
    def __init__(self, ship_dict : dict, player : int) -> None:
        self.player = player
        self.game = None
        self.name = ship_dict.get('name')
        self.point = ship_dict.get('point')

        self.max_hull = ship_dict.get('hull')
        self.size_class = ship_dict.get('size')
        self.token_size = SHIP_TOKEN_SIZE.get(self.size_class)
        self.base_size = SHIP_BASE_SIZE.get(self.size_class)

        self.battery = (ship_dict.get('battery')[0], ship_dict.get('battery')[1], ship_dict.get('battery')[2], ship_dict.get('battery')[1]) # (Front, Right, Rear, Left)
        self.navchart = ship_dict.get('navchart')
        self.max_shield = ship_dict.get('shield')
        self.point = ship_dict.get('point')

        self.front_arc = (ship_dict.get('front_arc_center'), ship_dict.get('front_arc_end')) 
        self.rear_arc = (ship_dict.get('rear_arc_center'), ship_dict.get('rear_arc_end'))

    def _get_coordination(self, vector : tuple[float]) -> tuple :
        x, y = self.x, self.y
        add_x, add_y = vector
        rotated_add_x = add_x * math.cos(self.orientation) + add_y * math.sin(self.orientation)
        rotated_add_y = -add_x * math.sin(self.orientation) + add_y * math.cos(self.orientation)
        return (x + rotated_add_x, y + rotated_add_y)
    
    def set_coordination(self) -> None: # update when deployed/moved
        self.front_right_token = self._get_coordination((self.token_size[0] / 2,0))
        self.front_left_token = self._get_coordination((- self.token_size[0] / 2,0))
        self.rear_right_token = self._get_coordination((self.token_size[0] / 2, - self.token_size[1]))
        self.rear_left_token = self._get_coordination((- self.token_size[0] / 2, - self.token_size[1]))

        self.front_right_arc = self._get_coordination((self.token_size[0]/2, - self.front_arc[1]))
        self.front_left_arc = self._get_coordination((-self.token_size[0]/2, - self.front_arc[1]))
        self.rear_right_arc = self._get_coordination((self.token_size[0]/2, - self.rear_arc[1]))
        self.rear_left_arc = self._get_coordination((-self.token_size[0]/2, - self.rear_arc[1]))

        self.front_arc_center = self._get_coordination((0, -self.front_arc[0]))
        self.rear_arc_center = self._get_coordination((0, -self.rear_arc[0]))

        self.front_right_base = self._get_coordination((self.base_size[0] / 2, (self.base_size[1] - self.token_size[1]) / 2))
        self.front_left_base = self._get_coordination((- self.base_size[0] / 2,(self.base_size[1] - self.token_size[1]) / 2))
        self.rear_right_base = self._get_coordination((self.base_size[0] / 2, (self.base_size[1] - self.token_size[1]) / 2 - self.base_size[1]))
        self.rear_left_base = self._get_coordination((- self.base_size[0] / 2, (self.base_size[1] - self.token_size[1]) / 2 - self.base_size[1]))
        
        self._make_polygon()

    def deploy(self, game : "Armada", x : float, y : float, orientation : float, speed : int, ship_id : int) -> None:
        self.game = game
        self.x = x # 위치는 맨 앞 중앙
        self.y = y
        self.orientation = orientation
        
        self.speed = speed
        self.hull = self.max_hull
        self.shield = [self.max_shield[0], self.max_shield[1], self.max_shield[2], self.max_shield[1]] # [Front, Right, Rear, Left]
        self.activated = False
        self.destroyed = False
        self.ship_id = ship_id
        self.set_coordination()
        self.game.visualize(f'{self.name} is deployed.')

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
    
    def measure_arc_and_range(self, from_hull : int, to_ship : "Ship", to_hull : int, extension_factor=1e4) -> int:
        """Measures the range and validity of a firing arc to a target.

        This method checks if a target hull is within the firing ship's
        specified arc and calculates the range if the line of sight is clear.

        Args:
            from_hull (int): The index of the firing hull section (0-3).
            to_ship (Ship): The target Ship object.
            to_hull (int): The index of the target hull section (0-3).

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
        to_hull_in_arc = to_ship.hull_polygon[to_hull].intersection(arc_polygon)

        if to_hull_in_arc.is_empty or not isinstance(to_hull_in_arc, Polygon) :
            return -1 # not in arc
        else :
            range_measure = LineString(shapely.ops.nearest_points(self.hull_polygon[from_hull], to_hull_in_arc))

            for hull_index in range(4) :
                if hull_index != to_hull and  range_measure.crosses(to_ship.hull_polygon[hull_index]) :
                    return -1 # range not valid
            distance = range_measure.length

            if distance <= 123.3 : return 0 # close range
            elif distance <= 186.5 : return 1 # medium range
            elif distance <= 304.8 : return 2 # long range
            else : return 3 # extreme range

    def attack(self, attack_hull : HullSection, defender : "Ship", defend_hull : HullSection) -> None :
        
        attack_range = self.measure_arc_and_range(attack_hull, defender, defend_hull)
        
        if attack_range == -1 : return # not in arc or invalid range

        self.game.visualize(f'{self.name} attacks {defender.name}! {attack_hull.name} to {defend_hull.name}! Range : {['close', 'medium', 'long'][attack_range]}')

        attack_pool = list(self.battery[attack_hull])
        for i in range(3) :
            if i < attack_range: attack_pool[i] = 0

        if sum(attack_pool) == 0 : return # empty attack pool

        attack_pool = roll_dice(attack_pool)
        self.game.visualize((f'''
          Dice Rolled!
          Black [Blank, Hit, Double] : {attack_pool[:3]}
          Blue [Hit, Critical, Accuracy] : {attack_pool[3:6]}
          Red [Blank, Hit, Critical, Double, Accuracy] : {attack_pool[6:]}
        '''))
        defender.defend(defend_hull, attack_pool)

    def defend(self, defend_hull : int, attack_pool : list) -> None:
        total_damage = sum([damage * dice for damage, dice in zip(DAMAGE_INDICES, attack_pool)]) # [black, blue, red] (dice module)
        
        self.game.visualize(f'{self.name} is defending. Total Damge is {total_damage}')

        # Absorb damage with shields first
        shield_damage = min(total_damage, self.shield[defend_hull])
        self.shield[defend_hull] -= shield_damage
        total_damage -= shield_damage

        # Apply remaining damage to the hull
        if total_damage > 0:
            self.hull -= total_damage
            critical = sum(attack_pool[i] for i in CRIT_INDICES)
            if critical:
                self.hull -= 1 # Critical hits add one damage

        self.game.visualize(f'{self.name} is defending. Remaining Hull : {max(0, self.hull)}, Remaining Sheid : {self.shield}')

        if self.hull <= 0 : self.destroy()

    def destroy(self) -> None:
        self.destroyed = True
        self.hull = 0
        self.shield = [0,0,0,0]
        self.battery = tuple((0, 0, 0) for _ in range(4))
        self.game.visualize(f'{self.name} is destroyed!')

    def determine_course(self) -> tuple[list, int]:
        # speed
        speed_value = model.choose_speed()
        for speed in range(5) :
            if abs(speed - self.speed) > 1 or (self.navchart.get(str(speed)) == None and speed != 0) :
                speed_value[speed] = model.MASK_VALUE
        speed_policy = model.softmax(speed_value)
        self.speed = np.argmax(speed_policy)

        course = list(0 for _ in range(self.speed)) # [yaw at  joint 1, yaw at joint 2, ...]
        
        # yaw
        for joint in range(self.speed) :
            current_yaw_value = model.choose_yaw(self.speed, joint + 1)
            for yaw in range(5) :
                if abs(yaw - 2) > self.navchart.get(str(self.speed))[joint] : current_yaw_value[yaw] = model.MASK_VALUE
            current_yaw_policy = model.softmax(current_yaw_value)
            course[joint] = np.argmax(current_yaw_policy) - 2

        # placement
        placement_value = model.choose_placement(course)
        if course[-1] > 0 : placement_value[0] = model.MASK_VALUE
        elif course[-1] < 0 : placement_value[1] = model.MASK_VALUE
        else : 
            if self.speed >= 2 and self.size_class != 'small' :
                if course[-2] > 0 : placement_value[0] = model.MASK_VALUE
                if course[-2] < 0 : placement_value[1] = model.MASK_VALUE
        placement_policy = model.softmax(placement_value)
        placement = np.argmax(placement_policy)
        placement = placement * 2 - 1 # right 1, left -1

        return course, placement

    def tool_coordination(self, course : list[int], placement : int) -> tuple[list[tuple[int]], list[int]] :
        (tool_x, tool_y) = self._get_coordination((placement * (self.base_size[0] + TOOL_WIDTH) / 2,0))
        tool_orientaion = self.orientation

        joint_coordination = [(tool_x, tool_y)]
        joint_orientation = []


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
    
    def maneuver_coordination(self, placement : int, tool_coordination : tuple[int], tool_orientaion : float) -> None :
        self.x = tool_coordination[0] - placement * math.cos(tool_orientaion) * (self.base_size[0] + TOOL_WIDTH) / 2
        self.y = tool_coordination[1] + placement * math.sin(tool_orientaion) * (self.base_size[0] + TOOL_WIDTH) / 2
        self.orientation = tool_orientaion
        self.set_coordination()
        

    def is_overlap(self) -> list[bool] :
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


    def move_ship(self, course : list[int], placement : int) -> list[bool]:
        original_x, original_y, original_orientaion = self.x, self.y, self.orientation

        joint_coordination, joint_orientaion = self.tool_coordination(course, placement)

        overlap_list = [False for _ in self.game.ships]

        while True :
            self.maneuver_coordination(placement, joint_coordination[-1], joint_orientaion[-1])
            self.game.visualize(f'{self.name} executes speed {len(joint_orientaion)} maneuver.', joint_coordination)

            current_overlap = self.is_overlap()

            if not any(current_overlap):
                break
            self.game.visualize(f'{self.name} overlaps ships at speed {len(joint_orientaion)} maneuver.')

            overlap_list = [overlap_list[i] or current_overlap[i] for i in range(len(overlap_list))]

            self.x, self.y, self.orientation = original_x, original_y, original_orientaion
            self.set_coordination()
            del joint_coordination[-2 :]
            del joint_orientaion[-1] 
            if len(joint_coordination) == 1 : break
        
        if self.out_of_board() :
            self.game.visualize(f'{self.name} is out of board!')
            self.destroy()
    
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

    def activate(self) -> None:
        self.game.visualize(f'{self.name} is activated.')

        # attack
        attack_count = 0
        attack_possible = [True, True, True, True]
        while attack_count < 2 and sum(attack_possible) > 0 :

            attack_hull_value = model.choose_attacker()
            for hull_index in range(4) :
                if not attack_possible[hull_index] :
                    attack_hull_value[hull_index] = model.MASK_VALUE
            attack_hull_policy = model.softmax(attack_hull_value)
            attack_hull = np.argmax(attack_hull_policy)
        
            defend_hull_value = model.choose_defender(self)
            
            for ship in self.game.ships :
                for hull_index in range(4) :
                    if ship.player == self.player :
                        defend_hull_value[ship.ship_id * 4 + hull_index] = model.MASK_VALUE
                        continue # Skip to the next ship
                    
                    attack_range = self.measure_arc_and_range(attack_hull, self.game.ships[ship.ship_id], hull_index)
                    if attack_range == -1 or attack_range == 3 : defend_hull_value[ship.ship_id * 4 + hull_index] = model.MASK_VALUE

            if np.sum(defend_hull_value == model.MASK_VALUE) == len(defend_hull_value):
                attack_possible[attack_hull] = False
                continue

            defend_hull_policy = model.softmax(defend_hull_value)
            defender = np.argmax(defend_hull_policy)
            attack_hull_section = HullSection(attack_hull)
            defend_ship = self.game.ships[defender // 4]
            defend_hull_section = HullSection(defender % 4)

            self.attack(attack_hull_section, defend_ship, defend_hull_section)
            attack_possible[attack_hull] = False
            attack_count += 1
        
        # maneuver
        course, placement = self.determine_course()
        self.move_ship(course, placement)
        
        self.activated = True
