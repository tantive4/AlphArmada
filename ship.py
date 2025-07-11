from shapely.geometry import Polygon, LineString
import shapely.ops
import numpy as np
import math
from dice import *
import model
from visualizer import visualize_board

class Ship:
    def __init__(self, ship_dict, player) :
        self.player = player
        self.game = None
        self.name = ship_dict.get('name')
        self.point = ship_dict.get('point')

        self.max_hull = ship_dict.get('hull')
        self.size = ship_dict.get('size')
        self.size_dimension = (76, 129) if self.size == 'large' else (61, 102) if self.size == 'medium' else (43, 71)
        self.battery = (ship_dict.get('battery')[0], ship_dict.get('battery')[1], ship_dict.get('battery')[2], ship_dict.get('battery')[1]) # (Front, Right, Rear, Left)
        self.navchart = ship_dict.get('navchart')
        self.max_shield = ship_dict.get('shield')
        self.point = ship_dict.get('point')

        self.front_arc = (ship_dict.get('front_arc_center'), ship_dict.get('front_arc_end')) 
        self.rear_arc = (ship_dict.get('rear_arc_center'), ship_dict.get('rear_arc_end'))

    def _get_coordination(self, vector) :
        x, y = self.x, self.y
        add_x, add_y = vector
        rotated_add_x = add_x * math.cos(self.orientation) + add_y * math.sin(self.orientation)
        rotated_add_y = -add_x * math.sin(self.orientation) + add_y * math.cos(self.orientation)
        return (x + rotated_add_x, y + rotated_add_y)
    
    def set_coordination(self) : # 위치 이동이 있을 때마다 실행시켜서 위치 업뎃해줄 것
        self.front_right_base = self._get_coordination((self.size_dimension[0] / 2,0))
        self.front_left_base = self._get_coordination((- self.size_dimension[0] / 2,0))
        self.rear_right_base = self._get_coordination((self.size_dimension[0] / 2, - self.size_dimension[1]))
        self.rear_left_base = self._get_coordination((- self.size_dimension[0] / 2, - self.size_dimension[1]))

        self.front_right_arc = self._get_coordination((self.size_dimension[0]/2, - self.front_arc[1]))
        self.front_left_arc = self._get_coordination((-self.size_dimension[0]/2, - self.front_arc[1]))
        self.rear_right_arc = self._get_coordination((self.size_dimension[0]/2, - self.rear_arc[1]))
        self.rear_left_arc = self._get_coordination((-self.size_dimension[0]/2, - self.rear_arc[1]))

        self.front_arc_center = self._get_coordination((0, -self.front_arc[0]))
        self.rear_arc_center = self._get_coordination((0, -self.rear_arc[0]))

        
        if self.game:
            self.game.visualization_counter += 1
            visualize_board(self.game.ships, self.game.player_edge, self.game.short_edge, self.game.visualization_counter)

    def deploy(self, game, x, y, orientation, speed, ship_index):
        self.game = game
        self.x = x # 위치는 맨 앞 중앙
        self.y = y
        self.orientation = orientation
        
        self.speed = speed
        self.hull = self.max_hull
        self.shield = [self.max_shield[0], self.max_shield[1], self.max_shield[2], self.max_shield[1]] # [Front, Right, Rear, Left]
        self.activated = False
        self.destroyed = False
        self.ship_id = ship_index
        self.set_coordination()
        print(f'{self.name} is deployed.')

    def measure_arc_and_range(self, from_hull, to_ship, to_hull, extension_factor=1e4) :
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

        if from_hull == 0 or from_hull == 1 :
            arc1 = (self.front_arc_center, self.front_right_arc)
        else : 
            arc1 = (self.rear_arc_center, self.rear_left_arc)
        
        if from_hull == 1 or from_hull == 2 :
            arc2 = (self.rear_arc_center, self.rear_right_arc)
        else :
            arc2 = (self.front_arc_center, self.front_left_arc)
        from_hull_polygon = (
                Polygon([ # front 0
                    self.front_right_base,
                    self.front_right_arc,
                    self.front_arc_center,
                    self.front_left_arc,
                    self.front_left_base
                ]),
                Polygon([ # right 1
                    self.front_arc_center,
                    self.front_right_arc,
                    self.rear_right_arc,
                    self.rear_arc_center
                ]),
                Polygon([ # rear 2
                    self.rear_right_arc,
                    self.rear_right_base,
                    self.rear_left_base,
                    self.rear_left_arc,
                    self.rear_arc_center
                ]),
                Polygon([ # left 3
                    self.front_arc_center,
                    self.front_left_arc,
                    self.rear_left_arc,
                    self.rear_arc_center
                ])
            )  
        to_hull_polygon = (
                Polygon([ # front 0
                    to_ship.front_right_base,
                    to_ship.front_right_arc,
                    to_ship.front_arc_center,
                    to_ship.front_left_arc,
                    to_ship.front_left_base
                ]),
                Polygon([ # right 1
                    to_ship.front_arc_center,
                    to_ship.front_right_arc,
                    to_ship.rear_right_arc,
                    to_ship.rear_arc_center
                ]),
                Polygon([ # rear 2
                    to_ship.rear_right_arc,
                    to_ship.rear_right_base,
                    to_ship.rear_left_base,
                    to_ship.rear_left_arc,
                    to_ship.rear_arc_center
                ]),
                Polygon([ # left 3
                    to_ship.front_arc_center,
                    to_ship.front_left_arc,
                    to_ship.rear_left_arc,
                    to_ship.rear_arc_center
                ])
            ) 
        
        arc1_vector = np.array(arc1[1]) - np.array(arc1[0])
        arc2_vector = np.array(arc2[1]) - np.array(arc2[0])

        arc_polygon = Polygon([arc1[0], arc1[1] + arc1_vector * extension_factor,
                            arc2[1] + arc2_vector * extension_factor, arc2[0]])
        to_hull_in_arc = to_hull_polygon[to_hull].intersection(arc_polygon)

        if to_hull_in_arc.is_empty or not isinstance(to_hull_in_arc, Polygon) :
            return -1 # not in arc
        else :
            range_measure = LineString(shapely.ops.nearest_points(from_hull_polygon[from_hull], to_hull_in_arc))

            for hull_index in range(4) :
                if hull_index != to_hull and  range_measure.intersects(to_hull_polygon[hull_index]) :
                    return -1 # range not valid
            distance = range_measure.length

            if distance <= 123.3 : return 0 # close range
            elif distance <= 186.5 : return 1 # medium range
            elif distance <= 30.48 : return 2 # long range
            else : return 3 # extreme range

    def attack(self, attack_hull, defender, defend_hull) :

        attack_range = self.measure_arc_and_range(attack_hull, defender, defend_hull)
        
        if attack_range == -1 : return # not in arc or invalid range

        print(f'{self.name} attacks {defender.name}! {['Front', 'Right', 'Rear', 'Left'][attack_hull]} to {['Front', 'Right', 'Rear', 'Left'][defend_hull]}! Range : {['close', 'medium', 'long'][attack_range]}')

        attack_pool = list(self.battery[attack_hull])
        for i in range(3) :
            if i < attack_range: attack_pool[i] = 0

        if sum(attack_pool) == 0 : return # empty attack pool

        attack_pool = roll_dice(attack_pool)

        defender.defend(defend_hull, attack_pool)

    def defend(self, defend_hull, attack_pool):
        total_damage = sum([damage * dice for damage, dice in zip(DAMAGE_INDICES, attack_pool)]) # [black, blue, red] (dice module)
        
        print(f'{self.name} is defending. Total Damge is {total_damage}')

        while total_damage > 0:
            if self.shield[defend_hull] > 0 :
                self.shield[defend_hull] -= 1
            total_damage -= 1

        if total_damage > 0 :
            self.hull -= total_damage
            critical = sum(attack_pool[i] for i in CRIT_INDICES)
            if critical : self.hull -= 1 # 크리티컬은 모두 Structural Damage

        print(f'{self.name} is defending. Remaining Hull : {max(0, self.hull)}, Remaining Sheid : {self.shield}')

        if self.hull <= 0 : self.destroy()

    def destroy(self):
        self.destroyed = True
        self.hull = 0
        self.shield = [0,0,0,0]
        self.battery = tuple((0, 0, 0) for _ in range(4))
        print(f'{self.name} is destroyed!')
        
    def maneuver(self, yaw) :
        # under construction , just speed 2 straight maneuver
        (self.x, self.y) = self._get_coordination((0, 75))
        self.set_coordination()
        
        print(f'{self.name} executes maneuver. {self.x, self.y}')

    def activate(self) :
        print(f'{self.name} is activated.')
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
                    if ship.ship_id == self.ship_id : defend_hull_value[ship.ship_id * 4 + hull_index] = model.MASK_VALUE
                    attack_range = self.measure_arc_and_range(attack_hull, self.game.ships[ship.ship_id], hull_index)
                    if attack_range == -1 or attack_range == 3 : defend_hull_value[ship.ship_id * 4 + hull_index] = model.MASK_VALUE

            if np.sum(defend_hull_value == model.MASK_VALUE) == len(defend_hull_value):
                attack_possible[attack_hull] = False
                continue

            defend_hull_policy = model.softmax(defend_hull_value)
            defender = np.argmax(defend_hull_policy)

            self.attack(attack_hull, self.game.ships[defender // 4], defender % 4)
            attack_possible[attack_hull] = False
            attack_count += 1
        
        self.maneuver(None) #under construction
        
        self.activated = True


Victory_2_dict = {
    'name' : 'Victory SD',
    'hull' : 8,
    'size' : 'medium', # 61 x 102
    'command' : 3,
    'squadron' : 3,
    'engineering' : 4,
    'defense_token' : ('brace', 'redirect', 'redirect'),
    'anti_squad' : (0, 1, 0),
    'shield' : (3, 3, 1),
    'battery' : ((0, 3, 3), (0, 1, 2), (0, 0, 2)),
    'navchart' : {1 : (1), 2 : (0, 1)},
    'point' : 80,
    'front_arc_center' : 48, # 전방 발포 호 선과 중앙선이 만나는 점과 함선 맨 앞 사이 거리(mm)
    'front_arc_end' : 24, # 전방 발포 호 선이 측면 끝과 만나는 점과 함선 맨 앞 앞 사이 거리 (이 둘만 있으면 발포 호 선 구현 가능)
    'rear_arc_center' : 48,
    'rear_arc_end' : 79
    }

CR90A_dict = {
    'name' : 'CR90 corvette',
    'hull' : 4,
    'size' : 'small', # 43 x 71
    'command' : 1,
    'squadron' : 1,
    'engineering' : 2,
    'defense_token' : ('evade', 'evade', 'redirect'),
    'anti_squad' : (0, 1, 0),
    'shield' : (2, 2, 1),
    'battery' : ((0, 1, 2), (0, 1, 1), (0, 0, 1)),
    'navchart' : {1 : (2), 2 : (1, 2), 3 : (0, 1, 2), 4 : (0, 1, 1, 2)},
    'point' : 44,
    'front_arc_center' : 43,
    'front_arc_end' : 23,
    'rear_arc_center' : 43,
    'rear_arc_end' : 62
    }

Neb_escort_dict = {
    'name' : 'Nebulon-B frigate',
    'hull' : 5,
    'size' : 'small',
    'command' : 2,
    'squadron' : 2,
    'engineering' : 3,
    'defense_token' : ('evade', 'brace', 'brace'),
    'anti_squad' : (0, 2, 0),
    'shield' : (3, 1, 2),
    'battery' : ((0, 0, 3), (0, 1, 1), (0, 0, 2)),
    'navchart' : {1 : (1), 2 : (1, 1), 3 : (0, 1, 2)},
    'point' : 47,
    'front_arc_center' : 35,
    'front_arc_end' : 0,
    'rear_arc_center' : 35,
    'rear_arc_end' : 0
    }