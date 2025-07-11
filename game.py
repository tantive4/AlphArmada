from ship import *
from squadron import *
from dice import *
from measure import *
import numpy as np
import model
from shapely.geometry import Polygon # install
import math

class armada:
    def __init__(self):
        self.player_edge = 30 # 3cm
        self.short_edge = 30 # 3cm
        self.ships = []  # max 3 total, 2 + 1

        self.round = 0
        self.roundphase = 0


    def deploy_ship(self, ship, x, y, direction, speed):
        self.ships.append(ship)
        ship.deploy(x, y, direction, speed, len(self.ships))

class ship:
    def __init__(self, ship_dict, Player) :
        self.Player = Player
        self.destroyed = False

        self.max_hull = ship_dict.get('hull')
        self.size = ship_dict.get('size')
        self.size_dimension = (76, 129) if self.size == 'large' else (61, 102) if self.size == 'medium' else (43, 71)
        self.battery = [ship_dict.get('battery')[0], ship_dict.get('battery')[1], ship_dict.get('battery')[2], ship_dict.get('battery')[1]] # [Front, Right, Rear, Left]
        self.navchart = ship_dict.get('navchart')
        self.max_shield = ship_dict.get('shield')
        self.point = ship_dict.get('point')

        self.front_arc = (ship_dict['front_arc_center'], ship_dict['front_arc_end']) 
        self.rear_arc = (ship_dict['front_arc_center'], ship_dict['front_arc_end'])

    def get_coordination(self, vector) :
        x, y = self.x, self.y
        add_x, add_y = vector
        rotated_add_x = add_x * math.cos(self.orientation) + add_y * math.sin(self.orientation)
        rotated_add_y = -add_x * math.sin(self.orientation) + add_y * math.cos(self.orientation)
        return (x + rotated_add_x, y + rotated_add_y)
    
    def deploy(self, x, y, orientation, speed, ship_index):
        self.x = x # 위치는 맨 앞 중앙
        self.y = y
        self.orientation = orientation
        self.speed = speed
        self.hull = self.max_hull
        self.shield = [self.max_shield[0], self.max_shield[1], self.max_shield[2], self.max_shield[1]] # [Front, Right, Rear, Left]
        self.activated = False
        self.ship_index = ship_index
        
    def attack(self, attack_hull, defender, defend_hull) :
        attack_range = range(self, attack_hull, defender, defend_hull) # close : 0, medium : 1, long : 2

        attack_pool = self.battery[attack_hull]
        for i in range(3) :
            if i < attack_range: attack_pool[i] = 0
        attack_pool = roll_dice(attack_pool)

        defender.defend(defend_hull, attack_pool)

    def defend(self, defend_hull, attack_pool):
        total_damage = [damage * dice for damage, dice in zip([0, 1, 2, 1, 1, 0, 0, 1, 1, 2, 0], attack_pool)] # [black, blue, red] (dice module)
        while total_damage > 0:
            if self.shield[defend_hull] > 0 :
                self.shield[defend_hull] -= 1
            else : self.hull -= 1
            total_damage -= 1
            
        critical = attack_pool[2] + attack_pool[4] + attack_pool[8]
        if critical : self.hull -= 1 # 크리티컬은 모두 Structural Damage

        if self.hull <= 0 : self.destroy()

    def destroy(self):
        pass
        


victory = ship(Victory_2_dict, -1)
cr90 = ship(CR90A_dict, 1)
nebulon = ship(Neb_escort_dict, 1)

game = armada()
player = 1
