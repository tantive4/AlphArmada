from ship import *
from squadron import *
from dice import *
import numpy as np
import model
from shapely.geometry import Polygon # install

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
        self.battery = [ship_dict.get('battery')[0], ship_dict.get('battery')[1], ship_dict.get('battery')[1], ship_dict.get('battery')[2]]
        self.navchart = ship_dict.get('navchart')
        self.max_shield = ship_dict.get('shield')
        self.point = ship_dict.get('point')

        

    def deploy(self, x, y, orientation, speed, ship_index):
        self.x = x # 위치는 맨 앞 중앙
        self.y = y
        self.orientation = orientation
        self.speed = speed
        self.hull = self.max_hull
        self.shield = [self.max_shield[0], self.max_shield[1], self.max_shield[1], self.max_shield[2]] # [Front, Left, Right, Rear]
        self.unactivated = False
        self.ship_index = ship_index
        
    def attack(self) :
        attack_hull = model.choose_attacker().index(1) # front : 0, left : 1, right : 2, rear : 3
        defend_hull = model.choose_defender() # []
        defender = game.ships[defend_hull.index(1) // 4]
        defend_hull = defend_hull % 4
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

        if self.hull <= 0 : self.destroyed()

    def destroyed(self):
        pass
        
def range(from_ship, from_hull, to_ship, to_hull):
    return 1 

victory = ship(Victory_2_dict, -1)
cr90 = ship(CR90A_dict, 1)
nebulon = ship(Neb_escort_dict, 1)

game = armada()
player = 1
