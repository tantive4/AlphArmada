from ship import *
from squadron import *
from dice import *
import numpy as np

class armada:
    def __init__(self):
        self.player_edge = 30 # 3cm
        self.short_edge = 30 # 3cm
        self.ship = []  # max 3 total, 2 + 1
        self.squadron = [] # max 10 total, 6 + 4

        self.round = 0
        self.roundphase = 0  


    def deploy_ship(self, ship, x, y, direction, speed):
        self.ship.append(ship)
        ship.deploy(x, y, direction, speed)

    def deploy_squadron(self, squadron, x, y):
        self.squadorn.append(squadron)
        squadron.deplay(x, y)
    
    def command_phase(self):
        pass

    def ship_phase(self):
        pass

    def squadron_phase(self):
        pass

    def status_phase(self):
        pass

class ship:
    def __init__(self, ship_dict, Player) :
        self.Player = Player
        self.destroyed = False

        self.max_hull = ship_dict.get('hull')
        self.size = ship_dict.get('size')
        self.command_value = ship_dict.get('command')
        self.squadron_value = ship_dict.get('squadron')
        self.engineering_value = ship_dict.get('engineering')
        self.defense_token = [ship_dict.get('DefenseToken').count(i) for i in ['brace', 'redirect', 'evade']]
        self.anti_squad = ship_dict.get('anti_squad')
        self.dice_front = ship_dict.get('battery')[0]
        self.dice_left = ship_dict.get('battery')[1]
        self.dice_right = ship_dict.get('battery')[1]
        self.dice_rear = ship_dict.get('battery')[2]
        self.navchart = ship_dict.get('navchart')
        self.max_shield = ship_dict.get('shield')
        self.point = ship_dict.get('point')

    def deploy(self, x, y, direction, speed):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = speed
        self.hull = self.max_hull
        self.shield = [self.max_shield[0], self.max_shield[1], self.max_shield[2], self.max_shield[1]] # [Front, Right, Rear, Left]
        self.unactivated = False
        

victory = ship(Victory_2_dict, -1)
cr90 = ship(CR90A_dict, 1)
nebulon = ship(Neb_escort_dict, 1)

game = armada()
player = 1
