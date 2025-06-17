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
        """
          1: ship activation, 2: squadron activation, 
          3: choose command, 4: command token conversion, 5: squadron command, 6: engineering command, 7: engineering point spend
          7: set target, 8: add concentrate fire dice, 9: reroll concentrate fire, 10: spend accuracy, 11: spend defense token,
          12: decide maneuver (speed 1, 2, 3, 4),
          13: squadron movement, 13: squadron set target
        """

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
        
    

class squadron:
    def __init__(self, squad_dict, Player):
        self.Player = Player
        
        self.hull = squad_dict.get('hull')
        self.speed = squad_dict.get('speed')
        self.point = squad_dict.get('point')
        self.anti_squad = squad_dict.get('anti_squad')
        self.battery = squad_dict.get('battery')
        self.escort = squad_dict.get('escort')
        self.bomber = squad_dict.get('bomber')
        self.swarm = squad_dict.get('swarm')

    def deplay(self, x, y):
        self.x = x
        self.y = y

victory = ship(Victory_2_dict, -1)
cr90 = ship(CR90A_dict, 1)
nebulon = ship(Neb_escort_dict, 1)

tie1 = squadron(tie_fighter_dict, -1)
tie2 = squadron(tie_fighter_dict, -1)
tie3 = squadron(tie_fighter_dict, -1)
tie4 = squadron(tie_fighter_dict, -1)
tie5 = squadron(tie_fighter_dict, -1)
tie6 = squadron(tie_fighter_dict, -1)
x1 = squadron(xwing_dict, 1)
x2 = squadron(xwing_dict, 1)
x3 = squadron(xwing_dict, 1)
x4 = squadron(xwing_dict, 1)

game = armada()
player = 1
