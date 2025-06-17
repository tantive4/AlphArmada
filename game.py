from ship import *
from squadron import *

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
"""
	possible outcome	
ship activation	3	
squadron activation	10	
choose command	4	
command token conversion	1	bool
squadron command	1	bool
engineering command	1	bool
engineering point spend	17	< move 4 * 3 + recover 4 + repair 1 (ignore critical)
set attacking hull	4	
set target	13	< squadron or ship hull 12
add dice	3	
reroll dice	12	< each dice with eye
spend accuracy	6	< each type, is exhausted
set attacking hull	4	
set target	13	< squadron or ship hull 12
add dice	3	
reroll dice	12	< each dice with eye
spend accuracy	6	< each type, is exhausted
spend defense token	6	< each type, is exhausted (player switch)
decide maneuver	6	< with caution, each speed joint
squadron speed	10	< 0.5 distance
squadron direction	20	< 30deg
squadron set target	22	< ship hull 12 + squad 10
"""



    def deploy_ship(self, ship, x, y, direction, speed):
        self.ship.append(ship)
        ship.deploy(x, y, direction, speed)

    def deplay_squadron(self, squadron, x, y):
        self.squadorn.append(squadron)
        squadron.deplay(x, y)

class ship:
    def __init__(self, ship_dict, Player) :
        self.Player = Player
        self.destroyed = False

        self.MaxHull = ship_dict.get('hull')
        self.Size = ship_dict.get('size')
        self.CommandValue = ship_dict.get('command')
        self.SquadronValue = ship_dict.get('squadron')
        self.EngineeringValue = ship_dict.get('engineering')
        self.DefenseToken = [ship_dict.get('DefenseToken').count(i) for i in ['brace', 'redirect', 'evade']]
        self.AntiSquad = ship_dict.get('anti_squad')
        self.FrontDice = ship_dict.get('battery')[0]
        self.LeftDice = ship_dict.get('battery')[1]
        self.RightDice = ship_dict.get('battery')[1]
        self.RearDice = ship_dict.get('battery')[2]
        self.NavChart = ship_dict.get('navchart')
        self.MaxShield = ship_dict.get('shield')
        self.Point = ship_dict.get('point')

    def deploy(self, x, y, direction, speed):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = speed
        self.hull = self.MaxHull
        self.shield = [self.maxshield[0], self.maxshield[1], self.maxshield[2], self.maxshield[1]] # [Front, Right, Rear, Left]
        self.unactivated = False

class squadron:
    def __init__(self, squad_dict, Player):
        self.Player = Player
        
        self.Hull = squad_dict.get('hull')
        self.Speed = squad_dict.get('speed')
        self.Point = squad_dict.get('point')
        self.AntiSquad = squad_dict.get('anti_squad')
        self.Battery = squad_dict.get('battery')
        self.Escort = squad_dict.get('escort')
        self.Bomber = squad_dict.get('bomber')
        self.Swarm = squad_dict.get('swarm')

    def deplay(self, x, y):
        self.x = x
        self.y = y


Victory = ship(Victory_2_dict, -1)
CR90 = ship(CR90A_dict, 1)
Nebulon = ship(Neb_escort_dict, 1)

TIE1 = squadron(TIE_fighter_dict, -1)
TIE2 = squadron(TIE_fighter_dict, -1)
TIE3 = squadron(TIE_fighter_dict, -1)
TIE4 = squadron(TIE_fighter_dict, -1)
TIE5 = squadron(TIE_fighter_dict, -1)
TIE6 = squadron(TIE_fighter_dict, -1)
X1 = squadron(xwing_dict, 1)
X2 = squadron(xwing_dict, 1)
X3 = squadron(xwing_dict, 1)
X4 = squadron(xwing_dict, 1)

game = armada()
player = 1
