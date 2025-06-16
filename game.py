from ship import *
from squadron import *

import numpy as np

class armada:
    def __init__(self):
        self.player_edge = 90 # cm
        self.short_edge = 90 # cm
        self.ship = [] # max 4 per player
        self.squadron = [] # max 6 per player

        self.round = 0
        self.roundphase = ""


class ship:
    def __init__(self, ship_dict, Player) :
        self.Player = Player

        self.Hull = ship_dict.get('hull')
        self.Size = ship_dict.get('size')
        self.CommandValue = ship_dict.get('command')
        self.SquadronValue = ship_dict.get('squadron')
        self.EngineeringValue = ship_dict.get('engineering')
        self.DefenseToken = {i : [ship_dict.get('DefenseToken')[i], 1] for i in range(len(ship_dict.get('DefenseToken')))}
        self.AntiSquad = ship_dict.get('anti_squad')
        self.FrontDice = ship_dict.get('battery')[0]
        self.LeftDice = ship_dict.get('battery')[1]
        self.RightDice = ship_dict.get('battery')[1]
        self.RearDice = ship_dict.get('battery')[2]
        self.NavChart = ship_dict.get('navchart')
        self.MaxShield = ship_dict.get('shield')
        self.Point = ship_dict.get('point')


    def deployship(self, x, y, direction, speed):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = speed
        self.shield = [self.maxshield[0], self.maxshield[1], self.maxshield[2], self.maxshield[1]] # [Front, Right, Rear, Left]

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