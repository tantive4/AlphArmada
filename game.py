from ship import *
from squadron import *
from dice import *
import numpy as np
import model
import math

class Armada:
    def __init__(self):
        self.player_edge = 900
        self.short_edge = 900
        self.ships = []  # max 3 total, 2 + 1

        self.round = 0
        self.roundphase = 0


    def deploy_ship(self, ship, x, y, orientation, speed):
        self.ships.append(ship)
        ship.deploy(self, x, y, orientation, speed, len(self.ships))

    def get_ship_count(self) :
        return len(self.ships)

    

game = Armada()

cr90 = Ship(CR90A_dict, 1)
nebulon = Ship(Neb_escort_dict, 1)
victory = Ship(Victory_2_dict, -1)


player = 1

game.deploy_ship(cr90,600, 175, 0, 2) # index = 0
game.deploy_ship(victory,450, 725, math.pi, 2) # 1
game.deploy_ship(nebulon, 300, 175, 0, 2) # 2

