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

        self.round = 1
        self.winner = None
        self.visualization_counter = 0


    def deploy_ship(self, ship, x, y, orientation, speed):
        self.ships.append(ship)
        ship.deploy(self, x, y, orientation, speed, len(self.ships) - 1)

    def get_ship_count(self):
        """Counts only the ships that have not been destroyed."""
        return sum(1 for ship in self.ships if not ship.destroyed)

    def ship_phase(self) :
        player = 1
        activation_count = 0

        while activation_count < self.get_ship_count() :
            ship_activate_value = model.choose_ship_activate()

            for ship in self.ships :
                if ship.player != player or ship.activated or ship.destroyed :
                    ship_activate_value[ship.ship_id] = model.MASK_VALUE

            if np.sum(ship_activate_value == model.MASK_VALUE) == len(ship_activate_value):
                player *= -1
                continue

            ship_activate_policy = model.softmax(ship_activate_value)
            ship_to_activate = self.ships[np.argmax(ship_activate_policy)]
            ship_to_activate.activate()
            activation_count += 1
            player *= -1

    def total_destruction(self, player) :
        player_ship_count = sum(1 for ship in self.ships if ship.player == player and not ship.destroyed)
        return player_ship_count == 0


    def status_phase(self) :
        for ship in self.ships :
            if not ship.destroyed : ship.activated = False
        if self.total_destruction(1) : self.winner = -1
        if self.total_destruction(-1) : self.winner = 1
    
    def get_point(self, player) :
        return sum(ship.point for ship in self.ships if ship.player != player and ship.destroyed)
    
    def play_round(self) :
        print(f'\n----------- ROUND {self.round} started -------------')
        self.ship_phase()
        self.status_phase()

    def play(self) :
        while self.round <= 6 :
            self.play_round()

            if self.winner != None :
                break
            self.round += 1

        if self.winner == None :
            self.winner = 1 if self.get_point(1) > self.get_point(-1) else -1

        print(f'Player {self.winner} has won!')

game = Armada()

cr90 = Ship(CR90A_dict, 1)
nebulon = Ship(Neb_escort_dict, 1)
victory = Ship(Victory_2_dict, -1)




game.deploy_ship(cr90,600, 175, 0, 2) # id = 0
game.deploy_ship(victory,450, 725, math.pi, 2) # 1
game.deploy_ship(nebulon, 300, 175, 0, 2) # 2

game.play()