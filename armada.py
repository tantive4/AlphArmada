import numpy as np
import model
from ship import Ship
import random
from shapely.geometry import Polygon
import visualizer



class Armada:
    def __init__(self) -> None:
        self.player_edge = 900
        self.short_edge = 900
        self.game_board = Polygon([
            (0,0),
            (0,self.player_edge),
            (self.short_edge, self.player_edge),
            (self.short_edge, 0),
        ])
        self.ships : list[Ship] = []  # max 3 total, 2 + 1

        self.round = 1
        self.winner : int | None = None
        self.image_counter = 0



    def deploy_ship(self, ship : "Ship", x : float, y : float, orientation : float, speed : int) -> None :
        self.ships.append(ship)
        ship.deploy(self, x, y, orientation, speed, len(self.ships) - 1)
        
        

    def get_ship_count(self) -> int :
        """Counts only the ships that have not been destroyed."""
        return sum(1 for ship in self.ships if not ship.destroyed)

    def ship_phase(self) -> None :
        player = 1

        while True :
            valid_activations = self.get_valid_activation(player)
            opponent_activations = self.get_valid_activation(-player)
            if not valid_activations and not opponent_activations:
                break
            if not valid_activations:
                player *= -1
                continue
            
            ship_to_activate = random.choice(valid_activations)
            ship_to_activate.activate()

            player *= -1

    def total_destruction(self, player : int) -> bool:
        player_ship_count = sum(1 for ship in self.ships if ship.player == player and not ship.destroyed)
        return player_ship_count == 0

    def get_valid_activation(self, player : int) -> list[Ship] :
        """
        Returns a list of ships that can be activated by the given player.
        A ship can be activated if it is not destroyed, not already activated, and belongs to the player.
        """
        return [ship for ship in self.ships if ship.player == player and not ship.destroyed and not ship.activated]
    
    def status_phase(self) -> None :
        for ship in self.ships :
            if not ship.destroyed : ship.activated = False
        if self.total_destruction(1) : self.winner = -1
        if self.total_destruction(-1) : self.winner = 1
    
    def get_point(self, player : int) -> int :
        return sum(ship.point for ship in self.ships if ship.player != player and ship.destroyed)
    
    def play_round(self) -> None :
        self.visualize(f'ROUND {self.round} started')
        self.ship_phase()
        self.status_phase()

    def play(self) -> None :
        while self.round <= 6 :
            self.play_round()

            if self.winner != None :
                break
            self.round += 1

        if self.winner == None :
            self.winner = 1 if self.get_point(1) > self.get_point(-1) else -1

        self.visualize(f'Player {self.winner} has won!')


    def visualize(self, title : str, maneuver_tool = None) -> None:
        visualizer.visualize(self, title, maneuver_tool)