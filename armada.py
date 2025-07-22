import numpy as np
from ship import Ship
import random
from shapely.geometry import Polygon
import visualizer
import copy
from mcts import MCTS, MCTSState, ActionType
from time import sleep



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
        self.current_player = 1
        self.simulation_mode = False

    def deploy_ship(self, ship : Ship, x : float, y : float, orientation : float, speed : int) -> None :
        self.ships.append(ship)
        ship.deploy(self, x, y, orientation, speed, len(self.ships) - 1)



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
            if not ship.destroyed : ship.refresh()
        if self.total_destruction(1) : self.winner = -1
        if self.total_destruction(-1) : self.winner = 1
        if not self.simulation_mode : print(f"End of Round {self.round}.")
        

    def get_point(self, player : int) -> int :
        return sum(ship.point for ship in self.ships if ship.player != player and ship.destroyed)

    def play_round(self) -> None:
        """
        Plays a single round of the game, alternating between players.
        Each player activates their ships, attacks, and maneuvers.
        """
        self.ship_phase()
        self.status_phase()
        self.round += 1


    def ship_phase(self) -> None:
        """
        The ship phase is where players activate their ships, attack, and maneuver.
        This method is called at the start of each round.
        """
        self.current_player = 1
        while True:
            p1_can_activate = any(self.get_valid_activation(1))
            p2_can_activate = any(self.get_valid_activation(-1))

            # If both players have activated all their ships, the round's activation phase is over.
            if not p1_can_activate and not p2_can_activate:
                break

            # Check if the current player has any ships to activate. If not, they pass.
            if not any(self.get_valid_activation(self.current_player)):
                print(f"Player {self.current_player} has no ships to activate and passes the turn.")
                self.current_player *= -1
                continue
            
            # It's the current player's turn to make a move.
            self.visualize(f'ROUND {self.round} | Player {self.current_player}\'s Turn')

            # Player 1's Turn (MCTS)
            if self.current_player == 1:
                print("Player 1 (MCTS) is playing...")
                self._execute_mcts_turn()

            # # Player -1's Turn (Random)
            elif self.current_player == -1:
                print("Player -1 (Random) is playing...")
                valid_activations = self.get_valid_activation(self.current_player)
                if valid_activations:
                    ship_to_activate = random.choice(valid_activations)
                    self._execute_random_activation(ship_to_activate)

            # Switch player for the next activation in the round
            self.current_player *= -1


    def play(self) -> None :
        """
        The main game loop, structured by rounds and alternating player turns.
        """
        while self.round <= 6 and self.winner is None:
            self.play_round()

        # End of Game: Determine winner if one wasn't already declared
        if self.winner is None:
            print("Game ends after 6 rounds. Calculating points...")
            p1_points = self.get_point(1)
            p2_points = self.get_point(-1)
            print(f"Player 1 Points: {p1_points} | Player -1 Points: {p2_points}")
            if p1_points > p2_points: self.winner = 1
            elif p2_points > p1_points: self.winner = -1
            else: self.winner = 0 # Draw

        self.visualize(f'Player {self.winner} has won!')
        print(f'Player {self.winner} has won!')

    def _execute_mcts_turn(self, iterations : int = 200):
        """
        Executes a full ship activation for the MCTS player by breaking it down
        into sequential decisions.
        """
        # === 1. CHOOSE SHIP TO ACTIVATE ===
        print(f"Player {self.current_player} is thinking about phase: activation")
        state : MCTSState = {
            "game": self, 
            "current_player": self.current_player, 
            "decision_phase": "ship_activation",
            "active_ship_id": None, 
            "declare_target": None,
            }
        mcts_state_copy = copy.deepcopy(state)
        mcts_state_copy['game'].simulation_mode = True
        mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
        mcts.search(iterations=iterations)
        action = mcts.get_best_action()

        if action is None or action[0] != 'activate_ship':
            raise ValueError('MCTS must choose ship to activate')
        
        active_ship_id : int = action[1]
        active_ship = self.ships[active_ship_id]
        print(f"Player {self.current_player} chose action: {action[0], active_ship.name}")


        # === 2. ATTACK PHASE ===

        while active_ship.attack_count < 2 :
            print(f"Player {self.current_player} is thinking about attack #{active_ship.attack_count + 1}...")
            

            state : MCTSState = {
                "game": self, 
                "current_player": self.current_player, 
                "decision_phase": "declare_target", 
                "active_ship_id": active_ship_id, 
                "declare_target": None
                }
            mcts_state_copy = copy.deepcopy(state)
            mcts_state_copy['game'].simulation_mode = True
            mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
            mcts.search(iterations=iterations)
            action = mcts.get_best_action()

            # If the AI decides to skip, break the attack loop
            if action is None or (action[0] != 'declare_target' and action[0] != 'skip_to_maneuver'):
                raise ValueError('MCTS must choose declare target or pass to maneuver')
            if action[0] == 'skip_to_maneuver':
                print(f"Player {self.current_player} skips to move ship step")
                break

            print(f"Player {self.current_player} has decided to attack. Determining optimal target...")


            # Decode the optimal attack path
            attacking_hull = action[1][0]
            defend_ship = self.ships[action[1][1]]
            defending_hull = action[1][2]

            # Execute the chosen attack in the REAL game
            print(f"Player {self.current_player} executes optimal attack: {active_ship.name} ({attacking_hull.name}) -> {defend_ship.name} ({defending_hull.name})")
            
            # Roll the dice and resolve the attack in the actual game state
            attack_pool = active_ship.roll_attack_dice(attacking_hull, defend_ship, defending_hull)
            if attack_pool:
                active_ship.resolve_damage(defend_ship, defending_hull, attack_pool)



        # === 3. MOVE SHIP PHASE ===
        print(f"Player {self.current_player} is thinking about the full maneuver...")

        # 1. Set up the initial state
        state : MCTSState = {
            "game": self,
            "current_player": self.current_player,
            "decision_phase": "determine_course",
            "active_ship_id": active_ship_id,
            "declare_target": None
            }
        mcts_state_copy = copy.deepcopy(state)
        mcts_state_copy['game'].simulation_mode = True

        # 2. Create and run MCTS
        mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
        mcts.search(iterations=iterations)

        action = mcts.get_best_action()
        if action is None or action[0] != 'determine_course' :
            raise ValueError('MCTS must choose course')
        
        course = action[1][0]
        placement = action[1][1]
        active_ship.speed = len(course)

        print(f"Player {self.current_player} chose final maneuver: Course {course}, Placement {'left' if placement == -1 else 'right'}")
        active_ship.move_ship(course, placement)
        

        # End of activation
        active_ship.activated = True




    def _execute_random_activation(self, ship_to_activate: Ship):
        """
        Executes a full, random activation for a given ship (for the non-MCTS player).
        """
        
        # 1. Attack Phase
        while ship_to_activate.attack_count < 2:
            valid_hulls = ship_to_activate.get_valid_attack_hull()
            if not valid_hulls: break
            
            attack_hull = random.choice(valid_hulls)
            defend_ship = random.choice(ship_to_activate.get_valid_target_ship(attack_hull))
            defend_hull = random.choice(ship_to_activate.get_valid_target_hull(attack_hull, defend_ship))
            attack_pool = ship_to_activate.roll_attack_dice(attack_hull, defend_ship, defend_hull)
            if attack_pool:
                ship_to_activate.resolve_damage(defend_ship, defend_hull, attack_pool)


        # 2. Maneuver Phase
        valid_speed = ship_to_activate.get_valid_speed()
        speed = random.choice(valid_speed)
        ship_to_activate.speed = speed

        course = []
        for joint in range(speed):
            valid_yaw = ship_to_activate.get_valid_yaw(speed, joint)
            yaw = random.choice(valid_yaw)
            course.append(yaw - 2)
        
        if speed > 0:
            valid_placement = ship_to_activate.get_valid_placement(course)
            placement = random.choice(valid_placement)
            ship_to_activate.move_ship(course, placement)
        
        ship_to_activate.activated = True


    def visualize(self, title : str, maneuver_tool = None) -> None:
        if self.simulation_mode:
            return
        visualizer.visualize(self, title, maneuver_tool)
        sleep(1)