import numpy as np
from ship import Ship
import random
from shapely.geometry import Polygon
import visualizer
import copy
from mcts import MCTS, MCTSState
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

    # def ship_phase(self) -> None :
        # player = 1

        # while True :
        #     valid_activations = self.get_valid_activation(player)
        #     opponent_activations = self.get_valid_activation(-player)
        #     if not valid_activations and not opponent_activations:
        #         break
        #     if not valid_activations:
        #         player *= -1
        #         continue

        #     ship_to_activate = random.choice(valid_activations)
        #     ship_to_activate.activate()

        #     player *= -1

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
        self.round += 1
        self.current_player = 1 # Player 1 always starts the new round.

    def get_point(self, player : int) -> int :
        return sum(ship.point for ship in self.ships if ship.player != player and ship.destroyed)
    
    def play(self) -> None :
        """
        The main game loop, structured by rounds and alternating player turns.
        """
        while self.round <= 6 and self.winner is None:
            self.visualize(f'ROUND {self.round} START')
            
            # Ship Phase for the current round
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

            # End of Round Status Phase
            self.status_phase()
            if self.winner is not None:
                break # A player was eliminated, end the game.
            


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
            "decision_phase": "activation", 
            "active_ship_id": None, 
            "maneuver_speed": None, 
            "maneuver_course": [], 
            "maneuver_joint_index": 0 
            }
        mcts_state_copy = copy.deepcopy(state)
        mcts_state_copy['game'].simulation_mode = True
        mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
        mcts.search(iterations=iterations)
        action = mcts.get_best_action()

        if action is None or action[0] != 'activate_ship':
            print(f"Player {self.current_player} chose not to activate. Passing turn.")
            return
        \
        active_ship_id : int = action[1]
        active_ship = self.ships[active_ship_id]
        print(f"Player {self.current_player} chose action: {action[0], active_ship.name}")

        # === 2. ATTACK PHASE ===
        for attack_num in range(2):
            print(f"Player {self.current_player} is thinking about phase: attack ({attack_num + 1}/2)")
            state = { 
                "game": self,
                "current_player": self.current_player, 
                "decision_phase": "attack", 
                "active_ship_id": active_ship_id, 
                "maneuver_speed": None, 
                "maneuver_course": [], 
                "maneuver_joint_index": 0 
                }
            mcts_state_copy = copy.deepcopy(state)
            mcts_state_copy['game'].simulation_mode = True
            mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
            mcts.search(iterations=iterations)
            attack_action = mcts.get_best_action()
            if attack_action is None or attack_action[0] == 'skip_to_maneuver':
                print(f"Player {self.current_player} skips remaining attacks.")
                break
            

            # Execute the chosen attack
            _, (attack_hull, target_ship_id, defend_hull) = attack_action
            defend_ship = self.ships[target_ship_id]
            print(f"Player {self.current_player} chose action: {attack_action[0]}\n {active_ship.name} attacks {defend_ship.name} {attack_hull.name} to {defend_hull.name}")
            attack_pool = active_ship.roll_attack_dice(attack_hull, defend_ship, defend_hull)
            if attack_pool:
                active_ship.resolve_damage(defend_ship, defend_hull, attack_pool)
            active_ship.attack_possible_hull[attack_hull.value] = False

        # === 3. MANEUVER PHASE ===
        # 3a. Choose Speed
        print(f"Player {self.current_player} is thinking about phase: maneuver_speed")
        state = {
            "game": self, 
            "current_player": self.current_player, 
            "decision_phase": "maneuver_speed", 
            "active_ship_id": active_ship_id,  
            "maneuver_speed": None, 
            "maneuver_course": [], 
            "maneuver_joint_index": 0 
            }
        mcts_state_copy = copy.deepcopy(state)
        mcts_state_copy['game'].simulation_mode = True
        mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
        mcts.search(iterations=iterations)
        speed_action = mcts.get_best_action()
        print(f"Player {self.current_player} chose action: {speed_action}")

        speed = speed_action[1]
        active_ship.speed = speed
        if speed == 0:
            active_ship.move_ship([], 1)
            active_ship.activated = True
            return

        # 3b. Choose Yaws
        course = []
        for joint_index in range(speed):
            print(f"Player {self.current_player} is thinking about phase: maneuver_yaw (joint {joint_index+1}/{speed})")
            state = { 
                "game": self, 
                "current_player": self.current_player,
                "decision_phase": "maneuver_yaw", 
                "active_ship_id": active_ship_id, 
                "maneuver_speed": speed, 
                "maneuver_course": course, 
                "maneuver_joint_index": joint_index 
                }
            mcts_state_copy = copy.deepcopy(state)
            mcts_state_copy['game'].simulation_mode = True
            mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
            mcts.search(iterations=iterations)
            yaw_action = mcts.get_best_action()
            print(f"Player {self.current_player} chose action: {yaw_action[0]}\n{yaw_action[1] - 2} at joint {joint_index+1}/{speed}")
            course.append(yaw_action[1] - 2)

        # 3c. Choose Placement
        print(f"Player {self.current_player} is thinking about phase: maneuver_placement")
        state = { 
            "game": self, 
            "current_player": self.current_player, 
            "decision_phase": "maneuver_placement", 
            "active_ship_id": active_ship_id,  
            "maneuver_speed": speed, 
            "maneuver_course": course, 
            "maneuver_joint_index": speed 
            }
        mcts_state_copy = copy.deepcopy(state)
        mcts_state_copy['game'].simulation_mode = True
        mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
        mcts.search(iterations=iterations)
        placement_action = mcts.get_best_action()
        print(f"Player {self.current_player} chose action: {placement_action[0]} {"left" if placement_action[1] == -1 else "right"}")
        
        placement = placement_action[1]
        active_ship.move_ship(course, placement)
        
        # Finally, mark the ship as activated for this round.
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
            valid_targets = ship_to_activate.get_valid_target(attack_hull)
            
            defend_ship, defend_hull = random.choice(valid_targets)
            
            attack_pool = ship_to_activate.roll_attack_dice(attack_hull, defend_ship, defend_hull)
            if attack_pool:
                ship_to_activate.resolve_damage(defend_ship, defend_hull, attack_pool)
            
            ship_to_activate.attack_count += 1
            ship_to_activate.attack_possible_hull[attack_hull.value] = False

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