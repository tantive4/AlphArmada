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
        self.winner : float | None = None
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
    

    def status_phase(self) -> None:
        # 1. Refresh all active ships for the next round
        for ship in self.ships:
            if not ship.destroyed:
                ship.refresh()

        # 2. Check for game-ending conditions
        player_1_eliminated = self.total_destruction(1)
        player_2_eliminated = self.total_destruction(-1)
        is_game_over = player_1_eliminated or player_2_eliminated or self.round == 6

        # 3. If the game is over, determine the winner
        if is_game_over:
            # This check is important to ensure the winner is only set once
            if self.winner is None:
                p1_points = self.get_point(1)
                p2_points = self.get_point(-1)
                margin_of_victory = p1_points - p2_points

                # Set a non-zero winner value based on the margin
                self.winner = margin_of_victory / 100
            return # Stop the function here; do not advance the round

        # 4. If the game is not over, advance to the next round
        else:
            self.round += 1
            self.current_player = 1
            if not self.simulation_mode:
                with open('simulation_log.txt', 'a') as f: f.write(f"\nEnd of Round {self.round}.")
        

    def get_point(self, player : int) -> int :
        return sum(ship.point for ship in self.ships if ship.player != player and ship.destroyed)

    def play_round(self) -> None:
        """
        Plays a single round of the game, alternating between players.
        Each player activates their ships, attacks, and maneuvers.
        """
        self.ship_phase()
        self.status_phase()


    def ship_phase(self) -> None:
        """
        The ship phase is where players activate their ships, attack, and maneuver.
        """
        # Continue as long as any ship can be activated by any player
        while any(self.get_valid_activation(1)) or any(self.get_valid_activation(-1)):

            # If the current player has no ships left to activate, pass the turn to the opponent.
            if not any(self.get_valid_activation(self.current_player)):
                self.current_player *= -1
                continue # Restart the loop for the other player's turn

            # It's the current player's turn to make a move.
            self.visualize(f'ROUND {self.round} | Player {self.current_player}\'s Turn')

            # --- MCTS Execution ---
            if self.current_player == 1:
                with open('simulation_log.txt', 'a') as f: f.write(f"\nPlayer 1 (MCTS) is playing...")
                self._execute_mcts_turn(iterations=1500)
            elif self.current_player == -1:
                with open('simulation_log.txt', 'a') as f: f.write(f"\nPlayer -1 (MCTS) is playing...")
                self._execute_mcts_turn(iterations=500)

            # --- Switch Player for the next activation ---
            # IMPORTANT: Only switch if the opponent can still activate.
            if any(self.get_valid_activation(-self.current_player)):
                self.current_player *= -1


    def play(self) -> None :
        """
        The main game loop, structured by rounds and alternating player turns.
        """
        while self.winner is None:
            self.play_round()

        self.visualize(f'Player {1 if self.winner is not None and self.winner > 0 else -1} has won!')
        with open('simulation_log.txt', 'a') as f: f.write(f"Player {1 if self.winner is not None and self.winner > 0 else -1} has won!")

    def _execute_mcts_turn(self, iterations : int = 1000):
        """
        Executes a full ship activation for the MCTS player by breaking it down
        into sequential decisions.
        """
        # === 1. CHOOSE SHIP TO ACTIVATE ===
        with open('simulation_log.txt', 'a') as f: f.write(f"\nPlayer {self.current_player} is thinking about phase: activation")
        state : MCTSState = {
            "game": self, 
            "current_player": self.current_player, 
            "decision_phase": "ship_activation",
            "active_ship_id": None, 
            "declare_target": None,
            "course": None
            }
        mcts_state_copy = copy.deepcopy(state)
        mcts_state_copy['game'].simulation_mode = True
        mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
        mcts.search(iterations=iterations)
        action = mcts.get_best_action()

        if action is None or action[0] != 'activate_ship_action':
            raise ValueError('MCTS must choose ship to activate')
        
        active_ship_id : int = action[1]
        active_ship = self.ships[active_ship_id]
        with open('simulation_log.txt', 'a') as f: f.write(f"\nPlayer {self.current_player} chose action: {action[0], active_ship.name}")


        # === 2. ATTACK PHASE ===

        while active_ship.attack_count < 2 :
            with open('simulation_log.txt', 'a') as f: f.write(f"\nPlayer {self.current_player} is thinking about attack #{active_ship.attack_count + 1}...")
            

            state : MCTSState = {
                "game": self, 
                "current_player": self.current_player, 
                "decision_phase": "declare_target", 
                "active_ship_id": active_ship_id, 
                "declare_target": None,
                "course": None,
                }
            mcts_state_copy = copy.deepcopy(state)
            mcts_state_copy['game'].simulation_mode = True
            mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
            mcts.search(iterations=iterations)
            action = mcts.get_best_action()

            # If the AI decides to skip, break the attack loop
            if action is None or (action[0] != 'declare_target_action' and action[0] != 'pass_attack'):
                raise ValueError('MCTS must choose declare target or pass to maneuver')
            if action[0] == 'pass_attack':
                with open('simulation_log.txt', 'a') as f: f.write(f"\nPlayer {self.current_player} skips to move ship step")
                break

            with open('simulation_log.txt', 'a') as f: f.write(f"\nPlayer {self.current_player} has decided to attack. Determining optimal target...")
            if action[1] is None :
                raise ValueError('Action must contain target information')

            # Decode the optimal attack path
            attacking_hull = action[1][0]
            defend_ship = self.ships[action[1][1]]
            defending_hull = action[1][2]

            # Execute the chosen attack in the REAL game
            with open('simulation_log.txt', 'a') as f: f.write(f"\nPlayer {self.current_player} executes optimal attack: {active_ship.name} ({attacking_hull.name}) -> {defend_ship.name} ({defending_hull.name})")
            
            # Roll the dice and resolve the attack in the actual game state
            attack_pool = active_ship.roll_attack_dice(attacking_hull, defend_ship, defending_hull)
            if attack_pool:
                active_ship.resolve_damage(defend_ship, defending_hull, attack_pool)



        # # === 3. MOVE SHIP PHASE ===
        # with open('simulation_log.txt', 'a') as f: f.write(f"\nPlayer {self.current_player} is thinking about the full maneuver...")

        # # 1. Set up the initial state
        # state : MCTSState = {
        #     "game": self,
        #     "current_player": self.current_player,
        #     "decision_phase": "determine_course",
        #     "active_ship_id": active_ship_id,
        #     "declare_target": None,
        #     "course": None
        #     }
        # mcts_state_copy = copy.deepcopy(state)
        # mcts_state_copy['game'].simulation_mode = True

        # # 2. Create and run MCTS
        # mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
        # mcts.search(iterations=iterations)

        # action = mcts.get_best_action()
        # if action is None or action[0] != 'determine_course_action' :
        #     raise ValueError('MCTS must choose course')
        
        # course = action[1][0]
        # placement = action[1][1]
        # active_ship.speed = len(course)

        # with open('simulation_log.txt', 'a') as f: f.write(f"\nPlayer {self.current_player} chose final maneuver: Course {course}, Placement {'left' if placement == -1 else 'right'}")
        # active_ship.move_ship(course, placement)


        # === 3. HIERARCHICAL MANEUVER PHASE (Single Search, Walk Down) ===
        with open('simulation_log.txt', 'a') as f: f.write(f"\nPhase: Maneuver Planning")
        state = {
            "game": self, "current_player": self.current_player, "decision_phase": "determine_speed",
            "active_ship_id": active_ship_id, "declare_target": None, "course": None
        }
        mcts_state_copy = copy.deepcopy(state)
        mcts_state_copy['game'].simulation_mode = True
        mcts = MCTS(initial_state=mcts_state_copy, player=self.current_player)
        mcts.search(iterations=iterations)


        # --- Walk down the tree to get the full maneuver ---
        # Get Speed
        best_node = mcts.get_best_child()
        with open('simulation_log.txt', 'a') as f: f.write(f"\nTotal Win {round(best_node.wins)}. Best Action {best_node.action} \n{[(node.action, round(node.wins), node.visits) for node in best_node.children]}")

        speed_action = best_node.action
        if speed_action is None or speed_action[0] != 'determine_speed_action': raise ValueError("MCTS failed to return a speed action.")
        speed = speed_action[1]
        active_ship.speed = speed

        course = []
        if speed > 0:
            # Get Yaws by walking down the tree
            for joint_index in range(speed):
                best_node = max(best_node.children, key=lambda c: c.visits)
                with open('simulation_log.txt', 'a') as f: f.write(f"\nTotal Win {round(best_node.wins)}. Best Action {best_node.action} \n{[(node.action, round(node.wins), node.visits) for node in best_node.children]}")
                
                yaw_action = best_node.action
                if yaw_action is None or yaw_action[0] != 'determine_yaw_action': raise ValueError("MCTS failed to return a yaw action.")
                course.append(yaw_action[1])

            # Get Placement
            best_node = max(best_node.children, key=lambda c: c.visits)
            with open('simulation_log.txt', 'a') as f: f.write(f"\nTotal Win {round(best_node.wins)}. Best Action {best_node.action} \n{[(node.action, round(node.wins), node.visits) for node in best_node.children]}")
            placement_action = best_node.action
            if placement_action is None or placement_action[0] != 'determine_placement_action': raise ValueError("MCTS failed to return a placement action.")
            placement = placement_action[1]
        else:
            placement = 1 # Default placement for speed 0
        
        placement_str = 'left' if placement == -1 else 'right'

        # --- Execute the final maneuver ---
        with open('simulation_log.txt', 'a') as f: f.write(f"\nAction: Execute Maneuver {course, placement_str}")
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
        # sleep(1)

    def refresh_ship_links(self) -> None:
        """Ensures all ship objects refer to this game instance."""
        for ship in self.ships:
            ship.game = self