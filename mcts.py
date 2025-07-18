import math
import random
import copy
from collections import defaultdict
from typing import TYPE_CHECKING

# Conditionally import Armada only for type checking
if TYPE_CHECKING:
    from armada import Armada
    from ship import Ship, HullSection

class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    Each node corresponds to a specific decision point in the game.
    """
    def __init__(self, state : dict[str, "str | Armada | Ship | int | list[int] | None"], parent : "Node | None" =None, action=None):
        self.parent = parent
        self.state = state  # The state dictionary, including game, phase, etc.
        self.action = action  # The action that led to this state
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions : list | None = None

    def uct_select_child(self, exploration_constant=1.414):
        """
        Selects a child node using the UCT formula.
        """
        # Add a small epsilon to visits to avoid division by zero on the first pass
        s = sorted(self.children, key=lambda c: c.wins / (c.visits + 1e-7) + exploration_constant * math.sqrt(math.log(self.visits + 1) / (c.visits + 1e-7)))
        return s[-1]

    def add_child(self, action, state):
        """
        Adds a new child node for the given action and state.
        """
        n = Node(parent=self, state=state, action=action)
        self.children.append(n)
        return n

    def update(self, result):
        """
        Updates the node's statistics from a simulation result.
        """
        self.visits += 1
        self.wins += result

class MCTS:
    """
    Hierarchical Monte Carlo Tree Search for Star Wars: Armada.
    This class is instantiated for a single decision point.
    """
    def __init__(self, initial_state: dict[str, "str | Armada | Ship | int | list[int] | None"], player: int):
        """
        Initializes the MCTS for a single decision.
        Args:
            initial_state (dict): The complete state dictionary from which to start the search.
            player (int): The player for whom the search is being conducted.
        """
        self.root = Node(state=initial_state)
        self.player = player

    def search(self, iterations: int):
        """
        Performs the MCTS search for a given number of iterations.
        """
        for _ in range(iterations):
            node = self.root
            state : dict = copy.deepcopy(node.state)

            # 1. Selection
            while node.untried_actions is not None and not node.untried_actions and node.children:
                node = node.uct_select_child()
                state = copy.deepcopy(node.state)

            # 2. Expansion
            if not state['game'].winner:
                if node.untried_actions is None:
                    node.untried_actions = self._get_possible_actions(state)

                if node.untried_actions:
                    action = node.untried_actions.pop()
                    child_state = self._apply_action(copy.deepcopy(state), action)
                    child_node = node.add_child(action, child_state)
                    node = child_node
                    state = child_state

            # 3. Simulation
            winner = self._simulate(state)

            # 4. Backpropagation
            while node is not None:
                # Result is from the perspective of the player who OWNS the MCTS agent
                result = 1 if winner == self.player else 0
                node.update(result)
                node = node.parent

    def _get_possible_actions(self, state: dict) -> list:
        """
        Returns a list of possible actions based on the current decision phase.
        """
        actions = []
        game : Armada = state['game']
        player : int = state['current_player']
        phase : str = state['decision_phase']
        
        active_ship : Ship | None = game.ships[state['active_ship_id']] if state['active_ship_id'] is not None else None

        if phase == "activation" or active_ship is None :
            valid_ships = game.get_valid_activation(player)
            for ship in valid_ships:
                actions.append(("activate_ship", ship.ship_id))
            if not valid_ships:
                actions.append(("pass_activation",))

        elif phase == "attack":
            if state['attack_count'] < 2:
                # Reset possible hulls at the start of the decision to ensure all options are considered
                active_ship.attack_possible_hull = [True] * 4
                valid_hulls = active_ship.get_valid_attack_hull()
                for hull in valid_hulls:
                    valid_targets = active_ship.get_valid_target(hull)
                    for target_ship, target_hull in valid_targets:
                        actions.append(("declare_attack", (hull, target_ship.ship_id, target_hull)))
            actions.append(("skip_to_maneuver",))

        elif phase == "maneuver_speed":
            valid_speeds = active_ship.get_valid_speed()
            for speed in valid_speeds:
                actions.append(("set_speed", speed))

        elif phase == "maneuver_yaw":
            speed = state['maneuver_speed']
            joint_index = state['maneuver_joint_index']
            valid_yaws = active_ship.get_valid_yaw(speed, joint_index)
            for yaw in valid_yaws:
                actions.append(("set_yaw", yaw))

        elif phase == "maneuver_placement":
            course = state['maneuver_course']
            valid_placements = active_ship.get_valid_placement(course)
            for placement in valid_placements:
                actions.append(("set_placement", placement))

        return actions

    def _apply_action(self, state: dict, action: tuple) -> dict:
        """
        Applies an action and transitions the state to the next decision point.
        """
        action_type = action[0]
        game : Armada = state['game']
        active_ship : Ship | None = game.ships[state['active_ship_id']] if state['active_ship_id'] is not None else None

        if action_type == "activate_ship" or active_ship is None:
            state['decision_phase'] = "attack"
            state['active_ship_id'] = action[1]

        elif action_type == "declare_attack":
            attack_hull, target_ship_id, defend_hull = action[1]
            defend_ship = next(s for s in game.ships if s.ship_id == target_ship_id)
            attack_pool = active_ship.roll_attack_dice(attack_hull, defend_ship, defend_hull)
            if attack_pool is not None:
                active_ship.resolve_damage(defend_ship, defend_hull, attack_pool)
            
            state['attack_count'] += 1
            # After the attack, the phase remains "attack" for the next decision
            # The game state has been updated, and the MCTS will be re-run from here.

        elif action_type == "skip_to_maneuver":
            state['decision_phase'] = "maneuver_speed"

        elif action_type == "set_speed":
            speed = action[1]
            state['maneuver_speed'] = speed
            if speed == 0:
                active_ship.speed = 0
                self._end_activation(state)
            else:
                state['decision_phase'] = "maneuver_yaw"
                state['maneuver_joint_index'] = 0
                state['maneuver_course'] = []

        elif action_type == "set_yaw":
            yaw = action[1]
            state['maneuver_course'].append(yaw - 2)
            state['maneuver_joint_index'] += 1
            if state['maneuver_joint_index'] >= state['maneuver_speed']:
                state['decision_phase'] = "maneuver_placement"

        elif action_type == "set_placement":
            placement = action[1]
            course = state['maneuver_course']
            active_ship.speed = state['maneuver_speed']
            active_ship.move_ship(course, placement)
            self._end_activation(state)
            
        elif action_type == "pass_activation":
             self._end_activation(state)

        return state

    def _end_activation(self, state: dict):
        """Helper to clean up after an activation and set up the next turn."""
        game = state['game']
        if state['active_ship_id'] is not None:
            ship = next(s for s in game.ships if s.ship_id == state['active_ship_id'])
            ship.activated = True

        # Reset for the next top-level decision
        state['decision_phase'] = "activation"
        state['active_ship_id'] = None
        state['attack_count'] = 0
        state['maneuver_speed'] = None
        state['maneuver_course'] = []
        state['maneuver_joint_index'] = 0

        # Switch player or end round
        if game.get_valid_activation(-state['current_player']):
            state['current_player'] *= -1
        elif not game.get_valid_activation(state['current_player']):
            game.status_phase()
            if not game.winner:
                game.round += 1
                state['current_player'] = 1

    def _simulate(self, state: dict) -> int:
        """
        Simulates a game from the given state to a terminal state using random moves.
        """
        sim_state = copy.deepcopy(state)
        sim_game = sim_state['game']
        
        max_simulation_steps = 500 # A generous safety net to prevent infinite loops
        steps = 0

        # Run until the game has a winner OR we hit the safety limit
        while sim_game.winner is None and steps < max_simulation_steps:
            possible_actions = self._get_possible_actions(sim_state)
            
            if not possible_actions:
                # If no actions, try to advance the game state (e.g., end of round)
                self._end_activation(sim_state)
                # If still no actions, the game is truly stuck or over.
                if not self._get_possible_actions(sim_state):
                    break
                continue

            random_action = random.choice(possible_actions)
            sim_state = self._apply_action(sim_state, random_action)
            sim_game = sim_state['game'] # Ensure we're referencing the updated game object
            steps += 1

        if sim_game.winner is None:
            p1_points = sim_game.get_point(1)
            p2_points = sim_game.get_point(-1)
            if p1_points > p2_points: sim_game.winner = 1
            elif p2_points > p1_points: sim_game.winner = -1
            else: sim_game.winner = 0 # Draw
        return sim_game.winner

    def get_best_action(self):
        """
        Returns the single best action from the root based on the most visits.
        """
        # if not self.root.children:
        #     return None
        # Find the child with the most visits (the "best" next move)
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.action
