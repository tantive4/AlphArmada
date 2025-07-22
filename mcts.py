import math
import random
import copy
from collections import defaultdict
from typing import TYPE_CHECKING, TypedDict, List, Optional, Dict
import sys
from itertools import combinations_with_replacement

# Conditionally import Armada only for type checking
if TYPE_CHECKING:
    from armada import Armada
    from ship import Ship, HullSection

class MCTSState(TypedDict):
    """
    A structured dictionary representing the complete state for an MCTS decision.
    This provides strong typing for each component of the state.
    """
    game: "Armada"
    current_player: int
    decision_phase: str
    active_ship_id: Optional[int]
    # Attack state
    attack_hull: Optional["HullSection"]
    defend_ship_id: Optional[int]
    defend_hull: Optional["HullSection"]
    # Maneuver state
    maneuver_speed: Optional[int]
    maneuver_course: List[int]
    maneuver_joint_index: int

class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    Can be a decision node (for a player) or a chance node (for random events).
    """
    def __init__(self, state : MCTSState, parent : "Node | None" =None, action=None):
        self.parent = parent
        self.state = state
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions : list | None = None
        self.is_chance_node = False

    def uct_select_child(self, exploration_constant=1.414, owner_player=1) -> "Node":
        """
        Selects a child node using the UCT formula. Should only be called on player decision nodes.
        """
        best_score = -float('inf')
        best_child = self.children[0]

        for child in self.children:
            win_rate = child.wins / (child.visits + 1e-7)
            
            if self.state['current_player'] != owner_player:
                win_rate = 1 - win_rate

            exploration = exploration_constant * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-7))
            
            uct_score = win_rate + exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child
            
        return best_child

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
    This class is instantiated for a single decision point and can recursively
    call itself to solve sub-problems. It also handles chance nodes for stochastic events.
    """
    # Defines which decision phases are sub-problems of a larger macro-action.
    # The iteration count is how much effort is spent on each step of the sub-problem.
    HIERARCHY_CONFIG = {
        # Attack Macro-Action Sub-problems
        "attack_choose_target_ship": 10,
        "attack_choose_target_hull": 10,
        # Maneuver Macro-Action Sub-problems
        "maneuver_yaw": 10,
        "maneuver_placement": 10
    }

    def __init__(self, initial_state: MCTSState, player: int, is_sub_search: bool = False):
        self.root = Node(state=initial_state)
        self.player = player
        self.is_sub_search = is_sub_search

    def search(self, iterations: int):
        possible_actions = self._get_possible_actions(self.root.state)
        if len(possible_actions) <= 1:
            if not self.is_sub_search:
                print("MCTS Thinking... [Only one move, skipping search.]")
            if possible_actions:
                action = possible_actions[0]
                child_state = self._apply_action(copy.deepcopy(self.root.state), action)
                self.root.add_child(action, child_state)
            return

        for i in range(iterations):
            node = self.root
            state = copy.deepcopy(node.state)

            # 1. Selection
            while node.untried_actions is not None and not node.untried_actions and node.children:
                if node.is_chance_node:
                    # For a chance node, sample a random outcome instead of using UCT.
                    if type(state) != MCTSState or state['active_ship_id'] is None or state['attack_hull'] is None or state['defend_ship_id'] is None or state['defend_hull'] is None:
                        raise ValueError("Invalid state for chance node: missing attack/defend info.")
                    
                    active_ship = state['game'].ships[state['active_ship_id']]
                    defend_ship = state['game'].ships[state['defend_ship_id']]
                    
                    # Use the ship's roll_dice to get a single, random outcome
                    random_outcome = active_ship.roll_attack_dice(state['attack_hull'], defend_ship, state['defend_hull'])
                    if not random_outcome:
                        raise ValueError("Chance node must have a valid random outcome.")
                    # Check if we've seen this random outcome before for this node
                    found_child = next((c for c in node.children if c.action[1] == random_outcome), None)
                    
                    if found_child:
                        # If we've seen it, continue selection down that path
                        node = found_child
                        state = node.state
                    else:
                        # If it's a new outcome, this is the node we will expand and simulate.
                        outcome_state = self._apply_dice_outcome(copy.deepcopy(state), random_outcome)
                        node = node.add_child(action=("dice_result", random_outcome), state=outcome_state)
                        state = node.state
                        break # Exit selection loop
                else:
                    # Standard player decision node, use UCT.
                    node = node.uct_select_child(owner_player=self.player)
                    state = copy.deepcopy(node.state)
            
            # 2. Expansion (for player decision nodes)
            if not state['game'].winner and not node.is_chance_node:
                if node.untried_actions is None:
                    node.untried_actions = self._get_possible_actions(state)

                if node.untried_actions:
                    action = node.untried_actions.pop()
                    child_state = self._apply_action(copy.deepcopy(state), action)
                    child_node = node.add_child(action, child_state)
                    node = child_node
                    state = child_state
                    
                    if action[0] == 'declare_full_attack':
                        node.is_chance_node = True

            # 3. Simulation
            winner = self._simulate(state)

            # 4. Backpropagation
            while node is not None:
                result = 1 if winner == self.player else 0
                node.update(result)
                node = node.parent
            
            if not self.is_sub_search:
                print(f"MCTS Iteration {i + 1}/{iterations} complete. Current best action: {self.get_best_action()}")

    def _simulate(self, state: MCTSState) -> int:
        sim_state = copy.deepcopy(state)

        # A top-level search hands off to an expert sub-search for hierarchical phases.
        if not self.is_sub_search and state['decision_phase'] in self.HIERARCHY_CONFIG:
            sub_search_iterations = self.HIERARCHY_CONFIG[state['decision_phase']]
            _best_action_sequence, resulting_state = self._run_sub_search(sim_state, sub_search_iterations)
            sim_state = resulting_state
        
        # Random rollout continues from the resulting state.
        sim_game = sim_state['game']
        max_simulation_steps = 500
        steps = 0
        while sim_game.winner is None and steps < max_simulation_steps:
            possible_actions = self._get_possible_actions(sim_state)
            
            if not possible_actions:
                self._end_activation(sim_state)
                if not self._get_possible_actions(sim_state):
                    break
                continue

            random_action = random.choice(possible_actions)
            sim_state = self._apply_action(sim_state, random_action)
            sim_game = sim_state['game']
            steps += 1

        if sim_game.winner is None:
            p1_points = sim_game.get_point(1)
            p2_points = sim_game.get_point(-1)
            if p1_points > p2_points: sim_game.winner = 1
            elif p2_points > p1_points: sim_game.winner = -1
            else: sim_game.winner = 0
        return sim_game.winner

    def _run_sub_search(self, state: MCTSState, iterations: int) -> tuple[List[tuple], MCTSState]:
        sub_state = copy.deepcopy(state)
        action_sequence = []

        # This general loop handles any sub-problem defined in the config.
        while sub_state['decision_phase'] in self.HIERARCHY_CONFIG:
            sub_mcts = MCTS(initial_state=sub_state, player=self.player, is_sub_search=True)
            sub_mcts.search(iterations=iterations)
            
            best_action = sub_mcts.get_best_action()

            if best_action is None:
                return [], sub_state

            action_sequence.append(best_action)
            sub_state = self._apply_action(sub_state, best_action)

            if sub_state['game'].winner:
                break
        
        return action_sequence, sub_state

    def _apply_dice_outcome(self, state: MCTSState, outcome: list) -> MCTSState:
        """
        Applies a specific dice roll outcome to the state, resolving damage and setting up the next phase.
        """
        outcome_state = copy.deepcopy(state)
        if outcome_state['active_ship_id'] is None or outcome_state['defend_ship_id'] is None or state['attack_hull'] is None or outcome_state['defend_hull'] is None:
            raise ValueError("Active ship and defend ship must be set before applying dice outcome.")
        
        active_ship_copy = outcome_state['game'].ships[outcome_state['active_ship_id']]
        defend_ship_copy = outcome_state['game'].ships[outcome_state['defend_ship_id']]
        
        active_ship_copy.resolve_damage(defend_ship_copy, outcome_state['defend_hull'], outcome)
        
        active_ship_copy.attack_count += 1
        active_ship_copy.attack_possible_hull[state['attack_hull'].value] = False
        
        # After one attack, the next decision is the start of the next attack
        outcome_state['decision_phase'] = 'attack'
        
        return outcome_state

    def _get_possible_actions(self, state: MCTSState) -> list:
        actions = []
        game = state['game']
        player = state['current_player']
        phase = state['decision_phase']
        active_ship : Ship | None = game.ships[state['active_ship_id']] if state['active_ship_id'] is not None else None

        if phase == "activation" or active_ship is None:
            valid_ships = game.get_valid_activation(player)
            for ship in valid_ships:
                actions.append(("activate_ship", ship.ship_id))
            if not valid_ships:
                actions.append(("pass_activation",))
        
        elif phase == "attack":
            if active_ship.attack_count < 2:
                valid_hulls = active_ship.get_valid_attack_hull()
                for hull in valid_hulls:
                    actions.append(("choose_attack_hull", hull))
            actions.append(("skip_to_maneuver",))
        
        elif phase == "attack_choose_target_ship":
            if state['attack_hull'] is None:
                raise ValueError("Attack hull must be selected before choosing target ship.")
            valid_ships = active_ship.get_valid_target_ship(state['attack_hull'])
            for ship in valid_ships:
                actions.append(("choose_target_ship", ship.ship_id))

        elif phase == "attack_choose_target_hull":
            if state['defend_ship_id'] is None or state['attack_hull'] is None:
                raise ValueError("Defend ship and attack hull must be selected before declaring full attack.")
            target_ship = game.ships[state['defend_ship_id']]
            valid_hulls = active_ship.get_valid_target_hull(state['attack_hull'], target_ship)
            for hull in valid_hulls:
                actions.append(("declare_full_attack", hull))

        elif phase == "maneuver_speed":
            valid_speeds = active_ship.get_valid_speed()
            for speed in valid_speeds:
                actions.append(("set_speed", speed))

        elif phase == "maneuver_yaw":
            if state['maneuver_speed'] is None:
                raise ValueError("Maneuver speed must be set before setting yaw.")
            speed = state['maneuver_speed']
            joint_index = state['maneuver_joint_index']
            valid_yaws = active_ship.get_valid_yaw(speed, joint_index)
            for yaw in valid_yaws:
                actions.append(("set_yaw", yaw))

        elif phase == "maneuver_placement":
            if state['maneuver_speed'] is None:
                raise ValueError("Maneuver speed must be set before setting yaw.")
            course = state['maneuver_course']
            valid_placements = active_ship.get_valid_placement(course)
            for placement in valid_placements:
                actions.append(("set_placement", placement))

        return actions

    def _apply_action(self, state: MCTSState, action: tuple) -> MCTSState:
        action_type = action[0]
        game : Armada = state['game']
        active_ship : Ship | None = game.ships[state['active_ship_id']] if state['active_ship_id'] is not None else None

        if action_type == "activate_ship":
            state['active_ship_id'] = action[1]
            state['decision_phase'] = "attack"
            state['attack_hull'] = None
            state['defend_ship_id'] = None
            state['defend_hull'] = None

        elif action_type == "choose_attack_hull":
            state['attack_hull'] = action[1]
            state['decision_phase'] = "attack_choose_target_ship"

        elif action_type == "choose_target_ship":
            state['defend_ship_id'] = action[1]
            state['decision_phase'] = "attack_choose_target_hull"

        elif action_type == "declare_full_attack":
            state['defend_hull'] = action[1]
            
        elif action_type == "skip_to_maneuver":
            state['decision_phase'] = "maneuver_speed"

        elif action_type == "set_speed":
            speed = action[1]
            state['maneuver_speed'] = speed
            if speed == 0:
                if active_ship: active_ship.speed = 0
                self._end_activation(state)
            else:
                state['decision_phase'] = "maneuver_yaw"
                state['maneuver_joint_index'] = 0
                state['maneuver_course'] = []

        elif action_type == "set_yaw":
            yaw = action[1]
            state['maneuver_course'].append(yaw - 2)
            state['maneuver_joint_index'] += 1
            if state['maneuver_speed'] is None:
                raise ValueError("Maneuver speed must be set before setting yaw.")
            if state['maneuver_joint_index'] >= state['maneuver_speed']:
                state['decision_phase'] = "maneuver_placement"

        elif action_type == "set_placement":
            if state['maneuver_speed'] is None:
                raise ValueError("Maneuver speed must be set before setting yaw.")
            placement = action[1]
            course = state['maneuver_course']
            if active_ship:
                active_ship.speed = state['maneuver_speed']
                active_ship.move_ship(course, placement)
            self._end_activation(state)
            
        elif action_type == "pass_activation":
             self._end_activation(state)

        return state

    def _end_activation(self, state: MCTSState):
        game = state['game']
        if state['active_ship_id'] is not None:
            ship = game.ships[state['active_ship_id']]
            ship.activated = True

        state['decision_phase'] = "activation"
        state['active_ship_id'] = None
        # Reset maneuver state
        state['maneuver_speed'] = None
        state['maneuver_course'] = []
        state['maneuver_joint_index'] = 0
        # Reset attack state
        state['attack_hull'] = None
        state['defend_ship_id'] = None
        state['defend_hull'] = None

        if game.get_valid_activation(-state['current_player']):
            state['current_player'] *= -1
        elif not game.get_valid_activation(state['current_player']):
            game.status_phase()
            if not game.winner:
                game.round += 1
                state['current_player'] = 1

    def get_best_action(self):
        if not self.root.children:
            possible_actions = self._get_possible_actions(self.root.state)
            return possible_actions[0] if possible_actions else None
            
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.action
