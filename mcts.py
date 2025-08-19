from __future__ import annotations

import math
import random
import copy
from typing import TYPE_CHECKING, TypedDict, List, NotRequired, TypeAlias, Tuple, Literal, Optional, cast
import itertools
from ship import Ship, HullSection
from time import sleep
from game_phase import GamePhase, ActionType
# Conditionally import Armada only for type checking
if TYPE_CHECKING:
    from armada import Armada




class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    Can be a decision node (for a player) or a chance node (for random events).
    """
    def __init__(self, decision_player : int | None, parent : Node | None =None, action : ActionType.Action | None=None, chance_node : bool = False) -> None :
        
        if (parent is None) != (action is None):
            raise ValueError("Root nodes must have no parent and no action, while child nodes must have both.")

        self.decision_player : int | None = decision_player
        self.parent : Node | None = parent
        self.action : ActionType.Action | None = action
        self.children : list[Node] = []
        self.wins : float = 0
        self.visits : int = 0
        self.untried_actions : Optional[list[ActionType.Action]] = None
        self.chance_node = chance_node


    def uct_select_child(self, exploration_constant=2) -> Node:
        """
        Selects a child node using the UCT formula with random tie-breaking.
        """
        if self.chance_node:
            raise ValueError("uct_select_child called on a chance node, which should not happen in MCTS.")

        if not self.children:
            # This should not happen if called on a non-terminal, expanded node
            raise ValueError("uct_select_child called on a node with no children")

        if self.visits == 0:
            # Fallback for an unvisited node, though MCTS should ideally not call UCT here.
            # The main loop should handle the initial expansion of each child once.
            return random.choice(self.children)
            
        log_parent_visits = math.log(self.visits)
        
        best_score = -float('inf')
        best_children = [] # Use a list to hold all children with the best score

        for child in self.children:
            # Exploit term (win rate)
            if child.visits == 0:
                return child
            win_rate = child.wins / (child.visits + 1e-7)

            # Explore term
            exploration = exploration_constant * math.sqrt(log_parent_visits / (child.visits + 1e-7))
            
            uct_score = win_rate + exploration

            if uct_score > best_score:
                best_score = uct_score
                best_children = [child] # Found a new best, start a new list
            elif uct_score == best_score:
                best_children.append(child) # It's a tie, add to the list

        return random.choice(best_children) # Randomly choose from the best options

    def add_child(self, action : ActionType.Action, game : Armada) -> Node :
        """
        Adds a new child node for the given action and game.
        """
        node = Node(decision_player=game.decision_player, parent=self, action=action)
        self.children.append(node)
        return node

    def update(self, result) -> None :
        """
        Updates the node's statistics from a simulation result.
        """
        self.visits += 1
        self.wins += result




class MCTS:
    """
    Basic Monte Carlo Tree Search for Star Wars: Armada.

    """

    def __init__(self, initial_game: Armada) -> None:
        self.game = initial_game
        self.root : Node = Node(decision_player=initial_game.decision_player)
        if self.root.decision_player is None :
            raise ValueError("MCTS requires a decision player to be set in the game.")




    def search(self, iterations: int) -> None:
        # single decision optimization

        possible_actions = self.game.get_possible_actions()
        if len(possible_actions) == 1:
            action = possible_actions[0]
            self.game.apply_action(action)
            self.root.add_child(action, self.game)
            return

        for i in range(iterations):
            game : Armada = copy.deepcopy(self.game)
            node : Node = self.root
            
            # 1. Selection
            while (node.untried_actions is not None and not node.untried_actions and node.children) or node.chance_node:
                if node.chance_node:
                    # For a chance node, sample a random outcome instead of using UCT.
                    if game['active_ship_id'] is None or game['declare_target'] is None :
                        raise ValueError("Invalid game for chance node: missing attack/defend info.")
                    
                    active_ship = game['game'].ships[game['active_ship_id']]
                    declare_target = game['declare_target']
                    attack_hull = declare_target[0]
                    defend_ship = game['game'].ships[declare_target[1]]
                    defend_hull = declare_target[2]

                    
                    # Use the ship's roll_dice to get a single, random outcome
                    attack_dice = active_ship.roll_attack_dice(attack_hull, defend_ship, defend_hull)
                    if not attack_dice:
                        raise ValueError("Invalid attack")

                    # Check if we've seen this random outcome before for this node
                    dice_roll_result_node = next((child for child in node.children if child.action is not None and child.action[1] == attack_dice), None)
                    
                    if dice_roll_result_node:
                        # If we've seen it, continue selection down that path
                        node = dice_roll_result_node
                        game = node.game
                    else:
                        # If it's a new outcome, this is the node we will expand and simulate.
                        outcome_game = self._apply_dice_outcome(copy.deepcopy(game), attack_dice)
                        node = node.add_child(action=("attack_dice_roll", attack_dice), game=outcome_game)
                        game = node.game
                        game['decision_phase'] = 'declare_target'  # Reset to next phase after dice roll
                        break # Exit selection loop
                else:
                    # Standard player decision node, use UCT.
                    node = node.uct_select_child()
                    if node.action is None:
                        raise ValueError("Child node must have an action.")
                    game.apply_action(node.action)
            
            # 2. Expansion (for player decision nodes)
            if not game.winner :
                if node.untried_actions is None:
                    node.untried_actions = game.get_possible_actions()
                    random.shuffle(node.untried_actions)

                if node.untried_actions:
                    action = node.untried_actions.pop()
                    game.apply_action(action)
                    child_node = node.add_child(action, game)
                    node = child_node
                    
                    if action[0] == 'roll_dice_action':
                        node.chance_node = True

            # 3. Simulation
            simulation_result = self._simulate(game)

            # 4. Backpropagation (Updated for -1 to 1 scoring)
            temp_node = node
            while temp_node is not None:
                # The result must be from the perspective of the player who made the move at the parent node.
                perspective_player = temp_node.decision_player
                
                # The simulation_result is always from Player 1's perspective.
                # If the current node's move was made by Player -1, we flip the score.
                result_for_node = simulation_result if perspective_player == 1 else -simulation_result
                
                temp_node.update(result_for_node)
                temp_node = temp_node.parent
            
            # if (i+1) % 20 == 0:
                # print(f"MCTS Iteration {i + 1}/{iterations} complete. Current best action: {self.get_best_action()} best wins: {int(self.get_best_child().wins)}, best visits: {self.get_best_child().visits}, depth: {depth}")
            with open('simulation_log.txt', 'a') as f: f.write(f"\n{i+1} iteration. Total Win {round(self.root.wins,2)}. Best Action {self.get_best_action()} \n{[(node.action, round(node.wins,2), node.visits) for node in self.root.children]}")



    def _simulate(self, game: Armada) -> float:
        """
        simulate random game from current game and return the winner
        winner is the MoV value, normalized as -1 ~ 1
        """
        sim_game = copy.deepcopy(game)
        
        # Random rollout continues from the resulting game.
        sim_game = sim_game['game']
        sim_game.refresh_ship_links()
        max_simulation_steps = 500
        steps = 0
        while sim_game.winner is None and steps < max_simulation_steps:
            if sim_game['decision_phase'] == 'attack_dice_roll':
                # Simulate a random dice outcome
                if sim_game['active_ship_id'] is None or sim_game['declare_target'] is None:
                    raise ValueError("Active ship and defend ship must be set before rolling dice.")
                active_ship = sim_game.ships[sim_game['active_ship_id']]
                declare_target = sim_game['declare_target']
                attack_hull = declare_target[0]
                defend_ship = sim_game.ships[declare_target[1]]
                defend_hull = declare_target[2]

                attack_dice = active_ship.roll_attack_dice(attack_hull, defend_ship, defend_hull)
                if not attack_dice:
                    raise ValueError("Invalid attack dice outcome during simulation")

                sim_game = self._apply_dice_outcome(sim_game, attack_dice)
                sim_game = sim_game['game']
                continue

            possible_actions = self._get_possible_actions(sim_game)
            
            if not possible_actions:
                raise ValueError("No action during simulation")

            random_action = random.choice(possible_actions)
            sim_game = self._apply_action(sim_game, random_action)

            sim_game = sim_game['game']
            steps += 1
        if sim_game.winner is None:
            raise RuntimeError("Simulation exceeded maximum steps without a winner.")
        with open('simulation_log.txt', 'a') as f: f.write(f"\nsimulation ends : {sim_game.winner}")
        return sim_game.winner

    def get_best_action(self) -> ActionType.Action:
        if not self.root.children:
            possible_actions = self._get_possible_actions(self.root.game)
            return possible_actions[0]
            
        best_child = max(self.root.children, key=lambda c: c.visits)
        if best_child.action is None :
            raise ValueError('Child Node needs action from parent')
        return best_child.action
    
    def get_best_child(self) -> Node:
        if not self.root.children:
            raise ValueError("No children available to select best child")
        return max(self.root.children, key=lambda c: c.visits)