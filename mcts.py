from __future__ import annotations
import math
import random
from typing import TYPE_CHECKING

import numpy as np

import dice
from action_phase import Phase, ActionType, get_action_str
if TYPE_CHECKING:
    from armada import Armada, AttackInfo


class Config:

    ITERATION =1600
    EXPLORATION_CONSTANT =2


class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    Can be a decision node (for a player) or a chance node (for random events).
    """
    def __init__(self, 
                 game : Armada,
                 action : ActionType, 
                 parent : Node | None =None, 
                 policy : float = 0,
                 action_index : int = 0,) -> None :
        
        self.snapshot = game.get_snapshot()
        self.decision_player : int | None = game.decision_player # decision player used when get_possible_action is called on this node
        self.chance_node : bool = self.snapshot['phase'] == Phase.ATTACK_ROLL_DICE
        self.information_set : bool = self.snapshot['phase'] == Phase.SHIP_REVEAL_COMMAND_DIAL


        self.parent : Node | None = parent
        self.action : ActionType = action
        self.children : list[Node] = []
        self.wins : float = 0
        self.visits : int = 0
        
        self.policy : float = policy # policy value from parent node state
        self_index : int = action_index # index of the action in the full action list from parent node state


    def select_child(self, use_policy : bool = False) -> Node:
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
            
        best_child = random.choice(self.children)
        best_ucb = -np.inf
        
        for child in self.children:
            if use_policy: ucb = self.get_pucb(child)
            else : ucb = self.get_ucb(child)
            
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child
    
    def get_ucb(self, child : Node, exploration_constant=2) -> float:
        if child.visits == 0:
            return float('inf')
        q_value : float = child.wins / child.visits
        return q_value + exploration_constant * math.sqrt(math.log(self.visits)) / child.visits

    def get_pucb(self, child : Node, exploration_constant=2) -> float:
        if child.visits == 0:
            q_value : float = 0
        else: 
            q_value : float = child.wins / child.visits
        return q_value + exploration_constant * child.policy * math.sqrt(self.visits) / (1 + child.visits)


    def add_child(self, action : ActionType, game : Armada, policy : float = 0, action_index : int = 0) -> Node :
        """
        Adds a new child node for the given action and game.
        Args:
            action : The action leading to the new child node.
            game : The game state after applying the action, used to determine the decision player for the child.
        """
        node = Node(game=game, parent=self, action=action, policy=policy, action_index=action_index)
        self.children.append(node)
        return node

    def update(self, result) -> None :
        """
        Updates the node's statistics from a simulation result.
        """
        self.visits += 1
        self.wins += result
    
    def backpropagate(self, simulation_result : float) -> None :
        """
        Backpropagates the simulation result up the tree.
        """
        node = self
        while node is not None:
            # The result must be from the perspective of the player who made the move at the parent node.
            if node.parent is not None :
                perspective_player = node.parent.decision_player
            else : perspective_player = None # do not update win value of root node (only update visits)
            
            # The simulation_result is always from Player 1's perspective.
            # If the current node's move was made by Player -1, we flip the score.
            if perspective_player == 1 :
                result_for_node = simulation_result
            elif perspective_player == -1:
                result_for_node = -simulation_result
            else :
                result_for_node = 0
            
            node.update(result_for_node)
            node = node.parent


class MCTS:
    """
    Basic Monte Carlo Tree Search for Star Wars: Armada.

    """

    def __init__(self, initial_game: Armada, config : Config) -> None:
        self.game = initial_game
        self.snapshot = self.game.get_snapshot()
        self.player_root : dict[int, Node] = { 1 : Node(game = initial_game, action = ('initialize_game', None)),
                                              -1 : Node(game = initial_game, action = ('initialize_game', None))}
        
        self.config = config

    def mcts_search(self, *,simulation_player :int) -> None:

        # single decision optimization
        possible_actions = self.game.get_valid_actions()
        if len(possible_actions) == 1:
            action = possible_actions[0]
            self.game.apply_action(action)
            self.player_root[simulation_player].add_child(action, self.game)
            self.game.revert_snapshot(self.snapshot)
            return

        for iteration in range(self.config.ITERATION):
            
            node : Node = self.player_root[simulation_player]
            
            # 1. Selection
            while node.children or node.chance_node:

                if node.chance_node:
                    self.game.revert_snapshot(node.snapshot)

                    # For a chance node, sample a random outcome instead of using UCT.
                    if self.game.attack_info is None :
                        raise ValueError("Invalid game for chance node: missing attack/defend info.")
                    
                    dice_roll = dice.roll_dice(self.game.attack_info.dice_to_roll)
                    action = ("roll_dice_action", dice_roll)

                    dice_roll_result_node = next((child for child in node.children if child.action is not None and child.action[1] == dice_roll), None)
                    if dice_roll_result_node:
                        # If we've seen this random outcome before for this node, continue selection down that path.
                        node = dice_roll_result_node
                    else:
                        # If it's a new outcome, this is the node we will expand and simulate.
                        self.game.apply_action(action)
                        node = node.add_child(action, self.game)
                        
                        break # Exit selection loop

                else:
                    # Standard player decision node, use UCT.
                    node = node.select_child()
            
            leaf_snapshot = node.snapshot
            self.game.revert_snapshot(leaf_snapshot)
            
            # 2. Expansion (for player decision nodes)
            if not self.game.winner :
                actions = self.game.get_valid_actions()

                for action in actions:
                    self.game.apply_action(action)
                    node.add_child(action, self.game)
                    self.game.revert_snapshot(leaf_snapshot)

            # 3. Simulation
            simulation_result = self.game.rollout(max_simulation_step=1000)
            # with open('simulation_log.txt', 'a') as f: f.write(f"\nSimulation Result: {simulation_result}")
            self.game.revert_snapshot(self.snapshot)
            
            # 4. Backpropagation (Updated for -1 to 1 scoring)
            node.backpropagate(simulation_result)

            if (iteration+1) % (self.config.ITERATION//4) == 0:
                print(f"Iteration {iteration + 1}/{self.config.ITERATION}: Total Visits : {self.player_root[simulation_player].visits} Total Wins: {round(sum([child.wins for child in self.player_root[simulation_player].children]), 2)}, Best Action | {get_action_str(self.game, self.get_best_action(simulation_player))}")
                with open('simulation_log.txt', 'a') as f: f.write(f"\n{iteration+1} iteration. Total Visits : {self.player_root[simulation_player].visits} Total Win {round(sum([child.wins for child in self.player_root[simulation_player].children]), 2)}. Best Action {self.get_best_action(simulation_player)} \n{[(node.action, round(node.wins,2), node.visits) for node in self.player_root[simulation_player].children]}")




    def advance_tree(self, action: ActionType, snapshot : dict) -> None:
        """
        Advances the tree to the next state by selecting the child
        corresponding to the given action as the new root.
        """

        self.game.revert_snapshot(snapshot)
        self.snapshot = snapshot

        # Find the child node that matches the action taken.
        for player in (-1, 1) :

            matching_child = next((child for child in self.player_root[player].children if child.action == action), None)

            if matching_child is None :
                matching_child = self.player_root[player].add_child(action, self.game)

            # The found child becomes the new root.
            self.player_root[player] = matching_child
            self.player_root[player].parent = None # The new root has no parent.



    def get_best_action(self, decision_player : int) -> ActionType:
        root_node = self.player_root[decision_player]
        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action
    
    def get_random_best_action(self, decision_player : int) -> ActionType:
        root_node = self.player_root[decision_player]
        visit_weights = [c.visits for c in root_node.children]
        chosen_child = random.choices(population=root_node.children, weights=visit_weights, k=1)[0]
        return chosen_child.action