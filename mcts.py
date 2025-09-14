from __future__ import annotations
import math
import random
from typing import TYPE_CHECKING

import torch
import numpy as np

from action_space import ActionManager, _make_hashable
from dummy_model import DummyModel
from armada_net import ArmadaNet
import dice
from game_phase import GamePhase, ActionType
from ship import _cached_range
if TYPE_CHECKING:
    from armada import Armada
    from self_play import Config




class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    Can be a decision node (for a player) or a chance node (for random events).
    """
    def __init__(self, 
                 game : Armada,
                 action : ActionType.Action, 
                 parent : Node | None =None, 
                 policy : float = 0,
                 action_index : int = 0,) -> None :
        
        self.snapshot = game.get_snapshot()
        self.decision_player : int | None = game.decision_player # decision player used when get_possible_action is called on this node
        self.chance_node : bool = self.snapshot['phase'] == GamePhase.SHIP_ATTACK_ROLL_DICE


        self.parent : Node | None = parent
        self.action : ActionType.Action = action
        self.children : list[Node] = []
        self.wins : float = 0
        self.visits : int = 0
        
        self.policy : float = policy # policy value from parent node state
        self.action_index : int = action_index # index of the action in the full action list from parent node state


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
        

    def add_child(self, action : ActionType.Action, game : Armada, policy : float = 0, action_index : int = 0) -> Node :
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

    def __init__(self, initial_game: Armada, action_manager : ActionManager, model, config : Config) -> None:
        self.root_game = initial_game
        self.snapshot = self.root_game.get_snapshot()
        self.root : Node = Node(game = initial_game, action = ('initialize_game', None))

        self.action_manager = action_manager
        self.model = model
        self.config = config

    def mcts_search(self, iterations: int) -> None:

        # single decision optimization
        possible_actions = self.root_game.get_valid_actions()
        if len(possible_actions) == 1:
            action = possible_actions[0]
            self.root_game.apply_action(action)
            self.root.add_child(action, self.root_game)
            self.root_game.revert_snapshot(self.snapshot)
            return

        for iteration in range(iterations):
            
            node : Node = self.root
            
            # 1. Selection
            while node.children or node.chance_node:

                if node.chance_node:
                    self.root_game.revert_snapshot(node.snapshot)

                    # For a chance node, sample a random outcome instead of using UCT.
                    if self.root_game.attack_info is None :
                        raise ValueError("Invalid game for chance node: missing attack/defend info.")
                    
                    dice_roll = dice.roll_dice(self.root_game.attack_info.dice_to_roll)
                    action = ("roll_dice_action", dice_roll)

                    dice_roll_result_node = next((child for child in node.children if child.action is not None and child.action[1] == dice_roll), None)
                    if dice_roll_result_node:
                        # If we've seen this random outcome before for this node, continue selection down that path.
                        node = dice_roll_result_node
                    else:
                        # If it's a new outcome, this is the node we will expand and simulate.
                        self.root_game.apply_action(action)
                        node = node.add_child(action, self.root_game)
                        
                        break # Exit selection loop

                else:
                    # Standard player decision node, use UCT.
                    node = node.select_child()
            
            leaf_snapshot = node.snapshot
            self.root_game.revert_snapshot(leaf_snapshot)
            
            # 2. Expansion (for player decision nodes)
            if not self.root_game.winner :
                actions = self.root_game.get_valid_actions()

                for action in actions:
                    self.root_game.apply_action(action)
                    node.add_child(action, self.root_game)
                    self.root_game.revert_snapshot(leaf_snapshot)

            # 3. Simulation
            simulation_result = self.root_game.play(max_simulation_step=1000)
            # with open('simulation_log.txt', 'a') as f: f.write(f"\nSimulation Result: {simulation_result}")
            self.root_game.revert_snapshot(self.snapshot)
            
            # 4. Backpropagation (Updated for -1 to 1 scoring)
            node.backpropagate(simulation_result)
            
            if (iteration+1) % (iterations//4) == 0:
                print(f"Iteration {iteration + 1}/{iterations}: Total Visits : {self.root.visits} Total Wins: {round(sum([child.wins for child in self.root.children]), 2)}, Best Action | {ActionType.get_action_str(self.root_game, self.get_best_action())}")
                with open('simulation_log.txt', 'a') as f: f.write(f"\n{iteration+1} iteration. Total Visits : {self.root.visits} Total Win {round(sum([child.wins for child in self.root.children]), 2)}. Best Action {self.get_best_action()} \n{[(node.action, round(node.wins,2), node.visits) for node in self.root.children]}")
        print(f'_RANGE CACHE INFO : {_cached_range.cache_info()}')



    def alpha_mcts_search(self) -> np.ndarray :
        
        for iteration in range(self.config.MCTS_ITERATION):
            
            node : Node = self.root
            
            # 1. Selection
            while node.children or node.chance_node:

                if node.chance_node:
                    self.root_game.revert_snapshot(node.snapshot)

                    # For a chance node, sample a random outcome instead of using UCT.
                    if self.root_game.attack_info is None :
                        raise ValueError("Invalid game for chance node: missing attack/defend info.")
                    
                    dice_roll = dice.roll_dice(self.root_game.attack_info.dice_to_roll)
                    action = ("roll_dice_action", dice_roll)

                    dice_roll_result_node = next((child for child in node.children if child.action is not None and child.action[1] == dice_roll), None)
                    if dice_roll_result_node:
                        # If we've seen this random outcome before for this node, continue selection down that path.
                        node = dice_roll_result_node
                    else:
                        # If it's a new outcome, this is the node we will expand and simulate.
                        self.root_game.apply_action(action)
                        node = node.add_child(action, self.root_game)
                        break # Exit selection loop

                else:
                    # Standard player decision node, use pUCT.
                    node = node.select_child(use_policy=True)

            leaf_snapshot = node.snapshot
            self.root_game.revert_snapshot(leaf_snapshot)

            encoded_state = self.root_game.get_encoded_state()
            
            # Convert numpy arrays to PyTorch tensors for the model
            scalar_tensor = torch.from_numpy(encoded_state['scalar']).float().to(self.config.DEVICE)
            entity_tensor = torch.from_numpy(encoded_state['entities']).float().to(self.config.DEVICE)
            spatial_tensor = torch.from_numpy(encoded_state['spatial']).float().to(self.config.DEVICE)
            relation_tensor = torch.from_numpy(encoded_state['relations']).float().to(self.config.DEVICE)
            
            self.model.eval()
            with torch.no_grad():
                policy_logits, value_tensor = self.model(
                    scalar_tensor, 
                    entity_tensor, 
                    spatial_tensor, 
                    relation_tensor, 
                    self.root_game.phase
                )
            value = value_tensor.item()
            policy = policy_logits.cpu().numpy()


            if self.root_game.winner is not None : value = self.root_game.winner
            else :

            # 2. Expansion (for player decision nodes)
                # create policy mask for valid actions
                action_map = self.action_manager.get_action_map(self.root_game.phase)
                action_to_index = action_map['action_to_index']
                total_actions_list = action_map['total_actions']
                valid_actions = self.root_game.get_valid_actions()
                valid_action_index = {}

                valid_moves_mask = np.zeros(len(total_actions_list), dtype=np.uint8)
                for i, action in enumerate(valid_actions):
                    # Ensure the action is hashable to be used as a dictionary key
                    hashable_action = _make_hashable(action[1])
                    action_key = (action[0], hashable_action)
                    
                    action_index = action_to_index[action_key]
                    valid_action_index[i] = action_index
                    valid_moves_mask[action_index] = 1
                    
                policy *= valid_moves_mask
                policy_sum = np.sum(policy)
                if policy_sum > 0:
                    policy /= policy_sum
                else:
                    # zero division fallback
                    policy = valid_moves_mask / np.sum(valid_moves_mask)

                # create child nodes for all valid actions
                for i, action in enumerate(valid_actions):
                    action_index : int = valid_action_index[i] # get the corresponding index in the full action list
                    action_policy : float = float(policy[action_index])

                    self.root_game.apply_action(action)
                    node.add_child(action, self.root_game, action_policy, action_index)
                    self.root_game.revert_snapshot(leaf_snapshot)

            self.root_game.revert_snapshot(self.snapshot)
            

            # 3. Backpropagation (Updated for -1 to 1 scoring)
            node.backpropagate(value)
            
            # if (iteration+1) % (iterations//4) == 0:
            #     print(f"Iteration {iteration + 1}/{iterations}: Total Visits : {self.root.visits} Total Wins: {round(sum([child.wins for child in self.root.children]), 2)}, Best Action | {ActionType.get_action_str(self.root_game, self.get_best_action())}")
            #     with open('simulation_log.txt', 'a') as f: f.write(f"\n{iteration+1} iteration. Total Visits : {self.root.visits} Total Win {round(sum([child.wins for child in self.root.children]), 2)}. Best Action {self.get_best_action()} \n{[(node.action, round(node.wins,2), node.visits) for node in self.root.children]}")
        
        # End of Search Iteration
        action_map = self.action_manager.get_action_map(self.root_game.phase)
        action_probs = np.zeros(len(action_map['total_actions']), dtype=np.float16)
        for child in self.root.children:
            action_probs[child.action_index] = child.visits

        action_probs /= np.sum(action_probs)
        return action_probs

    def advance_tree(self, action: ActionType.Action) -> None:
        """
        Advances the tree to the next state by selecting the child
        corresponding to the given action as the new root.
        """
        # Find the child node that matches the action taken.
        # If no child matches, a new tree must be started.
        self.root_game.apply_action(action)
        matching_child = next((child for child in self.root.children if child.action == action), None)
        if not matching_child :
            matching_child = self.root.add_child(action, self.root_game)

        
        # The found child becomes the new root.
        self.root = matching_child
        self.root.parent = None # The new root has no parent.

        self.snapshot = self.root_game.get_snapshot()

    def get_best_action(self) -> ActionType.Action:
        if not self.root.children:
            possible_actions = self.root_game.get_valid_actions()
            random.shuffle(possible_actions)
            return possible_actions[0]
            
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.action
    
    def get_random_best_action(self) -> ActionType.Action:
        visit_weights = [c.visits for c in self.root.children]
        chosen_child = random.choices(population=self.root.children, weights=visit_weights, k=1)[0]
        return chosen_child.action