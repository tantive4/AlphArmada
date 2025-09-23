from __future__ import annotations
import math
import random
from typing import TYPE_CHECKING

import torch
import numpy as np

from action_space import ActionManager
from dummy_model import DummyModel
from armada_net import ArmadaNet
from game_encoder import encode_game_state
import dice
from game_phase import GamePhase, ActionType, get_action_str
from ship import _cached_range
if TYPE_CHECKING:
    from armada import Armada, AttackInfo
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
        self.information_set : bool = self.snapshot['phase'] == GamePhase.SHIP_REVEAL_COMMAND_DIAL


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

    def __init__(self, games: list[Armada], action_manager : ActionManager, model : ArmadaNet, config : Config) -> None:
        self.games : list[Armada] = games
        self.snapshots :list[dict] = [game.get_snapshot() for game in games]
        self.player_roots : list[dict[int, Node]] = [{ 1 : Node(game = game, action = ('initialize_game', None)),
                                                      -1 : Node(game = game, action = ('initialize_game', None))} 
                                                      for game in games]
        self.action_manager : ActionManager = action_manager
        self.model : ArmadaNet = model
        self.config : Config = config

    def parallel_search(self, parallel_indices : list[int], simulation_players : list[int]) -> dict[int, np.ndarray]:
        for parallel_index, sim_player in zip(parallel_indices, simulation_players):
            if self.player_roots[parallel_index][sim_player].chance_node or self.player_roots[parallel_index][sim_player].information_set:
                raise ValueError("Don't use MCTS for chance node and information set")
            

            self.games[parallel_index].simulation_player = sim_player


        # one game / two tree
        

        for iteration in range(self.config.MCTS_ITERATION):
            parallel_nodes : dict[int, Node] = {idx: self.player_roots[idx][sim_player] for idx, sim_player in zip(parallel_indices, simulation_players)}
            expandable_node_indices : list[int] = []
            for parallel_index in parallel_indices:
                node = parallel_nodes[parallel_index]
            
                # 1. Selection
                while node.children or node.chance_node or node.information_set:

                    if node.chance_node:
                        self.games[parallel_index].revert_snapshot(node.snapshot)

                        # For a chance node, sample a random outcome instead of using UCT.
                        if self.games[parallel_index].attack_info is None :
                            raise ValueError("Invalid game for chance node: missing attack/defend info.")
                        dice_roll = dice.roll_dice(self.games[parallel_index].attack_info.dice_to_roll)
                        action = ("roll_dice_action", dice_roll)

                        matching_child = next((child for child in node.children if child.action[1] == dice_roll), None)
                        if matching_child : node = matching_child
                        else :
                            # dynamically expansion
                            self.games[parallel_index].apply_action(action)
                            node.add_child(action, self.games[parallel_index])

                    elif node.information_set :

                        if not node.children :
                            self.games[parallel_index].revert_snapshot(node.snapshot)
                            # expand all possible actions
                            for action in self.games[parallel_index].get_valid_actions() :
                                self.games[parallel_index].apply_action(action)
                                node.add_child(action, self.games[parallel_index])
                                self.games[parallel_index].revert_snapshot(node.snapshot)
                        
                        # don't use policy for secret information
                        # choose the best option using MCTS and UCB
                        node = node.select_child(use_policy=False)
                    
                    else:
                        # Standard player decision node, use pUCT.
                        node = node.select_child(use_policy=True)

                self.games[parallel_index].revert_snapshot(node.snapshot)
                parallel_nodes[parallel_index] = node


                if self.games[parallel_index].winner is not None : 
                    value = self.games[parallel_index].winner
                    parallel_nodes[parallel_index].backpropagate(value)
                    self.games[parallel_index].revert_snapshot(self.snapshots[parallel_index])
                else :
                    expandable_node_indices.append(parallel_index)

            # 2. Expansion (for player decision nodes)
            # note that leaf node is not chance node or information set node
            if expandable_node_indices :


                values, policies = self.get_value_policy(
                    [encode_game_state(self.games[idx]) for idx in expandable_node_indices],
                    [self.games[idx].phase for idx in expandable_node_indices],
                )
                for output_index, parallel_index in zip(expandable_node_indices, parallel_indices):
                    node = parallel_nodes[parallel_index]
                    value = float(values[output_index])
                    policy = policies[output_index]

                    # create policy mask for valid actions
                    action_map = self.action_manager.get_action_map(self.games[parallel_index].phase)
                    action_to_index = action_map['action_to_index']
                    total_actions_list = action_map['total_actions']
                    valid_actions : list[ActionType.Action]= self.games[parallel_index].get_valid_actions()
                    valid_action_indices = {}

                    valid_moves_mask = np.zeros(len(total_actions_list), dtype=np.uint8)
                    for valid_action_index, action in enumerate(valid_actions):                      
                        action_index = action_to_index[action]
                        valid_action_indices[valid_action_index] = action_index
                        valid_moves_mask[action_index] = 1
                        
                    policy *= valid_moves_mask

                    policy /= np.sum(policy)  

                    # create child nodes for all valid actions
                    for valid_action_index, action in enumerate(valid_actions):
                        action_index : int = valid_action_indices[valid_action_index] # get the corresponding index in the full action list
                        action_policy : float = float(policy[action_index])

                        self.games[parallel_index].apply_action(action)
                        node.add_child(action, self.games[parallel_index], action_policy, action_index)
                        self.games[parallel_index].revert_snapshot(node.snapshot)

                    self.games[parallel_index].revert_snapshot(self.snapshots[parallel_index])


                    # 3. Backpropagation (Updated for -1 to 1 scoring)
                    node.backpropagate(value)
            
        # End of Search Iteration
        parallel_action_probs : dict[int, np.ndarray] = {}
        for simulation_player, parallel_index in zip(simulation_players, parallel_indices):
            action_map = self.action_manager.get_action_map(self.games[parallel_index].phase)
            action_probs = np.zeros(len(action_map['total_actions']), dtype=np.float16)
            for child in self.player_roots[parallel_index][simulation_player].children:
                action_probs[child.action_index] = child.visits
            action_probs /= np.sum(action_probs)
            
            parallel_action_probs[parallel_index] = action_probs
        return parallel_action_probs

    def get_value_policy(self, encoded_states: list[dict[str, np.ndarray]], phases : list[GamePhase]) -> tuple:
        """
        Compute value and policy for a batch of encoded states.

        Inputs
        - encoded_states: list of encoded dicts with keys ['scalar','entities','spatial','relations']
        - phase: GamePhase for these states (assumed identical across the batch)

        Outputs
        - values: np.ndarray of shape [B] with scalar value predictions
        - policies: np.ndarray of shape [B, A] with policy probabilities for the given phase

        Notes
        - Uses np.stack to build batched tensors, but performs per-sample forward
          since ArmadaNet.forward currently expects single-sample inputs.
        """
        es_list = encoded_states

        # Build batched numpy arrays using np.stack
        scalar_batch = np.stack([es['scalar'] for es in es_list], axis=0)           # [B, S]
        entity_batch = np.stack([es['entities'] for es in es_list], axis=0)         # [B, MAX_SHIPS, F]
        spatial_batch = np.stack([es['spatial'] for es in es_list], axis=0)         # [B, C, H, W]
        relation_batch = np.stack([es['relations'] for es in es_list], axis=0)      # [B, R, R] or [B, 24, 24]

        # Convert numpy arrays to PyTorch tensors for the model
        scalar_tensor = torch.from_numpy(scalar_batch).float().to(self.config.DEVICE)
        entity_tensor = torch.from_numpy(entity_batch).float().to(self.config.DEVICE)
        spatial_tensor = torch.from_numpy(spatial_batch).float().to(self.config.DEVICE)
        relation_tensor = torch.from_numpy(relation_batch).float().to(self.config.DEVICE)

        self.model.eval()
        with torch.no_grad():
            # Per-sample inference with batched pre-processing
            values_list: list[float] = []
            policies_list: list[np.ndarray] = []
            B = scalar_tensor.shape[0]
            for i in range(B):
                policy_logits, value_tensor = self.model(
                    scalar_tensor[i],
                    entity_tensor[i],
                    spatial_tensor[i],
                    relation_tensor[i],
                    phases[i]
                )
                values_list.append(float(value_tensor.item()))
                policies_list.append(policy_logits.detach().cpu().numpy())

            values = np.asarray(values_list, dtype=np.float32)                  # [B]
            policies = np.stack(policies_list, axis=0)                          # [B, A]
            return values, policies

    def advance_tree(self, action: ActionType.Action, snapshot : dict) -> None:
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



    def get_best_action(self, decision_player : int) -> ActionType.Action:
        root_node = self.player_root[decision_player]
        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action
    
    def get_random_best_action(self, decision_player : int) -> ActionType.Action:
        root_node = self.player_root[decision_player]
        visit_weights = [c.visits for c in root_node.children]
        chosen_child = random.choices(population=root_node.children, weights=visit_weights, k=1)[0]
        return chosen_child.action