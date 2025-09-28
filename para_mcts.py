from __future__ import annotations
import math
import random
from typing import TYPE_CHECKING

import torch
import numpy as np
import torch.nn.functional as F

from action_space import ActionManager
from dummy_model import DummyModel
from armada_net import ArmadaNet
from game_encoder import encode_game_state
import dice
from game_phase import GamePhase, ActionType, get_action_str
from ship import _cached_range
if TYPE_CHECKING:
    from armada import Armada, AttackInfo
    from para_self_play import Config




class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    Can be a decision node (for a player) or a chance node (for random events).
    """
    def __init__(self, 
                 game : Armada,
                 action : ActionType.Action, 
                 config : Config,
                 parent : Node | None =None, 
                 policy : float = 0,
                 action_index : int = 0,
                 ) -> None :
        
        self.snapshot = game.get_snapshot()
        self.decision_player : int | None = game.decision_player # decision player used when get_possible_action is called on this node
        self.chance_node : bool = self.snapshot['phase'] == GamePhase.SHIP_ATTACK_ROLL_DICE
        self.information_set : bool = self.snapshot['phase'] == GamePhase.SHIP_REVEAL_COMMAND_DIAL
        self.config : Config = config


        self.parent : Node | None = parent
        self.action : ActionType.Action = action
        self.children : list[Node] = []
        self.wins : float = 0
        self.visits : int = 0
        
        self.policy : float = policy # policy value from parent node state
        self.action_index : int = action_index # index of the action in the full action list from parent node state


    def select_child(self, use_policy : bool) -> Node:
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
    
    def get_ucb(self, child : Node) -> float:
        if child.visits == 0:
            return float('inf')
        q_value : float = child.wins / child.visits
        return q_value + self.config.EXPLORATION_CONSTANT * math.sqrt(math.log(self.visits)) / child.visits

    def get_pucb(self, child : Node) -> float:
        if child.visits == 0:
            q_value : float = 0
        else: 
            q_value : float = child.wins / child.visits
        return q_value + self.config.EXPLORATION_CONSTANT * child.policy * math.sqrt(self.visits) / (1 + child.visits)


    def add_child(self, action : ActionType.Action, game : Armada, policy : float = 0, action_index : int = 0) -> Node :
        """
        Adds a new child node for the given action and game.
        Args:
            action : The action leading to the new child node.
            game : The game state after applying the action, used to determine the decision player for the child.
        """
        node = Node(game=game, parent=self, action=action, policy=policy, action_index=action_index, config=self.config)
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
        self.para_games : list[Armada] = games
        self.snapshots :list[dict] = [game.get_snapshot() for game in games]
        self.player_roots : list[dict[int, Node]] = [{ 1 : Node(game = game, action = ('initialize_game', None), config = config),
                                                      -1 : Node(game = game, action = ('initialize_game', None), config = config)} 
                                                      for game in games]
        self.action_manager : ActionManager = action_manager
        self.model : ArmadaNet = model
        self.config : Config = config

    def parallel_search(self, simulation_players : dict[int, int]) -> dict[int, np.ndarray]:
        parallel_indices = list(simulation_players.keys())
        for para_index in parallel_indices:
            sim_player = simulation_players[para_index]
            if self.player_roots[para_index][sim_player].chance_node or self.player_roots[para_index][sim_player].information_set:
                raise ValueError("Don't use MCTS for chance node and information set")
            

            self.para_games[para_index].simulation_player = sim_player


        # one game / two tree
        

        for _ in range(self.config.MCTS_ITERATION):
            para_nodes : dict[int, Node] = {para_index: self.player_roots[para_index][simulation_players[para_index]] for para_index in parallel_indices}
            expandable_node_indices : list[int] = []
            for para_index in parallel_indices:
                node = para_nodes[para_index]
            
                # 1. Selection
                while node.children or node.chance_node or node.information_set:

                    if node.chance_node:
                        self.para_games[para_index].revert_snapshot(node.snapshot)

                        # For a chance node, sample a random outcome instead of using UCT.
                       
                        if (attack_info := self.para_games[para_index].attack_info) is None :
                            raise ValueError("Invalid game for chance node: missing attack/defend info.")
                        dice_roll = dice.roll_dice(attack_info.dice_to_roll)
                        action = ("roll_dice_action", dice_roll)

                        matching_child = next((child for child in node.children if child.action[1] == dice_roll), None)
                        if matching_child : node = matching_child
                        else :
                            # dynamically expansion
                            self.para_games[para_index].apply_action(action)
                            node.add_child(action, self.para_games[para_index])

                    elif node.information_set :

                        if not node.children :
                            self.para_games[para_index].revert_snapshot(node.snapshot)
                            # expand all possible actions
                            for action in self.para_games[para_index].get_valid_actions() :
                                self.para_games[para_index].apply_action(action)
                                node.add_child(action, self.para_games[para_index])
                                self.para_games[para_index].revert_snapshot(node.snapshot)
                        
                        # don't use policy for secret information
                        # choose the best option using MCTS and UCB
                        node = node.select_child(use_policy=False)
                    
                    else:
                        # Standard player decision node, use pUCT.
                        node = node.select_child(use_policy=True)

                self.para_games[para_index].revert_snapshot(node.snapshot)
                para_nodes[para_index] = node

                
                # on terminal node, backpropagate the result
                if (value := self.para_games[para_index].winner) is not None : 
                    para_nodes[para_index].backpropagate(value)
                    self.para_games[para_index].revert_snapshot(self.snapshots[para_index])
                else :
                    expandable_node_indices.append(para_index)

            # 2. Expansion (for player decision nodes)
            # note that leaf node is not chance node or information set node
            if expandable_node_indices :

                values, policies = self.get_value_policy(
                    [encode_game_state(self.para_games[para_index]) for para_index in expandable_node_indices],
                    [self.para_games[para_index].phase for para_index in expandable_node_indices],
                )
                for output_index, para_index in enumerate(expandable_node_indices):
                    node = para_nodes[para_index]
                    value = float(values[output_index])
                    policy = policies[output_index]

                    action_map = self.action_manager.get_action_map(self.para_games[para_index].phase)
                    valid_actions: list[ActionType.Action] = self.para_games[para_index].get_valid_actions()
                    
                    # Create the mask with the same size as the policy array
                    valid_moves_mask = np.zeros_like(policy, dtype=np.uint8)

                    for action in valid_actions:
                        # get the action's index in the full action space
                        action_index = action_map[action]
                        valid_moves_mask[action_index] = 1
                    
                    policy *= valid_moves_mask

                    # Normalize the policy if there are any valid moves
                    policy_sum = np.sum(policy)
                    if policy_sum > 0:
                        policy /= policy_sum
                    else:
                        # Handle rare cases where the network assigns 0 probability to all valid moves
                        # Or if there are no valid moves (should not happen for an expandable node)
                        print(f"Warning: Zero policy sum for valid moves in phase {self.para_games[para_index].phase.name}. Using uniform distribution.")
                        valid_indices = [action_map.get(a) for a in valid_actions if action_map.get(a) is not None]
                        if valid_indices:
                            policy[valid_indices] = 1.0 / len(valid_indices)

                    # create child nodes for all valid actions
                    for action in valid_actions:
                        action_index = action_map[action]
                        action_policy: float = float(policy[action_index])

                        self.para_games[para_index].apply_action(action)
                        node.add_child(action, self.para_games[para_index], action_policy, action_index)
                        self.para_games[para_index].revert_snapshot(node.snapshot)

                    self.para_games[para_index].revert_snapshot(self.snapshots[para_index])


                    # 3. Backpropagation (Updated for -1 to 1 scoring)
                    node.backpropagate(value)
            
        # End of Search Iteration
        parallel_action_probs : dict[int, np.ndarray] = {}
        for para_index in parallel_indices:
            max_size = self.model.max_action_space
            action_probs = np.zeros(max_size, dtype=np.float32)
            sim_player = simulation_players[para_index]
            root_node = self.player_roots[para_index][sim_player]
            if not root_node.children:
                raise ValueError(f"No children found for root node after MCTS iterations.\n{self.para_games[para_index].get_snapshot()}\nsim_player: {sim_player}\npara_index: {para_index}")

            for child in root_node.children:
                action_probs[child.action_index] = child.visits
            action_probs /= np.sum(action_probs)
            
            parallel_action_probs[para_index] = action_probs
        return parallel_action_probs

    def get_value_policy(self, encoded_states: list[dict[str, np.ndarray]], phases: list[GamePhase]) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute value and policy for a BATCH of encoded states using a SINGLE model call.
        """
        if not encoded_states:
            return np.array([]), np.array([])

        # Build batched numpy arrays
        scalar_batch = np.stack([state['scalar'] for state in encoded_states])
        entity_batch = np.stack([state['entities'] for state in encoded_states])
        spatial_batch = np.stack([state['spatial'] for state in encoded_states])
        relation_batch = np.stack([state['relations'] for state in encoded_states])

        # Convert to PyTorch tensors
        scalar_tensor = torch.from_numpy(scalar_batch).float().to(self.config.DEVICE)
        entity_tensor = torch.from_numpy(entity_batch).float().to(self.config.DEVICE)
        spatial_tensor = torch.from_numpy(spatial_batch).float().to(self.config.DEVICE)
        relation_tensor = torch.from_numpy(relation_batch).float().to(self.config.DEVICE)

        self.model.eval()
        with torch.no_grad():
            # Perform a single, batched forward pass
            policy_logits, value_tensor = self.model(
                scalar_tensor,
                entity_tensor,
                spatial_tensor,
                relation_tensor,
                phases  # Pass the list of phases
            )

            # Apply softmax to get probabilities
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            
            # Squeeze to remove the last dimension (shape [B, 1] -> [B])
            values = value_tensor.squeeze(1).cpu().numpy()

        return values, policies
    
    def advance_tree(self, para_index : int, action: ActionType.Action, snapshot : dict) -> None:
        """
        Advances the tree to the next state by selecting the child
        corresponding to the given action as the new root.
        Args:
            para_index: Index of the parallel game/tree to advance.
            action: The action taken to reach the new state.
            snapshot: The game snapshot after applying the action.
        """

        self.para_games[para_index].revert_snapshot(snapshot)
        self.snapshots[para_index] = snapshot

        # Find the child node that matches the action taken.
        for player in (-1, 1) :

            matching_child = next((child for child in self.player_roots[para_index][player].children if child.action == action), None)

            if matching_child is None :
                matching_child = self.player_roots[para_index][player].add_child(action, self.para_games[para_index])

            # The found child becomes the new root.
            self.player_roots[para_index][player] = matching_child
            self.player_roots[para_index][player].parent = None # The new root has no parent.



    def get_best_action(self, para_index : int, decision_player : int) -> ActionType.Action:
        root_node = self.player_roots[para_index][decision_player]
        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action

    def get_random_best_action(self, para_index : int, decision_player : int) -> ActionType.Action:
        root_node = self.player_roots[para_index][decision_player]
        if not root_node.children:
            raise ValueError(f"No children found for root node during Advancing Tree\n{self.para_games[para_index].get_snapshot()}\npara_index: {para_index}")
        
        visit_weights = np.array([c.visits for c in root_node.children]) ** (1/self.config.TEMPERATURE)
        chosen_child = random.choices(population=root_node.children, weights=list(visit_weights), k=1)[0]
        return chosen_child.action