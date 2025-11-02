from __future__ import annotations
import random
from collections import deque

import torch
import numpy as np
import torch.nn.functional as F
cimport numpy as np

from libc.math cimport log, sqrt
from libc.stdlib cimport rand, RAND_MAX

from action_space import ActionManager
from armada_net import ArmadaNet
from game_encoder import encode_game_state
import dice
from action_phase import Phase, ActionType
from armada cimport Armada
from attack_info cimport AttackInfo
from self_play import Config


cdef class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    Can be a decision node (for a player) or a chance node (for random events).
    """
    cdef :
        public tuple snapshot
        public int decision_player
        public bint chance_node
        public bint information_set

        public tuple action
        public list children
        public float wins
        public int visits
    
        public float policy
        public float value
        public int action_index

    def __init__(self, 
                 Armada game,
                 action : ActionType, 
                 policy : float = 0,
                 value : float = 0,
                 action_index : int = 0,
                 ) -> None :
        
        self.snapshot = game.get_snapshot()
        self.decision_player  = game.decision_player # decision player used when get_possible_action is called on this node
        self.chance_node = <bint>(game.phase == Phase.ATTACK_ROLL_DICE)
        self.information_set = <bint>(game.phase == Phase.SHIP_REVEAL_COMMAND_DIAL)

        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0

        self.policy = policy # policy value from parent node state
        self.value = value # value from the perspective of the decision player at this node
        self.action_index = action_index # index of the action in the full action list from parent node state

    cpdef Node add_child(self, tuple action, Armada game, float policy=0.0, float value=0.0, int action_index=0) :
        """
        Adds a new child node for the given action and game.
        Args:
            action : The action leading to the new child node.
            game : The game state after applying the action, used to determine the decision player for the child.
        """
        cdef Node node = Node(game=game, action=action, policy=policy, value=value, action_index=action_index)
        self.children.append(node)
        return node

    cdef Node select_child(self):
        """
        Selects a child node using the UCT formula with random tie-breaking.
        """
        cdef:
            Node best_child
            Node child
            float best_ucb = -float('inf')
            float ucb

        if self.visits == 0:
            return random.choice(self.children)

        for child in self.children:
            # Don't use policy for secret information
            if self.information_set: ucb = self._get_ucb(child)
            # Use policy for regular decision nodes
            else: ucb = self._get_pucb(child)

            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child
    
    cdef void update(self, float result) :
        """
        Updates the node's statistics from a simulation result.
        """
        self.visits += 1
        self.wins += result

    cdef void reset_node(self):
        """
        Resets the node's statistics and children.
        """
        cdef Node child
        
        self.wins = 0
        self.visits = 0
        for child in self.children:
            child.reset_node()


    cdef float _get_ucb(self, Node child):
        cdef float q_value
        if child.visits == 0:
            return float('inf')
        q_value = child.wins / child.visits
        return q_value + Config.EXPLORATION_CONSTANT * <float>sqrt(log(<double>self.visits)) / child.visits

    cdef float _get_pucb(self, Node child):
        cdef float q_value = child.wins / child.visits if child.visits else 0.0
        return q_value + Config.EXPLORATION_CONSTANT * child.policy * <float>sqrt(<double>self.visits) / (1 + child.visits)


cdef class MCTS:
    """
    Parallel Monte Carlo Tree Search for multiple Armada games.
    """
    cdef :
        public list para_games, root_snapshots, player_roots
        public object action_manager
        public object model

    def __init__(self, list games, object action_manager, object model) -> None:
        self.para_games = games
        self.root_snapshots = [(<Armada>game).get_snapshot() for game in games]
        self.player_roots = [{ 1 : Node(game = game, action = ('initialize_game', None)),
                               -1 : Node(game = game, action = ('initialize_game', None))} 
                              for game in games]
        self.action_manager : ActionManager = action_manager
        self.model : ArmadaNet = model

    cpdef dict para_search(self, dict sim_players, bint deep_search):
        cdef:
            Armada game
            list para_indices, expandable_node_indices, valid_actions
            int para_index, sim_player, mcts_iteration, output_index, max_size
            Node root_node, node
            dict action_map, para_path, para_action_probs
            object path
            float value, winner
            np.ndarray policy_arr

        para_indices = list(sim_players.keys())
        for para_index in para_indices:
            sim_player = sim_players[para_index]
            root_node = self.player_roots[para_index][sim_player]
            if root_node.chance_node or root_node.information_set:
                raise ValueError("Don't use MCTS for chance node and information set")
            game = self.para_games[para_index]
            game.simulation_player = sim_player


        # one game / two tree
        

        mcts_iteration = Config.MCTS_ITERATION if deep_search else Config.MCTS_ITERATION_FAST
        for _ in range(mcts_iteration):
            para_path = {}
            expandable_node_indices = []
            for para_index in para_indices:
                node = self.player_roots[para_index][sim_players[para_index]]
            
                # 1. Selection
                path : deque[Node] = self._select(node, para_index)
                para_path[para_index] = path
                node = path[-1]
                game = self.para_games[para_index]
                game.revert_snapshot(node.snapshot)

                # on terminal node, backpropagate the result
                # also, if node has been expanded before, backpropagate the value
                if (winner := game.winner) != 0.0 or node.children:
                    value = winner if winner != 0.0 else node.value
                    self._backpropagate(path, value)
                    game.revert_snapshot(self.root_snapshots[para_index])
                else :
                    expandable_node_indices.append(para_index)

            # 2. Expansion (for player decision nodes)
            # note that leaf node is not chance node or information set node
            if expandable_node_indices :

                values, policies = self._get_value_policy(
                    [encode_game_state(self.para_games[para_index]) for para_index in expandable_node_indices],
                    [self.para_games[para_index].phase for para_index in expandable_node_indices],
                )



                for output_index, para_index in enumerate(expandable_node_indices):
                    path = para_path[para_index]
                    node = path[-1]
                    value = <float>(values[output_index])
                    policy_arr = policies[output_index]
                    game = self.para_games[para_index]

                    # Add Dirichlet noise for exploration at the root node only
                    if len(path) == 1:
                        policy_arr = (1-Config.DIRICHLET_EPSILON) * policy_arr + \
                                     Config.DIRICHLET_EPSILON * np.random.dirichlet(np.full(len(policy_arr), Config.DIRICHLET_ALPHA))

                    action_map = self.action_manager.get_action_map(game.phase)
                    valid_actions: list[ActionType] = game.get_valid_actions()
                    
                    policy_arr = self._mask_policy(policy_arr, valid_actions, action_map)

                    self._expand(node, para_index, valid_actions, action_map, value, policy_arr)

                    # 3. Backpropagation (Updated for -1 to 1 scoring)
                    self._backpropagate(path, value)

        # End of Search Iteration
        para_action_probs = {}
        max_size = self.model.max_action_space
        for para_index in para_indices:
            sim_player = sim_players[para_index]
            para_action_probs[para_index] = self._get_final_action_probs(para_index, sim_player, max_size)
        return para_action_probs

    cpdef object get_best_action(self, int para_index, int decision_player):
        cdef Node root_node, best_child, child
        cdef int max_visits = -1

        root_node = self.player_roots[para_index][decision_player]
        for child in root_node.children:
            if child.visits > max_visits:
                max_visits = child.visits
                best_child = child
                
        return best_child.action

    cpdef object get_random_best_action(self, int para_index, int decision_player, int game_round):
        cdef:
            Node root_node, child, chosen_child
            float temperature
            list children
            int num_children, i
            np.ndarray visit_counts, visit_weights 

        root_node = self.player_roots[para_index][decision_player]
        children = root_node.children
        num_children = len(children)
        visit_counts = np.empty(num_children, dtype=np.float32)
        for i in range(num_children):
            child = children[i]
            visit_counts[i] = child.visits

        temperature = Config.TEMPERATURE / <float>game_round
        # apply the temperature weighting
        visit_weights = visit_counts ** (1.0 / temperature)
        
        chosen_child = random.choices(population=children, weights=list(visit_weights), k=1)[0]
        return chosen_child.action

    cpdef void advance_tree(self, int para_index, tuple action, tuple snapshot):
        """
        Advances the tree to the next state by selecting the child
        corresponding to the given action as the new root.
        Args:
            para_index: Index of the parallel game/tree to advance.
            action: The action taken to reach the new state.
            snapshot: The game snapshot after applying the action.
        """
        cdef:
            int player
            Node root, matching_child
            bint found

        (<Armada>self.para_games[para_index]).revert_snapshot(snapshot)
        self.root_snapshots[para_index] = snapshot
        
        # Find the child node that matches the action taken.
        for player in (-1, 1):
            root = self.player_roots[para_index][player]
            found = <bint>False
            for child in root.children:
                if child.action == action:
                    matching_child = child
                    found = <bint>True
                    break
            if not found:
                matching_child = root.add_child(action, self.para_games[para_index])

            # The found child becomes the new root.
            matching_child.reset_node()
            self.player_roots[para_index][player] = matching_child

    cdef object _select(self, Node root_node, int para_index):
        """
        Select down the tree from a root node to a leaf node.
        """
        cdef:
            Node node = root_node
            object path = deque([node])
            Armada game 
            AttackInfo attack_info
            Node matching_child, child
            tuple action, dice_roll
            bint found
        
        game = self.para_games[para_index]
        while (node.visits > 0 and node.children) or node.chance_node or node.information_set:

            if node.chance_node:
                game.revert_snapshot(node.snapshot)

                # For a chance node, sample a random outcome instead of using UCT.
            
                if (attack_info := game.attack_info) is None :
                    raise ValueError("Invalid game for chance node: missing attack/defend info.")
                dice_roll = dice.roll_dice(attack_info.dice_to_roll)
                action = ("roll_dice_action", dice_roll)
                
                found = <bint>False
                for child in node.children:
                    if child.action[1] == dice_roll:
                        matching_child = child
                        found = <bint>True
                        break
                if found:
                    node = matching_child
                else:
                    # dynamically expansion
                    game.apply_action(action)
                    node = node.add_child(action, game)

            elif node.information_set :

                if not node.children :
                    game.revert_snapshot(node.snapshot)
                    # expand all possible actions
                    for action in game.get_valid_actions() :
                        game.apply_action(action)
                        node.add_child(action, game)
                        game.revert_snapshot(node.snapshot)
                node = node.select_child()
            
            else:
                node = node.select_child()
            path.append(node)
        return path

    cdef object _get_value_policy(self, encoded_states: list, phases: list):
        """
        Compute value and policy for a BATCH of encoded states using a SINGLE model call.
        """
        if not encoded_states:
            return np.array([]), np.array([])

        # Build batched numpy arrays
        scalar_batch = np.stack([state['scalar'] for state in encoded_states])
        ship_entity_batch = np.stack([state['ship_entities'] for state in encoded_states])
        squad_entity_batch = np.stack([state['squad_entities'] for state in encoded_states])
        spatial_batch = np.stack([state['spatial'] for state in encoded_states])
        relation_batch = np.stack([state['relations'] for state in encoded_states])

        # Convert to PyTorch tensors
        scalar_tensor = torch.from_numpy(scalar_batch).float().to(Config.DEVICE)
        ship_entity_tensor = torch.from_numpy(ship_entity_batch).float().to(Config.DEVICE)
        squad_entity_tensor = torch.from_numpy(squad_entity_batch).float().to(Config.DEVICE)
        spatial_tensor = torch.from_numpy(spatial_batch).float().to(Config.DEVICE)
        relation_tensor = torch.from_numpy(relation_batch).float().to(Config.DEVICE)

        with torch.no_grad():
            # Perform a single, batched forward pass
            outputs = self.model(
                scalar_tensor,
                ship_entity_tensor,
                squad_entity_tensor,
                spatial_tensor,
                relation_tensor,
                phases
            )

            policy_logits = outputs['policy_logits']
            value_tensor = outputs['value']
            # Apply softmax to get probabilities
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()

            # Squeeze to remove the last dimension (shape [B, 1] -> [B])
            values = value_tensor.squeeze(1).cpu().numpy()

        return values, policies
  
    cdef np.ndarray _mask_policy(self, np.ndarray policy, list valid_actions, dict action_map):
        cdef: 
            np.ndarray valid_moves_mask = np.zeros_like(policy, dtype=np.bool_)
            object action
            int action_index
            float policy_sum

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
            print(f"Warning: Zero policy sum for valid moves in {valid_actions}. Using uniform distribution.")
            policy = np.ones_like(policy, dtype=np.float32) / np.sum(valid_moves_mask)

        return policy

    cdef void _expand(self, Node node, int para_index, list valid_actions, dict action_map, float value, np.ndarray policy) :
        cdef: 
            tuple action
            int action_index
            float action_policy
            Armada game
            tuple node_snapshot = node.snapshot

        game = self.para_games[para_index]

        for action in valid_actions:
            action_index = action_map[action]
            action_policy = <float>(policy[action_index])
            node_value = value

            game.apply_action(action)
            node.add_child(action, game, policy=action_policy, value=node_value, action_index=action_index)
            game.revert_snapshot(node_snapshot)
        game.revert_snapshot(self.root_snapshots[para_index])

    cdef void _backpropagate(self, object path, float value):
        """
        Backpropagates the simulation result up the tree.
        """
        cdef Node node
        cdef int perspective_player
        cdef float result_for_node

        while path:
            node = path.pop()
            # The result must be from the perspective of the player who made the move at the parent node.
            if path :
                perspective_player = (<Node>path[-1]).decision_player
            else: perspective_player = 0  # do not update win value of root node (only update visits)

            # The simulation_result is always from Player 1's perspective.
            # If the current node's move was made by Player -1, we flip the score.
            result_for_node = value * perspective_player
            
            node.update(result_for_node)

    cdef np.ndarray _get_final_action_probs(self, int para_index, int sim_player, int max_size):
        cdef:
            np.ndarray action_probs = np.zeros(max_size, dtype=np.float32)
            Node root_node = self.player_roots[para_index][sim_player]
            Node child
            float total_visits

        for child in root_node.children:
            action_probs[child.action_index] = child.visits

        total_visits = np.sum(action_probs, dtype=float)
        action_probs /= total_visits

        return action_probs


