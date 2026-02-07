# cython: profile=True

from __future__ import annotations
import random
from collections import deque

import torch
import numpy as np
import torch.nn.functional as F
cimport numpy as cnp

from libc.math cimport log, sqrt

from action_manager cimport ActionManager
from big_deep import BigDeep
from game_encoder cimport encode_game_state
import dice
from action_phase import Phase, ActionType
from armada cimport Armada
from attack_info cimport AttackInfo
from configs import Config

cdef int max_ships = Config.MAX_SHIPS

cdef class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    Can be a decision node (for a player) or a chance node (for random events).
    """
    cdef :
        public tuple snapshot
        public int first_player_perspective
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

        self.first_player_perspective = game.decision_player # decision player used when get_possible_action is called on this node

        self.chance_node = <bint>(game.phase == Phase.ATTACK_ROLL_DICE)
        # simplified
        # self.information_set = <bint>(game.phase == Phase.SHIP_REVEAL_COMMAND_DIAL)
        self.information_set = False

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

    cdef void _add_root_noise(self):
        if not self.children:
            return
        
        cdef:
            int num_children = len(self.children)
            cnp.ndarray[cnp.float32_t, ndim=1] noise
            int i
            Node child
            float epsilon = Config.DIRICHLET_EPSILON
            float alpha = Config.DIRICHLET_ALPHA_SCALE / <float>num_children

        # Generate Dirichlet noise
        noise = np.random.dirichlet(np.full(num_children, alpha)).astype(np.float32)

        # Mix noise into the existing children's policies
        for i in range(num_children):
            child = self.children[i]
            # Note: This permanently alters the prior for this node, 
            child.policy = (1 - epsilon) * child.policy + epsilon * noise[i]


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
        public list para_games, root_snapshots, root_nodes
        public ActionManager action_manager
        public object model
        public object action_mask
        cnp.ndarray scalar_buffer, ship_entity_buffer, ship_coords_buffer, ship_def_token_buffer, spatial_buffer, relation_buffer, active_ship_indices_buffer, target_ship_indices_buffer

    def __init__(self, list para_games, ActionManager action_manager, object model) -> None:
        self.para_games = para_games
        self.root_snapshots = [(<Armada>game).get_snapshot() for game in para_games]
        self.root_nodes = [Node(game = game, action = ('initialize_game', None))
                              for game in para_games]
        
        self.model : BigDeep = model

        self.action_manager = action_manager
        self.action_mask = np.zeros(action_manager.max_action_space, dtype=np.bool_)
        
        self.scalar_buffer = np.zeros(
            (Config.GPU_INPUT_BATCH_SIZE, Config.SCALAR_FEATURE_SIZE), 
            dtype=np.float32
        )
        self.ship_entity_buffer = np.zeros(
            (Config.GPU_INPUT_BATCH_SIZE, Config.MAX_SHIPS, Config.SHIP_ENTITY_FEATURE_SIZE), 
            dtype=np.float32
        )
        self.ship_coords_buffer = np.zeros(
            (Config.GPU_INPUT_BATCH_SIZE, Config.MAX_SHIPS, 3), 
            dtype=np.float32
        )
        self.ship_def_token_buffer = np.zeros(
            (Config.GPU_INPUT_BATCH_SIZE, Config.MAX_SHIPS, Config.MAX_DEFENSE_TOKENS, Config.DEF_TOKEN_FEATURE_SIZE),
            dtype=np.float32
        )
        # Spatial buffer is the largest (ensure dims match BigDeep: [B, N, 10, H, W/8])
        self.spatial_buffer = np.zeros(
            (Config.GPU_INPUT_BATCH_SIZE, Config.MAX_SHIPS, 10, Config.BOARD_RESOLUTION[0], Config.BOARD_RESOLUTION[1]//8), 
            dtype=np.uint8
        )
        self.relation_buffer = np.zeros(
            (Config.GPU_INPUT_BATCH_SIZE, Config.MAX_SHIPS, Config.MAX_SHIPS, 20), 
            dtype=np.float32
        )
        self.active_ship_indices_buffer = np.full(Config.GPU_INPUT_BATCH_SIZE, Config.MAX_SHIPS, dtype=np.int8)
        self.target_ship_indices_buffer = np.full(Config.GPU_INPUT_BATCH_SIZE, Config.MAX_SHIPS, dtype=np.int8)

    cpdef dict para_search(self, list para_indices, bint deep_search, int manual_iteration = 0):
        cdef:
            Armada game
            list leaf_root_indices, expandable_indices, valid_actions
            int para_index, sim_player, mcts_iteration, output_index, max_size, action_exp_len
            Node root_node, node
            dict para_path, para_action_probs
            object path
            cnp.ndarray[cnp.float32_t, ndim=1] batch_value
            cnp.ndarray[cnp.float32_t, ndim=2] batch_policy
            float value, winner
            cnp.ndarray[cnp.float32_t, ndim=1] policy_arr

        # ===== Initialize Root Nodes and Add Noise =====
        
        leaf_root_indices = []
        for para_index in para_indices:
            root_node = self.root_nodes[para_index]
            game = self.para_games[para_index]
            game.revert_snapshot(root_node.snapshot)

            if root_node.chance_node or root_node.information_set:
                raise ValueError("Don't use MCTS for chance node and information set")
            if not root_node.children : 
                leaf_root_indices.append(para_index)
        
        # Expansion
        if leaf_root_indices:
            batch_value, batch_policy = self._get_value_policy(
                [encode_game_state(self.para_games[para_index]) for para_index in leaf_root_indices],
                [self.para_games[para_index].phase for para_index in leaf_root_indices],
            )

            for output_index, para_index in enumerate(leaf_root_indices):
                root_node = self.root_nodes[para_index]
                game = self.para_games[para_index]
                
                value = <float>(batch_value[output_index])
                policy_arr = batch_policy[output_index]

                valid_actions = game.get_valid_actions()

                policy_arr = self._mask_policy(policy_arr, <int>game.phase, valid_actions)
                self._expand(root_node, para_index, <int>game.phase, valid_actions, value, policy_arr)

        for para_index in para_indices:
            root_node = self.root_nodes[para_index]
            root_node._add_root_noise()
        

        # ===== MCTS Iterations =====

        mcts_iteration = Config.MCTS_ITERATION if deep_search else Config.MCTS_ITERATION_FAST
        if manual_iteration : mcts_iteration = manual_iteration
        for _ in range(mcts_iteration):
            para_path = {}
            expandable_indices = []
            for para_index in para_indices:
                root_node = self.root_nodes[para_index]
            
                # 1. Selection
                path : deque[Node] = self._select(root_node, para_index)
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
                    expandable_indices.append(para_index)

            # 2. Expansion (for player decision nodes)
            # note that leaf node is not chance node or information set node
            if expandable_indices:
                batch_value, batch_policy = self._get_value_policy(
                    [encode_game_state(self.para_games[para_index]) for para_index in expandable_indices],
                    [self.para_games[para_index].phase for para_index in expandable_indices],
                )

                for output_index, para_index in enumerate(expandable_indices):
                    path = para_path[para_index]
                    node = path[-1]
                    game = self.para_games[para_index]

                    value = <float>(batch_value[output_index])
                    policy_arr = batch_policy[output_index]
                    
                    valid_actions: list[ActionType] = game.get_valid_actions()
                    policy_arr = self._mask_policy(policy_arr, <int>game.phase, valid_actions)

                    self._expand(node, para_index, <int>game.phase, valid_actions, value, policy_arr)

                    # 3. Backpropagation (Updated for -1 to 1 scoring)
                    self._backpropagate(path, value)

        # ===== End of Search Iteration =====
        para_action_probs = {}
        for para_index in para_indices:
            para_action_probs[para_index] = self._get_final_action_probs(para_index)
        return para_action_probs

    cpdef object get_best_action(self, int para_index):
        cdef Node root_node, best_child, child
        cdef int max_visits = -1

        root_node = self.root_nodes[para_index]
        for child in root_node.children:
            if child.visits > max_visits:
                max_visits = child.visits
                best_child = child
                
        return best_child.action

    cpdef object get_random_best_action(self, int para_index, int game_round):
        cdef:
            Node root_node, child, chosen_child
            float temperature
            list children
            int num_children, i
            cnp.ndarray[cnp.float32_t, ndim=1] visit_counts, visit_weights 

        root_node = self.root_nodes[para_index]
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
            Node root, matching_child
            bint found

        (<Armada>self.para_games[para_index]).revert_snapshot(snapshot)
        self.root_snapshots[para_index] = snapshot
        
        # Find the child node that matches the action taken.
        root = self.root_nodes[para_index]
        found = <bint>False
        for child in root.children:
            if child.action == action:
                matching_child = child
                found = <bint>True
                break
        if not found:
            matching_child = root.add_child(action, self.para_games[para_index])

        # The found child becomes the new root.
        self.root_nodes[para_index] = matching_child

    # cdef void _expand_root(self,)

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

        cdef int current_batch_size = len(phases)
        cdef int bucket_step = 16 
        cdef int target_batch_size = ((current_batch_size + bucket_step - 1) // bucket_step) * bucket_step
        cdef int max_batch_size = Config.GPU_INPUT_BATCH_SIZE
        cdef int i

        if target_batch_size > max_batch_size:
            target_batch_size = max_batch_size
        
        # 1. Fill buffers with current data (No allocation)
        # This replaces np.stack which was the CPU bottleneck
        for i in range(current_batch_size):
            state = encoded_states[i]
            # Copy data from small dict arrays into the big pinned buffer
            self.scalar_buffer[i] = state['scalar']
            self.ship_entity_buffer[i] = state['ship_entities']
            self.ship_coords_buffer[i] = state['ship_coords']
            self.ship_def_token_buffer[i] = state['ship_def_tokens']
            self.spatial_buffer[i] = state['spatial']
            self.relation_buffer[i] = state['relations']
            self.active_ship_indices_buffer[i] = state['active_ship_id']
            self.target_ship_indices_buffer[i] = state['target_ship_id']

        # 2. Zero-out padding area if needed (to clean up dirty data from previous steps)
        # Only strictly necessary if Batch Normalization statistics are sensitive to garbage
        if target_batch_size > current_batch_size:
            self.scalar_buffer[current_batch_size:target_batch_size].fill(0)
            self.ship_entity_buffer[current_batch_size:target_batch_size].fill(0)
            self.ship_coords_buffer[current_batch_size:target_batch_size].fill(0)
            self.ship_def_token_buffer[current_batch_size:target_batch_size].fill(0)
            self.spatial_buffer[current_batch_size:target_batch_size].fill(0)
            self.relation_buffer[current_batch_size:target_batch_size].fill(0)
            self.active_ship_indices_buffer[current_batch_size:target_batch_size].fill(max_ships)
            self.target_ship_indices_buffer[current_batch_size:target_batch_size].fill(max_ships)

        # 3. Handle Phases (simple list padding is fast enough)
        pad_len = target_batch_size - current_batch_size
        phase_ints = [p.value for p in phases] + [0] * pad_len

        # 4. Convert to PyTorch tensors
        scalar_tensor = torch.from_numpy(self.scalar_buffer[:target_batch_size]).to(Config.DEVICE)
        ship_entity_tensor = torch.from_numpy(self.ship_entity_buffer[:target_batch_size]).to(Config.DEVICE)
        ship_coords_tensor = torch.from_numpy(self.ship_coords_buffer[:target_batch_size]).to(Config.DEVICE)
        ship_def_token_tensor = torch.from_numpy(self.ship_def_token_buffer[:target_batch_size]).to(Config.DEVICE)
        spatial_tensor = torch.from_numpy(self.spatial_buffer[:target_batch_size]).to(Config.DEVICE)
        relation_tensor = torch.from_numpy(self.relation_buffer[:target_batch_size]).to(Config.DEVICE)
        active_ship_indices_tensor = torch.from_numpy(self.active_ship_indices_buffer[:target_batch_size]).to(Config.DEVICE, dtype=torch.long)
        target_ship_indices_tensor = torch.from_numpy(self.target_ship_indices_buffer[:target_batch_size]).to(Config.DEVICE, dtype=torch.long)
        phases_tensor = torch.tensor(phase_ints, dtype=torch.long, device=Config.DEVICE)

        with torch.no_grad():
            # Perform a single, batched forward pass
            outputs = self.model(
                scalar_tensor,
                ship_entity_tensor,
                ship_coords_tensor,
                ship_def_token_tensor,
                spatial_tensor,
                relation_tensor,
                active_ship_indices_tensor,
                target_ship_indices_tensor,
                phases_tensor
            )

            policy_logits = outputs['policy_logits']
            value_tensor = outputs['value']
            
            # Apply softmax to get probabilities
            policies = F.softmax(policy_logits[:current_batch_size], dim=1).cpu().numpy()

            # Squeeze to remove the last dimension (shape [B, 1] -> [B])
            values = value_tensor[:current_batch_size].squeeze(1).cpu().numpy()

        return values, policies

    cdef cnp.ndarray[cnp.float32_t, ndim=1] _mask_policy(self, cnp.ndarray[cnp.float32_t, ndim=1] policy, int phase, list valid_actions):
        cdef: 
            object action
            int action_index
            float policy_sum = 0.0
            int num_valid
            int policy_len = policy.shape[0]
            cnp.ndarray[cnp.npy_bool, ndim=1] action_mask_view

        action_mask_view = self.action_mask
        action_mask_view.fill(0)
        
        for action in valid_actions:
            # get the action's index in the full action space
            action_index = self.action_manager.get_action_index(phase, action)
            action_mask_view[action_index] = 1
            policy_sum += policy[action_index]

        

        # Normalize the policy if there are any valid moves
        if policy_sum > 0:
            policy *= action_mask_view
            policy /= policy_sum
        else:
            # Handle rare cases where the network assigns 0 probability to all valid moves
            # Or if there are no valid moves (should not happen for an expandable node)
            raise ValueError(f"Warning: Zero policy sum for valid moves in \n{valid_actions}. \nUsing uniform distribution. \n\nOriginal Policy: \n{policy}, \n\nAction Mask: \n{action_mask_view}")
            num_valid = len(valid_actions)
            policy[action_mask_view] = 1.0 / num_valid

        return policy

    cdef void _expand(self, Node node, int para_index, int phase, list valid_actions, float value, cnp.ndarray[cnp.float32_t, ndim=1] policy) :
        cdef: 
            tuple action
            int action_index
            float action_policy
            Armada game
            tuple node_snapshot = node.snapshot

        game = self.para_games[para_index]

        for action in valid_actions:
            action_index = self.action_manager.get_action_index(phase, action)
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
            
            # Use the parent's perspective to update the child node's stats
            if path:
                perspective = (<Node>path[-1]).first_player_perspective
            else: 
                perspective = 0 

            # Calculate result: 
            # If First Player Won (+1) and Parent is First Player (+1) -> +1 (Good)
            # If First Player Won (+1) and Parent is Second Player (-1) -> -1 (Bad)
            result_for_node = value * perspective
            
            node.update(result_for_node)

    cdef cnp.ndarray[cnp.float32_t, ndim=1] _get_final_action_probs(self, int para_index):
        cdef:
            cnp.ndarray[cnp.float32_t, ndim=1] action_probs = np.zeros(self.action_manager.max_action_space, dtype=np.float32)
            Node root_node = self.root_nodes[para_index]
            Node child
            float total_visits

        for child in root_node.children:
            action_probs[child.action_index] = child.visits

        total_visits = np.sum(action_probs, dtype=float)
        action_probs /= total_visits

        return action_probs


