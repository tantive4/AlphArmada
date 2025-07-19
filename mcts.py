import math
import random
import copy
from typing import TypedDict, Any, Tuple, List, TYPE_CHECKING
if TYPE_CHECKING:
    from armada import Armada
    from ship import Ship, HullSection

class MCTSState(TypedDict):
    game: 'Armada'
    current_player: int
    decision_phase: str
    active_ship_id: int | None
    # The 'maneuver' field will store the sequence of choices building the full course
    maneuver: list[tuple] 

class Node:
    """
    Represents a node in the true Hierarchical MCTS tree.
    The node's position in the hierarchy (e.g., speed choice, yaw choice)
    is determined by the sequence of actions that led to it.
    """
    def __init__(self, state: MCTSState, parent: "Node | None" = None, action: Any = None):
        self.state = state
        self.parent = parent
        self.action = action # The specific action, e.g., ('set_speed', 2)
        
        self.children: list['Node'] = []
        self.wins = 0
        self.visits = 0
        
        self.untried_actions = self._get_possible_actions()
        self.is_terminal_maneuver_node = self._check_if_terminal_maneuver()

    def _get_possible_actions(self) -> list[Any]:
        """
        THIS IS THE CORE OF THE HIERARCHY.
        It determines the next level of actions based on the maneuver already built.
        """
        game = self.state['game']
        phase = self.state['decision_phase']
        
        # This function is now only for the 'maneuver' phase.
        # Other phases like 'activation' or 'attack' would need their own logic.
        if self.state['active_ship_id'] is None:
            return []

        if phase != 'maneuver':
            return []

        active_ship = game.ships[self.state['active_ship_id']]
        maneuver_steps = self.state['maneuver']

        # 1. TOP OF HIERARCHY: If no steps taken, choose SPEED.
        if not maneuver_steps:
            return [('set_speed', speed) for speed in active_ship.get_valid_speed()]

        # 2. MIDDLE OF HIERARCHY: If speed is set, choose YAW.
        speed_step = maneuver_steps[0]
        speed = speed_step[1]
        
        if speed == 0: # Speed 0 has no yaw or placement
            return []

        num_yaw_steps = len(maneuver_steps) - 1
        if num_yaw_steps < speed:
            joint_index = num_yaw_steps
            return [('set_yaw', yaw) for yaw in active_ship.get_valid_yaw(speed, joint_index)]

        # 3. BOTTOM OF HIERARCHY: If all yaws are set, choose PLACEMENT.
        if num_yaw_steps == speed:
            # Extract just the yaw values from the maneuver steps
            course = [step[1] - 2 for step in maneuver_steps[1:]]
            return [('set_placement', placement) for placement in active_ship.get_valid_placement(course)]

        return [] # All maneuver decisions made.

    def _check_if_terminal_maneuver(self) -> bool:
        """
        A node is "terminal" for the maneuver selection process if a full
        maneuver (speed + yaws + placement) has been selected. This is where a
        simulation should be run from.
        """
        if self.state['decision_phase'] != 'maneuver':
            return True # Not in maneuver phase, so not our concern
        
        maneuver_steps = self.state['maneuver']
        if not maneuver_steps:
            return False # No speed chosen yet

        speed = maneuver_steps[0][1]
        
        # Speed 0 is a full maneuver
        if speed == 0 and len(maneuver_steps) == 1:
            return True

        # Expected length is 1 (speed) + speed (yaws) + 1 (placement)
        expected_length = 1 + speed + 1
        return len(maneuver_steps) == expected_length


    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal_node(self) -> bool:
        return self.state['game'].winner is not None

    def select_best_child(self, exploration_constant: float = 1.41) -> 'Node':
        # Standard UCT selection
        best_score = -float('inf')
        best_children = []
        for child in self.children:
            if child.visits == 0:
                uct_score = float('inf')
            else:
                exploit_score = child.wins / child.visits
                explore_score = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
                uct_score = exploit_score + explore_score
            
            if uct_score > best_score:
                best_score = uct_score
                best_children = [child]
            elif uct_score == best_score:
                best_children.append(child)
        return random.choice(best_children)

    def expand(self) -> 'Node':
        action = self.untried_actions.pop()
        next_state = copy.deepcopy(self.state)
        # Add the chosen action to the maneuver sequence
        next_state['maneuver'].append(action)
        
        child_node = Node(state=next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

class MCTS:
    def __init__(self, initial_state: MCTSState, player: int):
        self.root = Node(state=initial_state)
        self.player = player

    def search(self, iterations: int):
        for _ in range(iterations):
            # 1. Selection: Traverse the hierarchy to a leaf node
            leaf = self._select(self.root)
            
            # 2. Expansion: If the maneuver isn't fully defined yet, expand
            if not leaf.is_terminal_maneuver_node:
                leaf = leaf.expand()
            
            # 3. Simulation: Run a random playout from the resulting state
            reward = self._simulate(leaf.state)
            
            # 4. Backpropagation: Update stats up the hierarchy
            self._backpropagate(leaf, reward)

    def _select(self, node: Node) -> Node:
        current_node = node
        # Traverse until we find a node that isn't fully expanded or is a terminal maneuver node
        while not current_node.is_terminal_maneuver_node:
            if not current_node.is_fully_expanded():
                return current_node
            else:
                current_node = current_node.select_best_child()
        return current_node

    def _simulate(self, state: MCTSState) -> int:
        """
        Runs a random playout. It first applies the chosen maneuver from the state,
        then continues the game randomly.
        """
        sim_game = copy.deepcopy(state['game'])
        sim_game.simulation_mode = True
        
        active_ship = sim_game.ships[state['active_ship_id']]
        maneuver = state['maneuver']

        # Apply the fully-defined maneuver to the simulation game state
        if maneuver:
            speed = maneuver[0][1]
            active_ship.speed = speed
            if speed > 0:
                course = [step[1] - 2 for step in maneuver[1:-1]]
                placement = maneuver[-1][1]
                active_ship.move_ship(course, placement)
            active_ship.activated = True
        
        # --- The rest of the simulation is similar to before ---
        current_player = sim_game.current_player * -1 # Next player's turn
        
        while sim_game.winner is None and sim_game.round <= 6:
            # (Random playout logic as in the previous version...)
            # This part can be copied from the previous implementation's _simulate method
            p1_can_activate = any(sim_game.get_valid_activation(1))
            p2_can_activate = any(sim_game.get_valid_activation(-1))

            if not p1_can_activate and not p2_can_activate:
                sim_game.status_phase()
                continue
            
            if not any(sim_game.get_valid_activation(current_player)):
                current_player *= -1
                continue

            valid_activations = sim_game.get_valid_activation(current_player)
            ship_to_activate = random.choice(valid_activations)
            ship_to_activate._execute_random_activation() # Assuming a helper for random moves
            current_player *= -1

        if sim_game.winner == self.player: return 1
        if sim_game.winner == -self.player: return -1
        return 0 # Draw or points based outcome

    def _backpropagate(self, node: Node, reward: int):
        # Standard backpropagation
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            # In a two-player game, the reward is inverted for the opponent's turn
            # Assuming MCTS is always called for the current player, this is simpler
            current_node.wins += reward 
            current_node = current_node.parent

    def get_best_action(self) -> list[tuple]:
        """
        After searching, finds the most promising path down the hierarchy
        and returns the full maneuver sequence.
        """
        if not self.root.children:
            return None
        
        best_maneuver = []
        current_node = self.root
        
        # Traverse down the tree, always choosing the most visited child
        while current_node.children:
            most_visited_child = max(current_node.children, key=lambda c: c.visits)
            best_maneuver.append(most_visited_child.action)
            current_node = most_visited_child
            
        return best_maneuver