import math
import random
import copy
from typing import TYPE_CHECKING, TypedDict, List, NotRequired, TypeAlias, Tuple, Literal, Optional
import itertools

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
    declare_target: Optional[tuple[HullSection, int, HullSection]]

class ActionType :
    ActiveShipAction: TypeAlias = Tuple[Literal["activate_ship"], int]
    DeclareTargetAction: TypeAlias = Tuple[Literal["declare_target"], Tuple[HullSection, int, HullSection]]
    DetermineCourseAction: TypeAlias = Tuple[Literal["determine_course"], Tuple[List[int], int]]
    AttackDiceAction: TypeAlias = Tuple[Literal["attack_dice"], List[List[int]]]
    PassAction: TypeAlias = Tuple[Literal["pass_activation"] | Literal["pass_attack"], None]

    Action: TypeAlias = (
        ActiveShipAction | 
        DeclareTargetAction | 
        DetermineCourseAction | 
        AttackDiceAction | 
        PassAction
    )

class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    Can be a decision node (for a player) or a chance node (for random events).
    """
    def __init__(self, state : MCTSState, parent : "Node | None" =None, action : Optional[ActionType.Action]=None, chance_node : bool = False):
        self.parent = parent
        self.state : MCTSState= state
        self.action = action
        self.children : list['Node']= []
        self.wins = 0
        self.visits = 0
        self.untried_actions : Optional[list[ActionType.Action]] = None
        self.chance_node = chance_node

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
    Basic Monte Carlo Tree Search for Star Wars: Armada.

    """


    def __init__(self, initial_state: MCTSState, player: int):
        self.root = Node(state=initial_state)
        self.player = player

    def search(self, iterations: int):
        possible_actions = self._get_possible_actions(self.root.state)
        if len(possible_actions) == 1:
            print("MCTS Thinking... [Only one move, skipping search.]")
            action = possible_actions[0]
            child_state = self._apply_action(copy.deepcopy(self.root.state), action)
            self.root.add_child(action, child_state)
            return

        for i in range(iterations):
            node = self.root
            state = copy.deepcopy(node.state)

            # 1. Selection
            while node.untried_actions is not None and not node.untried_actions and node.children:
                if node.chance_node:
                    # For a chance node, sample a random outcome instead of using UCT.
                    if state['active_ship_id'] is None or state['declare_target'] is None :
                        raise ValueError("Invalid state for chance node: missing attack/defend info.")
                    
                    active_ship = state['game'].ships[state['active_ship_id']]
                    declare_target = state['declare_target']
                    attack_hull = declare_target[0]
                    defend_ship = state['game'].ships[declare_target[1]]
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
                        state = node.state
                    else:
                        # If it's a new outcome, this is the node we will expand and simulate.
                        outcome_state = self._apply_dice_outcome(copy.deepcopy(state), attack_dice)
                        node = node.add_child(action=("attack_dice", attack_dice), state=outcome_state)
                        state = node.state
                        break # Exit selection loop
                else:
                    # Standard player decision node, use UCT.
                    node = node.uct_select_child(owner_player=self.player)
                    state = copy.deepcopy(node.state)
            
            # 2. Expansion (for player decision nodes)
            if not state['game'].winner and not node.chance_node:
                if node.untried_actions is None:
                    node.untried_actions = self._get_possible_actions(state)

                if node.untried_actions:
                    action = node.untried_actions.pop()
                    child_state = self._apply_action(copy.deepcopy(state), action)
                    child_node = node.add_child(action, child_state)
                    node = child_node
                    state = child_state
                    
                    if action[0] == 'declare_full_attack':
                        node.chance_node = True

            # 3. Simulation
            winner = self._simulate(state)

            # 4. Backpropagation
            while node is not None:
                result = 1 if winner == self.player else 0
                node.update(result)
                node = node.parent
            
            print(f"MCTS Iteration {i + 1}/{iterations} complete. Current best action: {self.get_best_action()}")

    def _simulate(self, state: MCTSState) -> int:
        sim_state = copy.deepcopy(state)
        
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

    def _apply_dice_outcome(self, state: MCTSState, outcome: list) -> MCTSState:
        """
        Applies a specific dice roll outcome to the state, resolving damage and setting up the next phase.
        """
        outcome_state = copy.deepcopy(state)
        if outcome_state['active_ship_id'] is None or outcome_state['declare_target'] is None:
            raise ValueError("Active ship and defend ship must be set before applying dice outcome.")
        
        active_ship_copy = outcome_state['game'].ships[outcome_state['active_ship_id']]
        defend_ship_copy = outcome_state['game'].ships[outcome_state['declare_target'][1]]
        
        active_ship_copy.resolve_damage(defend_ship_copy, outcome_state['declare_target'][2], outcome)
        
        # After one attack, the next decision is the start of the next attack
        if active_ship_copy.attack_count < 2 : outcome_state['decision_phase'] = 'declare_target'
        else : outcome_state['decision_phase'] = 'determine_course'

        return outcome_state

    def _get_possible_actions(self, state: MCTSState) -> list[ActionType.Action]:
        actions : List[ActionType.Action] = []
        game = state['game']
        player = state['current_player']
        phase = state['decision_phase']
        

        if phase == "activation" :
            valid_ships = game.get_valid_activation(player)
            for ship in valid_ships:
                actions.append(("activate_ship", ship.ship_id))
            if not valid_ships:
                actions.append(("pass_activation", None))

        if state['active_ship_id'] is None :
            raise ValueError('Active Ship must be chosen')
        active_ship = game.ships[state['active_ship_id']]


        if phase == "declare_target":
            if active_ship.attack_count < 2:
                for hull in active_ship.get_valid_attack_hull():
                    for defend_ship in active_ship.get_valid_target_ship(hull) :
                        for defend_hull in active_ship.get_valid_target_hull(hull, defend_ship):
                            actions.append(("declare_target", (hull, defend_ship.ship_id, defend_hull)))
            actions.append(("pass_attack", None))

        elif phase == "determine_course":
            for speed in active_ship.get_valid_speed():
                if speed == 0:
                    continue
                yaw_options_per_joint = [active_ship.get_valid_yaw(speed, joint) for joint in range(speed)]
                for yaw_tuple in itertools.product(*yaw_options_per_joint):
                    course = [yaw - 2 for yaw in yaw_tuple]
                    for placement in active_ship.get_valid_placement(course):
                        actions.append(("determine_course", (course, placement)))
        return actions

    def _apply_action(self, state: MCTSState, action: ActionType.Action) -> MCTSState:
        game = state['game']
        if state['active_ship_id'] is None :
            raise ValueError('Active Ship must be chosen to apply action')
        
        active_ship = game.ships[state['active_ship_id']]

        if action[0] == "activate_ship":
            state['active_ship_id'] = action[1]
            state['decision_phase'] = "declare_target"

        elif action[0] == "declare_target":
            attack_pool = active_ship.roll_attack_dice(action[1][0], game.ships[action[1][1]], action[1][2])
            if attack_pool:
                active_ship.resolve_damage(game.ships[action[1][1]], action[1][2], attack_pool)

        elif action[0] == "skip_to_maneuver":
            state['decision_phase'] = "maneuver_speed"

        elif action[0] == "determine_course":
            active_ship.move_ship(*action[1])
            self._end_activation(state)
            
        elif action[0] == "pass_activation":
             self._end_activation(state)

        return state

    def _end_activation(self, state: MCTSState):
        game = state['game']
        if state['active_ship_id'] is not None:
            ship = game.ships[state['active_ship_id']]
            ship.activated = True

        state['decision_phase'] = "activation"
        state['active_ship_id'] = None
        state['declare_target'] = None

        if game.get_valid_activation(-state['current_player']):
            state['current_player'] *= -1

        elif not game.get_valid_activation(state['current_player']):
            game.status_phase()
            if not game.winner:
                game.round += 1
                state['current_player'] = 1

    def get_best_action(self) -> ActionType.Action:
        if not self.root.children:
            possible_actions = self._get_possible_actions(self.root.state)
            return possible_actions[0]
            
        best_child = max(self.root.children, key=lambda c: c.visits)
        if best_child.action is None :
            raise ValueError('Child Node needs action from parent')
        return best_child.action