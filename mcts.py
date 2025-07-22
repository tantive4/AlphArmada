import math
import random
import copy
from typing import TYPE_CHECKING, TypedDict, List, NotRequired, TypeAlias, Tuple, Literal, Optional

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
    ActiveShipAction: TypeAlias = Tuple[Literal["active_ship"], int]
    DeclareTargetAction: TypeAlias = Tuple[Literal["declare_target"], Tuple[HullSection, int, HullSection]]
    DetermineCourseAction: TypeAlias = Tuple[Literal["determine_course"], Tuple[List[int], int]]
    AttackDiceAction: TypeAlias = Tuple[Literal["attack_dice"], List[List[int]]]
    Action: TypeAlias = (
        ActiveShipAction | 
        DeclareTargetAction | 
        DetermineCourseAction | 
        AttackDiceAction
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
        actions = []
        game = state['game']
        player = state['current_player']
        phase = state['decision_phase']
        

        if phase == "activation" :
            valid_ships = game.get_valid_activation(player)
            for ship in valid_ships:
                actions.append(("activate_ship", ship.ship_id))
            if not valid_ships:
                actions.append(("pass_activation",))

        if state['active_ship_id'] is None :
            raise ValueError('Active Ship must be chosen')
        active_ship = game.ships[state['active_ship_id']]


        if phase == "declare_target":
            if active_ship.attack_count < 2:
                valid_hulls = active_ship.get_valid_attack_hull()
                for hull in valid_hulls:
                    actions.append(("choose_attack_hull", hull))
            actions.append(("skip_to_maneuver",))

        elif phase == "determine_course":
            valid_speeds = active_ship.get_valid_speed()
            for speed in valid_speeds:
                actions.append(("set_speed", speed))


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

    def get_best_action(self) -> ActionType.Action:
        if not self.root.children:
            possible_actions = self._get_possible_actions(self.root.state)
            return possible_actions[0]
            
        best_child = max(self.root.children, key=lambda c: c.visits)
        if best_child.action is None :
            raise ValueError('Child Node needs action from parent')
        return best_child.action
