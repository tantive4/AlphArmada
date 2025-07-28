import math
import random
import copy
from typing import TYPE_CHECKING, TypedDict, List, NotRequired, TypeAlias, Tuple, Literal, Optional
import itertools
from ship import Ship, HullSection
from time import sleep
# Conditionally import Armada only for type checking
if TYPE_CHECKING:
    from armada import Armada
    

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
    ActiveShipAction: TypeAlias = Tuple[Literal["activate_ship_action"], int]
    DeclareTargetAction: TypeAlias = Tuple[Literal["declare_target_action"], Tuple[HullSection, int, HullSection]]
    DetermineCourseAction: TypeAlias = Tuple[Literal["determine_course_action"], Tuple[List[int], int]]
    AttackDiceAction: TypeAlias = Tuple[Literal["attack_dice_roll"], List[List[int]]]
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
        self.wins : float = 0
        self.visits : int = 0
        self.untried_actions : Optional[list[ActionType.Action]] = None
        self.chance_node = chance_node


    def uct_select_child(self, owner_player : int, exploration_constant=1.414,) -> "Node":
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

    def add_child(self, action : ActionType.Action, state : MCTSState):
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
            depth = 1
            while (node.untried_actions is not None and not node.untried_actions and node.children) or node.chance_node:
                depth += 1
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
                        node = node.add_child(action=("attack_dice_roll", attack_dice), state=outcome_state)
                        state = node.state
                        state['decision_phase'] = 'declare_target'  # Reset to next phase after dice roll
                        break # Exit selection loop
                else:
                    # Standard player decision node, use UCT.
                    node = node.uct_select_child(owner_player=self.player)
                    state = copy.deepcopy(node.state)
            
            # 2. Expansion (for player decision nodes)
            if not state['game'].winner :
                if node.untried_actions is None:
                    node.untried_actions = self._get_possible_actions(state)

                if node.untried_actions:
                    action = node.untried_actions.pop()
                    child_state = self._apply_action(copy.deepcopy(state), action)
                    child_node = node.add_child(action, child_state)
                    node = child_node
                    state = child_state
                    
                    if action[0] == 'declare_target_action':
                        node.chance_node = True

            # 3. Simulation
            normalized_margin_of_victory = self._simulate(state)

            # 4. Backpropagation
            my_win_prob = 0.0
            if self.player == 1:
                my_win_prob = (normalized_margin_of_victory + 1) / 2
            else:  # If the MCTS owner is Player -1
                # Their win probability is the inverse of Player 1's
                my_win_prob = (-normalized_margin_of_victory + 1) / 2
            
            temp_node = node
            while temp_node is not None:
                # The perspective for updating a node's stats is determined by the player
                # who made the decision at that node's PARENT.
                # If there's no parent, it's the root, and the perspective is the MCTS owner's.
                perspective_player = temp_node.parent.state['current_player'] if temp_node.parent else self.player
                
                if perspective_player != self.player:
                    # This node is a result of the opponent's choice, so update with their win prob
                    temp_node.update(1 - my_win_prob)
                else:
                    # This node is a result of our choice, so update with our win prob
                    temp_node.update(my_win_prob)

                temp_node = temp_node.parent
            
            # if (i+1) % 20 == 0:
                # print(f"MCTS Iteration {i + 1}/{iterations} complete. Current best action: {self.get_best_action()} best wins: {int(self.get_best_child().wins)}, best visits: {self.get_best_child().visits}, depth: {depth}")
            print(f"{i+1}iteration. Total Win {round(self.root.wins)}. Best Action {self.get_best_action()} \n{[(node.action, round(node.wins), node.visits) for node in self.root.children]}")



    def _simulate(self, state: MCTSState) -> float:

        sim_state = copy.deepcopy(state)
        
        # Random rollout continues from the resulting state.
        sim_game = sim_state['game']
        sim_game.refresh_ship_links()
        max_simulation_steps = 500
        steps = 0
        while sim_game.winner is None and steps < max_simulation_steps:
            if sim_state['decision_phase'] == 'attack_dice_roll':
                # Simulate a random dice outcome
                if sim_state['active_ship_id'] is None or sim_state['declare_target'] is None:
                    raise ValueError("Active ship and defend ship must be set before rolling dice.")
                active_ship = sim_game.ships[sim_state['active_ship_id']]
                declare_target = sim_state['declare_target']
                attack_hull = declare_target[0]
                defend_ship = sim_game.ships[declare_target[1]]
                defend_hull = declare_target[2]

                attack_dice = active_ship.roll_attack_dice(attack_hull, defend_ship, defend_hull)
                if not attack_dice:
                    raise ValueError("Invalid attack dice outcome during simulation")

                sim_state = self._apply_dice_outcome(sim_state, attack_dice)
                sim_game = sim_state['game']
                continue

            possible_actions = self._get_possible_actions(sim_state)
            
            if not possible_actions:
                raise ValueError("No action during simulation")

            random_action = random.choice(possible_actions)
            sim_state = self._apply_action(sim_state, random_action)

            sim_game = sim_state['game']
            steps += 1
        if sim_game.winner is None:
            raise RuntimeError("Simulation exceeded maximum steps without a winner.")

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
        

        if phase == "ship_activation" :
            valid_ships = game.get_valid_activation(player)
            if not valid_ships:
                actions.append(("pass_activation", None))
            for ship in valid_ships:
                actions.append(("activate_ship_action", ship.ship_id))

            return actions
        

        else:
            if state['active_ship_id'] is None :
                raise ValueError('Active Ship must be chosen')
            active_ship = game.ships[state['active_ship_id']]


        if phase == "declare_target":
            if active_ship.attack_count < 2:
                for hull in active_ship.get_valid_attack_hull():
                    for defend_ship in active_ship.get_valid_target_ship(hull) :
                        for defend_hull in active_ship.get_valid_target_hull(hull, defend_ship):
                            actions.append(("declare_target_action", (hull, defend_ship.ship_id, defend_hull)))
            actions.append(("pass_attack", None))

        elif phase == "determine_course":
            for speed in active_ship.get_valid_speed():
                if speed == 0:
                    actions.append(("determine_course_action", ([], 1)))
                    continue
                yaw_options_per_joint = [active_ship.get_valid_yaw(speed, joint) for joint in range(speed)]
                for yaw_tuple in itertools.product(*yaw_options_per_joint):
                    course = [yaw - 2 for yaw in yaw_tuple]
                    for placement in active_ship.get_valid_placement(course):
                        actions.append(("determine_course_action", (course, placement)))
        
        else :
            raise ValueError(f"Unknown decision phase: {phase}")
        return actions

    def _apply_action(self, state: MCTSState, action: ActionType.Action) -> MCTSState:
        game = state['game']

        if action[0] == "activate_ship_action":
            state['active_ship_id'] = action[1]
            state['decision_phase'] = "declare_target"
            return state
        elif action[0] == "pass_activation":
            self._end_ship_activation(state)
            return state

        else :
            if state['active_ship_id'] is None :
                raise ValueError('Active Ship must be chosen to apply action')
            active_ship = game.ships[state['active_ship_id']]

        if action[0] == "declare_target_action":
            state['declare_target'] = action[1]
            state['decision_phase'] = "attack_dice_roll"

        elif action[0] == "pass_attack":
            state['decision_phase'] = "determine_course"

        elif action[0] == "determine_course_action":
            active_ship.move_ship(*action[1])
            self._end_ship_activation(state)

        else :
            raise ValueError(f"Unknown action type: {action[0]}")

        return state

    def _end_ship_activation(self, state: MCTSState):
        game = state['game']
        if state['active_ship_id'] is not None:
            ship = game.ships[state['active_ship_id']]
            ship.activated = True

        state['decision_phase'] = "ship_activation"
        state['active_ship_id'] = None
        state['declare_target'] = None

        if game.get_valid_activation(-state['current_player']):
            state['current_player'] *= -1

        elif not game.get_valid_activation(state['current_player']):
            game.status_phase()

    def get_best_action(self) -> ActionType.Action:
        if not self.root.children:
            possible_actions = self._get_possible_actions(self.root.state)
            return possible_actions[0]
            
        best_child = max(self.root.children, key=lambda c: c.visits)
        if best_child.action is None :
            raise ValueError('Child Node needs action from parent')
        return best_child.action
    
    def get_best_child(self) -> Node:
        if not self.root.children:
            raise ValueError("No children available to select best child")
        return max(self.root.children, key=lambda c: c.visits)