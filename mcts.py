import math
import random
import copy
from typing import TYPE_CHECKING, TypedDict, List, NotRequired, TypeAlias, Tuple, Literal, Optional, cast
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
    course: Optional[list[Optional[int]]]

class ActionType :
    ActiveShipAction: TypeAlias = Tuple[Literal["activate_ship_action"], int]
    DeclareTargetAction: TypeAlias = Tuple[Literal["declare_target_action"], Tuple[HullSection, int, HullSection]]

    AttackDiceAction: TypeAlias = Tuple[Literal["attack_dice_roll"], List[List[int]]]
    PassAction: TypeAlias = Tuple[Literal["pass_activation"] | Literal["pass_attack"], None]

    # DetermineCourseAction: TypeAlias = Tuple[Literal["determine_course_action"], Tuple[List[int], int]]
    DetermineSpeedAction: TypeAlias = Tuple[Literal["determine_speed_action"], int]
    DetermineYawAction: TypeAlias = Tuple[Literal["determine_yaw_action"], int]
    DeterminePlacementAction: TypeAlias = Tuple[Literal["determine_placement_action"], int]


    Action: TypeAlias = (
        ActiveShipAction | 
        DeclareTargetAction | 
        AttackDiceAction | 
        PassAction |
        # DetermineCourseAction | 
        DetermineSpeedAction |
        DetermineYawAction |
        DeterminePlacementAction
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


    def uct_select_child(self, owner_player : int, exploration_constant=2,) -> "Node":
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


        # single decision optimization
        current_phase = self.root.state['decision_phase']
        is_maneuver_phase = current_phase in ["determine_speed", "determine_yaw", "determine_placement"]

        if not is_maneuver_phase:
            possible_actions = self._get_possible_actions(self.root.state)
            if len(possible_actions) == 1:
                if not self.root.state['game'].simulation_mode:
                    print(f"MCTS Info: Only one move for phase '{current_phase}', skipping search.")
                action = possible_actions[0]
                child_state = self._apply_action(copy.deepcopy(self.root.state), action)
                self.root.add_child(action, child_state)
                return

        for i in range(iterations):
            node = self.root
            state = copy.deepcopy(node.state)

            # 1. Selection
            while (node.untried_actions is not None and not node.untried_actions and node.children) or node.chance_node:
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
                    random.shuffle(node.untried_actions)

                if node.untried_actions:
                    action = node.untried_actions.pop()
                    child_state = self._apply_action(copy.deepcopy(state), action)
                    child_node = node.add_child(action, child_state)
                    node = child_node
                    state = child_state
                    
                    if action[0] == 'declare_target_action':
                        node.chance_node = True

            # 3. Simulation
            simulation_result = self._simulate(state)

            # 4. Backpropagation (Updated for -1 to 1 scoring)
            temp_node = node
            while temp_node is not None:
                # The result must be from the perspective of the player who made the move at the parent node.
                perspective_player = temp_node.parent.state['current_player'] if temp_node.parent else self.player
                
                # The simulation_result is always from Player 1's perspective.
                # If the current node's move was made by Player -1, we flip the score.
                result_for_node = simulation_result if perspective_player == 1 else -simulation_result
                
                temp_node.update(result_for_node)
                temp_node = temp_node.parent
            
            # if (i+1) % 20 == 0:
                # print(f"MCTS Iteration {i + 1}/{iterations} complete. Current best action: {self.get_best_action()} best wins: {int(self.get_best_child().wins)}, best visits: {self.get_best_child().visits}, depth: {depth}")
            with open('simulation_log.txt', 'a') as f: f.write(f"\n{i+1} iteration. Total Win {round(self.root.wins,2)}. Best Action {self.get_best_action()} \n{[(node.action, round(node.wins,2), node.visits) for node in self.root.children]}")



    def _simulate(self, state: MCTSState) -> float:
        """
        simulate random game from current state and return the winner
        winner is the MoV value, normalized as -1 ~ 1
        """
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
        with open('simulation_log.txt', 'a') as f: f.write(f"\nsimulation ends : {sim_game.winner}")
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
        else : outcome_state['decision_phase'] = 'determine_speed'

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

        # elif phase == "determine_course":
        #     for speed in active_ship.get_valid_speed():
        #         if speed == 0:
        #             actions.append(("determine_course_action", ([], 1)))
        #             continue
        #         yaw_options_per_joint = [active_ship.get_valid_yaw(speed, joint) for joint in range(speed)]
        #         for yaw_tuple in itertools.product(*yaw_options_per_joint):
        #             course = [yaw for yaw in yaw_tuple]
        #             for placement in active_ship.get_valid_placement(course):
        #                 actions.append(("determine_course_action", (course, placement)))


        # Hierarchical Maneuver Decision

        elif phase == "determine_speed":
            state['course'] = None
            for speed in active_ship.get_valid_speed():
                actions.append(("determine_speed_action", speed))

        elif phase == "determine_yaw":
            course = state['course']
            if course is None:
                raise ValueError("Course must be initialized to determine yaw.")
            
            # Find the first 'None' to determine which joint we are deciding
            try:
                joint_index = course.index(None)
                speed = len(course)
                for yaw in active_ship.get_valid_yaw(speed, joint_index):
                    actions.append(("determine_yaw_action", yaw))
            except ValueError:
                # This case should not be reached if logic is correct, as phase should have changed
                raise ValueError("Attempted to determine yaw for a fully defined course.")

        elif phase == "determine_placement":
            course = state['course']
            if course is None or None in course:
                 raise ValueError("Course must be fully defined to determine placement.")
            # We need to cast away the 'Optional' since we've confirmed no Nones exist
            course = cast(List[int], course)
            for placement in active_ship.get_valid_placement(course):
                actions.append(("determine_placement_action", placement))
        
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
            state['decision_phase'] = "determine_speed"
            

        # elif action[0] == "determine_course_action":
        #     active_ship.move_ship(*action[1])
        #     self._end_ship_activation(state)

        # --- HIERARCHICAL MANEUVER STATE TRANSITIONS ---
        elif action[0] == "determine_speed_action":
            speed = action[1]
            if speed == 0:
                # Speed 0 is a complete maneuver. Execute and end activation.
                active_ship.move_ship([], 1)
                self._end_ship_activation(state)
            else:
                # Initialize the course list with Nones and move to yaw decision.
                state['course'] = [None for _ in range(speed)]
                state['decision_phase'] = "determine_yaw"
        
        elif action[0] == "determine_yaw_action":
            yaw = action[1]
            course = state['course']
            if course is None:
                raise ValueError("Cannot apply yaw action, course is not initialized.")
            
            # Find the first None and replace it with the chosen yaw.
            joint_index = course.index(None)
            course[joint_index] = yaw
            
            # If there are no more Nones, the course is complete. Move to placement.
            if None not in course:
                state['decision_phase'] = "determine_placement"
                
        elif action[0] == "determine_placement_action":
            placement = action[1]
            course = state['course']
            if course is None or None in course:
                raise ValueError("Cannot apply placement, course is not fully defined.")
            
            # We cast the type because we've confirmed there are no 'None' values.
            final_course = cast(List[int], course)
            active_ship.move_ship(final_course, placement)
            self._end_ship_activation(state)

        elif action[0] == "attack_dice_roll":
            raise ValueError("Chance node action should not be applied in _apply_action")
        else:
            raise ValueError(f"Unknown action type in _apply_action: {action[0]}")

        return state

    def _end_ship_activation(self, state: MCTSState):
        game = state['game']
        if state['active_ship_id'] is not None:
            ship = game.ships[state['active_ship_id']]
            ship.activated = True

        state['decision_phase'] = "ship_activation"
        state['active_ship_id'] = None
        state['declare_target'] = None
        state['course'] = None

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