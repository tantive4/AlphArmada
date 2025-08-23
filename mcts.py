from __future__ import annotations

import math
import random
import copy
from typing import TYPE_CHECKING
import dice
from game_phase import GamePhase, ActionType
if TYPE_CHECKING:
    from armada import Armada




class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    Can be a decision node (for a player) or a chance node (for random events).
    """
    def __init__(self, decision_player : int | None, parent : Node | None =None, action : ActionType.Action | None=None, chance_node : bool = False) -> None :
        
        if (parent is None) != (action is None):
            raise ValueError("Root nodes must have no parent and no action, while child nodes must have both.")

        self.decision_player : int | None = decision_player
        self.parent : Node | None = parent
        self.action : ActionType.Action | None = action
        self.children : list[Node] = []
        self.wins : float = 0
        self.visits : int = 0
        self.untried_actions : list[ActionType.Action]| None = None
        self.chance_node = chance_node


    def uct_select_child(self, exploration_constant=2) -> Node:
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
        tie_count : int = 0

        for child in self.children:
            # Exploit term (win rate)
            if child.visits == 0:
                return child
            win_rate = child.wins / (child.visits)

            # Explore term
            exploration = exploration_constant * math.sqrt(log_parent_visits / (child.visits))
            
            uct_score = win_rate + exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child
                tie_count : int = 1  # Reset the count for the new best score
            elif uct_score == best_score:
                tie_count += 1
                # Reservoir sampling: with a 1/tie_count probability, replace the best child
                if random.randint(1, tie_count) == 1:
                    best_child = child

        return best_child # Randomly choose from the best options

    def add_child(self, action : ActionType.Action, game : Armada) -> Node :
        """
        Adds a new child node for the given action and game.
        """
        node = Node(decision_player=game.decision_player, parent=self, action=action)
        self.children.append(node)
        return node

    def update(self, result) -> None :
        """
        Updates the node's statistics from a simulation result.
        """
        self.visits += 1
        self.wins += result



class MCTS:
    """
    Basic Monte Carlo Tree Search for Star Wars: Armada.

    """

    def __init__(self, initial_game: Armada) -> None:
        self.game = initial_game
        self.root : Node = Node(decision_player=initial_game.decision_player)
        if self.root.decision_player is None :
            raise ValueError("MCTS requires a decision player to be set in the game.")

    def search(self, iterations: int) -> None:
        # single decision optimization

        possible_actions = self.game.get_possible_actions()
        if len(possible_actions) == 1:
            action = possible_actions[0]
            self.game.apply_action(action)
            self.root.add_child(action, self.game)
            return

        for i in range(iterations):
            game : Armada = copy.deepcopy(self.game)
            node : Node = self.root
            
            # 1. Selection
            while (node.untried_actions is not None and not node.untried_actions and node.children) or node.chance_node:

                if node.chance_node:
                    # For a chance node, sample a random outcome instead of using UCT.
                    if game.attack_info is None :
                        raise ValueError("Invalid game for chance node: missing attack/defend info.")
                    
                    dice_roll = dice.roll_dice(game.attack_info.dice_to_roll)
                    action = ("roll_dice_action", dice_roll)
                    game.apply_action(action)

                    dice_roll_result_node = next((child for child in node.children if child.action is not None and child.action[1] == dice_roll), None)
                    if dice_roll_result_node:
                        # If we've seen this random outcome before for this node, continue selection down that path.
                        node = dice_roll_result_node
                    else:
                        # If it's a new outcome, this is the node we will expand and simulate.
                        node = node.add_child(action, game)
                        break # Exit selection loop

                else:
                    # Standard player decision node, use UCT.
                    node = node.uct_select_child()
                    if node.action is None:
                        raise ValueError("Child node must have an action.")
                    game.apply_action(node.action)
            
            # 2. Expansion (for player decision nodes)
            if not game.winner :
                if node.untried_actions is None:
                    node.untried_actions = game.get_possible_actions()
                    random.shuffle(node.untried_actions)

                if node.untried_actions:
                    action = node.untried_actions.pop()
                    game.apply_action(action)
                    child_node = node.add_child(action, game)
                    node = child_node
                    
                    if game.phase == GamePhase.SHIP_ATTACK_ROLL_DICE :
                        node.chance_node = True

            # 3. Simulation
            simulation_result = game.play(max_simulation_step=1000)
            # with open('simulation_log.txt', 'a') as f: f.write(f"\nSimulation Result: {simulation_result}")

            # 4. Backpropagation (Updated for -1 to 1 scoring)
            temp_node = node
            while temp_node is not None:
                # The result must be from the perspective of the player who made the move at the parent node.
                perspective_player = temp_node.parent.decision_player if temp_node.parent else self.root.decision_player
                
                # The simulation_result is always from Player 1's perspective.
                # If the current node's move was made by Player -1, we flip the score.
                if perspective_player == 1 :
                    result_for_node = simulation_result
                elif perspective_player == -1:
                    result_for_node = -simulation_result
                else :
                    result_for_node = 0
                
                temp_node.update(result_for_node)
                temp_node = temp_node.parent
            
            # with open('simulation_log.txt', 'a') as f: f.write(f"\n{i+1} iteration. Total Win {round(self.root.wins,2)}. Best Action {self.get_best_action()} \n{[(node.action, round(node.wins,2), node.visits) for node in self.root.children]}")
            if i % 100 == 99:
                print(f"Iteration {i + 1}/{iterations}: Total Wins: {round(self.root.wins, 2)}, Best Action: {self.get_best_action()}")

    def get_best_action(self) -> ActionType.Action:
        if not self.root.children:
            possible_actions = self.game.get_possible_actions()
            random.shuffle(possible_actions)
            return possible_actions[0]
            
        best_child = max(self.root.children, key=lambda c: c.visits)
        if best_child.action is None :
            raise ValueError('Child Node needs action from parent')
        return best_child.action
    