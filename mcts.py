from __future__ import annotations
import multiprocessing
from multiprocessing.managers import BaseManager
import math
import random
import copy
from typing import TYPE_CHECKING
import dice
from game_phase import GamePhase, ActionType
if TYPE_CHECKING:
    from armada import Armada


class NodeManager(BaseManager):
    pass

class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
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
        self.virtual_visits = 0
        self.lock = multiprocessing.Lock()

    def uct_select_child(self, exploration_constant=2) -> Node:
        """
        Selects a child node using UCT, now including virtual visits for parallel search.
        """
        if self.chance_node:
            raise ValueError("uct_select_child called on a chance node.")
        if not self.children:
            raise ValueError("uct_select_child called on a node with no children.")

        log_parent_total_visits = math.log(self.visits + self.virtual_visits)
        
        best_score = -float('inf')
        tie_count = 0
        # Initialize best_child to a default value to handle all cases
        best_child = self.children[0] 

        for child in self.children:
            # Always prioritize expanding a node that has never been visited
            if child.visits + child.virtual_visits == 0:
                return child

            total_visits = child.visits + child.virtual_visits
            win_rate = child.wins / total_visits
            exploration = exploration_constant * math.sqrt(log_parent_total_visits / total_visits)
            uct_score = win_rate + exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child
                tie_count = 1
            elif uct_score == best_score:
                tie_count += 1
                if random.randint(1, tie_count) == 1:
                    best_child = child
        return best_child

    def add_child(self, action : ActionType.Action, game : Armada) -> Node :
        node = Node(decision_player=game.decision_player, parent=self, action=action)
        self.children.append(node)
        return node

    def backpropagate(self, simulation_result: float):
        """
        Thread-safe backpropagation that reverts virtual loss and applies the real result.
        """
        node = self
        while node is not None:
            with node.lock:
                node.virtual_visits -= 1
                node.visits += 1
                
                perspective_player = node.parent.decision_player if node.parent else node.decision_player
                
                if perspective_player == 1:
                    result_for_node = simulation_result
                elif perspective_player == -1:
                    result_for_node = -simulation_result
                else:
                    result_for_node = 0
                
                node.wins += result_for_node
            node = node.parent


NodeManager.register('Node', Node)


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
            # 1. SAVE STATE at the beginning of the iteration
            initial_snapshot = self.game.get_snapshot()
            node : Node = self.root
            
            # 2. Selection
            while (node.untried_actions is not None and not node.untried_actions and node.children) or node.chance_node:

                if node.chance_node:
                    # For a chance node, sample a random outcome instead of using UCT.
                    if self.game.attack_info is None :
                        raise ValueError("Invalid game for chance node: missing attack/defend info.")
                    
                    dice_roll = dice.roll_dice(self.game.attack_info.dice_to_roll)
                    action = ("roll_dice_action", dice_roll)
                    self.game.apply_action(action)

                    dice_roll_result_node = next((child for child in node.children if child.action is not None and child.action[1] == dice_roll), None)
                    if dice_roll_result_node:
                        # If we've seen this random outcome before for this node, continue selection down that path.
                        node = dice_roll_result_node
                    else:
                        # If it's a new outcome, this is the node we will expand and simulate.
                        node = node.add_child(action, self.game)
                        break # Exit selection loop

                else:
                    # Standard player decision node, use UCT.
                    node = node.uct_select_child()
                    if node.action is None:
                        raise ValueError("Child node must have an action.")
                    self.game.apply_action(node.action)
            
            # 3. Expansion (for player decision nodes)
            if not self.game.winner :
                if node.untried_actions is None:
                    node.untried_actions = self.game.get_possible_actions()
                    random.shuffle(node.untried_actions)

                if node.untried_actions:
                    action = node.untried_actions.pop()
                    self.game.apply_action(action)
                    child_node = node.add_child(action, self.game)
                    node = child_node
                    
                    if self.game.phase == GamePhase.SHIP_ATTACK_ROLL_DICE :
                        node.chance_node = True

            # 4. Simulation
            simulation_result = self.game.play(max_simulation_step=1000)
            # with open('simulation_log.txt', 'a') as f: f.write(f"\nSimulation Result: {simulation_result}")

            # 5. Backpropagation (Updated for -1 to 1 scoring)
            node.backpropagate(simulation_result)

            # 6. REVERT STATE at the end of the iteration
            self.game.revert_snapshot(initial_snapshot)
            
            if (i+1) % 400 == 0:
                print(f"Iteration {i + 1}/{iterations}: Total Wins: {round(self.root.wins, 2)}, Best Action | {ActionType.get_action_str(self.game, self.get_best_action())}")
                with open('simulation_log.txt', 'a') as f: f.write(f"\n{i+1} iteration. Total Win {round(self.root.wins,2)}. Best Action {self.get_best_action()} \n{[(node.action, round(node.wins,2), node.visits) for node in self.root.children]}")


    def search_parallel(self, iterations: int, num_processes: int = 4) -> None:
        if len(self.game.get_possible_actions()) == 1:
            self.game.apply_action(self.game.get_possible_actions()[0])
            return

        manager = NodeManager()
        manager.start()
        
        assert isinstance(manager, NodeManager)
        shared_root = manager.Node(decision_player=self.game.decision_player)  # type: ignore

        processes = []
        iters_per_worker = iterations // num_processes

        for _ in range(num_processes):
            p = multiprocessing.Process(
                target=mcts_worker, 
                args=(shared_root, self.game, iters_per_worker)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        self.root.wins = shared_root.wins
        self.root.visits = shared_root.visits
        self.root.children = shared_root.children
        
        manager.shutdown()

        print(f"Parallel search complete. Best Action | {ActionType.get_action_str(self.game, self.get_best_action())}")

    def get_best_action(self) -> ActionType.Action:
        if not self.root.children:
            possible_actions = self.game.get_possible_actions()
            random.shuffle(possible_actions)
            return possible_actions[0]
            
        best_child = max(self.root.children, key=lambda c: c.visits)
        if best_child.action is None :
            raise ValueError('Child Node needs action from parent')
        return best_child.action

# multiprocessing
def mcts_worker(shared_root: Node, game_prototype: Armada, iterations_per_worker: int):
    """The main function for each worker process in the parallel search."""
    game = copy.deepcopy(game_prototype)

    for _ in range(iterations_per_worker):
        initial_snapshot = game.get_snapshot()
        node = shared_root
        path = [node]

        # 1. Selection
        while (node.untried_actions is not None and not node.untried_actions and node.children) or node.chance_node:
            if node.chance_node:
                if game.attack_info is None:
                    raise ValueError("Invalid game for chance node: missing attack/defend info.")
                dice_roll = dice.roll_dice(game.attack_info.dice_to_roll)
                action = ("roll_dice_action", dice_roll)
                game.apply_action(action)
                child_node = next((child for child in node.children if child.action is not None and child.action[1] == dice_roll), None)
                if child_node:
                    node = child_node
                else:
                    node = node.add_child(action, game)
                    break
            else:
                node = node.uct_select_child()
                if node.action is None:
                    raise ValueError("Selected child node has no action.")
                game.apply_action(node.action)
            path.append(node)

        # 2. Apply Virtual Loss to the selected path
        for visited_node in path:
            with visited_node.lock:
                visited_node.virtual_visits += 1
        
        # 3. Expansion
        if not game.winner:
            with node.lock:
                if node.untried_actions is None:
                    node.untried_actions = game.get_possible_actions()
                    random.shuffle(node.untried_actions)
                
                if node.untried_actions:
                    action = node.untried_actions.pop()
                    game.apply_action(action)
                    node = node.add_child(action, game)
                    with node.lock:
                        node.virtual_visits += 1

        # 4. Simulation
        simulation_result = game.play(max_simulation_step=1000)

        # 5. Backpropagation
        node.backpropagate(simulation_result)

        # 6. Revert local game state
        game.revert_snapshot(initial_snapshot)
