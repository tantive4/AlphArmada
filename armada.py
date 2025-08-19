from __future__ import annotations
from ship import Ship, HullSection
import random
from shapely.geometry import Polygon
import visualizer
import dice
import copy
from mcts import MCTS
import itertools
from game_phase import GamePhase, ActionType
from typing import Callable

class Armada:
    def __init__(self) -> None:
        self.player_edge = 900
        self.short_edge = 900
        self.game_board = Polygon([
            (0,0),
            (0,self.player_edge),
            (self.short_edge, self.player_edge),
            (self.short_edge, 0),
        ])
        self.ships : list[Ship] = []

        self.round : int = 1
        self.phase = GamePhase.SHIP_PHASE
        self.current_player : int = 1
        self.decision_player : int | None = 1
        self.active_ship : Ship | None = None
        self.attack_info : AttackInfo | None = None
        self.determine_course : list[int | None] | None = None

        self.winner : float | None = None
        self.image_counter : int = 0
        self.simulation_mode : bool = False




    def play(self, 
             player1 : Callable[[], ActionType.Action] | None = None, 
             player2 : Callable[[], ActionType.Action] | None = None, 
             max_simulation_step : int | None = None) -> float :
        """
        The main game loop with two players defined by functions.
        Each player function should return an action based on the current game state.
        """
        if player1 is None : 
            player1 = self.random_decision
        if player2 is None : 
            player2 = self.random_decision
        simulation_counter = 0

        
        while self.winner is None:
            if self.phase == GamePhase.SHIP_ATTACK_ROLL_DICE : # chance node case
                if self.attack_info is None :
                    raise ValueError("No attack info for the current game phase.") 
                dice_roll = dice.roll_dice(self.attack_info.attack_pool)
                action = ('roll_dice_action', dice_roll)

            elif self.decision_player == 1 : 
                action : ActionType.Action = player1()
            elif self.decision_player == -1 :
                action : ActionType.Action = player2()

            else :
                action : ActionType.Action = self.get_possible_actions()[0]
            self.apply_action(action)

            simulation_counter += 1
            if max_simulation_step is not None and simulation_counter >= max_simulation_step:
                raise RuntimeError(f'Maximum simulation steps reached: {max_simulation_step}')
        return self.winner

    def random_decision(self) -> ActionType.Action:
        """
        A simple strategy that returns a random action from the list of possible actions.
        """
        actions = self.get_possible_actions()
        return random.choice(actions)


    def mcts_decision(self, iterations : int = 1000) -> ActionType.Action:
        game_copy : Armada = copy.deepcopy(self)
        game_copy.simulation_mode = True
        mcts = MCTS(game_copy)
        mcts.search(iterations)
        action : ActionType.Action = mcts.get_best_action()
        return action

    def update_decision_player(self) -> None:
        """
        Returns a list of possible actions based on the current game phase.
        """

        match self.phase:
            case GamePhase.SHIP_PHASE :
                self.decision_player = self.current_player

            case GamePhase.SHIP_ATTACK_DECLARE_TARGET :
                self.decision_player = self.current_player
            
            case GamePhase.SHIP_ATTACK_GATHER_DICE :
                self.decision_player = self.current_player

            case GamePhase.SHIP_ATTACK_ROLL_DICE :
                self.decision_player = None

            case GamePhase.SHIP_ATTACK_RESOLVE_DAMAGE :
                self.decision_player = self.current_player

            case GamePhase.SHIP_MANEUVER_DETERMINE_COURSE :
                self.decision_player = self.current_player

            case GamePhase.STATUS_PHASE :
                self.decision_player = None

            case _ :
                raise ValueError(f'Unknown game phase: {self.phase}')
    

    def get_possible_actions(self) -> list[ActionType.Action]:
        """
        Returns a list of possible actions based on the current game phase.
        """
        self.update_decision_player()
        actions : list[ActionType.Action] = []

        if self.phase > GamePhase.SHIP_PHASE and self.phase < GamePhase.SQUADRON_PHASE:
            if self.active_ship is None:
                raise ValueError('No active ship for the current game phase.')
            active_ship : Ship = self.active_ship
        if self.phase > GamePhase.SHIP_ATTACK_DECLARE_TARGET and self.phase < GamePhase.SHIP_EXECUTE_MANEUVER:
            if self.attack_info is None:
                raise ValueError('No attack info for the current game phase.')
            attack_info : AttackInfo = self.attack_info



        match self.phase:
            case GamePhase.SHIP_PHASE :
                if self.winner is not None:
                    raise ValueError('Game has already ended.')
                
                valid_ships = self.get_valid_activation(self.current_player)
                if valid_ships : 
                    actions = [('activate_ship_action', ship.ship_id) for ship in valid_ships]
                else :
                    actions = [('pass_ship_activation', None)]

            case GamePhase.SHIP_ATTACK_DECLARE_TARGET :
                if active_ship.attack_count < 2 :
                    actions = [('declare_target_action', (attack_hull, defend_ship.ship_id, defend_hull))
                        for attack_hull in active_ship.get_valid_attack_hull()
                        for defend_ship in active_ship.get_valid_target_ship(attack_hull)
                        for defend_hull in active_ship.get_valid_target_hull(attack_hull, defend_ship)]
                if not actions : actions = [('pass_attack', None)]
            
            case GamePhase.SHIP_ATTACK_GATHER_DICE :
                if attack_info.obstructed:
                    for index, value in enumerate(attack_info.attack_pool):
                        if value > 0:
                            new_pool = attack_info.attack_pool.copy()
                            new_pool[index] -= 1
                            actions.append(('gather_dice_action', new_pool))
                else:
                    actions = [('gather_dice_action', attack_info.attack_pool)]
            
            case GamePhase.SHIP_ATTACK_ROLL_DICE :
                actions = [('roll_dice_action', dice_roll) for dice_roll in dice.generate_all_dice_outcomes(attack_info.attack_pool)]
                

            case GamePhase.SHIP_ATTACK_RESOLVE_DAMAGE :
                actions = [('resolve_damage_action', None)]


            case GamePhase.SHIP_MANEUVER_DETERMINE_COURSE :
                for speed in active_ship.get_valid_speed():
                    if speed == 0:
                        actions.append(('determine_course_action', ([], 1)))
                        continue
                    yaw_options_per_joint = [active_ship.get_valid_yaw(speed, joint) for joint in range(speed)]
                    for yaw_tuple in itertools.product(*yaw_options_per_joint):
                        course = [yaw for yaw in yaw_tuple]
                        for placement in active_ship.get_valid_placement(course):
                            actions.append(('determine_course_action', (course, placement)))

            case GamePhase.STATUS_PHASE :
                actions = [('status_phase', None)]

            case _ :
                raise ValueError(f'Unknown game phase: {self.phase}')
            
        if not actions:
            raise ValueError(f'No valid actions available in phase {self.phase}')
        return actions
    
    def apply_action(self, action : ActionType.Action) -> None:
        """
        Applies the given action to the game state.
        """
        action_str : str = self._get_action_string(action)
        self.visualize(f"\n{self.phase.name}, Player {self.current_player}\n{action_str}")

        if self.phase > GamePhase.SHIP_PHASE and self.phase < GamePhase.SQUADRON_PHASE:
            if self.active_ship is None:
                raise ValueError("No active ship for the current game phase.")
            active_ship : Ship = self.active_ship
        if self.phase > GamePhase.SHIP_ATTACK_DECLARE_TARGET and self.phase < GamePhase.SHIP_EXECUTE_MANEUVER:
            if self.attack_info is None:
                raise ValueError("No attack info for the current game phase.")
            attack_info : AttackInfo = self.attack_info
        if self.phase > GamePhase.SHIP_ATTACK_ROLL_DICE and self.phase < GamePhase.SHIP_EXECUTE_MANEUVER:
            if attack_info.attack_pool_result is None:
                raise ValueError("No attack pool result for the current game phase.")
            attack_pool_result = attack_info.attack_pool_result
        
        match action[0]:
            case 'activate_ship_action':
                ship_id = action[1]
                self.active_ship = self.ships[ship_id]
                self.phase = GamePhase.SHIP_ATTACK_DECLARE_TARGET

            case 'pass_ship_activation':
                if self.get_valid_activation(-self.current_player):
                    self.current_player *= -1
                else :
                    self.phase = GamePhase.STATUS_PHASE
                    self.active_ship = None
            
            case 'declare_target_action':
                active_ship.attack_count += 1
                active_ship.attack_possible_hull[action[1][0].value] = False
                # gather initial dice pool here
                self.attack_info = AttackInfo(active_ship, action[1][0], self.ships[action[1][1]], action[1][2])
                self.phase = GamePhase.SHIP_ATTACK_GATHER_DICE
            
            case 'gather_dice_action':
                # update dice pool considering obstruction.etc
                attack_info.attack_pool = action[1]
                self.phase = GamePhase.SHIP_ATTACK_ROLL_DICE

            case 'roll_dice_action':
                attack_info.attack_pool_result = action[1]
                self.phase = GamePhase.SHIP_ATTACK_RESOLVE_DAMAGE

            case 'resolve_damage_action':
                black_critical = bool(attack_pool_result[dice.CRIT_INDICES[0]])
                blue_critical = bool(attack_pool_result[dice.CRIT_INDICES[1]])
                red_critical = bool(attack_pool_result[dice.CRIT_INDICES[2]])
                total_damage = sum(
                    sum(damage * dice for damage, dice in zip(damage_values, dice_counts)) for damage_values, dice_counts in zip(dice.DAMAGE_INDICES, attack_pool_result)
                    )
                
                critical = None
                if black_critical or blue_critical or red_critical :
                    critical = dice.Critical.STANDARD
                self.ships[attack_info.defend_ship_id].defend(attack_info.defend_hull, total_damage, critical)
                self.phase = GamePhase.SHIP_ATTACK_DECLARE_TARGET

            case 'pass_attack':
                self.attack_info = None
                self.phase = GamePhase.SHIP_MANEUVER_DETERMINE_COURSE

            case 'determine_course_action':
                course, placement = action[1]
                active_ship.move_ship(course, placement)
                active_ship.activated = True
                self.active_ship = None
                self.phase = GamePhase.SHIP_PHASE
                self.current_player *= -1
                
            case 'status_phase':
                self.status_phase()
                if self.winner is None :
                    self.phase = GamePhase.SHIP_PHASE

        self.update_decision_player()

    def _get_action_string(self, action : ActionType.Action) -> str:
        """
        Returns a string representation of the action for visualization.
        """
        match action[0]:
            case 'activate_ship_action':
                return f'Activating {self.ships[action[1]].name}'
            case 'pass_ship_activation':
                return 'Passing ship activation'
            case 'declare_target_action':
                attack_hull, defend_ship_id, defend_hull = action[1]
                defend_ship = self.ships[defend_ship_id]
                return f'Declaring target: from {attack_hull.name} to {defend_ship.name} {defend_hull.name}'
            case 'gather_dice_action':
                return f'Gathering dice: {action[1]}'
            case 'roll_dice_action':
                return f'Rolling dice: {action[1]}'
            case 'resolve_damage_action':
                return 'Resolving damage'
            case 'pass_attack':
                return 'Passing attack'
            case 'determine_course_action':
                course, placement = action[1]
                return f'Determining course: {course}, Placement: {placement}'
            case 'status_phase':
                return 'Status phase'
            case _:
                raise ValueError(f'Unknown action type: {action[0]}')

    def deploy_ship(self, ship : Ship, x : float, y : float, orientation : float, speed : int) -> None :
        self.ships.append(ship)
        ship.deploy(self, x, y, orientation, speed, len(self.ships) - 1)
        self.visualize(f'\n{ship.name} is deployed.')



    def total_destruction(self, player : int) -> bool:
        player_ship_count = sum(1 for ship in self.ships if ship.player == player and not ship.destroyed)
        return player_ship_count == 0

    def get_valid_activation(self, player : int) -> list[Ship] :
        """
        Returns a list of ships that can be activated by the given player.
        A ship can be activated if it is not destroyed, not already activated, and belongs to the player.
        """
        return [ship for ship in self.ships if ship.player == player and not ship.destroyed and not ship.activated]
    

    def status_phase(self) -> None:
        # 1. Refresh all active ships for the next round
        for ship in self.ships:
            if not ship.destroyed:
                ship.refresh()

        # 2. Check for game-ending conditions
        player_1_eliminated = self.total_destruction(1)
        player_2_eliminated = self.total_destruction(-1)
        is_game_over = player_1_eliminated or player_2_eliminated or self.round == 6

        # 3. If the game is over, determine the winner
        if is_game_over:
            # This check is important to ensure the winner is only set once
            if self.winner is None:
                p1_points = self.get_point(1)
                p2_points = self.get_point(-1)
                margin_of_victory = p1_points - p2_points

                # Set a non-zero winner value based on the margin
                self.winner = margin_of_victory / 100
            return # Stop the function here; do not advance the round

        # 4. If the game is not over, advance to the next round
        else:
            self.round += 1
            self.current_player = 1

        

    def get_point(self, player : int) -> int :
        return sum(ship.point for ship in self.ships if ship.player != player and ship.destroyed)

    def visualize(self, title : str, maneuver_tool = None) -> None:
        if self.simulation_mode:
            return
        visualizer.visualize(self, title, maneuver_tool)

    def refresh_ship_links(self) -> None:
        """Ensures all ship objects refer to this game instance."""
        for ship in self.ships:
            ship.game = self

class AttackInfo :
    def __init__(self, attack_ship : Ship, attack_hull : HullSection, defend_ship : Ship, defend_hull : HullSection) -> None:
        self.attack_player : int = attack_ship.player
        self.attack_ship_id : int = attack_ship.ship_id
        self.attack_hull : HullSection = attack_hull
        self.defend_ship_id : int = defend_ship.ship_id
        self.defend_hull : HullSection = defend_hull
        self.attack_range :int = attack_ship.measure_arc_and_range(attack_hull, defend_ship, defend_hull)
        self.obstructed : bool = attack_ship.measure_line_of_sight(attack_hull, defend_ship, defend_hull)[1]
        self.attack_pool : list[int] = attack_ship.gather_dice(attack_hull, self.attack_range)
        self.attack_pool_result : list[list[int]] | None = None