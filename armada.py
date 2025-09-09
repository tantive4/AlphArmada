from __future__ import annotations
import random
import copy
import itertools
from typing import Callable

from shapely.geometry import Polygon

from mcts import MCTS
from game_phase import GamePhase, ActionType
import visualizer
from ship import Ship, HullSection, Command, _cached_range
from defense_token import DefenseToken, TokenType, TOKEN_DICT
from dice import Dice, CRIT_INDEX, ACCURACY_INDEX, ICON_INDICES, DAMAGE_INDICES, roll_dice, generate_all_dice_outcomes, Critical, dice_choice_combinations
from measurement import AttackRange

class Armada:
    def __init__(self) -> None:
        self.player_edge = 900 # mm
        self.short_edge = 900 # mm
        self.game_board = Polygon([
            (0,0),
            (0,self.player_edge),
            (self.short_edge, self.player_edge),
            (self.short_edge, 0),
        ])
        self.ships : list[Ship] = []

        self.round : int = 1
        self.phase = GamePhase.COMMAND_PHASE
        self.current_player : int = 1
        self.decision_player : int | None = 1
        self.active_ship : Ship | None = None
        self.attack_info : AttackInfo | None = None

        self.winner : float | None = None
        self.image_counter : int = 0
        self.simulation_mode : bool = False
        self.simulation_player : int | None = None


    def play(self, 
             player1 : Callable[[], ActionType.Action] | None = None, 
             player2 : Callable[[], ActionType.Action] | None = None, 
             max_simulation_step : int = 1000) -> float :
        """
        The main game loop with two players defined by functions.
        Each player function should return an action based on the current game state.
        """
        if not self.simulation_mode :
            mcts_game = copy.deepcopy(self)
            mcts_game.simulation_mode = True
            self.game_tree = MCTS(mcts_game)

        if player1 is None : 
            player1 = self.random_decision
        if player2 is None : 
            player2 = self.random_decision
        simulation_counter = 0

    
        while self.winner is None:
            if self.phase == GamePhase.SHIP_ATTACK_ROLL_DICE : # chance node case
                if self.attack_info is None :
                    raise ValueError("No attack info for the current game phase.")
                dice_roll = roll_dice(self.attack_info.dice_to_roll)
                action = ('roll_dice_action', dice_roll)

            elif self.decision_player == 1 : 
                action : ActionType.Action = player1()
            elif self.decision_player == -1 :
                action : ActionType.Action = player2()
            else :
                action : ActionType.Action = self.get_valid_actions()[0]

            self.apply_action(action)
            if not self.simulation_mode : 
                self.game_tree.advance_tree(action)

                if self.get_snapshot() != self.game_tree.root_game.get_snapshot() : # debugging
                    print(self.get_snapshot())
                    print(self.game_tree.root_game.get_snapshot())
                    raise ValueError


            simulation_counter += 1
            if simulation_counter >= max_simulation_step:
                raise RuntimeError(f'Maximum simulation steps reached: {max_simulation_step}\n{self.phase}')
        return self.winner

    def random_decision(self) -> ActionType.Action:
        """
        A simple strategy that returns a random action from the list of possible actions.
        """
        actions = self.get_valid_actions()
        if not self.simulation_mode : 
            with open('simulation_log.txt', 'a') as f:
                f.write(f"{actions}\n")
        return random.choice(actions)


    def mcts_decision(self, iterations : int = 800) -> ActionType.Action:
        self.game_tree.root_game.simulation_player = self.current_player
        self.game_tree.mcts_search(iterations)
        action : ActionType.Action = self.game_tree.get_best_action()
        self.game_tree.root_game.simulation_player = None
        return action

    def alpha_mcts_decision(self, iterations : int = 800) -> ActionType.Action:
        self.game_tree.root_game.simulation_player = self.current_player
        self.game_tree.alpha_mcts_search(iterations)
        action : ActionType.Action = self.game_tree.get_best_action()
        self.game_tree.root_game.simulation_player = None
        return action

    def player_decision(self) -> ActionType.Action:
        actions = self.get_valid_actions()
        if len(actions) == 1 : return actions[0]
        for i, action in enumerate(actions,1) :
            print(f'{i} : {ActionType.get_action_str(self, action)}')
        player_index = int(input('Enter the action index (1, 2, 3 ...) : '))
        return actions[player_index - 1]

    
    def update_decision_player(self) -> None:
        """
        Returns a list of possible actions based on the current game phase.
        """
        if self.phase > GamePhase.SHIP_ATTACK_DECLARE_TARGET and self.phase < GamePhase.SHIP_EXECUTE_MANEUVER:
            if self.attack_info is None:
                raise ValueError('No attack info for the current game phase.')
            attack_info : AttackInfo = self.attack_info

        match self.phase:
            case GamePhase.COMMAND_PHASE :
                self.decision_player = self.current_player

            case GamePhase.SHIP_PHASE :
                self.decision_player = self.current_player
            case GamePhase.SHIP_REVEAL_COMMAND_DIAL :
                self.decision_player = self.current_player
            case GamePhase.SHIP_GAIN_COMMAND_TOKEN :
                self.decision_player = self.current_player
            case GamePhase.SHIP_DISCARD_COMMAND_TOKEN :
                self.decision_player = self.current_player

            case GamePhase.SHIP_RESOLVE_REPAIR :
                self.decision_player = self.current_player
            case GamePhase.SHIP_USE_ENGINEER_POINT :
                self.decision_player = self.current_player

            case GamePhase.SHIP_ATTACK_DECLARE_TARGET :
                self.decision_player = self.current_player
            case GamePhase.SHIP_ATTACK_GATHER_DICE :
                self.decision_player = attack_info.attack_player
            case GamePhase.SHIP_ATTACK_ROLL_DICE :
                self.decision_player = None
            case GamePhase.SHIP_ATTACK_RESOLVE_EFFECTS :
                self.decision_player = attack_info.attack_player
            case GamePhase.SHIP_ATTACK_SPEND_DEFENSE_TOKENS :
                self.decision_player = -attack_info.attack_player 
            case GamePhase.SHIP_ATTACK_USE_CRITICAL_EFFECT :
                self.decision_player = attack_info.attack_player
            case GamePhase.SHIP_ATTACK_RESOLVE_DAMAGE :
                self.decision_player = -attack_info.attack_player

            case GamePhase.SHIP_MANEUVER_DETERMINE_COURSE :
                self.decision_player = self.current_player

            case GamePhase.STATUS_PHASE :
                self.decision_player = None

            case _ :
                raise ValueError(f'Unknown game phase: {self.phase.name}')
    
    def get_valid_actions(self) -> list[ActionType.Action]:
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
            attack_pool_result = attack_info.attack_pool_result


        match self.phase:
            case GamePhase.COMMAND_PHASE :
                if self.winner is not None:
                    raise ValueError('Game has already ended.')
                ships_to_command = [ship.ship_id for ship in self.ships if ship.player == self.current_player and len(ship.command_stack) < ship.command_value]
                if ships_to_command : 
                    actions = [('set_command_action', (ship_id, command)) for command in Command for ship_id in ships_to_command]
                else :
                    actions = [('pass_command', None)]
            
            case GamePhase.SHIP_PHASE :
                valid_ships = self.get_valid_activation(self.current_player)
                if valid_ships : 
                    actions = [('activate_ship_action', ship.ship_id) for ship in valid_ships]
                else :
                    actions = [('pass_ship_activation', None)]
            
            # Reveal Command Sequence
            case GamePhase.SHIP_REVEAL_COMMAND_DIAL :
                if not self.simulation_mode or self.current_player == self.simulation_player :  # real game or player's simulation
                    actions = [('reveal_command_action', active_ship.command_stack[0])]
                else :                                                                          # secret information
                    actions = [('reveal_command_action', command) for command in Command]

            case GamePhase.SHIP_GAIN_COMMAND_TOKEN :
                actions = [('gain_command_token_action', command) for command in active_ship.command_dial if command not in active_ship.command_token]
                actions.append(('pass_command_token', None))

            case GamePhase.SHIP_DISCARD_COMMAND_TOKEN :
                actions = [('discard_command_token_action', command) for command in active_ship.command_token]


            # Engineering Sequence
            case GamePhase.SHIP_RESOLVE_REPAIR :
                dial = Command.REPAIR in active_ship.command_dial
                token_choices = [True, False] if Command.REPAIR in active_ship.command_token else [False]
                actions = [('resolve_repair_command_action', (dial, token)) for token in token_choices]

            case GamePhase.SHIP_USE_ENGINEER_POINT :
                if active_ship.engineer_point >= 3 and active_ship.hull < active_ship.max_hull :
                    actions.append(('repair_hull_action', None))
                if active_ship.engineer_point >= 2 :
                    actions.extend([('recover_shield_action', hull) for hull in HullSection if active_ship.shield[hull] < active_ship.max_shield[hull]])
                if not actions : actions = [('pass_repair', None)]

                if active_ship.engineer_point >= 1 :
                    actions.extend([('move_shield_action', (from_hull, to_hull)) for from_hull in HullSection for to_hull in HullSection
                                    if active_ship.shield[to_hull] < active_ship.max_shield[to_hull] and active_ship.shield[from_hull] > 0 and
                                        from_hull != to_hull and
                                        not from_hull in active_ship.repaired_hull])
            
            # Attack Sequence
            case GamePhase.SHIP_ATTACK_DECLARE_TARGET :
                actions = [('declare_target_action', (attack_hull, defend_ship.ship_id, defend_hull))
                    for attack_hull in active_ship.get_valid_attack_hull()
                    for defend_ship, defend_hull in active_ship.get_valid_target(attack_hull)]
                if not actions : actions = [('pass_attack', None)]

            case GamePhase.SHIP_ATTACK_GATHER_DICE :
                attack_ship = self.ships[attack_info.attack_ship_id]
                dice_to_roll = attack_ship.gather_dice(attack_info.attack_hull, attack_info.attack_range)

                if attack_info.obstructed:
                    if sum(dice_to_roll.values()) <= 1 : raise ValueError('Empty Attack Pool. Invalid Attack')
                    for dice_type in Dice :
                        if dice_to_roll[dice_type] > 0 :
                            dice_to_remove = {dice_type : 0 for dice_type in Dice}
                            dice_to_remove[dice_type] = 1
                            actions.append(('gather_dice_action', dice_to_remove))
                else:
                    actions = [('gather_dice_action', {dice_type : 0 for dice_type in Dice})]
            
            case GamePhase.SHIP_ATTACK_ROLL_DICE :
                actions = [('roll_dice_action', dice_roll) for dice_roll in generate_all_dice_outcomes(attack_info.dice_to_roll)]
            
            case GamePhase.SHIP_ATTACK_RESOLVE_EFFECTS :
                # spend accuracy
                blue_acc_count = attack_pool_result[Dice.BLUE][ACCURACY_INDEX[Dice.BLUE]]
                red_acc_count = attack_pool_result[Dice.RED][ACCURACY_INDEX[Dice.RED]]
                defend_ship = self.ships[attack_info.defend_ship_id]
                checked_tokens : list[DefenseToken]= []
                for index, token in defend_ship.defense_tokens.items():
                    if token in checked_tokens :
                        continue
                    checked_tokens.append(token)
                    if token.discarded or token.accuracy :
                        continue
                    if blue_acc_count : actions.append(('spend_accuracy_action', (Dice.BLUE, index)))
                    if red_acc_count : actions.append(('spend_accuracy_action', (Dice.RED, index)))


                # use con-fire command
                if attack_info.con_fire_dial :
                    actions.extend([('use_confire_dial_action', {dice: 1}) for dice in Dice if sum(attack_pool_result[dice])])

                if not actions : actions= [('pass_attack_effect', None)] # Above actions are MUST USED actions

                # use con-fire command
                if Command.CONFIRE not in active_ship.resolved_command :
                    dial = Command.CONFIRE in active_ship.command_dial
                    token = Command.CONFIRE in active_ship.command_token
                    dial_choices = [True, False] if dial else [False]
                    token_choices = [True, False] if token else [False]
                    all_combinations = itertools.product(dial_choices, token_choices)
                    for use_dial, use_token in all_combinations:
                        if not use_dial and not use_token:
                            continue  # Skip the (False, False) combination
                        actions.append(('resolve_con-fire_command_action', (use_dial, use_token)))

                if attack_info.con_fire_token and not attack_info.con_fire_dial : # Use reroll after adding dice
                    actions.extend([('use_confire_token_action', dice) for dice in dice_choice_combinations(attack_pool_result, 1)])


            case GamePhase.SHIP_ATTACK_SPEND_DEFENSE_TOKENS :
                attack_ship = self.ships[attack_info.attack_ship_id]
                defend_ship = self.ships[attack_info.defend_ship_id]
                
                if defend_ship.speed > 0 :
                    checked_tokens : list[DefenseToken]= []
                    for index, token in defend_ship.defense_tokens.items():

                        # do not double check the identical token
                        if token in checked_tokens :
                            continue
                        checked_tokens.append(token)
                        if (not token.discarded and not token.accuracy
                                and index not in attack_info.spent_token_indices
                                and token.type not in attack_info.spent_token_types):
                            
                            if token.type == TokenType.REDIRECT :
                                actions.append(('spend_redicect_token_action',(index, HullSection((attack_info.defend_hull + 1) % 4))))
                                actions.append(('spend_redicect_token_action',(index, HullSection((attack_info.defend_hull - 1) % 4))))

                            elif token.type == TokenType.EVADE :
                                # choose 1 die to affect
                                evade_dice_choices = dice_choice_combinations(attack_pool_result, 1)
                                for dice_choice in evade_dice_choices :
                                    actions.append(('spend_evade_token_action', (index, dice_choice)))
                                
                                # if defender is smaller, may choose 2 dice
                                if defend_ship.size_class < attack_ship.size_class :
                                    discard_evade_choices = dice_choice_combinations(attack_pool_result, 2)
                                    for dice_choice in discard_evade_choices :
                                        actions.append(('spend_evade_token_action', (index, dice_choice)))

                            else : actions.append(('spend_defense_token_action', index))

                actions.append(('pass_defense_token', None))

            case GamePhase.SHIP_ATTACK_USE_CRITICAL_EFFECT :
                black_crit = bool(attack_pool_result[Dice.BLACK][CRIT_INDEX[Dice.BLACK]])
                blue_crit = bool(attack_pool_result[Dice.BLUE][CRIT_INDEX[Dice.BLUE]])
                red_crit = bool(attack_pool_result[Dice.RED][CRIT_INDEX[Dice.RED]])

                attack_ship = self.ships[attack_info.attack_ship_id]
                critical_list = attack_ship.get_critical_effect(black_crit, blue_crit, red_crit)
                if critical_list :
                    actions = [('use_critical_action', critical) for critical in critical_list]
                else :
                    actions = [('pass_critical', None)]


            case GamePhase.SHIP_ATTACK_RESOLVE_DAMAGE:
                defend_ship = self.ships[attack_info.defend_ship_id]
                total_damage = attack_info.total_damage
                redirect_hulls = attack_info.redirect_hulls

                # Helper function to recursively find all combinations
                def find_redirect_combinations(hull_index, remaining_damage, current_combination):
                    # Base case: If we have processed all hulls, this combination is complete.
                    if hull_index == len(redirect_hulls):
                        actions.append(('resolve_damage_action', current_combination))
                        return

                    current_hull = redirect_hulls[hull_index]
                    max_shield = defend_ship.shield[current_hull]
                    
                    # Iterate through all possible damage amounts for the current hull.
                    for damage in range(min(remaining_damage, max_shield) + 1):
                        # Create the next combination by adding the current choice
                        next_combination = current_combination + [(current_hull, damage)]
                        # Recurse for the next hull with the updated remaining damage
                        find_redirect_combinations(hull_index + 1, remaining_damage - damage, next_combination)

                find_redirect_combinations(0, total_damage, [])
                

            # Maneuver Sequence
            case GamePhase.SHIP_MANEUVER_DETERMINE_COURSE:
               for speed in active_ship.get_valid_speed():
                    all_courses = active_ship.get_all_possible_courses(speed)

                    for course in all_courses:
                        for placement in active_ship.get_valid_placement(course):
                            actions.append(('determine_course_action', (course, placement)))

            case GamePhase.STATUS_PHASE :
                actions = [('status_phase', None)]

            case _ :
                raise ValueError(f'Unknown game phase: {self.phase.name}')
            
        if not actions:
            raise ValueError(f'No valid actions available in phase {self.phase.name}\n{self.get_snapshot()}')
        return actions
    
    
    def apply_action(self, action : ActionType.Action) -> None:
        """
        Applies the given action to the game state.
        """
        if self.phase > GamePhase.SHIP_PHASE and self.phase < GamePhase.SQUADRON_PHASE:
            if self.active_ship is None:
                raise ValueError("No active ship for the current game phase.")
            active_ship : Ship = self.active_ship
        if self.phase > GamePhase.SHIP_ATTACK_DECLARE_TARGET and self.phase < GamePhase.SHIP_EXECUTE_MANEUVER:
            if self.attack_info is None:
                raise ValueError("No attack info for the current game phase.")
            attack_info= self.attack_info
            attack_pool_result = attack_info.attack_pool_result

        self.visualize_action(action)
        
        match action:
            case 'set_command_action', (ship_id, command):
                command_ship = self.ships[ship_id]
                command_ship.command_stack.append(command)
                if [ship.ship_id for ship in self.ships if ship.player == self.current_player and len(ship.command_stack) < ship.command_value] :
                    self.phase = GamePhase.COMMAND_PHASE
                else :
                    if self.current_player == 1 : 
                        self.current_player = -1
                        self.phase = GamePhase.COMMAND_PHASE
                    else :
                        self.current_player = 1
                        self.phase = GamePhase.SHIP_PHASE
            
            case 'pass_command', _ :
                self.current_player = 1
                self.phase = GamePhase.SHIP_PHASE

            case 'activate_ship_action', ship_id:
                self.active_ship = self.ships[ship_id]
                self.phase = GamePhase.SHIP_REVEAL_COMMAND_DIAL

            case 'pass_ship_activation', _ :
                if self.get_valid_activation(-self.current_player):
                    self.current_player *= -1
                else :
                    self.phase = GamePhase.STATUS_PHASE
                    self.active_ship = None

            case 'reveal_command_action', command :
                active_ship.command_stack.pop()
                active_ship.command_dial.append(command)
                self.phase = GamePhase.SHIP_GAIN_COMMAND_TOKEN

            case 'gain_command_token_action', command :
                active_ship.command_dial.remove(command)
                active_ship.command_token.append(command)

                if len(active_ship.command_token) > active_ship.command_value :
                    self.phase = GamePhase.SHIP_DISCARD_COMMAND_TOKEN
                elif Command.REPAIR in active_ship.command_dial + active_ship.command_token :
                    self.phase = GamePhase.SHIP_RESOLVE_REPAIR
                else :
                    self.phase = GamePhase.SHIP_ATTACK_DECLARE_TARGET

            case 'pass_command_token', _ :
                if Command.REPAIR in active_ship.command_dial + active_ship.command_token :
                    self.phase = GamePhase.SHIP_RESOLVE_REPAIR
                else :
                    self.phase = GamePhase.SHIP_ATTACK_DECLARE_TARGET

            case 'discard_command_token_action', command :
                active_ship.command_token.remove(command)
                if len(active_ship.command_token) > active_ship.command_value :
                    self.phase = GamePhase.SHIP_DISCARD_COMMAND_TOKEN
                elif Command.REPAIR in active_ship.command_dial + active_ship.command_token :
                    self.phase = GamePhase.SHIP_RESOLVE_REPAIR
                else :
                    self.phase = GamePhase.SHIP_ATTACK_DECLARE_TARGET

            case 'resolve_repair_command_action', (dial, token) :
                if dial : active_ship.command_dial.remove(Command.REPAIR)
                if token : active_ship.command_token.remove(Command.REPAIR)
                if dial or token :
                    active_ship.resolved_command.append(Command.REPAIR)
                    active_ship.engineer_point = dial * active_ship.engineer_value + token * (active_ship.engineer_value + 1) // 2
                    self.phase = GamePhase.SHIP_USE_ENGINEER_POINT
                else : 
                    self.phase = GamePhase.SHIP_ATTACK_DECLARE_TARGET

            case 'repair_hull_action', _ :
                active_ship.engineer_point -= 3
                active_ship.hull += 1

                self.phase = GamePhase.SHIP_USE_ENGINEER_POINT

            case 'recover_shield_action', hull:
                active_ship.engineer_point -= 2
                active_ship.shield[hull] += 1
                active_ship.repaired_hull.append(hull)

                self.phase = GamePhase.SHIP_USE_ENGINEER_POINT

            case 'move_shield_action', (from_hull, to_hull) :
                active_ship.engineer_point -= 1
                active_ship.shield[from_hull] -= 1
                active_ship.shield[to_hull] += 1
                active_ship.repaired_hull.append(to_hull)

                self.phase = GamePhase.SHIP_USE_ENGINEER_POINT

            case 'pass_repair', _ :
                active_ship.repaired_hull = []
                active_ship.engineer_point = 0
                self.phase = GamePhase.SHIP_ATTACK_DECLARE_TARGET

            case 'declare_target_action', (attack_hull, defend_ship_id, defend_hull) :
                active_ship.attack_count += 1
                active_ship.attack_possible_hull[attack_hull.value] = False

                # gather initial dice pool here
                self.attack_info = AttackInfo(active_ship, attack_hull, self.ships[defend_ship_id], defend_hull)
                self.phase = GamePhase.SHIP_ATTACK_GATHER_DICE
            
            case 'gather_dice_action', dice_to_remove:
                # update dice pool considering obstruction.etc
                attack_info.dice_to_roll = {dice_type : attack_info.dice_to_roll[dice_type] - dice_to_remove[dice_type] for dice_type in Dice}
                self.phase = GamePhase.SHIP_ATTACK_ROLL_DICE

            case 'roll_dice_action', dice_roll:
                attack_info.dice_to_roll = {dice_type : 0 for dice_type in Dice}
                for dice_type in Dice :
                    current_results = attack_info.attack_pool_result[dice_type]
                    new_results = dice_roll[dice_type]
                    attack_info.attack_pool_result[dice_type] = [
                        current + new for current, new in zip(current_results, new_results)
                    ]
                if attack_info.phase == GamePhase.SHIP_ATTACK_RESOLVE_EFFECTS :
                    self.phase = GamePhase.SHIP_ATTACK_RESOLVE_EFFECTS
                else : self.phase = GamePhase.SHIP_ATTACK_SPEND_DEFENSE_TOKENS

            case 'spend_accuracy_action', (dice, index):
                defend_ship = self.ships[attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]
                token.accuracy = True
                if dice == Dice.BLUE:
                    attack_pool_result[Dice.BLUE][ACCURACY_INDEX[Dice.BLUE]] -= 1
                elif dice == Dice.RED:
                    attack_pool_result[Dice.RED][ACCURACY_INDEX[Dice.RED]] -= 1
                self.phase = GamePhase.SHIP_ATTACK_RESOLVE_EFFECTS

            case 'resolve_con-fire_command_action', (dial, token) :
                active_ship.resolved_command.append(Command.CONFIRE)
                attack_info.con_fire_dial, attack_info.con_fire_token = dial, token
                self.phase = GamePhase.SHIP_ATTACK_RESOLVE_EFFECTS
            
            case 'use_confire_dial_action', dice_to_remove :
                attack_info.dice_to_roll = dice_to_remove
                attack_info.con_fire_dial = False
                self.phase = GamePhase.SHIP_ATTACK_ROLL_DICE
                
            case 'use_confire_token_action', reroll_dice :
                attack_info.con_fire_token = False
                attack_info.attack_pool_result = {dice_type : [original_count - removed_count 
                                                            for original_count, removed_count in zip(attack_pool_result[dice_type], reroll_dice[dice_type])] 
                                                            for dice_type in Dice}
                attack_info.dice_to_roll = {dice_type : sum(reroll_dice[dice_type]) for dice_type in Dice}
                self.phase = GamePhase.SHIP_ATTACK_ROLL_DICE

            case 'pass_attack_effect', _ :
                attack_info.phase = GamePhase.SHIP_ATTACK_SPEND_DEFENSE_TOKENS
                self.phase = GamePhase.SHIP_ATTACK_SPEND_DEFENSE_TOKENS

            case 'spend_defense_token_action', index :
                defend_ship = self.ships[attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]

                attack_info.spent_token_indices.append(index)
                attack_info.spent_token_types.append(token.type)
                token.spend()
                self.phase = GamePhase.SHIP_ATTACK_SPEND_DEFENSE_TOKENS

            case 'spend_redicect_token_action', (index, hull) :
                defend_ship = self.ships[attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]

                attack_info.spent_token_indices.append(index)
                attack_info.spent_token_types.append(token.type)
                token.spend()
                attack_info.redirect_hulls.append(hull)
                self.phase = GamePhase.SHIP_ATTACK_SPEND_DEFENSE_TOKENS

            case 'spend_evade_token_action', (index, evade_dice) :
                defend_ship = self.ships[attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]

                attack_info.spent_token_indices.append(index)
                attack_info.spent_token_types.append(token.type)
                if sum([sum(evade_dice[dice_type]) for dice_type in Dice]) == 2:
                    token.discard()
                else :
                    token.spend()
                attack_info.attack_pool_result = {dice_type : [original_count - removed_count 
                                                            for original_count, removed_count in zip(attack_pool_result[dice_type], evade_dice[dice_type])] 
                                                            for dice_type in Dice}

                if attack_info.attack_range in [AttackRange.CLOSE, AttackRange.MEDIUM]:
                    # Reroll Evade Dice
                    attack_info.dice_to_roll = {dice_type : sum(evade_dice[dice_type]) for dice_type in Dice}
                    self.phase = GamePhase.SHIP_ATTACK_ROLL_DICE

                else:
                    self.phase = GamePhase.SHIP_ATTACK_SPEND_DEFENSE_TOKENS

            case 'pass_defense_token', _ :
                defend_ship = self.ships[attack_info.defend_ship_id]
                for token in defend_ship.defense_tokens.values() :
                    token.accuracy = False
                self.phase = GamePhase.SHIP_ATTACK_USE_CRITICAL_EFFECT

            case 'use_critical_action', critical :
                attack_info.critical = critical
                attack_info.calculate_total_damage()
                self.phase = GamePhase.SHIP_ATTACK_RESOLVE_DAMAGE

            case 'pass_critical', _ :
                attack_info.calculate_total_damage()
                self.phase = GamePhase.SHIP_ATTACK_RESOLVE_DAMAGE

            case 'resolve_damage_action', redirect_damge:
                defend_ship = self.ships[attack_info.defend_ship_id]
                total_damage = attack_info.total_damage
                # Redirect
                for hull, damage in redirect_damge :
                    defend_ship.shield[hull] -= damage
                    total_damage -= damage

                defend_ship.defend(attack_info.defend_hull, total_damage, attack_info.critical)
                self.phase = GamePhase.SHIP_ATTACK_DECLARE_TARGET
                self.attack_info = None

            case 'pass_attack', _:
                self.phase = GamePhase.SHIP_MANEUVER_DETERMINE_COURSE

            case 'determine_course_action', (course, placement):
                dial_used, token_used = active_ship.nav_command_used(course)
                if dial_used:
                    active_ship.command_dial.remove(Command.NAV)
                if token_used:
                    active_ship.command_token.remove(Command.NAV)

                active_ship.speed = len(course)
                active_ship.move_ship(course, placement)
                active_ship.end_activation()
                self.active_ship = None

                if self.get_valid_activation(self.current_player) or self.get_valid_activation(-self.current_player) :
                    self.phase = GamePhase.SHIP_PHASE
                    self.current_player *= -1
                else:
                    self.phase = GamePhase.STATUS_PHASE
                
            case 'status_phase', _:
                self.status_phase()
                if self.winner is None :
                    self.phase = GamePhase.COMMAND_PHASE

            case _ :
                raise ValueError(f'Unknown Action {action}')




        # decision player for NEXT PHASE (after applying action)
        self.update_decision_player()

    def visualize_action(self, action : ActionType.Action) -> None:
        """
        get action string and call visualizer
        """
        if self.simulation_mode : return

        action_str = ActionType.get_action_str(self, action)
        if action_str is None : return

        maneuver_tool = None
        if action[0] == 'determine_course_action':
            course, placement = action[1]
            if self.active_ship is None : raise ValueError('Need active ship to perform maneuver')
            maneuver_tool, _ = self.active_ship._tool_coordination(course, placement)

        self.visualize(f"Round {self.round} | {self.phase.name.replace("_"," ").title()} | Player {self.current_player}\n{action_str}", maneuver_tool)

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
        if not self.simulation_mode :
            print(f'\n{'-' * 10} Round {self.round} Ended {'-' * 10}\n')
            with open('simulation_log.txt', 'a') as f: f.write(f'\n{'-' * 10} Round {self.round} Ended {'-' * 10}\n\n')
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

    def visualize(self, title : str, maneuver_tool : list[tuple[float, float]] | None = None) -> None:
        if self.simulation_mode:
            return
        visualizer.visualize(self, title, maneuver_tool)

    def get_encoded_state(self) -> None :
        return

    def get_snapshot(self) -> dict:
        """Captures the essential state of the entire game."""
        return {
            "winner": self.winner,
            "round": self.round,
            "phase": self.phase,
            "current_player": self.current_player,
            "decision_player": self.decision_player,
            "active_ship_id": self.active_ship.ship_id if self.active_ship else None,
            "attack_info": copy.deepcopy(self.attack_info),
            "ship_states": [ship.get_snapshot() for ship in self.ships]
        }

    def revert_snapshot(self, snapshot: dict) -> None:
        """Restores the entire game state from a snapshot."""
        self.winner = snapshot["winner"]
        self.round = snapshot["round"]
        self.phase = snapshot["phase"]
        self.current_player = snapshot["current_player"]
        self.decision_player = snapshot["decision_player"]
        self.attack_info = copy.deepcopy(snapshot["attack_info"])
        
        active_ship_id = snapshot["active_ship_id"]
        self.active_ship = self.ships[active_ship_id] if active_ship_id is not None else None

        for i, ship_snapshot in enumerate(snapshot["ship_states"]):
            self.ships[i].revert_snapshot(ship_snapshot)



class AttackInfo :
    def __init__(self, attack_ship : Ship, attack_hull : HullSection, defend_ship : Ship, defend_hull : HullSection) -> None :
        self.attack_player : int = attack_ship.player
        self.attack_ship_id : int = attack_ship.ship_id
        self.attack_hull : HullSection = attack_hull
        self.defend_ship_id : int = defend_ship.ship_id
        self.defend_hull : HullSection = defend_hull
        self.attack_range : AttackRange = _cached_range(attack_ship.get_ship_hash_state(), defend_ship.get_ship_hash_state())[1][attack_hull][defend_hull]
        self.obstructed : bool = attack_ship.measure_line_of_sight(attack_hull, defend_ship, defend_hull)

        if self.attack_range == AttackRange.INVALID : raise ValueError('Cannot Declare Target of Invalid Range')

        self.dice_to_roll : dict[Dice, int] = attack_ship.gather_dice(attack_hull, self.attack_range)
        self.attack_pool_result : dict[Dice, list[int]]  = {dice_type : [0 for _ in ICON_INDICES[dice_type]] for dice_type in Dice}
        self.con_fire_dial = False
        self.con_fire_token = False

        self.dice_choice_action : ActionType.Action | None = None

        self.phase : GamePhase = GamePhase.SHIP_ATTACK_RESOLVE_EFFECTS

        self.spent_token_indices : list[int] = []
        self.spent_token_types : list[TokenType] = []
        self.redirect_hulls : list[HullSection] = []

        self.critical : Critical | None = None

    def __eq__(self,other) -> bool :
        if not isinstance(other, AttackInfo) : return False
        return self.__dict__ == other.__dict__
    
    def __str__(self) -> str :
        return str(self.__dict__)
    __repr__ = __str__
    



    def calculate_total_damage(self) -> int :
        total_damage = 0
        for dice_type in Dice :
            total_damage += sum([face_count * damage_value for face_count, damage_value in zip(self.attack_pool_result[dice_type], DAMAGE_INDICES[dice_type])])

        if TokenType.BRACE in self.spent_token_types :
            total_damage = (total_damage+1) // 2

        self.total_damage = total_damage
        return total_damage
