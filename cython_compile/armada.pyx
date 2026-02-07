# cython: profile=True

from __future__ import annotations
import random
import math

import numpy as np
cimport numpy as cnp
cnp.import_array()
from skimage.draw import polygon as draw_polygon

from configs import Config
from action_phase import Phase, ActionType, get_action_str, phase_type
import visualizer
from ship cimport Ship
from squad cimport Squad
from obstacle cimport Obstacle
from attack_info cimport AttackInfo
from enum_class import *
from dice import *
from measurement import *
import cache_function as cache
from defense_token cimport DefenseToken

cdef class Armada:
    """
    The main class representing the Armada game.
    """
    def __init__(self, faction: tuple[Faction, Faction], para_index:int = 0) -> None:
        self.player_edge = LONG_RANGE * 6
        self.short_edge = LONG_RANGE * 3
        self.ships : list[Ship] = []
        self.squads : list[Squad] = []
        self.obstacles : list[Obstacle] = []

        self.round = 1
        self.first_player = 1
        self.second_player = -1
        self.first_faction = faction[0].value
        self.second_faction = faction[1].value

        self.phase : Phase = Phase.SHIP_ACTIVATE    
        self.current_player : int = self.first_player
        self.decision_player : int = self.first_player
        self.active_ship : Ship | None = None
        self.defend_ship : Ship | None = None
        self.active_squad : Squad | None = None
        self.attack_info : AttackInfo | None = None
        self.squad_activation_count = 0

        self.winner = 0.0
        self.image_counter = 0
        self.debuging_visual = <bint>False
        self.simulation_player = 0
        self.para_index = para_index

        self.scalar_encode_array = np.zeros(Config.SCALAR_FEATURE_SIZE, dtype=np.float32)
        self.relation_encode_array = np.zeros((Config.MAX_SHIPS, Config.MAX_SHIPS, hull_type * hull_type + 4), dtype=np.float32)
        self.ship_encode_array = np.zeros((Config.MAX_SHIPS, Config.SHIP_ENTITY_FEATURE_SIZE), dtype=np.float32)
        self.ship_coords_array = np.zeros((Config.MAX_SHIPS, 3), dtype=np.float32)
        self.ship_def_token_array = np.zeros((Config.MAX_SHIPS, Config.MAX_DEFENSE_TOKENS, Config.DEF_TOKEN_FEATURE_SIZE), dtype=np.float32)
        
        h, w = Config.BOARD_RESOLUTION
        self.spatial_encode_array = np.zeros((Config.MAX_SHIPS, 10, h, w//8), dtype=np.uint8) # ships , presence+hull_type*range_type, height, width

    def rollout(self, max_simulation_step : int = 2000) -> float :
        """
        The main game loop with two players defined by functions.
        Each player function should return an action based on the current game state.
        """

        player1 = self.random_decision
        player2 = self.random_decision
        simulation_counter = 0

    
        while self.winner == 0.0:
            if self.phase == Phase.ATTACK_ROLL_DICE : # chance node case
                if self.attack_info is None :
                    raise ValueError("No attack info for the current game phase.")
                dice_roll = roll_dice(self.attack_info.dice_to_roll)
                action = ('roll_dice_action', dice_roll)

            elif self.decision_player == 1 : 
                action : ActionType = player1()
            elif self.decision_player == -1 :
                action : ActionType = player2()
            else :
                action : ActionType = self.get_valid_actions()[0]
            
            # print(get_action_str(self, action))
            
            self.apply_action(action)

            simulation_counter += 1
            if simulation_counter >= max_simulation_step:
                raise RuntimeError(f'Maximum simulation steps reached: {max_simulation_step}\n{self.phase}\n{self.get_snapshot()}')
        return self.winner

    def random_decision(self) -> ActionType:
        """
        A simple strategy that returns a random action from the list of possible actions.
        """
        actions = self.get_valid_actions()
        if self.debuging_visual : 
            with open('simulation_log.txt', 'a') as f:
                f.write(f"{actions}\n")
        return random.choice(actions)

    def player_decision(self) -> ActionType:
        actions = self.get_valid_actions()
        if len(actions) == 1 : return actions[0]
        for i, action in enumerate(actions,1) :
            print(f'{i} : {get_action_str(self, action)}')
        player_index = int(input('Enter the action index (1, 2, 3 ...) : '))
        return actions[player_index - 1]

    
    cdef void update_decision_player(self):
        """
        Returns a list of possible actions based on the current game phase.
        """

        # if self.phase in (Phase.SHIP_REVEAL_COMMAND_DIAL, Phase.ATTACK_ROLL_DICE):
        # Simplified
        if self.phase in (Phase.ATTACK_ROLL_DICE,) :
            self.decision_player = 0
        elif self.phase in (Phase.ATTACK_SPEND_DEFENSE_TOKENS, Phase.ATTACK_RESOLVE_DAMAGE): #, Phase.SHIP_PLACE_SQUAD):
            self.decision_player = -self.current_player
        else :
            self.decision_player = self.current_player

    cpdef list get_valid_actions(self) :
        """
        Returns a list of possible actions based on the current game phase.
        """
        cdef: 
            list actions = []
            Ship active_ship, attack_ship, ship, defend_ship
            Squad active_squad, attack_squad, squad
            AttackInfo attack_info
            int phase = self.phase
            int command
            bint dial, token
            int hull, from_hull, to_hull, attack_hull, defend_hull
            DefenseToken defense_token

        if self.active_ship is not None : active_ship = self.active_ship
        if self.defend_ship is not None : defend_ship = self.defend_ship
        if self.active_squad is not None : active_squad = self.active_squad
        if self.attack_info is not None : attack_info = self.attack_info

        # if phase == Phase.COMMAND_PHASE :
        #     for ship in self.ships:
        #         if ship.team == self.current_player and len(ship.command_stack) < ship.command_value :
        #             for command in COMMANDS :
        #                 actions.append(('set_command_action', (ship.id, command)))
        
        if phase == Phase.SHIP_ACTIVATE :
            for ship in self.get_valid_ship_activation(self.current_player) :
                actions.append(('activate_ship_action', ship.id))
        
        # Reveal Command Sequence
        elif phase == Phase.SHIP_REVEAL_COMMAND_DIAL :
            for command in COMMANDS:
                actions.append(('reveal_command_action', command))

            # simplified
            # if active_ship.team == self.simulation_player or self.simulation_player == 0 :  # player's simulation
            #     actions = [('reveal_command_action', active_ship.command_stack[0])]
            # else : 
            #     for command in COMMANDS:                                                      # secret information
            #         actions.append(('reveal_command_action', command))

        # elif phase == Phase.SHIP_GAIN_COMMAND_TOKEN :
        #     for command in active_ship.command_dial:
        #         if command not in active_ship.command_token:
        #             actions.append(('gain_command_token_action', command))
        #     actions.append(('pass_command_token', None))

        # elif phase == Phase.SHIP_DISCARD_COMMAND_TOKEN :
        #     for command in active_ship.command_token:
        #         actions.append(('discard_command_token_action', command))


        # Squad Command
        # elif phase == Phase.SHIP_RESOLVE_SQUAD :
        #     dial = Command.SQUAD in active_ship.command_dial
        #     token_choices = [True, False] if Command.SQUAD in active_ship.command_token else [False]
        #     for token in token_choices:
        #         actions.append(('resolve_squad_command_action', (dial, token)))


        # Engineering Sequence
        # elif phase == Phase.SHIP_RESOLVE_REPAIR :
        #     dial = Command.REPAIR in active_ship.command_dial
        #     token_choices = [True, False] if Command.REPAIR in active_ship.command_token else [False]
        #     for token in token_choices:
        #         actions.append(('resolve_repair_command_action', (dial, token)))

        elif phase == Phase.SHIP_USE_ENGINEER_POINT :
            if active_ship.engineer_point >= 3 and active_ship.hull < active_ship.max_hull :
                actions.append(('repair_hull_action', None))
            if active_ship.engineer_point >= 2 :
                for hull in HULL_SECTIONS :
                    if active_ship.shield[hull] < active_ship.max_shield[hull] :
                        actions.append(('recover_shield_action', hull))
            if not actions : actions.append(('pass_repair', None))

            if active_ship.engineer_point >= 1 :
                for from_hull in HULL_SECTIONS:
                    for to_hull in HULL_SECTIONS:
                        if active_ship.shield[to_hull] < active_ship.max_shield[to_hull] and active_ship.shield[from_hull] > 0 and \
                                from_hull != to_hull and \
                                not from_hull in active_ship.repaired_hull:
                            actions.append(('move_shield_action', (from_hull, to_hull)))

        # Attack Sequence
        elif phase == Phase.SHIP_CHOOSE_TARGET_SHIP :
            for attack_hull in active_ship.get_valid_attack_hull():
                for defend_ship, defend_hull in active_ship.get_valid_ship_target(attack_hull):
                    actions.append(('choose_target_ship_action', defend_ship.id))
            if not actions : actions = [('pass_attack', None)]

        elif phase == Phase.SHIP_DECLARE_TARGET :
            for attack_hull in active_ship.get_valid_attack_hull():
                for defend_hull in active_ship.get_valid_target_hull(attack_hull, defend_ship):
                    actions.append(('declare_target_action', (attack_hull, defend_hull)))
                # simplified
                # for squad in active_ship.get_valid_squad_target(attack_hull):
                #     actions.append(('declare_target_action', (attack_hull, squad.id)))

        # elif phase == Phase.ATTACK_GATHER_DICE :
        #     dice_to_roll = attack_info.dice_to_roll

        #     if attack_info.obstructed:
        #         if sum(dice_to_roll) <= 1 : print(f"WARNING: empty attack pool \n{self.get_snapshot()}\n{attack_info.get_snapshot()}")
        #         for dice_type in DICE :
        #             if dice_to_roll[dice_type] > 0 :
        #                 dice_to_remove = DICE_CHOICE_1[dice_type]
        #                 actions.append(('gather_dice_action', dice_to_remove))
        #     else:
        #         actions = [('gather_dice_action', (0,0,0))]
        
        elif phase == Phase.ATTACK_ROLL_DICE :
            raise NotImplementedError("This is a chance node. No player action available.")

        elif phase == Phase.ATTACK_RESOLVE_EFFECTS :

            if attack_info.is_attacker_ship :
                attack_ship :Ship = self.ships[attack_info.attack_ship_id]
            else :
                attack_squad :Squad= self.squads[attack_info.attack_squad_id]
            if attack_info.is_defender_ship:
                defender = <Ship>self.ships[attack_info.defend_ship_id]
            else :
                defender = <Squad>self.squads[attack_info.defend_squad_id]

            # spend accuracy
            blue_acc_count = attack_info.attack_pool_result[Dice.BLUE][ACCURACY_INDEX[Dice.BLUE]]
            red_acc_count = attack_info.attack_pool_result[Dice.RED][ACCURACY_INDEX[Dice.RED]]

            for index, defense_token in enumerate(defender.defense_tokens):
                if defense_token.discarded or defense_token.accuracy :
                    continue
                if blue_acc_count or red_acc_count : actions.append(('spend_accuracy_action', index))


            # use con-fire command
            # if attack_info.con_fire_dial :
            if Command.CONFIRE in attack_ship.command_dial and Command.CONFIRE not in attack_ship.resolved_command :
                for dice in Dice:
                    if sum(attack_info.attack_pool_result[dice]) :
                        actions.append(('use_confire_dial_action', DICE_CHOICE_1[dice]))

            actions.append(('pass_attack_effect', None)) # Above actions are MUST USED actions

            # simplified
            # # use con-fire command
            # if attack_info.is_attacker_ship and Command.CONFIRE not in attack_ship.resolved_command :
            #     dial = Command.CONFIRE in attack_ship.command_dial
            #     token = Command.CONFIRE in attack_ship.command_token
            #     if dial and token:
            #         actions.append(('resolve_con-fire_command_action', (True, True)))
            #     if dial:
            #         actions.append(('resolve_con-fire_command_action', (True, False)))
            #     if token:
            #         actions.append(('resolve_con-fire_command_action', (False, True)))

            # if attack_info.con_fire_token and not attack_info.con_fire_dial : # Use reroll after adding dice
            #     actions.extend([('use_confire_token_action', dice) for dice in dice_choices(attack_info.attack_pool_result, 1)])

            # # swarm reroll
            # if attack_info.swarm:
            #     actions.extend([('swarm_reroll_action', dice) for dice in dice_choices(attack_info.attack_pool_result, 1)])

        elif phase == Phase.ATTACK_SPEND_DEFENSE_TOKENS :
            if attack_info.is_defender_ship:
                defender = <Ship>self.ships[attack_info.defend_ship_id]
            else :
                defender = <Squad>self.squads[attack_info.defend_squad_id]

            if defender.speed > 0 :
                for defense_token in defender.defense_tokens:
                    index = defense_token.id
                    if (not defense_token.discarded and not defense_token.accuracy
                            and defense_token.id not in attack_info.spent_token_indices
                            and defense_token.type not in attack_info.spent_token_types):

                        actions.append(('spend_defense_token_action', index))

            actions.append(('pass_defense_token', None))

        # elif phase == Phase.ATTACK_USE_CRITICAL_EFFECT :
        #     if (attack_info.is_attacker_ship or attack_info.bomber) and attack_info.is_defender_ship:
        #         black_crit = bool(attack_info.attack_pool_result[Dice.BLACK][CRIT_INDEX[Dice.BLACK]])
        #         blue_crit = bool(attack_info.attack_pool_result[Dice.BLUE][CRIT_INDEX[Dice.BLUE]])
        #         red_crit = bool(attack_info.attack_pool_result[Dice.RED][CRIT_INDEX[Dice.RED]])

        #         if attack_info.is_attacker_ship :
        #             attacker = <Ship>self.ships[attack_info.attack_ship_id]
        #         else :
        #             attacker = <Squad>self.squads[attack_info.attack_squad_id]
        #         for critical in attacker.get_critical_effect(black_crit, blue_crit, red_crit):
        #             actions.append(('use_critical_action', critical))

        #     if not actions :
        #         actions = [('pass_critical', None)]


        elif phase == Phase.ATTACK_RESOLVE_DAMAGE:
            total_damage = attack_info.total_damage
            redirect_hull = attack_info.redirect_hull
            if redirect_hull is not None :
                for damage in range(min(total_damage, defend_ship.shield[redirect_hull]) + 1):
                    actions.append(('resolve_damage_action', (redirect_hull, damage)))

            actions.append(('resolve_damage_action', None))

        # elif phase == Phase.ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET :
        #     attack_ship = self.ships[attack_info.attack_ship_id]
        #     attack_hull = attack_info.attack_hull
        #     for squad in attack_ship.get_valid_squad_target(attack_hull) :
        #         if squad.id not in attack_info.squadron_target :
        #             actions.append(('declare_additional_squad_target_action', squad.id))
            
        #     if not actions :
        #         actions = [('pass_additional_squad_target', None)]



        # Maneuver Sequence
        elif phase == Phase.SHIP_DETERMINE_COURSE:
            for speed in active_ship.get_valid_speed():
                all_courses = active_ship.get_all_possible_courses(speed)

                for course in all_courses:
                    for placement in active_ship.get_valid_placement(course):
                        actions.append(('determine_course_action', (course, placement)))

        # elif phase == Phase.SHIP_PLACE_SQUAD :
        #     for squad in self.squads:
        #         if squad.overlap_ship_id is None: continue
        #         for index in active_ship.get_valid_squad_placement(squad):
        #             actions.append(('place_squad_action', (squad.id, index)))

        # === SQUADRON_PHASE ===
        # elif phase == Phase.SQUAD_ACTIVATE :
        #     if self.active_ship is not None :
        #         for squad in active_ship.get_squad_activation():
        #             if not squad.is_engaged(): actions.append(('activate_squad_move_action', squad.id))
        #             if squad.get_valid_target(): actions.append(('activate_squad_attack_action', squad.id))
        #     else :
        #         for squad in self.squads:
        #             if squad.team != self.current_player or squad.activated or squad.destroyed:continue
        #             if not squad.is_engaged(): actions.append(('activate_squad_move_action', squad.id))
        #             if squad.get_valid_target(): actions.append(('activate_squad_attack_action', squad.id))
        #     if not actions : actions = [('pass_activate_squad', None)]

        # elif phase == Phase.SQUAD_MOVE :
        #     for speed, angle in active_squad.get_valid_moves():
        #         actions.append(('move_squad_action', (speed, angle)))
        #     actions.append(('pass_move_squad', None))

        # elif phase == Phase.SQUAD_DECLARE_TARGET :
        #     for target in active_squad.get_valid_target():
        #         actions.append(('declare_squad_target_action', target))
        #     if not actions : actions = [('pass_attack_squad', None)]


        else :
            raise ValueError(f'Unknown game phase: {Phase(phase)}')
            
        if not actions:
            raise ValueError(f'No valid actions available in phase {Phase(phase)}\n{self.get_snapshot()}')

        return actions
    
    
    cpdef void apply_action(self, tuple action) :
        """
        Applies the given action to the game state.
        """
        cdef:
            Ship active_ship, command_ship, attack_ship, defend_ship, ship
            Squad active_squad, defend_squad, squad
            AttackInfo attack_info
            DefenseToken defense_token
            Obstacle obstacle
            str action_type
            object action_data
            bint dial, token
            list dice_pool = []
            list dice_result_color = []
        
        if self.active_ship is not None : active_ship = self.active_ship
        if self.defend_ship is not None : defend_ship = self.defend_ship
        if self.active_squad is not None : active_squad = self.active_squad
        if self.attack_info is not None : attack_info = self.attack_info

        self.visualize_action(action)
        action_type = action[0]
        action_data = action[1]

        if action_type == 'set_command_action':
            raise NotImplementedError(f"simplified {action_type}")
            (ship_id, command) = action_data
            command_ship = self.ships[ship_id]
            command_ship.command_stack += (command,)

            needs_commands = False
            for ship in self.ships:
                    if ship.team == self.current_player and len(ship.command_stack) < ship.command_value:
                        needs_commands = True
                        break 

            if needs_commands:
                self.phase = Phase.COMMAND_PHASE
            else:
                # No ships needed commands, so move to the next player or phase
                if self.current_player == self.first_player: 
                    self.current_player *= -1
                    self.phase = Phase.COMMAND_PHASE  # Set phase for the *other* player
                else:
                    self.current_player = self.first_player
                    self.phase = Phase.SHIP_ACTIVATE


        elif action_type == 'activate_ship_action':
            ship_id = action_data
            self.active_ship = self.ships[ship_id]
            self.current_player = self.active_ship.team
            self.phase = Phase.SHIP_REVEAL_COMMAND_DIAL


        elif action_type == 'reveal_command_action':
            command = action_data
            active_ship.command_dial += (command,)
            # simplified
            if command == Command.REPAIR :
                active_ship.engineer_point = active_ship.engineer_value
            # active_ship.command_stack = active_ship.command_stack[1:]
            # self.phase = Phase.SHIP_GAIN_COMMAND_TOKEN
            self.phase = Phase.SHIP_USE_ENGINEER_POINT

        elif action_type == 'gain_command_token_action':
            raise NotImplementedError(f"simplified {action_type}")
            command = action_data

            active_ship.spend_command_dial(command)
            active_ship.command_token += (command,)

            if len(active_ship.command_token) > active_ship.command_value :
                self.phase = Phase.SHIP_DISCARD_COMMAND_TOKEN
            else :self.phase = Phase.SHIP_RESOLVE_SQUAD

        elif action_type == 'pass_command_token':
            raise NotImplementedError(f"simplified {action_type}")
            _ = action_data
            self.phase = Phase.SHIP_RESOLVE_SQUAD

        elif action_type == 'discard_command_token_action':
            raise NotImplementedError(f"simplified {action_type}")
            command = action_data
            active_ship.spend_command_token(command)
            self.phase = Phase.SHIP_RESOLVE_SQUAD

        elif action_type == 'resolve_squad_command_action':
            raise NotImplementedError(f"simplified {action_type}")
            (dial, token) = action_data
            if dial : active_ship.spend_command_dial(Command.SQUAD)
            if token : active_ship.spend_command_token(Command.SQUAD)
            if dial or token :
                active_ship.resolved_command += (Command.SQUAD,)
                self.squad_activation_count = dial * active_ship.squad_value + token * 1
                self.phase = Phase.SQUAD_ACTIVATE
            else :
                self.phase = Phase.SHIP_RESOLVE_REPAIR


        elif action_type == 'resolve_repair_command_action':
            raise NotImplementedError(f"simplified {action_type}")
            (dial, token) = action_data
            if dial : active_ship.spend_command_dial(Command.REPAIR)
            if token : active_ship.spend_command_token(Command.REPAIR)
            if dial or token :
                active_ship.resolved_command += (Command.REPAIR,)
                active_ship.engineer_point = dial * active_ship.engineer_value + token * (active_ship.engineer_value + 1) // 2
                self.phase = Phase.SHIP_USE_ENGINEER_POINT
            else : 
                self.phase = Phase.SHIP_DECLARE_TARGET

        elif action_type == 'repair_hull_action':
            _ = action_data
            active_ship.engineer_point -= 3
            active_ship.hull += 1

        elif action_type == 'recover_shield_action':
            hull= action_data
            active_ship.engineer_point -= 2

            shield_list = list(active_ship.shield)
            shield_list[hull] += 1
            active_ship.shield = tuple(shield_list)

            active_ship.repaired_hull += (hull,)


        elif action_type == 'move_shield_action':
            (from_hull, to_hull) = action_data
            active_ship.engineer_point -= 1

            shield_list = list(active_ship.shield)
            shield_list[from_hull] -= 1
            shield_list[to_hull] += 1
            active_ship.shield = tuple(shield_list)

            active_ship.repaired_hull += (to_hull,)

        elif action_type == 'pass_repair':
            _ = action_data
            active_ship.repaired_hull = ()
            active_ship.engineer_point = 0
            self.phase = Phase.SHIP_CHOOSE_TARGET_SHIP


        elif action_type == 'choose_target_ship_action':
            defend_ship_id = action_data
            self.defend_ship = self.ships[defend_ship_id]
            self.phase = Phase.SHIP_DECLARE_TARGET

        elif action_type == 'declare_target_action':
            (attack_hull, defend_hull) = action_data
            defender = (defend_ship, defend_hull)
            # gather initial dice pool here
            self.attack_info = AttackInfo((active_ship, attack_hull), defender)

            # simplified
            # self.phase = Phase.ATTACK_GATHER_DICE
            self.phase = Phase.ATTACK_ROLL_DICE
        
        elif action_type == 'gather_dice_action':
            raise NotImplementedError(f"simplified {action_type}")
            dice_to_remove= action_data
            # update dice pool considering obstruction.etc
            for dice_type in DICE:
                new_count = attack_info.dice_to_roll[dice_type] - dice_to_remove[dice_type]
                dice_pool.append(new_count)
            attack_info.dice_to_roll = tuple(dice_pool)
            self.phase = Phase.ATTACK_ROLL_DICE

        elif action_type == 'roll_dice_action':
            dice_roll= action_data
            attack_info.dice_to_roll = (0,0,0)
            for dice_type in DICE:
                dice_result_color = []
                for original, new in zip(attack_info.attack_pool_result[dice_type], dice_roll[dice_type]):
                    dice_result_color.append(original + new)
                dice_pool.append(tuple(dice_result_color))
            attack_info.attack_pool_result = tuple(dice_pool)

            attack_info.calculate_total_damage()
            self.phase = attack_info.phase # either ATTACK_RESOLVE_EFFECTS or ATTACK_SPEND_DEFENSE_TOKENS

        elif action_type == 'spend_accuracy_action':
            index= action_data
            defense_token = defend_ship.defense_tokens[index]
            defense_token.accuracy = True
            if attack_info.attack_pool_result[Dice.RED][ACCURACY_INDEX[Dice.RED]]: accuracy_dice = Dice.RED
            else: accuracy_dice = Dice.BLUE

            dice_pool = list(EMPTY_DICE_POOL)
            dice_pool[accuracy_dice] = ACCURACY_DICE_1[accuracy_dice]
            
            attack_info.remove_dice(tuple(dice_pool))

            attack_info.calculate_total_damage()

        elif action_type == 'resolve_con-fire_command_action':
            raise NotImplementedError(f"simplified {action_type}")
            (dial, token) = action_data
            if dial : active_ship.spend_command_dial(Command.CONFIRE)
            if token : active_ship.spend_command_token(Command.CONFIRE)
            active_ship.resolved_command += (Command.CONFIRE,)
            attack_info.con_fire_dial, attack_info.con_fire_token = dial, token
            attack_info.calculate_total_damage()
        
        elif action_type == 'use_confire_dial_action':
            # simplified
            active_ship.spend_command_dial(Command.CONFIRE)
            active_ship.resolved_command += (Command.CONFIRE,)

            dice_to_add = action_data
            attack_info.dice_to_roll = dice_to_add
            attack_info.con_fire_dial = False
            attack_info.calculate_total_damage()
            self.phase = Phase.ATTACK_ROLL_DICE
            
        elif action_type == 'use_confire_token_action':
            raise NotImplementedError(f"simplified {action_type}")
            reroll_dice = action_data
            attack_info.con_fire_token = False

            attack_info.remove_dice(reroll_dice)

            attack_info.dice_to_roll = tuple([sum(reroll_dice[dice_type]) for dice_type in DICE])
            attack_info.calculate_total_damage()
            self.phase = Phase.ATTACK_ROLL_DICE

        elif action_type == 'swarm_reroll_action':
            raise NotImplementedError(f"simplified {action_type}")
            reroll_dice = action_data
            attack_info.swarm = False
            attack_info.remove_dice(reroll_dice)
            for dice_type in DICE:
                dice_pool.append(sum(reroll_dice[dice_type]))
            attack_info.dice_to_roll = tuple(dice_pool)
            attack_info.calculate_total_damage()
            self.phase = Phase.ATTACK_ROLL_DICE

        elif action_type == 'pass_attack_effect':
            _ = action_data
            attack_info.phase = Phase.ATTACK_SPEND_DEFENSE_TOKENS
            self.phase = Phase.ATTACK_SPEND_DEFENSE_TOKENS

        elif action_type == 'spend_defense_token_action':
            index = action_data
            defense_token = defend_ship.defense_tokens[index]

            attack_info.spent_token_indices += (index,)
            attack_info.spent_token_types += (defense_token.type,)
            defense_token.spend()
            attack_info.calculate_total_damage()

            if defense_token.type == TokenType.REDIRECT:
                hull = attack_info.defend_hull
                idx_left = (hull - 1) & 3
                idx_right = (hull + 1) & 3
                if defend_ship.shield[idx_left] > defend_ship.shield[idx_right] :
                    attack_info.redirect_hull = idx_left
                else:
                    attack_info.redirect_hull = idx_right

            elif defense_token.type == TokenType.EVADE:
                evade_dice = fast_dice_choice(attack_info.attack_pool_result)

                attack_info.remove_dice(evade_dice)

                if attack_info.attack_range in [AttackRange.CLOSE, AttackRange.MEDIUM]:
                    # Reroll Evade Dice
                    for dice_type in DICE:
                        dice_pool.append(sum(evade_dice[dice_type]))
                    attack_info.dice_to_roll = tuple(dice_pool)
                    self.phase = Phase.ATTACK_ROLL_DICE

        elif action_type == 'spend_redirect_token_action':
            raise NotImplementedError(f"simplified {action_type}")
            (index, hull) = action_data
            defense_token = defend_ship.defense_tokens[index]

            attack_info.spent_token_indices += (index,)
            attack_info.spent_token_types += (defense_token.type,)
            defense_token.spend()
            attack_info.redirect_hull = hull
            attack_info.calculate_total_damage()

        elif action_type == 'spend_evade_token_action':
            raise NotImplementedError(f"simplified {action_type}")
            (index, evade_dice) = action_data
            defense_token = defend_ship.defense_tokens[index]

            attack_info.spent_token_indices += (index,)
            attack_info.spent_token_types += (defense_token.type,)
            defense_token.spend()
            attack_info.remove_dice(evade_dice)

            if attack_info.attack_range in [AttackRange.CLOSE, AttackRange.MEDIUM]:
                # Reroll Evade Dice
                for dice_type in DICE:
                    dice_pool.append(sum(evade_dice[dice_type]))
                attack_info.dice_to_roll = tuple(dice_pool)
                self.phase = Phase.ATTACK_ROLL_DICE

        elif action_type == 'pass_defense_token':
            _ = action_data
            # Ship Defender
            if attack_info.is_defender_ship :
                if attack_info.is_attacker_ship or attack_info.bomber:
                    # simplified
                    # self.phase = Phase.ATTACK_USE_CRITICAL_EFFECT
                    self.phase = Phase.ATTACK_RESOLVE_DAMAGE
                else :
                    self.phase = Phase.ATTACK_RESOLVE_DAMAGE

            # Squadron Defender
            else :
                raise NotImplementedError(f"simplified {action_type}")
                defend_squad = <Squad>self.squads[attack_info.defend_squad_id]
                attack_info.calculate_total_damage()
                defend_squad.defend(attack_info.total_damage)

                # Ship to Squadron Attack
                if attack_info.is_attacker_ship :
                    self.phase = Phase.ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET

                # Squadron to Squadron Attack
                else :
                    # Counter Attack
                    if attack_info.attack_squad_id == active_squad.id and defend_squad.counter:
                        self.attack_info = AttackInfo(defend_squad, active_squad)
                        self.phase = Phase.ATTACK_GATHER_DICE
                        
                    elif active_squad.can_move and not active_squad.is_engaged() :
                        self.phase = Phase.SQUAD_MOVE
                    else :
                        active_squad.end_activation()

                        if self.squad_activation_count > 0 and self.get_valid_squad_activation(self.current_player) : self.phase = Phase.SQUAD_ACTIVATE
                        elif self.active_ship is not None : self.phase = Phase.SHIP_RESOLVE_REPAIR
                        elif self.get_valid_squad_activation(- self.current_player) :
                            self.current_player *= -1
                            self.squad_activation_count = 2
                            self.phase = Phase.SQUAD_ACTIVATE
                        elif self.get_valid_squad_activation(self.current_player) :
                            self.squad_activation_count = 2
                            self.phase = Phase.SQUAD_ACTIVATE
                        else :
                            self.status_phase()

        elif action_type == 'use_critical_action':
            raise NotImplementedError(f"simplified {action_type}")
            critical = action_data
            attack_info.critical = critical
            attack_info.calculate_total_damage()
            self.phase = Phase.ATTACK_RESOLVE_DAMAGE

        elif action_type == 'pass_critical':
            raise NotImplementedError(f"simplified {action_type}")
            _ = action_data
            attack_info.calculate_total_damage()
            self.phase = Phase.ATTACK_RESOLVE_DAMAGE

        elif action_type == 'resolve_damage_action':
            redirect_damage= action_data
            total_damage = attack_info.total_damage

            # Redirect
            if redirect_damage :
                hull, damage = redirect_damage
                shield_list = list(defend_ship.shield)
                shield_list[hull] -= damage
                defend_ship.shield = tuple(shield_list)
                total_damage -= damage

            defend_ship.defend(<int>attack_info.defend_hull, total_damage, attack_info.critical)
            for defense_token in defend_ship.defense_tokens :
                defense_token.accuracy = False

            self.attack_info = None
            self.defend_ship = None

            if attack_info.is_attacker_ship :
                if active_ship.attack_count < 2 : self.phase = Phase.SHIP_CHOOSE_TARGET_SHIP
                else : self.phase = Phase.SHIP_DETERMINE_COURSE
            else :
                raise NotImplementedError(f"simplified {action_type}")
                if active_squad.can_move and not active_squad.is_engaged() :
                    self.phase = Phase.SQUAD_MOVE
                else :
                    active_squad.end_activation()
                    if self.squad_activation_count > 0 and self.get_valid_squad_activation(self.current_player) : self.phase = Phase.SQUAD_ACTIVATE
                    elif self.active_ship is not None : self.phase = Phase.SHIP_RESOLVE_REPAIR
                    elif self.get_valid_squad_activation(- self.current_player) :
                        self.current_player *= -1
                        self.squad_activation_count = 2
                        self.phase = Phase.SQUAD_ACTIVATE
                    elif self.get_valid_squad_activation(self.current_player) :
                        self.squad_activation_count = 2
                        self.phase = Phase.SQUAD_ACTIVATE
                    else :
                        self.status_phase()

        elif action_type == 'declare_additional_squad_target_action':
            raise NotImplementedError(f"simplified {action_type}")
            squad_id = action_data
            attack_ship = self.ships[attack_info.attack_ship_id]
            attack_hull = attack_info.attack_hull
            defend_squad = self.squads[squad_id]
            attack_info.declare_additional_squad_target((attack_ship, attack_hull), defend_squad)
            self.phase = Phase.ATTACK_GATHER_DICE

        elif action_type == 'pass_attack':
            _= action_data
            self.phase = Phase.SHIP_DETERMINE_COURSE

        elif action_type == 'pass_additional_squad_target':
            raise NotImplementedError(f"simplified {action_type}")
            _ = action_data
            if active_ship.attack_count < 2 : self.phase = Phase.SHIP_DECLARE_TARGET
            else : self.phase = Phase.SHIP_DETERMINE_COURSE

        elif action_type == 'determine_course_action':
            (course, placement)= action_data
            dial, token = active_ship.nav_command_used(course)
            if dial:
                active_ship.spend_command_dial(Command.NAV)
            if token:
                active_ship.spend_command_token(Command.NAV)

            active_ship.speed = len(course)
            active_ship.execute_maneuver(course, placement)

            # for obstacle in self.obstacles:
            #     if active_ship.check_overlap(obstacle) :
            #         active_ship.overlap_obstacle(obstacle)


            if active_ship.is_overlap_squad() :
                raise NotImplementedError(f"simplified {action_type}")
                self.phase = Phase.SHIP_PLACE_SQUAD
            else :
                active_ship.end_activation()

                if self.get_valid_ship_activation(- self.current_player) :
                    # if opponent has ships left
                    self.current_player *= -1
                    self.phase = Phase.SHIP_ACTIVATE
                
                elif self.get_valid_ship_activation(self.current_player) :
                    # elif current player has ships left
                    self.phase = Phase.SHIP_ACTIVATE
                
                else :
                    if self.get_valid_squad_activation(self.first_player) :
                        raise NotImplementedError(f"simplified {action_type}")
                        self.current_player = self.first_player
                        self.squad_activation_count = 2
                        self.phase = Phase.SQUAD_ACTIVATE
                    elif self.get_valid_squad_activation(- self.first_player) :
                        raise NotImplementedError(f"simplified {action_type}")
                        self.current_player = - self.first_player
                        self.squad_activation_count = 2
                        self.phase = Phase.SQUAD_ACTIVATE
                    else :
                        self.status_phase()

        elif action_type == 'place_squad_action':
            raise NotImplementedError(f"simplified {action_type}")
            (squad_id, coords_index) = action_data
            squad = self.squads[squad_id]
            if coords_index is None :
                print(f"WARNING {squad} is destroyed due to no valid placement on {self.ships[squad.overlap_ship_id]}. This will be fixed in the future version.")
                visualizer.visualize(self,f'\n{squad} is destroyed due to no valid placement on {self.ships[squad.overlap_ship_id]}.')
                squad.destroy()
            else :
                coords : np.ndarray = cache._ship_coordinate(active_ship.get_ship_hash_state())['squad_placement_points'][coords_index]
                squad.place_squad(tuple(coords.tolist()))

            squad_to_place = False
            for squad in self.squads:
                if squad.overlap_ship_id is not None :
                    squad_to_place = True
                    break
            if squad_to_place:
                self.phase = Phase.SHIP_PLACE_SQUAD
            else :
                active_ship.end_activation()

                if self.get_valid_ship_activation(- self.current_player) :
                    # if opponent has ships left
                    self.current_player *= -1
                    self.phase = Phase.SHIP_ACTIVATE
                
                elif self.get_valid_ship_activation(self.current_player) :
                    # elif current player has ships left
                    self.phase = Phase.SHIP_ACTIVATE
                
                else :
                    if self.get_valid_squad_activation(self.first_player) :
                        self.current_player = self.first_player
                        self.squad_activation_count = 2
                        self.phase = Phase.SQUAD_ACTIVATE
                    elif self.get_valid_squad_activation(- self.first_player) :
                        self.current_player = - self.first_player
                        self.squad_activation_count = 2
                        self.phase = Phase.SQUAD_ACTIVATE
                    else :
                        self.status_phase()

        elif action_type == 'activate_squad_move_action':
            raise NotImplementedError(f"simplified {action_type}")  
            squad_id = action_data
            active_squad = self.squads[squad_id]
            active_squad.start_activation()
            if self.active_ship is None : active_squad.can_attack = False
            self.phase = Phase.SQUAD_MOVE

        elif action_type == 'activate_squad_attack_action':
            raise NotImplementedError(f"simplified {action_type}")  
            squad_id = action_data
            active_squad = self.squads[squad_id]
            active_squad.start_activation()
            if self.active_ship is None : active_squad.can_move = False
            self.phase = Phase.SQUAD_DECLARE_TARGET

        elif action_type == 'declare_squad_target_action':
            raise NotImplementedError(f"simplified {action_type}")  
            defender_id = action_data
            if isinstance(defender_id, tuple) : # ship target
                defend_ship_id, defend_hull = defender_id
                defender = (self.ships[defend_ship_id], defend_hull)
            else :
                defender = self.squads[defender_id]

            # gather initial dice pool here
            self.attack_info = AttackInfo(active_squad, defender)
            self.phase = Phase.ATTACK_GATHER_DICE

        elif action_type == 'move_squad_action':
            raise NotImplementedError(f"simplified {action_type}")
            (speed, angle) = action_data
            active_squad.move(speed, angle)

            if active_squad.can_attack : self.phase = Phase.SQUAD_DECLARE_TARGET
            else :
                active_squad.end_activation()
                if self.squad_activation_count > 0 and self.get_valid_squad_activation(self.current_player) : self.phase = Phase.SQUAD_ACTIVATE
                elif self.active_ship is not None : self.phase = Phase.SHIP_RESOLVE_REPAIR
                elif self.get_valid_squad_activation(- self.current_player) :
                    self.current_player *= -1
                    self.squad_activation_count = 2
                    self.phase = Phase.SQUAD_ACTIVATE
                elif self.get_valid_squad_activation(self.current_player) :
                    self.squad_activation_count = 2
                    self.phase = Phase.SQUAD_ACTIVATE
                else :
                    self.status_phase()

        elif action_type == 'pass_activate_squad':
            raise NotImplementedError(f"simplified {action_type}")
            _ = action_data
            self.squad_activation_count = 0
            if self.active_ship is None : raise ValueError('You cannot pass squad activation during Squadron Phase')
            self.phase = Phase.SHIP_RESOLVE_REPAIR
        
        elif action_type == 'pass_move_squad':
            raise NotImplementedError(f"simplified {action_type}")
            _ = action_data
            active_squad.can_move = False
            if active_squad.can_attack : self.phase = Phase.SQUAD_DECLARE_TARGET
            else :
                active_squad.end_activation()
                if self.squad_activation_count > 0 and self.get_valid_squad_activation(self.current_player) : self.phase = Phase.SQUAD_ACTIVATE
                elif self.active_ship is not None : self.phase = Phase.SHIP_RESOLVE_REPAIR
                elif self.get_valid_squad_activation(- self.current_player) :
                    self.current_player *= -1
                    self.squad_activation_count = 2
                    self.phase = Phase.SQUAD_ACTIVATE
                elif self.get_valid_squad_activation(self.current_player) :
                    self.squad_activation_count = 2
                    self.phase = Phase.SQUAD_ACTIVATE
                else :
                    self.status_phase()
        
        elif action_type == 'pass_attack_squad':
            raise NotImplementedError(f"simplified {action_type}")
            _ = action_data
            active_squad.can_attack = False
            if active_squad.can_move and not active_squad.is_engaged(): self.phase = Phase.SQUAD_MOVE
            else :
                active_squad.end_activation()
                if self.squad_activation_count > 0 and self.get_valid_squad_activation(self.current_player) : self.phase = Phase.SQUAD_ACTIVATE
                elif self.active_ship is not None : self.phase = Phase.SHIP_RESOLVE_REPAIR
                elif self.get_valid_squad_activation(- self.current_player) :
                    self.current_player *= -1
                    self.squad_activation_count = 2
                    self.phase = Phase.SQUAD_ACTIVATE
                elif self.get_valid_squad_activation(self.current_player) :
                    self.squad_activation_count = 2
                    self.phase = Phase.SQUAD_ACTIVATE
                else :
                    self.status_phase()

        else:
            raise ValueError(f'Unknown Action {action}')

        # decision player for NEXT PHASE (after applying action)
        self.update_decision_player()

    cpdef void visualize_action(self, object action) :
        """
        get action string and call visualizer
        """
        if not self.debuging_visual : return

        action_str = get_action_str(self, action)
        if action_str is None : return

        maneuver_tool = None
        if action[0] == 'determine_course_action':
            course, placement = action[1]
            if self.active_ship is None : raise ValueError('Need active ship to perform maneuver')
            maneuver_tool, _ = self.active_ship._tool_coordination(course, placement)

        self.visualize(f"Round {self.round} | {self.phase.name.replace('_',' ').title()} | Player {Faction(self.current_player)}\n{action_str}", maneuver_tool)

    cpdef void deploy_ship(self, Ship ship, float x, float y, float orientation, int speed) :
        self.ships.append(ship)
        ship.deploy(self, x, y, orientation, speed, len(self.ships) - 1)
        self.visualize(f'\n{ship.name} is deployed.')

        ship_view = self.ship_encode_array[ship.id]


        cdef float MAX_DICE = Config.GLOBAL_MAX_DICE
        
        # Assign features 0 to 26 in a single block
        # (11 Base Stats + 12 Battery Dice + 3 Anti-Squad Dice) = 26 values
        ship_view[0:26] = [
            # --- Status & Values (11 features) ---
            ship.team,                                    # 0: Team 
            ship.size_class / float(SizeClass.LARGE),     # 1: Size
            ship.command_value / 3.0,                     # 2: Command
            ship.squad_value / Config.GLOBAL_MAX_SQUAD_VALUE,    # 3: Squad
            ship.engineer_value / Config.GLOBAL_MAX_ENGINEER_VALUE, # 4: Engineer
            ship.point / 100.0,                           # 5: Points
            ship.max_hull / Config.GLOBAL_MAX_HULL,       # 6: Hull
            ship.max_shield[HullSection.FRONT] / Config.GLOBAL_MAX_SHIELDS, # 7: Front Shield
            ship.max_shield[HullSection.RIGHT] / Config.GLOBAL_MAX_SHIELDS, # 8: Right Shield
            ship.max_shield[HullSection.REAR] / Config.GLOBAL_MAX_SHIELDS,  # 9: Rear Shield
            ship.max_shield[HullSection.LEFT] / Config.GLOBAL_MAX_SHIELDS,  # 10: Left Shield
            
            # --- Battery Armament (4 hulls * 3 colors = 12 features) ---
            *[dice / MAX_DICE for hull_dice in ship.battery.values() for dice in hull_dice],

            # --- Anti-Squad Armament (3 features) ---
            *[dice / MAX_DICE for dice in ship.anti_squad]
        ]

        # Assign Navigation Chart (Features 26 to 36)
        ship_view[26:36] = ship.nav_chart_vector

    cpdef void deploy_squad(self, Squad squad, float x, float y) :
        raise NotImplementedError("simplified deploy_squad")
        self.squads.append(squad)
        squad.deploy(self, x, y, len(self.squads) - 1)
        self.visualize(f'\n{str(squad)} is deployed.')

        # Get a view to the specific row we will write to
        squad_view = self.squad_encode_array[squad.id]
        offset = 0 # Feature index for this row

        # --- Stats (15 features) ---
        squad_view[offset] = squad.max_hull / Config.GLOBAL_MAX_HULL; offset += 1
        squad_view[offset] = squad.team; offset += 1
        squad_view[offset] = squad.speed / 5.0; offset += 1
        squad_view[offset] = squad.point / 20.0; offset += 1
        squad_view[offset] = squad.swarm; offset += 1
        squad_view[offset] = squad.bomber; offset += 1
        squad_view[offset] = squad.escort; offset += 1
        squad_view[offset] = squad.heavy; offset += 1
        squad_view[offset] = squad.counter / 2.0; offset += 1

        squad_view[offset] = squad.battery[0] / Config.GLOBAL_MAX_DICE; offset += 1
        squad_view[offset] = squad.battery[1] / Config.GLOBAL_MAX_DICE; offset += 1
        squad_view[offset] = squad.battery[2] / Config.GLOBAL_MAX_DICE; offset += 1
        squad_view[offset] = squad.anti_squad[0] / Config.GLOBAL_MAX_DICE; offset += 1
        squad_view[offset] = squad.anti_squad[1] / Config.GLOBAL_MAX_DICE; offset += 1
        squad_view[offset] = squad.anti_squad[2] / Config.GLOBAL_MAX_DICE; offset += 1

    def place_obstacle(self, obstacle: Obstacle, x: float, y: float, orientation: float, flip: bool) -> None:
        raise NotImplementedError("simplified place_obstacle")
        self.obstacles.append(obstacle)
        self.obstacles.sort(key=lambda obstacle: obstacle.index)
        obstacle.place_obstacle(x, y, orientation, flip)
        self.visualize(f'\n{str(obstacle)} is placed.')

        obstacle_view = self.spatial_encode_array[obstacle.type]
        height_res, width_res = Config.BOARD_RESOLUTION
        width_step = self.player_edge / width_res
        height_step = self.short_edge / height_res
        scale_factor = 2
        scaled_vertices = obstacle.coordinates / [width_step, height_step]

        # Important: We must scale the vertices *and* the shape constraint
        high_res_vertices = scaled_vertices * scale_factor
        high_res_shape = (obstacle_view.shape[0] * scale_factor, obstacle_view.shape[1] * scale_factor)

        # 2. DRAW: Get coordinates on the high-res grid
        # Note: We pass the high_res_shape so it doesn't clip incorrectly
        rr, cc = draw_polygon(
            high_res_vertices[:, 1], 
            high_res_vertices[:, 0], 
            shape=high_res_shape
        )

        # 3. DOWNSAMPLE INDICES
        rr //= scale_factor
        cc //= scale_factor

        # 4. ACCUMULATE: Use add.at to handle overlapping indices correctly
        # Value to add is 1.0 / (scale_factor^2) -> 1.0 / 4 = 0.25
        value_per_hit = 1.0 / (scale_factor ** 2)

        np.add.at(obstacle_view, (rr, cc), value_per_hit)



    cdef bint total_destruction(self, int player) :
        cdef Ship ship
        for ship in self.ships:
            if ship.team == player and not ship.destroyed:
                return False
        return True

    cdef list get_valid_ship_activation(self, int player) :
        """
        Returns a list of ships that can be activated by the given player.
        A ship can be activated if it is not destroyed, not already activated, and belongs to the player.
        """
        cdef list valid_ships = []
        cdef Ship ship
        for ship in self.ships:
            if ship.team == player and not ship.destroyed and not ship.activated:
                valid_ships.append(ship)
        return valid_ships

    cdef list get_valid_squad_activation(self, int player) :
        """
        Returns a list of squadrons that can be activated by the given player.
        A squadron can be activated if it is not destroyed, not already activated, and belongs to the player.
        """
        cdef list valid_squadrons = []
        cdef Squad squad
        for squad in self.squads:
            if squad.team == player and not squad.destroyed and not squad.activated:
                valid_squadrons.append(squad)
        return valid_squadrons

    cdef void status_phase(self) :
        cdef:
            Ship ship
            Squad squad
            bint player_1_eliminated, player_2_eliminated, is_game_over
            int p1_points, p2_points

        # 1. Refresh all active ships for the next round
        for ship in self.ships:
            if not ship.destroyed:
                ship.status_phase()
        for squad in self.squads:
            if not squad.destroyed:
                squad.status_phase()

        # 2. Check for game-ending conditions
        player_1_eliminated = self.total_destruction(self.first_player)
        player_2_eliminated = self.total_destruction(-self.first_player)
        is_game_over = player_1_eliminated or player_2_eliminated or self.round == 6

        # 3. If the game is over, determine the winner
        if is_game_over:
            p1_points = self.get_point(self.first_player)
            p2_points = self.get_point(self.second_player)

            if player_1_eliminated and not player_2_eliminated:
                winner = -1
                margin_of_victory = max(p2_points - p1_points, 0)
            elif player_2_eliminated and not player_1_eliminated:
                winner = 1
                margin_of_victory = max(p1_points - p2_points, 0)
            elif player_1_eliminated and player_2_eliminated:
                winner = -1
                margin_of_victory = 0
            else :
                winner = self.first_player if p1_points > p2_points else self.second_player
                margin_of_victory = abs(p1_points - p2_points)
            
            tournament_point : int = margin_of_victory // 20 + 1
            self.winner = winner * tournament_point / 10

        # 4. If the game is not over, advance to the next round
        else:
            self.round += 1
            self.current_player = self.first_player
            self.phase = Phase.SHIP_ACTIVATE

    cpdef int get_point(self, int player) :
        cdef Ship ship
        cdef Squad squad
        cdef list ship_point, squad_point
        cdef point = 0
        for ship in self.ships :
            if ship.team != player and ship.destroyed :
                point += ship.point
        for squad in self.squads :
            if squad.team != player and squad.destroyed :
                point += squad.point
        return point

    def visualize(self, title : str, maneuver_tool : list[tuple[float, float]] | None = None) -> None:
        if not self.debuging_visual:
            return
        visualizer.visualize(self, title, maneuver_tool)

    cpdef object get_snapshot(self):
        """Captures the essential state of the entire game."""
        cdef list ship_snapshots = []
        cdef list squad_snapshots = []
        
        cdef Ship ship
        cdef Squad squad
        
        for ship in self.ships:
            ship_snapshots.append(ship.get_snapshot())
            
        for squad in self.squads:
            squad_snapshots.append(squad.get_snapshot())


        return (
            self.winner,
            self.round,
            self.phase,
            self.current_player,
            self.decision_player,
            (<Ship>self.active_ship).id if self.active_ship is not None else None,
            (<Ship>self.defend_ship).id if self.defend_ship is not None else None,
            (<Squad>self.active_squad).id if self.active_squad is not None else None,
            self.squad_activation_count,
            (<AttackInfo>self.attack_info).get_snapshot() if self.attack_info is not None else None,
            tuple(ship_snapshots),
            tuple(squad_snapshots)
        )

    cpdef void revert_snapshot(self, object snapshot):
        """Restores the entire game state from a snapshot."""
        (self.winner, self.round, self.phase, self.current_player, self.decision_player,
         active_ship_id, defend_ship_id, active_squad_id, self.squad_activation_count, attack_info_snapshot,
         ship_states, squad_states) = snapshot
        self.attack_info = AttackInfo.from_snapshot(attack_info_snapshot) if attack_info_snapshot is not None else None

        self.active_ship = self.ships[<int>active_ship_id] if active_ship_id is not None else None
        self.defend_ship = self.ships[<int>defend_ship_id] if defend_ship_id is not None else None  
        self.active_squad = self.squads[<int>active_squad_id] if active_squad_id is not None else None

        for i, ship_snapshot in enumerate(ship_states):
            (<Ship>self.ships[i]).revert_snapshot(ship_snapshot)
        for i, squad_snapshot in enumerate(squad_states):
            (<Squad>self.squads[i]).revert_snapshot(squad_snapshot)