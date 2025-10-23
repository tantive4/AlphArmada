from __future__ import annotations
import random
import math

import numpy as np

from action_phase import Phase, ActionType, get_action_str
import visualizer
from ship import Ship
from squad import Squad
from attack_info import AttackInfo
from enum_class import *
from dice import *
import cache_function as cache
from defense_token import DefenseToken, TokenType



def setup_game(*,debuging_visual:bool=False, para_index:int=0) -> Armada: 

    game = Armada(initiative=Player.REBEL, para_index=para_index) # randomly choose the first player
    game.debuging_visual = debuging_visual
    rebel_ships = (
        Ship(SHIP_DATA['CR90B'], Player.REBEL),
        Ship(SHIP_DATA['Neb-B Support'], Player.REBEL),  
        Ship(SHIP_DATA['CR90A'], Player.REBEL),
        Ship(SHIP_DATA['Neb-B Escort'], Player.REBEL))

    
    rebel_squads = (Squad(SQUAD_DATA['X-Wing'], Player.REBEL) for _ in range(3))

    empire_ships = (
        Ship(SHIP_DATA['VSD1'], Player.EMPIRE),
        Ship(SHIP_DATA['VSD2'], Player.EMPIRE),)

    empire_squads = (Squad(SQUAD_DATA['TIE Fighter'], Player.EMPIRE) for _ in range(6))

    rebel_ship_deployment :list[tuple[float, float, float]] = [(600, 175, math.pi/16), (700, 175, math.pi/16), (1200, 175, 0), (1400, 175, 0)]
    empire_ship_deployment :list[tuple[float, float, float]] = [(600, 725, math.pi*7/8), (1200, 725, math.pi)]

    for i, ship in enumerate(rebel_ships) :
        game.deploy_ship(ship, *rebel_ship_deployment[i], 3)
    for i, ship in enumerate(empire_ships): 
        game.deploy_ship(ship, *empire_ship_deployment[i], 2)

    for i, squad in enumerate(rebel_squads) :
        game.deploy_squad(squad,  1200 + i * 50, 250)
    for i, squad in enumerate(empire_squads) :
        game.deploy_squad(squad,  1000 - i * 50, 650)

    return game

class Armada:
    def __init__(self, * ,initiative: Player, para_index:int = 0) -> None:
        self.player_edge = 1800 # mm
        self.short_edge = 900 # mm
        self.ships : list[Ship] = []
        self.squads : list[Squad] = []

        self.round : int = 1
        self.first_player : int = initiative.value
        self.second_player : int = -initiative.value

        self.phase : Phase = Phase.COMMAND_PHASE    
        self.current_player : int = self.first_player
        self.decision_player : int = self.first_player
        self.active_ship : Ship | None = None
        self.active_squad : Squad | None = None
        self.attack_info : AttackInfo | None = None
        self.squad_activation_count : int = 0

        self.winner : float = 0.0
        self.image_counter : int = 0
        self.debuging_visual : bool = False
        self.simulation_player : int = 0
        self.para_index : int = para_index

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
                raise RuntimeError(f'Maximum simulation steps reached: {max_simulation_step}\n{self.phase}')
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

    
    def update_decision_player(self) -> None:
        """
        Returns a list of possible actions based on the current game phase.
        """

        if self.phase in (Phase.SHIP_REVEAL_COMMAND_DIAL, Phase.ATTACK_ROLL_DICE):
            self.decision_player = 0
        elif self.phase in (Phase.ATTACK_SPEND_DEFENSE_TOKENS, Phase.ATTACK_RESOLVE_DAMAGE, Phase.SHIP_PLACE_SQUAD):
            self.decision_player = -self.current_player
        else :
            self.decision_player = self.current_player

    def get_valid_actions(self) -> list[ActionType]:
        """
        Returns a list of possible actions based on the current game phase.
        """
        actions : list[ActionType] = []

        # if self.phase > Phase.SHIP_ACTIVATE and self.phase <= Phase.SHIP_PLACE_SQUAD:
        #     if (active_ship := self.active_ship) is None:
        #         raise ValueError(f'No active ship for the current game phase.\n{self.get_snapshot()}')
            
        # if self.phase > Phase.SQUAD_ACTIVATE and self.phase <= Phase.SQUAD_MOVE:
        #     if (active_squad := self.active_squad) is None:
        #         raise ValueError(f'No active squad for the current game phase.\n{self.get_snapshot()}')

        # if self.phase >= Phase.ATTACK_GATHER_DICE and self.phase <= Phase.ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET:
        #     if (attack_info := self.attack_info) is None:
        #         raise ValueError(f'No attack info for the current game phase.\n{self.get_snapshot()}')
        if self.active_ship is not None : active_ship = self.active_ship
        if self.active_squad is not None : active_squad = self.active_squad
        if self.attack_info is not None : attack_info = self.attack_info


        match self.phase:
            case Phase.COMMAND_PHASE :
                ships_to_command = [ship.id for ship in self.ships if ship.player == self.current_player and len(ship.command_stack) < ship.command_value] 
                actions = [('set_command_action', (ship_id, command)) for command in Command for ship_id in ships_to_command]
            
            case Phase.SHIP_ACTIVATE :
                valid_ships = self.get_valid_ship_activation(self.current_player)
                actions = [('activate_ship_action', ship.id) for ship in valid_ships]
            
            # Reveal Command Sequence
            case Phase.SHIP_REVEAL_COMMAND_DIAL :
                if active_ship.player == self.simulation_player or self.simulation_player == 0 :  # player's simulation
                    actions = [('reveal_command_action', active_ship.command_stack[0])]
                else :                                                                          # secret information
                    actions = [('reveal_command_action', command) for command in Command]

            case Phase.SHIP_GAIN_COMMAND_TOKEN :
                actions = [('gain_command_token_action', command) for command in active_ship.command_dial if command not in active_ship.command_token]
                actions.append(('pass_command_token', None))

            case Phase.SHIP_DISCARD_COMMAND_TOKEN :
                actions = [('discard_command_token_action', command) for command in active_ship.command_token]


            # Squad Command
            case Phase.SHIP_RESOLVE_SQUAD :
                dial = Command.SQUAD in active_ship.command_dial
                token_choices = [True, False] if Command.SQUAD in active_ship.command_token else [False]
                actions = [('resolve_squad_command_action', (dial, token)) for token in token_choices]


            # Engineering Sequence
            case Phase.SHIP_RESOLVE_REPAIR :
                dial = Command.REPAIR in active_ship.command_dial
                token_choices = [True, False] if Command.REPAIR in active_ship.command_token else [False]
                actions = [('resolve_repair_command_action', (dial, token)) for token in token_choices]

            case Phase.SHIP_USE_ENGINEER_POINT :
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
            case Phase.SHIP_DECLARE_TARGET :
                actions = [('declare_target_action', (attack_hull, (defend_ship.id, defend_hull)))
                    for attack_hull in active_ship.get_valid_attack_hull()
                    for defend_ship, defend_hull in active_ship.get_valid_ship_target(attack_hull)]
                actions.extend([('declare_target_action', (attack_hull, squad.id)) 
                                for attack_hull in active_ship.get_valid_attack_hull()
                                for squad in active_ship.get_valid_squad_target(attack_hull)])
                if not actions : actions = [('pass_attack', None)]

            case Phase.ATTACK_GATHER_DICE :
                dice_to_roll = attack_info.dice_to_roll

                if attack_info.obstructed:
                    if sum(dice_to_roll) <= 1 : raise ValueError('Empty Attack Pool. Invalid Attack')
                    for dice_type in Dice :
                        if dice_to_roll[dice_type] > 0 :
                            dice_to_remove = tuple(1 if i == dice_type else 0 for i in range(3))
                            actions.append(('gather_dice_action', dice_to_remove))
                else:
                    actions = [('gather_dice_action', (0,0,0))]
            
            case Phase.ATTACK_ROLL_DICE :
                raise NotImplementedError("This is a chance node. No player action available.")

            case Phase.ATTACK_RESOLVE_EFFECTS :

                if attack_info.is_attacker_ship :
                    attack_ship :Ship = self.ships[attack_info.attack_ship_id]
                else :
                    attack_squad :Squad= self.squads[attack_info.attack_squad_id]
                if attack_info.is_defender_ship:
                    defender = self.ships[attack_info.defend_ship_id]
                else :
                    defender = self.squads[attack_info.defend_squad_id]

                # spend accuracy
                blue_acc_count = attack_info.attack_pool_result[Dice.BLUE][ACCURACY_INDEX[Dice.BLUE]]
                red_acc_count = attack_info.attack_pool_result[Dice.RED][ACCURACY_INDEX[Dice.RED]]

                checked_tokens : list[DefenseToken]= []
                for index, token in defender.defense_tokens.items():
                    if token in checked_tokens :
                        continue
                    checked_tokens.append(token)
                    if token.discarded or token.accuracy :
                        continue
                    if blue_acc_count : actions.append(('spend_accuracy_action', (Dice.BLUE, index)))
                    if red_acc_count : actions.append(('spend_accuracy_action', (Dice.RED, index)))


                # use con-fire command
                if attack_info.con_fire_dial :
                    actions.extend([('use_confire_dial_action', tuple(1 if i == dice else 0 for i in range(3))) for dice in Dice if sum(attack_info.attack_pool_result[dice])])

                if not actions : actions = [('pass_attack_effect', None)] # Above actions are MUST USED actions

                # use con-fire command
                if attack_info.is_attacker_ship and Command.CONFIRE not in attack_ship.resolved_command :
                    dial = Command.CONFIRE in attack_ship.command_dial
                    token = Command.CONFIRE in attack_ship.command_token
                    if dial and token:
                        actions.append(('resolve_con-fire_command_action', (True, True)))
                    if dial:
                        actions.append(('resolve_con-fire_command_action', (True, False)))
                    if token:
                        actions.append(('resolve_con-fire_command_action', (False, True)))

                if attack_info.con_fire_token and not attack_info.con_fire_dial : # Use reroll after adding dice
                    actions.extend([('use_confire_token_action', dice) for dice in dice_choices(attack_info.attack_pool_result, 1)])

                # swarm reroll
                if attack_info.swarm:
                    actions.extend([('swarm_reroll_action', dice) for dice in dice_choices(attack_info.attack_pool_result, 1)])

            case Phase.ATTACK_SPEND_DEFENSE_TOKENS :
                if attack_info.is_defender_ship:
                    defender = self.ships[attack_info.defend_ship_id]
                else :
                    defender = self.squads[attack_info.defend_squad_id]
                
                if defender.speed > 0 :
                    checked_tokens : list[DefenseToken]= []
                    for index, token in defender.defense_tokens.items():

                        # do not double check the identical token
                        if token in checked_tokens :
                            continue
                        checked_tokens.append(token)
                        if (not token.discarded and not token.accuracy
                                and index not in attack_info.spent_token_indices
                                and token.type not in attack_info.spent_token_types):
                            
                            if token.type == TokenType.REDIRECT :
                                actions.append(('spend_redirect_token_action',(index, HullSection((attack_info.defend_hull + 1) % 4))))
                                actions.append(('spend_redirect_token_action',(index, HullSection((attack_info.defend_hull - 1) % 4))))

                            elif token.type == TokenType.EVADE :
                                # choose 1 die to affect
                                evade_dice_choices = dice_choices(attack_info.attack_pool_result, 1)
                                for dice_choice in evade_dice_choices :
                                    actions.append(('spend_evade_token_action', (index, dice_choice)))

                            else : actions.append(('spend_defense_token_action', index))

                actions.append(('pass_defense_token', None))

            case Phase.ATTACK_USE_CRITICAL_EFFECT :
                if (attack_info.is_attacker_ship or attack_info.bomber) and attack_info.is_defender_ship:
                    black_crit = bool(attack_info.attack_pool_result[Dice.BLACK][CRIT_INDEX[Dice.BLACK]])
                    blue_crit = bool(attack_info.attack_pool_result[Dice.BLUE][CRIT_INDEX[Dice.BLUE]])
                    red_crit = bool(attack_info.attack_pool_result[Dice.RED][CRIT_INDEX[Dice.RED]])

                    if attack_info.is_attacker_ship :
                        attacker = self.ships[attack_info.attack_ship_id]
                    else :
                        attacker = self.squads[attack_info.attack_squad_id]
                    critical_list = attacker.get_critical_effect(black_crit, blue_crit, red_crit)
                    actions = [('use_critical_action', critical) for critical in critical_list]

                if not actions :
                    actions = [('pass_critical', None)]


            case Phase.ATTACK_RESOLVE_DAMAGE:
                total_damage = attack_info.total_damage
                defender = self.ships[attack_info.defend_ship_id]
                redirect_hull = attack_info.redirect_hull
                if redirect_hull is not None :
                    actions = [('resolve_damage_action', (redirect_hull, damage)) for damage in (range(min(total_damage, defender.shield[redirect_hull]) + 1))]

                actions.append(('resolve_damage_action', None))

            case Phase.ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET :
                attack_ship = self.ships[attack_info.attack_ship_id]
                attack_hull = attack_info.attack_hull
                actions = [('declare_additional_squad_target_action', squad.id) 
                           for squad in attack_ship.get_valid_squad_target(attack_hull) 
                           if squad.id not in attack_info.squadron_target]
                
                if not actions :
                    actions = [('pass_additional_squad_target', None)]



            # Maneuver Sequence
            case Phase.SHIP_DETERMINE_COURSE:
               for speed in active_ship.get_valid_speed():
                    all_courses = active_ship.get_all_possible_courses(speed)

                    for course in all_courses:
                        for placement in active_ship.get_valid_placement(course):
                            actions.append(('determine_course_action', (course, placement)))

            case Phase.SHIP_PLACE_SQUAD :
                actions = [('place_squad_action', (squad.id, index)) 
                           for squad in self.squads if squad.overlap_ship_id is not None
                           for index in self.ships[squad.overlap_ship_id].get_valid_squad_placement(squad)]

            # === SQUADRON_PHASE ===
            case Phase.SQUAD_ACTIVATE :
                if self.active_ship is not None :
                    valid_squads = self.active_ship.get_squad_activation()
                else :
                    valid_squads = [squad for squad in self.squads if squad.player == self.current_player and not squad.activated and not squad.destroyed]
                actions = [('activate_squad_move_action', squad.id) for squad in valid_squads if not squad.is_engaged()]
                actions.extend([('activate_squad_attack_action', squad.id) for squad in valid_squads if squad.get_valid_target()])
                if not actions : actions = [('pass_activate_squad', None)]

            case Phase.SQUAD_MOVE :
                actions = [('move_squad_action', (speed, angle)) for speed, angle in active_squad.get_valid_moves()]
                actions.append(('pass_move_squad', None))

            case Phase.SQUAD_DECLARE_TARGET :
                actions = [('declare_squad_target_action', target) for target in active_squad.get_valid_target()]
                if not actions : actions = [('pass_attack_squad', None)]


            case _ :
                raise ValueError(f'Unknown game phase: {self.phase.name}')
            
        if not actions:
            raise ValueError(f'No valid actions available in phase {self.phase.name}\n{self.get_snapshot()}')

        return actions
    
    
    def apply_action(self, action : ActionType) -> None:
        """
        Applies the given action to the game state.
        """
        # if self.phase > Phase.SHIP_ACTIVATE and self.phase <= Phase.SHIP_PLACE_SQUAD:
        #     if (active_ship := self.active_ship) is None:
        #         raise ValueError(f'No active ship for the current game phase.\n{self.get_snapshot()}')
            
        # if self.phase > Phase.SQUAD_ACTIVATE and self.phase <= Phase.SQUAD_MOVE:
        #     if (active_squad := self.active_squad) is None:
        #         raise ValueError(f'No active squad for the current game phase.\n{self.get_snapshot()}')

        # if self.phase >= Phase.ATTACK_GATHER_DICE and self.phase <= Phase.ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET:
        #     if (attack_info := self.attack_info) is None:
        #         raise ValueError(f'No attack info for the current game phase.\n{self.get_snapshot()}')
        
        if self.active_ship is not None : active_ship = self.active_ship
        if self.active_squad is not None : active_squad = self.active_squad
        if self.attack_info is not None : attack_info = self.attack_info

        self.visualize_action(action)
        
        match action:
            case 'set_command_action', (ship_id, command):
                command_ship = self.ships[ship_id]
                command_ship.command_stack += (command,)
                if [ship.id for ship in self.ships if ship.player == self.current_player and len(ship.command_stack) < ship.command_value] :
                    self.phase = Phase.COMMAND_PHASE
                else :
                    if self.current_player == self.first_player : 
                        self.current_player *= -1
                        self.phase = Phase.COMMAND_PHASE
                    else :
                        self.current_player = self.first_player
                        self.phase = Phase.SHIP_ACTIVATE
            

            case 'activate_ship_action', ship_id:
                self.active_ship = self.ships[ship_id]
                self.current_player = self.active_ship.player
                self.phase = Phase.SHIP_REVEAL_COMMAND_DIAL


            case 'reveal_command_action', command :
                active_ship.command_stack = active_ship.command_stack[1:]
                active_ship.command_dial += (command,)
                self.phase = Phase.SHIP_GAIN_COMMAND_TOKEN

            case 'gain_command_token_action', command :
                active_ship.command_dial = tuple(cd for cd in active_ship.command_dial if cd != command)
                active_ship.command_token += (command,)

                if len(active_ship.command_token) > active_ship.command_value :
                    self.phase = Phase.SHIP_DISCARD_COMMAND_TOKEN
                else :self.phase = Phase.SHIP_RESOLVE_SQUAD

            case 'pass_command_token', _ :
                self.phase = Phase.SHIP_RESOLVE_SQUAD

            case 'discard_command_token_action', command :
                active_ship.command_token = tuple(ct for ct in active_ship.command_token if ct != command)
                self.phase = Phase.SHIP_RESOLVE_SQUAD

            case 'resolve_squad_command_action', (dial, token) :
                if dial : active_ship.command_dial = tuple(cd for cd in active_ship.command_dial if cd != Command.SQUAD)
                if token : active_ship.command_token = tuple(ct for ct in active_ship.command_token if ct != Command.SQUAD)
                if dial or token :
                    active_ship.resolved_command += (Command.SQUAD,)
                    self.squad_activation_count = dial * active_ship.squad_value + token * 1
                    self.phase = Phase.SQUAD_ACTIVATE
                else :
                    self.phase = Phase.SHIP_RESOLVE_REPAIR


            case 'resolve_repair_command_action', (dial, token) :
                if dial : active_ship.command_dial = tuple(cd for cd in active_ship.command_dial if cd != Command.REPAIR)
                if token : active_ship.command_token = tuple(ct for ct in active_ship.command_token if ct != Command.REPAIR)
                if dial or token :
                    active_ship.resolved_command += (Command.REPAIR,)
                    active_ship.engineer_point = dial * active_ship.engineer_value + token * (active_ship.engineer_value + 1) // 2
                    self.phase = Phase.SHIP_USE_ENGINEER_POINT
                else : 
                    self.phase = Phase.SHIP_DECLARE_TARGET

            case 'repair_hull_action', _ :
                active_ship.engineer_point -= 3
                active_ship.hull += 1

            case 'recover_shield_action', hull:
                active_ship.engineer_point -= 2

                shield_list = list(active_ship.shield)
                shield_list[hull] += 1
                active_ship.shield = tuple(shield_list)

                active_ship.repaired_hull += (hull,)


            case 'move_shield_action', (from_hull, to_hull) :
                active_ship.engineer_point -= 1

                shield_list = list(active_ship.shield)
                shield_list[from_hull] -= 1
                shield_list[to_hull] += 1
                active_ship.shield = tuple(shield_list)

                active_ship.repaired_hull += (to_hull,)


            case 'pass_repair', _ :
                active_ship.repaired_hull = ()
                active_ship.engineer_point = 0
                self.phase = Phase.SHIP_DECLARE_TARGET

            case 'declare_target_action', (attack_hull, defender_id) :
                if isinstance(defender_id, tuple) : # ship target
                    defend_ship_id, defend_hull = defender_id
                    defender = (self.ships[defend_ship_id], defend_hull)
                else :
                    defender = self.squads[defender_id]

                # gather initial dice pool here
                self.attack_info = AttackInfo((active_ship, attack_hull), defender)
                self.phase = Phase.ATTACK_GATHER_DICE
            
            case 'gather_dice_action', dice_to_remove:
                # update dice pool considering obstruction.etc
                attack_info.dice_to_roll = tuple(attack_info.dice_to_roll[dice_type] - dice_to_remove[dice_type] for dice_type in Dice)
                self.phase = Phase.ATTACK_ROLL_DICE

            case 'roll_dice_action', dice_roll:
                attack_info.dice_to_roll = tuple(0 for _ in Dice)
                attack_info.attack_pool_result = tuple(tuple(original + new for original, new 
                                                             in zip(attack_info.attack_pool_result[dice_type], dice_roll[dice_type])) for dice_type in Dice)
                attack_info.calculate_total_damage()
                self.phase = attack_info.phase # either ATTACK_RESOLVE_EFFECTS or ATTACK_SPEND_DEFENSE_TOKENS

            case 'spend_accuracy_action', (accuracy_dice, index):
                defend_ship = self.ships[attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]
                token.accuracy = True
                attack_info.attack_pool_result = tuple(
                    tuple(result) if dice_type != accuracy_dice else
                    tuple(count - 1 if i == ACCURACY_INDEX[accuracy_dice] else count
                          for i, count in enumerate(result))
                    for dice_type, result in zip(Dice, attack_info.attack_pool_result)
                )
                attack_info.calculate_total_damage()

            case 'resolve_con-fire_command_action', (dial, token) :
                active_ship.resolved_command += (Command.CONFIRE,)
                attack_info.con_fire_dial, attack_info.con_fire_token = dial, token
                attack_info.calculate_total_damage()
            
            case 'use_confire_dial_action', dice_to_remove :
                attack_info.dice_to_roll = dice_to_remove
                attack_info.con_fire_dial = False
                attack_info.calculate_total_damage()
                self.phase = Phase.ATTACK_ROLL_DICE
                
            case 'use_confire_token_action', reroll_dice :
                attack_info.con_fire_token = False
                attack_info.attack_pool_result = tuple(tuple(original_count - removed_count 
                                                            for original_count, removed_count in zip(attack_info.attack_pool_result[dice_type], reroll_dice[dice_type])) 
                                                            for dice_type in Dice)
                attack_info.dice_to_roll = tuple(sum(reroll_dice[dice_type]) for dice_type in Dice)
                attack_info.calculate_total_damage()
                self.phase = Phase.ATTACK_ROLL_DICE

            case 'swarm_reroll_action', reroll_dice :
                attack_info.swarm = False
                attack_info.attack_pool_result = tuple(tuple(original_count - removed_count 
                                                            for original_count, removed_count in zip(attack_info.attack_pool_result[dice_type], reroll_dice[dice_type])) 
                                                            for dice_type in Dice)
                attack_info.dice_to_roll = tuple(sum(reroll_dice[dice_type]) for dice_type in Dice)
                attack_info.calculate_total_damage()
                self.phase = Phase.ATTACK_ROLL_DICE

            case 'pass_attack_effect', _ :
                attack_info.phase = Phase.ATTACK_SPEND_DEFENSE_TOKENS
                self.phase = Phase.ATTACK_SPEND_DEFENSE_TOKENS

            case 'spend_defense_token_action', index :
                defend_ship = self.ships[attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]

                attack_info.spent_token_indices += (index,)
                attack_info.spent_token_types += (token.type,)
                token.spend()
                attack_info.calculate_total_damage()

            case 'spend_redirect_token_action', (index, hull) :
                defend_ship = self.ships[attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]

                attack_info.spent_token_indices += (index,)
                attack_info.spent_token_types += (token.type,)
                token.spend()
                attack_info.redirect_hull = hull
                attack_info.calculate_total_damage()

            case 'spend_evade_token_action', (index, evade_dice) :
                defend_ship = self.ships[attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]

                attack_info.spent_token_indices += (index,)
                attack_info.spent_token_types += (token.type,)
                token.spend()
                attack_info.attack_pool_result = tuple(tuple(original_count - removed_count 
                                                            for original_count, removed_count in zip(attack_info.attack_pool_result[dice_type], evade_dice[dice_type]))
                                                            for dice_type in Dice)
                attack_info.calculate_total_damage()
                if attack_info.attack_range in [AttackRange.CLOSE, AttackRange.MEDIUM]:
                    # Reroll Evade Dice
                    attack_info.dice_to_roll = tuple(sum(evade_dice[dice_type]) for dice_type in Dice)
                    self.phase = Phase.ATTACK_ROLL_DICE

            case 'pass_defense_token', _ :
                # Ship Defender
                if attack_info.is_defender_ship :
                    if attack_info.is_attacker_ship or attack_info.bomber:
                        self.phase = Phase.ATTACK_USE_CRITICAL_EFFECT
                    else :
                        self.phase = Phase.ATTACK_RESOLVE_DAMAGE

                # Squadron Defender
                else :
                    defender = self.squads[attack_info.defend_squad_id]
                    attack_info.calculate_total_damage()
                    defender.defend(attack_info.total_damage)
                    
                    # Ship to Squadron Attack
                    if attack_info.is_attacker_ship :
                        self.phase = Phase.ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET

                    # Squadron to Squadron Attack
                    else :
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

            case 'use_critical_action', critical :
                attack_info.critical = critical
                attack_info.calculate_total_damage()
                self.phase = Phase.ATTACK_RESOLVE_DAMAGE

            case 'pass_critical', _ :
                attack_info.calculate_total_damage()
                self.phase = Phase.ATTACK_RESOLVE_DAMAGE

            case 'resolve_damage_action', redirect_damage:
                total_damage = attack_info.total_damage
                defender = self.ships[attack_info.defend_ship_id]

                # Redirect
                if redirect_damage :
                    hull, damage = redirect_damage
                    shield_list = list(defender.shield)
                    shield_list[hull] -= damage
                    defender.shield = tuple(shield_list)
                    total_damage -= damage

                defender.defend(attack_info.defend_hull, total_damage, attack_info.critical)
                for token in defender.defense_tokens.values() :
                    token.accuracy = False

                self.attack_info = None

                if attack_info.is_attacker_ship :
                    if active_ship.attack_count < 2 : self.phase = Phase.SHIP_DECLARE_TARGET
                    else : self.phase = Phase.SHIP_DETERMINE_COURSE
                else :

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

            case 'declare_additional_squad_target_action', squad_id :
                attack_ship = self.ships[attack_info.attack_ship_id]
                attack_hull = attack_info.attack_hull
                defend_squad = self.squads[squad_id]
                attack_info.declare_additional_squad_target((attack_ship, attack_hull), defend_squad)
                self.phase = Phase.ATTACK_GATHER_DICE

            case 'pass_attack', _:
                self.phase = Phase.SHIP_DETERMINE_COURSE

            case 'pass_additional_squad_target', _ :
                if active_ship.attack_count < 2 : self.phase = Phase.SHIP_DECLARE_TARGET
                else : self.phase = Phase.SHIP_DETERMINE_COURSE

            case 'determine_course_action', (course, placement):
                dial_used, token_used = active_ship.nav_command_used(course)
                if dial_used:
                    active_ship.command_dial = tuple(cd for cd in active_ship.command_dial if cd != Command.NAV)
                if token_used:
                    active_ship.command_token = tuple(ct for ct in active_ship.command_token if ct != Command.NAV)

                active_ship.speed = len(course)
                active_ship.execute_maneuver(course, placement)


                if active_ship.is_overlap_squad() :
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

            case 'place_squad_action', (squad_id, coords_index) :
                squad = self.squads[squad_id]
                if coords_index is None :
                    squad.destroy()
                else :
                    coords : np.ndarray = cache._ship_coordinate(active_ship.get_ship_hash_state())['squad_placement_points'][coords_index]
                    squad.place_squad((float(coords[0]), float(coords[1])))

                if any(squad.overlap_ship_id is not None for squad in self.squads) :
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

            case 'activate_squad_move_action', squad_id :
                active_squad = self.squads[squad_id]
                active_squad.start_activation()
                if self.active_ship is None : active_squad.can_attack = False
                self.phase = Phase.SQUAD_MOVE

            case 'activate_squad_attack_action', squad_id :
                active_squad = self.squads[squad_id]
                active_squad.start_activation()
                if self.active_ship is None : active_squad.can_move = False
                self.phase = Phase.SQUAD_DECLARE_TARGET

            case 'declare_squad_target_action', defender_id :
                if isinstance(defender_id, tuple) : # ship target
                    defend_ship_id, defend_hull = defender_id
                    defender = (self.ships[defend_ship_id], defend_hull)
                else :
                    defender = self.squads[defender_id]

                # gather initial dice pool here
                self.attack_info = AttackInfo(active_squad, defender)
                self.phase = Phase.ATTACK_GATHER_DICE

            case 'move_squad_action', (speed, angle) :
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

            case 'pass_activate_squad', _ :
                self.squad_activation_count = 0
                if self.active_ship is None : raise ValueError('You cannot pass squad activation during Squadron Phase')
                self.phase = Phase.SHIP_RESOLVE_REPAIR
            
            case 'pass_move_squad', _ :
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
            
            case 'pass_attack_squad', _ :
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

            case _ :
                raise ValueError(f'Unknown Action {action}')

        # decision player for NEXT PHASE (after applying action)
        self.update_decision_player()

    def visualize_action(self, action : ActionType) -> None:
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

        self.visualize(f"Round {self.round} | {self.phase.name.replace("_"," ").title()} | Player {Player(self.current_player)}\n{action_str}", maneuver_tool)

    def deploy_ship(self, ship : Ship, x : float, y : float, orientation : float, speed : int) -> None :
        self.ships.append(ship)
        ship.deploy(self, x, y, orientation, speed, len(self.ships) - 1)
        self.visualize(f'\n{ship.name} is deployed.')

    def deploy_squad(self, squad : Squad, x : float, y : float) -> None :
        self.squads.append(squad)
        squad.deploy(self, x, y, len(self.squads) - 1)
        self.visualize(f'\n{str(squad)} is deployed.')

    def total_destruction(self, player : int) -> bool:
        player_ship_exists = any(ship for ship in self.ships if ship.player == player and not ship.destroyed)
        return not player_ship_exists

    def get_valid_ship_activation(self, player : int) -> list[Ship] :
        """
        Returns a list of ships that can be activated by the given player.
        A ship can be activated if it is not destroyed, not already activated, and belongs to the player.
        """
        return [ship for ship in self.ships if ship.player == player and not ship.destroyed and not ship.activated]

    def get_valid_squad_activation(self, player: int) -> list[Squad]:
        """
        Returns a list of squadrons that can be activated by the given player.
        A squadron can be activated if it is not destroyed, not already activated, and belongs to the player.
        """
        return [squad for squad in self.squads if squad.player == player and not squad.destroyed and not squad.activated]

    def status_phase(self) -> None:
        # if self.simulation_player == 0:
        #     print(f'Game {self.para_index+1 if self.para_index is not None else ''} Round {self.round} Ended')
            # with open('simulation_log.txt', 'a') as f: f.write(f'\n{'-' * 10} Round {self.round} Ended {'-' * 10}\n\n')

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
                winner = self.second_player
                margin_of_victory = max(p2_points - p1_points, 0)
            elif player_2_eliminated and not player_1_eliminated:
                winner = self.first_player
                margin_of_victory = max(p1_points - p2_points, 0)
            elif player_1_eliminated and player_2_eliminated:
                winner = self.second_player
                margin_of_victory = 0
            else :
                winner = self.first_player if p1_points > p2_points else self.second_player
                margin_of_victory = abs(p1_points - p2_points)
            
            tournament_point : int = margin_of_victory // 50 + 1
            self.winner = winner * tournament_point / 4

        # 4. If the game is not over, advance to the next round
        else:
            self.round += 1
            self.current_player = self.first_player
            self.phase = Phase.COMMAND_PHASE

    def get_point(self, player : int) -> int :
        destroyed_ships = [ship for ship in self.ships if ship.player != player and ship.destroyed]
        destroyed_squads = [squad for squad in self.squads if squad.player != player and squad.destroyed]
        return sum(ship.point for ship in destroyed_ships) + sum(squad.point for squad in destroyed_squads)

    def visualize(self, title : str, maneuver_tool : list[tuple[float, float]] | None = None) -> None:
        if not self.debuging_visual:
            return
        visualizer.visualize(self, title, maneuver_tool)

    def get_snapshot(self) -> tuple:
        """Captures the essential state of the entire game."""
        return (
            self.winner,
            self.round,
            self.phase,
            self.current_player,
            self.decision_player,
            self.active_ship.id if self.active_ship else None,
            self.active_squad.id if self.active_squad else None,
            self.squad_activation_count,
            self.attack_info.get_snapshot() if self.attack_info else None,
            tuple(ship.get_snapshot() for ship in self.ships),
            tuple(squad.get_snapshot() for squad in self.squads)
        )

    def revert_snapshot(self, snapshot: tuple) -> None:
        """Restores the entire game state from a snapshot."""
        (self.winner, self.round, self.phase, self.current_player, self.decision_player,
         active_ship_id, active_squad_id, self.squad_activation_count, attack_info_snapshot,
         ship_states, squad_states) = snapshot
        self.attack_info = AttackInfo.from_snapshot(attack_info_snapshot) if attack_info_snapshot else None

        self.active_ship = self.ships[active_ship_id] if active_ship_id is not None else None
        self.active_squad = self.squads[active_squad_id] if active_squad_id is not None else None

        for i, ship_snapshot in enumerate(ship_states):
            self.ships[i].revert_snapshot(ship_snapshot)
        for i, squad_snapshot in enumerate(squad_states):
            self.squads[i].revert_snapshot(squad_snapshot)

