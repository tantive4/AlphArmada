from __future__ import annotations
from enum import IntEnum
from typing import TypeAlias, Literal, TYPE_CHECKING
from ship import HullSection, Command
from defense_token import DefenseToken
from dice import Dice, Critical, dice_icon

if TYPE_CHECKING:
    from armada import Armada


class GamePhase(IntEnum):
    '''
    A single enum to track the entire hierarchical game state.
    - 1000s: Round phase
    - 100s:  Major action (e.g., Ship Phase step)
    - 1s:    Action sub-step (e.g., Attack Sequence step)
    '''
    # Round : COMMAND PHASE
    COMMAND_PHASE = 1000

    # Round : SHIP PHASE (2xxx)
    SHIP_PHASE = 2000
    
    # Ship Phase -> Reveal Dial (21xx)
    SHIP_REVEAL_COMMAND_DIAL = 2100
    SHIP_DISCARD_COMMAND_TOKEN = 2101
    
    # Ship Phase -> Attack (22xx)
    SHIP_ATTACK = 2200
    SHIP_ATTACK_DECLARE_ATTACK_HULL = 2201
    SHIP_ATTACK_DECLARE_TARGET = 2202
    SHIP_ATTACK_GATHER_DICE = 2203
    SHIP_ATTACK_ROLL_DICE = 2204
    SHIP_ATTACK_RESOLVE_EFFECTS = 2205
    SHIP_ATTACK_SPEND_DEFENSE_TOKENS = 2206
    SHIP_ATTACK_USE_CRITICAL_EFFECT = 2207
    SHIP_ATTACK_RESOLVE_DAMAGE = 2208
    # Note: Additional squadron target is part of the same sequence
    
    # Ship Phase -> Execute Maneuver (23xx)
    SHIP_EXECUTE_MANEUVER = 2300
    SHIP_MANEUVER_DETERMINE_COURSE = 2301
    SHIP_MANEUVER_MOVE_SHIP = 2302

    # Round : SQUADRON PHASE
    SQUADRON_PHASE = 3000

    # Round : STATUS PHASE
    STATUS_PHASE = 4000

    
class ActionType :
    SetCommandAction : TypeAlias = tuple[Literal['set_command_action'], tuple[int, Command]]

    ActiveShipAction: TypeAlias = tuple[Literal['activate_ship_action'], int]
    GainCommandTokenAction : TypeAlias = tuple[Literal['gain_command_token_action'], Command]
    ReavealCommandAction : TypeAlias = tuple[Literal['reveal_command_action'], Command]
    DiscardCommandTokenAction : TypeAlias = tuple[Literal['discard_command_token_action'], Command]
    ResolveCommandAction : TypeAlias = tuple[Literal['resolve_con-fire_command_action', 'resolve_nav_command_action'], tuple[bool, bool]]

    DeclareAttackHullAction : TypeAlias = tuple[Literal['declare_attack_hull_action'], HullSection]
    DeclareTargetAction: TypeAlias = tuple[Literal['declare_target_action'], tuple[HullSection, int, HullSection]]
    GatherDiceAction: TypeAlias = tuple[Literal['gather_dice_action'], dict[Dice, int]]
    RollDiceAction: TypeAlias = tuple[Literal['roll_dice_action'], dict[Dice, list[int]]]

    SpendAccuracyAction : TypeAlias = tuple[Literal['spend_accuracy_action'], tuple[Dice, int]]
    UseConFireDialAction : TypeAlias = tuple[Literal['use_confire_dial_action'], dict[Dice, int]]
    UseConFireTokenAction : TypeAlias = tuple[Literal['use_confire_token_action'], dict[Dice, list[int]]]
    ResolveAttackEffectAction : TypeAlias = (
        SpendAccuracyAction |
        UseConFireDialAction | 
        UseConFireTokenAction
    )

    SpendDefenseTokenAction: TypeAlias = (
        tuple[Literal['spend_defense_token_action'], int] | 
        tuple[Literal['spend_redicect_token_action'], tuple[int, HullSection]] | 
        tuple[Literal['spend_evade_token_action'], tuple[int, dict[Dice, list[int]]]])

    UseCriticalAction : TypeAlias = tuple[Literal['use_critical_action'], Critical | None]
    ResolveDamageAction: TypeAlias = tuple[Literal['resolve_damage_action'], list[tuple[HullSection,int]]]


    DetermineCourseAction: TypeAlias = tuple[Literal['determine_course_action'], tuple[list[int], int]]

    NoneValueAction: TypeAlias = tuple[Literal['pass_ship_activation', 
                                               'pass_command', 
                                               'pass_attack', 
                                               'pass_attack_effect', 
                                               'pass_defense_token', 
                                               'pass_critical',
                                               'status_phase'], None] 


    Action: TypeAlias = (
        SetCommandAction |
        ActiveShipAction | 
        GainCommandTokenAction |
        ReavealCommandAction |
        DiscardCommandTokenAction |
        ResolveCommandAction |
        DeclareAttackHullAction |
        DeclareTargetAction | 
        GatherDiceAction |
        RollDiceAction | 
        SpendDefenseTokenAction |
        ResolveAttackEffectAction |
        UseCriticalAction |
        NoneValueAction |
        DetermineCourseAction |
        ResolveDamageAction
    )

    @staticmethod
    def get_action_str(game : Armada, action : ActionType.Action) -> str | None:
        action_str = None

        match action[0]:
            case 'set_command_action' :
                action_str = f'Set {action[1][1]} Command on {game.ships[action[1][0]]}'

            case 'activate_ship_action':
                action_str = f'Activate Ship: {game.ships[action[1]].name}'

            case 'gain_command_token_action' :
                action_str = f'{game.active_ship} reveals {action[1]} Command and gain Token'
            
            case 'reveal_command_action' :
                action_str = f'{game.active_ship} reveals {action[1]} Command'

            case 'declare_attack_hull_action' :
                action_str = f'{game.active_ship} declares attack from {action[1]}'

            case 'declare_target_action':
                attack_hull, defend_ship_id, defend_hull = action[1]
                defend_ship = game.ships[defend_ship_id]
                action_str = f'Declare Target: from {game.active_ship} {attack_hull} to {defend_ship} {defend_hull}'

            case 'use_confire_token_action' :
                dice_choice = dice_icon(action[1])
                action_str = f'Use Confire Token : {dice_choice}'

            case 'roll_dice_action' :
                dice_result = dice_icon(action[1])
                action_str = f'Dice Roll {dice_result}'

            case 'spend_accuracy_action' :
                dice_type, index = action[1]
                if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
                defend_ship = game.ships[game.attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]
                action_str = f'Spend {dice_type} Accuracy : {token}'

            case 'spend_evade_token_action' :
                if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
                defend_ship = game.ships[game.attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[action[1][0]]
                evade_dice = action[1][1]
                action_str = f'{'Discard' if sum([sum(evade_dice[dice_type]) for dice_type in Dice]) == 2 else 'Spend'} {token} Token {dice_icon(evade_dice)}'

            case 'spend_redicect_token_action' :
                if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
                defend_ship = game.ships[game.attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[action[1][0]]
                hull = action[1][1]
                action_str = f'Spend {token} Token to {hull}'

            case 'spend_defense_token_action' :
                index = action[1]
                if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
                defend_ship = game.ships[game.attack_info.defend_ship_id]
                action_str = f'Spend {defend_ship.defense_tokens[index]} Token'
                
            case 'resolve_damage_action' :
                redirect_list = action[1]
                if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
                action_str = f'Resolve Total {game.attack_info.total_damage} Damage'
                if redirect_list : action_str += f', Redirect {[f'{damage} to {hull}' for hull, damage in redirect_list]}'


            case 'determine_course_action':
                course, placement = action[1]
                action_str = f'Determine Course: {course}, Placement: {'Right' if placement == 1 else 'Left'}'

            case _:
                if 'pass' in action[0] : return
                action_str = f'{action[0].replace('_action', '').replace('_', ' ').title().strip()} {f': {action[1]}' if action[1] else ''}'
        return action_str