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
    SHIP_GAIN_COMMAND_TOKEN = 2101
    SHIP_DISCARD_COMMAND_TOKEN = 2102
    SHIP_RESOLVE_REPAIR = 2103
    SHIP_USE_ENGINEER_POINT = 2104
    
    # Ship Phase -> Attack (22xx)
    SHIP_ATTACK = 2200 # checkpoint
    SHIP_ATTACK_DECLARE_TARGET = 2201
    SHIP_ATTACK_GATHER_DICE = 2202
    SHIP_ATTACK_ROLL_DICE = 2203
    SHIP_ATTACK_RESOLVE_EFFECTS = 2204
    SHIP_ATTACK_SPEND_DEFENSE_TOKENS = 2205
    SHIP_ATTACK_USE_CRITICAL_EFFECT = 2206
    SHIP_ATTACK_RESOLVE_DAMAGE = 2207
    # Note: Additional squadron target is part of the same sequence
    
    # Ship Phase -> Execute Maneuver (23xx)
    SHIP_EXECUTE_MANEUVER = 2300 # checkpoint
    SHIP_MANEUVER_DETERMINE_COURSE = 2301
    SHIP_MANEUVER_MOVE_SHIP = 2302 # checkpoint

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
    ResolveCommandAction : TypeAlias = tuple[Literal['resolve_con-fire_command_action', 'resolve_nav_command_action', 'resolve_repair_command_action'], tuple[bool, bool]]
    RepairAction : TypeAlias = (
        tuple[Literal['repair_hull_action'], None] | 
        tuple[Literal['recover_shield_action'], HullSection] |
        tuple[Literal['move_shield_action'], tuple[HullSection, HullSection]] 
    )

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


    DetermineCourseAction: TypeAlias = tuple[Literal['determine_course_action'], tuple[list[int], int]] # course, placement

    NoneValueAction: TypeAlias = tuple[Literal['initialize_game',
                                               'pass_ship_activation', 
                                               'pass_command', 
                                               'pass_command_token',
                                               'pass_repair',
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
        RepairAction |
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

        match action:
            case 'set_command_action', (ship_id, command) :
                action_str = f'Set {command} Command on {game.ships[ship_id]}'

            case 'activate_ship_action', ship_id:
                action_str = f'Activate Ship: {game.ships[ship_id].name}'


            case 'reveal_command_action', command :
                action_str = f'{game.active_ship} reveals {command} Command'

            case 'gain_command_token_action', command :
                action_str = f'{game.active_ship} gains {command} Token'

            case 'discard_command_token_action', command :
                action_str = f'{game.active_ship} discards {command} Token'
            

            case 'move_shield_action', (from_hull, to_hull) :
                action_str = f'Move Shield from {from_hull} to {to_hull}'


            case 'declare_target_action', (attack_hull, defend_ship_id, defend_hull):
                defend_ship = game.ships[defend_ship_id]
                action_str = f'Declare Attack: from {game.active_ship} {attack_hull} to {defend_ship} {defend_hull}'

            case 'gather_dice_action', dice_to_remove :
                if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
                action_str = f'Gather Dice {game.attack_info.dice_to_roll} {f"(Remove {dice_to_remove})" if any(dice_to_remove.values()) else ""}'

            case 'use_confire_token_action', dice :
                action_str = f'Use Confire Token to Reroll {dice_icon(dice)}'

            case 'roll_dice_action', dice :
                dice_result = dice_icon(dice)
                action_str = f'Dice Roll {dice_result}'

            case 'spend_accuracy_action', (dice_type, index) :
                if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
                defend_ship = game.ships[game.attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]
                action_str = f'Spend {dice_type} Accuracy : {token}'

            case 'spend_evade_token_action', (index, evade_dice) :
                if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
                defend_ship = game.ships[game.attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]
                action_str = f'{'Discard' if sum([sum(evade_dice[dice_type]) for dice_type in Dice]) == 2 else 'Spend'} {token} Token on {dice_icon(evade_dice)} ({game.attack_info.attack_range} Range)'

            case 'spend_redicect_token_action', (index, hull) :
                if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
                defend_ship = game.ships[game.attack_info.defend_ship_id]
                token = defend_ship.defense_tokens[index]
                action_str = f'Spend {token} Token to {hull}'

            case 'spend_defense_token_action', index :
                if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
                defend_ship = game.ships[game.attack_info.defend_ship_id]
                action_str = f'Spend {defend_ship.defense_tokens[index]} Token'
                
            case 'resolve_damage_action', redirect_list :
                if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
                action_str = f'Resolve Total {game.attack_info.total_damage} Damage'
                if redirect_list : action_str += f', Redirect {[f'{damage} to {hull}' for hull, damage in redirect_list]}'


            case 'determine_course_action', (course, placement) :
                if game.active_ship is None : raise ValueError('Need active ship to determine course')
                dial_used, token_used = game.active_ship.nav_command_used(course)
                action_str = f'Determine Course: {course}, Placement: {'Right' if placement == 1 else 'Left'} {" (Dial Spent)" if dial_used else ""} {" (Token Spent)" if token_used else ""}'

            case _:
                if 'pass' in action[0] : return
                action_str = f'{action[0].replace('_action', '').replace('_', ' ').title().strip()} {f': {action[1]}' if action[1] else ''}'
        return action_str