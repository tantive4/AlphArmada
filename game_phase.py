from enum import IntEnum
from typing import TypeAlias, Literal
from ship import HullSection, Command
from defense_token import DefenseToken
from dice import Dice, Critical
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
    SHIP_ATTACK_DECLARE_TARGET = 2201
    SHIP_ATTACK_GATHER_DICE = 2202
    SHIP_ATTACK_ROLL_DICE = 2203
    SHIP_ATTACK_RESOLVE_EFFECTS = 2204
    SHIP_ATTACK_SPEND_DEFENSE_TOKENS = 2205
    SHIP_ATTACK_USE_CRITICAL_EFFECT = 2206
    SHIP_ATTACK_RESOLVE_DAMAGE = 2207
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