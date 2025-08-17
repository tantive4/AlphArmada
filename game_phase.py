from enum import IntEnum
from typing import TypeAlias, Literal
from ship import HullSection

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
    
    # Ship Phase -> Attack (22xx)
    SHIP_ATTACK = 2200
    SHIP_ATTACK_DECLARE_TARGET = 2201
    SHIP_ATTACK_GATHER_DICE = 2202
    SHIP_ATTACK_ROLL_DICE = 2203
    SHIP_ATTACK_RESOLVE_EFFECTS = 2204
    SHIP_ATTACK_SPEND_DEFENSE_TOKENS = 2205
    SHIP_ATTACK_RESOLVE_DAMAGE = 2206
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
    ActiveShipAction: TypeAlias = tuple[Literal['activate_ship_action'], int]
    DeclareTargetAction: TypeAlias = tuple[Literal['declare_target_action'], tuple[HullSection, int, HullSection]]

    GatherDiceAction: TypeAlias = tuple[Literal['gather_dice_action'], list[int]]
    # AttackDiceAction: TypeAlias = tuple[Literal['attack_dice_roll'], list[list[int]]]
    RollDiceAction: TypeAlias = tuple[Literal['roll_dice_action'], list[list[int]]]

    NoneValueAction: TypeAlias = tuple[Literal['pass_ship_activation', 'pass_attack', 'status_phase'], None]

    DetermineCourseAction: TypeAlias = tuple[Literal['determine_course_action'], tuple[list[int], int]]
    # DetermineSpeedAction: TypeAlias = tuple[Literal['determine_speed_action'], int]
    # DetermineYawAction: TypeAlias = tuple[Literal['determine_yaw_action'], int]
    # DeterminePlacementAction: TypeAlias = tuple[Literal['determine_placement_action'], int]

    UnderConstructionAction: TypeAlias = tuple[Literal['resolve_damage_action'], None]

    Action: TypeAlias = (
        ActiveShipAction | 
        DeclareTargetAction | 
        GatherDiceAction |
        RollDiceAction | 
        NoneValueAction |
        DetermineCourseAction |
        UnderConstructionAction
        # DetermineSpeedAction |
        # DetermineYawAction |
        # DeterminePlacementAction
    )