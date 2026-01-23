from __future__ import annotations
from enum import IntEnum, auto
from typing import TypeAlias, Literal, TYPE_CHECKING

from dice import dice_icon
from enum_class import *
if TYPE_CHECKING:
    from armada import Armada


class Phase(IntEnum):
    # === COMMAND PHASE ===
    # COMMAND_PHASE = 0

    # === SHIP PHASE ===
    SHIP_ACTIVATE = 0 

    # Ship Phase -> Reveal Dial
    SHIP_REVEAL_COMMAND_DIAL = auto()
    # SHIP_GAIN_COMMAND_TOKEN = auto()
    # SHIP_DISCARD_COMMAND_TOKEN = auto()
    # SHIP_RESOLVE_SQUAD = auto()
    # SHIP_RESOLVE_REPAIR = auto()
    SHIP_USE_ENGINEER_POINT = auto()

    # Ship Phase -> Attack
    SHIP_CHOOSE_TARGET_SHIP = auto()
    SHIP_DECLARE_TARGET = auto()

    # Ship Phase -> Execute Maneuver
    SHIP_DETERMINE_COURSE = auto()
    # SHIP_PLACE_SQUAD = auto()

    # === SQUADRON_PHASE ===
    # SQUAD_ACTIVATE = auto() 
    # SQUAD_DECLARE_TARGET = auto()
    # SQUAD_MOVE = auto()

    # === Attack Step ===
    # ATTACK_GATHER_DICE = auto()
    ATTACK_ROLL_DICE = auto()
    ATTACK_RESOLVE_EFFECTS = auto()
    ATTACK_SPEND_DEFENSE_TOKENS = auto()
    # ATTACK_USE_CRITICAL_EFFECT = auto()
    ATTACK_RESOLVE_DAMAGE = auto()
    # ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET = auto()
    
    def __str__(self):
        return self.name
    __repr__ = __str__
phase_type = len(Phase)
POINTER_PHASE = (Phase.SHIP_ACTIVATE, Phase.SHIP_CHOOSE_TARGET_SHIP)

ActionType: TypeAlias = (
    tuple[Literal['set_command_action'], tuple[int, Command]] |

    # === Ship Phase ===
    tuple[Literal['activate_ship_action'], int] | 

    # Reveal Command Step
    tuple[Literal['gain_command_token_action'], Command] |
    tuple[Literal['reveal_command_action'], Command] |
    tuple[Literal['discard_command_token_action'], Command] |
    tuple[Literal['resolve_con-fire_command_action', 
                  'resolve_nav_command_action', 
                  'resolve_repair_command_action', 
                  'resolve_squad_command_action'], tuple[bool, bool]] |

    tuple[Literal['repair_hull_action'], None] | 
    tuple[Literal['recover_shield_action'], HullSection] |
    tuple[Literal['move_shield_action'], tuple[HullSection, HullSection]] |

    # Attack Step
    tuple[Literal['choose_target_ship_action'], int] |
    tuple[Literal['declare_target_action'], tuple[HullSection, HullSection]] | 
    tuple[Literal['gather_dice_action'], tuple[int, ...]] |
    tuple[Literal['roll_dice_action'], tuple[tuple[int, ...],...]] | 

    tuple[Literal['spend_accuracy_action'], tuple[Dice, int]] |
    tuple[Literal['use_confire_dial_action'], tuple[int, ...]]|
    tuple[Literal['use_confire_token_action', 
                  'swarm_reroll_action'], tuple[tuple[int, ...],...]]|

    tuple[Literal['spend_defense_token_action'], int] | 
    tuple[Literal['spend_redirect_token_action'], tuple[int, HullSection]] | 
    tuple[Literal['spend_evade_token_action'], tuple[int, tuple[tuple[int, ...],...]]] |

    tuple[Literal['use_critical_action'], Critical | None] |
    tuple[Literal['resolve_damage_action'], tuple[HullSection,int] | None]|
    tuple[Literal['declare_additional_squad_target_action'], int] |

    # Maneuver Step
    tuple[Literal['determine_course_action'], tuple[tuple[int, ...], int]] | # course, placement 
    tuple[Literal['place_squad_action'], tuple[int, int|None]] |


    # === Squad Phase ===
    tuple[Literal['activate_squad_move_action', 'activate_squad_attack_action'], int] |
    tuple[Literal['move_squad_action'], tuple[int, float]] |
    tuple[Literal['declare_squad_target_action'], int | tuple[int, HullSection]] |

    # None action
    tuple[Literal['initialize_game',
                  'pass_command_token',
                  'pass_repair',
                  'pass_attack', 
                  'pass_attack_effect', 
                  'pass_additional_squad_target',
                  'pass_defense_token', 
                  'pass_critical',
                  'pass_activate_squad',
                  'pass_move_squad',
                  'pass_attack_squad'
                  ], None] 
)


def get_action_str(game : Armada, action : ActionType) -> str | None:
    action_str = None

    match action:
        case 'set_command_action', (ship_id, command) :
            action_str = f'Set {Command(command)} Command on {game.ships[ship_id]}'

        case 'activate_ship_action', ship_id:
            action_str = f'Activate Ship: {game.ships[ship_id]}'


        case 'reveal_command_action', command :
            action_str = f'{game.active_ship} reveals {Command(command)} Command'

        case 'gain_command_token_action', command :
            action_str = f'{game.active_ship} gains {Command(command)} Token'

        case 'discard_command_token_action', command :
            action_str = f'{game.active_ship} discards {Command(command)} Token'

        case 'resolve_repair_command_action', (use_dial, use_token) :
            if use_dial or use_token :
                action_str = f'Resolve Repair Command{" (Dial)" if use_dial else ""}{" (Token)" if use_token else ""}'
            else : action_str = None

        case 'move_shield_action', (from_hull, to_hull) :
            action_str = f'Move Shield from {HullSection(from_hull)} to {HullSection(to_hull)}'

        case 'choose_target_ship_action', defend_ship_id :
            defend_ship = game.ships[defend_ship_id]
            action_str = f'Choose Target Ship: {defend_ship}'

        case 'declare_target_action', (attack_hull, defend_hull):
            action_str = f'Declare Attack: from {game.active_ship} {HullSection(attack_hull)} to {game.defend_ship} {HullSection(defend_hull)}'


        case 'declare_additional_squad_target_action', squad_id :
            if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
            defend_squad = game.squads[squad_id]
            action_str = f'Declare Additional Squad Target: from {game.ships[game.attack_info.attack_ship_id]} to {defend_squad}'


        case 'gather_dice_action', dice_to_remove :
            if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
            action_str = f'Gather Dice {game.attack_info.dice_to_roll} {f"(Remove {Dice(dice_to_remove.index(1))} Die)" if any(dice_to_remove) else ""}'

        case 'use_confire_token_action', dice :
            action_str = f'Use Confire Token to Reroll {dice_icon(dice)}'

        case 'roll_dice_action', dice :
            dice_result = dice_icon(dice)
            action_str = f'Dice Roll {dice_result}'

        case 'spend_accuracy_action', (dice_type, index) :
            if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
            defend_ship = game.ships[game.attack_info.defend_ship_id]
            token = defend_ship.defense_tokens[index]
            action_str = f'Spend {Dice(dice_type)} Accuracy : {token}'

        case 'spend_evade_token_action', (index, evade_dice) :
            if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
            defend_ship = game.ships[game.attack_info.defend_ship_id]
            token = defend_ship.defense_tokens[index]
            action_str = f'Spend {token} Token on {dice_icon(evade_dice)} ({AttackRange(game.attack_info.attack_range)} Range)'

        case 'spend_redirect_token_action', (index, hull) :
            if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
            defend_ship = game.ships[game.attack_info.defend_ship_id]
            token = defend_ship.defense_tokens[index]
            action_str = f'Spend {token} Token to {HullSection(hull)}'

        case 'spend_defense_token_action', index :
            if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
            defend_ship = game.ships[game.attack_info.defend_ship_id]
            action_str = f'Spend {defend_ship.defense_tokens[index]} Token'
            
        case 'resolve_damage_action', redirect_damage :
            if game.attack_info is None : raise ValueError('Need attack info to resolve attack effect')
            action_str = f'Resolve Total {game.attack_info.total_damage} Damage'
            if redirect_damage : 
                redirect_hull, damage = redirect_damage
                action_str += f', Redirect {[f"{damage} to {HullSection(redirect_hull)}"]}'


        case 'determine_course_action', (course, placement) :
            if game.active_ship is None : raise ValueError('Need active ship to determine course')
            dial_used, token_used = game.active_ship.nav_command_used(course)
            action_str = f'Determine Course: {course}, Placement: {("Right" if placement == 1 else "Left")} {" (Dial Spent)" if dial_used else ""} {" (Token Spent)" if token_used else ""}'


        case 'activate_squad_move_action' | 'activate_squad_attack_action', squad_id :
            squad = game.squads[squad_id]
            action_str = f'Activate Squad: {squad} to {"Move" if action[0]=="activate_squad_move_action" else "Attack"}'

        case 'move_squad_action', (squad_id, distance) :
            action_str = f'Move Squad: {game.active_squad} Distance: {distance}'

        case 'place_squad_action', (squad_id, coords_index) :
            squad = game.squads[squad_id]
            if coords_index is None :
                action_str = f'Place Squad: {squad} (Destroy)'
            else :
                index_to_placement = lambda x: (x * 2 + 1) // 11
                place_hull : HullSection = HullSection(index_to_placement(coords_index))
                action_str = f'Place Squad: {squad} at {game.active_ship} {place_hull}'

        case 'declare_squad_target_action', defender :
            if isinstance(defender, tuple) :
                defend_ship_id, defend_hull = defender
                defend_ship = game.ships[defend_ship_id]
                action_str = f'Declare Squad Attack: from {game.active_squad} to {defend_ship} {HullSection(defend_hull)}'
            else :
                defend_squad_id = defender
                defend_squad = game.squads[defend_squad_id]
                action_str = f'Declare Squad Attack: from {game.active_squad} to {defend_squad}'

        
        case _:
            # if 'pass' in action[0] : return
            action_str = f'{action[0].replace("_action", "").replace("_", " ").title().strip()} {f": {action[1]}" if action[1] else ""}'
    return action_str

if __name__ == '__main__':
    print(phase_type)