# generate_action_space.py
import json
import itertools
from game_phase import GamePhase, ActionType
from ship import Command, HullSection
from dice import Dice, Critical, dice_choice_combinations, FULL_DICE_POOL
from defense_token import TokenType, TOKEN_DICT

def _make_hashable(action_value):
    """
    Recursively converts an action's payload into a fully hashable type.
    """
    if isinstance(action_value, list):
        return tuple(_make_hashable(item) for item in action_value)
    if isinstance(action_value, dict):
        return tuple((k, _make_hashable(v)) for k, v in action_value.items())
    return action_value

# --- Helper to make enums JSON-serializable ---
def enum_serializer(obj):
    if isinstance(obj, (Command, HullSection, Dice, Critical)):
        return obj.name # Save enums by their string name
    if isinstance(obj, set):
        return list(obj) # Convert sets to lists
    return obj.__dict__

def generate_all_maps():
    """Generates the total action space for every game phase."""
    all_maps_raw = {}

    # Define the maximums your game will support
    MAX_SHIPS = 6 
    MAX_DEFENSE_TOKENS_PER_TYPE = 2
    
    for phase in GamePhase:
        actions : list[ActionType.Action] = []
        # Use simple loops to generate every possible action combination
        match phase:
        
            case GamePhase.COMMAND_PHASE :
                actions = [('set_command_action', (ship_id, command)) for command in Command for ship_id in range(MAX_SHIPS)]
                actions.append(('pass_command', None))
            
            case GamePhase.SHIP_PHASE :
                actions = [('activate_ship_action', ship_id) for ship_id in range(MAX_SHIPS)]
                actions.append(('pass_ship_activation', None))
            
            # Reveal Command Sequence
            case GamePhase.SHIP_REVEAL_COMMAND_DIAL :
                actions = [('reveal_command_action', command) for command in Command]

            case GamePhase.SHIP_GAIN_COMMAND_TOKEN :
                actions = [('gain_command_token_action', command) for command in Command]
                actions.append(('pass_command_token', None))
                
            case GamePhase.SHIP_DISCARD_COMMAND_TOKEN :
                actions = [('discard_command_token_action', command) for command in Command]


            # Engineering Sequence
            case GamePhase.SHIP_RESOLVE_REPAIR :
                actions = [('resolve_repair_command_action', (dial, token)) for dial, token in zip((True, False), (True, False))]

            case GamePhase.SHIP_USE_ENGINEER_POINT :
                actions = [('repair_hull_action', None)]
                actions.extend([('recover_shield_action', hull) for hull in HullSection])
                actions.extend([('move_shield_action', (from_hull, to_hull)) for from_hull in HullSection for to_hull in HullSection])
                actions.append(('pass_repair', None))

            # Attack Sequence
            case GamePhase.SHIP_ATTACK_DECLARE_TARGET :
                actions = [('declare_target_action', (attack_hull, defend_ship_id, defend_hull))
                           for attack_hull, defend_ship_id, defend_hull in zip(HullSection, range(MAX_SHIPS), HullSection)]
                actions.append(('pass_attack', None))
                
            case GamePhase.SHIP_ATTACK_GATHER_DICE :
                for dice_type in Dice :
                    dice_to_remove = {dice_type : 0 for dice_type in Dice}
                    dice_to_remove[dice_type] = 1
                    actions.append(('gather_dice_action', dice_to_remove))
                actions = [('gather_dice_action', {dice_type : 0 for dice_type in Dice})]

            case GamePhase.SHIP_ATTACK_ROLL_DICE : # chance node
                pass

            case GamePhase.SHIP_ATTACK_RESOLVE_EFFECTS :
                # spend accuracy
                actions = [('spend_accuracy_action', (dice_type, index)) for dice_type in (Dice.BLUE, Dice.RED) for index in range(len(TokenType) * MAX_DEFENSE_TOKENS_PER_TYPE)]

                # use con-fire command
                actions.extend([('resolve_con-fire_command_action', (use_dial, use_token)) for use_dial, use_token in ((True, False), (True, False)) if use_dial or use_token])
                actions.extend([('use_confire_dial_action', {dice: 1}) for dice in Dice])
                actions.extend([('use_confire_token_action', dice) for dice in dice_choice_combinations(FULL_DICE_POOL, 1)])
                
                actions.append(('pass_attack_effect', None))


            case GamePhase.SHIP_ATTACK_SPEND_DEFENSE_TOKENS :
                for index, token in TOKEN_DICT.items():
                    if token.type == TokenType.REDIRECT :
                        actions.extend([('spend_redicect_token_action',(index, hull)) for hull in HullSection])   

                    elif token.type == TokenType.EVADE :
                        # choose 1 die to affect
                        evade_dice_choices = dice_choice_combinations(FULL_DICE_POOL, 1)
                        for dice_choice in evade_dice_choices :
                            actions.append(('spend_evade_token_action', (index, dice_choice)))
                        # if defender is smaller, may choose 2 dice
                        discard_evade_choices = dice_choice_combinations(FULL_DICE_POOL, 2)
                        for dice_choice in discard_evade_choices :
                            actions.append(('spend_evade_token_action', (index, dice_choice))) # dice choice edit!!!

                    else : actions.append(('spend_defense_token_action', index))

                actions.append(('pass_defense_token', None))

            case GamePhase.SHIP_ATTACK_USE_CRITICAL_EFFECT :
                actions = [('use_critical_action', critical) for critical in Critical]
                actions.append(('pass_critical', None))


            case GamePhase.SHIP_ATTACK_RESOLVE_DAMAGE:
                # consider standard redirect with max damage 4
                actions = [('resolve_damage_action', [(hull, damage)]) for hull in HullSection for damage in range(5)]

            # Maneuver Sequence
            case GamePhase.SHIP_MANEUVER_DETERMINE_COURSE :
                actions = [('determine_course_action',([], 0))]
                for speed in range(1, 5):
                    # All possible yaw combinations (-2 to 2 for each joint)
                    yaw_options = range(-2, 3)
                    all_courses = list(itertools.product(yaw_options, repeat=speed))

                    for course in all_courses:
                        # Placement can be Left (-1) or Right (1)
                        for placement in [-1, 1]:
                            actions.append(('determine_course_action', (list(course), placement)))

            case GamePhase.STATUS_PHASE :
                actions = [('status_phase', None)]
        
        if actions:
            all_maps_raw[phase.name] = actions
            
    # --- NEW: Post-processing step to make everything hashable ---
    all_maps_hashable = {}
    for phase_name, action_list in all_maps_raw.items():
        # Use a list comprehension to apply the conversion to the entire list
        all_maps_hashable[phase_name] = [
            (action_name, _make_hashable(action_value))
            for action_name, action_value in action_list
        ]

    # Write the generated data to a JSON file
    with open('action_space.json', 'w') as f:
        # Use the custom serializer for enums
        json.dump(all_maps_hashable, f, indent=2, default=enum_serializer)
    
    print("action_space.json has been generated successfully!")

if __name__ == '__main__':
    generate_all_maps()