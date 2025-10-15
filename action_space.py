import json
import itertools

from action_phase import Phase, ActionType
from enum_class import *
from dice import dice_choices, FULL_DICE_POOL
from defense_token import TokenType, TOKEN_DICT

def _make_hashable(action_value):
    """
    Recursively converts an action's payload (often loaded from JSON as a list)
    into a fully hashable type (a tuple).
    """
    if isinstance(action_value, (list, tuple)):
        return tuple(_make_hashable(item) for item in action_value)
    return action_value

def generate_all_maps():
    """
    Generates the total action space for every game phase.
    
    Skip Chance node (dice roll) & Information node (command stack)
    """
    all_maps_raw = {}

    # Define the maximums your game will support
    MAX_SHIPS = 6 
    MAX_SQUADS = 9
    MAX_DEFENSE_TOKENS_PER_TYPE = 2
    SQUAD_PLACEMENT_RESOLUTION = 22
    
    for phase in Phase:
        actions : list[ActionType] = []
        # Use simple loops to generate every possible action combination
        match phase:
            
            # === Command Phase ===
            case Phase.COMMAND_PHASE :
                actions = [('set_command_action', (ship_id, command)) for command in Command for ship_id in range(MAX_SHIPS)]
            

            # === Ship Phase ===
            case Phase.SHIP_ACTIVATE :
                actions = [('activate_ship_action', ship_id) for ship_id in range(MAX_SHIPS)]
            
            # Reveal Command 
            case Phase.SHIP_REVEAL_COMMAND_DIAL : # information node
                pass

            case Phase.SHIP_GAIN_COMMAND_TOKEN :
                actions = [('gain_command_token_action', command) for command in Command]
                actions.append(('pass_command_token', None))
                
            case Phase.SHIP_DISCARD_COMMAND_TOKEN :
                actions = [('discard_command_token_action', command) for command in Command]

            case Phase.SHIP_RESOLVE_SQUAD:
                actions = [('resolve_squad_command_action', (dial, token)) for dial in (True, False) for token in (True, False)]

            case Phase.SHIP_RESOLVE_REPAIR :
                actions = [('resolve_repair_command_action', (dial, token)) for dial in (True, False) for token in (True, False)]

            case Phase.SHIP_USE_ENGINEER_POINT :
                actions = [('repair_hull_action', None)]
                actions.extend([('recover_shield_action', hull) for hull in HullSection])
                actions.extend([('move_shield_action', (from_hull, to_hull)) for from_hull in HullSection for to_hull in HullSection if from_hull != to_hull])
                actions.append(('pass_repair', None))

            # Attack 
            case Phase.SHIP_DECLARE_TARGET :
                actions = [('declare_target_action', (attack_hull, (defend_ship_id, defend_hull)))
                           for attack_hull, defend_ship_id, defend_hull in itertools.product(HullSection, range(MAX_SHIPS), HullSection)]
                actions.extend([('declare_target_action', (attack_hull, defend_squad_id)) for attack_hull, defend_squad_id in itertools.product(HullSection, range(MAX_SQUADS))])
                actions.append(('pass_attack', None))

            # Execute Maneuver
            case Phase.SHIP_DETERMINE_COURSE :
                actions = [('determine_course_action', ((), 0))]
                for speed in range(1, 5):
                    # All possible yaw combinations (-2 to 2 for each joint)
                    yaw_options = range(-2, 3)
                    all_courses = list(itertools.product(yaw_options, repeat=speed))

                    for course in all_courses:
                        # Placement can be Left (-1) or Right (1)
                        for placement in [-1, 1]:
                            if course[-1] * placement < 0: continue 
                            actions.append(('determine_course_action', (course, placement)))

            case Phase.SHIP_PLACE_SQUAD :
                for squad_id in range(MAX_SQUADS) :
                    actions.extend([('place_squad_action', (squad_id, coord_index)) for coord_index in range(SQUAD_PLACEMENT_RESOLUTION)])
                    actions.append(('place_squad_action', (squad_id, None)))


            # === SQUADRON_PHASE ===
            case Phase.SQUAD_ACTIVATE:
                actions = [('activate_squad_move_action', squad_id) for squad_id in range(MAX_SQUADS)] 
                actions.extend([('activate_squad_attack_action', squad_id) for squad_id in range(MAX_SQUADS)])
                actions.append(('pass_activate_squad', None))
                
            case Phase.SQUAD_DECLARE_TARGET :
                actions = [('declare_squad_target_action', (defend_ship_id, defend_hull))
                           for defend_ship_id, defend_hull in itertools.product(range(MAX_SHIPS), HullSection)]
                actions.extend([('declare_squad_target_action', defend_squad_id) for defend_squad_id in range(MAX_SQUADS)])
                actions.append(('pass_attack_squad', None))

            case Phase.SQUAD_MOVE :
                moves : list[tuple[int, float]] = []
                for speed in range(6) :
                    if speed == 0 :
                        moves.append((0, 0))
                        continue
                    for angle in range(0, 360, 90 // speed) :
                        moves.append((speed, angle))

                actions = [('move_squad_action', move) for move in moves]
                actions.append(('pass_move_squad', None))


            # === Attack Step ===
            case Phase.ATTACK_GATHER_DICE :
                for dice_type in Dice :
                    dice_to_remove = [0,0,0]
                    dice_to_remove[dice_type] = 1
                    dice_to_remove = tuple(dice_to_remove)
                    actions.append(('gather_dice_action', dice_to_remove))
                actions.append(('gather_dice_action', (0, 0, 0)))

            case Phase.ATTACK_ROLL_DICE : # chance node
                pass

            case Phase.ATTACK_RESOLVE_EFFECTS :
                actions = [('spend_accuracy_action', (dice_type, index)) for dice_type in (Dice.BLUE, Dice.RED) for index in range(len(TokenType) * MAX_DEFENSE_TOKENS_PER_TYPE)]
                actions.extend([('resolve_con-fire_command_action', (use_dial, use_token)) for use_dial, use_token in itertools.product((True, False), repeat=2) if use_dial or use_token])
                actions.extend([('use_confire_dial_action', tuple(1 if i == dice else 0 for i in range(3))) for dice in Dice])
                actions.extend([('use_confire_token_action', dice) for dice in dice_choices(FULL_DICE_POOL, 1)])
                actions.extend([('swarm_reroll_action', dice) for dice in dice_choices(FULL_DICE_POOL, 1)])
                actions.append(('pass_attack_effect', None))

            case Phase.ATTACK_SPEND_DEFENSE_TOKENS :
                for index, token in TOKEN_DICT.items():
                    if token.type == TokenType.REDIRECT :
                        actions.extend([('spend_redirect_token_action',(index, hull)) for hull in HullSection])   

                    elif token.type == TokenType.EVADE :
                        evade_dice_choices = dice_choices(FULL_DICE_POOL, 1)
                        for dice_choice in evade_dice_choices :
                            actions.append(('spend_evade_token_action', (index, dice_choice)))

                    else : actions.append(('spend_defense_token_action', index))

                actions.append(('pass_defense_token', None))

            case Phase.ATTACK_USE_CRITICAL_EFFECT :
                actions = [('use_critical_action', critical) for critical in Critical]
                actions.append(('pass_critical', None))

            case Phase.ATTACK_RESOLVE_DAMAGE:
                # consider standard redirect with max damage 4
                actions = [('resolve_damage_action', (hull, damage)) for hull in HullSection for damage in range(5)]
                actions.append(('resolve_damage_action', None))

            case Phase.ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET :
                actions = [('declare_additional_squad_target_action', defend_squad_id) for defend_squad_id in range(MAX_SQUADS)]
                actions.append(('pass_additional_squad_target', None))

            case _:
                print(f"Warning: No action generation logic defined for phase {phase.name}")

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



    def enum_serializer(obj):
        if isinstance(obj, (Command, HullSection, Dice, Critical)):
            return obj.name # Save enums by their string name
        if isinstance(obj, set):
            return list(obj) # Convert sets to lists
        return obj.__dict__

    # Write the generated data to a JSON file
    with open('action_space.json', 'w') as f:
        # Use the custom serializer for enums
        json.dump(all_maps_hashable, f, indent=2, default=enum_serializer)
    
    print("action_space.json has been generated successfully!")

class ActionManager:
    """
    Loads the pre-computed total action space from a JSON file and creates
    the essential action-to-index lookup dictionary for each phase.
    """
    def __init__(self, filepath='action_space.json'):
        self.action_maps: dict[Phase, dict[ActionType, int]] = {}
        
        with open(filepath, 'r') as f:
            raw_maps = json.load(f)

        for phase_name, total_actions_list in raw_maps.items():
            # This check will skip phases that might not be in your Phase enum
            if phase_name in Phase.__members__:
                phase = Phase[phase_name]
                
                # Create the action-to-index dictionary from the loaded list.
                action_to_index_dict :dict[ActionType, int] = {
                    # Convert the action_value to a hashable tuple before creating the key
                    (action_name, _make_hashable(action_value)): i
                    for i, (action_name, action_value) in enumerate(total_actions_list)
                }
                
                self.action_maps[phase] = action_to_index_dict

    def get_action_map(self, phase: Phase) -> dict[ActionType, int]:
        """Returns the action map for a given game phase."""
        return self.action_maps[phase]



    
# --- Main execution block for testing ---
if __name__ == '__main__':
    generate_all_maps()

    action_manager = ActionManager()
    print("ActionManager initialized successfully.")
    print("\n--- Action Space Sizes ---")
    for phase in Phase:
        if phase in action_manager.action_maps:
            action_map = action_manager.get_action_map(phase)
            print(f"Phase: {phase.name:<40} Total Actions: {len(action_map)}")