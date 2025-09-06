# action_space.py
import json
from game_phase import GamePhase

def _make_hashable(action_value):
    """
    Recursively converts an action's payload (often loaded from JSON as a list)
    into a fully hashable type (a tuple).
    """
    if isinstance(action_value, list):
        return tuple(_make_hashable(item) for item in action_value)
    # Dictionaries are not used in your action space, but this is good practice
    if isinstance(action_value, dict):
        return tuple((k, _make_hashable(v)) for k, v in action_value.items())
    return action_value

class ActionManager:
    """
    Loads the pre-computed total action space from a JSON file and creates
    the essential action-to-index lookup dictionary for each phase.
    """
    def __init__(self, filepath='action_space.json'):
        self.action_maps = {}
        
        with open(filepath, 'r') as f:
            # Loads the raw map where keys are phase names and values are lists of actions
            raw_maps = json.load(f)

        for phase_name, total_actions_list in raw_maps.items():
            phase = GamePhase[phase_name]
            
            # --- THIS IS THE CRUCIAL STEP YOU IDENTIFIED ---
            # Create the action-to-index dictionary from the loaded list.
            # The key is the hashable (name, value) tuple, and the value is its index.
            action_to_index_dict = {
                (action_name, action_value): i
                for i, (action_name, action_value) in enumerate(total_actions_list)
            }
            
            self.action_maps[phase] = {
                'total_actions': total_actions_list,
                'action_to_index': action_to_index_dict  # <--- Store the dictionary
            }
        print("ActionManager initialized and action-to-index maps created successfully.")

    def get_action_map(self, phase: GamePhase) -> dict | None:
        """Returns the action map for a given game phase."""
        return self.action_maps.get(phase)