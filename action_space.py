import json
from game_phase import GamePhase

def _make_hashable(action_value):
    """
    Recursively converts an action's payload (often loaded from JSON as a list)
    into a fully hashable type (a tuple).
    """
    if isinstance(action_value, list | tuple):
        return tuple(_make_hashable(item) for item in action_value)
    if isinstance(action_value, dict):
        # Sort by key to ensure consistency
        return tuple(sorted((k, _make_hashable(v)) for k, v in action_value.items()))
    return action_value

class ActionManager:
    """
    Loads the pre-computed total action space from a JSON file and creates
    the essential action-to-index lookup dictionary for each phase.
    """
    def __init__(self, filepath='action_space.json'):
        self.action_maps: dict[GamePhase, dict] = {}
        
        with open(filepath, 'r') as f:
            raw_maps = json.load(f)

        for phase_name, total_actions_list in raw_maps.items():
            # This check will skip phases that might not be in your GamePhase enum
            if phase_name in GamePhase.__members__:
                phase = GamePhase[phase_name]
                
                # Create the action-to-index dictionary from the loaded list.
                action_to_index_dict = {
                    # Convert the action_value to a hashable tuple before creating the key
                    (action_name, _make_hashable(action_value)): i
                    for i, (action_name, action_value) in enumerate(total_actions_list)
                }
                
                self.action_maps[phase] = {
                    'total_actions': total_actions_list,
                    'action_to_index': action_to_index_dict
                }

    def get_action_map(self, phase: GamePhase) -> dict:
        """Returns the action map for a given game phase."""
        return self.action_maps[phase]

# --- Main execution block for testing ---
if __name__ == '__main__':
    try:
        action_manager = ActionManager()
        print("ActionManager initialized successfully.")
        print("\n--- Action Space Sizes ---")
        for phase in GamePhase:
            if phase in action_manager.action_maps:
                action_map = action_manager.get_action_map(phase)
                print(f"Phase: {phase.name:<35} Total Actions: {len(action_map['total_actions'])}")
    except Exception as e:
        print(f"An error occurred: {e}")
