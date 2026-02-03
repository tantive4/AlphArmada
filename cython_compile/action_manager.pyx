# cython: profile=True

from enum_class import *
import json
from action_phase import Phase, ActionType
from action_space import _make_hashable
from configs import Config

cdef class ActionManager:
    """
    Loads the pre-computed total action space from a JSON file and creates
    the essential action-to-index lookup dictionary for each phase.
    """

    def __init__(self, filepath='action_space.json'):
        self.action_maps = []

        pointer_action_names = {
            "activate_ship_action",
            "choose_target_ship_action",
            "spend_accuracy_action",
            "spend_defense_token_action"
        }

        with open(filepath, 'r') as f:
            raw_maps = json.load(f)

        for phase_name, total_actions_list in raw_maps.items():
            if phase_name not in Phase.__members__:
                raise ValueError(f"Phase '{phase_name}' not found in Phase enum.")

            action_to_index_dict = {}
            static_count = 0

            # Rewrite loop as requested: unrolled logic for pointer vs static indices
            for i, (action_name, action_value) in enumerate(total_actions_list):
                # Convert value to hashable
                action_key = (action_name, _make_hashable(action_value))
                
                # Check if this is a pointer action
                is_pointer = action_name in pointer_action_names

                if is_pointer:
                    # Pointer actions keep their original index 'i'
                    # (User assumes these are encoded at the front of the list 0..K)
                    action_to_index_dict[action_key] = i
                else:
                    # Static actions start from N (MAX_SHIPS) + counter
                    action_to_index_dict[action_key] = Config.MAX_SHIPS + static_count
                    static_count += 1


            self.action_maps.append(action_to_index_dict)
        self.max_action_space = max(len(amap) for amap in self.action_maps) + Config.MAX_SHIPS

    cpdef dict get_action_map(self, int phase):
        """Returns the action map for a given game phase."""
        return self.action_maps[phase]

    cdef int get_action_index(self, int phase, tuple action):
        """Returns the action index for a given game phase."""
        cdef dict action_map = self.action_maps[phase]
        return action_map[action]