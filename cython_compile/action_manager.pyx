# cython: profile=True

from enum_class import *
import json
from action_phase import Phase, ActionType
from action_space import _make_hashable

cdef class ActionManager:
    """
    Loads the pre-computed total action space from a JSON file and creates
    the essential action-to-index lookup dictionary for each phase.
    """

    def __init__(self, filepath='action_space.json'):
        self.action_maps = []
        
        with open(filepath, 'r') as f:
            raw_maps = json.load(f)

        for phase_name, total_actions_list in raw_maps.items():
            # This check will skip phases that might not be in your Phase enum
            if phase_name not in Phase.__members__:
                raise ValueError(f"Phase '{phase_name}' not found in Phase enum.")

            # Create the action-to-index dictionary from the loaded list.
            action_to_index_dict :dict[ActionType, int] = {
                # Convert the action_value to a hashable tuple before creating the key
                (action_name, _make_hashable(action_value)): i
                for i, (action_name, action_value) in enumerate(total_actions_list)
            }

            self.action_maps.append(action_to_index_dict)
        self.max_action_space = max(len(amap) for amap in self.action_maps)

    cpdef dict get_action_map(self, int phase):
        """Returns the action map for a given game phase."""
        return self.action_maps[phase]

    cdef int get_action_index(self, int phase, tuple action):
        """Returns the action index for a given game phase."""
        cdef dict action_map = self.action_maps[phase]
        return action_map[action]