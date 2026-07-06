# cython: profile=True

from armada_game.helpers.enum_class import *
import json
from armada_game.helpers.action_phase import Phase, ActionType
from armada_game.helpers.action_space import _make_hashable
from learning.params.configs import Config
from armada_game.helpers.paths import data_path

cdef class ActionManager:
    """
    Loads the pre-computed total action space from a JSON file and creates
    the essential action-to-index lookup dictionary for each phase.
    """

    def __init__(self, filepath=None):
        self.action_maps = []

        ship_pointer_action_names = {
            "activate_ship_action",
            "choose_target_ship_action",
        }
        token_pointer_action_names = {
            "spend_accuracy_action",
            "spend_defense_token_action"
        }
        token_action_offset = Config.MAX_SHIPS
        static_action_offset = Config.MAX_SHIPS + Config.MAX_DEFENSE_TOKENS

        if filepath is None:
            filepath = data_path("action_space.json")

        with open(filepath, 'r', encoding="utf-8") as f:
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

                if action_name in ship_pointer_action_names:
                    action_to_index_dict[action_key] = action_value
                elif action_name in token_pointer_action_names:
                    action_to_index_dict[action_key] = token_action_offset + action_value
                else:
                    action_to_index_dict[action_key] = static_action_offset + static_count
                    static_count += 1


            self.action_maps.append(action_to_index_dict)
        self.max_action_space = max(max(amap.values()) + 1 for amap in self.action_maps if amap)

    cpdef dict get_action_map(self, int phase):
        """Returns the action map for a given game phase."""
        return self.action_maps[phase]

    cdef int get_action_index(self, int phase, tuple action):
        """Returns the action index for a given game phase."""
        cdef dict action_map = self.action_maps[phase]
        return action_map[action]
