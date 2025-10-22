from defense_token cimport DefenseToken
from armada cimport Armada

cdef class Ship:
    cdef :
        public Armada game
        public int max_hull, point, command_value, engineer_value, squad_value, id, speed, hull, engineer_point, attack_count
        public float x, y, orientation
        public str name
        public object player, size_class, battery, battery_range, anti_squad, anti_squad_range, defense_tokens, nav_chart, max_shield, _course_cache, 
        public bint destroyed, activated
        public tuple base_size, token_size, shield, command_stack, command_dial, command_token, resolved_command, attack_impossible_hull, repaired_hull
        public object template_base_vertices, template_token_vertices, template_targeting_points_and_maneuver_tool_insert, template_hull_vertices
    cdef object get_snapshot(self)
    cdef void revert_snapshot(self, object snapshot)
    cpdef object get_ship_hash_state(self)