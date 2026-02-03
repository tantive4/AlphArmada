from armada cimport Armada
from squad cimport Squad
from obstacle cimport Obstacle
cimport numpy as cnp

cdef class Ship:
    cdef :
        public Armada game
        public int team, faction, max_hull, point, command_value, engineer_value, squad_value, id, speed, hull, engineer_point, attack_count
        public float x, y, orientation
        public cnp.ndarray nav_chart_vector
        public str name
        public int size_class
        public dict battery, battery_range, anti_squad_range, nav_chart, max_shield, _course_cache
        public bint destroyed, activated
        public tuple base_size, token_size, shield, defense_tokens, command_stack, command_dial, command_token, resolved_command, attack_history, repaired_hull, anti_squad
        public object template_base_vertices, template_token_vertices, template_targeting_points_and_maneuver_tool_insert, template_hull_vertices
        public cnp.ndarray rotation_matrix
    cdef object get_snapshot(self)
    cdef void revert_snapshot(self, object snapshot)
    cpdef object get_ship_hash_state(self)
    cpdef void asign_command(self, int command)
    cpdef void destroy(self)
    cpdef void status_phase(self)
    cpdef void end_activation(self)
    cpdef void execute_maneuver(self, tuple course, int placement)
    cpdef set move_ship(self, tuple course, int placement, set overlap_ships)
    cpdef bint is_overlap_squad(self)
    cpdef list get_valid_squad_placement(self, Squad squad)
    cpdef bint is_obstruct_s2s(self, int from_hull, Ship to_ship, int to_hull)
    cpdef bint is_obstruct_s2q(self, int from_hull, Squad to_squad)
    cpdef tuple gather_dice(self, int attack_hull, int attack_range, bint is_ship)
    cpdef void defend(self, int defend_hull, int total_damage, object critical)
    cpdef list get_valid_ship_target(self, int attack_hull)
    cpdef list get_valid_target_hull(self, int attack_hull, Ship target_ship)
    cpdef list get_valid_squad_target(self, int attack_hull)
    cpdef list get_valid_attack_hull(self)
    cpdef list get_critical_effect(self, bint black_crit, bint blue_crit, bint red_crit)
    cpdef list get_squad_activation(self)
    cpdef set is_overlap(self)
    cpdef void overlap_damage(self, set overlap_list)
    cpdef bint out_of_board(self)
    cpdef list get_valid_speed(self)
    cpdef list get_valid_yaw(self, int speed, int joint)
    cpdef tuple nav_command_used(self, tuple course)
    cpdef bint is_standard_course(self, tuple course)
    cpdef list get_all_possible_courses(self, int speed)
    cpdef list get_valid_placement(self, tuple course)
    cpdef void spend_command_dial(self, int command)
    cpdef void spend_command_token(self, int command)
    cpdef bint check_overlap(self, object obstacle)
    cpdef void overlap_obstacle(self, Obstacle obstacle)