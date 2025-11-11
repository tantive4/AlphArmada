cimport numpy as cnp
from ship cimport Ship
from squad cimport Squad

cdef class Armada:
    cdef :
        public int round, first_player, second_player, current_player, decision_player, simulation_player, squad_activation_count, image_counter, para_index
        public float player_edge, short_edge, winner 
        public list ships, squads
        public object phase, active_ship, active_squad, attack_info
        public bint debuging_visual
        public object scalar_encode_array
        public object relation_encode_array, ship_encode_array, squad_encode_array
        public object spatial_encode_array

    
    # Declare the C-level methods
    cpdef object get_snapshot(self)
    cpdef void revert_snapshot(self, object snapshot)
    cdef void update_decision_player(self)
    cpdef int get_point(self, int player)
    cpdef list get_valid_actions(self)
    cpdef void apply_action(self, tuple action)
    cpdef void visualize_action(self, object action)
    cpdef void deploy_ship(self, Ship ship, float x, float y, float orientation, int speed)
    cpdef void deploy_squad(self, Squad squad, float x, float y)
    cdef bint total_destruction(self, int player)
    cdef list get_valid_ship_activation(self, int player)
    cdef list get_valid_squad_activation(self, int player)
    cdef void status_phase(self)