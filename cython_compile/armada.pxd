
from ship cimport Ship 
from squad cimport Squad
from attack_info cimport AttackInfo

cdef class Armada:
    cdef :
        public int round, first_player, second_player, current_player, decision_player, simulation_player, squad_activation_count, image_counter, para_index
        public float player_edge, short_edge, winner 
        public list ships, squads
        public object phase, active_ship, active_squad, attack_info
        public bint debuging_visual

    
    # Declare the C-level methods
    cpdef object get_snapshot(self)
    cdef void revert_snapshot(self, object snapshot)
    cdef void update_decision_player(self)
    cpdef int get_point(self, int player)
