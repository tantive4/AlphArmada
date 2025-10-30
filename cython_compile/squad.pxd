from defense_token cimport DefenseToken
from armada cimport Armada
cdef class Squad:
    cdef :
        public Armada game
        public int player, max_hull, speed, point, id, hull
        public str name
        public tuple battery, anti_squad, coords
        public object overlap_ship_id
        public dict defense_tokens
        public bint unique, swarm, escort, bomber, destroyed, activated, can_attack, can_move
    cdef object get_snapshot(self)
    cdef void revert_snapshot(self, object snapshot)
    cpdef tuple get_squad_hash_state(self)