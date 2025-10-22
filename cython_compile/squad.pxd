from defense_token cimport DefenseToken
from armada cimport Armada
cdef class Squad:
    cdef :
        public Armada game
        public int max_hull, speed, point, id, hull
        public str name
        public tuple battery, anti_squad, coords
        public object defense_tokens, overlap_ship_id, player
        public bint unique, swarm, escort, bomber, destroyed, activated, can_attack, can_move
    cdef object get_snapshot(self)
    cdef void revert_snapshot(self, object snapshot)