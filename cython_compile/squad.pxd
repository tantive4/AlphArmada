from armada cimport Armada
cdef class Squad:
    cdef :
        public Armada game
        public int player, max_hull, speed, point, id, hull
        public str name
        public tuple battery, anti_squad, coords
        public object overlap_ship_id
        public dict defense_tokens
        public bint destroyed, activated, can_attack, can_move

        public bint unique, swarm, escort, bomber, heavy
        public int counter

    cpdef bint is_overlap(self)
    cdef object get_snapshot(self)
    cdef void revert_snapshot(self, object snapshot)
    cpdef tuple get_squad_hash_state(self)
    cpdef bint in_distance(self, Squad other, float distance, float distance_sq)
    cpdef bint is_engaged(self)
    cpdef bint is_engage_with(self, Squad other)