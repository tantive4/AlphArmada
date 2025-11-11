from armada cimport Armada
from ship cimport Ship
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
    cpdef void status_phase(self)
    cpdef void destroy(self)
    cpdef void start_activation(self)
    cpdef void end_activation(self)
    cpdef void defend(self, int total_damage)
    cpdef list get_valid_target(self)
    cpdef bint is_obstruct_q2q(self, Squad to_squad)
    cpdef bint is_obstruct_q2s(self, Ship to_ship, int to_hull)
    cpdef list get_critical_effect(self, bint black_crit, bint blue_crit, bint red_crit)
    cpdef void move(self, int speed, float angle)
    cpdef list get_valid_moves(self)
    cpdef void place_squad(self, tuple coords)
    cpdef bint out_of_board(self)
    cpdef tuple gather_dice(self, bint is_ship, bint is_counter)