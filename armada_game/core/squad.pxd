from armada cimport Armada
from ship cimport Ship
cdef class Squad:
    cdef :
        public Armada game
        public int team, max_hull, speed, point, id, hull, counter
        public int battery_dice, anti_squad_dice
        public str name
        public tuple battery, anti_squad, coords
        public bint destroyed, activated

        public bint unique, swarm, escort, bomber, heavy

    cdef object get_snapshot(self)
    cdef void revert_snapshot(self, object snapshot)
    cpdef tuple get_squad_hash_state(self)
    cpdef void status_phase(self)
    cpdef void destroy(self)
    cpdef void defend(self, int total_damage)
    cpdef void move_by_cells(self, int di, int dj)
    cpdef void place(self, double x, double y)
    cpdef tuple gather_dice(self, bint is_ship, bint is_counter)
    cpdef bint is_obstruct_q2q(self, Squad to_squad)
    cpdef bint is_obstruct_q2s(self, Ship to_ship, int to_hull)
