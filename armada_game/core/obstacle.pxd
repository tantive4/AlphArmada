cimport numpy as cnp

cdef class Obstacle:
    cdef: 
        public object type
        public float x, y, orientation
        public cnp.ndarray coordinates
        public int index
        public tuple hash_state

    cpdef void place_obstacle(self, float x, float y, float orientation, bint flip)
    cpdef tuple get_hash_state(self)