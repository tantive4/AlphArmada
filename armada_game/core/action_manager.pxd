cdef class ActionManager:
    cdef public list action_maps
    cdef public int max_action_space
    cdef int get_action_index(self, int phase, tuple action)
    cpdef dict get_action_map(self, int phase)