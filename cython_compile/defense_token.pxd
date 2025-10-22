cdef class DefenseToken:
    cdef :
        public object type
        public bint readied, discarded, accuracy
    cdef tuple get_snapshot(self)
    cdef void revert_snapshot(self, tuple snapshot)