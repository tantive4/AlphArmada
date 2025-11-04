from armada cimport Armada
cimport numpy as cnp

cpdef tuple get_terminal_value(Armada game)
cpdef dict encode_game_state(Armada game)
cdef cnp.ndarray[cnp.float32_t, ndim=1] encode_scalar_features(Armada game)
cdef cnp.ndarray[cnp.float32_t, ndim=2] encode_ship_entity_features(Armada game)
cdef cnp.ndarray[cnp.float32_t, ndim=2] encode_squad_entity_features(Armada game)
cdef cnp.ndarray[cnp.float32_t, ndim=3] encode_spatial_features(Armada game, tuple resolution)
cdef cnp.ndarray[cnp.float32_t, ndim=2] encode_relation_matrix(Armada game)