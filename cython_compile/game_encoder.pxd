from armada cimport Armada
cimport numpy as cnp

cpdef tuple get_terminal_value(Armada game)
cpdef dict encode_game_state(Armada game)
cdef void encode_scalar_features(Armada game)
cdef void encode_ship_entity_features(Armada game)
cdef void encode_squad_entity_features(Armada game)
cdef void encode_spatial_mask(Armada game)
cdef void encode_relation_matrix(Armada game)