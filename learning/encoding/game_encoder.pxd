from armada cimport Armada
cimport numpy as cnp

cpdef tuple get_terminal_value(Armada game)

cpdef tuple encode_game_state(Armada game, 
                              float[:] scalar_buffer, 
                              float[:, :] ship_entity_buffer, 
                              float[:, :] ship_coords_buffer, 
                              float[:, :, :] ship_def_token_buffer, 
                              cnp.uint8_t[:, :, :, :] spatial_buffer, 
                              float[:, :, :] relation_buffer)

cdef void encode_scalar_features(Armada game, float[:] scalar_view)
cdef void encode_ship_entity_features(Armada game, 
                                      float[:, :] ship_view_buffer, 
                                      float[:, :] coords_view_buffer, 
                                      float[:, :, :] def_token_view_buffer)
# cdef void encode_squad_entity_features(Armada game)
cdef void encode_spatial_mask(Armada game, cnp.uint8_t[:, :, :, :] planes_view)
cdef void encode_relation_matrix(Armada game, float[:, :, :] rel_matrix)