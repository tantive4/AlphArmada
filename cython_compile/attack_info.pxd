# cython: language_level=3

cdef class AttackInfo:
    
    # --- Declare ALL public C-level attributes ---
    cdef public bint is_attacker_ship
    cdef public bint is_defender_ship
    cdef public bint obstructed
    cdef public bint con_fire_dial
    cdef public bint con_fire_token
    cdef public bint bomber
    cdef public bint swarm
    
    cdef public int attack_ship_id
    cdef public int attack_squad_id
    cdef public int defend_ship_id
    cdef public int defend_squad_id
    cdef public int total_damage
    
    cdef public object attack_hull
    cdef public object defend_hull
    cdef public object attack_range
    cdef public object dice_to_roll
    cdef public object squadron_target
    cdef public object phase
    cdef public object attack_pool_result
    cdef public object spent_token_indices
    cdef public object spent_token_types
    cdef public object redirect_hull
    cdef public object critical


    cpdef dict get_snapshot(self)