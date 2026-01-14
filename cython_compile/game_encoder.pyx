# cython: profile=True

from __future__ import annotations
from libc.math cimport sin, cos
from typing import TYPE_CHECKING

import numpy as np
cimport numpy as cnp
cnp.import_array()
import cython

from configs import Config
from action_phase import phase_type
from dice import *
from enum_class import *
from measurement import *
import cache_function as cache


from armada cimport Armada
from ship cimport Ship
from squad cimport Squad
from attack_info cimport AttackInfo
from defense_token cimport DefenseToken

cdef:
    int c_command_type = command_type
    int c_phase_type = phase_type
    int c_critical_type = critical_type
    int c_hull_type = hull_type
    int c_obstacle_type = obstacle_type
    int max_ships = <int>Config.MAX_SHIPS
    int max_squads = <int>Config.MAX_SQUADS
    int max_command_stack = <int>Config.MAX_COMMAND_STACK
    int max_defense_tokens = <int>Config.MAX_DEFENSE_TOKENS
    int max_squad_defense_tokens = <int>Config.MAX_SQUAD_DEFENSE_TOKENS
    int max_squad_value = <int>Config.GLOBAL_MAX_SQUAD_VALUE
    int max_engineer_value = <int>Config.GLOBAL_MAX_ENGINEER_VALUE
    int global_max_hull = <int>Config.GLOBAL_MAX_HULL
    int global_max_shields = <int>Config.GLOBAL_MAX_SHIELDS
    int global_max_squad_value = <int>Config.GLOBAL_MAX_SQUAD_VALUE
    int global_max_engineer_value = <int>Config.GLOBAL_MAX_ENGINEER_VALUE
    int global_max_dice = <int>Config.GLOBAL_MAX_DICE
    tuple board_resolution = Config.BOARD_RESOLUTION
    int height_res = board_resolution[0]
    int width_res = board_resolution[1]
    float width_step = LONG_RANGE * 6 / width_res
    float height_step = LONG_RANGE * 3 / height_res

    int SHIP_STATS_FEATURES = 36



cpdef tuple get_terminal_value(Armada game):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] ship_hulls, squad_hulls, game_length
    cdef Ship ship
    cdef Squad squad

    if game.winner == 0.0 :
        raise ValueError("Game is not in a terminal state.")

    ship_hulls = np.zeros(max_ships, dtype=np.float32)
    for ship in game.ships:
        if ship.id >= max_ships or ship.destroyed: continue
        ship_hulls[ship.id] = ship.hull / ship.max_hull

    game_length = np.zeros(6, dtype=np.float32)
    game_length[game.round - 1] = 1.0

    return game.winner, {'game_length': game_length, 'ship_hulls': ship_hulls}


cpdef dict encode_game_state(Armada game):
    """Main function to encode the entire game state into numpy arrays for the NN."""
    encode_scalar_features(game)
    encode_ship_entity_features(game)
    encode_spatial_mask(game)
    encode_relation_matrix(game)

    return {
        'scalar': game.scalar_encode_array,
        'ship_entities': game.ship_encode_array,
        'ship_coords': game.ship_coords_array,
        'spatial': game.spatial_encode_array,
        'relations': game.relation_encode_array,
        'active_ship_id': (<Ship>game.active_ship).id if game.active_ship is not None else Config.MAX_SHIPS
    }

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void encode_scalar_features(Armada game):
    """
    Encodes high-level, non-spatial game state information, including crucial
    context about an ongoing attack from game.attack_info.
    """
    cdef int offset = 0
    cdef AttackInfo attack_info

    cdef tuple pool_result 

    cdef cnp.ndarray[cnp.float32_t, ndim=1] scalar_array = game.scalar_encode_array
    cdef cnp.float32_t[:] scalar_view = scalar_array

    # Reset the pre-allocated array
    scalar_array.fill(0.0)

    # --- Base Game State Features ---
    
    # Simple scalar values (3 features)
    scalar_view[0] = game.round / 6.0
    scalar_view[1] = game.get_point(1) / 200.0
    scalar_view[2] = game.get_point(-1) / 200.0
    offset = 3
    
    # One-hot encoded initiative (2 features)
    if game.first_player == 1:
        scalar_view[offset] = 1.0
    else:
        scalar_view[offset + 1] = 1.0
    offset += 2
    
    # One-hot encoded phase (phase_type features)
    scalar_view[offset + <int>game.phase - 1] = 1.0
    offset += phase_type # phase_type = 21
    
    # One-hot encoded current player (2 features)
    if game.current_player == 1:
        scalar_view[offset] = 1.0
    else:
        scalar_view[offset + 1] = 1.0
    offset += 2
    
    # --- Attack Context Features (17 features) ---
    
    if game.attack_info is None:
        return  # No attack in progress, return early
        
    attack_info = game.attack_info
    
    # Attack availability flags (3 features)
    scalar_view[offset] = float(attack_info.con_fire_dial)
    scalar_view[offset + 1] = float(attack_info.con_fire_token)
    scalar_view[offset + 2] = float(attack_info.swarm)
    offset += 3
    
    # Dice Pool Result (11 features)
    # Black (3)
    pool_result = attack_info.attack_pool_result[Dice.BLACK]
    scalar_view[offset] = <float>pool_result[0]
    scalar_view[offset + 1] = <float>pool_result[1]
    scalar_view[offset + 2] = <float>pool_result[2]
    offset += 3
    
    # Blue (3)
    pool_result = attack_info.attack_pool_result[Dice.BLUE]
    scalar_view[offset] = <float>pool_result[0]
    scalar_view[offset + 1] = <float>pool_result[1]
    scalar_view[offset + 2] = <float>pool_result[2]
    offset += 3
    
    # Red (5)
    pool_result = attack_info.attack_pool_result[Dice.RED]
    scalar_view[offset] = <float>pool_result[0]
    scalar_view[offset + 1] = <float>pool_result[1]
    scalar_view[offset + 2] = <float>pool_result[2]
    scalar_view[offset + 3] = <float>pool_result[3]
    scalar_view[offset + 4] = <float>pool_result[4]
    offset += 5
    
    # Critical Effect (one-hot, 1 feature)
    if attack_info.critical is not None:
        scalar_view[offset + <int>attack_info.critical] = 1.0
    offset += c_critical_type # critical_type is 1

    # Range & Obstruction (2 features)
    scalar_view[offset] = (<int>attack_info.attack_range) / 4.0
    scalar_view[offset + 1] = 1.0 if attack_info.obstructed else 0.0
    offset += 2 # End of array
    



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void encode_ship_entity_features(Armada game):
    """
    Encodes a detailed vector for each ship, now including its role in an active attack.
    
    This optimized version writes directly into the pre-allocated 
    game.ship_encode_array to avoid memory allocations within the loop.
    """
    # --- C-level variable definitions ---
    cdef Ship ship
    cdef DefenseToken token
    cdef AttackInfo attack_info
    
    # Memory view for the target row
    cdef cnp.float32_t[:] ship_view
    cdef cnp.float32_t[:] ship_coords_view
    
    # Indices and iterators
    cdef int offset, stack_idx, defense_start_idx, defense_idx, hull, command
    
    # Booleans
    cdef bint is_attack = False

    # --- End definitions ---

    # Zero out the entire array at the beginning.
    # This is crucial so we only have to write non-zero values.
    game.ship_encode_array[:, SHIP_STATS_FEATURES:].fill(0.0)
    game.ship_coords_array.fill(0.0)

    # Get attack_info once outside the loop
    if game.attack_info is not None:
        attack_info = game.attack_info
        is_attack = True
    
    for ship in game.ships:
        if ship.id >= max_ships or ship.destroyed: continue

        ship_coords_view = game.ship_coords_array[ship.id]
        ship_coords_view[0] = ship.x / game.player_edge
        ship_coords_view[1] = ship.y / game.short_edge

        ship_view = game.ship_encode_array[ship.id]

        offset = SHIP_STATS_FEATURES

        # --- Status (8 features) ---
        ship_view[offset] = float(ship == game.active_ship); offset += 1
        ship_view[offset] = float(ship == game.defend_ship); offset += 1
        ship_view[offset] = float(ship.activated); offset += 1
        ship_view[offset] = ship.speed / 4.0; offset += 1
        for hull in range(c_hull_type):
            if ship.attack_history[hull] is not None:
                ship_view[offset + hull] = 1.0
        offset += c_hull_type

        # --- Hull and Shields (5 features) ---
        ship_view[offset] = ship.hull / global_max_hull; offset += 1
        for hull in range(c_hull_type):
            ship_view[offset + hull] = ship.shield[hull] / global_max_shields
        offset += c_hull_type

        # --- Position and Orientation (4 features) ---
        ship_view[offset] = ship.x / game.player_edge; offset += 1
        ship_view[offset] = ship.y / game.short_edge; offset += 1
        ship_view[offset] = sin(ship.orientation); offset += 1
        ship_view[offset] = cos(ship.orientation); offset += 1
        
        # --- Command Stack (12 features) ---
        if ship.player == game.simulation_player:
            for stack_idx, command in enumerate(ship.command_stack):
                ship_view[offset + stack_idx * c_command_type + command] = 1.0
        offset += max_command_stack * c_command_type # Advance offset by the block size
        
        # --- Command Dials (4 features) ---
        for command in ship.command_dial:
            ship_view[offset + command] = 1.0
        offset += c_command_type # Advance offset by block size

        # --- Command Tokens (4 features) ---
        for command in ship.command_token:
            ship_view[offset + command] = 1.0
        offset += c_command_type # Advance offset by block size

        # --- Attack Role (8 features) ---
        if is_attack:
            if attack_info.is_attacker_ship and attack_info.attack_ship_id == ship.id:
                ship_view[offset + <int>attack_info.attack_hull] = 1.0 # is_attacking_hull (one-hot)
            if attack_info.is_defender_ship and ship.id == attack_info.defend_ship_id:
                ship_view[offset + c_hull_type + <int>attack_info.defend_hull] = 1.0 # is_defending_hull (one-hot)
        offset += 8 # Advance offset by block size

        # --- Defense Tokens (24 features) ---
        defense_start_idx = offset
        for defense_idx, token in ship.defense_tokens.items():
            if token.discarded: continue

            if token.readied: ship_view[defense_start_idx + defense_idx * 4] = 1.0
            else: ship_view[defense_start_idx + defense_idx * 4 + 1] = 1.0

            if not token.accuracy: ship_view[defense_start_idx + defense_idx * 4 + 2] = 1.0

            if not(is_attack and (defense_idx in attack_info.spent_token_indices or token.type in attack_info.spent_token_types)):
                ship_view[defense_start_idx + defense_idx * 4 + 3] = 1.0
        offset += max_defense_tokens * 4


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void encode_squad_entity_features(Armada game):
    """
    Encodes a detailed vector for each squad in-place in game.squad_encode_array
    to avoid all memory allocations, then returns a reference to the array.
    """
    
    cdef Squad squad
    cdef cnp.float32_t[:] squad_view
    cdef int defense_idx, offset, overlap_start_idx, defense_start_idx
    cdef DefenseToken token
    cdef AttackInfo attack_info
    cdef bint is_attack = False

    # Zero out the array to clear stale data
    game.squad_encode_array[:, SQUAD_STATS_FEATURES:].fill(0.0)

    if game.attack_info is not None:
        attack_info = game.attack_info
        is_attack = True

    for squad in game.squads:
        if squad.id >= max_squads or squad.destroyed:
            continue
            
        # Get a view to the specific row we will write to
        squad_view = game.squad_encode_array[squad.id]
        offset = SQUAD_STATS_FEATURES

        # --- Status (6 features) ---
        squad_view[offset] = squad.hull / global_max_hull; offset += 1
        squad_view[offset] = squad.activated; offset += 1
        squad_view[offset] = squad.can_attack; offset += 1
        squad_view[offset] = squad.can_move; offset += 1
        squad_view[offset] = squad.coords[0] / game.player_edge; offset += 1
        squad_view[offset] = squad.coords[1] / game.short_edge; offset += 1
        
        # --- Overlap (MAX_SHIPS=6 features) ---
        overlap_start_idx = offset
        if squad.overlap_ship_id is not None:
            squad_view[overlap_start_idx + <int>squad.overlap_ship_id] = 1.0
        offset += max_ships

        # --- Attack Role (2 features) ---
        if is_attack:
            if not attack_info.is_attacker_ship and squad.id == attack_info.attack_squad_id:
                squad_view[offset] = 1.0
            if not attack_info.is_defender_ship and squad.id == attack_info.defend_squad_id:
                squad_view[offset + 1] = 1.0
        offset += 2
        
        # # --- Defense Token (2*4 = 8 features) ---
        # defense_start_idx = offset
        # for defense_idx, token in squad.defense_tokens.items():
        #     if token.discarded: continue

        #     if token.readied :squad_view[defense_start_idx + defense_idx * 4] = 1.0
        #     else: squad_view[defense_start_idx + defense_idx * 4 + 1] = 1.0

        #     if not token.accuracy: squad_view[defense_start_idx + defense_idx * 4 + 2] = 1.0

        #     if not(is_attack and (defense_idx in attack_info.spent_token_indices or token.type in attack_info.spent_token_types)):
        #         squad_view[defense_start_idx + defense_idx * 4 + 3] = 1.0
        # offset += max_squad_defense_tokens * 4



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void encode_spatial_mask(Armada game):
    """
    Fills the game.spatial_planes 2D grid representations of the game board
    in-place and returns a reference to it.
    """

    cdef cnp.uint8_t[:, :, :, ::1] planes_view = game.spatial_encode_array
    
    # Standard cleanup
    game.spatial_encode_array.fill(0.0)

    cdef int i, r, c
    cdef long[:] rr_view, cc_view 
    cdef int hull, attack_range, channel_idx
    cdef dict threat_plane_dict, ranges_dict
    cdef Ship ship
    cdef tuple ship_hash
    
    for ship in game.ships:
        if ship.id >= Config.MAX_SHIPS or ship.destroyed:
            continue
            
        ship_hash = ship.get_ship_hash_state()

        # --- 1. Ship Presence ---
        # Get arrays as usual
        rr, cc = cache._ship_presence_indices(ship_hash)
        
        # Cast to memoryviews for C-level access
        rr_view = rr
        cc_view = cc
        
        # C-Level Loop (No GIL, No Python Overhead)
        for i in range(rr_view.shape[0]):
            r = rr_view[i]
            c = cc_view[i]

            # Bitwise Packing: Set bit (c % 8) in byte (c // 8)
            planes_view[ship.id, 0, r, c >> 3] |= (1 << (c & 7))

        # --- 2. Ship Threat ---
        threat_plane_dict = cache._ship_threat_indices(ship_hash)
        
        # Iterate Dictionary Items to avoid KeyError risks
        for hull, ranges_dict in threat_plane_dict.items():
            for attack_range, coords in ranges_dict.items():
                
                # Calculate channel carefully
                channel_idx = 1 + (hull * 3) + attack_range
                
                rr_view = coords[0]
                cc_view = coords[1]
                
                for i in range(rr_view.shape[0]):
                    r = rr_view[i]
                    c = cc_view[i]

                    # Bitwise Packing
                    planes_view[ship.id, channel_idx, r, c >> 3] |= (1 << (c & 7))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void encode_relation_matrix(Armada game):
    """
    Optimized pairwise range encoding.
    """
    
    cdef int i, j, attacker_id, defender_id
    cdef int from_hull, to_hull
    cdef int flat_idx
    cdef int attack_range
    cdef list ships = game.ships 
    cdef int n_ships = len(ships)
    cdef Ship attacker, defender 
    
    cdef cnp.uint8_t[:, :, ::1] rel_matrix = game.relation_encode_array
    rel_matrix[:] = 0

    cdef list range_list, attack_range_list

    # --- MAIN LOOP ---
    for i in range(n_ships):
        attacker = ships[i]
        
        # Check destroyed status (assuming boolean property)
        if attacker.destroyed:
            continue
            
        attacker_id = attacker.id
        
        for j in range(n_ships):
            # Self-check optimization
            if i == j: 
                continue

            defender = ships[j]
            
            if defender.destroyed:
                continue
                
            defender_id = defender.id

            _, range_list = cache.attack_range_s2s(
                attacker.get_ship_hash_state(), 
                defender.get_ship_hash_state()
            )

            for from_hull in range(c_hull_type):
                attack_range_list = range_list[from_hull] 
                
                for to_hull in range(c_hull_type):
                    attack_range = <int>attack_range_list[to_hull]
                    
                    flat_idx = from_hull * c_hull_type + to_hull
                    
                    rel_matrix[attacker_id, defender_id, flat_idx] = attack_range