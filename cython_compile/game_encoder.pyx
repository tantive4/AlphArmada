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

    int SHIP_STATS_FEATURES = 36
    int SQUAD_STATS_FEATURES = 15



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

    squad_hulls = np.zeros(max_squads, dtype=np.float32)
    for squad in game.squads:
        if squad.id >= max_squads or squad.destroyed: continue
        squad_hulls[squad.id] = squad.hull / squad.max_hull

    game_length = np.zeros(6, dtype=np.float32)
    game_length[game.round - 1] = 1.0

    return game.winner, {'game_length': game_length, 'ship_hulls': ship_hulls, 'squad_hulls': squad_hulls}


cpdef dict encode_game_state(Armada game):
    """Main function to encode the entire game state into numpy arrays for the NN."""
    return {
        'scalar': encode_scalar_features(game),
        'ship_entities': encode_ship_entity_features(game),
        'squad_entities': encode_squad_entity_features(game),
        'spatial': encode_spatial_features(game, board_resolution),
        'relations': encode_relation_matrix(game)
    }

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[cnp.float32_t, ndim=1] encode_scalar_features(Armada game):
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
        return scalar_array  # No attack in progress, return early
        
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
    
    return scalar_array


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[cnp.float32_t, ndim=2] encode_ship_entity_features(Armada game):
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
    
    # Indices and iterators
    cdef int offset, stack_idx, defense_idx, hull, command
    
    # Booleans
    cdef bint is_attack = False

    # --- End definitions ---

    # Zero out the entire array at the beginning.
    # This is crucial so we only have to write non-zero values.
    game.ship_encode_array[:, SHIP_STATS_FEATURES:].fill(0.0)

    # Get attack_info once outside the loop
    if game.attack_info is not None:
        attack_info = game.attack_info
        is_attack = True
    
    for ship in game.ships:
        if ship.id >= max_ships or ship.destroyed: continue

        ship_view = game.ship_encode_array[ship.id]
        offset = SHIP_STATS_FEATURES

        # --- Status (6 features) ---
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

    # Return the populated array
    return game.ship_encode_array

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[cnp.float32_t, ndim=2] encode_squad_entity_features(Armada game):
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

    # Return the reference to the modified array
    return game.squad_encode_array

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[cnp.float32_t, ndim=3] encode_spatial_features(Armada game, tuple resolution):
    """
    Fills the game.spatial_planes 2D grid representations of the game board
    in-place and returns a reference to it.
    """
    cdef int width_res, height_res
    width_res, height_res = resolution

    cdef cnp.ndarray[cnp.float32_t, ndim=3] planes = game.spatial_encode_array

    # CRITICAL: Zero out the existing array
    planes.fill(0.0)

    cdef float width_step = game.player_edge / width_res
    cdef float height_step = game.short_edge / height_res
    cdef float value
    cdef Ship ship
    cdef Squad squad
    cdef cnp.ndarray[cnp.float32_t, ndim=2] squad_plane

    for ship in game.ships:
        if ship.id >= max_ships: 
            continue
        if ship.destroyed: 
            continue

        value = (ship.hull / ship.max_hull) * ship.player
        
        planes[2 * ship.id] = cache._ship_presence_plane(
            ship.get_ship_hash_state(), value, width_step, height_step, width_res, height_res
        )
        planes[2 * ship.id + 1] = cache._threat_plane(
            ship.get_ship_hash_state(), width_step, height_step, width_res, height_res
        )

    for squad in game.squads:
        if squad.id >= max_squads: 
            continue
        if squad.destroyed: continue
        
        squad_plane = cache._squad_presence_plane(
            squad.get_squad_hash_state(), 
            squad.hull / squad.max_hull, 
            width_step, height_step, width_res, height_res
        )
        
        if squad.player == 1:
            planes[2*max_ships] += squad_plane
        else:
            planes[2*max_ships + 1] += squad_plane

    # Return the reference to the array modified in-place
    return game.spatial_encode_array

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[cnp.float32_t, ndim=2] encode_relation_matrix(Armada game):
    """
    Encodes the pairwise range relationships between every hull section of every ship.
    """
    
    cdef Ship attacker, defender
    cdef int attacker_id, defender_id, from_hull, to_hull, from_idx, to_idx
    
    # --- IMPROVEMENT 3: Type variables in the hot loop ---
    cdef int attack_range
    cdef list range_list, attack_range_list
    
    # Get the numpy array from the game object
    cdef cnp.ndarray[cnp.float32_t, ndim=2] rel_matrix = game.relation_encode_array
    
    # --- IMPROVEMENT 2: Use a Typed Memoryview ---
    # This view provides direct, C-level buffer access to the array's data.
    cdef cnp.float32_t[:, :] rel_matrix_view = rel_matrix
    
    # .fill() is efficient for initialization
    rel_matrix.fill(0.0)
    
    for attacker in game.ships:
        attacker_id = attacker.id
        
        if attacker.destroyed: 
            continue
        
        for defender in game.ships:
            defender_id = defender.id

            if attacker_id == defender_id:
                continue  # Don't check against self

            if defender.destroyed:
                continue

            _, range_list = cache.attack_range_s2s(attacker.get_ship_hash_state(), defender.get_ship_hash_state())

            for from_hull in range(c_hull_type):
                attack_range_list = range_list[from_hull]
                for to_hull in range(c_hull_type):
                    # Calculate the flattened index for the matrix
                    from_idx = attacker_id * c_hull_type + from_hull
                    to_idx = defender_id * c_hull_type + to_hull

                    attack_range = <int>attack_range_list[to_hull]
                    
                    rel_matrix_view[from_idx, to_idx] = (attack_range + 1.0) / 4.0
                    
    return game.relation_encode_array
