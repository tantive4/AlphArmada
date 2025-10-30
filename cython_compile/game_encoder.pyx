from __future__ import annotations
import math
from typing import TYPE_CHECKING

import numpy as np
cimport numpy as np

from configs import Config
from action_phase import Phase
from dice import *
from enum_class import *
import cache_function as cache


from armada cimport Armada
from ship cimport Ship
from squad cimport Squad
from attack_info cimport AttackInfo
from defense_token cimport DefenseToken

cdef: 
    int command_type = len(Command)
    int phase_type = len(Phase)
    int critical_type = len(Critical)
    int hull_type = len(HullSection)

cpdef tuple get_terminal_value(Armada game):
    cdef np.ndarray[np.float32_t, ndim=1] ship_hulls, squad_hulls, game_length

    if game.winner == 0.0 :
        raise ValueError("Game is not in a terminal state.")

    ship_hulls = np.zeros(<int>Config.MAX_SHIPS, dtype=np.float32)
    for ship in game.ships:
        if ship.id >= <int>Config.MAX_SHIPS or ship.destroyed: continue
        ship_hulls[ship.id] = ship.hull / ship.max_hull

    squad_hulls = np.zeros(<int>Config.MAX_SQUADS, dtype=np.float32)
    for squad in game.squads:
        if squad.id >= <int>Config.MAX_SQUADS or squad.destroyed: continue
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
        'spatial': encode_spatial_features(game, <tuple>Config.BOARD_RESOLUTION),
        'relations': encode_relation_matrix(game)
    }

cdef np.ndarray[np.float32_t, ndim=1] encode_scalar_features(Armada game):
    """
    Encodes high-level, non-spatial game state information, including crucial
    context about an ongoing attack from game.attack_info.
    """
    # --- Base Game State Features (10 features) ---
    
    # Simple scalar values (3 features)
    cdef np.ndarray[np.float32_t, ndim=1] base_scalars = np.array([
        game.round / 6.0,
        game.get_point(1) / 200.0,
        game.get_point(-1) / 200.0
    ], dtype=np.float32)

    # One-hot encoded phase (len(Phase) features)
    cdef np.ndarray[np.float32_t, ndim=1] phase_feature = np.zeros(phase_type, dtype=np.float32)
    phase_feature[<int>game.phase - 1] = 1.0

    # One-hot encoded initiative (2 features)
    cdef np.ndarray[np.float32_t, ndim=1] initiative_feature = np.array([1, 0] if game.first_player == 1 else [0, 1], dtype=np.float32)
    
    # One-hot encoded current player (2 features)
    cdef np.ndarray[np.float32_t, ndim=1] player_feature = np.array([1, 0] if game.current_player == 1 else [0, 1], dtype=np.float32)

    # Combine all base features using np.concatenate
    cdef np.ndarray base_features = np.concatenate([
        base_scalars,
        initiative_feature,
        phase_feature,
        player_feature
    ])

    # --- Attack Context Features (17 features) ---
    cdef np.ndarray[np.float32_t, ndim=1] attack_features = np.zeros(17, dtype=np.float32)
    cdef AttackInfo attack_info
    cdef np.ndarray[np.float32_t, ndim=1] availability_flags, dice_pool, crit_effect, range_obstruction

    if game.attack_info is not None:
        attack_info = game.attack_info
        # Attack availability flags (3 features)
        availability_flags = np.array([
            float(attack_info.con_fire_dial),
            float(attack_info.con_fire_token),
            float(attack_info.swarm)
        ], dtype=np.float32)

        # Dice Pool Result (3+3+5 = 11 features)
        dice_pool = np.concatenate([
            np.array(attack_info.attack_pool_result[Dice.BLACK], dtype=np.float32),
            np.array(attack_info.attack_pool_result[Dice.BLUE], dtype=np.float32),
            np.array(attack_info.attack_pool_result[Dice.RED], dtype=np.float32)
        ])

        # Critical Effect (one-hot, len(Critical) = 1 feature)
        crit_effect = np.zeros(critical_type, dtype=np.float32)
        if attack_info.critical:
            crit_effect[<int>attack_info.critical] = 1.0

        # Range & Obstruction (2 features)
        range_obstruction = np.array([
            (<int>attack_info.attack_range) / 4.0,
            1.0 if attack_info.obstructed else 0.0
        ], dtype=np.float32)

        # Combine all attack features using np.concatenate
        attack_features = np.concatenate([
            availability_flags,
            dice_pool,
            crit_effect,
            range_obstruction
        ])

    # Combine base and attack features for the final vector
    return np.concatenate([base_features, attack_features])


cdef np.ndarray[np.float32_t, ndim=2] encode_ship_entity_features(Armada game):
    """
    Encodes a detailed vector for each ship, now including its role in an active attack.
    """
    cdef np.ndarray[np.float32_t, ndim=2] ship_entity_vectors = np.zeros(
        (<int>Config.MAX_SHIPS, <int>Config.SHIP_ENTITY_FEATURE_SIZE), dtype=np.float32
    )
    cdef Ship ship
    cdef np.ndarray[np.float32_t, ndim=1] hull_point, shield, position, command_scalars, command_stack, command_dials, command_tokens, \
                                          attack_role, defense_tokens, armament, status, nav_chart_vector
    cdef int stack_idx, defense_idx
    cdef object command
    cdef DefenseToken token

    # Get attack_info once outside the loop
    cdef AttackInfo attack_info
    cdef bint is_attack = False
    if game.attack_info is not None:
        attack_info = game.attack_info
        is_attack = True
    


    for ship in game.ships:
        if ship.id >= <int>Config.MAX_SHIPS or ship.destroyed:
            continue

        # Hull and Shield (10 features)
        hull_point = np.array([
            ship.max_hull,
            ship.hull
        ], dtype=np.float32)/ <int>Config.GLOBAL_MAX_HULL
        shield = np.array([
            ship.max_shield[HullSection.FRONT],
            ship.max_shield[HullSection.RIGHT],
            ship.max_shield[HullSection.REAR],
            ship.max_shield[HullSection.LEFT],
            ship.shield[HullSection.FRONT],
            ship.shield[HullSection.RIGHT],
            ship.shield[HullSection.REAR],
            ship.shield[HullSection.LEFT]
        ], dtype=np.float32)/ <int>Config.GLOBAL_MAX_SHIELDS

        # Position and Orientation (4 features)
        position = np.array([
            ship.x / game.player_edge,
            ship.y / game.short_edge,
            math.sin(ship.orientation),
            math.cos(ship.orientation)
        ], dtype=np.float32)


        # Stats (4 features)
        command_scalars = np.array([
            ship.command_value / <int>Config.MAX_COMMAND_STACK,
            ship.squad_value / <int>Config.GLOBAL_MAX_SQUAD_VALUE,
            ship.engineer_value / <int>Config.GLOBAL_MAX_ENGINEER_VALUE,
            ship.point / 100.0
        ], dtype=np.float32)
        
        # Command Stack (12 features)
        command_stack = np.zeros(
            <int>Config.MAX_COMMAND_STACK * command_type, dtype=np.float32
        )
        if ship.player == game.simulation_player:
            for stack_idx, command in enumerate(ship.command_stack):
                command_stack[stack_idx * command_type + <int>command] = 1.0
        
        # Command Dials and Tokens (8 features)
        command_dials = np.zeros(command_type, dtype=np.float32)
        for command in ship.command_dial:
            command_dials[<int>command] = 1.0
        command_tokens = np.zeros(command_type, dtype=np.float32)
        for command in ship.command_token:
            command_tokens[<int>command] = 1.0


        # --- Attack Role (10 features) ---
        attack_role = np.zeros(10, dtype=np.float32)
        if is_attack:
            if attack_info.is_attacker_ship and attack_info.attack_ship_id == ship.id:
                attack_role[0] = 1.0
                attack_role[<int>attack_info.attack_hull + 1] = 1.0 # is_attacking_hull (one-hot)
            if attack_info.is_defender_ship and ship.id == attack_info.defend_ship_id:
                attack_role[5] = 1.0
                attack_role[5 + <int>attack_info.defend_hull] = 1.0 # is_defending_hull (one-hot)

        # --- Defense Tokens (12 features) ---
        defense_tokens = np.zeros(Config.MAX_DEFENSE_TOKENS * 2, dtype=np.float32)

        for defense_idx, token in ship.defense_tokens.items():
            is_unavailable = (
                token.discarded or 
                token.accuracy or
                (is_attack and (defense_idx in attack_info.spent_token_indices or token.type in attack_info.spent_token_types))
            )
            if not token.discarded:
                defense_tokens[defense_idx * 2 + 0] = 1.0 if token.readied else 0.5 # ready / exhausted
                defense_tokens[defense_idx * 2 + 1] = 1.0 if not is_unavailable else 0.0 # is_available
        
        # Armament (15 features)
        armament = np.zeros(15, dtype=np.float32)
        for hull in HULL_SECTIONS:
            if hull in ship.attack_impossible_hull : continue
            armament[hull*3:(hull+1)*3] = [dice/Config.GLOBAL_MAX_DICE for dice in ship.battery[hull]]
        armament[12:15] = [dice/Config.GLOBAL_MAX_DICE for dice in ship.anti_squad]

        # Status (6 features)
        status = np.array([
            float(ship.activated), 
            ship.player,
            <int>ship.size_class / <int>SizeClass.LARGE, 
            ship.speed / 4.0,
            ship.attack_count / 2.0,
        ], dtype=np.float32)
        
        # Nav Chart Vector (10 features)
        nav_chart_vector = ship.nav_chart_vector

        ship_entity_vectors[ship.id] = np.concatenate([
            hull_point, shield, position, command_scalars, command_stack, command_dials, command_tokens,
            attack_role, defense_tokens, armament, status, nav_chart_vector
        ])

    return ship_entity_vectors

cdef np.ndarray[np.float32_t, ndim=2] encode_squad_entity_features(Armada game):
    """
    Encodes a detailed vector for each squad, summarizing the ships in the squad.
    """
    cdef np.ndarray[np.float32_t, ndim=2] squad_entity_vectors = np.zeros((<int>Config.MAX_SQUADS, <int>Config.SQUAD_ENTITY_FEATURE_SIZE), dtype=np.float32)
    cdef Squad squad
    cdef np.ndarray[np.float32_t, ndim=1] info, status, overlap_ship, attack_role, defense_tokens, armament
    cdef int defense_idx
    cdef DefenseToken token

    cdef AttackInfo attack_info
    cdef bint is_attack = False
    if game.attack_info is not None:
        attack_info = game.attack_info
        is_attack = True

    for squad in game.squads:
        if squad.id >= <int>Config.MAX_SQUADS or squad.destroyed:
            continue

        # Stats (8 features)
        info = np.array([
            squad.max_hull / <int>Config.GLOBAL_MAX_HULL,
            float(squad.player),
            squad.speed / 5.0,
            squad.point / 20,
            float(squad.unique),
            float(squad.swarm), 
            float(squad.bomber), 
            float(squad.escort),
        ], dtype=np.float32)

        # status (6 features)
        status = np.array([
            squad.hull / <int>Config.GLOBAL_MAX_HULL,
            float(squad.activated),
            float(squad.can_attack),
            float(squad.can_move),
            squad.coords[0] / game.player_edge,
            squad.coords[1] / game.short_edge,
        ], dtype=np.float32)

        # Overlap (MAX_SHIPS=6 features)
        overlap_ship = np.zeros(Config.MAX_SHIPS, dtype=np.float32)
        if squad.overlap_ship_id is not None :
            overlap_ship[<int>squad.overlap_ship_id] = 1.0

        # Armament (6 features)
        armament = np.zeros(6, dtype=np.float32)
        armament[0:3] = [dice/Config.GLOBAL_MAX_DICE for dice in squad.battery]
        armament[3:6] = [dice/Config.GLOBAL_MAX_DICE for dice in squad.anti_squad]

        # Attack Role (2 features)
        attack_role = np.zeros(2, dtype=np.float32)
        if is_attack:
            if not attack_info.is_attacker_ship and squad.id == attack_info.attack_squad_id:
                attack_role[0] = 1.0
            if not attack_info.is_defender_ship and squad.id == attack_info.defend_squad_id:
                attack_role[1] = 1.0

        # Defense Token (4 features)
        defense_tokens = np.zeros(Config.MAX_SQUAD_DEFENSE_TOKENS * 2, dtype=np.float32)
        for defense_idx, token in squad.defense_tokens.items():
            is_unavailable = (
                token.discarded or 
                token.accuracy or
                (game.attack_info and (defense_idx in attack_info.spent_token_indices or token.type in attack_info.spent_token_types))
            )

            if not token.discarded:
                defense_tokens[defense_idx * 2 + 0] = 1.0 if token.readied else 0.5 # ready / exhausted / 0 if discarded
                defense_tokens[defense_idx * 2 + 1] = 1.0 if not is_unavailable else 0.0 # is_available

        squad_entity_vectors[squad.id] = np.concatenate([
            info, status, overlap_ship, attack_role, defense_tokens, armament
        ])

    return squad_entity_vectors

cdef np.ndarray[np.float32_t, ndim=3] encode_spatial_features(Armada game, tuple resolution):
    """
    Creates 2D grid representations of the game board.
    This is now a wrapper function for clarity and profiling.
    """
    cdef int width_res, height_res
    width_res, height_res = resolution  # width along player_edge, height along short_edge

    cdef np.ndarray[np.float32_t, ndim=3] planes = np.zeros((Config.MAX_SHIPS * 2 + 2, height_res, width_res), dtype=np.float32)

    cdef float width_step = game.player_edge / width_res
    cdef float height_step = game.short_edge / height_res
    cdef float value
    cdef Ship ship
    for ship in game.ships:
        if ship.destroyed: continue

        value = (ship.hull / ship.max_hull) * ship.player
        planes[2 * ship.id] = cache._ship_presence_plane(
            ship.get_ship_hash_state(), value, width_step, height_step, width_res, height_res
        )
        planes[2 * ship.id + 1] = cache._threat_plane(
            ship.get_ship_hash_state(), width_step, height_step, width_res, height_res
        )
        planes[-2] = sum(cache._squad_presence_plane(<Squad>squad.get_squad_hash_state(), (<Squad>squad).hull / (<Squad>squad).max_hull, width_step, height_step, width_res, height_res) for squad in game.squads if not squad.destroyed and squad.player == 1)
        planes[-1] = sum(cache._squad_presence_plane(<Squad>squad.get_squad_hash_state(), (<Squad>squad).hull / (<Squad>squad).max_hull, width_step, height_step, width_res, height_res) for squad in game.squads if not squad.destroyed and squad.player == -1)

    return planes

cdef np.ndarray[np.float32_t, ndim=2] encode_relation_matrix(Armada game) :
    """
    Encodes the pairwise range relationships between every hull section of every ship.
    """
    cdef int num_hulls = <int>Config.MAX_SHIPS * hull_type
    cdef np.ndarray[np.float32_t, ndim=2] matrix = np.zeros((num_hulls, num_hulls), dtype=np.float32)
    cdef Ship attacker, defender
    cdef int i, j, from_idx, to_idx
    cdef dict range_dict
    cdef object from_hull, to_hull

    for i, attacker in enumerate(game.ships):
        if i >= <int>Config.MAX_SHIPS or attacker.destroyed: continue
        for j, defender in enumerate(game.ships):
            if j >= <int>Config.MAX_SHIPS or defender.destroyed or i == j: continue

            _, range_dict = cache.attack_range_s2s(attacker.get_ship_hash_state(), defender.get_ship_hash_state())

            for from_hull in HULL_SECTIONS:
                for to_hull in HULL_SECTIONS:
                    # Calculate the flattened index for the matrix
                    from_idx = i * 4 + <int>from_hull
                    to_idx = j * 4 + <int>to_hull

                    attack_range = range_dict[from_hull][to_hull]
                    
                    # Normalize range to a value between 0 and 1
                    matrix[from_idx, to_idx] = (<int>attack_range + 1) / 4
    return matrix
