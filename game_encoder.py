from __future__ import annotations
import math
from typing import TYPE_CHECKING

import numpy as np

from configs import Config
from action_phase import Phase
from dice import *
from enum_class import *
import cache_function as cache

if TYPE_CHECKING:
    from armada import Armada


def get_terminal_value(game: Armada) -> tuple[float, dict[str, np.ndarray]]:
    if game.winner is None:
        raise ValueError("Game is not in a terminal state.")
    ship_hulls = np.zeros(Config.MAX_SHIPS, dtype=np.float32)
    for i, ship in enumerate(game.ships):
        if i >= Config.MAX_SHIPS or ship.destroyed: continue
        ship_hulls[i] = ship.hull / ship.max_hull

    squad_hulls = np.zeros(Config.MAX_SQUADS, dtype=np.float32)
    for i, squad in enumerate(game.squads):
        if i >= Config.MAX_SQUADS or squad.destroyed: continue
        squad_hulls[i] = squad.hull / squad.max_hull

    game_length = np.zeros(6, dtype=np.float32)
    game_length[game.round - 1] = 1.0

    return game.winner, {'game_length': game_length, 'ship_hulls': ship_hulls, 'squad_hulls': squad_hulls}


def encode_game_state(game: Armada) -> dict[str, np.ndarray]:
    """Main function to encode the entire game state into numpy arrays for the NN."""
    return {
        'scalar': encode_scalar_features(game),
        'ship_entities': encode_ship_entity_features(game),
        'squad_entities': encode_squad_entity_features(game),
        'spatial': encode_spatial_features(game, Config.BOARD_RESOLUTION),
        'relations': encode_relation_matrix(game)
    }

def encode_scalar_features(game: Armada) -> np.ndarray:
    """
    Encodes high-level, non-spatial game state information, including crucial
    context about an ongoing attack from game.attack_info.
    """
    # --- Base Game State Features (10 features) ---
    round_feature = game.round / 6.0
    phase_feature = np.zeros(len(Phase), dtype=np.float32)
    phase_feature[game.phase - 1] = 1.0
    initiative_feature = np.array([1, 0] if game.first_player == 1 else [0, 1], dtype=np.float32)
    player_feature = np.array([1, 0] if game.current_player == 1 else [0, 1], dtype=np.float32)
    p1_points = game.get_point(1) / 200.0
    p2_points = game.get_point(-1) / 200.0

    base_features = np.array(
        [round_feature, p1_points, p2_points] +
        initiative_feature.tolist() +
        phase_feature.tolist() +
        player_feature.tolist()
    )

    # --- Attack Context Features (17 features) ---
    attack_features = np.zeros(17, dtype=np.float32)

    if game.attack_info:
        info = game.attack_info
        
        # Attack availability flags (3 features)
        con_fire_dial = float(info.con_fire_dial)
        con_fire_token = float(info.con_fire_token)
        swarm = float(info.swarm)

        # Dice Pool Result (3+3+5 = 11 features)
        dice_pool = np.concatenate([
            np.array(info.attack_pool_result[Dice.BLACK], dtype=np.float32),
            np.array(info.attack_pool_result[Dice.BLUE], dtype=np.float32),
            np.array(info.attack_pool_result[Dice.RED], dtype=np.float32)
        ])

        # Critical Effect (one-hot, len(Critical) = 1 feature)
        crit_effect = np.zeros(len(Critical), dtype=np.float32)
        if info.critical :
            crit_effect[info.critical.value] = 1.0

        # Range & Obstruction (2 features)
        attack_range = (info.attack_range.value) / 4
        obstructed = 1.0 if info.obstructed else 0.0

        attack_features = np.concatenate([
            np.array([con_fire_dial, con_fire_token, swarm]),
            dice_pool,
            crit_effect,
            np.array([attack_range]),
            np.array([obstructed])
        ])

    return np.concatenate([base_features, attack_features])


def encode_ship_entity_features(game: Armada) -> np.ndarray:
    """
    Encodes a detailed vector for each ship, now including its role in an active attack.
    """
    ship_entity_vectors = np.zeros((Config.MAX_SHIPS, Config.SHIP_ENTITY_FEATURE_SIZE), dtype=np.float32)


    for i, ship in enumerate(game.ships):
        if i >= Config.MAX_SHIPS or ship.destroyed:
            continue

        # --- Base Stats and Position (14 features) ---
        base_and_pos = np.array([
            ship.max_hull / Config.GLOBAL_MAX_HULL,
            ship.hull / Config.GLOBAL_MAX_HULL,
            ship.max_shield[HullSection.FRONT] / Config.GLOBAL_MAX_SHIELDS,
            ship.max_shield[HullSection.RIGHT] / Config.GLOBAL_MAX_SHIELDS,
            ship.max_shield[HullSection.REAR] / Config.GLOBAL_MAX_SHIELDS,
            ship.max_shield[HullSection.LEFT] / Config.GLOBAL_MAX_SHIELDS,
            ship.shield[HullSection.FRONT] / Config.GLOBAL_MAX_SHIELDS,
            ship.shield[HullSection.RIGHT] / Config.GLOBAL_MAX_SHIELDS,
            ship.shield[HullSection.REAR] / Config.GLOBAL_MAX_SHIELDS,
            ship.shield[HullSection.LEFT] / Config.GLOBAL_MAX_SHIELDS,
            ship.x / game.player_edge,
            ship.y / game.short_edge,
            math.sin(ship.orientation),
            math.cos(ship.orientation)
        ], dtype=np.float32)

        # --- Command Info (17 features) ---
        command_value = ship.command_value / Config.MAX_COMMAND_STACK
        squad_value = ship.squad_value / Config.GLOBAL_MAX_SQUAD_VALUE
        engineer_value = ship.engineer_value / Config.GLOBAL_MAX_ENGINEER_VALUE
        point_cost = ship.point / 100
        command_stack_feature = np.zeros(Config.MAX_COMMAND_STACK * len(Command), dtype=np.float32)
        if ship.player == game.simulation_player:
            for stack_idx, command in enumerate(ship.command_stack):
                command_stack_feature[stack_idx * len(Command) + command] = 1.0
        
        command_dials = np.zeros(len(Command), dtype=np.float32)
        command_tokens = np.zeros(len(Command), dtype=np.float32)
        for dial in ship.command_dial:
            command_dials[dial] = 1.0
        for token in ship.command_token:
            command_tokens[token] = 1.0

        # --- Attack Role (10 features) ---
        attack_role = np.zeros(10, dtype=np.float32)
        if attack_info := game.attack_info:
            if attack_info.is_attacker_ship and ship.id == attack_info.attack_ship_id:
                attack_role[0] = 1.0
                attack_role[attack_info.attack_hull + 1] = 1.0 # is_attacking_hull (one-hot)
            if attack_info.is_defender_ship and ship.id == attack_info.defend_ship_id:
                attack_role[5] = 1.0
                attack_role[5 + attack_info.defend_hull] = 1.0 # is_defending_hull (one-hot)

        # --- Defense Tokens (12 features: 2 states for 6 token slots) ---
        defense_tokens = np.zeros(Config.MAX_DEFENSE_TOKENS * 2, dtype=np.float32)
        for idx, token in ship.defense_tokens.items():
            is_unavailable = (
                token.discarded or 
                token.accuracy or
                (game.attack_info and (idx in game.attack_info.spent_token_indices or token.type in game.attack_info.spent_token_types))
            )
            if not token.discarded:
                defense_tokens[idx * 2 + 0] = 1.0 if token.readied else 0.5 # ready / exhausted / 0 if discarded
                defense_tokens[idx * 2 + 1] = 1.0 if not is_unavailable else 0.0 # is_available
        
        # --- Armament, Status, Speed (24 features) ---
        armament = np.zeros(15, dtype=np.float32)
        for hull in HullSection:
            if hull in ship.attack_impossible_hull : continue
            armament[hull*3:(hull+1)*3] = [dice/Config.GLOBAL_MAX_DICE for dice in ship.battery[hull]]
        armament[12:15] = [dice/Config.GLOBAL_MAX_DICE for dice in ship.anti_squad]

        status = np.array([
            float(ship.activated), 
            ship.player,
            ship.size_class / SizeClass.LARGE, 
            ship.speed / 4.0,
            ship.attack_count / 2.0,
        ])
        
        nav_chart_vector = np.zeros(10, dtype=np.float32) # For speed 0 to 4
        for speed in range(5):
            if speed in ship.nav_chart:
             clicks = ship.nav_chart[speed]
             for i, click in enumerate(clicks):
                 nav_chart_vector[speed+i-1] = click / 2.0 # Normalize by max clicks

        ship_entity_vectors[i] = np.concatenate([
            base_and_pos,
            np.array([command_value, squad_value, engineer_value, point_cost]), command_stack_feature, command_dials, command_tokens,
            attack_role, defense_tokens, armament, status, nav_chart_vector
        ])

    return ship_entity_vectors

def encode_squad_entity_features(game: Armada) -> np.ndarray:
    """
    Encodes a detailed vector for each squad, summarizing the ships in the squad.
    """
    squad_entity_vectors = np.zeros((Config.MAX_SQUADS, Config.SQUAD_ENTITY_FEATURE_SIZE), dtype=np.float32)


    for i, squad in enumerate(game.squads):
        if i >= Config.MAX_SQUADS or squad.destroyed:
            continue

        # 8 features
        info = np.array([
            squad.max_hull / Config.GLOBAL_MAX_HULL,
            float(squad.player),
            squad.speed / 5.0,
            squad.point / 20,
            float(squad.unique),
            float(squad.swarm), 
            float(squad.bomber), 
            float(squad.escort),
        ])

        # 12 features
        status = np.array([
            squad.hull / Config.GLOBAL_MAX_HULL,
            float(squad.activated),
            float(squad.can_attack),
            float(squad.can_move),
            squad.coords[0] / game.player_edge,
            squad.coords[1] / game.short_edge,
        ])

        overlap_ship = np.zeros(Config.MAX_SHIPS, dtype=np.float32)
        if squad.overlap_ship_id is not None :
            overlap_ship[squad.overlap_ship_id] = 1.0
        status = np.concatenate([status, overlap_ship])

        # 6 features
        armament = np.zeros(6, dtype=np.float32)
        armament[0:3] = [dice/Config.GLOBAL_MAX_DICE for dice in squad.battery]
        armament[3:6] = [dice/Config.GLOBAL_MAX_DICE for dice in squad.anti_squad]

        # 2 features
        attack_role = np.zeros(2, dtype=np.float32)
        if attack_info := game.attack_info:
            if not attack_info.is_attacker_ship and i == attack_info.attack_squad_id:
                attack_role[0] = 1.0
            if not attack_info.is_defender_ship and i == attack_info.defend_squad_id:
                attack_role[1] = 1.0

        # 4 features
        defense_tokens = np.zeros(Config.MAX_SQUAD_DEFENSE_TOKENS * 2, dtype=np.float32)
        for idx, token in squad.defense_tokens.items():
            is_unavailable = (
                token.discarded or 
                token.accuracy or
                (game.attack_info and (idx in game.attack_info.spent_token_indices or token.type in game.attack_info.spent_token_types))
            )

            if not token.discarded:
                defense_tokens[idx * 2 + 0] = 1.0 if token.readied else 0.5 # ready / exhausted / 0 if discarded
                defense_tokens[idx * 2 + 1] = 1.0 if not is_unavailable else 0.0 # is_available

        squad_entity_vectors[i] = np.concatenate([
            info, status, attack_role, defense_tokens, armament
        ])

    return squad_entity_vectors

def encode_spatial_features(game: Armada, resolution: tuple[int, int]) -> np.ndarray:
    """
    Creates 2D grid representations of the game board.
    This is now a wrapper function for clarity and profiling.
    """
    width_res, height_res = resolution  # width along player_edge, height along short_edge
    planes = np.zeros((Config.MAX_SHIPS * 2 + 2, height_res, width_res), dtype=np.float32)

    width_step = game.player_edge / width_res
    height_step = game.short_edge / height_res
    for i, ship in enumerate(game.ships):
        if ship.destroyed: continue

        value = (ship.hull / ship.max_hull) * ship.player
        planes[2 * i] = cache._ship_presence_plane(
            ship.get_ship_hash_state(), value, width_step, height_step, width_res, height_res
        )
        planes[2 * i + 1] = cache._threat_plane(
            ship.get_ship_hash_state(), width_step, height_step, width_res, height_res
        )
        planes[-2] = sum(cache._squad_presence_plane(squad.get_squad_hash_state(), squad.hull / squad.max_hull, width_step, height_step, width_res, height_res) for squad in game.squads if not squad.destroyed and squad.player == 1)
        planes[-1] = sum(cache._squad_presence_plane(squad.get_squad_hash_state(), squad.hull / squad.max_hull, width_step, height_step, width_res, height_res) for squad in game.squads if not squad.destroyed and squad.player == -1)

    return planes

def encode_relation_matrix(game: Armada) -> np.ndarray:
    """
    Encodes the pairwise range relationships between every hull section of every ship.
    """
    num_hulls = Config.MAX_SHIPS * len(HullSection)
    matrix = np.zeros((num_hulls, num_hulls), dtype=np.float32)

    for i, attacker in enumerate(game.ships):
        if i >= Config.MAX_SHIPS or attacker.destroyed: continue
        for j, defender in enumerate(game.ships):
            if j >= Config.MAX_SHIPS or defender.destroyed or i == j: continue

            _, range_dict = cache.attack_range_s2s(attacker.get_ship_hash_state(), defender.get_ship_hash_state())

            for from_hull in HullSection:
                for to_hull in HullSection:
                    # Calculate the flattened index for the matrix
                    from_idx = i * 4 + from_hull
                    to_idx = j * 4 + to_hull
                    
                    attack_range = range_dict[from_hull][to_hull]
                    
                    # Normalize range to a value between 0 and 1
                    matrix[from_idx, to_idx] = (attack_range + 1) / 4
    return matrix
