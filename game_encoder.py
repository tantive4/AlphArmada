from __future__ import annotations
import numpy as np
import math
from typing import TYPE_CHECKING

from ship import Ship, HullSection, Command, SizeClass, _cached_range, _cached_point_range, _cached_polygons
from game_phase import GamePhase
from dice import Dice, Critical
from measurement import AttackRange
from skimage.draw import polygon as draw_polygon

if TYPE_CHECKING:
    from armada import Armada

# --- Configuration Constants ---
MAX_SHIPS = 6
MAX_COMMAND_STACK = 3
MAX_DEFENSE_TOKENS = 6 
BOARD_RESOLUTION = 16
ENTITY_FEATURE_SIZE = 78
RELATION_FEATURE_SIZE = 12
SCALAR_FEATURE_SIZE = 38

def encode_game_state(game: Armada) -> dict[str, np.ndarray]:
    """Main function to encode the entire game state into numpy arrays for the NN."""
    return {
        'scalar': encode_scalar_features(game),
        'entities': encode_entity_features(game),
        'spatial': encode_spatial_features(game, BOARD_RESOLUTION),
        'relations': encode_relation_matrix(game)
    }

def encode_scalar_features(game: Armada) -> np.ndarray:
    """
    Encodes high-level, non-spatial game state information, including crucial
    context about an ongoing attack from game.attack_info.
    """
    # --- Base Game State Features (10 features) ---
    round_feature = game.round / 6.0
    phase_feature = np.zeros(len(GamePhase), dtype=np.float32)
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

    # --- Attack Context Features (16 features) ---
    attack_features = np.zeros(16, dtype=np.float32)

    if game.attack_info:
        info = game.attack_info
        
        # Attack availability flags (2 features)
        con_fire_dial = 1.0 if info.con_fire_dial else 0.0
        con_fire_token = 1.0 if info.con_fire_token else 0.0

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
            np.array([con_fire_dial, con_fire_token]),
            dice_pool,
            crit_effect,
            np.array([attack_range]),
            np.array([obstructed])
        ])

    return np.concatenate([base_features, attack_features])


def encode_entity_features(game: Armada) -> np.ndarray:
    """
    Encodes a detailed vector for each ship, now including its role in an active attack.
    """
    entity_vectors = np.zeros((MAX_SHIPS, ENTITY_FEATURE_SIZE), dtype=np.float32)
    GLOBAL_MAX_HULL = 8.0
    GLOBAL_MAX_SHIELDS = 4.0
    GLOBAL_MAX_DICE = 4.0 

    for i, ship in enumerate(game.ships):
        if i >= MAX_SHIPS or ship.destroyed:
            continue

        # --- Base Stats and Position (14 features) ---
        base_and_pos = np.array([
            ship.max_hull / GLOBAL_MAX_HULL,
            ship.hull / GLOBAL_MAX_HULL,
            ship.max_shield[HullSection.FRONT] / GLOBAL_MAX_SHIELDS,
            ship.max_shield[HullSection.RIGHT] / GLOBAL_MAX_SHIELDS,
            ship.max_shield[HullSection.REAR] / GLOBAL_MAX_SHIELDS,
            ship.max_shield[HullSection.LEFT] / GLOBAL_MAX_SHIELDS,
            ship.shield[HullSection.FRONT] / GLOBAL_MAX_SHIELDS,
            ship.shield[HullSection.RIGHT] / GLOBAL_MAX_SHIELDS,
            ship.shield[HullSection.REAR] / GLOBAL_MAX_SHIELDS,
            ship.shield[HullSection.LEFT] / GLOBAL_MAX_SHIELDS,
            ship.x / game.player_edge,
            ship.y / game.short_edge,
            math.sin(ship.orientation),
            math.cos(ship.orientation)
        ])

        # --- Command Info (14 features) ---
        command_value = ship.command_value / float(MAX_COMMAND_STACK)
        command_stack_feature = np.zeros(MAX_COMMAND_STACK * len(Command), dtype=np.float32)
        if ship.player == game.simulation_player:
            for stack_idx, cmd in enumerate(ship.command_stack):
                command_stack_feature[stack_idx * len(Command) + (cmd.value)] = 1.0
        
        command_dials = np.zeros(len(Command), dtype=np.float32)
        command_tokens = np.zeros(len(Command), dtype=np.float32)
        for dial in ship.command_dial:
            command_dials[dial.value] = 1.0
        for token in ship.command_token:
            command_tokens[token.value] = 1.0

        # --- Attack Role (10 features) ---
        attack_role = np.zeros(10, dtype=np.float32)
        if game.attack_info:
            info = game.attack_info
            if i == info.attack_ship_id:
                attack_role[0] = 1.0
                attack_role[info.attack_hull.value + 1] = 1.0 # is_attacking_hull (one-hot)
            if i == info.defend_ship_id:
                attack_role[5] = 1.0
                attack_role[5 + info.defend_hull.value] = 1.0 # is_defending_hull (one-hot)

        # --- Defense Tokens (12 features: 2 states for 6 token slots) ---
        defense_tokens = np.zeros(MAX_DEFENSE_TOKENS * 2, dtype=np.float32)
        for idx, token in ship.defense_tokens.items():
            is_unavailable = (
                token.discarded or 
                token.accuracy or
                (game.attack_info and (idx in game.attack_info.spent_token_indices or token.type in game.attack_info.spent_token_types))
            )
            if not token.discarded:
                defense_tokens[idx * 2 + 0] = 1.0 if token.readied else 0.5 # ready / exhausted / 0 if discarded
                defense_tokens[idx * 2 + 1] = 1.0 if not is_unavailable else 0.0 # is_available
        
        # --- Armament, Status, Speed (21 features) ---
        armament = np.zeros(12, dtype=np.float32)
        for hull_idx, hs in enumerate(HullSection):
            armament[hull_idx*3:(hull_idx+1)*3] = [d/GLOBAL_MAX_DICE for d in ship.battery[hs]]

        status = np.array([
            1.0 if ship.activated else 0.0, 
            1.0 if ship.player == 1 else -1.0,
            ship.size_class.value / float(SizeClass.LARGE.value), 
            ship.speed / 4.0
        ])
        
        nav_chart_vector = np.zeros(10, dtype=np.float32) # For speed 0 to 4
        for speed in range(5):
            if speed in ship.nav_chart:
             clicks = ship.nav_chart[speed]
             for i, click in enumerate(clicks):
                 nav_chart_vector[speed+i-1] = click / 2.0 # Normalize by max clicks

        entity_vectors[i] = np.concatenate([
            base_and_pos,
            np.array([command_value]), command_stack_feature, command_dials, command_tokens,
            attack_role, defense_tokens, armament, status, nav_chart_vector
        ])

    return entity_vectors


def encode_spatial_features(game: Armada, resolution: int) -> np.ndarray:
    """
    Creates 2D grid representations of the game board using the Per-Entity Plane approach.
    This provides high-fidelity orientation and shape information.

    Output Shape: (MAX_SHIPS + 2, resolution, resolution)
    - Planes 0-5: Presence map for Ship ID 0-5 respectively.
        - Value is +health_ratio for friendly ships.
        - Value is -health_ratio for enemy ships.
    - Plane 6: Friendly Threat Map
    - Plane 7: Enemy Threat Map
    """
    planes = np.zeros((MAX_SHIPS + 2, resolution, resolution), dtype=np.float32)
    width_step = game.player_edge / resolution
    height_step = game.short_edge / resolution

    for i, ship in enumerate(game.ships):
        if ship.destroyed: continue
        
        value = (ship.hull / ship.max_hull) * ship.player
        scaled_vertices = np.array(_cached_polygons(ship.get_ship_hash_state())['base'].exterior.coords) / [width_step, height_step]
        rr, cc = draw_polygon(scaled_vertices[:, 1], scaled_vertices[:, 0], shape=planes[i].shape)
        planes[i, rr, cc] = value

    threat_plane_offset = MAX_SHIPS
    for ship in game.ships:
        if ship.destroyed: continue
            
        threat_plane_idx = threat_plane_offset + (0 if ship.player == game.current_player else 1)
        for r in range(resolution):
            for c in range(resolution):
                point = ((c + 0.5) * width_step, (r + 0.5) * height_step)
                total_threat = 0
                range_dict = _cached_point_range(ship.get_ship_hash_state(), point)
                for attack_hull in HullSection:
                    dice = ship.gather_dice(attack_hull, range_dict[attack_hull])
                    total_threat += sum(dice)
                if total_threat > 0:
                    planes[threat_plane_idx, r, c] += total_threat / 10.0
    return planes


def encode_relation_matrix(game: Armada) -> np.ndarray:
    """
    Encodes the pairwise range relationships between every hull section of every ship.
    """
    num_hulls = MAX_SHIPS * len(HullSection)
    matrix = np.zeros((num_hulls, num_hulls), dtype=np.float32)

    for i, attacker in enumerate(game.ships):
        if i >= MAX_SHIPS or attacker.destroyed: continue
        for j, defender in enumerate(game.ships):
            if j >= MAX_SHIPS or defender.destroyed or i == j: continue

            _, range_dict = _cached_range(attacker.get_ship_hash_state(), defender.get_ship_hash_state())

            for from_hull in HullSection:
                for to_hull in HullSection:
                    # Calculate the flattened index for the matrix
                    from_idx = i * 4 + from_hull
                    to_idx = j * 4 + to_hull
                    
                    attack_range = range_dict[from_hull][to_hull]
                    
                    # Normalize range to a value between 0 and 1
                    matrix[from_idx, to_idx] = (attack_range + 1) / 4
    return matrix
