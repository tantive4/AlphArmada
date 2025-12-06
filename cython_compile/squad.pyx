# cython: profile=True

from __future__ import annotations
from typing import TYPE_CHECKING
from collections import Counter

import numpy as np
import math
cimport numpy as cnp
from libc.math cimport abs, M_PI, sin, cos

from measurement import *
from enum_class import *
from dice import *
import cache_function as cache

from armada cimport Armada
from ship cimport Ship
from obstacle cimport Obstacle
from defense_token cimport DefenseToken





cdef class Squad :
    def __init__(self, squad_dict : dict, player : Player) -> None:
        self.player : int = player.value
        self.name : str = squad_dict['name']
        self.unique : bool = squad_dict['unique']

        self.max_hull : int = squad_dict['hull']
        self.speed : int = squad_dict['speed']
        self.battery : tuple[int,...] = tuple(squad_dict['battery'])
        self.anti_squad : tuple[int, ...] = tuple(squad_dict['anti_squad'])

        self.defense_tokens: dict[int, DefenseToken] = {}
        token_counts = Counter()
        # Iterate through the list of token strings from the JSON
        for token_type_str in squad_dict['defense_token']:
            token_enum = TokenType[token_type_str.upper()]
            # token_counts[token_enum] will be 0 for the first, 1 for the second, etc.
            key = token_enum * 2 + token_counts[token_enum]
            # Add the token to the dictionary and increment the count for that type
            self.defense_tokens[key] = DefenseToken(token_type_str)
            token_counts[token_enum] += 1

        self.point : int = squad_dict['point']

        self.swarm : bool = 'swarm' in squad_dict['keywords']
        self.escort : bool = 'escort' in squad_dict['keywords']
        self.bomber : bool = 'bomber' in squad_dict['keywords']
        self.heavy : bool = 'heavy' in squad_dict['keywords']
        self.counter : int = squad_dict.get('counter', 0)

    def __str__(self):
        return self.name
    __repr__ = __str__

    def deploy(self, Armada game, float x, float y, int squad_id):
        """
        deploy the squad in the game board
        param game: the Armada game instance
        param x: x coordinate of the squad center
        param y: y coordinate of the squad center
        param squad_id: the unique id of the squad in the game
        """
        self.game : Armada = game
        self.destroyed : bool = False
        self.id : int = squad_id
        self.hull :int = self.max_hull
        self.coords : tuple[float, float] = (x, y)
        self.activated : bool = False
        self.can_move : bool = False
        self.can_attack : bool = False
        self.overlap_ship_id : int|None = None

    cpdef void status_phase(self) :
        """
        refresh the squad status at the end of the round
        """
        cdef DefenseToken token
        self.activated : bool = False
        for token in self.defense_tokens.values():
            token.ready()

    cpdef void destroy(self) :
        cdef DefenseToken token
        self.destroyed = True
        self.hull = 0
        for token in self.defense_tokens.values() :
            if not token.discarded : token.discard()
        self.game.visualize(f'{self} is destroyed!')

    cpdef void start_activation(self) :
        self.game.active_squad = self
        self.game.squad_activation_count -= 1
        self.can_move = True
        self.can_attack = True

    cpdef void end_activation(self) :
        self.game.active_squad = None
        self.activated = True

    cpdef void defend(self, int total_damage) :
        """
        apply damage to the squad
        param total_damage: the total damage to be applied
        """
        self.hull -= total_damage
        if self.hull <= 0 :
            self.destroy()

    cpdef tuple get_squad_hash_state(self) :
        """
        get a hashable state of the squad for caching purpose
        """
        cdef int x_int = <int>(self.coords[0] * HASH_PRECISION)
        cdef int y_int = <int>(self.coords[1] * HASH_PRECISION)
        return (x_int, y_int)
    
    cpdef bint is_engaged(self):
        """
        check if the squad is engaged with any enemy squadron
        return: a list of engaged enemy squadrons
        """
        cdef:
            float engage_distance = Q2Q_RANGE
            float engage_distance_sq = engage_distance * engage_distance
            Squad squad

        for squad in self.game.squads :
            if squad.player == self.player or squad.destroyed or squad.heavy :
                continue

            if not self.in_distance(squad, engage_distance, engage_distance_sq):
                continue

            if not self.is_obstruct_q2q(squad) :
                return True
        return False

    cpdef bint is_engage_with(self, Squad other):
        """
        Check if this squad is engaged with another squad.
        """
        cdef:
            float engage_distance = Q2Q_RANGE
            float engage_distance_sq = engage_distance * engage_distance
        return self.in_distance(other, engage_distance, engage_distance_sq) and not self.is_obstruct_q2q(other)

    cpdef list get_valid_target(self) :
        """
        get a list of valid targets for the squad to attack
        Returns:
            list[int | tuple[int, HullSection]]
        """
        cdef:
            list escort_target = []
            list valid_target = []
            Squad squad
            Ship ship
            float distance1 = <float>Q2Q_RANGE
            float distance1_sq = distance1 * distance1
            int hull
            bint in_range

        
        for squad in self.game.squads :
            if squad.player == self.player or squad.destroyed :
                continue
            if self.in_distance(squad, distance1, distance1_sq):
                if sum(self.anti_squad) <= 1 and self.is_obstruct_q2q(squad): continue
                valid_target.append(squad.id)
                if squad.escort and not self.is_obstruct_q2q(squad):
                    escort_target.append(squad.id)

        if escort_target :
            return escort_target
        if self.is_engaged() :
            return valid_target
        
        for ship in self.game.ships :
            if ship.player == self.player or ship.destroyed :
                continue
            for hull in HULL_SECTIONS :
                in_range = cache.attack_range_q2s(self.get_squad_hash_state(), ship.get_ship_hash_state())[hull]
                if not in_range :
                    continue
                if not self.is_obstruct_q2s(ship, hull) :
                    valid_target.append((ship.id, hull))
                elif sum(self.battery) > 1:
                    valid_target.append((ship.id, hull))

        return valid_target


    cpdef bint is_obstruct_q2q(self, Squad to_squad) :
        """
        Checks if the line of sight between two squads is obstructed.

        Args:
            squad (Squad): The target squad.
        """
        cdef:
            tuple line_of_sight = (self.coords, to_squad.coords)
            Ship ship
            Obstacle obstacle

        for ship in self.game.ships:
            if ship.destroyed :
                continue
            if cache.is_obstruct_ship(line_of_sight, ship.get_ship_hash_state()):
                return True

        for obstacle in self.game.obstacles:
            if cache.is_obstruct_obstacle(line_of_sight, obstacle.get_hash_state()):
                return True

        return False

    cpdef bint is_obstruct_q2s(self, Ship to_ship, int to_hull) :
        """
        Checks if the line of sight between a squad and a ship is obstructed.

        Args:
            squad (Squad): The target squad.
        """
        return to_ship.is_obstruct_s2q(to_hull, self)

    cpdef list get_critical_effect(self, bint black_crit, bint blue_crit, bint red_crit) :
        cdef list critical_list = []
        if black_crit or blue_crit or red_crit :
            critical_list.append(Critical.STANDARD)
        return critical_list

    cpdef void move(self, int speed, float angle) :
        """
        move the squad
        Args:
            speed: the speed to move
            angle: the angle to move, in degree, 0 is to up **on player's perspective**, 90 is to right
        """
        cdef float angle_rad = (M_PI*angle/180.0) * self.player # go "up" on player's perspective
        self.coords = (<float>self.coords[0] + <float>DISTANCE[speed] * sin(angle_rad), <float>self.coords[1] + <float>DISTANCE[speed] * cos(angle_rad))
        self.can_move = False

    cpdef list get_valid_moves(self) :
        """
        get a list of valid moves for the squad
        return: 
            a list of valid moves, each move is a tuple of (speed, angle)
        """
        cdef:
            list valid_moves = []
            tuple original_coords = self.coords
            int speed
            float angle

        for speed in range(self.speed + 1) :

            if speed == 0 :
                valid_moves.append((0, 0))
                continue

            # 3 * speed directions, evenly distributed
            for angle in range(0, 360, 90 // speed) :

                # simulate the move
                self.move(speed, angle)

                if not self.out_of_board() and not self.is_overlap():
                    valid_moves.append((speed, angle))
                self.coords = original_coords

        return valid_moves

    cpdef bint in_distance(self, Squad other, float distance, float distance_sq) :
        
        cdef:
            float dx = self.coords[0] - other.coords[0]
            float dy = self.coords[1] - other.coords[1]

        # L1 Norm Check
        if (abs(dx) > distance) or (abs(dy) > distance):
            return False  # Skip to the next squad, don't do the expensive check

        # Distance Check
        if (dx**2 + dy**2) <= distance_sq:
            return True

    cpdef bint is_overlap(self) :
        """
        Check if the squad is overlapping with any other squad or ship.
        Args:
            touching_ship_id: the id of the ship that has overlapped with the squad, if any. This ship will be ignored in the overlap check.
        Returns:
            True if the squad is overlapping with any other squad or ship, False otherwise.
        """
        cdef:
            Squad squad
            Ship ship
            float dx, dy
            float q2q_touch = SQUAD_BASE_RADIUS * 2
            float q2q_touch_square = q2q_touch ** 2
            
        for squad in self.game.squads:
            if squad.id == self.id or squad.destroyed or squad.overlap_ship_id is not None:
                continue
            if self.in_distance(squad, q2q_touch, q2q_touch_square): 
                return True

        for ship in self.game.ships:
            # ignore destroyed ships and the ship this squad is currently overlapping with
            if ship.destroyed or ship.id == self.overlap_ship_id:
                continue
            if cache.is_overlap_s2q(ship.get_ship_hash_state(), self.get_squad_hash_state()) :
                return True
        return False

    cpdef void place_squad(self, tuple coords) :
        """
        place the squad at the given coordinates
        Args:
            coords: the coordinates to place the squad at
        """
        self.coords = coords
        self.overlap_ship_id = None

    cpdef bint out_of_board(self) :
        """
        check if the squad is out of the board
        """
        cdef:
            float x = self.coords[0]
            float y = self.coords[1]
            float radius = SQUAD_BASE_RADIUS
        return x < radius or x > self.game.player_edge - radius or y < radius or y > self.game.short_edge - radius

    cpdef tuple gather_dice(self, bint is_ship, bint is_counter) :
        if is_ship :
            return self.battery
        if is_counter:
            return (0,self.counter, 0)
        return self.anti_squad
        
    cdef object get_snapshot(self) :
        """
        get a snapshot of the squad for saving and loading purpose
        """
        return (
            self.hull,
            self.coords,
            self.destroyed,
            self.activated,
            self.can_move,
            self.can_attack,
            self.overlap_ship_id,
            {<int>key: <DefenseToken>dt.get_snapshot()
             for key, dt in self.defense_tokens.items()}
        )

    cdef void revert_snapshot(self, object snapshot):
        """
        revert the squad to a previous snapshot
        """
        (
            self.hull,
            self.coords,
            self.destroyed,
            self.activated,
            self.can_move,
            self.can_attack,
            self.overlap_ship_id,
            defense_tokens_state
        ) = snapshot

        for key, token_state in defense_tokens_state.items():
            (<DefenseToken>self.defense_tokens[<int>key]).revert_snapshot(token_state)