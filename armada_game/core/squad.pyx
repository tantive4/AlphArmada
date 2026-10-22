# cython: profile=True

from __future__ import annotations

from armada_game.helpers.measurement import *
from armada_game.helpers.enum_class import *

from armada cimport Armada
from ship cimport Ship


cdef double MOVE_PITCH = SQUAD_MOVE_PITCH


cdef class Squad :
    """
    A squadron: a disk-shaped base with no facing and no hull zones (RRG
    Squadron).  Only hull, position and the activation slider change during a
    game; every rule that needs the other squadrons or the ships - engagement,
    targets, legal moves, placement - lives on Armada, which owns that state and
    caches it per game state.
    """
    def __init__(self, squad_dict : dict, team : int) -> None:
        self.team : int = team
        self.name : str = squad_dict['name']
        self.unique : bool = squad_dict['unique']

        self.max_hull : int = squad_dict['hull']
        self.speed : int = squad_dict['speed']
        self.battery : tuple[int,...] = tuple(squad_dict['battery'])
        self.anti_squad : tuple[int, ...] = tuple(squad_dict['anti_squad'])
        self.battery_dice : int = sum(self.battery)
        self.anti_squad_dice : int = sum(self.anti_squad)
        self.point : int = squad_dict['point']

        # Only unique squadrons carry defense tokens (RRG Defense Tokens), and
        # none of the squadrons in the data are unique.  Fail loudly rather than
        # silently dropping tokens if the data ever gains one.
        if squad_dict['defense_token']:
            raise NotImplementedError(f"{self.name}: squadron defense tokens are not modelled")

        self.swarm : bool = 'swarm' in squad_dict['keywords']
        self.escort : bool = 'escort' in squad_dict['keywords']
        self.bomber : bool = 'bomber' in squad_dict['keywords']
        self.heavy : bool = 'heavy' in squad_dict['keywords']
        self.counter : int = squad_dict.get('counter', 0)

    def __str__(self):
        return f"{self.name}(ID:{self.id})"
    __repr__ = __str__

    def deploy(self, Armada game, float x, float y, int squad_id):
        """
        deploy the squad in the game board
        param game: the Armada game instance
        param x: x coordinate of the squad center
        param y: y coordinate of the squad center
        param squad_id: the unique id of the squad in the game
        """
        self.game = game
        self.id = squad_id
        self.hull = self.max_hull
        self.destroyed = False
        self.activated = False
        self.coords = (<double>x, <double>y)

    cpdef void status_phase(self) :
        """Toggle the activation slider back for the next round."""
        self.activated = False

    cpdef void destroy(self) :
        self.destroyed = True
        self.hull = 0
        self.game.invalidate_geometry()
        self.game.visualize(f'{self} is destroyed!')

    cpdef void defend(self, int total_damage) :
        """
        A squadron loses hull points equal to the damage (RRG Damage); squadrons
        have no shields and are never dealt damage cards.
        """
        self.hull -= total_damage
        if self.hull <= 0 :
            self.destroy()

    cpdef tuple get_squad_hash_state(self) :
        """
        get a hashable state of the squad for caching purpose
        """
        cdef int x_int = <int>(<double>self.coords[0] * HASH_PRECISION)
        cdef int y_int = <int>(<double>self.coords[1] * HASH_PRECISION)
        return (x_int, y_int)

    cpdef void move_by_cells(self, int di, int dj) :
        """
        Move on the absolute squadron move grid (TODO D2).  Legality - inside
        the speed disk, inside the play area, overlapping nothing - is enforced
        by Armada.squad_move_actions(), which is the only source of payloads.
        """
        self.coords = (<double>self.coords[0] + di * MOVE_PITCH,
                       <double>self.coords[1] + dj * MOVE_PITCH)
        self.game.invalidate_geometry()

    cpdef void place(self, double x, double y) :
        """Place the squad directly, e.g. when a ship overlapped it."""
        self.coords = (x, y)
        self.game.invalidate_geometry()

    cpdef tuple gather_dice(self, bint is_ship, bint is_counter) :
        """
        Battery armament against a ship, anti-squadron armament against a
        squadron (RRG Attack), or X blue dice for a counter attack (RRG Squadron
        Keywords, Counter).  Squadron attacks are at distance 1, inside close
        range, so no colour is filtered out.
        """
        if is_ship :
            return self.battery
        if is_counter:
            return (0, self.counter, 0)
        return self.anti_squad

    cpdef bint is_obstruct_q2q(self, Squad to_squad) :
        """
        Line of sight between two squadrons is traced between base centres
        (RRG Line of Sight).  Delegated so attacks and engagement read the same
        cached answer: engaged == at distance 1 and not obstructed.
        """
        return self.game.squad_pair_obstructed(self, to_squad)

    cpdef bint is_obstruct_q2s(self, Ship to_ship, int to_hull) :
        """
        Checks if the line of sight between a squad and a ship is obstructed.
        """
        return to_ship.is_obstruct_s2q(to_hull, self)

    cdef object get_snapshot(self) :
        """
        Hull, position and activation slider are the squad's entire mutable
        state.  `destroyed` is derived: destroy() always zeroes the hull.
        """
        return (self.hull, self.coords, self.activated)

    cdef void revert_snapshot(self, object snapshot):
        """
        revert the squad to a previous snapshot
        """
        self.hull, self.coords, self.activated = snapshot
        self.destroyed = self.hull <= 0
