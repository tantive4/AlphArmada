from __future__ import annotations
from typing import TYPE_CHECKING
from collections import Counter

from defense_token import DefenseToken, TokenType
from measurement import *
from enum_class import *
from dice import *
import cache_function as cache
if TYPE_CHECKING:
    from armada import Armada
    from ship import Ship





class Squad :

    def __init__(self, squad_dict : dict, player : int) -> None:
        self.player : int = player
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
        
        

    def __str__(self):
        return self.name
    __repr__ = __str__

    def deploy(self, game : Armada , x : float, y : float, squad_id : int) -> None:
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

    def status_phase(self) -> None :
        """
        refresh the squad status at the end of the round
        """
        self.activated : bool = False
        for token in self.defense_tokens.values():
            token.ready()

    def destroy(self) -> None:
        self.destroyed = True
        self.hull = 0
        for token in self.defense_tokens.values() :
            if not token.discarded : token.discard()
        self.game.visualize(f'{self} is destroyed!')

    def start_activation(self) -> None :
        self.game.active_squad = self
        self.game.squad_activation_count -= 1
        self.can_move = True
        self.can_attack = True
    
    def end_activation(self) -> None :
        self.game.active_squad = None
        self.activated = True

    def defend(self, total_damage) -> None :
        """
        apply damage to the squad
        param total_damage: the total damage to be applied
        """
        self.hull -= total_damage
        if self.hull <= 0 :
            self.destroy()

    def get_squad_hash_state(self) -> tuple[int, int] :
        """
        get a hashable state of the squad for caching purpose
        """
        return int(self.coords[0]*HASH_PRECISION), int(self.coords[1]*HASH_PRECISION)

    def is_engaged(self) -> bool :
        """
        check if the squad is engaged with any enemy squadron
        return: a list of engaged enemy squadrons
        """
        engage_distance = Q2Q_RANGE
        for squad in self.game.squads :
            if squad.player == self.player or squad.destroyed :
                continue
            distance = np.hypot(self.coords[0] - squad.coords[0], self.coords[1] - squad.coords[1])
            if distance > engage_distance :
                continue
            if not self.is_obstruct_q2q(squad) :
                return True
        return False

    
    def get_valid_target(self) -> list[int | tuple[int, HullSection]] :
        """
        get a list of valid targets for the squad to attack
        return: a list of valid targets, which can be either enemy squadrons or ship hull sections
        """
        valid_target : list[int | tuple[int, HullSection]] = []

        
        for squad in self.game.squads :
            if squad.player == self.player or squad.destroyed :
                continue
            distance = np.hypot(self.coords[0] - squad.coords[0], self.coords[1] - squad.coords[1])
            if distance <= Q2Q_RANGE :
                valid_target.append(squad.id)

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
                    # since there is no 2 dice battery squad YET
                    valid_target.append((ship.id, hull))

        return valid_target


    def is_obstruct_q2q(self, to_squad: Squad) -> bool :
        """
        Checks if the line of sight between two squads is obstructed.

        Args:
            squad (Squad): The target squad.
        """
        line_of_sight : tuple[tuple[float, float], ...] = (self.coords, to_squad.coords)

        for ship in self.game.ships:
            if ship.destroyed :
                continue
            if cache.is_obstruct(line_of_sight, ship.get_ship_hash_state()):
                return True
        return False
    
    def is_obstruct_q2s(self, to_ship: Ship, to_hull: HullSection) -> bool :
        """
        Checks if the line of sight between a squad and a ship is obstructed.

        Args:
            squad (Squad): The target squad.
        """
        return to_ship.is_obstruct_s2q(to_hull, self)
    
    def get_critical_effect(self, black_crit : bool, blue_crit : bool, red_crit : bool) -> list[Critical] :
        critical_list : list[Critical] = []
        if black_crit or blue_crit or red_crit :
            critical_list.append(Critical.STANDARD)
        return critical_list

    def move(self, speed: int, angle: float) -> None:
        """
        move the squad
        Args:
            speed: the speed to move
            angle: the angle to move, in degree, 0 is to up **on player's perspective**, 90 is to right
        """
        angle_rad = np.deg2rad(angle) * self.player # go "up" on player's perspective
        self.coords = (self.coords[0] + DISTANCE[speed] * np.sin(angle_rad), self.coords[1] + DISTANCE[speed] * np.cos(angle_rad))
        self.can_move = False

    def get_valid_moves(self) -> list[tuple[int, float]] :
        """
        get a list of valid moves for the squad
        return: 
            a list of valid moves, each move is a tuple of (speed, angle)
        """
        valid_moves : list[tuple[int, float]] = []
        original_coords = self.coords
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

    def is_overlap(self) -> bool :
        """
        Check if the squad is overlapping with any other squad or ship.
        Args:
            touching_ship_id: the id of the ship that has overlapped with the squad, if any. This ship will be ignored in the overlap check.
        Returns:
            True if the squad is overlapping with any other squad or ship, False otherwise.
        """
        for squad in self.game.squads :
            if squad.id == self.id or squad.destroyed or squad.overlap_ship_id is not None:
                continue
            distance = np.hypot(self.coords[0] - squad.coords[0], self.coords[1] - squad.coords[1])
            if distance <= SQUAD_BASE_RADIUS * 2 :
                return True
        for ship in self.game.ships :
            # ignore destroyed ships and the ship this squad is currently overlapping with
            if ship.destroyed or ship.id == self.overlap_ship_id:
                continue
            if cache.is_overlap_s2q(ship.get_ship_hash_state(), self.get_squad_hash_state()) :
                return True
        return False

    def place_squad(self, coords:tuple[float, float]) -> None :
        """
        place the squad at the given coordinates
        Args:
            coords: the coordinates to place the squad at
        """
        self.coords = coords
        self.overlap_ship_id = None

    def out_of_board(self) -> bool :
        """
        check if the squad is out of the board
        """
        return self.coords[0] < SQUAD_BASE_RADIUS or self.coords[0] > self.game.player_edge - SQUAD_BASE_RADIUS or self.coords[1] < SQUAD_BASE_RADIUS or self.coords[1] > self.game.short_edge - SQUAD_BASE_RADIUS

    def gather_dice(self, *, is_ship:bool) -> tuple[int, ...] :
        if is_ship :
            return self.battery
        else :
            return self.anti_squad
        
    def get_snapshot(self) -> tuple :
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
            {key: (dt.readied, dt.discarded, dt.accuracy)
             for key, dt in self.defense_tokens.items()}
        )

    def revert_snapshot(self, snapshot: tuple) -> None :
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
            self.defense_tokens[key].readied, self.defense_tokens[key].discarded, self.defense_tokens[key].accuracy = token_state