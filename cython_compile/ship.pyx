# cython: profile=True

from __future__ import annotations
from typing import TYPE_CHECKING
import itertools

import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos

from dice import *
from defense_token cimport DefenseToken
from measurement import *
from enum_class import *
import cache_function as cache

from armada cimport Armada
from squad cimport Squad
from obstacle cimport Obstacle


cdef class Ship:
    def __init__(self, ship_dict : dict, team : int) -> None:
        self.team : int = team
        self.name : str = ship_dict['name']
        self.faction : int = ship_dict['faction']

        self.max_hull : int = ship_dict['hull']
        self.size_class : SizeClass = SizeClass[ship_dict['size_class']]
        self.token_size : tuple [float, float] = SHIP_TOKEN_SIZE[self.size_class]
        self.base_size : tuple [float, float] = SHIP_BASE_SIZE[self.size_class]

        self.battery :  dict[HullSection, list[int]] = {HullSection.FRONT : ship_dict['battery'][0], 
                                                        HullSection.RIGHT : ship_dict['battery'][1], 
                                                        HullSection.REAR : ship_dict['battery'][2], 
                                                        HullSection.LEFT : ship_dict['battery'][1]}
        self.battery_range : dict[HullSection, dict[AttackRange, tuple[int, ...]]] = {
            hull: {
                attack_range : tuple(self.battery[hull][dice_type] if dice_type >= attack_range else 0 for dice_type in DICE)
                for attack_range in ATTACK_RANGES if attack_range != AttackRange.INVALID
            } for hull in HULL_SECTIONS
        }

        self.anti_squad :tuple[int,...] = tuple(ship_dict['anti_squad'])
        self.anti_squad_range : dict[AttackRange, tuple[int, ...]] = {
            attack_range : tuple(self.anti_squad[dice_type] if dice_type >= attack_range else 0 for dice_type in DICE)
            for attack_range in ATTACK_RANGES if attack_range != AttackRange.INVALID
        }
        
        self.defense_tokens: tuple[DefenseToken, ...] = tuple(
            DefenseToken(token_type_str, id) 
            for id, token_type_str in enumerate(ship_dict['defense_token'])
            )

        self.nav_chart : dict[int, list[int]] = {int(k) : v for k, v in ship_dict['navchart'].items()}
        self.nav_chart_vector = np.zeros(10, dtype=np.float32)
        for speed in range(5):
            if speed in self.nav_chart:
                 clicks = self.nav_chart[speed]
                 for i, click in enumerate(clicks): 
                     self.nav_chart_vector[speed+i-1] = click / 2.0 # Normalize by max clicks
        self.rotation_matrix = np.zeros((2,2), dtype=np.float32)
        self.max_shield : dict[HullSection, int] = {HullSection.FRONT : ship_dict['shield'][0], 
                                                    HullSection.RIGHT : ship_dict['shield'][1], 
                                                    HullSection.REAR : ship_dict['shield'][2], 
                                                    HullSection.LEFT : ship_dict['shield'][1]}
        self.point : int = ship_dict['point']
        self.command_value : int = ship_dict['command']
        self.engineer_value : int = ship_dict['engineering']
        self.squad_value : int = ship_dict['squadron']
        self.destroyed = False
        
        self._create_template_geometries(ship_dict)
        self._course_cache : dict[tuple[int, bool], list[tuple[int, ...]]]= {}
        
    def __str__(self):
        return f"{self.name}({self.id})"
    __repr__ = __str__

    def deploy(self, Armada game, float x, float y, float orientation, int speed, int ship_id):
        """
        deploy the ship to the game board

        Args:
            game (Armada) : the game
            (x, y) (float) : coordination of front center of the **ship token**
            orientation (float) : ship orientation
            speed (int) : ship speed
            ship_id (int) : ship id
        """
        self.game = game
        self.x = x
        self.y = y
        self.orientation = orientation
        self.speed = speed

        self.hull: int = self.max_hull
        self.shield: tuple[int, ...] = tuple(self.max_shield[hull] for hull in HULL_SECTIONS)
        self.id: int = ship_id
        self.command_stack: tuple[Command, ...] = ()
        self.command_dial: tuple[Command, ...] = ()
        self.command_token: tuple[Command, ...] = ()
        self.resolved_command: tuple[Command, ...] = ()
        self.engineer_point: int = 0
        self.attack_count: int = 0
        self.attack_history: tuple[tuple[int, HullSection]|tuple[int, ...]|None, ...] = (None, None, None, None)
        self.repaired_hull: tuple[HullSection, ...] = ()
        self.status_phase()
    
    cpdef void asign_command(self, int command) :
        if len(self.command_stack) >= self.command_value : raise ValueError("Cannot asigne more command then Command Value")
        self.command_stack += (command,)

    cpdef void spend_command_dial(self, int command) :
        if command not in self.command_dial : raise ValueError("Cannot spend command not in command dial")
        cdef: 
            list new_dial = []
            int dial
        for dial in self.command_dial:
            if dial != command:
                new_dial.append(dial)
        self.command_dial = tuple(new_dial)

    cpdef void spend_command_token(self, int command) :
        if command not in self.command_token : raise ValueError("Cannot spend command not in command token")
        cdef: 
            list new_token = []
            int token
        for token in self.command_token:
            if token != command:
                new_token.append(token)
        self.command_token = tuple(new_token)

    cpdef void destroy(self) :
        cdef DefenseToken token

        self.destroyed = True
        self.activated = True
        self.attack_history = (None, None, None, None)
        self.attack_count = 0
        self.command_dial = ()
        self.resolved_command = ()

        self.hull = 0
        self.shield = (0, 0, 0, 0)
        for token in self.defense_tokens.values() :
            if not token.discarded : token.discard()
        self.game.visualize(f'{self} is destroyed!')

    cpdef void status_phase(self) :
        cdef DefenseToken token

        self.activated = False
        for token in self.defense_tokens.values():
            if not token.discarded:
                token.ready()

    cpdef void end_activation(self) :
        self.activated = True
        self.game.active_ship = None
        self.attack_history = (None, None, None, None)
        self.attack_count = 0
        self.command_dial = ()
        self.resolved_command = ()

    cpdef void execute_maneuver(self, tuple course, int placement) :
        cdef set overlap_ships = self.move_ship(course, placement, set())
        self.overlap_damage(overlap_ships)
        if self.out_of_board() :
            self.game.visualize(f'{self} is out of board!')
            self.destroy()

    cpdef set move_ship(self, tuple course, int placement, set overlap_ships) :

        cdef:
            float original_x = self.x
            float original_y = self.y
            float original_orientation = self.orientation
            cnp.ndarray[cnp.float32_t, ndim=1] tool_translation
            float tool_rotation
            float c, s
            set current_overlap, new_overlap
        if not course :
            self.game.visualize(f'{self} executes speed 0 maneuver.')
            return overlap_ships
        if self.game.debuging_visual : tool_coord = self._tool_coordination(course, placement)[0]

        tool_translation, tool_rotation = cache.maneuver_tool(self.size_class, course, placement)
        # change translation vector according to the current orientation
        # rotation matrix use CW orientation    
        c = cos(-original_orientation)
        s = sin(-original_orientation)
        self.rotation_matrix[0, 0] = c
        self.rotation_matrix[0, 1] = -s
        self.rotation_matrix[1, 0] = s
        self.rotation_matrix[1, 1] = c

        tool_translation = self.rotation_matrix @ tool_translation

        self.x = original_x + tool_translation[0]
        self.y = original_y + tool_translation[1]
        self.orientation += tool_rotation

        if self.game.debuging_visual :self.game.visualize(f'{self} executes speed {len(course)} maneuver.',tool_coord)
        current_overlap = self.is_overlap()
        
        if current_overlap:
            if self.game.debuging_visual :self.game.visualize(f'{self} overlaps ships at speed {len(course)} maneuver.',tool_coord)
            overlap_ships = overlap_ships.union(current_overlap)
            self.x, self.y, self.orientation = original_x, original_y, original_orientation
            new_overlap = self.move_ship(course[:-1], placement, overlap_ships)
            overlap_ships = overlap_ships.union(new_overlap)

        return overlap_ships

    cpdef bint is_overlap_squad(self) :
        """
        determines which squad overlaps to this ship at current location
        
        Returns:
            is_overlap(bool): A set indicating which squads were overlapped.
        """
        cdef: 
            Squad squad
            bint is_overlap = False

        if self.destroyed : return False
        for squad in self.game.squads:
            if squad.destroyed :
                continue

            if cache.is_overlap_s2q(self.get_ship_hash_state(), squad.get_squad_hash_state()):
                is_overlap = True
                squad.overlap_ship_id = self.id
        return is_overlap

    cpdef list get_valid_squad_placement(self, Squad squad) :
        """
        Get a list of valid placements for a squad at the current ship location.
        """
        cdef:
            list valid_placements = []
            tuple original_coords = squad.coords
            cnp.ndarray[cnp.float32_t, ndim=2] squad_placement_points = cache._ship_coordinate(self.get_ship_hash_state())['squad_placement_points']
            int index
            cnp.ndarray[cnp.float32_t, ndim=1] point
            tuple coords

        for index, point in enumerate(squad_placement_points):
            coords = tuple(point.tolist())
            squad.coords = coords
            if squad.out_of_board() or squad.is_overlap():
                continue

            valid_placements.append(index)
        squad.coords = original_coords
        if not valid_placements:
            valid_placements = [None]
        return valid_placements

    cpdef bint check_overlap(self, object obstacle) :
        """
        Check if the ship overlaps with the given obstacle.

        Args:
            obstacle (Obstacle): The obstacle to check against.

        Returns:
            bool: True if there is an overlap, False otherwise.
        """
        return cache.is_overlap_s2o(self.get_ship_hash_state(), obstacle.get_hash_state())

    cpdef void overlap_obstacle(self, Obstacle obstacle) :
        obstacle_type = obstacle.type
        if obstacle_type == ObstacleType.STATION: 
            self.hull = min(self.hull+1, self.max_hull)
        elif obstacle_type == ObstacleType.DEBRIS: 
            max_shield = max(self.shield)
            for hull in HULL_SECTIONS:
                if self.shield[hull] == max_shield: 
                    selected_hull = hull
                    break
            self.defend(selected_hull,2,None)

        elif obstacle_type == ObstacleType.ASTEROID: 
            self.hull = self.hull - 2


# sub method for ship dimension

    def _create_template_geometries(self, ship_dict : dict) -> None:
        """
        Used for visualizer (not game logic)
        """
        front_arc : tuple[float, float] = (ship_dict['front_arc_center'], ship_dict['front_arc_end']) 
        rear_arc : tuple[float, float] = (ship_dict['rear_arc_center'], ship_dict['rear_arc_end'])

        token_half_w = self.token_size[0] / 2
        base_half_w = self.base_size[0] / 2
        base_front_y = (self.base_size[1] - self.token_size[1]) / 2
        base_rear_y = base_front_y - self.base_size[1]


        self.template_base_vertices = np.array([
            [base_half_w, base_front_y], [-base_half_w, base_front_y],
            [-base_half_w, base_rear_y], [base_half_w, base_rear_y]
        ]) + np.array([0, self.token_size[1]/2])

        front_arc_center_pt = (0, -front_arc[0])
        front_left_arc_pt = (-token_half_w, -front_arc[1])
        front_right_arc_pt = (token_half_w, -front_arc[1])
        rear_arc_center_pt = (0, -rear_arc[0])
        rear_left_arc_pt = (-token_half_w, -rear_arc[1])
        rear_right_arc_pt = (token_half_w, -rear_arc[1])
        front_left_token_pt = (-token_half_w, 0)
        front_right_token_pt = (token_half_w, 0)
        rear_left_token_pt = (-token_half_w, -self.token_size[1])
        rear_right_token_pt = (token_half_w, -self.token_size[1])

        self.template_token_vertices = np.array([
            front_right_token_pt, front_left_token_pt,
            rear_left_token_pt, rear_right_token_pt
        ]) + np.array([0, self.token_size[1]/2])

        self.template_hull_vertices = {
            HullSection.FRONT: np.array([
                front_arc_center_pt, front_right_arc_pt, front_right_token_pt, 
                front_left_token_pt, front_left_arc_pt
            ]) + np.array([0, self.token_size[1]/2]),
            HullSection.RIGHT: np.array([
                front_arc_center_pt, front_right_arc_pt, 
                rear_right_arc_pt, rear_arc_center_pt
            ]) + np.array([0, self.token_size[1]/2]),
            HullSection.REAR: np.array([
                rear_arc_center_pt, rear_right_arc_pt, rear_right_token_pt,
                rear_left_token_pt, rear_left_arc_pt
            ]) + np.array([0, self.token_size[1]/2]),
            HullSection.LEFT: np.array([
                front_arc_center_pt, front_left_arc_pt,
                rear_left_arc_pt, rear_arc_center_pt
            ]) + np.array([0, self.token_size[1]/2])
        }
        
        # 0~3 : targeting points
        # 4,5 : Right/Left maneuver tool insert point
        # 6 : ship token center
        self.template_targeting_points_and_maneuver_tool_insert = np.array([
            [0, -ship_dict['front_targeting_point']],
            [ship_dict['side_targeting_point'][0], -ship_dict['side_targeting_point'][1]],
            [0, -ship_dict['rear_targeting_point']],
            [- ship_dict['side_targeting_point'][0], -ship_dict['side_targeting_point'][1]],
            [ (self.base_size[0] + TOOL_WIDTH_HALF)/2, (self.base_size[1]-self.token_size[1])/2],
            [-(self.base_size[0] + TOOL_WIDTH_HALF)/2, (self.base_size[1]-self.token_size[1])/2],
            [0, -self.token_size[1]/2]
        ]) + np.array([0, self.token_size[1]/2])



# sub method for attack
    cpdef bint is_obstruct_s2s(self, int from_hull, Ship to_ship, int to_hull) :
        """
        Checks if the line of sight between two ships is obstructed.

        Args:
            from_hull (HullSection): The firing hull section of the attacking ship.
            to_ship (Ship): The target ship.
            to_hull (HullSection): The targeted hull section of the defending ship.

        Returns:
            obstructed (bool)
        """
        return False
        # simplified
        cdef:
            tuple line_of_sight = (tuple(cache._ship_coordinate(self.get_ship_hash_state())['targeting_points'][from_hull]), tuple(cache._ship_coordinate(to_ship.get_ship_hash_state())['targeting_points'][to_hull]))
            Ship ship
            Obstacle obstacle
            tuple self_point, other_point
        
        self_point = cache.los_point_ship(line_of_sight[0],line_of_sight[1], self.get_ship_hash_state())
        other_point = cache.los_point_ship(line_of_sight[1],line_of_sight[0], to_ship.get_ship_hash_state())
        line_of_sight = (self_point, other_point)
        for ship in self.game.ships:

            if ship.id == self.id or ship.id == to_ship.id:
                continue
            if ship.destroyed:
                continue

            if cache.is_obstruct_ship(line_of_sight, ship.get_ship_hash_state()):
                return True

        for obstacle in self.game.obstacles:
            if cache.is_obstruct_obstacle(line_of_sight, obstacle.get_hash_state()):
                return True

        return False

    cpdef bint is_obstruct_s2q(self, int from_hull, Squad to_squad) :
        """
        Checks if the line of sight between a ship and a squad is obstructed.

        Args:
            hull (HullSection): The firing hull section of the attacking ship.
            squad (Squad): The target squad.
        """
        cdef:
            tuple line_of_sight = (tuple(cache._ship_coordinate(self.get_ship_hash_state())['targeting_points'][from_hull]), to_squad.coords)
            Ship ship
            Obstacle obstacle
            tuple self_point

        self_point = cache.los_point_ship(line_of_sight[0],line_of_sight[1], self.get_ship_hash_state())
        line_of_sight = (self_point, line_of_sight[1])

        for ship in self.game.ships:

            if ship.id == self.id:
                continue
            if ship.destroyed:
                continue

            if cache.is_obstruct_ship(line_of_sight, ship.get_ship_hash_state()):
                return True
                
        for obstacle in self.game.obstacles:
            if cache.is_obstruct_obstacle(line_of_sight, obstacle.get_hash_state()):
                return True

        return False
        

    cpdef tuple gather_dice(self, int attack_hull, int attack_range, bint is_ship) :
        if attack_range in (AttackRange.INVALID, AttackRange.EXTREME): return (0, 0, 0)
        if is_ship : 
            return self.battery_range[attack_hull][attack_range]
        else :
            return self.anti_squad_range[attack_range]

    cpdef void defend(self, int defend_hull, int total_damage, object critical) :
        cdef:
            int shield_damage = min(total_damage, self.shield[defend_hull])
            list shield_list = list(self.shield) 
        shield_list[defend_hull] -= shield_damage
        self.shield = tuple(shield_list)

        total_damage -= shield_damage

        # Apply remaining damage to the hull
        if total_damage > 0:
            self.hull -= total_damage
            if critical == Critical.STANDARD : self.hull -= 1 # Structural Damage

        if self.hull <= 0 : self.destroy()

    cpdef list get_valid_ship_target(self, int attack_hull) :
        cdef:
            list valid_ship_targets = []
            Ship ship
            list target_dict, range_dict
            int target_hull, attack_range
            int dice_count

        for ship in self.game.ships:
            if ship.id == self.id or ship.destroyed or ship.team == self.team: continue

            target_dict, range_dict =  cache.attack_range_s2s(self.get_ship_hash_state(), ship.get_ship_hash_state())

            if not target_dict[attack_hull] : continue

            for target_hull in HULL_SECTIONS :
                attack_range = range_dict[attack_hull][target_hull] 
                
                if attack_range in (AttackRange.INVALID, AttackRange.EXTREME): continue

                dice_count = sum(self.gather_dice(attack_hull, attack_range, is_ship=True))
                if dice_count == 0 : continue
                elif dice_count == 1:
                    if self.is_obstruct_s2s(attack_hull, ship, target_hull) : continue
                valid_ship_targets.append((ship, target_hull))

        return valid_ship_targets

    cpdef list get_valid_target_hull(self, int attack_hull, Ship target_ship) :
        cdef:
            list valid_target_hulls = []
            int target_hull
            int attack_range
            int dice_count
            list range_dict, hull_range_dict

        _, range_dict =  cache.attack_range_s2s(self.get_ship_hash_state(), target_ship.get_ship_hash_state())
        hull_range_dict = range_dict[attack_hull]

        for target_hull in HULL_SECTIONS :
            attack_range = hull_range_dict[target_hull] 
            if attack_range in (AttackRange.INVALID, AttackRange.EXTREME): continue

            dice_count = sum(self.gather_dice(attack_hull, attack_range, is_ship=True))
            if dice_count == 0 : continue
            elif dice_count == 1:
                if self.is_obstruct_s2s(attack_hull, target_ship, target_hull) : continue
            valid_target_hulls.append(target_hull)

        return valid_target_hulls

    cpdef list get_valid_squad_target(self, int attack_hull) :
        cdef:
            list valid_squad_targets = []
            Squad squad
            list range_dict
            int attack_range
            int dice_count

        for squad in self.game.squads:
            if squad.team == self.team or squad.destroyed : continue

            range_dict =  cache.attack_range_s2q(self.get_ship_hash_state(), squad.get_squad_hash_state())
            attack_range = range_dict[attack_hull]
            if attack_range in (AttackRange.INVALID, AttackRange.EXTREME): continue
            dice_count = sum(self.gather_dice(attack_hull, attack_range, is_ship=False))
            if dice_count == 0 : continue
            elif dice_count == 1:
                if self.is_obstruct_s2q(attack_hull, squad) : continue
            valid_squad_targets.append(squad)

        return valid_squad_targets

    cpdef list get_valid_attack_hull(self) :
        """
        Get a list of valid attacking hull sections for the ship.

        Returns:
            valid_attacker (list[HullSection]): A list of valid attacking hull sections.
        """
        cdef:
            int hull
            list valid_attacker = []

        for hull in HULL_SECTIONS :
            if self.attack_history[hull] is None :
                valid_attacker.append(hull)

        return valid_attacker

    cpdef list get_critical_effect(self, bint black_crit, bint blue_crit, bint red_crit) :
        cdef list critical_list = []
        if black_crit or blue_crit or red_crit :
            critical_list.append(Critical.STANDARD)
        return critical_list

    cpdef list get_squad_activation(self) :
        """
        Get the number of squads that can be activated by this ship.

        Returns:
            list[Squad]: A list of squads that can be activated.
        """
        cdef:
            Squad squad
            list valid_squads = []
        for squad in self.game.squads:
            if squad.team == self.team and not squad.activated and not squad.destroyed \
               and cache.range_s2q(self.get_ship_hash_state(), squad.get_squad_hash_state()) <= AttackRange.MEDIUM:
                valid_squads.append(squad)
        return valid_squads

# sub method for execute maneuver
    def _tool_coordination(self, course : tuple[int, ...], placement : int) -> tuple[list[tuple[float, float]], list[float]]:
        """
        Calculates the coordinates and orientations along a maneuver tool's path using NumPy.
        """

        tool_coords = cache._ship_coordinate(self.get_ship_hash_state())['tool_insert_points']
        tool_coord : tuple[float, float] = tool_coords[0] if placement == 1 else tool_coords[1]
        
        if not course:
            # For speed 0, return the starting point and orientation
            return [tool_coord], [self.orientation]
        
        # --- Step 1: Set up initial conditions ---
        initial_position = np.array(tool_coord)
        initial_orientation = self.orientation
        speed = len(course)

        # --- Step 2: Calculate all joint orientations at once ---
        # This array will have shape (speed + 1) and includes the initial orientation
        yaw_changes = np.array([0] + list(course)) * (np.pi / 8)
        joint_orientations = initial_orientation + np.cumsum(yaw_changes)

        # --- Step 3: Calculate the direction vectors for each segment ---
        long_segment_orientations = joint_orientations[:-1]
        short_segment_orientations = joint_orientations[1:]
        
        segment_orientations = np.empty(2 * speed, dtype=np.float32)
        segment_orientations[0::2] = long_segment_orientations
        segment_orientations[1::2] = short_segment_orientations
        
        segment_lengths = np.tile([TOOL_LENGTH, TOOL_PART_LENGTH], speed)
        direction_vectors = np.array([np.sin(segment_orientations), np.cos(segment_orientations)]).T

        # --- Step 4: Calculate all position vectors ---
        position_vectors = direction_vectors * segment_lengths[:, np.newaxis]
        
        # This creates an array of all points along the path, including the start point.
        # The shape will be (2 * speed + 1, 2)
        all_points = np.vstack([initial_position, np.cumsum(position_vectors, axis=0) + initial_position])

        # --- Step 5: Convert back to the required list format (Corrected) ---
        # Return the full list of orientations to match the original function's output.
        # The shape will now be (speed + 1)
        return all_points.tolist(), joint_orientations.tolist()
        
    cpdef set is_overlap(self) :        
        """
        determines which ship overlaps to this ship at current location
        
        Returns:
            overlap_list (set[int]): A list indicating which ships were overlapped.
        """
        cdef:
            set overlap_list = set()
            Ship ship

        for ship in self.game.ships:
            if self.id == ship.id or ship.destroyed :
                continue

            if cache.is_overlap_s2s(self.get_ship_hash_state(), ship.get_ship_hash_state()):
                overlap_list.add(ship.id)
        return overlap_list

    cpdef void overlap_damage(self, set overlap_list) :
        """
        Determines which of the overlapping ships is closest and handles the collision.

        Args:
            overlap_list (set[int]): A set indicating which ships were overlapped.
        """
        if not overlap_list: return
        cdef:
            Ship closest_ship
            float min_distance = float('inf')
            int ship_id
            Ship other_ship
            float distance

        for ship_id in overlap_list:
            other_ship = self.game.ships[ship_id]

            # Calculate the distance between the two ship bases.
            distance = cache.distance_s2s(self.get_ship_hash_state(), other_ship.get_ship_hash_state())
            if distance < min_distance:
                min_distance = distance
                closest_ship = other_ship

        self.hull -= 1
        closest_ship.hull -= 1
        self.game.visualize(f"\n{self} overlaps to {closest_ship}.")
        if self.hull <= 0 : self.destroy()
        if closest_ship.hull <= 0 :closest_ship.destroy()

    cpdef bint out_of_board(self) :
        """
        Checks if the ship's base is completely within the game board.

        Returns:
            bool: True if the ship is out of the board, False otherwise.
        """
        cdef:
            dict coords = cache._ship_coordinate(self.get_ship_hash_state())
            cnp.ndarray[cnp.float32_t, ndim=2] base_corners = coords['base_corners']
            float min_x, min_y, max_x, max_y
        min_x, min_y = base_corners.min(axis=0)
        max_x, max_y = base_corners.max(axis=0)

        if min_x >= 0 and max_x <= self.game.player_edge and min_y >= 0 and max_y <= self.game.short_edge:
            return False
        
        return True
    
    cpdef list get_valid_speed(self) :
        """
        Get a list of valid speeds for the ship based on its navchart.

        Returns:
            list[int]: A list of valid speeds.
        """
        cdef:
            list valid_speed = []
            int speed_change = int(Command.NAV in self.command_dial) + int(Command.NAV in self.command_token)
            int speed

        for speed in range(5):
            if abs(speed - self.speed) > speed_change:
                continue
            if self.nav_chart.get(speed) is not None or speed == 0:
                valid_speed.append(speed)

        return valid_speed

    cpdef list get_valid_yaw(self, int speed, int joint) :
        """
        Get a list of valid yaw adjustments for the ship based on its navchart.

        Args:
            speed (int): The current speed of the ship.
            joint (int): The joint index for which to get valid yaw adjustments.

        Returns:
            list[int]: A list of valid yaw adjustments. -2 ~ 2
        """
        cdef:
            list valid_yaw = []
            int yaw
            int max_yaw = self.nav_chart[speed][joint]

        for yaw in range(5):
            if abs(yaw - 2) > max_yaw:
                continue
            valid_yaw.append(yaw - 2)

        return valid_yaw
    
    cpdef tuple nav_command_used(self, tuple course) :
        """
        Args:
            course (tuple[int, ...]): The maneuver course of the ship.
        Returns:
            tuple (nav_dial_used (bool), nav_token_used (bool))
        """
        cdef:
            bint nav_dial_used = False
            bint nav_token_used = False
            int new_speed = len(course)
            int speed_change = abs(new_speed - self.speed)
            bint is_standard

        # 1. SPEED CHANGE VALIDATION
        if speed_change > 2:
            raise ValueError(f"Invalid speed change of {speed_change}. Maximum is 2.")

        if speed_change == 1:
            # Must use one NAVIGATE command from either dial or token
            if Command.NAV in self.command_dial:
                nav_dial_used = True
            elif Command.NAV in self.command_token:
                nav_token_used = True
            else:
                raise ValueError("Speed change of 1 requires a NAVIGATE command from a dial or token.")

        if speed_change == 2:
            # Must use two NAVIGATE commands, one from each source
            if Command.NAV in self.command_dial and Command.NAV in self.command_token:
                nav_dial_used = True
                nav_token_used = True
            else:
                raise ValueError("Speed change of 2 requires NAVIGATE commands from BOTH a dial and a token.")

        # 2. COURSE VALIDITY CHECK (for extra clicks)
        is_standard = self.is_standard_course(course)
        
        if not is_standard:
            # This is a special course that requires an extra click from the command dial.
            if Command.NAV in self.command_dial:
                nav_dial_used = True
            else:
                raise ValueError("This course requires an extra click, which needs a NAVIGATE command from the dial.")
        return nav_dial_used, nav_token_used
    
    cpdef bint is_standard_course(self, tuple course) :
        """
        Checks if a given course is a standard maneuver for a given speed,
        without using any special abilities like adding a click.
        """
        cdef:
            int speed = len(course)
            int joint, yaw, max_yaw
        if speed == 0 : return True

        for joint, yaw in enumerate(course) :
            max_yaw = self.nav_chart[speed][joint]
            if abs(yaw) > max_yaw : return False
        return True

    cpdef list get_all_possible_courses(self, int speed) :
        """
        Gets all possible maneuver courses for a given speed.

        If can_add_click is True, it will also generate additional courses by modifying
        a single joint of any originally valid course by one click (yaw).

        Args:
            speed (int): The maneuver speed.

        Returns:
            A list of all unique, valid course lists.
        """
        cdef:
            bint has_nav_dial = Command.NAV in self.command_dial
            tuple cache_key = (speed, has_nav_dial)
            list original_yaw_options
            set all_courses, standard_courses
            int i
            list final_result
            int original_yaw, modified_yaw_add, modified_yaw_sub

        if cache_key in self._course_cache :
            return self._course_cache[cache_key]
        
        if speed == 0:
            return [()] # A speed 0 maneuver has an empty course

        # 1. Get the base valid yaw options for each joint from the nav chart.
        original_yaw_options = [self.get_valid_yaw(speed, joint) for joint in range(speed)]

        # 2. Generate all standard courses using the Cartesian product.
        # Using a set handles duplicates automatically.
        all_courses = set(itertools.product(*original_yaw_options))

        # 3. If the special condition isn't met, return the standard courses.
        if not Command.NAV in self.command_dial:
            return [course for course in all_courses]

        # 4. If the condition is met, generate new courses from the standard ones.
        # We start with a copy of the original courses to iterate over.
        standard_courses = all_courses.copy()
        for course_tuple in standard_courses:
            course_list = list(course_tuple)
            # Iterate through each joint in the current standard course.
            for i in range(len(course_list)):
                original_yaw = course_list[i]

                # Option A: Add one click to the current joint
                modified_yaw_add = original_yaw + 1
                if abs(modified_yaw_add) <= 2:
                    new_course = list(course_list) # Create a copy
                    new_course[i] = modified_yaw_add
                    all_courses.add(tuple(new_course))

                # Option B: Subtract one click from the current joint
                modified_yaw_sub = original_yaw - 1
                if abs(modified_yaw_sub) <= 2:
                    new_course = list(course_list) # Create a copy
                    new_course[i] = modified_yaw_sub
                    all_courses.add(tuple(new_course))


        final_result = sorted(list(all_courses))
        self._course_cache[cache_key] = final_result
        return final_result

    cpdef list get_valid_placement(self, tuple course) :
        """
        Get a list of valid placements for the ship based on its navchart.

        Args:
            course (list[int]): The course of the ship.

        Returns:
            list[int]: A list of valid placements. 
                1 for right, -1 for left
        """
        cdef:
            int speed = len(course)
            list valid_placement = [-1, 1]
        if speed == 0 : return [0]
        
        if speed > 0 :
            if course[-1] > 0 : valid_placement.remove(-1)
            elif course[-1] < 0 : valid_placement.remove(1)
            else : 
                if speed >= 2 and self.size_class > <int>SizeClass.SMALL :
                    if course[-2] > 0 : valid_placement.remove(-1)
                    elif course[-2] < 0 : valid_placement.remove(1)
        return valid_placement

    cdef object get_snapshot(self) :
        return (
            self.x, self.y, self.orientation, self.speed,
            self.hull, self.shield,
            self.destroyed, self.activated,
            self.command_stack, self.command_dial, self.command_token, self.resolved_command,
            self.attack_count, self.attack_history, 
            self.engineer_point, self.repaired_hull, 
            [(<DefenseToken>dt).get_snapshot() for dt in self.defense_tokens]
        )
    
    cdef void revert_snapshot(self, object snapshot):
        """Restores the ship's state from a snapshot."""
        (
            self.x, self.y, self.orientation, self.speed,
            self.hull, self.shield,
            self.destroyed, self.activated,
            self.command_stack, self.command_dial, self.command_token, self.resolved_command,
            self.attack_count, self.attack_history, 
            self.engineer_point, self.repaired_hull, 
            defense_tokens_state
        ) = snapshot

        for id, token_state in enumerate(defense_tokens_state):
            (<DefenseToken>self.defense_tokens[id]).revert_snapshot(token_state)

    cpdef object get_ship_hash_state(self):
        """Returns a hashable tuple representing the ship's state."""
        cdef int x_int = <int>(self.x * HASH_PRECISION)
        cdef int y_int = <int>(self.y * HASH_PRECISION)
        cdef int orientation_int = <int>(self.orientation * HASH_PRECISION)

        return (self.name, x_int, y_int, orientation_int)

