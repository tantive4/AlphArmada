from __future__ import annotations
from typing import TYPE_CHECKING
from collections import Counter
import itertools

import numpy as np


from dice import *
from defense_token import DefenseToken, TokenType
from measurement import *
from enum_class import *
import cache_function as cache
if TYPE_CHECKING:
    from armada import Armada
    from squad import Squad




class Ship:
    def __init__(self, ship_dict : dict, player : int) -> None:
        self.player : int = player
        self.name : str = ship_dict['name']

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
        
        self.defense_tokens: dict[int, DefenseToken] = {}
        token_counts = Counter()
        # Iterate through the list of token strings from the JSON
        for token_type_str in ship_dict['defense_token']:
            token_enum = TokenType[token_type_str.upper()]
            # token_counts[token_enum] will be 0 for the first, 1 for the second, etc.
            key = token_enum * 2 + token_counts[token_enum]
            # Add the token to the dictionary and increment the count for that type
            self.defense_tokens[key] = DefenseToken(token_type_str)
            token_counts[token_enum] += 1

        self.nav_chart : dict[int, list[int]] = {int(k) : v for k, v in ship_dict['navchart'].items()}
        self.max_shield : dict[HullSection, int] = {HullSection.FRONT : ship_dict['shield'][0], 
                                                    HullSection.RIGHT : ship_dict['shield'][1], 
                                                    HullSection.REAR : ship_dict['shield'][2], 
                                                    HullSection.LEFT : ship_dict['shield'][1]}
        self.point : int = ship_dict['point']
        self.command_value : int = ship_dict['command']
        self.engineer_value : int = ship_dict['engineering']
        self.squad_value : int = ship_dict['squadron']
        self.destroyed = False

        self.front_arc : tuple[float, float] = (ship_dict['front_arc_center'], ship_dict['front_arc_end']) 
        self.rear_arc : tuple[float, float] = (ship_dict['rear_arc_center'], ship_dict['rear_arc_end'])
        
        self._create_template_geometries(ship_dict)
        self._course_cache : dict[tuple[int, bool], list[tuple[int, ...]]]= {}
        
    def __str__(self):
        return self.name
    __repr__ = __str__



    def deploy(self, game : Armada , x : float, y : float, orientation : float, speed : int, ship_id : int) -> None:
        """
        deploy the ship to the game board

        Args:
            game (Armada) : the game
            (x, y) (float) : coordination of front center of the **ship token**
            orientation (float) : ship orientation
            speed (int) : ship speed
            ship_id (int) : ship id
        """
        self.game : Armada = game
        self.x: float = x
        self.y: float = y
        self.orientation: float = orientation
        self.speed: int = speed

        self.hull: int = self.max_hull
        self.shield: tuple[int, ...] = tuple(self.max_shield[hull] for hull in HULL_SECTIONS)
        self.id: int = ship_id
        self.command_stack: tuple[Command, ...] = ()
        self.command_dial : tuple[Command, ...] = ()
        self.command_token : tuple[Command, ...] = ()
        self.resolved_command : tuple[Command, ...] = ()
        self.engineer_point : int = 0
        self.attack_count : int = 0
        self.attack_history : tuple[tuple[int, HullSection]|tuple[int, ...]|None, ...] = (None, None, None, None)
        self.repaired_hull : tuple[HullSection, ...] = ()
        self.status_phase()
    
    def asign_command(self, Command) -> None :
        if len(self.command_stack) >= self.command_value : raise ValueError("Cannot asigne more command then Command Value")
        self.command_stack += (Command,)


    def destroy(self) -> None:
        self.destroyed = True
        self.hull = 0
        self.shield = (0, 0, 0, 0)
        for token in self.defense_tokens.values() :
            if not token.discarded : token.discard()
        self.game.visualize(f'{self} is destroyed!')

    def status_phase(self) -> None:
        self.activated = False
        for token in self.defense_tokens.values():
            if not token.discarded:
                token.ready()

    def end_activation(self) -> None :
        self.activated = True
        self.game.active_ship = None
        self.attack_history = (None, None, None, None)
        self.attack_count = 0
        self.command_dial = ()
        self.resolved_command = ()

    def execute_maneuver(self, course : tuple[int, ...], placement : int) -> None:
        overlap_ships = self.move_ship(course, placement, set())
        self.overlap_damage(overlap_ships)
        if self.out_of_board() :
            self.game.visualize(f'{self} is out of board!')
            self.destroy()
            
            

    def move_ship(self, course : tuple[int, ...], placement : int, overlap_ships : set[int]) -> set[int]:
        if not course :
            self.game.visualize(f'{self} executes speed 0 maneuver.')
            return overlap_ships
        if self.game.debuging_visual : tool_coord = self._tool_coordination(course, placement)[0]
        
        original_position, original_orientation = np.array([self.x, self.y]), self.orientation

        tool_translation, tool_rotation = cache.maneuver_tool(self.size_class, course, placement)
        # change translation vector according to the current orientation
        # rotation matrix use CW orientation    
        c = np.cos(-original_orientation)
        s = np.sin(-original_orientation)
        rotation_matrix = np.array([[c, -s],
                                    [s,  c]])
        tool_translation = rotation_matrix @ tool_translation
        self.x, self.y = original_position + tool_translation
        self.orientation += tool_rotation
        if self.game.debuging_visual :self.game.visualize(f'{self} executes speed {len(course)} maneuver.',tool_coord)
        current_overlap = self.is_overlap()
        
        if current_overlap:
            if self.game.debuging_visual :self.game.visualize(f'{self} overlaps ships at speed {len(course)} maneuver.',tool_coord)
            overlap_ships = overlap_ships.union(current_overlap)
            self.x, self.y, self.orientation = *original_position, original_orientation
            new_overlap = self.move_ship(course[:-1], placement, overlap_ships)
            overlap_ships = overlap_ships.union(new_overlap)

        return overlap_ships

    def is_overlap_squad(self) -> bool:
        """
        determines which squad overlaps to this ship at current location
        
        Returns:
            overlap_list (set[Squad]): A set indicating which squads were overlapped.
        """
        is_overlap : bool = False
        for squad in self.game.squads:
            if squad.destroyed :
                continue

            if cache.is_overlap_s2q(self.get_ship_hash_state(), squad.get_squad_hash_state()):
                is_overlap = True
                squad.overlap_ship_id = self.id
        return is_overlap

    def get_valid_squad_placement(self, squad : Squad) -> list[int|None] :
        """
        Get a list of valid placements for a squad at the current ship location.

        Returns:
            valid_placements (list[tuple[float, float]]): A list of valid (x, y) placements for the squad.
        """
        valid_placements : list[int|None] = []
        original_coords = squad.coords

        squad_placement_points = cache._ship_coordinate(self.get_ship_hash_state())['squad_placement_points']
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


# sub method for ship dimension

    def _create_template_geometries(self, ship_dict : dict) -> None:
        """
        Creates template vertices (points) for all ship shapes, relative to a single
        (0,0) pivot at the token's front-center.
        """
        token_half_w = self.token_size[0] / 2

        base_half_w = self.base_size[0] / 2
        base_front_y = (self.base_size[1] - self.token_size[1]) / 2
        base_rear_y = base_front_y - self.base_size[1]
        self.template_base_vertices = np.array([
            [base_half_w, base_front_y], [-base_half_w, base_front_y],
            [-base_half_w, base_rear_y], [base_half_w, base_rear_y]
        ])

        front_arc_center_pt = (0, -self.front_arc[0])
        front_left_arc_pt = (-token_half_w, -self.front_arc[1])
        front_right_arc_pt = (token_half_w, -self.front_arc[1])
        rear_arc_center_pt = (0, -self.rear_arc[0])
        rear_left_arc_pt = (-token_half_w, -self.rear_arc[1])
        rear_right_arc_pt = (token_half_w, -self.rear_arc[1])
        front_left_token_pt = (-token_half_w, 0)
        front_right_token_pt = (token_half_w, 0)
        rear_left_token_pt = (-token_half_w, -self.token_size[1])
        rear_right_token_pt = (token_half_w, -self.token_size[1])

        self.template_token_vertices = np.array([
            front_right_token_pt, front_left_token_pt,
            rear_left_token_pt, rear_right_token_pt
        ])

        self.template_hull_vertices = {
            HullSection.FRONT: np.array([
                front_arc_center_pt, front_right_arc_pt, front_right_token_pt, 
                front_left_token_pt, front_left_arc_pt
            ]),
            HullSection.RIGHT: np.array([
                front_arc_center_pt, front_right_arc_pt, 
                rear_right_arc_pt, rear_arc_center_pt
            ]),
            HullSection.REAR: np.array([
                rear_arc_center_pt, rear_right_arc_pt, rear_right_token_pt,
                rear_left_token_pt, rear_left_arc_pt
            ]),
            HullSection.LEFT: np.array([
                front_arc_center_pt, front_left_arc_pt,
                rear_left_arc_pt, rear_arc_center_pt
            ])
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
        ])





# sub method for attack
    def is_obstruct_s2s(self, from_hull : HullSection, to_ship : Ship, to_hull : HullSection) -> bool :
        """
        Checks if the line of sight between two ships is obstructed.

        Args:
            from_hull (HullSection): The firing hull section of the attacking ship.
            to_ship (Ship): The target ship.
            to_hull (HullSection): The targeted hull section of the defending ship.

        Returns:
            obstructed (bool)
        """
        line_of_sight : tuple[tuple[float, float], ...] = (tuple(cache._ship_coordinate(self.get_ship_hash_state())['targeting_points'][from_hull]), tuple(cache._ship_coordinate(to_ship.get_ship_hash_state())['targeting_points'][to_hull]))

        for ship in self.game.ships:

            if ship.id == self.id or ship.id == to_ship.id:
                continue
            if ship.destroyed:
                continue

            if cache.is_obstruct(line_of_sight, ship.get_ship_hash_state()):
                return True

        return False

    def is_obstruct_s2q(self, from_hull: HullSection, to_squad: Squad) -> bool :
        """
        Checks if the line of sight between a ship and a squad is obstructed.

        Args:
            hull (HullSection): The firing hull section of the attacking ship.
            squad (Squad): The target squad.
        """
        line_of_sight : tuple[tuple[float, float], ...] = (tuple(cache._ship_coordinate(self.get_ship_hash_state())['targeting_points'][from_hull]), to_squad.coords)

        for ship in self.game.ships:

            if ship.id == self.id:
                continue
            if ship.destroyed:
                continue

            if cache.is_obstruct(line_of_sight, ship.get_ship_hash_state()):
                return True

        return False
        

    def gather_dice(self, attack_hull : HullSection, attack_range : AttackRange, *, is_ship : bool) -> tuple[int, ...] :
        if attack_range in (AttackRange.INVALID, AttackRange.EXTREME): return (0, 0, 0)
        if is_ship :
            return self.battery_range[attack_hull][attack_range]
        else :
            return self.anti_squad_range[attack_range]

    def defend(self, defend_hull : HullSection, total_damage : int, critical: Critical | None) -> None:
        # Absorb damage with shields first
        shield_damage = min(total_damage, self.shield[defend_hull])

        shield_list = list(self.shield)
        shield_list[defend_hull] -= shield_damage
        self.shield = tuple(shield_list)

        total_damage -= shield_damage

        # Apply remaining damage to the hull
        if total_damage > 0:
            self.hull -= total_damage
            if critical == Critical.STANDARD : self.hull -= 1 # Structural Damage

        if self.hull <= 0 : self.destroy()

    def get_valid_ship_target(self, attack_hull : HullSection) -> list[tuple[Ship, HullSection]] :
        valid_ship_targets : list[tuple[Ship, HullSection]]= []

        for ship in self.game.ships:
            if ship.id == self.id or ship.destroyed or ship.player == self.player: continue

            target_dict, range_dict =  cache.attack_range_s2s(self.get_ship_hash_state(), ship.get_ship_hash_state())

            if not target_dict[attack_hull] : continue

            for target_hull in HULL_SECTIONS :
                attack_range : AttackRange = range_dict[attack_hull][target_hull] 
                
                if attack_range in (AttackRange.INVALID, AttackRange.EXTREME): continue

                dice_count = sum(self.gather_dice(attack_hull, attack_range, is_ship=True))
                if dice_count == 0 : continue
                elif dice_count == 1:
                    if self.is_obstruct_s2s(attack_hull, ship, target_hull) : continue
                valid_ship_targets.append((ship, target_hull))

        return valid_ship_targets

    def get_valid_squad_target(self, attack_hull:HullSection) -> list[Squad] :
        valid_squad_targets : list[Squad]= []
        for squad in self.game.squads:
            if squad.player == self.player or squad.destroyed : continue

            range_dict :list[AttackRange] =  cache.attack_range_s2q(self.get_ship_hash_state(), squad.get_squad_hash_state())
            attack_range : AttackRange = range_dict[attack_hull]
            if attack_range in (AttackRange.INVALID, AttackRange.EXTREME): continue
            dice_count = sum(self.gather_dice(attack_hull, attack_range, is_ship=False))
            if dice_count == 0 : continue
            elif dice_count == 1:
                if self.is_obstruct_s2q(attack_hull, squad) : continue
            valid_squad_targets.append(squad)

        return valid_squad_targets

    def get_valid_attack_hull(self) -> list[HullSection]:
        """
        Get a list of valid attacking hull sections for the ship.

        Returns:
            valid_attacker (list[HullSection]): A list of valid attacking hull sections.
        """
        valid_attacker = [hull for hull in HULL_SECTIONS if self.attack_history[hull] is None]

        return valid_attacker

    def get_critical_effect(self, black_crit : bool, blue_crit : bool, red_crit : bool) -> list[Critical] :
        critical_list : list[Critical] = []
        if black_crit or blue_crit or red_crit :
            critical_list.append(Critical.STANDARD)
        return critical_list

    def get_squad_activation(self) -> list[Squad] :
        """
        Get the number of squads that can be activated by this ship.

        Returns:
            list[Squad]: A list of squads that can be activated.
        """
        return [squad for squad in self.game.squads 
                if squad.player == self.player and not squad.activated and not squad.destroyed 
                and cache.range_s2q(self.get_ship_hash_state(), squad.get_squad_hash_state()) <= AttackRange.MEDIUM]


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
        
    def is_overlap(self) -> set[int] :        
        """
        determines which ship overlaps to this ship at current location
        
        Returns:
            overlap_list (list[bool]): A list indicating which ships were overlapped.
        """
        overlap_list = set()
        for ship in self.game.ships:
            if self.id == ship.id or ship.destroyed :
                continue

            if cache.is_overlap_s2s(self.get_ship_hash_state(), ship.get_ship_hash_state()):
                overlap_list.add(ship.id)
        return overlap_list

    def overlap_damage(self, overlap_list : set[int]) -> None:
        """
        Determines which of the overlapping ships is closest and handles the collision.

        Args:
            overlap_list (set[int]): A set indicating which ships were overlapped.
        """
        closest_ship = None
        min_distance = float('inf')

        for ship_id in overlap_list:
            other_ship = self.game.ships[ship_id]

            # Calculate the distance between the two ship bases.
            distance = cache.distance_s2s(self.get_ship_hash_state(), other_ship.get_ship_hash_state())
            if distance < min_distance:
                min_distance = distance
                closest_ship = other_ship

        if closest_ship:
            self.hull -= 1
            closest_ship.hull -= 1
            self.game.visualize(f"\n{self} overlaps to {closest_ship}.")
            if self.hull <= 0 : self.destroy()
            if closest_ship.hull <= 0 :closest_ship.destroy()

    def out_of_board(self) -> bool:
        """
        Checks if the ship's base is completely within the game board.

        Returns:
            bool: True if the ship is out of the board, False otherwise.
        """
        coords = cache._ship_coordinate(self.get_ship_hash_state())
        base_corners = coords['base_corners']
        min_x, min_y = base_corners.min(axis=0)
        max_x, max_y = base_corners.max(axis=0)

        if min_x >= 0 and max_x <= self.game.player_edge and min_y >= 0 and max_y <= self.game.short_edge:
            return False
        
        return True
    
    def get_valid_speed(self) -> list[int]:
        """
        Get a list of valid speeds for the ship based on its navchart.

        Returns:
            list[int]: A list of valid speeds.
        """
        valid_speed = []
        speed_change : int = int(Command.NAV in self.command_dial) + int(Command.NAV in self.command_token)

        for speed in range(5):
            if abs(speed - self.speed) > speed_change:
                continue
            if self.nav_chart.get(speed) is not None or speed == 0:
                valid_speed.append(speed)

        return valid_speed
    
    def get_valid_yaw(self, speed : int, joint : int) -> list[int]:
        """
        Get a list of valid yaw adjustments for the ship based on its navchart.

        Args:
            speed (int): The current speed of the ship.
            joint (int): The joint index for which to get valid yaw adjustments.

        Returns:
            list[int]: A list of valid yaw adjustments. -2 ~ 2
        """
        valid_yaw = []
        
        for yaw in range(5):
            if abs(yaw - 2) > self.nav_chart[speed][joint]:
                continue
            valid_yaw.append(yaw -2)

        return valid_yaw
    
    def nav_command_used(self, course: tuple[int, ...]) -> tuple[bool, bool] :
            nav_dial_used = False
            nav_token_used = False
            new_speed = len(course)

            # 1. SPEED CHANGE VALIDATION
            speed_change = abs(new_speed - self.speed)

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
    
    def is_standard_course(self, course:tuple[int, ...]) -> bool :
        """
        Checks if a given course is a standard maneuver for a given speed,
        without using any special abilities like adding a click.
        """
        speed = len(course)
        if speed == 0 : return True

        for joint, yaw in enumerate(course) :
            if abs(yaw) > self.nav_chart[speed][joint] : return False
        return True

    def get_all_possible_courses(self, speed: int) -> list[tuple[int, ...]]:
        """
        Gets all possible maneuver courses for a given speed.

        If can_add_click is True, it will also generate additional courses by modifying
        a single joint of any originally valid course by one click (yaw).

        Args:
            speed (int): The maneuver speed.

        Returns:
            A list of all unique, valid course lists.
        """
        has_nav_dial = Command.NAV in self.command_dial
        cache_key = (speed, has_nav_dial)

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
    
    def get_valid_placement(self, course : tuple[int, ...]) -> list[int]:
        """
        Get a list of valid placements for the ship based on its navchart.

        Args:
            course (list[int]): The course of the ship.

        Returns:
            list[int]: A list of valid placements. 
                1 for right, -1 for left
        """
        speed = len(course)
        if speed == 0 : return [0]
        
        valid_placement = [-1, 1]
        if speed > 0 :
            if course[-1] > 0 : valid_placement.remove(-1)
            elif course[-1] < 0 : valid_placement.remove(1)
            else : 
                if speed >= 2 and self.size_class > SizeClass.SMALL :
                    if course[-2] > 0 : valid_placement.remove(-1)
                    elif course[-2] < 0 : valid_placement.remove(1)
        return valid_placement

    def get_snapshot(self) -> tuple :
        return (
            self.x, self.y, self.orientation, self.speed,
            self.hull, self.shield,
            self.destroyed, self.activated,
            self.command_stack, self.command_dial, self.command_token, self.resolved_command,
            self.attack_count, self.attack_history, 
            self.engineer_point, self.repaired_hull, 
            {key: (dt.readied, dt.discarded, dt.accuracy) 
             for key, dt in self.defense_tokens.items()}
        )
    
    def revert_snapshot(self, snapshot: tuple) -> None:
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

        for key, token_state in defense_tokens_state.items():
            self.defense_tokens[key].readied, self.defense_tokens[key].discarded, self.defense_tokens[key].accuracy = token_state

    def get_ship_hash_state(self) -> tuple[str, int, int, int]:
        """Returns a hashable tuple representing the ship's state."""
        return (self.name, int(self.x*HASH_PRECISION), int(self.y*HASH_PRECISION), int(self.orientation*HASH_PRECISION))

