# cython: profile=True

from __future__ import annotations
import random
import math

import numpy as np
cimport numpy as cnp
cnp.import_array()
from libc.math cimport ceil, floor, fabs, cos, sin, M_PI
from libc.string cimport memset
from learning.params.configs import Config
from armada_game.helpers.action_phase import Phase, ActionType, get_action_str, phase_type
from armada_game.helpers import visualizer
from ship cimport Ship
from squad cimport Squad
from obstacle cimport Obstacle
from attack_info cimport AttackInfo
from squad_geometry cimport Rect, rect_dist_sq, rect_bounds, segment_hits_rect, segment_hits_polygon
from armada_game.helpers.enum_class import *
from armada_game.helpers.dice import *
from armada_game.helpers.damage_deck import *
from armada_game.helpers.measurement import *
from armada_game.helpers import cache_function as cache
from defense_token cimport DefenseToken


# --- Squadron geometry constants (TODO.md, G1-G19) ---
cdef enum:
    MOVE_GRID_N = 33
if SQUAD_MOVE_GRID != MOVE_GRID_N:
    raise ImportError(f'SQUAD_MOVE_GRID ({SQUAD_MOVE_GRID}) must match the compiled move grid ({MOVE_GRID_N})')

cdef int MOVE_CENTER = SQUAD_MOVE_CENTER
cdef double MOVE_PITCH = SQUAD_MOVE_PITCH
cdef double SQ_R = SQUAD_BASE_RADIUS
cdef double Q2Q = Q2Q_RANGE
cdef double Q2S = Q2S_RANGE
# Strict overlap thresholds on squared distances (G4, G6).  Touching is not
# overlapping, so each test sits SQUAD_TOUCH_EPS inside the touching distance.
cdef double SQUAD_SQUAD_OVERLAP_SQ = (2.0 * SQUAD_BASE_RADIUS - SQUAD_TOUCH_EPS) ** 2
cdef double SQUAD_SHIP_OVERLAP_SQ = (SQUAD_BASE_RADIUS - SQUAD_TOUCH_EPS) ** 2
cdef double SQUAD_SHIP_TOUCH_SQ = (SQUAD_BASE_RADIUS + SQUAD_TOUCH_EPS) ** 2
cdef double COMMAND_RANGE_SQ = (MEDIUM_RANGE + SQUAD_BASE_RADIUS) ** 2

# Static reachable disk per speed (G7, G14): a cell (di, dj) is reachable at
# speed s iff its centre is within DISTANCE[s] of the squadron's centre (G1).
_move_disk = np.zeros((6, MOVE_GRID_N, MOVE_GRID_N), dtype=np.uint8)
for _speed in range(6):
    _reach = DISTANCE[_speed] / SQUAD_MOVE_PITCH
    for _i in range(MOVE_GRID_N):
        for _j in range(MOVE_GRID_N):
            if (_i - SQUAD_MOVE_CENTER) ** 2 + (_j - SQUAD_MOVE_CENTER) ** 2 <= _reach * _reach + 1e-9:
                _move_disk[_speed, _i, _j] = 1
cdef cnp.uint8_t[:, :, ::1] MOVE_DISK = _move_disk

# Prebuilt action tuples: a SQUAD_MOVE node can expand up to 861 children.
cdef list MOVE_ACTIONS = [[('move_squad_action', (i - SQUAD_MOVE_CENTER, j - SQUAD_MOVE_CENTER))
                           for j in range(MOVE_GRID_N)] for i in range(MOVE_GRID_N)]
cdef list PLACE_ACTIONS = [('place_squad_action', slot) for slot in range(SQUAD_PLACEMENT_SLOTS)]

# Second-ring placement candidates (RRG Overlapping): 12 positions touching an
# anchor squadron, every 30 degrees.
cdef int RING2_COUNT = 12
cdef list RING2_OFFSETS = [(2.0 * SQUAD_BASE_RADIUS * math.cos(k * math.pi / 6),
                            2.0 * SQUAD_BASE_RADIUS * math.sin(k * math.pi / 6)) for k in range(12)]

# --- Forced single-action steps ---
# Each of these is the only legal action whenever it is offered alone, and
# taking it changes nothing the player could have chosen differently.  Such a
# step is resolved inline instead of costing MCTS a network evaluation.
cdef tuple PASS_COMMAND_TOKEN = ('pass_command_token', None)
cdef tuple PASS_SQUAD_COMMAND = ('pass_squad_command', None)
cdef tuple PASS_REPAIR_COMMAND = ('pass_repair_command', None)
cdef tuple PASS_REPAIR = ('pass_repair', None)
cdef tuple PASS_ATTACK = ('pass_attack', None)
cdef tuple GATHER_UNOBSTRUCTED = ('gather_dice_action', (0, 0, 0))
cdef tuple PASS_ATTACK_EFFECT = ('pass_attack_effect', None)
cdef tuple PASS_DEFENSE_TOKEN = ('pass_defense_token', None)
cdef tuple RESOLVE_NO_REDIRECT = ('resolve_damage_action', None)
cdef tuple PASS_ADDITIONAL_SQUAD_TARGET = ('pass_additional_squad_target', None)
cdef tuple PASS_ACTIVATE_SQUAD = ('pass_activate_squad', None)
cdef tuple PASS_ATTACK_SQUAD = ('pass_attack_squad', None)
cdef frozenset FORCEABLE_ACTIONS = frozenset((
    PASS_COMMAND_TOKEN, PASS_SQUAD_COMMAND, PASS_REPAIR_COMMAND, PASS_REPAIR, PASS_ATTACK,
    GATHER_UNOBSTRUCTED, PASS_ATTACK_EFFECT, PASS_DEFENSE_TOKEN, RESOLVE_NO_REDIRECT,
    PASS_ADDITIONAL_SQUAD_TARGET, PASS_ACTIVATE_SQUAD, PASS_ATTACK_SQUAD,
))


cdef class Armada:
    """
    The main class representing the Armada game.
    """
    def __init__(self, faction: tuple[Faction, Faction], para_index:int = 0) -> None:
        self.player_edge = LONG_RANGE * 3
        self.short_edge = LONG_RANGE * 3
        self.ships : list[Ship] = []
        self.squads : list[Squad] = []
        self.obstacles : list[Obstacle] = []

        self.round = 1
        self.first_player = 1
        self.second_player = -1
        self.first_faction = faction[0].value
        self.second_faction = faction[1].value

        self.phase : Phase = Phase.COMMAND_PHASE
        self.current_player : int = self.first_player
        self.decision_player : int = self.first_player
        self.active_ship : Ship | None = None
        self.defend_ship : Ship | None = None
        self.active_squad : Squad | None = None
        self.defend_squad : Squad | None = None
        self.attack_info : AttackInfo | None = None
        self.squad_activation_count = 0
        self.squad_can_move = False
        self.squad_can_attack = False
        self.pending_placement = ()

        # Pending faceup damage card draws (Phase.DRAW_DAMAGE_CARD chance node)
        self.damage_draw_ship_id = -1
        self.damage_draw_count = 0
        self.damage_draw_source = DAMAGE_SOURCE_NONE

        self.winner = 0.0
        self.image_counter = 0
        self.debuging_visual = <bint>False
        self.simulation_player = 0
        self.para_index = para_index
        self.fleet_check_pending = False
        self.verify_forced = False

        self.ship_geometry_valid = False
        self.ship_geometry = np.zeros((0, 8), dtype=np.float64)
        self.squad_rows_valid = 0
        self.squad_in_range = []
        self.squad_engaged = []
        self.squad_pair_los = {}
        self.heavy_mask = 0
        self.escort_mask = 0
        self.obstacle_polygons = []
        self.obstacle_bounds = np.zeros((0, 4), dtype=np.float64)

        self.ship_static_encode_array = np.zeros((Config.MAX_SHIPS, Config.SHIP_STATIC_OFFSET), dtype=np.float32)

    def rollout(self, max_simulation_step : int = 2000) -> float :
        """
        The main game loop with two players defined by functions.
        Each player function should return an action based on the current game state.
        """

        player1 = self.random_decision
        player2 = self.random_decision
        simulation_counter = 0


        while self.winner == 0.0:
            if self.decision_player == 0 : # chance node case
                action : ActionType = self.sample_chance_action()

            elif self.decision_player == 1 :
                action : ActionType = player1()
            else :
                action : ActionType = player2()


            # print(get_action_str(self, action))

            self.apply_action(action)

            simulation_counter += 1
            if simulation_counter >= max_simulation_step:
                raise RuntimeError(f'Maximum simulation steps reached: {max_simulation_step}\n{Phase(self.phase)}\n{self.get_snapshot()}')
        return self.winner

    def random_decision(self) -> ActionType:
        """
        A simple strategy that returns a random action from the list of possible actions.
        """
        actions = self.get_valid_actions()
        if self.debuging_visual :
            with open('simulation_log.txt', 'a') as f:
                f.write(f"{actions}\n")
        return random.choice(actions)

    def player_decision(self) -> ActionType:
        actions = self.get_valid_actions()
        if len(actions) == 1 : return actions[0]
        for i, action in enumerate(actions,1) :
            print(f'{i} : {get_action_str(self, action)}')
        player_index = int(input('Enter the action index (1, 2, 3 ...) : '))
        return actions[player_index - 1]


    cdef void update_decision_player(self):
        """
        Chance phases have no decision player.  The defender decides its own
        defense steps, and the player who is not moving the ship places the
        squadrons that ship overlapped (RRG Overlapping).
        """

        # if self.phase in (Phase.SHIP_REVEAL_COMMAND_DIAL, Phase.ATTACK_ROLL_DICE):
        # Simplified
        if self.phase in (Phase.ATTACK_ROLL_DICE, Phase.DRAW_DAMAGE_CARD) :
            self.decision_player = 0
        elif self.phase in (Phase.ATTACK_SPEND_DEFENSE_TOKENS, Phase.CHOOSE_DEFEND_DICE, Phase.ATTACK_RESOLVE_DAMAGE, Phase.SHIP_PLACE_SQUAD):
            self.decision_player = -self.current_player
        else :
            self.decision_player = self.current_player

    cpdef list get_valid_actions(self) :
        """
        Returns a list of possible actions based on the current game phase.
        """
        cdef:
            list actions = []
            Ship active_ship, ship, defend_ship
            Squad active_squad, defend_squad
            AttackInfo attack_info
            int phase = self.phase
            int command
            bint dial, token
            int attack_hull, defend_hull

        if self.active_ship is not None : active_ship = self.active_ship
        if self.defend_ship is not None : defend_ship = self.defend_ship
        if self.active_squad is not None : active_squad = self.active_squad
        if self.attack_info is not None : attack_info = self.attack_info

        if phase == Phase.COMMAND_PHASE :
            # assign command to 'active_ship' (fixed ordering), no ship pointer.
            # A squadron dial can only ever activate friendly squadrons, so with
            # none left it is weakly dominated by every other command.
            for command in COMMANDS :
                if command == Command.SQUAD and not self.has_live_squad(active_ship.team) :
                    continue
                actions.append(('assign_command_action', (command, len(active_ship.command_stack) + 1)))

        elif phase == Phase.SHIP_ACTIVATE :
            for ship in self.get_valid_ship_activation(self.current_player) :
                actions.append(('activate_ship_action', ship.id))

        # Reveal Command Sequence
        elif phase == Phase.SHIP_REVEAL_COMMAND_DIAL :
            # no hidden information
            actions = [('reveal_command_action', active_ship.command_stack[0])]

        elif phase == Phase.SHIP_GAIN_COMMAND_TOKEN :
            revealed = active_ship.command_dial[0]
            if revealed not in active_ship.command_token :
                if len(active_ship.command_token) < active_ship.command_value :
                    actions.append(('gain_command_token_action', revealed))
                else :
                    for command in active_ship.command_token :
                        if command != revealed :
                            actions.append(('gain_and_discard_command_token_action', (revealed, command)))
            actions.append(PASS_COMMAND_TOKEN)

        # Squadron command: resolve the dial/token commitment, then activate.
        elif phase == Phase.SHIP_RESOLVE_SQUAD :
            actions = self.squad_command_resolve_actions()
            actions.append(PASS_SQUAD_COMMAND)

        # Engineering Sequence: resolve dial/token commitment first...
        elif phase == Phase.SHIP_RESOLVE_REPAIR :
            dial = Command.REPAIR in active_ship.command_dial
            token = Command.REPAIR in active_ship.command_token
            if dial and token:
                actions.append(('resolve_repair_command_action', (True, True)))
            if dial:
                actions.append(('resolve_repair_command_action', (True, False)))
            if token:
                actions.append(('resolve_repair_command_action', (False, True)))
            actions.append(PASS_REPAIR_COMMAND)

        # ...then spend the resulting engineering points.
        elif phase == Phase.SHIP_REPAIR_EFFECT :
            actions = self.repair_effect_actions()

        # Attack Sequence
        elif phase == Phase.SHIP_CHOOSE_TARGET_SHIP :
            actions = self.ship_target_actions(False)
            if not actions : actions = [PASS_ATTACK]

        elif phase == Phase.SHIP_DECLARE_TARGET :
            if self.defend_squad is not None :
                # Anti-squadron attack: the target is chosen, pick the hull zone.
                defend_squad = self.defend_squad
                for attack_hull in active_ship.get_valid_attack_hull():
                    if active_ship.is_valid_squad_target(attack_hull, defend_squad) :
                        actions.append(('declare_squad_target_action', attack_hull))
            else :
                for attack_hull in active_ship.get_valid_attack_hull():
                    for defend_hull in active_ship.get_valid_target_hull(attack_hull, defend_ship):
                        actions.append(('declare_target_action', (attack_hull, defend_hull)))

        elif phase == Phase.ATTACK_GATHER_DICE :
            # Obstruction removes exactly one die before rolling.  One-die
            # obstructed declarations are pruned by target validation, so
            # every offered removal leaves at least one die for training.
            if attack_info.obstructed:
                for dice_type in DICE :
                    if attack_info.dice_to_roll[dice_type] > 0:
                        actions.append(('gather_dice_action', DICE_CHOICE_1[dice_type]))
            else:
                actions = [GATHER_UNOBSTRUCTED]

        elif phase == Phase.ATTACK_ROLL_DICE or phase == Phase.DRAW_DAMAGE_CARD :
            raise NotImplementedError("This is a chance node. No player action available.")

        elif phase == Phase.ATTACK_RESOLVE_EFFECTS :
            actions = self.attack_effect_actions()
            actions.append(PASS_ATTACK_EFFECT) # Above actions are MUST USED actions

        elif phase == Phase.ATTACK_SPEND_DEFENSE_TOKENS :
            actions = self.defense_token_actions()
            actions.append(PASS_DEFENSE_TOKEN)

        elif phase == Phase.CHOOSE_DEFEND_DICE :
            for dice in dice_single_choice(attack_info.attack_pool_result) :
                actions.append(('spend_evade_dice_action', dice))

            if attack_info.is_attacker_ship :
                attack_ship = self.ships[attack_info.attack_ship_id]
                if defend_ship.size_class < attack_ship.size_class :
                    for dice_pair in dice_pair_choices(attack_info.attack_pool_result) :
                        actions.append(('discard_evade_dice_action', dice_pair))

        elif phase == Phase.ATTACK_RESOLVE_DAMAGE:
            actions = self.redirect_actions()
            actions.append(RESOLVE_NO_REDIRECT)

        elif phase == Phase.ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET :
            actions = self.additional_squad_target_actions()
            actions.append(PASS_ADDITIONAL_SQUAD_TARGET)

        # Maneuver Sequence: resolve dial/token commitment and determine
        # course are offered from the same phase, so a ship can maneuver
        # without ever resolving NAV.
        elif phase == Phase.SHIP_DETERMINE_COURSE:
            if Command.NAV not in active_ship.resolved_command :
                dial = Command.NAV in active_ship.command_dial
                token = Command.NAV in active_ship.command_token
                if dial and token:
                    actions.append(('resolve_nav_command_action', (True, True)))
                if dial:
                    actions.append(('resolve_nav_command_action', (True, False)))
                if token:
                    actions.append(('resolve_nav_command_action', (False, True)))

            for speed in active_ship.get_valid_speed():
                all_courses = active_ship.get_all_possible_courses(speed)

                for course in all_courses:
                    for placement in active_ship.get_valid_placement(course):
                        actions.append(('determine_course_action', (course, placement)))

        elif phase == Phase.SHIP_PLACE_SQUAD :
            actions = self.placement_actions()

        # === SQUADRON PHASE / squadron command ===
        elif phase == Phase.SQUAD_ACTIVATE :
            actions = self.squad_activation_actions()
            if self.active_ship is not None :
                # Up to the command's count (RRG Commands); the Squadron Phase
                # has no pass while squadrons remain (RRG Squadron Phase).
                actions.append(PASS_ACTIVATE_SQUAD)

        elif phase == Phase.SQUAD_MOVE :
            actions = self.squad_move_actions(active_squad)

        elif phase == Phase.SQUAD_DECLARE_TARGET :
            actions = self.squad_attack_target_actions(active_squad, False)
            if self.squad_attack_pass_offered() :
                actions.append(PASS_ATTACK_SQUAD)

        else :
            raise ValueError(f'Unknown game phase: {Phase(phase)}')

        if not actions:
            raise ValueError(f'No valid actions available in phase {Phase(phase)}\n{self.get_snapshot()}')

        return actions

    # --- Legal-action builders, shared by get_valid_actions() and the forced
    # single-action check so the two can never disagree. ---

    cdef list squad_command_resolve_actions(self) :
        """
        Resolve the squadron command from the dial, the token, or both
        (RRG Commands).  Offered only when some friendly squadron could be
        activated: with none in range, resolving produces no effect and the
        token version strictly loses the token.
        """
        cdef:
            list actions = []
            Ship active_ship = self.active_ship
            bint dial = Command.SQUAD in active_ship.command_dial
            bint token = Command.SQUAD in active_ship.command_token
        if not (dial or token) or not self.command_eligible_squads(active_ship) :
            return actions
        if dial and token:
            actions.append(('resolve_squad_command_action', (True, True)))
        if dial:
            actions.append(('resolve_squad_command_action', (True, False)))
        if token:
            actions.append(('resolve_squad_command_action', (False, True)))
        return actions

    cdef list repair_effect_actions(self) :
        cdef:
            list actions = []
            Ship active_ship = self.active_ship
            int hull, from_hull, to_hull, card_index
            bint repair_hull_legal, recover_shield_legal, forced

        # Repair Hull discards one damage card of the ship's choice: any
        # faceup type it holds, or an anonymous facedown card.  hull <
        # max_hull guarantees at least one card exists, so this always
        # offers at least one action when it is legal at all.
        repair_hull_legal = active_ship.engineer_budget >= 3 and active_ship.hull < active_ship.max_hull
        if repair_hull_legal :
            for card_index in range(DAMAGE_CARD_TYPES) :
                if active_ship.damage_cards[card_index] > 0 :
                    actions.append(('repair_hull_action', card_index))
            if active_ship.facedown_damage_count() > 0 :
                actions.append(('repair_hull_action', FACEDOWN_CARD_INDEX))

        recover_shield_legal = False
        if active_ship.engineer_budget >= 2 :
            for hull in HULL_SECTIONS :
                if active_ship.shield[hull] < active_ship.max_shield[hull] and \
                        not hull in active_ship.reduced_hull :
                    actions.append(('recover_shield_action', hull))
                    recover_shield_legal = True

        forced = repair_hull_legal or recover_shield_legal
        if not forced :
            actions.append(PASS_REPAIR)

        if active_ship.engineer_budget >= 1 :
            for from_hull in HULL_SECTIONS:
                for to_hull in HULL_SECTIONS:
                    if active_ship.shield[to_hull] < active_ship.max_shield[to_hull] and active_ship.shield[from_hull] > 0 and \
                            from_hull != to_hull and \
                            not from_hull in active_ship.repaired_hull and \
                            not to_hull in active_ship.reduced_hull:
                        actions.append(('move_shield_action', (from_hull, to_hull)))
        return actions

    cdef list ship_target_actions(self, bint stop_at_first) :
        """
        Every target the active ship could attack from an unused hull zone:
        enemy ships (a ship pointer) and enemy squadrons.  One action per
        target, not per (hull, target): the payload is only the target, so a
        target reachable from two hull zones would otherwise be offered twice.
        The attacking hull is chosen in SHIP_DECLARE_TARGET.
        """
        cdef:
            list actions = []
            Ship active_ship = self.active_ship
            Ship defend_ship
            Squad squad
            set seen_target_ids = set()
            int attack_hull
            list attack_hulls = active_ship.get_valid_attack_hull()

        for attack_hull in attack_hulls:
            for defend_ship in active_ship.get_valid_ship_target(attack_hull):
                if defend_ship.id in seen_target_ids : continue
                seen_target_ids.add(defend_ship.id)
                actions.append(('choose_target_ship_action', defend_ship.id))
                if stop_at_first : return actions

        seen_target_ids = set()
        for squad in self.squads :
            if squad.team == active_ship.team or squad.destroyed : continue
            for attack_hull in attack_hulls :
                if active_ship.is_valid_squad_target(attack_hull, squad) :
                    actions.append(('choose_target_squad_action', squad.id))
                    if stop_at_first : return actions
                    break
        return actions

    cdef list attack_effect_actions(self) :
        cdef:
            list actions = []
            AttackInfo attack_info = self.attack_info
            Ship attack_ship, defender
            DefenseToken defense_token
            int index, blue_acc_count, red_acc_count
            bint dial, token

        # Spend accuracy: only ships carry defense tokens here.
        if attack_info.is_defender_ship :
            defender = self.ships[attack_info.defend_ship_id]
            blue_acc_count = attack_info.attack_pool_result[Dice.BLUE][ACCURACY_INDEX[Dice.BLUE]]
            red_acc_count = attack_info.attack_pool_result[Dice.RED][ACCURACY_INDEX[Dice.RED]]
            if blue_acc_count or red_acc_count :
                for index, defense_token in enumerate(defender.defense_tokens):
                    if defense_token.discarded or defense_token.accuracy :
                        continue
                    actions.append(('spend_accuracy_action', index))

        # con-fire command
        if attack_info.is_attacker_ship :
            attack_ship = self.ships[attack_info.attack_ship_id]
            # resolve : commit dial and/or token. Once per activation, and before
            # either granted effect is used (no incremental top-up afterwards).
            if Command.CONFIRE not in attack_ship.resolved_command :
                dial = Command.CONFIRE in attack_ship.command_dial
                token = Command.CONFIRE in attack_ship.command_token
                if dial and token:
                    actions.append(('resolve_confire_command_action', (True, True)))
                if dial:
                    actions.append(('resolve_confire_command_action', (True, False)))
                if token:
                    actions.append(('resolve_confire_command_action', (False, True)))

            # use the granted effects, legal any time before pass_attack_effect
            if attack_info.confire_dial : # add a die of a color already in the pool
                for dice in DICE:
                    if sum(attack_info.attack_pool_result[dice]) :
                        actions.append(('use_confire_dial_action', DICE_CHOICE_1[dice]))

            if attack_info.confire_token :
                for dice in dice_single_choice(attack_info.attack_pool_result) :
                    actions.append(('use_confire_token_action', dice))

        # Swarm: reroll 1 die (RRG Squadron Keywords).  The condition is fixed
        # when the attack is declared; AttackInfo.swarm is cleared once used.
        if attack_info.swarm :
            for dice in dice_single_choice(attack_info.attack_pool_result) :
                actions.append(('swarm_reroll_action', dice))
        return actions

    cdef list defense_token_actions(self) :
        cdef:
            list actions = []
            AttackInfo attack_info = self.attack_info
            Ship defender
            DefenseToken defense_token
            bint pool_has_dice = False

        # Only ships carry defense tokens here (no squadron in the data is
        # unique, and only unique squadrons have them).
        if not attack_info.is_defender_ship :
            return actions
        defender = self.ships[attack_info.defend_ship_id]

        for color in attack_info.attack_pool_result :
            if sum(color) :
                pool_has_dice = True
                break

        if defender.speed > 0 :
            for defense_token in defender.defense_tokens:
                if (not defense_token.discarded and not defense_token.accuracy
                        and defense_token.id not in attack_info.spent_token_indices
                        and defense_token.type not in attack_info.spent_token_types
                        and (defense_token.type != TokenType.EVADE or pool_has_dice)):

                    actions.append(('spend_defense_token_action', defense_token.id))
        return actions

    cdef list redirect_actions(self) :
        cdef:
            list actions = []
            AttackInfo attack_info = self.attack_info
            Ship defend_ship
            int defend_hull, idx_left, idx_right, hull, damage, total_damage

        if TokenType.REDIRECT in attack_info.spent_token_types :
            defend_ship = self.defend_ship
            total_damage = attack_info.total_damage
            defend_hull = attack_info.defend_hull
            idx_left = (defend_hull - 1) & 3
            idx_right = (defend_hull + 1) & 3
            for hull in (idx_left, idx_right) :
                for damage in range(1, min(total_damage, defend_ship.shield[hull]) + 1):
                    actions.append(('resolve_damage_action', (hull, damage)))
        return actions

    cdef list additional_squad_target_actions(self) :
        """
        After attacking a squadron, the ship may declare another enemy squadron
        in the same hull zone's arc and range, each squadron once per attack
        (RRG Attack, Declare Additional Squadron Target).
        """
        cdef:
            list actions = []
            AttackInfo attack_info = self.attack_info
            Ship attack_ship = self.ships[attack_info.attack_ship_id]
            int attack_hull = attack_info.attack_hull
            Squad squad
        for squad in self.squads :
            if squad.id in attack_info.squadron_target : continue
            if attack_ship.is_valid_squad_target(attack_hull, squad) :
                actions.append(('declare_additional_squad_target_action', squad.id))
        return actions

    cdef list squad_activation_actions(self) :
        """
        Which squadron activates, and which step it takes first (RRG Squadron
        Activation).  Squadron Phase: one step only.  Squadron command: both
        steps, in either order.  An engaged squadron cannot take the move step
        unless every squadron engaging it has heavy; one with no legal target
        cannot take the attack step.  Every candidate keeps at least one option:
        a squadron that cannot move is engaged, so the squadron engaging it is
        an unobstructed target at distance 1.
        """
        cdef:
            list actions = []
            list candidates
            Squad squad
        if self.active_ship is not None :
            candidates = self.command_eligible_squads(self.active_ship)
        else :
            candidates = self.get_valid_squad_activation(self.current_player)
        for squad in candidates :
            if not self.squad_movement_restricted(squad) :
                actions.append(('activate_squad_move_action', squad.id))
            if self.squad_attack_target_actions(squad, True) :
                actions.append(('activate_squad_attack_action', squad.id))
        return actions

    cdef list squad_attack_target_actions(self, Squad squad, bint stop_at_first) :
        """
        Targets of a (non-counter) squadron attack, all at distance 1 (RRG
        Attack).  Any enemy squadron at distance 1 can be attacked, engaged or
        not (RRG FAQ).  Engaged with an escort, only squadrons with escort
        (RRG Squadron Keywords).  Engaged with any squadron that lacks heavy, no
        ship (RRG Engagement, Heavy).
        """
        cdef:
            list actions = []
            Ship ship
            Rect base
            int index, hull
            long long in_range, engaged, targets
            double x, y
            list in_range_list

        self.ensure_squad_row(squad.id)
        in_range = <long long>self.squad_in_range[squad.id]
        engaged = <long long>self.squad_engaged[squad.id]

        targets = in_range
        if engaged & self.escort_mask :
            targets &= self.escort_mask
        if squad.anti_squad_dice == 0 :
            targets = 0
        elif squad.anti_squad_dice == 1 :
            # Simplified training policy, as for ships: obstruction would remove
            # the only die.  Among squadrons at distance 1, unobstructed is
            # exactly "engaged" (G2 + line of sight).
            targets &= engaged
        index = 0
        while targets :
            if targets & 1 :
                actions.append(('squad_attack_squad_action', index))
                if stop_at_first : return actions
            targets >>= 1
            index += 1

        if engaged & ~self.heavy_mask : return actions
        if squad.battery_dice == 0 : return actions

        x = squad.coords[0]
        y = squad.coords[1]
        for ship in self.ships :
            if ship.team == squad.team or ship.destroyed : continue
            # Exact pre-reject (G3): hull zones lie inside the base, so a base
            # beyond distance 1 puts every hull zone beyond it.
            base = self.ship_base_rect(ship.id)
            if rect_dist_sq(&base, x, y) > Q2S * Q2S : continue
            in_range_list = cache.attack_range_q2s(squad.get_squad_hash_state(), ship.get_ship_hash_state())
            for hull in HULL_SECTIONS :
                if not in_range_list[hull] : continue
                if squad.battery_dice == 1 and squad.is_obstruct_q2s(ship, hull) : continue
                actions.append(('squad_attack_ship_action', (ship.id, hull)))
                if stop_at_first : return actions
        return actions

    cdef bint squad_attack_pass_offered(self) :
        """
        Declining the attack is offered only where it reaches a state no other
        choice reaches: after the move step of a squadron command, or when the
        squadron cannot take the move step at all.  Otherwise "activate and do
        nothing" (RRG Squadron Activation) is already the move step's
        remain-in-place cell, and a pass would only duplicate it.
        """
        cdef Squad squad = self.active_squad
        if self.active_ship is not None and not self.squad_can_move :
            return True
        return self.squad_movement_restricted(squad)

    cdef tuple forced_pass_action(self) :
        """
        The single legal action of the current step when that action is a
        pass-like no-op, else None.  Each check is the cheap form of the phase's
        legal-action builder; `verify_forced` cross-checks them against
        get_valid_actions() in testing.
        """
        cdef:
            int phase = self.phase
            Ship active_ship = self.active_ship
            AttackInfo attack_info = self.attack_info
            tuple forced = None
            list actions

        if phase == Phase.SHIP_GAIN_COMMAND_TOKEN :
            if active_ship.command_dial[0] in active_ship.command_token : forced = PASS_COMMAND_TOKEN
        elif phase == Phase.SHIP_RESOLVE_SQUAD :
            if not self.squad_command_resolve_actions() : forced = PASS_SQUAD_COMMAND
        elif phase == Phase.SHIP_RESOLVE_REPAIR :
            if Command.REPAIR not in active_ship.command_dial and Command.REPAIR not in active_ship.command_token :
                forced = PASS_REPAIR_COMMAND
        elif phase == Phase.SHIP_REPAIR_EFFECT :
            actions = self.repair_effect_actions()
            if len(actions) == 1 and actions[0] == PASS_REPAIR : forced = PASS_REPAIR
        elif phase == Phase.SHIP_CHOOSE_TARGET_SHIP :
            if not self.ship_target_actions(True) : forced = PASS_ATTACK
        elif phase == Phase.ATTACK_GATHER_DICE :
            if not attack_info.obstructed : forced = GATHER_UNOBSTRUCTED
        elif phase == Phase.ATTACK_RESOLVE_EFFECTS :
            if not self.attack_effect_actions() : forced = PASS_ATTACK_EFFECT
        elif phase == Phase.ATTACK_SPEND_DEFENSE_TOKENS :
            if not self.defense_token_actions() : forced = PASS_DEFENSE_TOKEN
        elif phase == Phase.ATTACK_RESOLVE_DAMAGE :
            if not self.redirect_actions() : forced = RESOLVE_NO_REDIRECT
        elif phase == Phase.ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET :
            if not self.additional_squad_target_actions() : forced = PASS_ADDITIONAL_SQUAD_TARGET
        elif phase == Phase.SQUAD_ACTIVATE :
            # Squadron command only: every eligible squadron has an option (see
            # squad_activation_actions), so the pass is forced exactly when no
            # squadron is eligible.  The Squadron Phase never offers a pass.
            if active_ship is not None and not self.command_eligible_squads(active_ship) :
                forced = PASS_ACTIVATE_SQUAD
        elif phase == Phase.SQUAD_DECLARE_TARGET :
            if self.squad_attack_pass_offered() and \
                    not self.squad_attack_target_actions(self.active_squad, True) :
                forced = PASS_ATTACK_SQUAD

        if self.verify_forced :
            actions = self.get_valid_actions()
            if forced is None :
                if len(actions) == 1 and actions[0] in FORCEABLE_ACTIONS :
                    raise AssertionError(f'Missed forced step {actions[0]} in {Phase(phase)}')
            elif actions != [forced] :
                raise AssertionError(f'{forced} forced in {Phase(phase)} but legal actions are {actions}')
        return forced


    cpdef void apply_action(self, tuple action) :
        """
        Applies the given action to the game state.
        """
        cdef:
            Ship active_ship, command_ship, attack_ship, defend_ship, ship, next_ship, other_ship
            Squad active_squad, defend_squad, squad
            AttackInfo attack_info
            DefenseToken defense_token
            Obstacle obstacle
            str action_type
            object action_data
            bint dial, token
            list dice_pool = []
            list dice_result_color = []
            list pending
            int card_index, cards_dealt, asteroid_count, source
            bint has_crit
            tuple forced
            Rect moving_base

        if self.active_ship is not None : active_ship = self.active_ship
        if self.defend_ship is not None : defend_ship = self.defend_ship
        if self.active_squad is not None : active_squad = self.active_squad
        if self.attack_info is not None : attack_info = self.attack_info

        self.visualize_action(action)
        action_type = action[0]
        action_data = action[1]

        if action_type == 'assign_command_action':
            (command, round) = action_data
            active_ship.asign_command(command)

            next_ship = self.get_next_command_ship(self.current_player)
            if next_ship is not None:
                self.active_ship = next_ship
            else:
                other_ship = self.get_next_command_ship(-self.current_player)
                if other_ship is not None:
                    self.current_player *= -1
                    self.active_ship = other_ship
                else:
                    self.active_ship = None
                    self.current_player = self.first_player
                    self.phase = Phase.SHIP_ACTIVATE


        elif action_type == 'activate_ship_action':
            ship_id = action_data
            self.active_ship = self.ships[ship_id]
            self.current_player = self.active_ship.team
            self.phase = Phase.SHIP_REVEAL_COMMAND_DIAL


        elif action_type == 'reveal_command_action':
            command = action_data
            active_ship.gain_command_dial(command)
            active_ship.command_stack = active_ship.command_stack[1:]
            self.phase = Phase.SHIP_GAIN_COMMAND_TOKEN

        elif action_type == 'gain_command_token_action':
            command = action_data

            active_ship.discard_command_dial(command)
            active_ship.gain_command_token(command)
            self.phase = Phase.SHIP_RESOLVE_SQUAD

        elif action_type == 'gain_and_discard_command_token_action':
            (gain_command, discard_command) = action_data

            active_ship.discard_command_dial(gain_command)
            active_ship.discard_command_token(discard_command)
            active_ship.gain_command_token(gain_command)
            self.phase = Phase.SHIP_RESOLVE_SQUAD

        elif action_type == 'pass_command_token':
            _ = action_data
            self.phase = Phase.SHIP_RESOLVE_SQUAD

        # Squadron and Repair both "resolve after revealing the ship's command
        # dial" (RRG Commands).  The engine fixes squadron first; the two never
        # interact, so the order cannot change the outcome.
        elif action_type == 'resolve_squad_command_action':
            (dial, token) = action_data
            if dial : active_ship.discard_command_dial(Command.SQUAD)
            if token : active_ship.discard_command_token(Command.SQUAD)
            active_ship.resolved_command += (Command.SQUAD,)
            # Dial: up to the ship's squadron value; token: one (RRG Commands).
            self.squad_activation_count = dial * active_ship.squad_value + token
            self.phase = Phase.SQUAD_ACTIVATE

        elif action_type == 'pass_squad_command':
            _ = action_data
            self.phase = Phase.SHIP_RESOLVE_REPAIR

        elif action_type == 'resolve_repair_command_action':
            (dial, token) = action_data
            if dial : active_ship.discard_command_dial(Command.REPAIR)
            if token : active_ship.discard_command_token(Command.REPAIR)
            active_ship.resolved_command += (Command.REPAIR,)
            active_ship.engineer_budget = dial * active_ship.engineer_value + token * (active_ship.engineer_value + 1) // 2
            self.phase = Phase.SHIP_REPAIR_EFFECT

        elif action_type == 'pass_repair_command':
            _ = action_data
            self.phase = Phase.SHIP_REPAIR_EFFECT

        elif action_type == 'repair_hull_action':
            card_index = action_data
            active_ship.engineer_budget -= 3
            active_ship.engineer_spent += 3
            active_ship.discard_damage_card(card_index)

        elif action_type == 'recover_shield_action':
            hull= action_data
            active_ship.engineer_budget -= 2
            active_ship.engineer_spent += 2

            shield_list = list(active_ship.shield)
            shield_list[hull] += 1
            active_ship.shield = tuple(shield_list)

            active_ship.repaired_hull += (hull,)


        elif action_type == 'move_shield_action':
            (from_hull, to_hull) = action_data
            active_ship.engineer_budget -= 1
            active_ship.engineer_spent += 1

            shield_list = list(active_ship.shield)
            shield_list[from_hull] -= 1
            shield_list[to_hull] += 1
            active_ship.shield = tuple(shield_list)

            active_ship.repaired_hull += (to_hull,)
            active_ship.reduced_hull += (from_hull,)

        elif action_type == 'pass_repair':
            _ = action_data
            active_ship.repaired_hull = ()
            active_ship.reduced_hull = ()
            active_ship.engineer_budget = 0
            active_ship.engineer_spent = 0
            self.phase = Phase.SHIP_CHOOSE_TARGET_SHIP


        elif action_type == 'choose_target_ship_action':
            defend_ship_id = action_data
            self.defend_ship = self.ships[defend_ship_id]
            self.phase = Phase.SHIP_DECLARE_TARGET

        elif action_type == 'choose_target_squad_action':
            self.defend_squad = self.squads[<int>action_data]
            self.phase = Phase.SHIP_DECLARE_TARGET

        elif action_type == 'declare_target_action':
            (attack_hull, defend_hull) = action_data
            defender = (defend_ship, defend_hull)
            # gather initial dice pool here
            self.attack_info = AttackInfo((active_ship, attack_hull), defender)

            self.phase = Phase.ATTACK_GATHER_DICE

        elif action_type == 'declare_squad_target_action':
            attack_hull = action_data
            self.attack_info = AttackInfo((active_ship, attack_hull), self.defend_squad)
            self.phase = Phase.ATTACK_GATHER_DICE

        elif action_type == 'gather_dice_action':
            dice_to_remove = action_data

            if attack_info.obstructed:
                if sum(dice_to_remove) != 1:
                    raise ValueError('An obstructed attack must remove exactly one gathered die')
            elif any(dice_to_remove):
                raise ValueError('An unobstructed attack cannot remove a gathered die')

            # Remove the chosen color from the gathered pool.
            for dice_type in DICE:
                if dice_to_remove[dice_type] < 0 or dice_to_remove[dice_type] > attack_info.dice_to_roll[dice_type]:
                    raise ValueError('Cannot remove a die that is not in the gathered attack pool')
                dice_pool.append(attack_info.dice_to_roll[dice_type] - dice_to_remove[dice_type])
            attack_info.dice_to_roll = tuple(dice_pool)

            self.phase = Phase.ATTACK_ROLL_DICE

        elif action_type == 'roll_dice_action':
            dice_roll= action_data
            attack_info.dice_to_roll = (0,0,0)
            for dice_type in DICE:
                dice_result_color = []
                for original, new in zip(attack_info.attack_pool_result[dice_type], dice_roll[dice_type]):
                    dice_result_color.append(original + new)
                dice_pool.append(tuple(dice_result_color))
            attack_info.attack_pool_result = tuple(dice_pool)

            attack_info.calculate_total_damage()
            self.phase = attack_info.phase # either ATTACK_RESOLVE_EFFECTS or ATTACK_SPEND_DEFENSE_TOKENS

        elif action_type == 'spend_accuracy_action':
            index= action_data
            defense_token = defend_ship.defense_tokens[index]
            defense_token.accuracy = True
            if attack_info.attack_pool_result[Dice.RED][ACCURACY_INDEX[Dice.RED]]: accuracy_dice = Dice.RED
            else: accuracy_dice = Dice.BLUE

            dice_pool = list(EMPTY_DICE_POOL)
            dice_pool[accuracy_dice] = ACCURACY_DICE_1[accuracy_dice]

            attack_info.remove_dice(tuple(dice_pool))

            attack_info.calculate_total_damage()

        elif action_type == 'resolve_confire_command_action':
            (dial, token) = action_data
            if dial : active_ship.discard_command_dial(Command.CONFIRE)
            if token : active_ship.discard_command_token(Command.CONFIRE)
            active_ship.resolved_command += (Command.CONFIRE,)
            attack_info.confire_dial, attack_info.confire_token = dial, token

        elif action_type == 'use_confire_dial_action':
            dice_to_add = action_data
            attack_info.dice_to_roll = dice_to_add
            attack_info.confire_dial = False
            self.phase = Phase.ATTACK_ROLL_DICE

        elif action_type == 'use_confire_token_action':
            reroll_dice = action_data
            attack_info.confire_token = False

            attack_info.remove_dice(reroll_dice)

            for dice_type in DICE:
                dice_pool.append(sum(reroll_dice[dice_type]))
            attack_info.dice_to_roll = tuple(dice_pool)
            self.phase = Phase.ATTACK_ROLL_DICE

        elif action_type == 'swarm_reroll_action':
            reroll_dice = action_data
            attack_info.swarm = False
            attack_info.remove_dice(reroll_dice)
            for dice_type in DICE:
                dice_pool.append(sum(reroll_dice[dice_type]))
            attack_info.dice_to_roll = tuple(dice_pool)
            self.phase = Phase.ATTACK_ROLL_DICE

        elif action_type == 'pass_attack_effect':
            _ = action_data
            attack_info.phase = Phase.ATTACK_SPEND_DEFENSE_TOKENS
            self.phase = Phase.ATTACK_SPEND_DEFENSE_TOKENS

        elif action_type == 'spend_defense_token_action':
            index = action_data
            defense_token = defend_ship.defense_tokens[index]

            attack_info.spent_token_indices += (index,)
            attack_info.spent_token_types += (defense_token.type,)
            defense_token.spend()
            attack_info.calculate_total_damage()

            if defense_token.type == TokenType.EVADE:
                self.phase = Phase.CHOOSE_DEFEND_DICE

        elif action_type == 'spend_evade_dice_action':
            evade_dice = action_data
            attack_info.remove_dice(evade_dice)

            if AttackRange.CLOSE <= attack_info.attack_range <= AttackRange.MEDIUM:
                # Reroll Evade Dice
                for dice_type in DICE:
                    dice_pool.append(sum(evade_dice[dice_type]))
                attack_info.dice_to_roll = tuple(dice_pool)
                self.phase = Phase.ATTACK_ROLL_DICE
            else:
                self.phase = Phase.ATTACK_SPEND_DEFENSE_TOKENS

        elif action_type == 'discard_evade_dice_action':
            (dice_a, dice_b) = action_data
            defense_token = defend_ship.defense_tokens[attack_info.spent_token_indices[-1]]
            defense_token.discard()

            combined_dice = sum_dice_pools(dice_a, dice_b)
            attack_info.remove_dice(combined_dice)

            if AttackRange.CLOSE <= attack_info.attack_range <= AttackRange.MEDIUM:
                # Reroll both Evade Dice
                for dice_type in DICE:
                    dice_pool.append(sum(combined_dice[dice_type]))
                attack_info.dice_to_roll = tuple(dice_pool)
                self.phase = Phase.ATTACK_ROLL_DICE
            else:
                self.phase = Phase.ATTACK_SPEND_DEFENSE_TOKENS

        elif action_type == 'pass_defense_token':
            _ = action_data
            self.phase = Phase.ATTACK_RESOLVE_DAMAGE

        elif action_type == 'resolve_damage_action':
            redirect_damage= action_data
            total_damage = attack_info.total_damage

            if attack_info.is_defender_ship :
                # Redirect
                if redirect_damage :
                    hull, damage = redirect_damage
                    shield_list = list(defend_ship.shield)
                    shield_list[hull] -= damage
                    defend_ship.shield = tuple(shield_list)
                    total_damage -= damage

                # Standard critical effect: "If the defender is dealt at least one
                # damage card by this attack, deal the first damage card faceup."
                # It adds no damage - it only turns one of the cards this attack
                # already deals faceup, so count them before defend() applies them.
                # Squadrons resolve critical effects only through bomber, and
                # never suffer them (RRG Critical Effects, Squadron Keywords).
                cards_dealt = total_damage - min(total_damage, defend_ship.shield[<int>attack_info.defend_hull])
                has_crit = <bint>False
                if attack_info.is_attacker_ship or attack_info.bomber :
                    for dice_type in DICE :
                        if attack_info.attack_pool_result[dice_type][CRIT_INDEX[dice_type]] :
                            has_crit = <bint>True
                            break

                defend_ship.defend(<int>attack_info.defend_hull, total_damage)
                for defense_token in defend_ship.defense_tokens :
                    defense_token.accuracy = False

                if has_crit and cards_dealt > 0 and not defend_ship.destroyed :
                    # attack_info / defend_ship stay set: after_attack_damage()
                    # needs them once the chance node resolves.
                    self.damage_draw_ship_id = defend_ship.id
                    self.damage_draw_count = 1
                    self.damage_draw_source = DAMAGE_SOURCE_ATTACK
                    self.phase = Phase.DRAW_DAMAGE_CARD
                else :
                    self.after_attack_damage()
            else :
                # A squadron loses hull points; it has no shields and is never
                # dealt damage cards (RRG Damage).
                defend_squad = self.squads[attack_info.defend_squad_id]
                defend_squad.defend(total_damage)
                self.after_attack_damage()

        elif action_type == 'draw_damage_card_action':
            card_index = action_data
            if self.damage_draw_count <= 0 :
                raise ValueError('No pending damage card draw')
            ship = self.ships[self.damage_draw_ship_id]

            if self.damage_draw_source == DAMAGE_SOURCE_ATTACK :
                # The card was already dealt by defend(); only turn it faceup.
                ship.flip_damage_card_faceup(card_index)
            else :
                # Asteroid overlap deals a brand new faceup card.
                ship.deal_faceup_damage_card(card_index)

            self.damage_draw_count -= 1

            if self.damage_draw_count > 0 and not ship.destroyed :
                pass # stay in Phase.DRAW_DAMAGE_CARD for the next draw
            else :
                source = self.damage_draw_source
                self.damage_draw_ship_id = -1
                self.damage_draw_count = 0
                self.damage_draw_source = DAMAGE_SOURCE_NONE

                if source == DAMAGE_SOURCE_ATTACK :
                    self.after_attack_damage()
                else :
                    self.end_ship_activation()

        elif action_type == 'declare_additional_squad_target_action':
            defend_squad = self.squads[<int>action_data]
            attack_ship = self.ships[attack_info.attack_ship_id]
            attack_info.declare_additional_squad_target((attack_ship, attack_info.attack_hull), defend_squad)
            self.defend_squad = defend_squad
            self.phase = Phase.ATTACK_GATHER_DICE

        elif action_type == 'pass_additional_squad_target':
            _ = action_data
            self.finish_ship_attack()

        elif action_type == 'pass_attack':
            _= action_data
            self.phase = Phase.SHIP_DETERMINE_COURSE

        elif action_type == 'resolve_nav_command_action':
            (dial, token) = action_data
            if dial : active_ship.discard_command_dial(Command.NAV)
            if token : active_ship.discard_command_token(Command.NAV)
            active_ship.resolved_command += (Command.NAV,)
            active_ship.nav_dial_granted = dial
            active_ship.nav_token_granted = token

        elif action_type == 'determine_course_action':
            (course, placement)= action_data
            active_ship.nav_dial_granted = False
            active_ship.nav_token_granted = False

            active_ship.speed = len(course)
            active_ship.execute_maneuver(course, placement)

            # Obstacle overlap: only the asteroid effect is live.  It deals one
            # faceup damage card per overlapped asteroid, which needs the
            # DRAW_DAMAGE_CARD chance node, so it cannot go through
            # Ship.overlap_obstacle().  Station and debris stay disabled.
            asteroid_count = 0
            pending = []
            if not active_ship.destroyed :
                for obstacle in self.obstacles:
                    if obstacle.type == ObstacleType.ASTEROID and active_ship.check_overlap(obstacle) :
                        asteroid_count += 1

                # Squadrons under the ship's final position are moved aside and
                # placed touching it by the opponent once the maneuver is
                # finished (RRG Overlapping).  Touching is not overlapping.
                moving_base = self.ship_base_rect(active_ship.id)
                for squad in self.squads :
                    if not squad.destroyed and \
                            rect_dist_sq(&moving_base, squad.coords[0], squad.coords[1]) < SQUAD_SHIP_OVERLAP_SQ :
                        pending.append(squad.id)

            # Placement first, then the asteroid draws: the ship must still be
            # present for squadrons to be placed touching it.
            if asteroid_count :
                self.damage_draw_ship_id = active_ship.id
                self.damage_draw_count = asteroid_count
                self.damage_draw_source = DAMAGE_SOURCE_OBSTACLE
            self.pending_placement = tuple(pending)
            self.resolve_pending_placements()

        elif action_type == 'place_squad_action':
            squad = self.squads[<int>self.pending_placement[0]]
            point = cache._ship_coordinate(active_ship.get_ship_hash_state())['squad_placement_points'][<int>action_data]
            squad.place(<double>point[0], <double>point[1])
            self.pending_placement = self.pending_placement[1:]
            self.resolve_pending_placements()

        # === SQUADRON ACTIVATION ===
        elif action_type == 'activate_squad_move_action' or action_type == 'activate_squad_attack_action':
            active_squad = self.squads[<int>action_data]
            self.active_squad = active_squad
            self.squad_activation_count -= 1
            # Squadron Phase: move OR attack.  Squadron command: both, in
            # either order (RRG Squadron Activation).
            if action_type == 'activate_squad_move_action' :
                self.squad_can_move = True
                self.squad_can_attack = self.active_ship is not None
                self.phase = Phase.SQUAD_MOVE
            else :
                self.squad_can_move = self.active_ship is not None
                self.squad_can_attack = True
                self.phase = Phase.SQUAD_DECLARE_TARGET

        elif action_type == 'pass_activate_squad':
            _ = action_data
            self.squad_activation_count = 0
            self.phase = Phase.SHIP_RESOLVE_REPAIR

        elif action_type == 'move_squad_action':
            (di, dj) = action_data
            active_squad.move_by_cells(di, dj)
            self.squad_can_move = False
            if self.squad_can_attack : self.phase = Phase.SQUAD_DECLARE_TARGET
            else : self.end_squad_activation()

        elif action_type == 'squad_attack_squad_action':
            defend_squad = self.squads[<int>action_data]
            self.attack_info = AttackInfo(active_squad, defend_squad)
            self.phase = Phase.ATTACK_GATHER_DICE

        elif action_type == 'squad_attack_ship_action':
            (defend_ship_id, defend_hull) = action_data
            self.defend_ship = self.ships[defend_ship_id]
            self.attack_info = AttackInfo(active_squad, (self.defend_ship, defend_hull))
            self.phase = Phase.ATTACK_GATHER_DICE

        elif action_type == 'pass_attack_squad':
            _ = action_data
            self.squad_can_attack = False
            if self.squad_can_move and not self.squad_movement_restricted(active_squad) :
                self.phase = Phase.SQUAD_MOVE
            else :
                self.end_squad_activation()

        else:
            raise ValueError(f'Unknown Action {action}')

        # decision player for NEXT PHASE (after applying action)
        self.update_decision_player()

        # A fleet with no ships left ends the game at once (RRG Winning and
        # Losing), before anything else resolves.
        if self.fleet_check_pending :
            self.fleet_check_pending = False
            self.check_fleet_elimination()

        # Resolve a forced pass-like step inline; it recurses through any chain.
        if self.winner == 0.0 and self.decision_player != 0 :
            forced = self.forced_pass_action()
            if forced is not None :
                self.apply_action(forced)

    cdef void after_attack_damage(self) :
        """
        Tail of the Resolve Damage step, reached directly or after the standard
        critical's faceup card is drawn.  A ship that attacked a squadron may
        declare another squadron target (RRG Attack); otherwise the attack ends.
        """
        cdef AttackInfo attack_info = self.attack_info
        if attack_info.is_attacker_ship :
            if attack_info.is_defender_ship :
                self.finish_ship_attack()
            else :
                self.phase = Phase.ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET
        else :
            self.finish_squad_attack()

    cdef void finish_ship_attack(self) :
        """
        Clear the attack context and hand the attacking ship its next step.
        """
        cdef Ship active_ship = self.active_ship

        self.attack_info = None
        self.defend_ship = None
        self.defend_squad = None

        if active_ship.attack_count < 2 : self.phase = Phase.SHIP_CHOOSE_TARGET_SHIP
        else : self.phase = Phase.SHIP_DETERMINE_COURSE

    cdef void finish_squad_attack(self) :
        """
        Clear the attack context.  Under a squadron command the squadron may
        still move, unless engaged by a squadron that lacks heavy - and the
        attack may have just destroyed the last one engaging it (RRG Engagement).
        """
        self.attack_info = None
        self.defend_ship = None
        self.squad_can_attack = False
        if self.squad_can_move and not self.squad_movement_restricted(self.active_squad) :
            self.phase = Phase.SQUAD_MOVE
        else :
            self.end_squad_activation()

    cdef void end_squad_activation(self) :
        """
        Toggle the squadron's activation slider and pass to the next one: within
        the squadron command, back to the ship's activation, or to the next
        Squadron Phase turn.
        """
        cdef Squad squad = self.active_squad
        squad.activated = True
        self.active_squad = None
        self.squad_can_move = False
        self.squad_can_attack = False

        if self.active_ship is not None :
            if self.squad_activation_count > 0 : self.phase = Phase.SQUAD_ACTIVATE
            else : self.phase = Phase.SHIP_RESOLVE_REPAIR
        else :
            self.next_squadron_turn()

    cdef void next_squadron_turn(self) :
        """
        Players activate two squadrons per turn, one at a time, and a player
        with none left passes for the rest of the phase (RRG Squadron Phase).
        """
        if self.squad_activation_count > 0 and self.get_valid_squad_activation(self.current_player) :
            self.phase = Phase.SQUAD_ACTIVATE
        elif self.get_valid_squad_activation(-self.current_player) :
            self.current_player *= -1
            self.squad_activation_count = 2
            self.phase = Phase.SQUAD_ACTIVATE
        elif self.get_valid_squad_activation(self.current_player) :
            self.squad_activation_count = 2
            self.phase = Phase.SQUAD_ACTIVATE
        else :
            self.squad_activation_count = 0
            self.status_phase()

    cdef void start_squadron_phase(self) :
        """The first player activates first (RRG Squadron Phase)."""
        self.active_ship = None
        if self.get_valid_squad_activation(self.first_player) :
            self.current_player = self.first_player
        elif self.get_valid_squad_activation(-self.first_player) :
            self.current_player = -self.first_player
        else :
            self.status_phase()
            return
        self.squad_activation_count = 2
        self.phase = Phase.SQUAD_ACTIVATE

    cdef void end_ship_activation(self) :
        """
        Tail of the Execute Maneuver step: end the active ship's activation and
        pass to the next activation, or to the Squadron Phase.  Reached
        directly, after overlapped squadrons are placed, or after asteroid
        damage cards are drawn.
        """
        cdef Ship active_ship = self.active_ship

        active_ship.end_activation()

        if self.get_valid_ship_activation(- self.current_player) :
            # if opponent has ships left
            self.current_player *= -1
            self.phase = Phase.SHIP_ACTIVATE

        elif self.get_valid_ship_activation(self.current_player) :
            # elif current player has ships left
            self.phase = Phase.SHIP_ACTIVATE

        else :
            self.start_squadron_phase()

    cdef void resolve_pending_placements(self) :
        """
        Place the squadrons the moving ship overlapped, one at a time in id
        order.  The opponent chooses among the touching slots; a squadron that
        cannot touch the ship is placed touching a squadron that does (RRG
        Overlapping), automatically, nearest to where it was.  Then the
        maneuver step resumes with any asteroid draws.
        """
        cdef Squad squad
        while self.pending_placement :
            squad = self.squads[<int>self.pending_placement[0]]
            if self.placement_actions() :
                self.phase = Phase.SHIP_PLACE_SQUAD
                return
            if not self.place_on_second_ring(squad) :
                # Deviation: no legal position at all, so the squadron is lost.
                squad.destroy()
            self.pending_placement = self.pending_placement[1:]

        if self.damage_draw_count > 0 :
            self.phase = Phase.DRAW_DAMAGE_CARD
        else :
            self.end_ship_activation()

    cdef bint place_on_second_ring(self, Squad squad) :
        cdef:
            Squad anchor
            Rect moving_base = self.ship_base_rect((<Ship>self.active_ship).id)
            double ox = squad.coords[0], oy = squad.coords[1]
            double ax, ay, px, py, d2
            double best_d2 = -1.0, best_x = 0.0, best_y = 0.0
            int k
        for anchor in self.squads :
            if anchor.destroyed or anchor.id in self.pending_placement : continue
            ax = anchor.coords[0]
            ay = anchor.coords[1]
            if rect_dist_sq(&moving_base, ax, ay) > SQUAD_SHIP_TOUCH_SQ : continue
            for k in range(RING2_COUNT) :
                px = ax + <double>RING2_OFFSETS[k][0]
                py = ay + <double>RING2_OFFSETS[k][1]
                if not self.position_is_clear(px, py) : continue
                d2 = (px - ox) * (px - ox) + (py - oy) * (py - oy)
                if best_d2 < 0.0 or d2 < best_d2 :
                    best_d2 = d2
                    best_x = px
                    best_y = py
        if best_d2 < 0.0 :
            return False
        squad.place(best_x, best_y)
        return True

    cdef void check_fleet_elimination(self) :
        """
        If all ships in a fleet are destroyed, ignoring squadrons, the game ends
        immediately.  If both fleets' last ships go at the same time, the higher
        score wins and a tie goes to the second player (RRG Winning and Losing).
        """
        cdef:
            bint first_out = self.total_destruction(self.first_player)
            bint second_out = self.total_destruction(self.second_player)
            int first_points, second_points
        if not (first_out or second_out) :
            return
        if first_out and second_out :
            first_points = self.get_point(self.first_player)
            second_points = self.get_point(self.second_player)
            self.winner = self.first_player if first_points > second_points else self.second_player
        elif first_out :
            self.winner = self.second_player
        else :
            self.winner = self.first_player

    cpdef tuple sample_chance_action(self) :
        """
        Samples the outcome of the current chance node.  Every driver loop
        (rollout, self-play, evaluation, MCTS selection) routes chance phases
        through here so they all sample identically.
        """
        cdef AttackInfo attack_info
        cdef Ship ship

        if self.phase == Phase.ATTACK_ROLL_DICE :
            if (attack_info := self.attack_info) is None :
                raise ValueError("No attack info for the current game phase.")
            return ('roll_dice_action', roll_dice(attack_info.dice_to_roll))

        if self.phase == Phase.DRAW_DAMAGE_CARD :
            if self.damage_draw_count <= 0 :
                raise ValueError("No pending damage card draw for the current game phase.")
            ship = self.ships[self.damage_draw_ship_id]
            return ('draw_damage_card_action', draw_damage_card(self.get_dealt_damage_counts(ship.team)))

        raise ValueError(f'{Phase(self.phase)} is not a chance node')

    cpdef tuple get_dealt_damage_counts(self, int player) :
        """
        Per-type count of the faceup damage cards held by `player`'s surviving
        ships.  Each player draws from their own deck, so this is exactly what
        is missing from that deck.  Facedown cards are ignored, and a destroyed
        ship's cards return to circulation.
        """
        cdef:
            list counts = [0] * DAMAGE_CARD_TYPES
            Ship ship
            int card_index

        for ship in self.ships:
            if ship.team != player or ship.destroyed :
                continue
            for card_index in range(DAMAGE_CARD_TYPES) :
                counts[card_index] += <int>ship.damage_cards[card_index]
        return tuple(counts)

    cpdef void visualize_action(self, object action) :
        """
        get action string and call visualizer
        """
        if not self.debuging_visual : return

        action_str = get_action_str(self, action)
        if action_str is None or 'pass' in action_str: return

        maneuver_tool = None
        if action[0] == 'determine_course_action':
            course, placement = action[1]
            if self.active_ship is None : raise ValueError('Need active ship to perform maneuver')
            maneuver_tool, _ = self.active_ship._tool_coordination(course, placement)
        turn = 'First' if self.current_player == self.first_player else 'Second'
        self.visualize(f"Round {self.round} | {Phase(self.phase).name.replace('_',' ').title()} | {turn} Player\n{action_str}", maneuver_tool)

    cpdef void deploy_ship(self, Ship ship, float x, float y, float orientation, int speed) :
        self.ships.append(ship)
        ship.deploy(self, x, y, orientation, speed, len(self.ships) - 1)
        self.invalidate_geometry()
        self.visualize(f'\n{ship.name} is deployed.')
        self.active_ship = self.get_next_command_ship(self.current_player)

        ship_view = self.ship_static_encode_array[ship.id]


        cdef float MAX_DICE = Config.GLOBAL_MAX_DICE

        # Assign features 0 to 26 in a single block
        # (11 Base Stats + 12 Battery Dice + 3 Anti-Squad Dice) = 26 values
        ship_view[0:26] = [
            # --- Status & Values (11 features) ---
            ship.team,                                    # 0: Team
            ship.size_class / float(SizeClass.LARGE),     # 1: Size
            ship.command_value / 3.0,                     # 2: Command
            ship.squad_value / Config.GLOBAL_MAX_SQUAD_VALUE,    # 3: Squad
            ship.engineer_value / Config.GLOBAL_MAX_ENGINEER_VALUE, # 4: Engineer
            ship.point / 100.0,                           # 5: Points
            ship.max_hull / Config.GLOBAL_MAX_HULL,       # 6: Hull
            ship.max_shield[HullSection.FRONT] / Config.GLOBAL_MAX_SHIELDS, # 7: Front Shield
            ship.max_shield[HullSection.RIGHT] / Config.GLOBAL_MAX_SHIELDS, # 8: Right Shield
            ship.max_shield[HullSection.REAR] / Config.GLOBAL_MAX_SHIELDS,  # 9: Rear Shield
            ship.max_shield[HullSection.LEFT] / Config.GLOBAL_MAX_SHIELDS,  # 10: Left Shield

            # --- Battery Armament (4 hulls * 3 colors = 12 features) ---
            *[dice / MAX_DICE for hull in range(hull_type) for dice in ship.battery[hull]],

            # --- Anti-Squad Armament (3 features) ---
            *[dice / MAX_DICE for dice in ship.anti_squad]
        ]

        # Assign Navigation Chart (Features 26 to 36)
        ship_view[26:36] = ship.nav_chart_vector

    cpdef void deploy_squad(self, Squad squad, float x, float y) :
        """
        Add a squadron to the play area.  Placement legality (RRG Setup) is the
        caller's job; this only records it.  Static keyword masks are kept as
        bitmasks so engagement queries stay integer operations.
        """
        if len(self.squads) >= 63 :
            raise ValueError('Squadron bitmasks support at most 63 squadrons')
        self.squads.append(squad)
        squad.deploy(self, x, y, len(self.squads) - 1)
        self.squad_in_range.append(0)
        self.squad_engaged.append(0)
        if squad.heavy : self.heavy_mask |= (<long long>1) << squad.id
        if squad.escort : self.escort_mask |= (<long long>1) << squad.id
        self.invalidate_geometry()
        self.visualize(f'\n{str(squad)} is deployed.')

    def place_obstacle(self, obstacle: Obstacle, x: float, y: float, orientation: float, flip: bool) -> None:
        """Places one static obstacle in the play area.

        Spatial planes are encoded from the obstacle state on demand.  Keeping
        no mutable raster here ensures snapshots and deep-copied games always
        derive geometry from the authoritative obstacle objects.  The squadron
        line-of-sight polygons are the same cached world coordinates, kept as
        float64 because obstacles never move.
        """
        obstacle.place_obstacle(x, y, orientation, flip)
        self.obstacles.append(obstacle)
        self.obstacles.sort(key=lambda obstacle: obstacle.index)

        self.obstacle_polygons = [
            np.ascontiguousarray(cache._obstacle_coordinate(o.get_hash_state()), dtype=np.float64)
            for o in self.obstacles
        ]
        self.obstacle_bounds = np.array([
            (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())
            for poly in self.obstacle_polygons
        ], dtype=np.float64).reshape(-1, 4)
        self.invalidate_geometry()
        self.visualize(f'\n{str(obstacle)} is placed.')

    # --- Squadron geometry (TODO.md, G1-G19) ---

    cdef void invalidate_geometry(self) :
        """
        Drop every cached geometric answer.  Called whenever a ship or
        squadron moves or is destroyed, and on every revert.  Cheap when the
        cache is already empty, which is the common case during MCTS.
        """
        self.ship_geometry_valid = False
        self.squad_rows_valid = 0
        if self.squad_pair_los :
            self.squad_pair_los.clear()

    cdef void ensure_ship_geometry(self) :
        """
        Base and token rectangles for every ship, from the same hashed pose
        that `_ship_coordinate()` uses, so these tests agree with the cached
        corners (to ~5e-5 mm).
        """
        cdef:
            Ship ship
            Py_ssize_t k, n = len(self.ships)
            double theta
            double[:, ::1] geometry
            tuple hash_state
        if self.ship_geometry_valid :
            return
        if self.ship_geometry.shape[0] != n :
            self.ship_geometry = np.zeros((n, 8), dtype=np.float64)
        geometry = self.ship_geometry
        for k in range(n) :
            ship = self.ships[k]
            hash_state = ship.get_ship_hash_state()
            theta = <double>(<int>hash_state[3]) / HASH_PRECISION
            geometry[k, 0] = <double>(<int>hash_state[1]) / HASH_PRECISION
            geometry[k, 1] = <double>(<int>hash_state[2]) / HASH_PRECISION
            geometry[k, 2] = cos(theta)
            geometry[k, 3] = sin(theta)
            geometry[k, 4] = <double>ship.base_size[0] / 2.0
            geometry[k, 5] = <double>ship.base_size[1] / 2.0
            geometry[k, 6] = <double>ship.token_size[0] / 2.0
            geometry[k, 7] = <double>ship.token_size[1] / 2.0
        self.ship_geometry_valid = True

    cdef Rect ship_base_rect(self, int ship_id) :
        """The ship base, shield-dial frames included (RRG Overlapping, FAQ)."""
        cdef Rect rect
        cdef double[:, ::1] geometry
        self.ensure_ship_geometry()
        geometry = self.ship_geometry
        rect.cx = geometry[ship_id, 0]
        rect.cy = geometry[ship_id, 1]
        rect.c = geometry[ship_id, 2]
        rect.s = geometry[ship_id, 3]
        rect.hw = geometry[ship_id, 4]
        rect.hl = geometry[ship_id, 5]
        return rect

    cdef Rect ship_token_rect(self, int ship_id) :
        """The ship token, which range is measured from (RRG Measuring)."""
        cdef Rect rect
        cdef double[:, ::1] geometry
        self.ensure_ship_geometry()
        geometry = self.ship_geometry
        rect.cx = geometry[ship_id, 0]
        rect.cy = geometry[ship_id, 1]
        rect.c = geometry[ship_id, 2]
        rect.s = geometry[ship_id, 3]
        rect.hw = geometry[ship_id, 6]
        rect.hl = geometry[ship_id, 7]
        return rect

    cdef void ensure_squad_row(self, int squad_id) :
        """
        Build squadron `squad_id`'s row of the distance-1 and engagement
        bitmasks.  Enemies only - a squadron does not engage friendly squadrons
        (RRG Engagement).  Distance 1 is a centre radius (G2) behind an exact
        axis pre-reject (G5); line of sight is tested only for pairs inside it
        (G15), and each pair's answer is shared by both rows (G9).
        """
        cdef:
            Squad squad, other
            long long in_range = 0, engaged = 0
            double x, y, dx, dy
            int key
            bint obstructed
        if (self.squad_rows_valid >> squad_id) & 1 :
            return
        squad = self.squads[squad_id]
        if not squad.destroyed :
            x = squad.coords[0]
            y = squad.coords[1]
            for other in self.squads :
                if other.team == squad.team or other.destroyed : continue
                dx = <double>other.coords[0] - x
                dy = <double>other.coords[1] - y
                if fabs(dx) > Q2Q or fabs(dy) > Q2Q : continue
                if dx * dx + dy * dy > Q2Q * Q2Q : continue
                in_range |= (<long long>1) << other.id
                key = squad_id * 64 + other.id if squad_id < other.id else other.id * 64 + squad_id
                if key in self.squad_pair_los :
                    obstructed = self.squad_pair_los[key]
                else :
                    obstructed = self.q2q_obstructed(x, y, other.coords[0], other.coords[1])
                    self.squad_pair_los[key] = obstructed
                # Obstructed line of sight breaks engagement, though the two can
                # still attack each other (RRG Engagement).
                if not obstructed :
                    engaged |= (<long long>1) << other.id
        self.squad_in_range[squad_id] = in_range
        self.squad_engaged[squad_id] = engaged
        self.squad_rows_valid |= (<long long>1) << squad_id

    cdef bint q2q_obstructed(self, double ax, double ay, double bx, double by) :
        """
        Line of sight between two squadron centres (RRG Line of Sight) crosses
        an obstacle or a ship's base.  Squadrons never obstruct.  A squadron
        sitting on an obstacle traces from a point inside it, so it is always
        obstructed and can never be engaged.
        """
        cdef:
            Ship ship
            Rect rect
            Py_ssize_t k
            double xmin = ax if ax < bx else bx
            double xmax = bx if ax < bx else ax
            double ymin = ay if ay < by else by
            double ymax = by if ay < by else ay
            double rxmin, rymin, rxmax, rymax
            double[:, ::1] bounds = self.obstacle_bounds
        for ship in self.ships :
            if ship.destroyed : continue
            rect = self.ship_base_rect(ship.id)
            rect_bounds(&rect, 0.0, &rxmin, &rymin, &rxmax, &rymax)
            if rxmax < xmin or rxmin > xmax or rymax < ymin or rymin > ymax : continue
            if segment_hits_rect(&rect, ax, ay, bx, by) :
                return True
        for k in range(bounds.shape[0]) :
            if bounds[k, 2] < xmin or bounds[k, 0] > xmax or bounds[k, 3] < ymin or bounds[k, 1] > ymax : continue
            if segment_hits_polygon(self.obstacle_polygons[k], ax, ay, bx, by) :
                return True
        return False

    cpdef bint squad_pair_obstructed(self, Squad attacker, Squad defender) :
        """Attack obstruction between two squadrons, read from the shared cache."""
        self.ensure_squad_row(attacker.id)
        if ((<long long>self.squad_in_range[attacker.id]) >> defender.id) & 1 :
            return not (((<long long>self.squad_engaged[attacker.id]) >> defender.id) & 1)
        return self.q2q_obstructed(attacker.coords[0], attacker.coords[1], defender.coords[0], defender.coords[1])

    cpdef bint squad_engaged_with_other(self, Squad squad, Squad other) :
        """Is `squad` engaged with any squadron besides `other`?  (Swarm.)"""
        self.ensure_squad_row(squad.id)
        return ((<long long>self.squad_engaged[squad.id]) & ~((<long long>1) << other.id)) != 0

    cpdef bint squad_is_engaged(self, Squad squad) :
        self.ensure_squad_row(squad.id)
        return <long long>self.squad_engaged[squad.id] != 0

    cpdef tuple squad_engagement_masks(self, Squad squad) :
        """(enemies at distance 1, enemies engaged) as bitmasks over squad ids."""
        self.ensure_squad_row(squad.id)
        return (self.squad_in_range[squad.id], self.squad_engaged[squad.id])

    cpdef bint squad_movement_restricted(self, Squad squad) :
        """
        An engaged squadron cannot move, and cannot attack ships, unless every
        squadron engaging it has heavy (RRG Engagement, Heavy).  Heavy squadrons
        still engage: they count for swarm and escort.
        """
        self.ensure_squad_row(squad.id)
        return ((<long long>self.squad_engaged[squad.id]) & ~self.heavy_mask) != 0

    cpdef list squad_move_actions(self, Squad squad) :
        """
        Legal cells of the 33x33 absolute move grid (TODO D2).  A cell is legal
        when its centre is within the squadron's speed (G1, static disk G7/G14),
        the whole base stays in the play area (G12: a contiguous index window),
        and it overlaps no squadron or ship (G13: each blocker marks only the
        cells it can reach).  Obstacles do not block squadrons (RRG Obstacles).
        The centre cell is always legal: it is "remain in place" (RRG Squadron
        Movement).
        """
        cdef:
            list actions = []
            unsigned char blocked[MOVE_GRID_N][MOVE_GRID_N]
            int speed = squad.speed
            double x0 = squad.coords[0], y0 = squad.coords[1]
            double ox, oy, px, py, dx, dy
            double bxmin, bymin, bxmax, bymax
            int i_lo, i_hi, j_lo, j_hi, a_lo, a_hi, b_lo, b_hi, a, b
            Squad other
            Ship ship
            Rect rect

        if speed < 0 : speed = 0
        if speed > 5 : speed = 5
        memset(blocked, 0, sizeof(blocked))

        i_lo = max(0, <int>ceil((SQ_R - x0) / MOVE_PITCH) + MOVE_CENTER)
        i_hi = min(MOVE_GRID_N - 1, <int>floor((self.player_edge - SQ_R - x0) / MOVE_PITCH) + MOVE_CENTER)
        j_lo = max(0, <int>ceil((SQ_R - y0) / MOVE_PITCH) + MOVE_CENTER)
        j_hi = min(MOVE_GRID_N - 1, <int>floor((self.short_edge - SQ_R - y0) / MOVE_PITCH) + MOVE_CENTER)

        for other in self.squads :
            if other is squad or other.destroyed : continue
            ox = other.coords[0]
            oy = other.coords[1]
            a_lo = max(i_lo, <int>ceil((ox - 2.0 * SQ_R - x0) / MOVE_PITCH) + MOVE_CENTER)
            a_hi = min(i_hi, <int>floor((ox + 2.0 * SQ_R - x0) / MOVE_PITCH) + MOVE_CENTER)
            b_lo = max(j_lo, <int>ceil((oy - 2.0 * SQ_R - y0) / MOVE_PITCH) + MOVE_CENTER)
            b_hi = min(j_hi, <int>floor((oy + 2.0 * SQ_R - y0) / MOVE_PITCH) + MOVE_CENTER)
            for a in range(a_lo, a_hi + 1) :
                dx = x0 + (a - MOVE_CENTER) * MOVE_PITCH - ox
                for b in range(b_lo, b_hi + 1) :
                    dy = y0 + (b - MOVE_CENTER) * MOVE_PITCH - oy
                    if dx * dx + dy * dy < SQUAD_SQUAD_OVERLAP_SQ :
                        blocked[a][b] = 1

        for ship in self.ships :
            if ship.destroyed : continue
            rect = self.ship_base_rect(ship.id)
            rect_bounds(&rect, SQ_R, &bxmin, &bymin, &bxmax, &bymax)
            a_lo = max(i_lo, <int>ceil((bxmin - x0) / MOVE_PITCH) + MOVE_CENTER)
            a_hi = min(i_hi, <int>floor((bxmax - x0) / MOVE_PITCH) + MOVE_CENTER)
            b_lo = max(j_lo, <int>ceil((bymin - y0) / MOVE_PITCH) + MOVE_CENTER)
            b_hi = min(j_hi, <int>floor((bymax - y0) / MOVE_PITCH) + MOVE_CENTER)
            for a in range(a_lo, a_hi + 1) :
                px = x0 + (a - MOVE_CENTER) * MOVE_PITCH
                for b in range(b_lo, b_hi + 1) :
                    py = y0 + (b - MOVE_CENTER) * MOVE_PITCH
                    if rect_dist_sq(&rect, px, py) < SQUAD_SHIP_OVERLAP_SQ :
                        blocked[a][b] = 1

        for a in range(i_lo, i_hi + 1) :
            for b in range(j_lo, j_hi + 1) :
                if MOVE_DISK[speed, a, b] and not blocked[a][b] :
                    actions.append((<list>MOVE_ACTIONS[a])[b])
        return actions

    cpdef list command_eligible_squads(self, Ship ship) :
        """
        Friendly squadrons a squadron command can still activate: unactivated
        and at close-medium range of the ship (RRG Commands), measured from the
        ship token to the squadron base (G3).
        """
        cdef:
            list squads = []
            Squad squad
            Rect token = self.ship_token_rect(ship.id)
        for squad in self.squads :
            if squad.team != ship.team or squad.destroyed or squad.activated : continue
            if rect_dist_sq(&token, squad.coords[0], squad.coords[1]) <= COMMAND_RANGE_SQ :
                squads.append(squad)
        return squads

    cdef bint position_is_clear(self, double x, double y) :
        """
        A squadron centre at (x, y) is wholly inside the play area and overlaps
        no ship or squadron.  Squadrons still waiting to be placed have been
        moved out of the way, so they do not block (RRG Overlapping).
        """
        cdef:
            Ship ship
            Squad squad
            Rect rect
            double dx, dy
        if x < SQ_R or x > self.player_edge - SQ_R or y < SQ_R or y > self.short_edge - SQ_R :
            return False
        for ship in self.ships :
            if ship.destroyed : continue
            rect = self.ship_base_rect(ship.id)
            if rect_dist_sq(&rect, x, y) < SQUAD_SHIP_OVERLAP_SQ :
                return False
        for squad in self.squads :
            if squad.destroyed or squad.id in self.pending_placement : continue
            dx = <double>squad.coords[0] - x
            dy = <double>squad.coords[1] - y
            if dx * dx + dy * dy < SQUAD_SQUAD_OVERLAP_SQ :
                return False
        return True

    cpdef list placement_actions(self) :
        """Touching slots open for the next overlapped squadron."""
        cdef:
            list actions = []
            int slot
            object points = cache._ship_coordinate((<Ship>self.active_ship).get_ship_hash_state())['squad_placement_points']
        for slot in range(SQUAD_PLACEMENT_SLOTS) :
            if self.position_is_clear(<double>points[slot, 0], <double>points[slot, 1]) :
                actions.append(PLACE_ACTIONS[slot])
        return actions

    cdef bint has_live_squad(self, int player) :
        cdef Squad squad
        for squad in self.squads :
            if squad.team == player and not squad.destroyed :
                return True
        return False

    cdef bint total_destruction(self, int player) :
        cdef Ship ship
        for ship in self.ships:
            if ship.team == player and not ship.destroyed:
                return False
        return True

    cdef list get_valid_ship_activation(self, int player) :
        """
        Returns a list of ships that can be activated by the given player.
        A ship can be activated if it is not destroyed, not already activated, and belongs to the player.
        """
        cdef list valid_ships = []
        cdef Ship ship
        for ship in self.ships:
            if ship.team == player and not ship.destroyed and not ship.activated:
                valid_ships.append(ship)
        return valid_ships

    cpdef Ship get_next_command_ship(self, int player) :
        """
        Returns the ship of `player` that still needs a command dial assigned
        this round (len(command_stack) < command_value), preferring highest
        point cost, then highest current hull, then lowest ship id. None if
        every ship of `player` already has a full command stack.
        """
        cdef:
            Ship ship
            Ship best = None
        for ship in self.ships:
            if ship.team != player or ship.destroyed:
                continue
            if len(ship.command_stack) >= ship.command_value:
                continue
            if best is None or (ship.point, ship.hull, -ship.id) > (best.point, best.hull, -best.id):
                best = ship
        return best

    cdef list get_valid_squad_activation(self, int player) :
        """
        Returns a list of squadrons that can be activated by the given player.
        A squadron can be activated if it is not destroyed, not already activated, and belongs to the player.
        """
        cdef list valid_squadrons = []
        cdef Squad squad
        for squad in self.squads:
            if squad.team == player and not squad.destroyed and not squad.activated:
                valid_squadrons.append(squad)
        return valid_squadrons

    cdef void status_phase(self) :
        cdef:
            Ship ship
            Squad squad
            int p1_points, p2_points

        # 1. Refresh all active ships and squadrons for the next round
        for ship in self.ships:
            if not ship.destroyed:
                ship.status_phase()
        for squad in self.squads:
            if not squad.destroyed:
                squad.status_phase()

        # 2. A wiped-out fleet already ended the game when its last ship was
        # destroyed (check_fleet_elimination).  Otherwise the game ends after
        # six rounds: highest score wins, a tie goes to the second player
        # (RRG Winning and Losing).
        if self.round == 6:
            p1_points = self.get_point(self.first_player)
            p2_points = self.get_point(self.second_player)
            self.winner = self.first_player if p1_points > p2_points else self.second_player

        # 3. If the game is not over, advance to the next round
        else:
            self.round += 1
            self.current_player = self.first_player
            self.active_ship = self.get_next_command_ship(self.first_player)
            self.phase = Phase.COMMAND_PHASE

    cpdef int get_point(self, int player) :
        cdef Ship ship
        cdef Squad squad
        cdef list ship_point, squad_point
        cdef point = 0
        for ship in self.ships :
            if ship.team != player and ship.destroyed :
                point += ship.point
        for squad in self.squads :
            if squad.team != player and squad.destroyed :
                point += squad.point
        return point

    def visualize(self, title : str, maneuver_tool : list[tuple[float, float]] | None = None) -> None:
        if not self.debuging_visual:
            return
        visualizer.visualize(self, title, maneuver_tool)

    cpdef object get_snapshot(self):
        """Captures the essential state of the entire game."""
        cdef list ship_snapshots = []
        cdef list squad_snapshots = []

        cdef Ship ship
        cdef Squad squad

        for ship in self.ships:
            ship_snapshots.append(ship.get_snapshot())

        for squad in self.squads:
            squad_snapshots.append(squad.get_snapshot())


        return (
            self.winner,
            self.round,
            self.phase,
            self.current_player,
            self.decision_player,
            (<Ship>self.active_ship).id if self.active_ship is not None else None,
            (<Ship>self.defend_ship).id if self.defend_ship is not None else None,
            (<Squad>self.active_squad).id if self.active_squad is not None else None,
            (<Squad>self.defend_squad).id if self.defend_squad is not None else None,
            self.squad_activation_count,
            self.squad_can_move,
            self.squad_can_attack,
            self.pending_placement,
            (<AttackInfo>self.attack_info).get_snapshot() if self.attack_info is not None else None,
            (self.damage_draw_ship_id, self.damage_draw_count, self.damage_draw_source),
            tuple(ship_snapshots),
            tuple(squad_snapshots)
        )

    cpdef void revert_snapshot(self, object snapshot):
        """Restores the entire game state from a snapshot."""
        (self.winner, self.round, self.phase, self.current_player, self.decision_player,
         active_ship_id, defend_ship_id, active_squad_id, defend_squad_id,
         self.squad_activation_count, self.squad_can_move, self.squad_can_attack, self.pending_placement,
         attack_info_snapshot, damage_draw_state, ship_states, squad_states) = snapshot
        self.attack_info = AttackInfo.from_snapshot(attack_info_snapshot) if attack_info_snapshot is not None else None
        (self.damage_draw_ship_id, self.damage_draw_count, self.damage_draw_source) = damage_draw_state

        self.active_ship = self.ships[<int>active_ship_id] if active_ship_id is not None else None
        self.defend_ship = self.ships[<int>defend_ship_id] if defend_ship_id is not None else None
        self.active_squad = self.squads[<int>active_squad_id] if active_squad_id is not None else None
        self.defend_squad = self.squads[<int>defend_squad_id] if defend_squad_id is not None else None

        for i, ship_snapshot in enumerate(ship_states):
            (<Ship>self.ships[i]).revert_snapshot(ship_snapshot)
        for i, squad_snapshot in enumerate(squad_states):
            (<Squad>self.squads[i]).revert_snapshot(squad_snapshot)
        self.fleet_check_pending = False
        self.invalidate_geometry()
