cimport numpy as cnp
from ship cimport Ship
from squad cimport Squad
from attack_info cimport AttackInfo
from squad_geometry cimport Rect

cdef class Armada:
    cdef :
        public int first_player, second_player, current_player, decision_player, simulation_player, first_faction, second_faction
        public int round, squad_activation_count, image_counter, para_index
        public int phase
        public int damage_draw_ship_id, damage_draw_count, damage_draw_source
        public float player_edge, short_edge, winner
        public list ships, squads, obstacles
        public Ship active_ship, defend_ship
        public object active_squad, defend_squad
        public AttackInfo attack_info
        public bint debuging_visual
        public cnp.ndarray ship_static_encode_array

        # Active squadron's remaining steps this activation (RRG Squadron
        # Activation): Squadron Phase grants one, a squadron command both.
        public bint squad_can_move, squad_can_attack
        # Squadrons overlapped by the moving ship, placed one at a time in id
        # order by the opponent (RRG Overlapping).
        public tuple pending_placement

        # Set by Ship.destroy(); checked once the current action has resolved,
        # because a wiped-out fleet ends the game immediately.
        public bint fleet_check_pending
        # Debug: assert each skipped forced step really had one legal action.
        public bint verify_forced

        # --- Per-state geometry cache (TODO G9-G11, G16) ---
        # Never snapshotted: any change of position, destruction or revert
        # invalidates it, and it is rebuilt lazily on first use.
        bint ship_geometry_valid
        cnp.ndarray ship_geometry        # (ships, 8) cx, cy, cos, sin, base hw/hl, token hw/hl
        long long squad_rows_valid       # bit i: row i of the masks below is current
        list squad_in_range              # enemies at distance 1, as a bitmask per squad
        list squad_engaged               # ...and also with unobstructed line of sight
        dict squad_pair_los              # pair key -> obstructed, shared by both rows
        long long heavy_mask, escort_mask
        list obstacle_polygons           # float64 (n, 2) vertices; obstacles never move
        cnp.ndarray obstacle_bounds      # (obstacles, 4) xmin, ymin, xmax, ymax

    # Declare the C-level methods
    cpdef object get_snapshot(self)
    cpdef void revert_snapshot(self, object snapshot)
    cdef void update_decision_player(self)
    cpdef int get_point(self, int player)
    cpdef list get_valid_actions(self)
    cpdef void apply_action(self, tuple action)
    cpdef void visualize_action(self, object action)
    cpdef void deploy_ship(self, Ship ship, float x, float y, float orientation, int speed)
    cpdef void deploy_squad(self, Squad squad, float x, float y)
    cdef bint total_destruction(self, int player)
    cdef list get_valid_ship_activation(self, int player)
    cdef list get_valid_squad_activation(self, int player)
    cdef void status_phase(self)
    cpdef Ship get_next_command_ship(self, int player)
    cdef void after_attack_damage(self)
    cdef void finish_ship_attack(self)
    cdef void finish_squad_attack(self)
    cdef void end_ship_activation(self)
    cdef void end_squad_activation(self)
    cdef void next_squadron_turn(self)
    cdef void start_squadron_phase(self)
    cdef void resolve_pending_placements(self)
    cdef void check_fleet_elimination(self)
    cpdef tuple sample_chance_action(self)
    cpdef tuple get_dealt_damage_counts(self, int player)

    # --- forced single-action steps ---
    cdef tuple forced_pass_action(self)
    cdef list squad_command_resolve_actions(self)
    cdef list repair_effect_actions(self)
    cdef list ship_target_actions(self, bint stop_at_first)
    cdef list attack_effect_actions(self)
    cdef list defense_token_actions(self)
    cdef list redirect_actions(self)
    cdef list additional_squad_target_actions(self)
    cdef list squad_activation_actions(self)
    cdef list squad_attack_target_actions(self, Squad squad, bint stop_at_first)
    cdef bint squad_attack_pass_offered(self)

    # --- squadron geometry ---
    cdef void invalidate_geometry(self)
    cdef void ensure_ship_geometry(self)
    cdef Rect ship_base_rect(self, int ship_id)
    cdef Rect ship_token_rect(self, int ship_id)
    cdef void ensure_squad_row(self, int squad_id)
    cdef bint q2q_obstructed(self, double ax, double ay, double bx, double by)
    cpdef bint squad_pair_obstructed(self, Squad attacker, Squad defender)
    cpdef bint squad_engaged_with_other(self, Squad squad, Squad other)
    cpdef bint squad_is_engaged(self, Squad squad)
    cpdef tuple squad_engagement_masks(self, Squad squad)
    cpdef bint squad_movement_restricted(self, Squad squad)
    cpdef list squad_move_actions(self, Squad squad)
    cpdef list command_eligible_squads(self, Ship ship)
    cdef bint position_is_clear(self, double x, double y)
    cpdef list placement_actions(self)
    cdef bint place_on_second_ring(self, Squad squad)
    cdef bint has_live_squad(self, int player)
