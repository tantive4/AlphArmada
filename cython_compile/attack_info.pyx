from __future__ import annotations

from ship cimport Ship
from squad cimport Squad
from enum_class import *
import cache_function as cache
from dice import *
from action_phase import Phase
from defense_token import TokenType

cdef class AttackInfo :

    def __init__(self, attacker:tuple[Ship,HullSection]|Squad, defender:tuple[Ship, HullSection] | Squad) -> None :
        cdef Ship attack_ship
        cdef Squad attack_squad
        cdef Ship defend_ship
        cdef Squad defend_squad
        
        self.initialize_attribute()

        if isinstance(attacker, tuple) :
            self.is_attacker_ship : bool= True
            attack_ship, attack_hull = attacker
            self.attack_ship_id : int = attack_ship.id
            self.attack_hull : HullSection = attack_hull
            attack_ship.attack_count += 1
            attack_ship.attack_impossible_hull += (attack_hull,)
        else :
            self.is_attacker_ship : bool= False
            attack_squad = attacker
            self.attack_squad_id : int = attack_squad.id
            attack_squad.can_attack = False

        if isinstance(defender, tuple) :
            self.is_defender_ship : bool = True
            defend_ship, defend_hull = defender
            self.defend_ship_id : int = defend_ship.id
            self.defend_hull : HullSection = defend_hull
        else :
            self.is_defender_ship : bool = False
            defend_squad = defender
            self.defend_squad_id : int = defend_squad.id


        if self.is_attacker_ship :

            self.con_fire_dial = False
            self.con_fire_token = False
            
            if self.is_defender_ship :
                self.attack_range : AttackRange = cache.attack_range_s2s(attack_ship.get_ship_hash_state(), defend_ship.get_ship_hash_state())[1][attack_hull][defend_hull]
                self.obstructed : bool = attack_ship.is_obstruct_s2s(attack_hull, defend_ship, defend_hull)
            else :
                self.attack_range : AttackRange = cache.attack_range_s2q(attack_ship.get_ship_hash_state(), defend_squad.get_squad_hash_state())[attack_hull]
                self.obstructed : bool = attack_ship.is_obstruct_s2q(attack_hull, defend_squad)
                self.squadron_target : tuple[int, ...] = (self.defend_squad_id,)

            self.dice_to_roll : tuple[int, ...] = attack_ship.gather_dice(attack_hull, self.attack_range, is_ship=self.is_defender_ship)


        else :
            self.attack_range : AttackRange = AttackRange.CLOSE
            self.dice_to_roll : tuple[int, ...] = attack_squad.gather_dice(is_ship=self.is_defender_ship)
            
            self.bomber : bool = attack_squad.bomber

            if self.is_defender_ship :
                self.obstructed : bool = attack_squad.is_obstruct_q2s(defend_ship, defend_hull)
            else :
                self.obstructed : bool = attack_squad.is_obstruct_q2q(defend_squad)
                self.swarm : bool = attack_squad.swarm


    def declare_additional_squad_target(self, attacker:tuple[Ship, HullSection], defend_squad : Squad) -> None :
        """
        Re-Initialize for additional squadron target
        """
        if not self.is_attacker_ship or self.is_defender_ship :
            raise ValueError('This is not a ship-to-squadron attack')
        if defend_squad.id in self.squadron_target :
            raise ValueError('This squadron has already been targeted')
        
        self.initialize_attribute()

        attack_ship, attack_hull = attacker
        self.defend_squad_id = defend_squad.id
        self.squadron_target += (defend_squad.id,)

        self.attack_range : AttackRange = cache.attack_range_s2q(attack_ship.get_ship_hash_state(), defend_squad.get_squad_hash_state())[attack_hull]
        self.obstructed : bool = attack_ship.is_obstruct_s2q(attack_hull, defend_squad)

        self.dice_to_roll : tuple[int, ...] = attack_ship.gather_dice(attack_hull, self.attack_range, is_ship=self.is_defender_ship)

    def initialize_attribute(self) -> None :
        self.phase : Phase = Phase.ATTACK_RESOLVE_EFFECTS
        self.attack_pool_result : tuple[tuple[int, ...], ...]  = EMPTY_DICE_POOL

        self.con_fire_dial : bool = False
        self.con_fire_token : bool = False
        self.swarm : bool = False
        self.bomber : bool = False

        self.spent_token_indices : tuple[int,...] = ()
        self.spent_token_types : tuple[TokenType,...] = ()
        self.redirect_hull : HullSection | None = None
        self.critical : Critical | None = None
        self.total_damage : int = 0

    def __eq__(self, other):
        if not isinstance(other, AttackInfo): return False
        return self.get_snapshot() == other.get_snapshot()

    def __str__(self):
        return str(self.get_snapshot())
    __repr__ = __str__

    def calculate_total_damage(self) -> int :
        total_damage = 0
        damage_indices = SHIP_DAMAGE_INDICES if (self.is_attacker_ship or self.bomber) and self.is_defender_ship else SQUAD_DAMAGE_INDICES
        for dice_type in DICE :
            total_damage += sum([face_count * damage_value for face_count, damage_value in zip(self.attack_pool_result[dice_type], damage_indices[dice_type])])

        if TokenType.BRACE in self.spent_token_types :
            total_damage = (total_damage+1) // 2

        self.total_damage = total_damage
        return total_damage

    cpdef dict get_snapshot(self):
        return {
            'is_attacker_ship': self.is_attacker_ship,
            'is_defender_ship': self.is_defender_ship,
            'obstructed': self.obstructed,
            'con_fire_dial': self.con_fire_dial,
            'con_fire_token': self.con_fire_token,
            'bomber': self.bomber,
            'swarm': self.swarm,
            'attack_ship_id': self.attack_ship_id,
            'attack_squad_id': self.attack_squad_id,
            'defend_ship_id': self.defend_ship_id,
            'defend_squad_id': self.defend_squad_id,
            'total_damage': self.total_damage,
            'attack_hull': self.attack_hull,
            'defend_hull': self.defend_hull,
            'attack_range': self.attack_range,
            'dice_to_roll': self.dice_to_roll,
            'squadron_target': self.squadron_target,
            'phase': self.phase,
            'attack_pool_result': self.attack_pool_result,
            'spent_token_indices': self.spent_token_indices,
            'spent_token_types': self.spent_token_types,
            'redirect_hull': self.redirect_hull,
            'critical': self.critical,
        }

    @classmethod
    def from_snapshot(cls, dict snapshot):
        # Create a new, uninitialized instance
        cdef AttackInfo instance = cls.__new__(cls) 
        
        # Manually populate C-level attributes from the snapshot
        instance.is_attacker_ship = snapshot['is_attacker_ship']
        instance.is_defender_ship = snapshot['is_defender_ship']
        instance.obstructed = snapshot['obstructed']
        instance.con_fire_dial = snapshot['con_fire_dial']
        instance.con_fire_token = snapshot['con_fire_token']
        instance.bomber = snapshot['bomber']
        instance.swarm = snapshot['swarm']
        instance.attack_ship_id = snapshot['attack_ship_id']
        instance.attack_squad_id = snapshot['attack_squad_id']
        instance.defend_ship_id = snapshot['defend_ship_id']
        instance.defend_squad_id = snapshot['defend_squad_id']
        instance.total_damage = snapshot['total_damage']
        instance.attack_hull = snapshot['attack_hull']
        instance.defend_hull = snapshot['defend_hull']
        instance.attack_range = snapshot['attack_range']
        instance.dice_to_roll = snapshot['dice_to_roll']
        instance.squadron_target = snapshot['squadron_target']
        instance.phase = snapshot['phase']
        instance.attack_pool_result = snapshot['attack_pool_result']
        instance.spent_token_indices = snapshot['spent_token_indices']
        instance.spent_token_types = snapshot['spent_token_types']
        instance.redirect_hull = snapshot['redirect_hull']
        instance.critical = snapshot['critical']
        
        return instance