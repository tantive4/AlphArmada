from __future__ import annotations

from ship import Ship
from squad import Squad
from enum_class import *
import cache_function as cache
from dice import *
from action_phase import Phase
from defense_token import TokenType

class AttackInfo :
    def __init__(self, attacker:tuple[Ship,HullSection]|Squad, defender:tuple[Ship, HullSection] | Squad) -> None :
        self.initialize_attribute()

        if isinstance(attacker, tuple) :
            self.is_attacker_ship : bool= True
            attack_ship, attack_hull = attacker
            self.attack_ship_id : int = attack_ship.id
            self.attack_hull : HullSection = attack_hull
            attack_ship.attack_count += 1

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
                attack_hist_list = list(attack_ship.attack_history)
                attack_hist_list[attack_hull] = (defend_ship.id, defend_hull)
                attack_ship.attack_history = tuple(attack_hist_list)

                self.attack_range : AttackRange = cache.attack_range_s2s(attack_ship.get_ship_hash_state(), defend_ship.get_ship_hash_state())[1][attack_hull][defend_hull]
                self.obstructed : bool = attack_ship.is_obstruct_s2s(attack_hull, defend_ship, defend_hull)
            else :
                attack_hist_list = list(attack_ship.attack_history)
                attack_hist_list[attack_hull] = (defend_squad.id,)
                attack_ship.attack_history = tuple(attack_hist_list)

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

        attack_hist_list = list(attack_ship.attack_history)
        attack_hist_list[attack_hull] = self.squadron_target
        attack_ship.attack_history = tuple(attack_hist_list)
        
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

    def __eq__(self,other) -> bool :
        if not isinstance(other, AttackInfo) : return False
        return self.__dict__ == other.__dict__
    
    def __str__(self) -> str :
        return str(self.__dict__)
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

    def get_snapshot(self) -> dict:
        """Creates a serializable snapshot of the AttackInfo state."""
        return self.__dict__.copy()

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> AttackInfo:
        """Reconstructs an AttackInfo object from a snapshot."""
        # We can't call the original __init__ because it recalculates things.
        # So we create a dummy instance and then populate its __dict__.
        # This is a common pattern for serialization/deserialization.
        instance = cls.__new__(cls)
        instance.__dict__.update(snapshot)
        
        return instance