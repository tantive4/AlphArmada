# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AlphArmada: an AlphaZero-style self-play engine for a simplified Star Wars: Armada
tabletop game. Cython rules engine + Cython batched MCTS + PyTorch policy/value
network (`BigDeep`), with a Vessl-based distributed worker/downloader/trainer loop.
`README.md` has the full architecture writeup (model structure, MCTS loop, training
workflow) and `STRUCTURE.md` has an exhaustive file-by-file guide — read those for
breadth. This file focuses on the parts that aren't obvious from either: the
cross-file contract you must maintain when touching the game's phase/action-space
machinery, since that's where the sharp edges are.

Squadron **rules** are live (Stage 1 of the squadron plan in `TODO.md`): the Squadron
Phase, the squadron command, ship/squadron attacks in every combination, swarm,
escort, heavy, bomber, overlap placement and the 33x33 move grid. Counter is Stage 2,
and `setup_game()` still deploys no squadrons (`max_squad_points=0`, Stage 3), so real
self-play is squadron-free until then; the encoder and model do not see squadrons yet
(Stages 5-6). A few unrelated handlers in `armada.pyx` remain stubbed with
`raise NotImplementedError(f"simplified ...")` guarding otherwise-dead code — see the
Cython gotcha below before touching anything near those. Obstacles are live: all six
are placed at setup and they obstruct line of sight; of their overlap effects only
the asteroid's is implemented (see the obstacle and damage-card sections below).
Damage cards are tracked per type but have no faceup effects yet. Command stacks
(`Config.MAX_COMMAND_STACK = 3`) are live: ships assign a command dial per round in
`COMMAND_PHASE` (FIFO queue sized to `command_value`, fixed assignment order —
highest point cost first, tie-break by hull then id) and reveal the top of the stack
at the start of their own activation. All four commands resolve fully,
from a dial and/or a held token, and all four share **one shape**: an explicit
`resolve_*_command_action` carrying `(use_dial, use_token)` that discards the chosen
resources, appends to `Ship.resolved_command`, and stores the resulting *grant* —
then separate actions spend that grant.

| Command | Resolve action (phase) | Grant stored on | Spend action (phase) |
|---|---|---|---|
| Nav | `resolve_nav_command_action` (`SHIP_DETERMINE_COURSE`) | `Ship.nav_dial_granted` / `nav_token_granted` | `determine_course_action` (same phase) |
| Squad | `resolve_squad_command_action` (`SHIP_RESOLVE_SQUAD`) | `Armada.squad_activation_count` | `activate_squad_*_action` (`SQUAD_ACTIVATE`) |
| Repair | `resolve_repair_command_action` (`SHIP_RESOLVE_REPAIR`) | `Ship.engineer_budget` | `repair_hull_action` / `recover_shield_action` / `move_shield_action` (`SHIP_REPAIR_EFFECT`) |
| Con-fire | `resolve_confire_command_action` (`ATTACK_RESOLVE_EFFECTS`) | `AttackInfo.confire_dial` / `confire_token` | `use_confire_dial_action` / `use_confire_token_action` (same phase) |

Squad and Repair both "resolve after revealing the ship's command dial"; the engine
fixes squad first. They never interact, so the order cannot change an outcome.

This pre-commit shape is what the RRG actually prescribes — "A ship must decide
whether it is spending the dial, the token, or both **before** resolving that
command's effects" (RRG *Commands*) — so do not reintroduce the old post-hoc
derivation (`Ship.nav_command_used()` / `Ship.repair_command_used()`, both deleted)
that inferred the spend from the course or from `engineer_spent` after the fact.
Each command is one-shot per activation, gated by `Command.X not in
resolved_command`, which `end_activation()` clears; a ship activates once per round,
so that is also the RRG's "cannot resolve the same command more than once per round".

Two structural consequences worth knowing:

- **Repair needed a phase split** (`SHIP_RESOLVE_REPAIR` → `SHIP_REPAIR_EFFECT`)
  because the budget must exist before the spend menu can be enumerated. Nav and
  con-fire did **not**: their resolve action deliberately leaves `self.phase`
  untouched, so the ship re-enters the same phase with a wider menu and the resolve
  option gone. That is the cheap way to add a pre-commit step — reach for a new
  phase only when the follow-up menu depends on the grant.
- **Nav grant flags are cleared by `determine_course_action`**, not by
  `end_activation()`. Every path out of `SHIP_DETERMINE_COURSE` runs that action
  (including the asteroid `DRAW_DAMAGE_CARD` detour, which clears before suspending),
  and `destroy()` clears them too, so they cannot leak into another activation. This
  is correct by reachability rather than by construction; it is a deliberate choice,
  verified, and not something to "fix".

## Commands

Build the Cython extensions (required after any `.pyx`/`.pxd` edit — needs MSVC Build
Tools on Windows; a compiler toolchain on Linux):

```bash
python tools/cython_setup.py build_ext --inplace
```

Regenerate the action-space map (required after any change to phases or the actions
a phase generates):

```bash
python -m armada_game.helpers.action_space
```

Run the distributed roles:

```bash
python worker.py --worker_id 01
python downloader.py --num_worker 20
python trainer.py
```

There is no formal test suite (the scripts under `evaluation/debug/` are stale
one-off helpers, not tests — see `STRUCTURE.md`'s Active File Guide for which ones).
The de facto verification method for a rules/action change is a scripted random
rollout, run with the repo root on `PYTHONPATH` (`$env:PYTHONPATH = "<repo root>"` on
Windows) since it isn't a package script itself:

```python
import random
from armada_game.helpers.setup_game import setup_game
from armada_game.helpers.action_phase import Phase

N_GAMES = 100
counters = {"my_new_phase_visits": 0, "branch_a_chosen": 0, "branch_b_chosen": 0}
errors = 0

for i in range(N_GAMES):
    game = setup_game(debuging_visual=False)
    step = 0
    try:
        while game.winner == 0.0:
            if not game.decision_player:
                # chance node, not a player decision — sample it directly.
                # covers ATTACK_ROLL_DICE and DRAW_DAMAGE_CARD alike.
                action = game.sample_chance_action()
            else:
                actions = game.get_valid_actions()

                # hook in here, before the random pick, to inspect what's on offer
                if game.phase == Phase.MY_NEW_PHASE:
                    counters["my_new_phase_visits"] += 1
                    for a in actions:
                        assert <a's payload is legal against current state>

                action = random.choice(actions)
                if action[0] == 'branch_a_action': counters["branch_a_chosen"] += 1
                if action[0] == 'branch_b_action': counters["branch_b_chosen"] += 1

            game.apply_action(action)
            step += 1
            if step > 3000: raise RuntimeError("step limit")
    except Exception as e:
        errors += 1
        print(f"[game {i}] {type(e).__name__}: {e}")

print(f"errors: {errors}", counters)
```

Run this over 60-150 fresh random games. The two things it checks are different and
both matter:

- **No exceptions** — catches phase-transition bugs and illegal-action crashes
  (a wrong `self.phase` assignment or a malformed payload usually surfaces within a
  handful of games, not all 100).
- **Non-zero, plausible counters** — a clean run with `branch_a_chosen: 0` doesn't
  mean branch A is correct, it means it was never exercised, which random legal
  play can easily fail to hit (e.g. a size-class-gated option, or a rare empty-pool
  edge case). Print a count for every new action name and every new phase visited,
  and add inline `assert`s on the specific invariant you're changing (e.g. "every
  offered dice-pick payload must fit within the live pool", "this action should
  only appear when `defend_ship.size_class < attack_ship.size_class`") — that turns
  a silent wrong-but-legal-looking action into a hard crash you'll actually see.

This is cheap — seconds for 100+ games — and needs no MCTS or trained model at all,
since it only exercises `get_valid_actions()`/`apply_action()` directly.

## The action-space pipeline

This is the part that requires reading four files together, and it's exactly what
you'll be touching when you add phases/actions closer to the real ruleset. Get this
contract wrong and you get either a silent illegal-action bug or a `ValueError: No
valid actions available` crash mid-rollout — not a compile error.

**The four layers, in the order you edit them:**

1. **`armada_game/helpers/action_phase.py`** — `Phase` IntEnum (add new phases here),
   the `ActionType` type-alias documenting every `(action_name, payload)` shape, and
   `get_action_str()` for debug/visualization text.
2. **`armada_game/core/armada.pyx`** — the actual state machine:
   - `get_valid_actions()` (~line 126): given `self.phase`, returns the *exact* list
     of legal `(action_name, payload)` tuples for the current state.
   - `apply_action()` (~line 413): mutates state for one chosen action and sets
     `self.phase` to whatever comes next.
   - `update_decision_player()`: most phases are decided by
     `self.current_player` (the active player); attack-defense phases
     (`ATTACK_SPEND_DEFENSE_TOKENS`, `CHOOSE_DEFEND_DICE`, `ATTACK_RESOLVE_DAMAGE`)
     are decided by the *defender*, and `SHIP_PLACE_SQUAD` by the player who is not
     moving the ship (RRG Overlapping) — if you add a new phase decided by the
     non-active player, it must be added to that tuple or turns will silently go to
     the wrong player. Chance
     phases (`ATTACK_ROLL_DICE`, `DRAW_DAMAGE_CARD`) set `decision_player = 0`;
     every driver loop treats a falsy `decision_player` as "call
     `sample_chance_action()`", so that is the only place a new chance phase
     needs registering.
3. **`armada_game/helpers/action_space.py`** — a second, independent enumeration of
   the *canonical* (state-independent) action space per phase, used only to size the
   model's output and build `action_space.json`. This must generate the same
   `(action_name, payload)` shapes `get_valid_actions()` can produce, just over the
   full unconstrained domain instead of what's legal right now (e.g. `armada.pyx`
   only offers `resolve_damage_action` payloads legal for the current shields;
   `action_space.py` generates all of them). After editing, regenerate with
   `python -m armada_game.helpers.action_space` — nothing reads stale JSON
   automatically.
4. **`armada_game/core/action_manager.pyx`** — loads `action_space.json` and
   classifies every action *by name* into one of three index ranges: `ship_pointer`,
   `token_pointer`, or (the `else` branch) plain `static`. This is almost never a
   file you need to edit for a new action — a brand-new action name not in
   `ship_pointer_action_names`/`token_pointer_action_names` automatically falls into
   the static bucket with no special-casing needed.

**`learning/model/big_deep.py` usually needs zero changes.** `BigDeep.__init__`
iterates `Phase` and calls `action_manager.get_action_map(phase)`, auto-discovering
each phase's static-action width and building the stacked per-phase MLP
(`w1_stack`/`w2_stack`/`w3_stack`, sized by `max_static_action_space`) and the
phase→type lookup (`ship_pointer_phases`/`token_pointer_phases` around line 277)
entirely from that. A brand-new phase that's purely static — the common case — is
picked up automatically, same mechanism as `SHIP_DECLARE_TARGET`. You only need to
touch `big_deep.py` if you're introducing a genuinely new *pointer* category: an
action that selects among a variable-identity, variable-order collection of objects
that isn't already "which ship" or "which defense token" (e.g. "which squadron").

**Rule of thumb for flatten vs. pointer vs. phase-split**, worked out this session
while rebuilding the Redirect and Evade defense-token flows:

- **Flatten into one static action** when every axis of the decision is a fixed,
  canonical menu — hull section, dice color/face, damage amount. Example:
  `resolve_damage_action`'s `(hull, damage)` grid in `ATTACK_RESOLVE_DAMAGE`, or
  `declare_target_action`'s `(attack_hull, defend_hull)` grid. Cost is just output
  width, which is cheap (`SHIP_DETERMINE_COURSE` is 940 wide and sets
  `Config.MAX_ACTION_SPACE`'s static term on its own).
- **Use a pointer** when one axis selects among objects whose identity/order isn't
  canonical — which of up to 4 defense tokens, which of up to 10 ships. A fixed
  "slot 2" doesn't mean the same thing across two different ships' token loadouts,
  so a static head can't generalize across states the way a pointer (dot-product
  attention over the actual candidate embeddings) can.
- **Split into a new phase** (pointer step, then a follow-up static or pointer step)
  when a sub-choice is only legal/knowable *after* a prior pointer decision resolves,
  so it can't be flattened into one node. `CHOOSE_DEFEND_DICE` exists because you
  can't know "which die to reroll" until you know an evade token (a pointer pick)
  was actually spent. Contrast with Redirect's hull+damage choice, which doesn't
  need a phase split because it's resolved entirely within the already-static
  `ATTACK_RESOLVE_DAMAGE` phase.

Whichever shape you pick, the model has no idea which specific *values* are legal in
the current state — that's enforced entirely by `get_valid_actions()` plus
`shared_mcts.pyx`/`para_mcts.pyx`'s `_mask_policy`/`_expand`, which look up each
legal action's index via `action_manager.get_action_index()` and zero out everything
else before normalizing. Getting a new action into `get_valid_actions()` correctly is
what actually gates legality; getting it into `action_space.py` correctly just gates
whether the model has a slot to put a useful prior in.

**`get_valid_actions()` must never return the same `(name, payload)` twice.** Nothing
downstream deduplicates, and all three consumers break differently: `_expand` appends
one child per list entry, so identical children each carry the *full* prior for that
one `action_index` and collectively get N× the PUCT weight; `_add_root_noise` sizes
`alpha = DIRICHLET_ALPHA_SCALE / num_children` off the inflated count and draws
independent noise per copy; and `_get_final_action_probs` does
`action_probs[child.action_index] = child.visits` — an *assignment*, so copies
overwrite each other and the policy target keeps only the last one's share of visits
while the search split them across all of them. The easy way to introduce this is a
nested loop whose outer variable isn't in the payload — `SHIP_CHOOSE_TARGET_SHIP`
iterates attacking hull zones but its payload is only the target ship id, so it needs
an explicit `seen_target_ids` guard (the attacking hull is chosen later, in
`SHIP_DECLARE_TARGET`). When adding a phase shaped that way, assert
`len(actions) == len(set(actions))` in the rollout harness.

## The state chain, and where it deliberately leaves the RRG

The whole game is one graph over `Phase`, and the full set of reachable edges is
small enough to hold in your head. These are the natural edges; every one was observed
in random rollouts with squadrons deployed (153 distinct post-action transitions over
120 games, all consistent with this graph):

```
COMMAND_PHASE --assign_command_action--> COMMAND_PHASE (loop over every ship of both
    players, one dial at a time) --> SHIP_ACTIVATE

SHIP_ACTIVATE --> SHIP_REVEAL_COMMAND_DIAL --> SHIP_GAIN_COMMAND_TOKEN
    --{gain | gain_and_discard | pass}--> SHIP_RESOLVE_SQUAD
    --resolve_squad_command--> SQUAD_ACTIVATE (squadron command, see below)
    --pass_squad_command-----> SHIP_RESOLVE_REPAIR
    --{resolve_repair_command | pass_repair_command}--> SHIP_REPAIR_EFFECT
    --{repair_hull | recover_shield | move_shield}--> SHIP_REPAIR_EFFECT (self-loop)
    --pass_repair--> SHIP_CHOOSE_TARGET_SHIP

SHIP_CHOOSE_TARGET_SHIP --choose_target_ship--> SHIP_DECLARE_TARGET --declare_target-------> ATTACK_GATHER_DICE
                        --choose_target_squad-> SHIP_DECLARE_TARGET --declare_squad_target-> ATTACK_GATHER_DICE
                        --pass_attack--------> SHIP_DETERMINE_COURSE   (only when no target)

ATTACK_GATHER_DICE --> ATTACK_ROLL_DICE [chance] --> attack_info.phase
ATTACK_RESOLVE_EFFECTS --{spend_accuracy | resolve_confire_command}--> itself
                       --{use_confire_dial | use_confire_token | swarm_reroll}--> ATTACK_ROLL_DICE
                       --pass_attack_effect--> ATTACK_SPEND_DEFENSE_TOKENS
ATTACK_SPEND_DEFENSE_TOKENS --spend_defense_token--> itself (brace/redirect)
                            --spend_defense_token--> CHOOSE_DEFEND_DICE (evade)
                            --pass_defense_token--> ATTACK_RESOLVE_DAMAGE
CHOOSE_DEFEND_DICE --{spend_evade_dice | discard_evade_dice}--> ATTACK_ROLL_DICE (close/medium
                       reroll) | ATTACK_SPEND_DEFENSE_TOKENS (long cancel)
ATTACK_RESOLVE_DAMAGE --resolve_damage--> DRAW_DAMAGE_CARD [chance] | after_attack_damage()
after_attack_damage() --> ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET (a ship attacked a squadron)
                        | finish_ship_attack() | finish_squad_attack()
ATTACK_SHIP_ADDITIONAL_SQUADRON_TARGET --declare_additional_squad_target--> ATTACK_GATHER_DICE
                                       --pass_additional_squad_target----> finish_ship_attack()
finish_ship_attack()  --> SHIP_CHOOSE_TARGET_SHIP (attack_count < 2) | SHIP_DETERMINE_COURSE
finish_squad_attack() --> SQUAD_MOVE (command, move step left, not restricted) | end_squad_activation()

SHIP_DETERMINE_COURSE --resolve_nav_command--> itself (phase deliberately unchanged)
                      --determine_course----> SHIP_PLACE_SQUAD (it overlapped squadrons)
                                            | DRAW_DAMAGE_CARD [chance, per asteroid]
                                            | end_ship_activation()
SHIP_PLACE_SQUAD --place_squad--> SHIP_PLACE_SQUAD (next overlapped squadron)
                                | DRAW_DAMAGE_CARD [asteroids] | end_ship_activation()
end_ship_activation() --> SHIP_ACTIVATE (opponent, else self) | start_squadron_phase()

SQUAD_ACTIVATE --activate_squad_move----> SQUAD_MOVE --move_squad--> SQUAD_DECLARE_TARGET (command)
                                                                   | end_squad_activation()
               --activate_squad_attack--> SQUAD_DECLARE_TARGET
               --pass_activate_squad----> SHIP_RESOLVE_REPAIR   (squadron command only)
SQUAD_DECLARE_TARGET --{squad_attack_squad | squad_attack_ship}--> ATTACK_GATHER_DICE
                     --pass_attack_squad--> SQUAD_MOVE | end_squad_activation()
end_squad_activation() --> SQUAD_ACTIVATE (command count left) | SHIP_RESOLVE_REPAIR (command over)
                         | next_squadron_turn()
start_squadron_phase(), next_squadron_turn() --> SQUAD_ACTIVATE | status_phase()
status_phase() --> COMMAND_PHASE (round+1) | game over (after round 6)
any action destroying a fleet's last ship --> game over at once (check_fleet_elimination)
```

`DRAW_DAMAGE_CARD` resumes whichever tail suspended it via `damage_draw_source`, so
it re-enters `after_attack_damage()` or `end_ship_activation()` — it is never a dead
end. `SHIP_PLACE_SQUAD` stashes pending asteroid draws in the same `damage_draw_*`
fields and resumes them once the last squadron is placed.

**Observed transitions usually skip edges.** A step whose only legal action is a
pass-like no-op is never visited: `apply_action()` resolves it inline (see *Forced
single-action steps* below), so `declare_target_action` typically lands straight on
`ATTACK_ROLL_DICE`, and one `reveal_command_action` can run through four forced steps
to `SHIP_DETERMINE_COURSE`.

**Deliberate deviations from the RRG in this chain.** These are choices, not bugs;
do not "fix" one without saying so:

- **Attacking is mandatory whenever a legal target exists.** `pass_attack` is offered
  only when `SHIP_CHOOSE_TARGET_SHIP` produced no target at all. The RRG says a ship
  performs *up to* two attacks. Declining an attack is close to strictly dominated in
  this simplified fleet (no salvo, no counter, no crit downside), and forcing it
  prunes a large branch. One attack is still reachable: attack once, then the second
  `SHIP_CHOOSE_TARGET_SHIP` visit offers `pass_attack` once the remaining hull zones
  have no target.
- **Repair spend is forced once the command is resolved.** `forced =
  repair_hull_legal or recover_shield_legal`. The RRG permits resolving a command and
  choosing to produce no effect. Since the dial/token commitment is now explicit and
  already paid at that point, declining the spend is pure loss. Note `move_shield`
  alone never forces, because `forced` is computed before those actions are appended
  — so a 1-point budget still offers `pass_repair`.
- **`repaired_hull` / `reduced_hull` block shield churn.** You cannot move a shield
  out of a zone you just filled, or into a zone you just drained, and cannot recover
  into a drained zone. The RRG has no such restriction ("A ship can pay for and
  resolve each effect multiple times"); this exists purely to kill A→B→A loops.
- **Redirect picks its hull zone at `ATTACK_RESOLVE_DAMAGE`, not at token spend.**
  The RRG chooses the adjacent zone when the token is spent and only the amount later.
  The engine defers both, which is strictly more permissive for the defender and
  keeps the whole choice inside one static action. `game_encoder.pyx` has a matching
  comment where the old pre-chosen-hull feature slot now stays zero.
- **Spending a dial for a command token you already hold is not offered.** The RRG
  allows it (you gain, then immediately discard the duplicate) purely as a way to
  waste a dial.
- **Squadron attacks stay optional** (a Squadron Phase activation is move *or*
  attack, and forcing the attack would delete repositioning), but the mandatory
  ship-attack rule above now covers squadron targets too: counter never triggers on a
  ship, so declining an anti-squadron attack is still dominated.
- **The one-die obstructed pruning applies to squadron attacks as well**, ship or
  squadron target, for the same reason as for ships.
- **Squadron moves that leave the play area are not offered.** The RRG allows them
  and destroys the squadron; that is strictly dominated by staying put.
- **Squadron move destinations are grid cells** (33x33 absolute grid, 18.47 mm
  pitch). The reach bound itself is exact (TODO.md G1).
- **Overlapped squadrons are placed in squadron-id order** (RRG: any order), into 28
  discrete slots touching the ship. Ring 1 is filled greedily, so the RRG's global
  "as many as possible touching" is not guaranteed when the ring is congested. A
  squadron with no touching slot is placed automatically, touching a squadron that
  touches the ship, nearest to where it was; with no legal spot at all it is
  destroyed.
- **Two dominance prunes for the squadron command**: resolving it is offered only
  when a friendly squadron is eligible, and `COMMAND_PHASE` offers no squadron dial
  to a player with no squadrons left.
- **No pass tokens.** A player only passes when they have no unactivated ships.
- **Accuracy die color is not a choice** — red is spent first when both colors show
  an accuracy icon. Spending removes the die either way, so damage is unaffected.
- **Command dials are assigned sequentially and in the open**, not secretly and
  simultaneously. This is the one parked information-set problem, not a simplification
  to be papered over; it needs dual per-player MCTS.

**RRG rules that *are* implemented faithfully**, and are easy to break by accident:
the obstructed attack removes exactly one die before rolling regardless of how many
obstacles are crossed; brace halves the total (round up) *before* redirect subtracts
from it; the extra evade die against a larger attacker discards the spent token; the
standard critical adds no damage and is skipped when the attack destroys the
defender; `move_ship()`'s recursion implements the RRG's overlap speed-reduction loop
and deals one facedown card to the mover and the closest ship overlapped; a ship
whose base leaves the play area is destroyed; yaw `II` cannot be increased, which
falls out of the `abs(modified_yaw) <= 2` cap in `get_all_possible_courses()`; and a
tie after six rounds goes to the second player. The game ends **immediately** when a
fleet's last ship is destroyed, and if both fleets' last ships go in the same action
the higher score wins, tie to the second player (`check_fleet_elimination()`, run at
the end of any action in which a ship was destroyed).

Squadron rules that are easy to break: engagement is distance 1 **and** unobstructed
line of sight, and heavy squadrons still engage (they only waive the move/attack
restriction); a squadron may attack any enemy squadron at distance 1, engaged or not
(FAQ); escort restricts only squadrons *engaged with* it; swarm needs the defender
engaged with a squadron other than the attacker; criticals for squadrons come only
from bomber against a ship; a ship attacks a squadron with its anti-squadron dice
filtered by range colour, exactly as against a ship; touching is not overlapping
(FAQ); and a squadron sitting on an obstacle traces line of sight from a point inside
it, so it is always obstructed and can never be engaged.

There is **no `CONTAIN` or `SALVO` token type** in `TokenType`, and no ship in
`ship_dict.json` carries `SCATTER` — the live loadouts are brace/redirect/evade only.
So the RRG's contain-blocks-the-standard-critical interaction is genuinely out of
scope, not a missing check.

## Forced single-action steps

MCTS spends a full network evaluation on every decision node, including one with a
single legal action, and `_expand` applies *every* legal action to build child
snapshots. So `apply_action()` ends by asking `forced_pass_action()` whether the
new step's only legal action is a pass-like no-op (`FORCEABLE_ACTIONS` in
`armada.pyx`: the `pass_*` actions plus `('gather_dice_action', (0, 0, 0))` and
`('resolve_damage_action', None)`), and if so applies it recursively. The chain stops
at a chance node, a real choice, or the end of the game. Such states never reach the
encoder or the replay buffer.

The contract when you touch a phase:

- `get_valid_actions()` and `forced_pass_action()` share the per-phase list builders
  (`repair_effect_actions()`, `attack_effect_actions()`, ...), so they cannot drift.
  Where a cheap test stands in for a builder (a held-token membership check, or "no
  eligible squadron" for the squadron command), its equivalence is argued in a
  comment.
- **`forced_pass_action()` runs inside `_expand` once per child.** Keep each check
  cheap. Never call `get_valid_actions()` from it for a phase that can have hundreds
  of children — a `SQUAD_MOVE` node expands up to 861.
- A new phase whose pass can be the only legal action must get a branch in
  `forced_pass_action()`, or it silently costs MCTS a network evaluation.
- `Armada.verify_forced = True` (testing only) cross-checks every decision against
  `get_valid_actions()` in both directions. Run the rollout harness with it on after
  touching any of this.

Forced steps that are *not* pass-like stay visible on purpose: the always-single
`SHIP_REVEAL_COMMAND_DIAL`, and a lone `CHOOSE_DEFEND_DICE`, `SHIP_DECLARE_TARGET` or
`SHIP_PLACE_SQUAD` option.

## Squadrons: geometry and the per-state cache

Every squadron rule reduces to exact closed forms, proven and numerically checked in
`TODO.md` (G1-G19). The core is `armada_game/core/squad_geometry.pxd`, a
header-only set of `nogil` inline functions with no `.pyx`. It holds a ship base or
token as a rotated rectangle centred on the ship's hashed pose, which matches
`_ship_coordinate()` to ~5e-5 mm. Disk-vs-rectangle overlap is a clamped local
distance, and segment tests use Liang-Barsky or orientation predicates. All
squadron↔ship overlap goes through it, with one tolerance (`SQUAD_TOUCH_EPS`), because
touching is not overlapping.

`Armada` caches derived geometry per game state, never snapshotted:

- per-ship rectangles (`ensure_ship_geometry()`);
- per-squadron bitmask rows of "enemies at distance 1" and "enemies engaged"
  (`ensure_squad_row()`, built lazily, one row at a time);
- a pair line-of-sight dict shared by both rows of a pair.

**Any code that moves or destroys a ship or squadron must call
`game.invalidate_geometry()`**: `Squad.move_by_cells/place/destroy`, `Ship.destroy`,
`Ship.execute_maneuver`, `deploy_*`, `place_obstacle` and `revert_snapshot` all do. A
missed call means stale engagement, not a crash. The rollout harness checks
engagement against a shapely reference on every state, so run it after any such
change.

Squadron *activation* state lives on `Armada`, not `Squad`: `active_squad`,
`squad_can_move`, `squad_can_attack`, `squad_activation_count`, plus
`pending_placement` for overlapped squadrons. A `Squad` snapshot is only
`(hull, coords, activated)`; `destroyed` is derived from the hull. Keep it that lean:
`_expand` snapshots every child.

## Guardrail: `big_deep.py` and `game_encoder.pyx`

**Do not modify `learning/model/big_deep.py` or `learning/encoding/game_encoder.pyx`
without first asking the user for the specific change wanted, even if a task seems
to imply one is needed.** This is a standing workflow rule, not a caution about
checkpoint compatibility — the model is being retrained from a zero base, so
existing `model_checkpoints/*.pth` files are not a constraint and shape/layout
changes to either file are fine to make freely once the change is actually
specified. The reason to stop and ask first is that these two files have to stay in
exact sync with each other and with `Config`'s feature-size constants, and it's easy
to get that subtly wrong without a precise spec. If a request would touch either
file and the exact change hasn't been spelled out, stop and ask rather than
inferring or designing it yourself.

## `AttackInfo` — the in-progress-attack context

`armada_game/core/attack_info.pyx` (`.pxd` for the Cython attribute declarations) is
the scratch object that lives for the duration of one attack, across all the
`ATTACK_*` and `CHOOSE_DEFEND_DICE` phases: attacker/defender ids and hull sections,
range, obstruction, the live `attack_pool_result` dice pool, `spent_token_indices` /
`spent_token_types`, critical, and `total_damage`. Most of the interesting state for
an in-progress attack decision lives here, not on `Ship`. When a phase needs to know
"which token was just spent" without a dedicated field for it, the established
pattern (see `CHOOSE_DEFEND_DICE`) is reading `spent_token_indices[-1]`, since a
token-spend always synchronously transitions into whatever phase resolves it.

`remove_dice()` and the phase-transition handlers in `armada.pyx` call
`calculate_total_damage()` for you at the right points — don't call it redundantly.

## Damage cards and the `DRAW_DAMAGE_CARD` chance node

`Ship.hull` is still the authoritative "hull remaining" counter, so
`max_hull - hull` is the total number of damage cards a ship holds.
`Ship.damage_cards` is a 22-tuple counting the **faceup** cards by type;
facedown cards are anonymous and **derived**, never stored
(`Ship.facedown_damage_count()` = `max_hull - hull - sum(damage_cards)`). That is
deliberate: four separate sites decrement `hull` (`defend()`, `overlap_damage()`,
`deal_faceup_damage_card()`, and `discard_damage_card()` in reverse), and a stored
facedown counter would desync the first time one of them was missed.

Card types are **anonymous indices 0-21** with no effects implemented; only their
copy counts matter (`DAMAGE_CARD_COPIES` in `armada_game/helpers/damage_deck.py`:
type 6 has 4 copies, type 21 has 8, the rest 2, summing to 52).

**Two sources deal a faceup card, and they are not the same operation:**

- **Standard critical** (`resolve_damage_action`): any crit icon in the *final*
  attack pool, when the attack deals at least one card and the attacker is a ship or
  a squadron with bomber (squadrons cannot otherwise resolve critical effects, RRG
  Critical Effects). It adds **no** damage —
  it only reclassifies one card the attack already dealt, so
  `flip_damage_card_faceup()` must not touch `hull`. (The old
  `if critical == Critical.STANDARD: self.hull -= 1` in `Ship.defend()` was a
  wrong reading of this rule and has been deleted.) Skipped entirely when the
  attack destroys the defender.
- **Asteroid overlap** (`determine_course_action`): a genuinely new card, so
  `deal_faceup_damage_card()` decrements `hull` and can destroy the ship. One
  draw per asteroid overlapped, resolved sequentially.

**Deck model** (deliberate deviations from the RRG, which has one shared deck):
each player has their own 52-card deck, and sampling excludes only the faceup
cards on that player's **surviving** ships (`Armada.get_dealt_damage_counts()`).
Facedown cards count as still in the deck, and there is no discard pile —
repaired and destroyed-ship cards silently return to circulation.

**Chance-node plumbing.** `Phase.DRAW_DAMAGE_CARD` mirrors `ATTACK_ROLL_DICE`:
`decision_player = 0`, `get_valid_actions()` raises, and `action_space.py` emits
an empty list so `BigDeep` skips the phase automatically. `Armada` carries
`damage_draw_ship_id` / `damage_draw_count` / `damage_draw_source` (all in the
snapshot) because the draw suspends a step that must be resumed — hence the two
extracted tails `after_attack_damage()` and `end_ship_activation()`, which the draw
handler dispatches to on `damage_draw_source`. The attack case deliberately keeps
`attack_info` and `defend_ship` alive across the chance node so
`after_attack_damage()` can still read them (and route a bomber squadron's attack
back to its activation).

`Armada.sample_chance_action()` is the single sampler for *every* chance phase;
`rollout()`, `alpharmada.py`, both `evaluation/` drivers, the matchup notebook, and
both MCTS `_select` loops all call it, and MCTS matches chance children on the full
action tuple. Adding another chance phase means extending that one method plus
`update_decision_player()` — no driver loop should ever test a phase by name again.

**Repair Hull** now chooses its target: `('repair_hull_action', card_index)`, where
0-21 discards that faceup type and `FACEDOWN_CARD_INDEX` (22) discards an anonymous
facedown card. `get_valid_actions()` only offers cards the ship actually holds.

## Obstacles, line of sight, and the spatial planes

All six ordinary obstacles are placed every game and they obstruct attacks. Only
the **asteroid** overlap effect is implemented: ending a maneuver on an asteroid
deals one faceup damage card per asteroid overlapped, resolved through the
`DRAW_DAMAGE_CARD` chance node from `determine_course_action` (see the damage-card
section). Because that needs a chance node it cannot go through
`Ship.overlap_obstacle()`, which now raises `NotImplementedError` — its station and
debris branches remain dead code, so a ship parked on a station or debris field
still pays none of the RRG cost while hiding that obstacle from its own line of
sight (see below).

**Placement** (`get_random_obstacle_layout()` in `armada_game/helpers/setup_game.py`)
builds a complete layout by rejection sampling before mutating the game, then raises
rather than starting a game with fewer than six. The zone is the board inset by
distance 3 on **all four** edges, which is what the RRG prescribes here: for a 3' x 3'
play area the whole play area is the setup area, and obstacles must be beyond
distance 3 of its edges and beyond distance 1 of each other. Do not "restore" a
distance-5 short-edge inset — that is the 6' x 3' rule and does not apply until the
board grows. Random order, orientation, and mirroring stand in for the alternating
player choices, which the engine has no setup action phase for.

**Obstruction** (`Ship.is_obstruct_s2s()`) resolves one trimmed segment and feeds it
to both loops:

- `cache.visible_los_segment_s2s(attacker_state, from_hull, defender_state, to_hull)`
  returns the LOS trimmed to the part visible outside both endpoint **tokens**. The
  token is the opaque geometry; the larger transparent base hides nothing, so an
  obstacle under a ship's base still obstructs while one under its token does not.
- A third ship obstructs when the line crosses its full **base**, not its token.
  Trimming cannot hide a third ship, because a token is strictly inside its own base
  and bases never overlap — so the same segment is correct for both checks.

Key that cache on ship states plus hull indices, never on the targeting points. The
caller then never builds the raw point tuple, and building plus hashing that tuple
dominated the call: passing the segment in instead is about ten times slower per
call, and `is_obstruct_s2s()` runs ~1,440 times per `encode_relation_matrix()`.
`visible_los_segment_s2s()` carries a `NOTE:` about the one case it gets wrong
(overlapping ship tokens, currently unreachable) — read it before reusing the
function anywhere ships can be placed without a base-overlap check.

`get_valid_ship_target()` / `get_valid_target_hull()` prune declarations whose pool is
a single die when the attack is obstructed, since obstruction would remove that die
and leave nothing. That is a deliberate training simplification, not the RRG, and it
now fires on roughly 7% of otherwise-legal declarations.

**Spatial planes.** `Config.SPATIAL_CHANNELS = OBSTACLE_SPATIAL_OFFSET (10) +
OBSTACLE_SPATIAL_CHANNELS (3)`: plane 0 is ship presence, 1-9 threat geometry, and
10-12 are station / debris / asteroid occupancy indexed by `ObstacleType` value. The
obstacle planes are global, written identically into every ship slot, and `BigDeep`
collapses them with `amax(dim=1)`. Every consumer sizes itself from `Config`
(`game_encoder.pyx`, `big_deep.py`, `disk_manager.py`, `alpharmada.py`, both MCTS
`.pyx`), and `encode_game_state()` asserts the channel count, so change `Config` and
rebuild rather than editing shapes by hand — subject to the `big_deep.py` /
`game_encoder.pyx` guardrail above.

Two things that must stay in sync with the board size: `width_step` in
`cache_function.py` is `LONG_RANGE * 3 / width_res` and has to match
`Armada.player_edge`, because the rasterized planes and `BigDeep`'s `grid_sample()`
gather over normalized `ship_coords` share that scale — a mismatch silently offsets
scatter from gather. And any change to the spatial layout invalidates existing replay
chunks: `ArmadaChunkDataset._load_chunk()` memmaps with the current `Config` shape and
raises on a stale file, so purge `replay_buffers/` and the remote chunks.

## Cython gotchas hit while extending this file

- **Generator expressions inside `cpdef` functions don't compile**: `any(x for x in
  y)` fails with `closures inside cpdef functions not yet supported`. Use a plain
  `for` loop or a list comprehension (comprehensions are fine; genexprs are not).
- **`cdef class` attributes are resolved at compile time**, including inside code
  that's unconditionally unreachable (e.g. guarded by `raise NotImplementedError` at
  the top of a branch, a pattern used throughout `armada.pyx` for the
  simplified-away squadron/command mechanics). Renaming or removing a `.pxd`-declared
  attribute breaks every reference to it, live or dead — Cython still type-checks
  code Python would never execute. Check `attack_info.pxd`/`ship.pxd`/etc. before
  assuming a "dead" branch is safe to leave alone.
- **numba's `cache=True` freezes module globals at compile time** and invalidates
  only when its *own* source file changes. A constant imported from another module
  (`Q2S_RANGE`, `SQUAD_BASE_POLY` from `measurement.py`) can stay stale on disk.
  After changing such a constant, delete `armada_game/helpers/__pycache__/*.nbi`
  and `*.nbc`.
- **Fixed-size C arrays need a compile-time size.** Use a `cdef enum` constant and
  assert it against the `measurement.py` value at import — `armada.pyx` does this for
  the 33x33 move grid (`MOVE_GRID_N`).
- **Squadron masks are `long long` bitmasks**, so `deploy_squad()` refuses a 64th
  squadron rather than silently overflowing.
- **The scalar encoding writes `26 + len(Phase)` slots with `boundscheck(False)`.**
  `SCALAR_FEATURE_SIZE = 64` leaves headroom (48 used at 22 phases). Adding phases
  past it overruns the buffer silently.
- The IDE's inline Cython diagnostics flag a lot of `cdef public object` /
  `cdef public bint` operator/assignment "errors" that are pre-existing noise (the
  linter doesn't model Cython's Python-object fallback for `object`-typed
  attributes) — treat `python tools/cython_setup.py build_ext --inplace` as the
  actual source of truth, not the inline diagnostics.
