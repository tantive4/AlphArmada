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

Squadrons are present in the code but disabled in the active config
(`Config.MAX_SQUADS = 0`). Many of their handlers in `armada.pyx` are stubbed with
`raise NotImplementedError(f"simplified ...")` guarding otherwise-dead code — see the
Cython gotcha below before touching anything near those. Obstacles are live: all six
are placed at setup and they obstruct line of sight, but their overlap effects stay
disabled (see the obstacle section below). Command stacks
(`Config.MAX_COMMAND_STACK = 3`) are live: ships assign a command dial per round in
`COMMAND_PHASE` (FIFO queue sized to `command_value`, fixed assignment order —
highest point cost first, tie-break by hull then id) and reveal the top of the stack
at the start of their own activation. All three live commands resolve fully,
from a dial and/or a held token: nav (`Ship.nav_command_used()`), repair
(`Ship.repair_command_used()` + `Phase.SHIP_RESOLVE_REPAIR`) and con-fire
(`resolve_confire_command_action` + `use_confire_dial_action` /
`use_confire_token_action` in `ATTACK_RESOLVE_EFFECTS`). Nav and repair derive
which resources were spent *post-hoc* from what the ship actually did;
con-fire instead pre-commits via an explicit resolve action and is one-shot
per activation — see `TODO.md` for why that asymmetry is deliberate. Squad
command remains stubbed, blocked on squadrons.

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
from armada_game.helpers.dice import roll_dice

N_GAMES = 100
counters = {"my_new_phase_visits": 0, "branch_a_chosen": 0, "branch_b_chosen": 0}
errors = 0

for i in range(N_GAMES):
    game = setup_game(debuging_visual=False)
    step = 0
    try:
        while game.winner == 0.0:
            if game.phase == Phase.ATTACK_ROLL_DICE:
                # chance node, not a player decision — sample it directly
                action = ('roll_dice_action', roll_dice(game.attack_info.dice_to_roll))
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
   - `update_decision_player()` (~line 112): most phases are decided by
     `self.current_player` (the active player); attack-defense phases
     (`ATTACK_SPEND_DEFENSE_TOKENS`, `CHOOSE_DEFEND_DICE`, `ATTACK_RESOLVE_DAMAGE`)
     are decided by the *defender* — if you add a new defender-side phase, it must be
     added to that tuple or turns will silently go to the wrong player.
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
  width, which is cheap (`SHIP_DETERMINE_COURSE`'s course space alone is ~937 wide).
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

## Obstacles, line of sight, and the spatial planes

All six ordinary obstacles are placed every game and they obstruct attacks. Their
*overlap* effects are still disabled — the `active_ship.overlap_obstacle(obstacle)`
call in `armada.pyx` is commented out, so `Ship.overlap_obstacle()` never runs. Be
aware of the training artifact that creates: a ship parked on an obstacle hides it
from its own line of sight (see below) and pays none of the RRG damage for doing so,
so sitting on an asteroid is currently a free advantage.

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
- The IDE's inline Cython diagnostics flag a lot of `cdef public object` /
  `cdef public bint` operator/assignment "errors" that are pre-existing noise (the
  linter doesn't model Cython's Python-object fallback for `object`-typed
  attributes) — treat `python tools/cython_setup.py build_ext --inplace` as the
  actual source of truth, not the inline diagnostics.
