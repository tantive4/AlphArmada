# Face-up damage implementation audit

Date: 2026-08-22  
Scope: the implementation described in `TODO.md`, checked against `armada_game/rules/Rule Reference 1.6.md` and the simplified choices supplied for this audit.

## Verdict

**Core gameplay result: PASS under the stated simplifications.** Standard critical damage, asteroid damage, repair choices, deck sampling, state copying, encoding of held face-up cards, and MCTS chance handling all worked in focused and randomized tests.

**Repository-readiness result: NOT READY AS-IS.** Two concrete integration defects should be resolved before accepting the change:

1. `armada_game/helpers/damage_deck.py` is untracked even though the modified compiled code imports it. A commit or deployment that omits this file will fail to build or import.
2. The encoder identifies the wrong active ship during reachable asteroid `DRAW_DAMAGE_CARD` states. This is hidden from the current MCTS path because chance nodes are sampled without neural-network evaluation, but direct state encoding is incorrect and the unchecked Cython cast is unsafe.

No implementation source was changed during this audit. This report is the only file added.

## Findings

### F1 — High: required damage-deck module is untracked

- `armada_game/helpers/damage_deck.py` is currently reported by Git as `??`.
- The changed Cython implementation imports its constants and sampler.
- The working tree runs because the file exists locally; it is not part of the tracked patch unless explicitly added.
- Consequence: a clean checkout of the patch can fail during Cython generation/build or at import time.

### F2 — Medium: asteroid chance-state encoding uses ship ID 0

- `learning/encoding/game_encoder.pyx:99` selects `defend_ship.id` whenever `decision_player != current_player`.
- During an asteroid `DRAW_DAMAGE_CARD` decision, `decision_player == 0`, `current_player` is `1` or `-1`, `defend_ship` is `None`, and `active_ship` is the ship receiving damage.
- Reproduction with actual active ship IDs 1 and 2 encoded ID 0 in both cases. The resulting header was `(0, 10, 14)` rather than the active ship's ID.
- The current MCTS implementations bypass model encoding at chance nodes, so normal self-play did not crash. Any caller that encodes every state, debugger/training tooling that observes the pending draw state, or a future MCTS change can receive corrupt state information.

### F3 — Medium: `TODO.md` contains mutually incompatible asteroid requirements

- The face-up damage section and verification checklist say asteroid effects are enabled, including at speed 0 (`TODO.md:19-22`, `38`, `77-80`).
- The older obstacle section says overlap effects remain disabled and specifically says a speed-0 asteroid causes no damage (`TODO.md:106`, `108`, `122`, `140`).
- The implementation follows the newer damage section: asteroid overlap deals a face-up card, multiple asteroids resolve sequentially, and speed 0 still triggers the effect.
- This behavior matches the RRG: asteroid damage is at line 765, multiple obstacle effects are resolved one at a time at line 813, overlap effects occur after executing a maneuver at line 889, and a speed-0 maneuver is still executed at line 1179.

### F4 — Medium: existing tracked checkpoints are incompatible

- `load_recent_model()` fails on the repository's tracked `model999` checkpoint with missing/unexpected keys and a spatial-convolution input mismatch (`36` versus `39` channels).
- Some mismatch is broader than this feature, but this patch also changes phase buffers, feature meaning/offsets, and repair action indices. Old compatible checkpoints/replays would therefore still require an explicit migration or fresh initialization.
- `TODO.md` accounts for purging replay buffers but does not state the checkpoint policy.
- A fresh current model passes forward inference and full-game evaluation.

### F5 — Low: automatic standard critical is not recorded in `attack_info.critical`

- The attack resolver detects a qualifying critical and schedules a standard critical draw, but it does not set `attack_info.critical = Critical.STANDARD`.
- The draw still occurs correctly through `damage_draw_source`, so gameplay is unaffected.
- During the pending draw, the snapshot/critical scalar says no critical effect was selected. That makes observability and encoded semantics inconsistent with the action being resolved.

### F6 — Low: damage-count feature scaling can exceed 1

- The 22 held face-up damage counts are encoded at offsets 81–102 and divided by `2.0`.
- Twenty card types have two copies and therefore encode at most `1.0`, but the four-copy type can encode `2.0` and the eight-copy type can encode `4.0`.
- This may be intentional raw multiplicity scaling, but it differs from the apparent normalized range used by most nearby features.

### F7 — Low, unreachable through legal action generation: invalid repair partly mutates state

- Repair cost/budget is consumed before `discard_damage_card()` validates the selected card.
- Calling the public repair application with an unavailable face-up type raises `ValueError` after engineering budget becomes 0 and engineering spent becomes 3.
- The generated repair menu only exposes valid choices, so normal play is safe. This is still a non-atomic API failure mode.

### F8 — Low, accepted scope exception: destroyed-ship damage helpers expose phantom cards

- `Ship.destroy()` clears damage tracking and sets hull to 0.
- `facedown_damage_count()` then derives `max_hull - hull - faceup_count`, making a destroyed ship appear to hold its full hull value in face-down cards.
- A direct discard call can consequently increase a destroyed ship's hull while it remains destroyed.
- Legal game flow skips destroyed ships, and the supplied rules explicitly permit ignoring destroyed ships' cards, so this is not counted as a failure for this implementation.

### F9 — Low, unreachable through internal callers: sampler accepts malformed deck tuples

- The sampler accepts a short tuple, silently ignores entries beyond 22 in a long tuple, and accepts negative counts.
- The internal game getter always supplies the exact 22 non-negative counts, so legal play is unaffected.

### F10 — Documentation evidence gap: the local RRG does not define the 22-type copy map

- The local RRG specifies damage-card behavior and a damage deck, but it does not enumerate all 22 card identities or their multiplicities.
- The implementation uses 52 cards: twenty types ×2, one type ×4, and one type ×8. The total of 52 is supported by the [official core-set component listing](https://www.atomicmassgames.com/product_type/star-wars-armada-core-set/); the detailed multiplicities are corroborated by the community [Armada damage-deck listing](https://starwars-armada.fandom.com/wiki/Damage_Deck), not by the checked-in RRG itself.
- Because card text effects are deliberately omitted, anonymous stable indices are sufficient for the current feature.

## Rule and implementation checks

| Requirement | Result | Evidence |
|---|---:|---|
| Only the standard critical effect is used | Pass, subject to Question 1 | A qualifying final attack pool schedules exactly one face-up draw. |
| No damage-card text effects | Pass | All 22 types behave identically except for their tracked index. |
| Critical eligibility uses the final attack pool | Pass | Brace/redirect/shield tests preserved the critical result correctly; a canceled/no-critical pool did not draw. |
| Black, blue, and red critical icons are recognized | Pass | Every color's critical face was exercised; non-critical black was also checked. |
| Standard critical changes the first dealt hull card to face-up | Pass | Shielded damage did not consume a card; unshielded damage produced one face-up card and remaining hull cards face-down. |
| Multiple critical icons do not create multiple face-up cards | Pass | Multi-critical pools still drew exactly one face-up card. |
| Lethal attacks need not draw a critical card | Pass as an approved deviation | The implementation stops when the ship is destroyed. |
| Per-player deck accounts for surviving face-up cards only | Pass | Fresh 52-card sampling, depletion, repair return, snapshot, and deepcopy were verified. Face-down, destroyed, and discarded cards are ignored as directed. |
| Deck exhaustion is handled | Pass | Forced sequential draws consumed exactly all 52 cards; the next draw raised the expected error. |
| Asteroid overlap deals one face-up card and bypasses shields | Pass, subject to Question 2 | Single, repeated, mixed-obstacle, lethal, and speed-0 cases passed. |
| Multiple asteroid effects resolve sequentially and stop on destruction | Pass | Two-asteroid and lethal-first-obstacle cases passed, including through MCTS. |
| Debris field and station effects remain disabled | Pass | Neither scheduled a damage-card draw. |
| Engineering may discard a selected face-up or face-down card for 3 points | Pass | All 22 face-up choices plus the face-down sentinel were tested; the legal action menu contained exactly the held choices. |
| Draw decisions are chance nodes rather than player choices | Pass | Both MCTS implementations sampled legal weighted outcomes without exposing a player choice. |
| State snapshots/deepcopies preserve pending draws and deck state | Pass | Attack and obstacle pending states round-tripped correctly and copies diverged independently. |
| Damage-card counts are encoded | Pass, subject to Questions 3 and 4 | All 22 offsets updated and repaired correctly; the active-ship header defect is F2. |
| Action-space/config updates are internally consistent | Pass | 15 phases, 40 repair actions, maximum phase-action count 937, and global maximum action size 951 matched the generated JSON. |
| Current neural model accepts the new shapes | Pass | Finite outputs: policy `(1,951)`, value `(1,1)`, raw-point `(1,2)`, hull `(1,10)`, game-length `(1,6)`. |
| Contain and Bomber interactions | Not reachable in the current ruleset | No Contain token exists in the enum/rosters and `MAX_SQUADS == 0`; see Question 6. |

## Verification performed

No persistent test files were present or created. `pytest` is not installed, so the checks were run with isolated Python/Cython harnesses and temporary build directories.

- **156 focused assertions passed** across constants, forced outcomes, exact deck exhaustion, weighted sampling, ship primitives, attack criticals, shields, redirects, brace, repair, action JSON, chance APIs, snapshots, deepcopy, encoder features, and neural inference.
- **14 focused obstacle assertions passed** for one/two asteroids, lethal stopping, station/debris suppression, mixed obstacles, and speed-0 repeat overlap.
- **All 22 damage types** were individually forced, drawn, encoded, and repaired.
- **Distribution check passed:** 104,000 fresh-deck draws had a maximum absolute frequency error of `0.001615` from the configured weights.
- **Two independent randomized campaigns passed:** 200 complete games, 78,248 actions, 851 face-up draws, all 22 card types observed, and zero exceptions or invariant failures.
  - Campaign A: 39,582 actions; 433 draws (241 critical, 192 asteroid); 37 face-up and 95 face-down repairs.
  - Campaign B: 38,666 actions; 418 draws (208 critical, 210 asteroid); 44 face-up and 81 face-down repairs.
- **30 engine rollout games passed** without exceptions.
- **Both parallel and shared MCTS passed** focused attack and consecutive-asteroid chance-node tests; returned policies were finite, normalized, and length 951.
- **Fresh-model evaluation passed** two full games at a one-iteration smoke setting. That artificial setting emitted a pre-existing zero-visit normalization warning; it did not affect the face-up-damage checks.
- **Out-of-tree Cython builds passed:** all ten extensions compiled/imported, including forced C regeneration. Temporary output was used so generated repository files were not touched.
- **Action-space regeneration matched** the checked-in `action_space.json` byte-for-data in a temporary destination.
- Driver imports and static parsing passed. The destructive/external worker constructor was not invoked; equivalent engine, MCTS, encoding, and evaluation paths were exercised directly.

## Questions requiring a decision

1. **Is the standard critical mandatory whenever the final pool contains a critical icon, or may the attacker decline it?** The RRG says the attacker *can* resolve a critical effect (`Rule Reference 1.6.md:273-283`). I interpreted “no critical effect choice: just standard critical” as automatic/mandatory, which matches the implementation.
2. **Which asteroid contract in `TODO.md` is authoritative?** The newer face-up-damage section enables asteroid damage, including at speed 0; the older obstacle section disables all overlap effects and explicitly expects no speed-0 damage. The code and RRG match the newer section.
3. **Must pending chance states be valid inputs to `GameEncoder`?** If yes, F2 is a definite defect. If the encoder contract explicitly excludes chance states, it remains an unsafe latent path but does not affect current MCTS.
4. **Should the 22 damage-count features be normalized per card type, or is division by 2 intentional?** Current maxima are 1, 2, and 4 depending on copy count.
5. **Are all tracked/pre-feature checkpoints intentionally obsolete?** The current loader cannot use `model999`; acceptance needs either a fresh-model policy or a migration/conversion requirement.
6. **May Contain and Bomber remain outside this simplified implementation?** Neither is reachable with the current enum/rosters/configuration, so I treated both as out of scope rather than missing behavior.
7. **Should an automatically resolved standard critical be stored in `attack_info.critical`?** It is currently executed but represented as `None` in that field and its scalar encoding.

## Acceptance gate

Before merging, at minimum:

1. Ensure `damage_deck.py` is included in the tracked change.
2. Resolve or explicitly document the asteroid chance-state encoder contract and F2.
3. Answer Questions 1–7 and make `TODO.md` internally consistent.
4. Declare the checkpoint migration/fresh-start policy.

With those integration points resolved, the implemented core behavior is well covered and passed every legal-gameplay test run in this audit.
