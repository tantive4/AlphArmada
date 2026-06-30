# AlphArmada

AlphArmada is an AlphaZero-style self-play project for a simplified
Star Wars: Armada environment. It combines a Cython rules engine, a
PyTorch neural network, batched Monte Carlo Tree Search, disk-backed replay
storage, and Vessl-based multi-worker orchestration.

The goal of the project is to learn tactical ship activation, attack,
defense-token, repair, and maneuver decisions directly from self-play rather
than from scripted heuristics.

## Results

The reported training run used 20 GTX1080 workers in a Vessl environment to
generate 512k self-play game replays, about 40M encoded states, over roughly
10 days. Training ran in parallel on a MacBook Pro and advanced the model to
iteration 1000. The relative Elo estimate rose from 1000 at initialization to
1312 at model version 1000.

![AlphArmada Elo progression](result/Elo_score.png)

The final human matchup video is available at
[result/game_visual.mp4](result/game_visual.mp4).

## What Is Implemented

AlphArmada models a ship-focused version of Armada:

- Rebel and Empire fleets are generated from JSON ship data.
- Fleets are randomly deployed on a 3ft x 3ft board coordinate system.
- Ships have hull zones, shields, hull points, speed, navigation charts,
  attack dice, command dials, and defense tokens.
- The active game phases include ship activation, command reveal, repair,
  target choice, target declaration, dice rolling, attack effects, defense
  tokens, damage resolution, and maneuver execution.
- Dice rolls are treated as chance nodes instead of player decisions.
- Legal actions are generated from the current game state and masked against a
  global phase-aware action space.
- The game engine, encoder, MCTS, and hot loops are implemented in Cython for
  speed.

The current training setup intentionally simplifies parts of the full tabletop
game. Squadrons are disabled (`MAX_SQUADS = 0`), command stacks are disabled
(`MAX_COMMAND_STACK = 0`), and randomized setup currently places no obstacles,
although squadron, obstacle, and fuller command logic remains present in the
codebase.

## Model Structure

The main neural network is `BigDeep` in `learning/model/big_deep.py`.
It receives a structured encoding of the current game state:

- `scalar`: 48 global features including round, score, factions, player,
  phase, attack range, obstruction, and dice-pool context.
- `ship_entities`: up to 10 ships, each with 128 static and dynamic features.
- `ship_coords`: normalized ship position and orientation.
- `ship_def_tokens`: up to 4 defense tokens per ship, with readiness,
  accuracy-lock, spendability, and token type.
- `spatial`: bit-packed 64 x 64 ship presence and threat planes.
- `relations`: 10 x 10 pairwise ship relation features, including hull-zone
  ranges and relative geometry.

The architecture is a transformer-centric "sandwich" model:

1. A scalar token and per-ship tokens are embedded into a 256-dimensional token
   space.
2. Defense tokens are separately embedded and fused into ship embeddings.
3. Fourier coordinate features and learned geometric relation bias inform
   multi-head attention.
4. A first 3-layer transformer block reasons over global and ship tokens.
5. Ship tokens are scattered into spatial presence/threat maps, processed by a
   6-block ResNet with global-pooling bias, then gathered back into the token
   stream.
6. A second 3-layer transformer block performs tactical reasoning with the
   spatial context fused back in.

The network has multiple heads:

- Policy head: phase-specific action logits, with pointer-style heads for
  choosing ships or defense tokens and static MLP heads for fixed action sets.
- Value head: scalar win/loss value in `[-1, 1]`.
- Auxiliary heads: raw point prediction, per-ship remaining hull prediction,
  and game-length prediction.

## MCTS and AlphaZero Loop

The self-play policy is produced by Cython MCTS implementations in
`learning/mcts/para_mcts.pyx` and `learning/mcts/shared_mcts.pyx`.

At each decision state:

1. The game state is encoded into the model input buffers.
2. `BigDeep` predicts a value and policy prior.
3. The prior is masked to legal actions from the current phase.
4. MCTS expands children using the masked prior and backs up model values.
5. Selection uses a PUCT-style score with visit counts, value estimates, and
   policy priors.
6. Dirichlet noise is mixed into root priors during self-play for exploration.
7. Dice rolls are sampled as chance nodes.
8. The final policy target is the normalized root visit-count distribution.

The default search settings are:

- 200 simulations for deep search states.
- 50 simulations for fast search states.
- 25% deep-search sampling ratio for replay saving.
- Temperature decays by game round (`TEMPERATURE / round`).
- Root Dirichlet noise uses `epsilon = 0.25`.

Only deep-search decision states are stored as replay targets, so the trainer
learns from higher-quality MCTS distributions while workers still advance games
quickly.

## Training Workflow

The distributed workflow has three long-running roles.

### Worker

```bash
python -m orchestration.main_run --mode worker --worker_id 01
```

A worker downloads the latest model from the Vessl `BigDeep` model repository,
runs 128 parallel self-play games per batch (`16` diverse setups x `8` copies),
saves replay arrays to disk-backed `.npy` memmaps, trims them to the exact
number of generated states, and uploads the result plus a timestamp commit flag.

### Downloader

```bash
python -m orchestration.main_run --mode downloader --num_worker 20
```

The downloader polls worker timestamp flags, downloads new worker buffers into
`staging/`, aggregates every 8 worker outputs, shuffles the merged samples, and
writes chunk folders under `replay_buffers/`.

### Trainer

```bash
python -m orchestration.main_run --mode trainer
```

The trainer waits for at least 4 replay chunks, keeps a sliding window of up to
40 chunks, loads the latest checkpoint, trains with AdamW, writes
`model_iter_XXX.pth`, and uploads the newest checkpoint back to Vessl.

The training loss combines:

- Policy cross entropy against MCTS visit distributions.
- Value MSE against terminal game outcome.
- Raw point, ship hull, and game-length auxiliary losses.

## Repository Layout

```text
armada_game/
  core/         Cython game engine: Armada, Ship, Squad, tokens, attacks, actions
  helpers/      setup, phases, dice, geometry, measurement, visualization
  data/         ship data, squad data, generated action-space JSON
  assets/       fonts and obstacle images

learning/
  model/        BigDeep neural network
  mcts/         parallel and shared-tree Cython MCTS
  encoding/     Cython game-state encoder
  replay/       disk replay buffer and iterable dataset
  params/       training, model, action-space, and hardware config

orchestration/
  main_run.py   worker, trainer, and downloader entrypoint
  alpharmada.py self-play worker and trainer classes
  storage_*     Vessl model and replay transfer helpers
  vessl/        worker launch YAML

evaluation/
  evaluation.py model-vs-model evaluation
  matchup_self.py model gameplay / visualization helper
  notebooks/    Elo and matchup notebooks

result/
  Elo_score.png
  game_visual.mp4
```

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install torch
```

Build the Cython extensions in place:

```bash
python tools/cython_setup.py build_ext --inplace
```

The active extension modules are built as top-level imports:

```text
armada, ship, squad, obstacle, defense_token, attack_info,
action_manager, game_encoder, para_mcts, shared_mcts
```

## Evaluation

Compare two uploaded model versions:

```bash
python -m evaluation.evaluation --versions 1000 800
```

Run the local model matchup/visualization helper:

```bash
python -m evaluation.matchup_self
```

Generated visual frames are written under `game_visuals/` when visualization is
enabled.
