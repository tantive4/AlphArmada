# AlphArmada Structure and Execution Guide

This repository is now organized around four active package areas:

- `armada_game/`: Star Wars: Armada rules, data, geometry helpers, visualization assets, and Cython game-core sources.
- `learning/`: neural network, MCTS sources, encoder, replay storage, and training parameters.
- `evaluation/`: evaluation scripts, debug/verification helpers, notebooks, and sample result artifacts.
- `orchestration/`: worker, trainer, downloader, storage, and Vessl launch helpers.

The root `main_run.py` is intentionally kept as a tiny compatibility wrapper. The real implementation lives in `orchestration/main_run.py`, so both of these remain valid:

```bash
python main_run.py --mode worker --worker_id 01
python -m orchestration.main_run --mode worker --worker_id 01
```

The active Cython extension modules are still built as top-level imports named `armada`, `ship`, `squad`, `obstacle`, `defense_token`, `attack_info`, `action_manager`, `game_encoder`, `para_mcts`, and `shared_mcts`. Their source files moved into the package folders, but the extension names stayed top-level to preserve the existing Cython `cimport` relationships.

## Layout

```text
.
├── main_run.py                         # root compatibility launcher
├── tools/
│   └── cython_setup.py                 # builds all Cython extensions in-place
├── armada_game/
│   ├── core/                           # Cython game-core sources
│   ├── helpers/                        # Python game/rules/geometry helpers
│   ├── data/                           # JSON game data and generated action map
│   └── assets/                         # font and obstacle images
├── learning/
│   ├── params/                         # Config
│   ├── model/                          # BigDeep model
│   ├── mcts/                           # Cython MCTS sources
│   ├── encoding/                       # Cython state encoder
│   └── replay/                         # disk replay buffer and dataset
├── orchestration/
│   ├── main_run.py                     # actual worker/trainer/downloader entrypoint
│   ├── alpharmada.py                   # worker/trainer classes
│   ├── storage_manager.py              # Vessl model/replay transfer
│   ├── downloader.py                   # downloader state JSON helper
│   ├── launcher.py                     # Vessl run creation
│   └── vessl/                          # worker YAML
├── evaluation/
│   ├── evaluation.py
│   ├── matchup_self.py
│   ├── debug/
│   ├── notebooks/
│   └── results/
└── legacy/
    └── python_deprecated/              # old pure-Python references
```

## Runtime Roles

### Worker

Command:

```bash
python -m orchestration.main_run --mode worker --worker_id 01
```

Flow:

1. `orchestration.main_run.main()` parses CLI args.
2. `work(worker_id)` downloads the latest remote model with `orchestration.storage_manager.download_recent_model()`.
3. `learning.model.big_deep.load_model()` loads `model_checkpoints/model_best.pth`.
4. `orchestration.alpharmada.AlphArmadaWorker`:
   - switches the model to eval mode,
   - calls `model.compile_fast_policy()`,
   - clears local `replay_buffers/`,
   - creates a `learning.replay.disk_manager.DiskReplayBuffer`.
5. `AlphArmadaWorker.self_play()`:
   - creates randomized games with `armada_game.helpers.setup_game.setup_game()`,
   - deep-copies them into the configured parallel game batch,
   - uses `para_mcts.MCTS` for decision nodes,
   - uses `armada_game.helpers.dice.roll_dice()` for chance dice nodes,
   - applies actions through `Armada.apply_action()`,
   - stores terminal game memories through `save_game_data()`.
6. `save_game_data()` rewinds snapshots, calls `game_encoder.encode_game_state()`, and writes replay arrays plus policy/value/auxiliary targets.
7. `DiskReplayBuffer.trim_buffer()` truncates memmaps and writes `metadata.pkl`.
8. `orchestration.storage_manager.upload_replay_result()` uploads worker replay data and a timestamp commit flag to Vessl.

If worker self-play raises, `work()` uploads only `output/` with `upload_replay=False`, logs the error to Vessl, and does not publish the replay timestamp.

### Downloader

Command:

```bash
python -m orchestration.main_run --mode downloader --num_worker 20
```

Flow:

1. `download_all(num_worker)` loads `downloader_state.json` through `orchestration.downloader.load_state()`.
2. It clears and recreates local `staging/`.
3. For every worker id, it calls `get_worker_timestamp(worker_id)`.
4. If a worker timestamp is new, it downloads that worker volume into `staging/replayNN`.
5. After 8 staged worker buffers, it calls `learning.replay.disk_manager.aggregate_staging_buffers()`.
6. Aggregation writes one shuffled chunk under `replay_buffers/YYYYMMDD_HHMMSS`.
7. `staging/` is reset and polling continues.

The timestamp file in `alpharmada-worker-common/output_XX/timestamp` is the commit signal. The downloader does not inspect partially uploaded worker volumes.

### Trainer

Command:

```bash
python -m orchestration.main_run --mode trainer
```

Flow:

1. `train()` scans `Config.REPLAY_BUFFER_DIR`, currently `replay_buffers/`.
2. It waits until at least 4 chunk folders exist.
3. It keeps at most 40 chunk folders by deleting oldest chunks.
4. It loads or initializes the latest checkpoint with `learning.model.big_deep.load_recent_model()`.
5. It creates `AlphArmadaTrainer(model, optimizer)`.
6. `AlphArmadaTrainer.train_model()` creates `ArmadaChunkDataset` and a `DataLoader`.
7. Every step flattens `[chunk_count, seq_len, ...]` into a training batch and calls `train()`.
8. Loss combines policy, value, raw point, hull, and game-length heads.
9. The trainer saves `model_checkpoints/model_iter_XXX.pth`.
10. `orchestration.storage_manager.upload_model()` uploads the newest checkpoint to Vessl model repository `BigDeep`.

## Multi-Machine Data Flow

```text
Worker
  download latest model_best.pth
  -> self-play with BigDeep + para_mcts
  -> write replay_buffers/*.npy + metadata.pkl
  -> upload worker volume
  -> upload timestamp flag

Downloader
  poll worker timestamp flags
  -> download new worker volumes into staging/
  -> aggregate every 8 staged buffers
  -> write replay_buffers/<timestamp>/ chunks

Trainer
  wait for >= 4 chunks
  -> train BigDeep
  -> save model_iter_XXX.pth
  -> upload checkpoint

Next workers
  download uploaded checkpoint as model_best.pth
```

## Build and Verification

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install torch vessl
```

The Vessl worker YAML installs a CUDA Torch wheel instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126 --quiet
```

Build Cython extensions:

```bash
python tools/cython_setup.py build_ext --inplace
```

`tools/cython_setup.py` reads sources from:

- `armada_game/core/*.pyx`
- `learning/mcts/*.pyx`
- `learning/encoding/game_encoder.pyx`

and builds top-level modules used by runtime imports:

- `armada`
- `ship`
- `squad`
- `obstacle`
- `defense_token`
- `attack_info`
- `action_manager`
- `game_encoder`
- `para_mcts`
- `shared_mcts`

Local verification performed after the restructure:

```bash
python -m compileall -q armada_game learning orchestration evaluation tools main_run.py
python -c "from armada_game.helpers.enum_class import SHIP_DATA, SQUAD_DATA; from learning.params.configs import Config; from orchestration.downloader import load_state; print(len(SHIP_DATA), len(SQUAD_DATA), Config.DEVICE, load_state())"
```

Both passed.

The Cython setup command now reaches native compilation, but this local Windows environment stops with:

```text
Microsoft Visual C++ 14.0 or greater is required.
```

Install Microsoft C++ Build Tools, then rerun `python tools/cython_setup.py build_ext --inplace` to produce the `.pyd` extension modules. On Vessl/Linux, the worker image is expected to have a compiler toolchain available.

## Active File Guide

### `armada_game/helpers/`

`paths.py`

Central package-relative path helper:

- `data_path(filename)` resolves `armada_game/data/<filename>`.
- `asset_path(filename)` resolves `armada_game/assets/<filename>`.

This removes dependence on the process working directory for JSON and font files.

`enum_class.py`

Loads `ship_dict.json` and `squad_dict.json` from `armada_game/data/`, then defines shared enums:

- `Faction`
- `Dice`
- `HullSection`
- `SizeClass`
- `Command`
- `AttackRange`
- `Critical`
- `ObstacleType`
- `TokenType`

It also exports enum-derived tuples and counts used by Cython code and model code.

`action_phase.py`

Defines the simplified phase state machine and the `ActionType` type alias. Also provides `get_action_str(game, action)` for visualization/debug output.

`action_space.py`

Generates `armada_game/data/action_space.json`. Run it after changing phases, action definitions, or action-space limits:

```bash
python -m armada_game.helpers.action_space
```

`dice.py`

Contains dice pools, probability tables, face damage metadata, `roll_dice()`, `dice_choices()`, `fast_dice_choice()`, and `dice_icon()`. The display icons were normalized to ASCII labels so the module is UTF-8 importable.

`measurement.py`

Contains physical board constants, range bands, ship base/token dimensions, obstacle polygons, ship template polygons, and precomputed threat-zone templates.

`jit_geometry.py`

Numba-compiled geometry kernels: SAT overlap, line intersections, polygon distances, clipping, attack ranges, maneuver transforms, and range conversion.

`cache_function.py`

LRU-cached adapter layer around geometry:

- ship/obstacle coordinate transforms,
- ship-to-ship and ship-to-squad attack ranges,
- obstruction and overlap checks,
- maneuver tool transforms,
- spatial encoding indices.

`setup_game.py`

Builds randomized game instances from static JSON data and compiled game classes. It currently uses the simplified setup with no obstacles and no squadrons.

`visualizer.py`

Pillow debug renderer. It reads the font through `asset_path("ARIAL.TTF")` and writes images to `game_visuals/`.

### `armada_game/core/`

These are the Cython game-core sources. They still compile into top-level extension modules.

`action_manager.pyx/.pxd`

Loads `armada_game/data/action_space.json`, builds per-phase action-to-index maps, and exposes `get_action_map()` and `get_action_index()`.

`armada.pyx/.pxd`

Main game state machine:

- stores board state, ships, squads, obstacles, active/defending units, attack info, phase, current player, decision player, and winner,
- exposes `get_valid_actions()`,
- mutates state through `apply_action()`,
- snapshots/restores state for MCTS,
- handles activation, attack, maneuver, status, scoring, and visualization hooks.

`ship.pyx/.pxd`

Ship state and behavior:

- deployment,
- commands,
- defense tokens,
- attack target selection,
- maneuver generation and movement,
- overlap/collision handling,
- defense/damage,
- snapshots and hash state.

`squad.pyx/.pxd`

Squadron state and behavior. Present for the broader ruleset but mostly inactive because `Config.MAX_SQUADS = 0` in the current simplified training setup.

`attack_info.pyx/.pxd`

Attack-context object. Tracks attacker/defender ids, hull sections, range, obstruction, dice pool, dice result, spent tokens, redirect hull, critical, and total damage.

`defense_token.pyx/.pxd`

Defense token state: ready/exhausted/discarded/accuracy plus spend, discard, ready, snapshot, and restore.

`obstacle.pyx/.pxd`

Obstacle geometry state and placement transform.

### `armada_game/data/`

`ship_dict.json`

Static ship definitions. Currently contains 10 ships.

`squad_dict.json`

Static squadron definitions. Currently contains 8 squadrons.

`action_space.json`

Generated phase-to-action map consumed by `ActionManager`.

`simplified.txt`

Simplified-rules note. The text appears encoding-damaged, so use cautiously.

### `armada_game/assets/`

`ARIAL.TTF`

Font for `visualizer.py`.

`obstacle_image/`

Obstacle source PNGs and `polygon_numpy.py`, a helper for extracting/simplifying obstacle polygons from images.

### `learning/params/`

`configs.py`

Central `Config` class:

- game limits,
- encoding tensor sizes,
- hardware device,
- self-play batch sizes,
- MCTS parameters,
- replay buffer sizes,
- training hyperparameters,
- checkpoint/replay paths.

### `learning/model/`

`big_deep.py`

Defines the `BigDeep` neural network and checkpoint helpers.

Main architecture:

1. scalar token embedding,
2. ship entity embedding plus defense-token and coordinate Fourier features,
3. relation-biased transformer block,
4. spatial sandwich with bit-packed board planes and ResNet,
5. second transformer block,
6. policy/value/auxiliary heads.

Helpers:

- `load_recent_model()`
- `load_model(version=None)`
- `compile_fast_policy()`

### `learning/mcts/`

`para_mcts.pyx`

Active batched MCTS for self-play/evaluation. It keeps one root per parallel game and batches model calls through preallocated encoder buffers.

`shared_mcts.pyx`

Single-game shared-root MCTS used by `evaluation/matchup_self.py`.

### `learning/encoding/`

`game_encoder.pyx/.pxd`

Encodes game state directly into supplied NumPy memory views and returns `(active_ship_id, target_ship_id, phase)`.

Writes:

- scalar features,
- ship entity features,
- ship coordinates,
- ship defense token features,
- bit-packed spatial planes,
- pairwise relation matrix.

Also provides `get_terminal_value(game)` for value and auxiliary training targets.

### `learning/replay/`

`disk_manager.py`

Replay storage and loading:

- `DiskReplayBuffer`: worker-side memmap writer,
- `aggregate_staging_buffers()`: downloader-side merger/shuffler,
- `ArmadaChunkDataset`: trainer-side infinite iterable dataset.

### `orchestration/`

`main_run.py`

Actual CLI entrypoint. Modes:

- `worker`
- `trainer`
- `downloader`

`alpharmada.py`

High-level self-play and training orchestration:

- `AlphArmadaWorker`
- `AlphArmadaTrainer`

`storage_manager.py`

Vessl storage/model operations:

- worker timestamp check,
- replay upload/download,
- latest model download,
- versioned model download,
- checkpoint upload.

`downloader.py`

Reads/writes `downloader_state.json`.

`launcher.py`

Generates Vessl runs for workers `01` through `20`. The launch template now uses:

```bash
python tools/cython_setup.py build_ext --inplace
python -u -m orchestration.main_run --mode worker --worker_id XX
```

`vessl/alpharmada-worker-xx.yaml`

Checked-in example worker YAML using the new build and module-run commands.

### `evaluation/`

`evaluation.py`

Downloads two model versions, loads both, runs paired MCTS games, and reports win/draw rate.

`matchup_self.py`

Runs a single model-vs-model game with `shared_mcts`.

`debug/check_model.py`

Model-policy inspection helper. It is still partly stale relative to the active encoder signature and should be refreshed before relying on it.

`debug/sample_rollout.py`

Rollout/visualization helper. It is also stale because the active encoder writes into supplied buffers rather than returning a dictionary.

`debug/test.py`

Single-batch overfit/checkpoint reload helper. It is partly stale because it references older win-probability output names.

`notebooks/`

Notebook experiments:

- `Elo-calculate.ipynb`
- `matchup_player.ipynb`

`results/replay_stats.txt`

Checked-in sample stats artifact. Runtime workers write current stats to `output/replay_stats.txt`.

### `legacy/python_deprecated/`

Old pure-Python references and backups. These are not active runtime code.

## Maintenance Notes

1. Rebuild Cython after changing any `.pyx` or `.pxd` file:

   ```bash
   python tools/cython_setup.py build_ext --inplace
   ```

2. Regenerate the action map after changing phases/actions:

   ```bash
   python -m armada_game.helpers.action_space
   ```

3. Keep `Config.MAX_ACTION_SPACE` synchronized with `action_space.json`. `BigDeep.__init__()` checks this through `ActionManager`.

4. Worker mode deletes local `replay_buffers/` when `AlphArmadaWorker` starts. Do not run worker mode in the same local directory used by a trainer/downloader for persistent chunks.

5. The active compiled modules remain top-level. Do not change extension names in `tools/cython_setup.py` without also updating Cython `cimport` statements and Python imports.

6. Several debug scripts under `evaluation/debug/` predate the current in-place encoder and current model output names. Treat the orchestration path, `learning/model`, `learning/replay`, `learning/encoding`, `learning/mcts`, and `armada_game/core` as the source of truth.
