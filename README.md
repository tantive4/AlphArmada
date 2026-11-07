# AlphArmada

Headless Star Wars: Armada rules engine and search infrastructure, based on the
fan-made SWArmada 0.2.7 implementation with additional squadron, objective,
movement and tournament rules. It runs without Unity or a game window.

## Repository layout

- `src/Swarmada.Headless`: rules, geometry, deterministic state recovery and MCTS.
- `src/Swarmada.Headless/Learning`: visible state, spatial/candidate tensors, batch
  collation, lossless replay transport and terminal auxiliary targets.
- `src/Swarmada.Headless.Cli`: persistent JSON worker and diagnostic CLI.
- `src/Swarmada.Headless.Worker`: bounded native self play, replay shards and telemetry.
- `python/swarmada_headless`: Python client for the worker.
- `orchestration/vessl_worker`: worker-only VESSL bootstrap and test supervision.
- `data`: five runtime catalogs required by the engine.

The previous Python/Cython AlphArmada project is preserved on the
[`alpharmada_v1` branch](https://github.com/tantive4/AlphArmada/tree/alpharmada_v1).
In the original working copy it is also archived locally in `alpharmada_v1/`.
Local tests, benchmarks, reference files, test data and planning notes are kept
in `.local/`; those folders and the original SWArmada distribution are not
published on `main`.

## Build and use

Requires the .NET 9 SDK. The Python client uses the Python standard library.
No Unity installation or external NuGet packages are required.

```powershell
dotnet build AlphArmada.sln -c Release
dotnet src/Swarmada.Headless.Cli/bin/Release/net9.0/Swarmada.Headless.Cli.dll status --data data
```

Use the persistent Python client after building:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'python')
python -c "from swarmada_headless import HeadlessClient; c = HeadlessClient(); print(c.status); c.close()"
```

The native library exposes `GameSession.GetDecision()`, `Apply()`,
`SampleChance()`, `DeepClone()`, `CopyFrom()` and typed snapshots. Player choices
and chance outcomes are explicit. Coordinates are millimetres; players are 1
and 2. The Python worker exposes the same game operations over JSON.

MCTS uses caller-owned finite action domains, conditional static/pointer choices,
root-copy-and-replay recovery, cached decision descriptors, compatible subtree
reuse, and independent environment/CPU/inference scheduling. Unknown enemy dials
use non-neural UCT at reveal, their first simulated private look, or the first
pre-reveal step whose legal effects depend on the dial. Simulated assignments
stay separate from each player's legally seen dials. Already resolved dials stay
fixed until an actual queue change; discarded tops expose a new slot rather than
reusing the old value. The structured encoder is available through
`ProductionSearchEncoder`; BigDeep and the optimizer/trainer remain separate
work. `TensorRandomSearchModel` consumes real padded tensors and supplies
untrained random policy/value outputs.

Create a learning search explicitly (the legacy constructor default retains the
small diagnostic encoder for existing tests):

```csharp
var encoder = new ProductionSearchEncoder(data);
var environment = new NativeMctsEnvironment(game, knowledge[actor - 1],
    SearchInformationMode.DeferredUctReveal, encoder: encoder,
    observerKnowledge: knowledge);
await using var inference = new InferenceBatcher(new TensorRandomSearchModel(),
    maximumBatch: 32, maximumDelay: TimeSpan.FromMilliseconds(1));
var result = await new MctsSearch(environment, inference,
    new MctsOptions(Simulations: 200)).RunAsync();
var samples = TrainingRecordCapture.Capture(game, knowledge, result, encoder);
var action = NativeMctsEnvironment.ApplySelectedPath(game, result.SelectedPath,
    knowledge);
```

Each environment owns its encoder/workspace; independent games share an inference
batcher. Inputs use actual entity counts. The default encoder capacity profile is
24 ships, 48 squadrons and 384 upgrades, configurable through `EncoderLimits`;
overflow fails explicitly. Squadrons use coordinates/radii without headings or
individual raster planes. Ships use position and sine/cosine heading. Physical
relations and native legal candidate facts remain separate from the shared
64-by-128 terrain raster.

`EncoderManifest.Describe(encoder.Schema)` exports categorical tables, field
layout and schema fingerprints for replay/checkpoint compatibility.
`ReplayTensorCodec` transports actual-count tensors losslessly as raw or zlib
records. `python/swarmada_headless/encoding.py` decodes them into byte-backed
views and optional NumPy arrays/batches. Observations exclude unknown commands,
unknown facedown identities, future deck order and RNG; authoritative game
snapshots remain a separate recovery API. Conditional policy targets are exported
per selected prefix; hidden-command UCT branches produce no policy targets.
`TerminalTrainingTargets.FromTerminal` retains result, raw-score, ship-hull,
game-round and squadron-hull supervision after a genuine terminal outcome.

The native self-play worker runs real MCTS and production encoding with the
explicit `TensorRandomSearchModel` substitute. It streams compressed observations
and sparse conditional visit targets to disk, publishes a replay shard only after
a genuine terminal outcome, and records search, batching, memory, game-length and
transfer statistics. Limits and cancellation leave diagnostics without terminal
training labels. Optional exact action replay is available for verification.

```powershell
dotnet src/Swarmada.Headless.Worker/bin/Release/net9.0/Swarmada.Headless.Worker.dll run --data data --output .local/worker --games 2 --preset small --simulations 200 --concurrency 2 --cpu-workers 2
```

See [worker storage](python/swarmada_headless/WORKER_STORAGE.md) for the immutable
READY/checksum protocol and [VESSL workers](orchestration/vessl_worker/README.md)
for bounded deployment and retrieval. Cloud authentication remains outside source
and deployment configuration. The worker test path contains no trainer.

Search keeps simulated command assignments separate from legal observations.
The eight-column command row ends with an explicit known-enemy-dial flag;
internal materialization status is never a model feature (state schema v3).
`SearchRecovery` checkpoints actual roots. Capturing a worker-managed simulation
as a complete recovery checkpoint is rejected because its private continuation
and materialization overlay are wrapper state; MCTS continues using root replay.

## Rules and coverage limits

Each standard match takes exactly one supplied objective. All 36 official
objectives have executable handlers. The default `static-map-v1` training
variant disables obstacle and existing map-token relocation, including Strategic,
Ezra and Grav Shift Reroute. Setup remains caller-owned. Token pickup, scoring,
ordinary unit movement and reserve deployment remain available. Explicitly enable
`mapMovementEnabled` to use retained relocation rules.

The additional 20 unique squadron abilities, Cloak, offboard destruction,
tournament scoring and Mercy Mission corrections are implemented. Fighter-group
cards are excluded. The 13 obstacle silhouettes support reflected placement;
small source-to-physical calibration differences remain.

Damage-deck initialization and exhaustion deliberately retain the source behavior:
the order derives from `matchId`, and an exhausted deck draws Structural Damage.
The user's reduced-endpoint ram, retryable blocked Fleet Ambush, persistent
initial exogorth host and no late-round Rieekan rulings remain applied.

Full official interaction parity is not claimed. Some private card interactions
raise explicit search coverage errors, and the full before/when/after timing and
per-card policy registry remain incomplete. The status flags `fullGameReady` and
`trainingReady` remain conservative. Tests and replay checks do not establish
exhaustive correctness for every card combination.
