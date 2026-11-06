# AlphArmada

Headless Star Wars: Armada rules engine and search infrastructure, based on the
fan-made SWArmada 0.2.7 implementation with additional squadron, objective,
movement and tournament rules. It runs without Unity or a game window.

## Repository layout

- `src/Swarmada.Headless`: rules, geometry, deterministic state recovery and MCTS.
- `src/Swarmada.Headless.Cli`: persistent JSON worker and diagnostic CLI.
- `python/swarmada_headless`: Python client for the worker.
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
use the accepted deferred-reveal UCT approximation. The included encoder and
model are diagnostic mocks; production encoding, BigDeep and training integration
remain unfinished.

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
