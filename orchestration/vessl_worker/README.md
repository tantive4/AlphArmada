# Bounded VESSL headless workers

This controller starts worker slots 01–20. It has no trainer, optimizer, model
seed, legacy replay import, or automatic restart service. It does not modify
the old `AlphArmada_vessl_test` project, its runs, or its storage volumes.

Copy `config.example.json` to a local ignored configuration. Set a fresh
`headless-*` namespace and `alpharmada-headless-*` volume; exact public Git
commit; runtime archive path and SHA256; and the user-start and absolute
deadline UNIX timestamps. The deadline must be at most four hours after user
start. Set `integration_reviewed=true` after reviewing the concrete settings.
The example is deliberately not launchable. Runtime archive paths resolve
relative to the configuration file. Configuration and YAML must contain no
personal credentials. Authentication uses the existing VESSL profile locally
and injected run credentials remotely.

The verified cluster configuration is organization/cluster
`snu-eng-gtx1080`, project `alpharmada`, preset `gpu-1`. The previous deployment
documented GTX1080 8 GiB, 11 CPU cores and 14 GiB RAM per preset. Available
capacity is shared. This worker uses the CPU random model; allocating a GPU
does not turn its measurements into neural inference benchmarks. The pinned
image is `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`; bootstrap installs the
configured .NET 9 SDK and validates the exact version. It verifies Microsoft's
SHA512 for the SDK archive. It never builds the old Cython engine or imports
the old BigDeep implementation.

Run from the repository root with a Python environment containing
`requirements-control.txt` (the existing sibling `.venv-vessl` is suitable).
The SDK uses the saved profile. If login is necessary, run `vessl configure`
and complete its normal browser authentication.

```powershell
python orchestration/vessl_worker/control.py --config .local/remote/config.json prepare
python orchestration/vessl_worker/control.py --config .local/remote/config.json pilot --compatibility
python orchestration/vessl_worker/control.py --config .local/remote/config.json health
python orchestration/vessl_worker/control.py --config .local/remote/config.json status
python orchestration/vessl_worker/control.py --config .local/remote/config.json logs RUN_ID --tail 100
# Review the Linux compatibility result, then run the actual runtime pilot.
python orchestration/vessl_worker/control.py --config .local/remote/config.json pilot --replace-terminal
# Review that pilot before expanding to all twenty slots.
python orchestration/vessl_worker/control.py --config .local/remote/config.json launch --pilot-reviewed
python orchestration/vessl_worker/control.py --config .local/remote/config.json fetch
python orchestration/vessl_worker/control.py --config .local/remote/config.json stop
```

`launch --slots 3 4 --pilot-reviewed --replace-terminal` replaces only the named
terminal worker slots. Other active or terminal slots are left alone, and the
twenty-active-worker limit still applies across the entire namespace.

`prepare` uploads the runtime archive using the storage module's runtime bundle
format: payload, checksum manifest, then the hashed READY commit marker.
The operations bundle embeds bootstrap, settings, pinned requirements,
standalone storage helper and sanitized configuration with per-file hashes.
The YAML also verifies the exact bundle archive SHA256 before extraction.
Bootstrap installs Git when the base image lacks it, fetches the exact Git
commit, verifies the runtime READY/manifest, archive hash and commit, then
invokes its configured Python entry:

```text
python runtime.py --config CONFIG --slot N --deadline-unix ABSOLUTE
  --repository /root/headless-engine --work-root /root/headless-worker
```

`ALPHARMADA_ARTIFACT_PREFIX`, repository, work-root, slot and deadline environment
variables are available as well. Include the Worker project in
`dotnet_projects` for real self play. The compatibility pilot builds the CLI,
checks its native status and proves a checksum-verified remote artifact
upload/download. It does not run a fake self-play workload. Runtime telemetry
and artifacts belong under the fresh namespace; bootstrap health is under
`NAMESPACE/health/worker-NN.json`, artifacts under
`NAMESPACE/artifacts/worker-NN/RUN_ID/`. Bootstrap leaves the runtime's richer
completion health record intact, including `final_prefix` and final transfer
receipt. No remote deletion is implemented.

## Runtime matrix and transfers

`mock_model_prefix` points to a separately published, READY-committed
`mock-model` bundle inside the same namespace. It contains `model.json` with
the seeded random policy/value configuration. The native worker reads the
actual encoded tensors. This transport exercise measures that real small
artifact, not future BigDeep weights or GPU model speed.

By default, the twenty slots cover small/mixed fleets, 200/600 MCTS
simulations and concurrent environments 1/2/4/8/12. Slots assigned 12 also run
a separate 16-environment cell afterward. Each cell has its own timings,
completed/unfinished counts, resource samples and byte counters. The extra
16-environment rows are sequential measurements, not replicated comparisons
against another simultaneous worker. Compare rates using each cell's actual
active time. A `matrix_by_slot` entry overrides the default for that slot:

```json
{
  "matrix_by_slot": {
    "1": [{"id":"pilot-small-200-c1","preset":"small","simulations":200,
           "concurrency":1,"games":2,"duration_seconds":900,
           "drain_seconds":120,"replay":true}]
  }
}
```

Optional runtime settings include `dotnet_command` (default `dotnet`),
`threads_per_worker` (default 10) and `native_memory_limit_mb` (default 10240).
The runtime reserves time for completed-game uploads before the global
deadline. Only genuine terminal games publish training-ready shards;
unfinished games retain their diagnostic status and do not receive invented
terminal targets. Failed uploads keep persistent retry metadata and local
data. Any READY shards still unpublished at shutdown are reported explicitly.

Replay READY bundles use
`NAMESPACE/replay/RUN_ID/worker-NN/SHARD_ID/`; log READY bundles use
`ARTIFACT_PREFIX/logs/chunk-NNNNNN/`. Each chunk contains short flat
`fragment-NNNN.bin.gz` files plus a `fragments.json` map with format
`swarmada.worker-log-fragments.v1`. Map entries provide `file`, original
`source`, byte `offset`, and `bytes`; reconstruct each source by offset
rather than parsing fragments as independent JSON. `codec="gzip"` and
`storedBytes` describe compressed storage; `bytes` retains the exact decoded
length. Earlier raw `.bin` fragments remain readable. Flat names avoid long
nested Windows paths. The final
parseable report is `ARTIFACT_PREFIX/final/runtime-summary.json`.
Completion health points to that final bundle and records its transfer
separately, avoiding a self-referential size total inside its own payload.
Receipts report confirmed payload/metadata byte transfers and elapsed time;
they do not estimate TLS overhead or unobservable SDK retry traffic.

Controller fetch verifies READY, manifest and every payload checksum before
exposing a local artifact directory. Its default prefix fetches logs/final
reports. Fetch replay explicitly:

```powershell
python orchestration/vessl_worker/control.py --config .local/remote/config.json fetch --prefix NAMESPACE/replay/
```

Each run has the earlier of its configured duration and global absolute
deadline. Every bootstrap subprocess is bounded by that deadline, including
installation and compilation. Workers run once and exit; failures are not
restarted automatically. `watch-deadline` is an optional independent local
stop-only watchdog. If launched in Windows with Start-Process, use
`-WindowStyle Hidden`. The root runtime must also enforce its deadline and
publish its final manifest before exit. `stop` disables all matching slots
locally and terminates only this exact namespace's worker names. Storage is
preserved. Confirm terminal statuses after requesting stop.

Local journals/manifests/status live under `.local/vessl-worker/NAMESPACE/`.
One controller may mutate this namespace at a time. A creation intent is
journaled before POST. Unknown POST outcomes are reconciled against the
server listing and never automatically retried; `--retry-missing-intents`
requires an operator to review the unresolved intent first. Existing terminal
slots need `--replace-terminal`. At most twenty matching workers may be active;
there is no compensating trainer or twenty-first slot. Pending capacity still
consumes the same absolute session budget. No monetary cost rate is assumed.

## Offline experiment report

After verified artifact fetch, summarize the experiment without cloud access:

```powershell
python orchestration/vessl_worker/report.py .local/vessl-worker/NAMESPACE/artifacts --output .local/reports/experiment.json --csv-prefix .local/reports/experiment --health .local/vessl-worker/NAMESPACE/health.json
```

The command writes JSON plus `.cells.csv`, `.games.csv`, `.transfers.csv`,
`.runs.csv` and `.transferTotals.csv`. Multiple input directories are allowed.
It rechecks READY, manifest and payload hashes, deduplicates fetched copies
by manifest and run/cell/game identity, and reports conflicting or incomplete
artifacts. `--no-logs` uses final reports only. Optional
`--reconstruct .local/reports/reconstructed` writes flat source-log files
and a source mapping, reconstructing `fragments.json` entries safely.

Completed-game wall times have average, median and p95 statistics. Nonterminal
games are recorded as censored and excluded from those completed-game timing
quantiles. Throughput divides completed games, searches, actions and published
records by the native cell's active wall time, not the sum of concurrently
running games. The supervisor's enclosing cell duration is also retained.
Missing finals can be recovered from verified native log summaries; when
only partial game logs survive, the report marks incompleteness and does not
invent a throughput denominator.

Reports separate attempted records from published terminal-game records, raw
and compressed tensor bytes from logical replay shard bytes, and confirmed
transfer receipts from those logical sizes. Replay/model/log/final transfers
remain separate; any explicitly synthetic transfer stays labelled synthetic.
Pass the downloaded controller health JSON to include the final report's own
transfer receipt. Search-latency p95 requires surviving search log events;
per-game average search time remains available in native summaries. The
measurements describe the CPU random model and make no neural GPU inference
or playing-strength claim.

Current workers gzip each log fragment and upload native JSONL once; fatal
stdout/stderr tails remain available. The fragment map binds compressed file
lengths to exact uncompressed source offsets. Rotated log generations retain
separate identities, so replacement files cannot overwrite earlier log bytes.
The reporter also accepts earlier raw fragments.

Verified supervisor log events also supply per-worker/execution hardware:
CPU model, reported logical CPUs, affinity count, allocated GPU description,
platform, and raw cgroup CPU/RAM quotas. CPU quota cores and RAM bytes are
reported separately from host-visible CPU counts. Runs and cells retain an
explicit hardware profile; differing or conflicting observations are not
silently combined. `endToEndCompletedGamesPerHour` uses the enclosing
supervisor cell duration, while `completedGamesPerHour` uses native active
cell time. Hardware fields remain unknown when those log events are absent
or `--no-logs` is used.
