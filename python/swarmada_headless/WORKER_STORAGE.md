# Worker storage protocol

`worker_storage.py` handles worker artifacts and transfers. It does not train a
model, launch cloud runs, or delete remote data. `VesslStore` requires an explicit
fresh namespace and rejects object keys outside it. Its lazy dependencies are
`vessl==0.1.199`, `boto3==1.43.118`, and `botocore==1.43.118`. The adapter restores
SDK TLS verification and obtains temporary, in-memory S3 federation credentials.

## Completed game directory

Native workers create `output/shards/<gameId>/` only for terminal games:

- `inputs.bin`: ASCII `SWAREC01`, little-endian uint32 version `1`, followed by
  repeated uint32 record lengths and exact `ReplayTensorCodec` records. Each
  record includes its compression envelope and full input schema identifier.
- `targets.json`: conditional decisions with `observer`, `prefix`,
  `candidateIndices`, positive integer `visits`, `inputOffset`, and `inputBytes`.
  Offsets point to record payloads, after their length fields. `terminalTargets`
  contains winner, two raw scores, final ship/squadron hull fractions and round.
- `game.json`: game configuration, genuine outcome and worker measurements.
- `encoder-manifest.json`: categorical vocabularies and feature/schema metadata.
- `manifest.json`: format `swarmada.worker-shard.v1`, `runId`, `workerId`,
  `shardId`, creation time, version registry, terminal game registry and payload
  files. Each file has a relative POSIX `path`, exact `bytes` and SHA-256.
- `READY`: format `swarmada.worker-shard-ready.v1`, exact `manifestBytes` and
  `manifestSha256`. The producer writes this atomically last.

The version registry requires `schemaVersion`, `catalogFingerprint`, and
`codeVersion`; it may also record model, policy, action-catalog and rule versions.
Partial, rejected or canceled games do not receive READY or terminal labels.

`validate_ready_artifact()` checks the marker, manifest, file lengths and hashes.
`validate_game_records()` additionally checks stream offsets, sparse visits,
terminal arrays and, by default, decodes the actual tensor records against the
declared schema. This is an integrity check, not a trainer.

## Publication and recovery

`publish_ready_artifact(store, directory, prefix)` uploads payload files, the
unchanged manifest bytes, then READY. A retry accepts an existing matching READY;
the same identity with a different manifest is an error. Downloaders ignore
directories without READY. `fetch_ready_artifact()` checks every downloaded file
in a temporary directory and exposes the final local directory only after full
verification. Schema expectations can be supplied explicitly.

`WorkerSpool(output_shards, store, limits=SpoolLimits(...))` uploads ready direct
children under `<namespace>/replay/<runId>/<workerId>/<shardId>`. `pump_once()`
persists retry attempts and exponential backoff in `.retry.json`, and writes
`.uploaded.json` only after checking remote READY. SDK exception messages are not
persisted because they may contain signed URLs. Restarting retries pending work;
an ambiguous successful final upload is resolved by its matching remote marker.

`require_capacity()` provides backpressure before generating another game. The
byte bound includes unfinished local directories. Native production must honor
the same configured byte bound while writing. No unacknowledged replay is evicted.
Acknowledged data is retained by default. Explicit local pruning rechecks remote
READY and preserves a receipt in `.receipts/`; it never deletes remote data.

## Runtime, mock models and logs

`publish_bundle()` / `fetch_bundle()` use the same marker/checksum mechanism with
format `swarmada.worker-bundle.v1` and kind `runtime`, `mock-model`, or `logs`.
Kind and version checks happen before exposing a download.

`write_mock_model()` creates an explicit random policy/value seed configuration.
`fetch_mock_model()` verifies it and reports its actual configuration bytes and
zero neural-weight bytes. These transfers do not estimate BigDeep checkpoint
size or GPU inference cost. Export logs as immutable copies/chunks; appending to
an already-ready bundle invalidates its checksums.

## Measurements and command line

Transfer methods return confirmed bytes, elapsed seconds and cache/skip status.
Artifact receipts separately report payload bytes, manifest/marker metadata
bytes, confirmed transferred bytes and elapsed seconds. Spool receipts include
attempt/retry counts. `VesslStore.counters` adds JSON bytes, operation counts,
federation renewals and expired-token retries. Counts are application bytes;
they exclude TLS/S3 framing and opaque SDK-internal retry traffic.

Run `python -m swarmada_headless.worker_storage --help`. Global backend arguments
precede subcommands: `upload-spool`, `download-verify`, `export-logs`,
`model-download`, and `list`. Use `--local-store` for offline protocol tests.

Offline fault coverage is in `.local/tools/test_worker_storage.py`; optional
`--native-shard PATH` also validates a genuine completed native worker shard.
