"""Bounded worker supervisor: real self play, durable replay upload, and telemetry.

This process does not train a model. Its downloaded model artifact is explicitly a
seeded random-policy/value substitute; it never imports torch or uses the GPU.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import threading
import time


def default_cells(slot: int) -> list[dict]:
    """Twenty slots cover two fleet sizes, two budgets, and five concurrencies."""
    index = slot - 1
    preset = "small" if index < 10 else "mixed"
    simulations = 200 if index % 10 < 5 else 600
    concurrency = (1, 2, 4, 8, 12)[index % 5]
    cells = [dict(id=f"{preset}-{simulations}-c{concurrency}", preset=preset,
        simulations=simulations, concurrency=concurrency, games=32,
        duration_seconds=900 if preset == "small" else 2400,
        drain_seconds=120 if preset == "small" else 600)]
    if concurrency == 12:
        cells.append(dict(cells[0], id=f"{preset}-{simulations}-c16", concurrency=16, games=32))
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--deadline-unix", required=True, type=float)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--work-root", required=True)
    args = parser.parse_args()
    if not 1 <= args.slot <= 20:
        raise ValueError("Worker slot must be 1..20")
    repository = Path(args.repository).resolve()
    sys.path.insert(0, str(repository / "python"))
    from swarmada_headless.worker_storage import (LocalStore, VesslStore, WorkerSpool,
        SpoolLimits, atomic_json, fetch_bundle, fetch_ready_artifact, publish_bundle,
        validate_game_records, validate_ready_artifact)
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    deadline = min(args.deadline_unix, float(config["deadline_unix"]))
    if deadline <= time.time() + 10:
        raise TimeoutError("No time remains for worker startup")
    namespace = config["namespace"]
    worker = f"worker-{args.slot:02d}"
    identity = os.environ.get("VESSL_RUN_ID") or os.environ.get("ALPHARMADA_EXECUTION_ID")
    if not identity and not config.get("local_store"):
        raise ValueError("Remote worker requires a unique execution identity")
    execution = re.sub(r"[^a-zA-Z0-9._-]", "-", identity or "local")[:40]
    run_id = namespace[:40] + "-" + hashlib.sha256((namespace + ":" + execution).encode()).hexdigest()[:16]
    work = Path(args.work_root).resolve() / "runs" / (worker + "-" + execution)
    work.mkdir(parents=True, exist_ok=True)
    artifact_prefix = os.environ.get("ALPHARMADA_ARTIFACT_PREFIX", f"{namespace}/artifacts/{worker}/{execution}")
    store = LocalStore(config["local_store"], namespace) if config.get("local_store") else VesslStore(
        config["organization"], config["volume"], storage=config.get("storage", "vessl-storage"), namespace=namespace)
    log_path = work / "supervisor.jsonl"
    lock = threading.Lock()
    failures: list[dict] = []
    receipts: list[dict] = []
    cells_done: list[dict] = []
    spools: list[WorkerSpool] = []
    stop = threading.Event()
    active: list[subprocess.Popen] = []
    log_offsets: dict[str, int] = {}
    log_sequence = 0
    start = time.time()

    def event(name: str, **data) -> None:
        row = dict(event=name, time=time.time(), worker=worker, execution=execution, **data)
        with lock, log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        print(json.dumps(row, separators=(",", ":"), allow_nan=False), flush=True)

    def request_stop(_signum=None, _frame=None):
        stop.set()
        for process in tuple(active):
            if process.poll() is None:
                process.terminate()

    def cleanup_children():
        request_stop()
        for process in tuple(active):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    atexit.register(cleanup_children)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def health(phase: str, **data) -> None:
        # Mutable status is observational. Immutable READY bundles contain data.
        try:
            store.put_json(f"{namespace}/health/{worker}.json", dict(phase=phase, time=time.time(),
                slot=args.slot, execution=execution, commit=config["repository_commit"],
                deadline_unix=deadline, artifact_prefix=artifact_prefix, **data))
        except Exception as error:
            event("health-upload-failed", errorType=type(error).__name__)

    def log_files() -> list[Path]:
        return [log_path, *work.glob("cells/*/console.jsonl"), *work.glob("cells/*/logs/*"),
                *work.glob("cells/*/summaries/*"), *work.glob("cells/*/failures/*")]

    def upload_logs(final: bool = False, targets: dict[str, int] | None = None) -> bool:
        nonlocal log_sequence
        files = log_files()
        pieces: list[tuple[str, int, bytes]] = []
        new_offsets = {}
        for path in sorted(set(files)):
            if not path.is_file():
                continue
            name = path.relative_to(work).as_posix()
            if targets is not None and name not in targets:
                continue
            offset = log_offsets.get(name, 0)
            length = path.stat().st_size if targets is None else targets.get(name, 0)
            if length < offset:
                offset = 0  # Native log rotation starts a new file generation.
            with path.open("rb") as source:
                source.seek(offset)
                data = source.read(max(0, min(16 * 1024 * 1024, length - offset)))
            if not final and path.suffix == ".jsonl" and data:
                end = data.rfind(b"\n")
                data = data[:end + 1] if end >= 0 else b""
            if data:
                pieces.append((name, offset, data))
                new_offsets[name] = offset + len(data)
        if not pieces:
            return False
        log_sequence += 1
        bundle = work / "log-bundles" / f"chunk-{log_sequence:06d}"
        bundle.mkdir(parents=True, exist_ok=False)
        fragments = []
        for index, (name, offset, data) in enumerate(pieces):
            part = f"fragment-{index:04d}.bin"
            target = bundle / part
            target.write_bytes(data)
            fragments.append(dict(file=part, source=name, offset=offset, bytes=len(data)))
        atomic_json(bundle / "fragments.json", dict(format="swarmada.worker-log-fragments.v1", fragments=fragments))
        receipt = publish_bundle(store, bundle, f"{artifact_prefix}/logs/{bundle.name}", dict(
            kind="logs", runId=run_id, workerId=worker, bundleId=bundle.name,
            versions={"codeVersion": config["repository_commit"]}, final=final))
        log_offsets.update(new_offsets)
        atomic_json(work / "log-offsets.json", log_offsets)
        receipts.append(dict(kind="logs", **receipt))
        event("logs-uploaded", **receipt)
        return True

    def pump(spool: WorkerSpool) -> None:
        report = spool.pump_once()
        receipts.extend(dict(kind="replay", **item) for item in report["published"])
        for failed in report["failed"]:
            failures.append(dict(stage="upload", **failed))

    def safe_upload_logs(final: bool = False) -> None:
        try:
            targets = {p.relative_to(work).as_posix(): p.stat().st_size for p in log_files() if p.is_file()} if final else None
            while time.time() < deadline - 30:
                if not upload_logs(final=final, targets=targets) or not final:
                    break
        except Exception as error:
            event("log-upload-failed", errorType=type(error).__name__)

    event("worker-start", python=sys.version, platform=platform.platform(),
        cpu_count=os.cpu_count(), affinity=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        mock_model=True, gpu_compute=False, commit=config["repository_commit"])
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        model_name = next((line.partition(":")[2].strip() for line in cpuinfo.read_text().splitlines()
                           if line.startswith("model name")), "unknown")
        event("cpu-model", model=model_name)
    if sys.platform == "linux":
        try:
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"], text=True, timeout=15).strip()
            event("allocated-gpu", description=gpu, computation="unused")
        except (FileNotFoundError, subprocess.SubprocessError):
            event("allocated-gpu", description="unavailable", computation="unused")
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            event("resource-quota", resource=name, value=path.read_text().strip())
    health("model-download")
    model_directory = work / "mock-model"
    model_receipt = fetch_bundle(store, config["mock_model_prefix"], model_directory, kind="mock-model",
        expected_versions={"codeVersion": config["repository_commit"]})
    model_artifact = validate_ready_artifact(model_directory)
    model_files = [f for f in model_artifact.manifest["files"] if f["path"] == "model.json"]
    if len(model_files) != 1:
        raise ValueError("Mock model bundle must contain model.json")
    model_definition = json.loads((model_directory / "model.json").read_text(encoding="utf-8"))
    published_version = model_artifact.manifest.get("modelVersion")
    if model_definition.get("format") != "swarmada.tensor-random-model.v1" or not model_definition.get("modelVersion"):
        raise ValueError("Unsupported mock model definition")
    if published_version and published_version != model_definition["modelVersion"]:
        raise ValueError("Mock model payload differs from its published model version")
    event("model-downloaded", **model_receipt, modelBytes=model_files[0]["bytes"],
        limitation="Seeded mock only; production BigDeep weight size is not measured")
    receipts.append(dict(kind="mock-model", **model_receipt))
    configured = config.get("matrix_by_slot", {}).get(str(args.slot))
    cells = configured if configured is not None else default_cells(args.slot)
    if not isinstance(cells, list) or not cells:
        raise ValueError("Worker needs at least one matrix cell")
    assembly = repository / "src/Swarmada.Headless.Worker/bin/Release/net9.0/Swarmada.Headless.Worker.dll"
    dotnet = config.get("dotnet_command", "dotnet")
    roundtrip_done = False
    for index, cell in enumerate(cells):
        if stop.is_set() or time.time() >= deadline - 90:
            break
        cell_id = cell["id"]
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", cell_id):
            raise ValueError("Unsafe cell id")
        output = work / "cells" / cell_id
        options = dict(data=str(repository / "data"), output=str(output), runId=run_id,
            workerId=worker, cellId=cell_id, games=int(cell.get("games", 32)),
            sourceRevision=config["repository_commit"],
            preset=cell["preset"], simulations=int(cell["simulations"]), concurrency=int(cell["concurrency"]),
            cpuWorkers=int(config.get("threads_per_worker", 10)), batchSize=int(cell.get("batch_size", 32)),
            batchDelayMs=float(cell.get("batch_delay_ms", 1)), seed=int(cell.get("seed", 7331)),
            durationSeconds=min(float(cell.get("duration_seconds", 1200)), max(1, deadline - time.time() - 120)),
            drainSeconds=min(float(cell.get("drain_seconds", 180)), max(1, deadline - time.time() - 90)),
            maxGameSeconds=float(cell.get("max_game_seconds", 3600)), maxSteps=100000,
            memoryLimitMb=int(config.get("native_memory_limit_mb", 10240)), maxOutputBytes=2 * 1024**3,
            model=str(model_directory / "model.json"), replay=bool(cell.get("replay", False)),
            stopFile=str(output / "STOP"))
        if cell.get("objective"):
            options["objective"] = cell["objective"]
        output.mkdir(parents=True, exist_ok=True)
        config_file = output / "native-config.json"
        atomic_json(config_file, options)
        def storage_event(row):
            event("storage", sourceEvent=row.get("event"), **{k: v for k, v in row.items() if k != "event"})
        spool = WorkerSpool(output / "shards", store, limits=SpoolLimits(2 * 1024**3, 128), log=storage_event)
        spools.append(spool)
        command = [dotnet, str(assembly), "run", "--config", str(config_file)]
        event("cell-start", cell=cell, nativeOptions=options)
        health("self-play", cell=cell_id)
        console = (output / "console.jsonl").open("ab")
        process = subprocess.Popen(command, cwd=repository, stdout=console, stderr=subprocess.STDOUT,
            start_new_session=False)
        active.append(process)
        cell_start = time.time()
        last_health = last_logs = last_pump = 0.0
        capacity_stopped = False
        while process.poll() is None:
            now = time.time()
            if now >= deadline - 75 or stop.is_set():
                request_stop()
                try:
                    process.wait(timeout=max(1, min(30, deadline - time.time() - 40)))
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            if now - last_pump >= 5:
                pump(spool)
                last_pump = now
                usage = spool.usage()
                if not capacity_stopped and usage["bytes"] >= 1800 * 1024**2:
                    (output / "STOP").touch()
                    capacity_stopped = True
                    event("spool-backpressure", cell=cell_id, usage=usage)
            if now - last_health >= 30:
                health("self-play", cell=cell_id, spool=spool.usage(), elapsed_seconds=now-cell_start)
                event("heartbeat", cell=cell_id, spool=spool.usage(), elapsedSeconds=now-cell_start)
                last_health = now
            if now - last_logs >= 60:
                safe_upload_logs()
                last_logs = now
            stop.wait(.5)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        active.remove(process)
        console.close()
        pump(spool)
        summaries = []
        for path in output.glob("summaries/*.json"):
            summaries.append(json.loads(path.read_text(encoding="utf-8-sig")))
        result = dict(cell=cell_id, exitCode=process.returncode, elapsedSeconds=time.time()-cell_start,
            summaries=summaries, spool=spool.usage())
        cells_done.append(result)
        event("cell-finished", cell=cell_id, exitCode=process.returncode,
            elapsedSeconds=result["elapsedSeconds"], spool=result["spool"],
            completedGames=sum(s.get("completedGames", 0) for s in summaries),
            abortedGames=sum(s.get("abortedGames", 0) for s in summaries),
            errorGames=sum(s.get("errorGames", 0) for s in summaries))
        # One actual replay download verifies the eventual consumer's byte stream.
        # It is a worker transport test, with no optimizer or training dependency.
        if not roundtrip_done:
            for directory in sorted((output / "shards").iterdir()) if (output / "shards").exists() else []:
                if not (directory / ".uploaded.json").exists():
                    continue
                try:
                    receipt = json.loads((directory / ".uploaded.json").read_text())
                    destination = work / "roundtrip" / directory.name
                    transfer = fetch_ready_artifact(store, receipt["prefix"], destination)
                    checked = validate_game_records(destination, decode_inputs=True)
                    event("replay-roundtrip", **transfer, validation=checked)
                    receipts.append(dict(kind="replay-validation-download", **transfer))
                    roundtrip_done = True
                except Exception as error:
                    failure = dict(stage="replay-validation-attempt", cell=cell_id, errorType=type(error).__name__)
                    failures.append(failure)
                    event("replay-roundtrip-failed", **failure)
                break
        if process.returncode != 0:
            failures.append(dict(stage="native", cell=cell_id, exitCode=process.returncode))
        safe_upload_logs(final=True)
    # Give persistent retries time to publish completed games before exiting.
    flush_until = min(deadline - 35, time.time() + 120)
    while any(spool.usage()["pendingShards"] for spool in spools) and time.time() < flush_until:
        for spool in spools:
            pump(spool)
        stop.wait(2)
    pending = sum(spool.usage()["pendingShards"] for spool in spools)
    if pending:
        failures.append(dict(stage="unpublished-replay", pendingShards=pending))
        event("unpublished-replay", pendingShards=pending)
    if not roundtrip_done and any(spool.usage()["readyShards"] for spool in spools):
        failures.append(dict(stage="replay-validation-missing"))
    summary = dict(format="swarmada.worker-runtime-summary.v1", worker=worker, execution=execution,
        commit=config["repository_commit"], mockModel=True, gpuCompute=False, startedUnix=start,
        finishedUnix=time.time(), elapsedSeconds=time.time()-start, cells=cells_done,
        failures=failures, transfers=receipts, storeCounters=getattr(store, "counters", {}),
        replayRoundtripValidated=roundtrip_done)
    atomic_json(work / "runtime-summary.json", summary)
    # Summary is copied into an immutable log bundle; health points to its prefix.
    final = work / "final-bundle"
    final.mkdir(exist_ok=True)
    (final / "runtime-summary.json").write_bytes((work / "runtime-summary.json").read_bytes())
    for attempt in range(5):
        try:
            final_receipt = publish_bundle(store, final, f"{artifact_prefix}/final", dict(kind="logs",
                runId=run_id, workerId=worker, bundleId="final", versions={"codeVersion": config["repository_commit"]}))
            break
        except Exception as error:
            event("final-upload-failed", errorType=type(error).__name__, attempt=attempt + 1)
            if attempt == 4 or time.time() >= deadline - 15:
                raise
            stop.wait(min(2 ** attempt, max(0, deadline - time.time() - 15)))
    health("complete" if not failures else "complete-with-errors", final_prefix=f"{artifact_prefix}/final",
        cells=len(cells_done), replay_roundtrip=roundtrip_done, final_receipt=final_receipt,
        transfer_counters_before_health=getattr(store, "counters", {}))
    event("worker-finished", finalReceipt=final_receipt, cells=len(cells_done), failures=len(failures))
    return 1 if any(f.get("stage") in {"native", "unpublished-replay", "replay-validation-missing"} for f in failures) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # SDK exceptions may contain signed URLs; log the safe type only.
        import traceback
        frames = [dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                  for f in traceback.extract_tb(error.__traceback__)]
        print(json.dumps(dict(event="worker-fatal", errorType=type(error).__name__, frames=frames)), flush=True)
        raise SystemExit(1)
