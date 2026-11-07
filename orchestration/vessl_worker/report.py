"""Offline summaries of checksum-verified worker artifacts; never contacts VESSL."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from swarmada_headless.worker_storage import validate_ready_artifact

RUNTIME = "swarmada.worker-runtime-summary.v1"
NATIVE = "swarmada.worker-run.v1"


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def number(value, default=0):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else default


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def distribution(values) -> dict:
    values = sorted(number(value, None) for value in values if number(value, None) is not None)
    def percentile(p):
        if not values:
            return None
        index = (len(values) - 1) * p
        lower = math.floor(index)
        return values[lower] + (values[math.ceil(index)] - values[lower]) * (index - lower)
    return dict(count=len(values), average=statistics.fmean(values) if values else None,
        median=percentile(.5), p95=percentile(.95), minimum=min(values) if values else None, maximum=max(values) if values else None)


def histogram(values) -> dict:
    counts = Counter()
    for value in values:
        for key, count in (value or {}).items():
            try:
                key, count = int(key), int(count)
            except (TypeError, ValueError):
                continue
            if count > 0:
                counts[key] += count
    total = sum(counts.values())
    def ranked(rank):
        for key, count in sorted(counts.items()):
            if rank < count:
                return key
            rank -= count
        return None
    def percentile(p):
        if not total:
            return None
        index = (total - 1) * p
        lower, upper = ranked(math.floor(index)), ranked(math.ceil(index))
        return lower + (upper - lower) * (index - math.floor(index))
    return dict(count=total, average=ratio(sum(key * count for key, count in counts.items()), total),
        median=percentile(.5), p95=percentile(.95), maximum=max(counts) if counts else None,
        histogram={str(key): count for key, count in sorted(counts.items())})


def safe_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid fragment path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Fragment path escapes its artifact")
    return path


def read_json(path: Path):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Report JSON input exceeds 64 MiB")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_report(roots, *, health_files=(), reconstruct: Path | None = None, include_logs=True) -> dict:
    issues, artifacts, manifest_seen = [], [], {}
    roots = [Path(root).resolve() for root in roots]
    candidates = set()
    for root in roots:
        if root.is_file() and root.name == "READY":
            candidates.add(root.parent)
        elif root.is_dir():
            if (root / "READY").is_file():
                candidates.add(root)
            candidates.update(path.parent for path in root.rglob("READY") if path.is_file())
        else:
            issues.append(dict(kind="missing-input", path=str(root)))
    duplicate_artifacts = 0
    for directory in sorted(candidates):
        try:
            artifact = validate_ready_artifact(directory)
            if artifact.manifest_sha256 in manifest_seen:
                duplicate_artifacts += 1
                continue
            manifest_seen[artifact.manifest_sha256] = directory
            artifacts.append(artifact)
        except Exception as error:
            issues.append(dict(kind="invalid-artifact", path=str(directory), errorType=type(error).__name__))

    runtime, native, events = {}, {}, {}
    fragments = defaultdict(list)
    health_records = []
    def register(table, identity, value, priority, source):
        previous = table.get(identity)
        if previous and canonical(previous[0]) == canonical(value):
            return
        if previous and priority == previous[1]:
            issues.append(dict(kind="conflicting-identity", identity=list(identity), source=source))
            return
        if previous is None or priority > previous[1]:
            table[identity] = (value, priority, source)
    def runtime_record(value, priority, source):
        identity = (value.get("commit", ""), value.get("worker", ""), value.get("execution", ""))
        register(runtime, identity, value, priority, source)
        for cell in value.get("cells", []):
            for summary in cell.get("summaries", []):
                native_record(summary, priority, source, runtime_identity=identity, supervisor_cell=cell)
    def native_record(value, priority, source, runtime_identity=None, supervisor_cell=None):
        if value.get("format") != NATIVE:
            return
        identity = (value.get("runId", ""), value.get("workerId", ""), value.get("cellId", ""), value.get("executionId", ""))
        record = dict(summary=value, runtimeIdentity=runtime_identity, supervisorCell=supervisor_cell)
        register(native, identity, record, priority, source)
    def observe(value, source, priority=1):
        if not isinstance(value, dict):
            return
        if value.get("format") == RUNTIME:
            runtime_record(value, priority, source)
        elif value.get("format") == NATIVE:
            native_record(value, priority, source)
        elif value.get("event") == "cell-finished":
            for summary in value.get("summaries", []):
                native_record(summary, priority, source, supervisor_cell=value)
        elif value.get("eventType"):
            digest = hashlib.sha256(canonical(value).encode()).hexdigest()
            events.setdefault(digest, value)
            if value["eventType"] == "finished":
                native_record(value.get("data", {}).get("summary", {}), priority, source)

    for artifact in artifacts:
        manifest, directory = artifact.manifest, artifact.directory
        if manifest.get("kind") != "logs":
            continue
        payloads = {item["path"] for item in manifest["files"]}
        if "runtime-summary.json" in payloads:
            try:
                observe(read_json(directory / "runtime-summary.json"), str(directory), 3)
            except Exception as error:
                issues.append(dict(kind="invalid-final-json", path=str(directory), errorType=type(error).__name__))
        if not include_logs or "fragments.json" not in payloads:
            continue
        try:
            mapping = read_json(directory / "fragments.json")
            if mapping.get("format") != "swarmada.worker-log-fragments.v1":
                raise ValueError("Unknown log fragment map")
            for item in mapping["fragments"]:
                file = safe_relative(item["file"])
                source = safe_relative(item["source"])
                if file.as_posix() not in payloads:
                    raise ValueError("Fragment is absent from verified payload registry")
                offset, size = item["offset"], item["bytes"]
                if type(offset) is not int or type(size) is not int or offset < 0 or size < 0 or (directory / file).stat().st_size != size:
                    raise ValueError("Invalid log fragment byte range")
                identity = (manifest.get("runId", ""), manifest.get("workerId", ""), source.as_posix())
                fragments[identity].append((offset, size, directory / file))
        except Exception as error:
            issues.append(dict(kind="invalid-fragment-map", path=str(directory), errorType=type(error).__name__))

    reconstructed = []
    with tempfile.TemporaryDirectory(prefix="swarmada-report-") as temporary:
        target_directory = Path(reconstruct).resolve() if reconstruct else Path(temporary)
        target_directory.mkdir(parents=True, exist_ok=True)
        for identity, pieces in sorted(fragments.items()):
            extension = ".json" if identity[2].endswith(".json") else ".jsonl"
            target = target_directory / ("log-" + hashlib.sha256(canonical(identity).encode()).hexdigest()[:24] + extension)
            coverage, gaps, conflict = 0, [], False
            try:
                with target.open("w+b") as output:
                    for offset, size, piece in sorted(pieces, key=lambda item: (item[0], item[1])):
                        data = piece.read_bytes()
                        if offset > coverage:
                            gaps.append([coverage, offset])
                        overlap = min(size, max(0, coverage - offset))
                        if overlap:
                            output.seek(offset)
                            if output.read(overlap) != data[:overlap]:
                                conflict = True
                                break
                        output.seek(offset + overlap)
                        output.write(data[overlap:])
                        coverage = max(coverage, offset + size)
                row = dict(runId=identity[0], worker=identity[1], source=identity[2], file=target.name,
                    observedBytes=coverage, gaps=gaps, conflicting=conflict)
                reconstructed.append(row)
                if conflict:
                    issues.append(dict(kind="conflicting-fragment-generation", source=list(identity)))
                    continue
                if gaps:
                    issues.append(dict(kind="missing-log-fragments", source=list(identity), gaps=gaps))
                    continue
                if extension == ".json":
                    observe(read_json(target), "verified-log:" + "/".join(identity), 2)
                else:
                    with target.open("r", encoding="utf-8-sig", errors="replace") as source:
                        for line in source:
                            try:
                                observe(json.loads(line), "verified-log:" + "/".join(identity))
                            except (ValueError, TypeError):
                                # A killed process can leave one incomplete JSONL tail.
                                continue
            except Exception as error:
                issues.append(dict(kind="unreadable-reconstructed-log", source=list(identity), errorType=type(error).__name__))
        if reconstruct:
            (target_directory / "reconstructed.json").write_text(json.dumps(reconstructed, indent=2), encoding="utf-8")

    for path in health_files:
        try:
            value = read_json(Path(path))
            records = value if isinstance(value, list) else value.get("heartbeats", [value])
            health_records.extend(record for record in records if isinstance(record, dict))
        except Exception as error:
            issues.append(dict(kind="invalid-health-json", path=str(path), errorType=type(error).__name__))

    event_groups = defaultdict(list)
    for value in events.values():
        event_groups[(value.get("runId", ""), value.get("workerId", ""), value.get("cellId", ""))].append(value)
    # Recover a missing final native summary from terminal/aborted game log rows.
    known_groups = {identity[:3] for identity in native}
    for identity, rows in event_groups.items():
        if identity in known_groups:
            continue
        started = next((row for row in rows if row["eventType"] == "started"), None)
        if not started:
            continue
        data = started.get("data", {})
        games, begun, last_active = {}, {}, {}
        for row in rows:
            details = row.get("data", {})
            if row["eventType"] == "game_started":
                begun[details.get("gameId")] = details
            elif row["eventType"] in ("game_terminal", "game_aborted") and details.get("metrics"):
                games[details["metrics"]["gameId"]] = details["metrics"]
            elif row["eventType"] == "heartbeat":
                for game in details.get("activeGames", []):
                    last_active[game["gameId"]] = game
        for game_id, game in begun.items():
            if game_id not in games:
                games[game_id] = dict(last_active.get(game_id, {}), gameId=game_id, status="censored-no-final", index=game.get("index"), seed=game.get("seed"))
        summary = dict(format=NATIVE, runId=identity[0], workerId=identity[1], cellId=identity[2],
            executionId=data.get("executionId", ""), configuration=data.get("options", {}), versions=data.get("versions", {}),
            games=list(games.values()), incompleteSummary=True, stopReason="missing-final-summary")
        native_record(summary, 0, "verified-native-log-recovery")

    transfers, transfer_seen = [], {}
    def add_transfer(receipt, owner, default_kind=None):
        if not isinstance(receipt, dict) or not receipt.get("prefix"):
            return
        direction = "upload" if "confirmedUploadedBytes" in receipt else "download"
        identity = (owner, receipt["prefix"], direction, receipt.get("manifestSha256"))
        kind = default_kind or receipt.get("kind", "unknown")
        if receipt["prefix"].endswith("/final"):
            kind = "final"
        synthetic = bool(receipt.get("synthetic", False) or "synthetic" in kind)
        row = dict(worker=owner[0], execution=owner[1], kind=kind, direction=direction,
            prefix=receipt["prefix"], manifestSha256=receipt.get("manifestSha256"),
            payloadBytes=number(receipt.get("payloadBytes")), metadataBytes=number(receipt.get("metadataBytes")),
            confirmedBytes=number(receipt.get("confirmedUploadedBytes" if direction == "upload" else "confirmedDownloadedBytes")),
            elapsedSeconds=number(receipt.get("elapsedSeconds")), retryCount=number(receipt.get("retryCount")), synthetic=synthetic)
        if identity in transfer_seen:
            index = transfer_seen[identity]
            if row["confirmedBytes"] > transfers[index]["confirmedBytes"]:
                transfers[index] = row
            return
        transfer_seen[identity] = len(transfers)
        transfers.append(row)
    for value, _, _ in runtime.values():
        for receipt in value.get("transfers", []):
            add_transfer(receipt, (value.get("worker", ""), value.get("execution", "")))
    for value in health_records:
        slot = value.get("slot")
        owner = (value.get("worker") or (f"worker-{slot:02d}" if type(slot) is int else "unknown"), str(value.get("execution", value.get("run_id", ""))))
        for field in ("finalReceipt", "final_receipt", "finalTransfer", "final_transfer"):
            if value.get(field):
                add_transfer(value[field], owner, "final")

    games_table, cells_table = [], []
    seen_games = {}
    for identity, (record, _, source) in sorted(native.items()):
        summary = record["summary"]
        config = summary.get("configuration", {})
        group_events = event_groups.get(identity[:3], [])
        native_searches = [row["data"] for row in group_events if row["eventType"] == "search_completed"]
        metrics = []
        for game in summary.get("games", []):
            game_key = (identity[0], identity[1], game.get("gameId"))
            if game_key in seen_games:
                if canonical(seen_games[game_key]) != canonical(game):
                    issues.append(dict(kind="conflicting-game-identity", identity=list(game_key)))
                continue
            seen_games[game_key] = game
            metrics.append(game)
            shard = game.get("shard") or {}
            published = game.get("status") == "terminal" and bool(shard)
            games_table.append(dict(runId=identity[0], worker=identity[1], cell=identity[2], nativeExecution=identity[3],
                gameId=game.get("gameId"), preset=config.get("preset"), budget=config.get("simulations"), concurrency=config.get("concurrency"),
                status=game.get("status"), censored=game.get("status") != "terminal", published=published,
                round=game.get("round"), wallSeconds=ratio(game.get("elapsedMilliseconds"), 1000) if game.get("elapsedMilliseconds") is not None else None,
                actions=number(game.get("actualNativeActions")), searches=number(game.get("searches")),
                simulations=number(game.get("simulations")), searchedPrefixEdges=number(game.get("searchedPrefixEdges")),
                trainingRecords=number(game.get("trainingRecords")), publishedRecords=number(shard.get("decisions")) if published else 0,
                rawTensorBytes=number(game.get("rawTensorBytes")), compressedTensorBytes=number(game.get("compressedTensorBytes")),
                logicalUploadBytes=number(shard.get("uploadBytes")), sparsePolicyBytes=number(game.get("sparsePolicyBytes")),
                exactReplay=game.get("exactReplay")))
        terminal = [game for game in metrics if game.get("status") == "terminal"]
        published = [game for game in terminal if game.get("shard")]
        counts = Counter(game.get("status", "unknown") for game in metrics)
        active_seconds = ratio(summary.get("elapsedMilliseconds"), 1000) if summary.get("elapsedMilliseconds") is not None else None
        supervisor = record.get("supervisorCell") or {}
        def total(field): return sum(number(game.get(field)) for game in metrics)
        records = sum(number(game.get("shard", {}).get("decisions")) for game in published)
        raw = sum(number(game.get("rawTensorBytes")) for game in published)
        compressed = sum(number(game.get("compressedTensorBytes")) for game in published)
        upload = sum(number(game["shard"].get("uploadBytes")) for game in published)
        ending = summary.get("endingMemory", {})
        peak = max([number(ending.get("peakRssBytes"))] + [number(row.get("data", {}).get("process", {}).get("peakRssBytes")) for row in group_events])
        inference = summary.get("inference", {})
        row = dict(runId=identity[0], worker=identity[1], cell=identity[2], nativeExecution=identity[3], source=source,
            preset=config.get("preset"), budget=config.get("simulations"), concurrency=config.get("concurrency"), cpuWorkers=config.get("cpuWorkers"),
            batchSize=config.get("batchSize"), batchDelayMs=config.get("batchDelayMs"), replayEnabled=config.get("replay"),
            observedGames=len(metrics), completedGames=len(terminal), abortedGames=sum(counts[status] for status in ("time-limit", "canceled")),
            errorGames=sum(count for status, count in counts.items() if status not in ("terminal", "time-limit", "canceled", "censored-no-final")),
            censoredGames=len(metrics) - len(terminal), unpublishedGames=len(metrics) - len(published),
            unstartedGames=max(0, int(config.get("games", 0)) - len(metrics)) if config.get("games", 0) else None,
            incompleteSummary=bool(summary.get("incompleteSummary")), statusCounts=dict(counts), stopReason=summary.get("stopReason"),
            activeSeconds=active_seconds, supervisorCellSeconds=supervisor.get("elapsedSeconds"),
            completedGamesPerHour=ratio(len(terminal) * 3600, active_seconds), searchesPerSecond=ratio(total("searches"), active_seconds),
            simulationsPerSecond=ratio(total("simulations"), active_seconds), nativeActionsPerSecond=ratio(total("actualNativeActions"), active_seconds),
            publishedRecordsPerSecond=ratio(records, active_seconds),
            completedGameWallSeconds=distribution(number(game.get("elapsedMilliseconds"), None) / 1000 for game in terminal if number(game.get("elapsedMilliseconds"), None) is not None),
            censoredObservedWallSeconds=distribution(number(game.get("elapsedMilliseconds"), None) / 1000 for game in metrics if game.get("status") != "terminal" and number(game.get("elapsedMilliseconds"), None) is not None),
            completedRounds=distribution(game.get("round") for game in terminal), actions=total("actualNativeActions"), searches=total("searches"),
            simulations=total("simulations"), attemptedTrainingRecords=total("trainingRecords"), publishedTrainingRecords=records,
            prefixLengths=histogram(game.get("playedPrefixLengths") for game in metrics),
            rootCandidateCounts=histogram(game.get("rootCandidateCounts") for game in metrics),
            searchLatencyMilliseconds=distribution(search.get("elapsedMilliseconds") for search in native_searches),
            gameAverageSearchMilliseconds=distribution(ratio(game.get("searchMilliseconds", 0), game.get("searches", 0)) for game in metrics if game.get("searches")),
            peakRssBytes=peak or None, cpuSeconds=summary.get("cpuSeconds"), averageCpuCores=ratio(number(summary.get("cpuSeconds")), active_seconds),
            publishedRawTensorBytes=raw, publishedCompressedTensorBytes=compressed, logicalReplayUploadBytes=upload,
            rawBytesPerPublishedGame=ratio(raw, len(published)), compressedBytesPerPublishedGame=ratio(compressed, len(published)),
            logicalUploadBytesPerPublishedGame=ratio(upload, len(published)), rawBytesPerPublishedDecision=ratio(raw, records),
            compressedBytesPerPublishedDecision=ratio(compressed, records), logicalUploadBytesPerPublishedDecision=ratio(upload, records),
            compressionRatio=ratio(raw, compressed), inference=inference, cpuScheduling=summary.get("cpuScheduling", {}), model=summary.get("model", {}),
            mcts={field: total(field) for field in ("nodes", "actionSlots", "modelRequests", "uctNodes", "cachedTraversals", "inheritedVisits", "reusedRoots", "traversalNativeActions", "traversalDecisionEdges")},
            versions=summary.get("versions", {}))
        cells_table.append(row)

    runs_table = []
    for identity, (value, _, source) in sorted(runtime.items()):
        runs_table.append(dict(commit=identity[0], worker=identity[1], execution=identity[2], source=source,
            finalPresent=True, cells=len(value.get("cells", [])), elapsedSeconds=value.get("elapsedSeconds"),
            failures=value.get("failures", []), failureCount=len(value.get("failures", [])), replayRoundtripValidated=value.get("replayRoundtripValidated"),
            mockModel=value.get("mockModel"), gpuCompute=value.get("gpuCompute"), storeCounters=value.get("storeCounters", {})))
    for identity, (record, _, _) in native.items():
        if not any(value[0].get("worker") == identity[1] and any(summary.get("runId") == identity[0]
            for cell in value[0].get("cells", []) for summary in cell.get("summaries", [])) for value in runtime.values()):
            key = (identity[0], identity[1])
            if not any(row.get("recoveredRunId") == key[0] and row["worker"] == key[1] for row in runs_table):
                runs_table.append(dict(recoveredRunId=key[0], worker=key[1], finalPresent=False, source="verified-log-recovery"))
    transfer_totals = []
    for kind, direction, synthetic in sorted({(item["kind"], item["direction"], item["synthetic"]) for item in transfers}):
        rows = [item for item in transfers if (item["kind"], item["direction"], item["synthetic"]) == (kind, direction, synthetic)]
        transfer_totals.append(dict(kind=kind, direction=direction, synthetic=synthetic, operations=len(rows),
            confirmedBytes=sum(item["confirmedBytes"] for item in rows), elapsedSeconds=sum(item["elapsedSeconds"] for item in rows),
            payloadBytes=sum(item["payloadBytes"] for item in rows), metadataBytes=sum(item["metadataBytes"] for item in rows)))
    return dict(format="swarmada.worker-experiment-report.v1", scope="Offline checksum-verified worker artifacts. CPU tensor-reading random policy/value only; no trained model, neural GPU throughput or playing-strength claim. Throughput uses each native cell's active wall time, never summed concurrent game durations. Nonterminal games are censored and excluded from completed-game duration quantiles.",
        verifiedArtifacts=len(artifacts), duplicateArtifactCopies=duplicate_artifacts, recoveredLogSources=len(reconstructed),
        runs=runs_table, cells=cells_table, games=games_table, transfers=transfers, transferTotals=transfer_totals, issues=issues)


def flatten(value, prefix="") -> dict:
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        elif isinstance(item, list):
            # Keep histograms/details in JSON; CSV remains a compact scalar table.
            continue
        else:
            result[name] = item
    return result


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar = [flatten(row) for row in rows]
    names = list(dict.fromkeys(key for row in scalar for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(scalar)


def compact_cells(rows):
    base = ("runId", "worker", "cell", "nativeExecution", "preset", "budget", "concurrency", "cpuWorkers",
        "batchSize", "batchDelayMs", "replayEnabled", "observedGames", "completedGames", "abortedGames", "errorGames",
        "censoredGames", "unstartedGames", "incompleteSummary", "stopReason", "activeSeconds", "supervisorCellSeconds",
        "completedGamesPerHour", "searchesPerSecond", "simulationsPerSecond", "nativeActionsPerSecond", "publishedRecordsPerSecond",
        "actions", "searches", "simulations", "attemptedTrainingRecords", "publishedTrainingRecords", "peakRssBytes", "cpuSeconds",
        "averageCpuCores", "publishedRawTensorBytes", "publishedCompressedTensorBytes", "logicalReplayUploadBytes",
        "rawBytesPerPublishedGame", "compressedBytesPerPublishedGame", "logicalUploadBytesPerPublishedGame",
        "rawBytesPerPublishedDecision", "compressedBytesPerPublishedDecision", "logicalUploadBytesPerPublishedDecision", "compressionRatio")
    for row in rows:
        result = {key: row.get(key) for key in base}
        for key in ("completedGameWallSeconds", "censoredObservedWallSeconds", "completedRounds", "prefixLengths",
            "rootCandidateCounts", "searchLatencyMilliseconds", "gameAverageSearchMilliseconds"):
            for field in ("count", "average", "median", "p95", "maximum"):
                result[key + "." + field] = row.get(key, {}).get(field)
        for field in ("requests", "batches", "averageBatch", "maximumBatch", "averageQueueMilliseconds", "modelMilliseconds"):
            result["inference." + field] = row.get("inference", {}).get(field)
        yield result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-prefix", type=Path)
    parser.add_argument("--health", action="append", default=[], type=Path)
    parser.add_argument("--reconstruct", type=Path)
    parser.add_argument("--no-logs", action="store_true")
    args = parser.parse_args()
    report = build_report(args.roots, health_files=args.health, reconstruct=args.reconstruct, include_logs=not args.no_logs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.csv_prefix:
        for table in ("cells", "games", "transfers", "runs", "transferTotals"):
            write_csv(Path(str(args.csv_prefix) + "." + table + ".csv"), list(compact_cells(report[table])) if table == "cells" else report[table])
    print(json.dumps(dict(verifiedArtifacts=report["verifiedArtifacts"], cells=len(report["cells"]), games=len(report["games"]),
        completedGames=sum(row["completedGames"] for row in report["cells"]), censoredGames=sum(row["censoredGames"] for row in report["cells"]),
        issues=len(report["issues"]), report=str(args.output))))
    return 0 if report["cells"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
