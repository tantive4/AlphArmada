"""Scoped, explicit control for twenty bounded headless test workers only."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import time
import uuid

from settings import TERMINAL, atomic_json, load, sha256

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY / "python"))


def backend(config):
    from swarmada_headless.worker_storage import VesslStore
    return VesslStore(config["organization"], config["volume"], storage=config["storage"], namespace=config["namespace"])


def raw(config, method, **kwargs):
    from swarmada_headless.worker_storage import vessl_raw
    return vessl_raw(method, organization_name=config["organization"], project_name=config["project"], **kwargs)


def directory(config):
    path = REPOSITORY / ".local" / "vessl-worker" / config["namespace"]
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def locked(config):
    with (directory(config) / "controller.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def matching_runs(config):
    expression = re.compile(r"^aa-" + re.escape(config["namespace"]) + r"-worker-(\d{2})-[0-9a-f]{12}$")
    result = []
    offset = 0
    while True:
        page = raw(config, "run_execution_list_api", limit=100, offset=offset)
        for run in page["results"]:
            title = run["run_spec"]["title"]
            match = expression.fullmatch(title)
            if match and 1 <= int(match[1]) <= 20:
                result.append(dict(id=run["id"], name=title, slot=int(match[1]), status=run["status"], reason=run.get("status_reason")))
        if len(page["results"]) < 100:
            return result
        offset += 100


def ops_bundle(config):
    members = {name: (HERE / name).read_bytes() for name in ("bootstrap.py", "settings.py", "requirements-control.txt")}
    members["worker_storage.py"] = (REPOSITORY / "python/swarmada_headless/worker_storage.py").read_bytes()
    members["config.json"] = json.dumps(config, sort_keys=True).encode()
    manifest = dict(version=1, commit=config["repository_commit"], runtime_sha256=config["runtime_sha256"],
        files={name: hashlib.sha256(data).hexdigest() for name, data in members.items()})
    members["ops-manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))
    content = buffer.getvalue()
    return base64.b64encode(content).decode(), hashlib.sha256(content).hexdigest(), manifest


def definition(config, slot, encoded, bundle_hash):
    name = f"aa-{config['namespace']}-worker-{slot:02d}-{uuid.uuid4().hex[:12]}"
    command = (
        "set -eu\nmkdir -p /opt/alpharmada-headless-ops\n"
        f"printf %s {encoded} | base64 -d > /opt/alpharmada-headless-ops/ops.tar.gz\n"
        f"printf '%s  %s\\n' '{bundle_hash}' '/opt/alpharmada-headless-ops/ops.tar.gz' | sha256sum -c -\n"
        "tar -xzf /opt/alpharmada-headless-ops/ops.tar.gz -C /opt/alpharmada-headless-ops\n"
        f"exec python -u /opt/alpharmada-headless-ops/bootstrap.py --config /opt/alpharmada-headless-ops/config.json --slot {slot}\n"
    )
    return dict(name=name, description=f"Bounded headless worker only; commit {config['repository_commit']}; ops {bundle_hash}",
        tags=["alpharmada", "headless-worker", config["namespace"], f"worker-{slot:02d}"],
        resources=dict(cluster=config["cluster"], preset=config["preset"]), image=config["image"],
        run=[dict(command=command, workdir="/root")],
        env=dict(VESSL_DEFAULT_ORGANIZATION=config["organization"], VESSL_DEFAULT_PROJECT=config["project"],
            PYTHONUNBUFFERED="1", VESSL_DATA_COLLECTION_OPT_OUT="true", ENABLE_SENTRY="false", VESSL_SAVE_CONFIG="false",
            ALPHARMADA_EXECUTION_ID=name[-12:]))


def prepare(config, config_path):
    package = Path(config["runtime_package"])
    if not package.is_absolute():
        package = config_path.parent / package
    if sha256(package) != config["runtime_sha256"]:
        raise ValueError("Local runtime package bytes do not match runtime_sha256")
    import vessl.storage
    # Authentication is initialized through the helper, which restores SDK TLS.
    from swarmada_headless.worker_storage import vessl_api
    vessl_api()
    volumes = vessl.storage.list_volumes(storage_name=config["storage"], keyword=config["volume"], organization_name=config["organization"])
    if config["volume"] not in {volume.name for volume in volumes}:
        vessl.storage.create_volume(config["volume"], config["storage"], tags=("alpharmada", "headless-worker", config["namespace"]),
            organization_name=config["organization"])
    store = backend(config)
    from swarmada_headless.worker_storage import publish_bundle
    expected = dict(version=1, kind="runtime", versions={"code": config["repository_commit"]},
        commit=config["repository_commit"], sha256=config["runtime_sha256"],
        key=config["runtime_key"], size=package.stat().st_size)
    with tempfile.TemporaryDirectory(prefix="runtime-publication-", dir=directory(config)) as temporary:
        staging = Path(temporary)
        shutil.copyfile(package, staging / "runtime.tar.gz")
        receipt = publish_bundle(store, staging, config["runtime_key"].rsplit("/", 1)[0], expected)
    atomic_json(directory(config) / "prepared.json", expected)
    print(json.dumps(dict(prepared=True, volume=config["volume"], namespace=config["namespace"], runtime=expected, transfer=receipt), indent=2))


def launch(config, args):
    if config["deadline_unix"] - time.time() < 180:
        raise ValueError("Fewer than three minutes remain before the global deadline; no new run is started")
    if args.command == "launch" and not args.pilot_reviewed:
        raise ValueError("Inspect the single-worker pilot first, then explicitly pass --pilot-reviewed")
    if args.command == "launch" and args.compatibility:
        raise ValueError("Compatibility mode is restricted to the single-worker pilot")
    if args.command == "pilot" and args.compatibility:
        config = dict(config, pilot_mode="compatibility")
    elif backend(config).get_json(config["runtime_manifest_key"]).get("sha256") != config["runtime_sha256"]:
        raise ValueError("Prepare the pinned runtime package before launching workers")
    journal_path = directory(config) / "fleet.json"
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    disabled_path = directory(config) / "disabled.json"
    disabled = set(json.loads(disabled_path.read_text())) if disabled_path.exists() else set()
    runs = matching_runs(config)
    if args.command == "launch" and not any(run["slot"] == 1 for run in runs):
        raise ValueError("A single-worker pilot must exist before expanding the fleet")
    active = [run for run in runs if run["status"] not in TERMINAL]
    if len(active) > 20 or len({run["slot"] for run in active}) != len(active):
        raise RuntimeError("Duplicate or excess active workers detected; refusing additional creation")
    encoded, bundle_hash, manifest = ops_bundle(config)
    atomic_json(directory(config) / "ops-manifest.json", manifest)
    slots = [1] if args.command == "pilot" else list(range(1, 21))
    for slot in slots:
        if time.time() >= config["deadline_unix"] - 180:
            print("Absolute deadline is near; stopped launching additional slots")
            break
        key = str(slot)
        prior = journal.get(key, {})
        live = [run for run in active if run["slot"] == slot]
        if live:
            print(json.dumps(dict(slot=slot, existing=live[0])))
            continue
        if key in disabled:
            print(json.dumps(dict(slot=slot, disabled=True)))
            continue
        intent = prior.get("intent")
        if intent:
            observed = [run for run in runs if run["name"] == intent]
            if observed:
                prior.update(id=observed[0]["id"], intent=None)
                journal[key] = prior
                atomic_json(journal_path, journal)
                if observed[0]["status"] not in TERMINAL:
                    continue
            elif not args.retry_missing_intents:
                raise RuntimeError(f"Unresolved creation intent for slot {slot}; server listing was reconciled, no POST retried. Inspect it before --retry-missing-intents")
        if any(run["slot"] == slot for run in runs) and not args.replace_terminal:
            print(json.dumps(dict(slot=slot, terminal=True, action="use --replace-terminal only after reviewing previous run")))
            continue
        if len(active) >= 20:
            raise RuntimeError("Twenty active workers already exist")
        spec = definition(config, slot, encoded, bundle_hash)
        journal[key] = dict(intent=spec["name"], time=time.time(), runtime_sha256=config["runtime_sha256"],
            commit=config["repository_commit"], deadline_unix=config["deadline_unix"])
        atomic_json(journal_path, journal)
        # Store intent before POST. Never retry a create request automatically.
        response = raw(config, "run_spec_create_from_yamlapi",
            run_spec_create_from_yamlapi_input={"yaml_spec": __import__("yaml").safe_dump(spec, sort_keys=False)})
        run = response["run_execution"]
        journal[key].update(id=run["id"], intent=None)
        atomic_json(journal_path, journal)
        active.append(dict(id=run["id"], slot=slot, name=spec["name"], status=run.get("status", "pending")))
        print(json.dumps(dict(slot=slot, id=run["id"], name=spec["name"], deadline_unix=config["deadline_unix"])), flush=True)


def stop(config):
    atomic_json(directory(config) / "disabled.json", [str(slot) for slot in range(1, 21)])
    stopped, errors = [], []
    for run in matching_runs(config):
        if run["status"] not in TERMINAL:
            try:
                raw(config, "run_execution_terminate_api", execution_id=run["id"])
                stopped.append(run["id"])
            except Exception as error:
                errors.append(dict(id=run["id"], error=type(error).__name__))
    print(json.dumps(dict(termination_requested=stopped, errors=errors, namespace=config["namespace"], storage_preserved=True)))
    if errors:
        raise RuntimeError("Some termination requests failed; all matching runs were attempted, verify server status")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    for name in ("pilot", "launch"):
        command = commands.add_parser(name)
        command.add_argument("--replace-terminal", action="store_true")
        command.add_argument("--retry-missing-intents", action="store_true")
        command.add_argument("--pilot-reviewed", action="store_true")
        command.add_argument("--compatibility", action="store_true")
    commands.add_parser("status")
    commands.add_parser("health")
    command = commands.add_parser("logs")
    command.add_argument("id", type=int)
    command.add_argument("--tail", type=int, default=100)
    command = commands.add_parser("fetch")
    command.add_argument("--destination", type=Path)
    command.add_argument("--prefix")
    commands.add_parser("stop")
    commands.add_parser("watch-deadline")
    args = parser.parse_args()
    config = load(args.config, launching=args.command in ("prepare", "pilot", "launch"))
    if args.command == "prepare":
        with locked(config):
            prepare(config, args.config.resolve())
    elif args.command in ("pilot", "launch"):
        with locked(config):
            launch(config, args)
    elif args.command == "status":
        result = dict(namespace=config["namespace"], deadline_unix=config["deadline_unix"], runs=matching_runs(config))
        atomic_json(directory(config) / "status.json", result)
        print(json.dumps(result, indent=2))
    elif args.command == "health":
        store = backend(config)
        result = [store.get_json(item["key"]) for item in store.list(f"{config['namespace']}/health/") if item["key"].endswith(".json")]
        atomic_json(directory(config) / "health.json", result)
        print(json.dumps(result, indent=2))
    elif args.command == "logs":
        if args.id not in {run["id"] for run in matching_runs(config)}:
            raise ValueError("Run ID does not belong to this headless namespace")
        result = raw(config, "run_execution_logs_api", execution_id=args.id, log_limit=args.tail)
        for line in result.get("logs", []):
            print(line["message"].replace("\\n", "\n"))
    elif args.command == "fetch":
        from swarmada_headless.worker_storage import fetch_ready_artifact
        store = backend(config)
        destination = (args.destination or directory(config) / "artifacts").resolve()
        prefix = args.prefix or f"{config['namespace']}/artifacts/"
        receipts = []
        ready = [item["key"].removesuffix("/READY") for item in store.list(prefix) if item["key"].endswith("/READY")]
        for artifact in ready:
            relative = PurePosixPath(artifact).relative_to(config["namespace"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError("Artifact path is not a safe relative path")
            receipts.append(fetch_ready_artifact(store, artifact, destination.joinpath(*relative.parts)))
        print(json.dumps(dict(verified_artifacts=len(receipts), destination=str(destination), transfers=receipts)))
    elif args.command == "stop":
        with locked(config):
            stop(config)
    elif args.command == "watch-deadline":
        # No restart loop. This optional independent controller only enforces stop.
        while time.time() < config["deadline_unix"]:
            time.sleep(max(0, min(30, config["deadline_unix"] - time.time())))
        with locked(config):
            stop(config)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Worker controller failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
