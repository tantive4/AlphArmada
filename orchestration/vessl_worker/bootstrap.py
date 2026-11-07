"""One bounded Linux worker run. No trainer, optimizer, seed, or automatic restart."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request

from settings import atomic_json, load, sha256

_children: set[subprocess.Popen] = set()


def hard_deadline(_signum, _frame) -> None:
    # Network libraries can be waiting/retrying when the wall-clock limit lands.
    # This independent process alarm covers those operations as well as run().
    for child in tuple(_children):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    print("Absolute worker deadline reached; exiting without restart", file=sys.stderr, flush=True)
    os._exit(124)


def requested_stop(_signum, _frame) -> None:
    for child in tuple(_children):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    os._exit(143)


def extract(archive: Path, destination: Path) -> None:
    """Extract ordinary files/directories only, without links or path escapes."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as contents:
        for member in contents:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name or not (member.isfile() or member.isdir()):
                raise ValueError("Archive contains an unsupported path/link")
            target = destination.joinpath(*name.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with contents.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)


def run(command: list[str], deadline: float, *, cwd: Path | None = None, capture: bool = False) -> str:
    remaining = deadline - time.time()
    if remaining <= 5:
        raise TimeoutError("Absolute worker deadline reached")
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True,
        stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None, text=True)
    _children.add(process)
    try:
        output, _ = process.communicate(timeout=max(.1, remaining - 5))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=max(.1, min(4, deadline - time.time())))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1)
        raise TimeoutError("Worker command stopped at its absolute deadline")
    if process.returncode:
        _children.discard(process)
        raise RuntimeError(f"Worker command failed with exit code {process.returncode}: {command[0]}")
    _children.discard(process)
    if capture and output:
        print(output, end="", flush=True)
    return output or ""


def download(url: str, target: Path, deadline: float) -> None:
    if time.time() >= deadline - 5:
        raise TimeoutError("No download after the absolute worker deadline")
    with urllib.request.urlopen(url, timeout=min(60, deadline - time.time())) as source, target.open("wb") as output:
        while True:
            if time.time() >= deadline - 5:
                raise TimeoutError("Download stopped at the absolute worker deadline")
            block = source.read(1024 * 1024)
            if not block:
                return
            output.write(block)


def install_dotnet(version: str, directory: Path, deadline: float) -> Path:
    current = shutil.which("dotnet")
    if current and run([current, "--version"], deadline, capture=True).strip() == version:
        return Path(current)
    archive = directory.parent / "dotnet-sdk.tar.gz"
    checksum = directory.parent / "dotnet-sdk.sha512"
    url = f"https://builds.dotnet.microsoft.com/dotnet/Sdk/{version}/dotnet-sdk-{version}-linux-x64.tar.gz"
    download(url, archive, deadline)
    download(url + ".sha512", checksum, deadline)
    expected = checksum.read_text().strip().split()[0].lower()
    if len(expected) != 128:
        raise ValueError("Official .NET SDK SHA512 response is invalid")
    digest = hashlib.sha512()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("Official .NET SDK archive checksum mismatch")
    extract(archive, directory)
    binary = directory / "dotnet"
    os.environ["DOTNET_ROOT"] = str(directory)
    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
    if run([str(binary), "--version"], deadline, capture=True).strip() != version:
        raise ValueError("Installed .NET SDK version differs from its pinned version")
    return binary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--slot", type=int, required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load(config_path, launching=True)
    if not 1 <= args.slot <= 20:
        raise ValueError("Worker slot must be 1..20")
    deadline = min(config["deadline_unix"], time.time() + config["per_run_seconds"])
    signal.signal(signal.SIGALRM, hard_deadline)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()))
    signal.signal(signal.SIGTERM, requested_stop)
    ops = Path(__file__).resolve().parent
    manifest = json.loads((ops / "ops-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(ops / name) != digest:
            raise ValueError("Embedded worker operations checksum mismatch")
    os.environ.update(VESSL_DATA_COLLECTION_OPT_OUT="true", ENABLE_SENTRY="false", VESSL_SAVE_CONFIG="false",
        PYTHONUNBUFFERED="1", DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1", DEBIAN_FRONTEND="noninteractive")
    threads = str(config.get("threads_per_worker", 10))
    os.environ.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
    work = Path("/root/headless-worker")
    work.mkdir(parents=True, exist_ok=True)
    repository = Path("/root/headless-engine")
    run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(ops / "requirements-control.txt")], deadline)
    if shutil.which("git") is None:
        run(["apt-get", "update"], deadline)
        run(["apt-get", "install", "-y", "--no-install-recommends", "git", "ca-certificates"], deadline)
    # Standalone helper is part of the verified operations bundle. This supports
    # a Linux compatibility pilot against the preceding engine-only Git commit.
    from worker_storage import VesslStore, fetch_bundle, publish_bundle
    store = VesslStore(config["organization"], config["volume"], storage=config["storage"], namespace=config["namespace"])
    slot = f"worker-{args.slot:02d}"
    run_id = os.environ.get("VESSL_RUN_ID") or os.environ.get("ALPHARMADA_EXECUTION_ID")
    if not run_id:
        raise ValueError("A remote worker requires a unique run/execution ID")
    prefix = f"{config['namespace']}/artifacts/{slot}/{run_id}"
    def health(phase: str, **extra):
        store.put_json(f"{config['namespace']}/health/{slot}.json", dict(phase=phase, time=time.time(), slot=args.slot,
            run_id=run_id, deadline_unix=deadline, commit=config["repository_commit"], **extra))
    health("bootstrap")
    repository.mkdir(parents=True, exist_ok=True)
    run(["git", "init", str(repository)], deadline)
    run(["git", "-C", str(repository), "remote", "add", "origin", config["repository_url"]], deadline)
    run(["git", "-C", str(repository), "fetch", "--depth=1", "origin", config["repository_commit"]], deadline)
    run(["git", "-C", str(repository), "checkout", "--detach", "FETCH_HEAD"], deadline)
    actual_commit = run(["git", "-C", str(repository), "rev-parse", "HEAD"], deadline, capture=True).strip()
    if actual_commit != config["repository_commit"]:
        raise ValueError("Fetched source differs from the exact pinned commit")
    dotnet = install_dotnet(config["dotnet_sdk_version"], work / "dotnet", deadline)
    for project in config.get("dotnet_projects", []):
        run([str(dotnet), "build", project, "-c", "Release", "-p:UseAppHost=false", "-p:UseSharedCompilation=false"], deadline, cwd=repository)
    if config.get("pilot_mode") == "compatibility":
        assembly = repository / "src/Swarmada.Headless.Cli/bin/Release/net9.0/Swarmada.Headless.Cli.dll"
        result = json.loads(run([str(dotnet), str(assembly), "status", "--data", str(repository / "data")], deadline, capture=True))
        if result.get("protocolVersion") != 1 or not result.get("headlessMatchReady"):
            raise ValueError("Linux CLI compatibility check failed")
        record = dict(version=1, mode="compatibility", slot=args.slot, run_id=run_id, commit=actual_commit,
            dotnet_sdk=config["dotnet_sdk_version"], status=result, time=time.time())
        artifact_directory = work / "compatibility-artifact"
        artifact_directory.mkdir(exist_ok=True)
        artifact = artifact_directory / "compatibility.json"
        atomic_json(artifact, record)
        upload = publish_bundle(store, artifact_directory, prefix,
            dict(kind="logs", versions={"code": actual_commit}, commit=actual_commit, mode="compatibility"))
        downloaded = work / "compatibility-roundtrip"
        fetched = fetch_bundle(store, prefix, downloaded, kind="logs", expected_versions={"code": actual_commit})
        if (downloaded / "compatibility.json").read_bytes() != artifact.read_bytes():
            raise ValueError("Remote artifact upload/download roundtrip failed")
        health("compatibility-complete", artifact_prefix=prefix, upload=upload, download=fetched)
        return 0
    runtime_publication = work / "runtime-publication"
    fetch_bundle(store, config["runtime_key"].rsplit("/", 1)[0], runtime_publication,
        kind="runtime", expected_versions={"code": actual_commit})
    package_manifest = json.loads((runtime_publication / "manifest.json").read_text())
    if package_manifest.get("sha256") != config["runtime_sha256"] or package_manifest.get("commit") != actual_commit:
        raise ValueError("Published runtime package is incompatible with this source commit")
    archive = runtime_publication / "runtime.tar.gz"
    if sha256(archive) != config["runtime_sha256"]:
        raise ValueError("Runtime archive checksum mismatch")
    package = work / "runtime"
    extract(archive, package)
    entry = package / config["runtime_entry"]
    if not entry.is_file():
        raise ValueError("Verified runtime archive does not contain the configured entry point")
    os.environ.update(ALPHARMADA_REPOSITORY=str(repository), ALPHARMADA_WORK_ROOT=str(work),
        ALPHARMADA_WORKER_SLOT=str(args.slot), ALPHARMADA_DEADLINE_UNIX=str(deadline), ALPHARMADA_ARTIFACT_PREFIX=prefix)
    health("runtime-starting", artifact_prefix=prefix, runtime_sha256=config["runtime_sha256"])
    run([sys.executable, "-u", str(entry), "--config", str(config_path), "--slot", str(args.slot),
        "--deadline-unix", str(deadline), "--repository", str(repository), "--work-root", str(work),
        *config.get("runtime_arguments", [])], deadline, cwd=repository)
    # The runtime publishes its final_prefix and final transfer receipt in this
    # same health object. Do not replace that richer completion record here.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Worker bootstrap failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
