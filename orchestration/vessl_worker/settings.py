"""Validated worker-only deployment settings; contains no authentication data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse


MAX_WORKERS = 20
MAX_SESSION_SECONDS = 4 * 60 * 60
TERMINAL = frozenset({"failed", "terminated", "completed", "stopped", "cancelled", "canceled"})


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: str | Path, *, launching: bool = False) -> dict:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")
    def reject_secrets(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if re.search(r"secret|password|credential|access.?key|api.?key|(^|_)token($|_)", key, re.I):
                    raise ValueError("Do not put authentication fields into deployment configuration")
                reject_secrets(item)
        elif isinstance(value, list):
            for item in value:
                reject_secrets(item)
    reject_secrets(config)
    if config.get("worker_only") is not True or config.get("workers") != MAX_WORKERS:
        raise ValueError("This controller supports exactly 20 worker slots and no trainer")
    for key in ("organization", "project", "cluster", "preset", "image", "volume", "namespace"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing configuration string: {key}")
    if not re.fullmatch(r"headless-[a-z0-9][a-z0-9-]{2,60}", config["namespace"]):
        raise ValueError("Use a fresh headless-* namespace")
    if not re.fullmatch(r"alpharmada-headless-[a-z0-9][a-z0-9-]{2,60}", config["volume"]):
        raise ValueError("Use a fresh alpharmada-headless-* volume; old training volumes are excluded")
    if not re.fullmatch(r"[0-9a-f]{40}", config.get("repository_commit", "")):
        raise ValueError("repository_commit must be an exact lowercase 40-character Git commit")
    source = urlparse(config.get("repository_url", ""))
    if source.scheme != "https" or source.netloc != "github.com" or source.username or source.password or source.query or source.fragment:
        raise ValueError("repository_url must be a public HTTPS github.com URL without embedded credentials")
    if not re.fullmatch(r"[0-9a-f]{64}", config.get("runtime_sha256", "")):
        raise ValueError("runtime_sha256 must pin the exact runtime archive bytes")
    for name in ("runtime_entry",):
        value = config.get(name, "")
        if not isinstance(value, str) or "\\" in value or Path(value).is_absolute() or ".." in Path(value).parts or not value.endswith(".py"):
            raise ValueError("runtime_entry must be a relative Python file inside the runtime package")
    start, deadline = config.get("started_unix"), config.get("deadline_unix")
    if not isinstance(start, (int, float)) or not isinstance(deadline, (int, float)) or start <= 0 or not 0 < deadline - start <= MAX_SESSION_SECONDS:
        raise ValueError("Set the absolute user-start/deadline times with a session lasting at most four hours")
    if type(config.get("per_run_seconds")) is not int or not 1 <= config["per_run_seconds"] <= MAX_SESSION_SECONDS:
        raise ValueError("per_run_seconds must be between 1 and 14400")
    if not re.fullmatch(r"9\.0\.\d{3}", config.get("dotnet_sdk_version", "")):
        raise ValueError("Pin a .NET 9 SDK version")
    for project in config.get("dotnet_projects", []):
        if not isinstance(project, str) or "\\" in project or Path(project).is_absolute() or ".." in Path(project).parts or not project.endswith(".csproj"):
            raise ValueError("dotnet_projects must contain repository-relative .csproj paths")
    for name in ("runtime_arguments", "native_command"):
        if not isinstance(config.get(name, []), list) or any(not isinstance(item, str) for item in config.get(name, [])):
            raise ValueError(f"{name} must be a list of literal string arguments")
    if type(config.get("threads_per_worker", 10)) is not int or not 1 <= config.get("threads_per_worker", 10) <= 64:
        raise ValueError("threads_per_worker must be an integer between 1 and 64")
    if launching:
        if config.get("integration_reviewed") is not True:
            raise ValueError("Set integration_reviewed=true only after reviewing this worker-only deployment")
        if time.time() >= deadline:
            raise ValueError("The absolute four-hour test deadline has passed; no run may be launched")
    config["storage"] = config.get("storage", "vessl-storage")
    config["runtime_key"] = f"{config['namespace']}/runtime/{config['runtime_sha256']}/runtime.tar.gz"
    config["runtime_manifest_key"] = f"{config['namespace']}/runtime/{config['runtime_sha256']}/manifest.json"
    return config


def atomic_json(path: str | Path, data) -> None:
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
