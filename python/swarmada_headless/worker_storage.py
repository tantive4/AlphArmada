"""Durable, namespaced worker transfers; no training or cloud-run creation.

Native workers finish immutable game directories with a checksummed ``manifest.json``
and an atomically renamed ``READY`` marker. Transfers publish payloads, the exact
manifest bytes, then READY. Downloaders ignore partial remote directories and verify
every file before exposing a local READY directory. Replays contain the actual
lossless tensor records, conditional sparse visits, and genuine terminal auxiliaries.

The VESSL adapter follows the verified sibling template's temporary S3 federation
path. Credentials stay in memory, SDK TLS verification is restored explicitly, and
all object operations are confined to one fresh caller-supplied namespace.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import struct
import tempfile
import time
from typing import Any, Callable, Iterable, Protocol
import uuid


SHARD_FORMAT = "swarmada.worker-shard.v1"
BUNDLE_FORMAT = "swarmada.worker-bundle.v1"
READY_FORMAT = "swarmada.worker-shard-ready.v1"
INPUT_MAGIC = b"SWAREC01"
INPUT_VERSION = 1
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_READY_BYTES = 4096
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INTERNAL_FILES = {"manifest.json", "READY", ".uploaded.json", ".retry.json"}


class StorageIntegrityError(ValueError):
    """A ready artifact is malformed, incompatible, incomplete, or corrupted."""


class StorageConflictError(StorageIntegrityError):
    """An immutable remote/local identity already has different contents."""


class SpoolFullError(RuntimeError):
    """Backpressure: retain pending data and finish uploads before generating more."""


class ObjectStore(Protocol):
    namespace: str

    def list(self, prefix: str = "") -> list[dict[str, Any]]: ...
    def get_json(self, key: str) -> Any: ...
    def put_json(self, key: str, value: Any) -> None: ...
    def upload(self, path: str | Path, key: str) -> dict[str, Any]: ...
    def download(self, key: str, path: str | Path, sha256: str | None = None) -> dict[str, Any]: ...


def digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StorageIntegrityError("Duplicate JSON object key")
        result[key] = value
    return result


def _decode_json(data: bytes) -> Any:
    try:
        return json.loads(data, object_pairs_hook=_unique_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(StorageIntegrityError("Nonfinite JSON number")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StorageIntegrityError("Malformed UTF-8 JSON artifact") from error


def _read_json(path: Path, maximum: int = MAX_MANIFEST_BYTES) -> tuple[Any, bytes]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        raise StorageIntegrityError("Missing, linked, or oversized JSON artifact: " + path.name)
    data = path.read_bytes()
    if len(data) > maximum:
        raise StorageIntegrityError("JSON artifact grew beyond its limit")
    return _decode_json(data), data


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_bytes(path, _json_bytes(value))


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise StorageIntegrityError("Invalid safe identifier: " + label)
    return value


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value or ":" in value:
        raise StorageIntegrityError("Artifact path must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise StorageIntegrityError("Artifact path escapes its directory")
    return path.as_posix()


def _local_file(root: Path, relative: str) -> Path:
    relative = _relative_path(relative)
    root = root.resolve()
    result = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for component in PurePosixPath(relative).parts:
        current /= component
        if current.is_symlink():
            raise StorageIntegrityError("Symlinks are not transferable artifacts")
    if not result.resolve().is_relative_to(root):
        raise StorageIntegrityError("Artifact path escapes its directory")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StorageIntegrityError(label + " must be a nonnegative integer")
    return value


def _manifest_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files or len(files) > 100_000:
        raise StorageIntegrityError("Manifest has no bounded file registry")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise StorageIntegrityError("Invalid manifest file entry")
        name = _relative_path(item.get("path"))
        if name.casefold() in seen or name in _INTERNAL_FILES:
            raise StorageIntegrityError("Duplicate or reserved manifest file path")
        seen.add(name.casefold())
        _nonnegative_integer(item.get("bytes"), "file bytes")
        if not isinstance(item.get("sha256"), str) or not _SHA256.fullmatch(item["sha256"]):
            raise StorageIntegrityError("Invalid file SHA-256")
    return files


def validate_manifest(manifest: Any, *, expected_versions: dict[str, str] | None = None) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("format") not in {SHARD_FORMAT, BUNDLE_FORMAT}:
        raise StorageIntegrityError("Unsupported worker artifact manifest format")
    _manifest_files(manifest)
    if manifest["format"] == SHARD_FORMAT:
        for field in ("runId", "workerId", "shardId"):
            _safe_id(manifest.get(field), field)
        versions = manifest.get("versions")
        if not isinstance(versions, dict) or any(not isinstance(versions.get(field), str) or not versions[field]
                                                  for field in ("schemaVersion", "catalogFingerprint", "codeVersion")):
            raise StorageIntegrityError("Replay manifest needs schema, catalog and code versions")
        games = manifest.get("games")
        if not isinstance(games, list) or not games:
            raise StorageIntegrityError("Replay shard has no genuine terminal game registry")
        seen: set[str] = set()
        for game in games:
            if not isinstance(game, dict) or game.get("status") != "terminal":
                raise StorageIntegrityError("Unfinished/rejected games cannot be ready training data")
            identity = _safe_id(game.get("gameId"), "gameId")
            if identity in seen:
                raise StorageIntegrityError("Duplicate game identity")
            seen.add(identity)
            _nonnegative_integer(game.get("decisions"), "game decisions")
    else:
        if manifest.get("kind") not in {"runtime", "mock-model", "logs"}:
            raise StorageIntegrityError("Only runtime, explicit mock models and logs are worker bundles")
        versions = manifest.get("versions", {})
        if not isinstance(versions, dict):
            raise StorageIntegrityError("Invalid bundle version registry")
    for key, expected in (expected_versions or {}).items():
        if versions.get(key) != expected:
            raise StorageIntegrityError("Artifact version mismatch: " + key)
    return manifest


@dataclass(frozen=True)
class ReadyArtifact:
    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    manifest_bytes: int
    ready_bytes: int

    @property
    def payload_bytes(self) -> int:
        return sum(item["bytes"] for item in self.manifest["files"])

    @property
    def metadata_bytes(self) -> int:
        return self.manifest_bytes + self.ready_bytes

    @property
    def total_bytes(self) -> int:
        return self.payload_bytes + self.metadata_bytes


def validate_ready_artifact(directory: str | Path, *, expected_versions: dict[str, str] | None = None) -> ReadyArtifact:
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise StorageIntegrityError("Ready artifact must be an actual directory")
    ready, ready_data = _read_json(directory / "READY", MAX_READY_BYTES)
    manifest, manifest_data = _read_json(directory / "manifest.json")
    if not isinstance(ready, dict) or ready.get("format") != READY_FORMAT:
        raise StorageIntegrityError("Unsupported READY marker")
    checksum = hashlib.sha256(manifest_data).hexdigest()
    if ready.get("manifestBytes") != len(manifest_data) or ready.get("manifestSha256") != checksum:
        raise StorageIntegrityError("READY marker does not bind the exact manifest bytes")
    manifest = validate_manifest(manifest, expected_versions=expected_versions)
    for item in manifest["files"]:
        path = _local_file(directory, item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise StorageIntegrityError("Artifact file checksum/length mismatch: " + item["path"])
    return ReadyArtifact(directory.resolve(), manifest, checksum, len(manifest_data), len(ready_data))


def seal_directory(directory: str | Path, metadata: dict[str, Any]) -> ReadyArtifact:
    """Finalize a caller-owned artifact directory; native producers use the same contract."""
    directory = Path(directory)
    if (directory / "READY").exists():
        raise StorageConflictError("An already-ready directory is immutable")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise StorageIntegrityError("Symlinks are not transferable artifacts")
        if path.is_file() and path.relative_to(directory).as_posix() not in _INTERNAL_FILES:
            files.append({"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = validate_manifest({**metadata, "files": files})
    atomic_json(directory / "manifest.json", manifest)
    data = (directory / "manifest.json").read_bytes()
    atomic_json(directory / "READY", {"format": READY_FORMAT, "manifestBytes": len(data), "manifestSha256": hashlib.sha256(data).hexdigest()})
    return validate_ready_artifact(directory)


class _Namespaced:
    def __init__(self, namespace: str):
        self.namespace = _safe_id(namespace, "namespace")

    def _key(self, key: str, *, listing: bool = False) -> str:
        key = key or (self.namespace + "/" if listing else "")
        if not isinstance(key, str) or "\\" in key or key.startswith("/") or "\x00" in key:
            raise StorageIntegrityError("Object keys must be relative POSIX paths")
        if any(part in {"", ".", ".."} for part in key.rstrip("/").split("/")):
            raise StorageIntegrityError("Invalid object key")
        if key != self.namespace and not key.startswith(self.namespace + "/"):
            raise StorageIntegrityError("Object operation is outside this fresh worker namespace")
        if not listing and (key == self.namespace or key.endswith("/")):
            raise StorageIntegrityError("Expected an object key, not a prefix")
        if listing and key == self.namespace:
            key += "/"
        return key


class LocalStore(_Namespaced):
    """Same immutable transfer protocol on disk, for offline integration/fault tests."""
    def __init__(self, root: str | Path, namespace: str):
        super().__init__(namespace)
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return _local_file(self.root, self._key(key))

    def list(self, prefix: str = "") -> list[dict[str, Any]]:
        prefix = self._key(prefix, listing=True)
        return [{"key": path.relative_to(self.root).as_posix(), "size": path.stat().st_size}
                for path in sorted((self.root / self.namespace).rglob("*"))
                if path.is_file() and not path.is_symlink() and path.relative_to(self.root).as_posix().startswith(prefix)]

    def get_json(self, key: str) -> Any:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(key)
        return _read_json(path)[0]

    def put_json(self, key: str, value: Any) -> None:
        atomic_json(self._path(key), value)

    def upload(self, path: str | Path, key: str) -> dict[str, Any]:
        started = time.monotonic(); size = Path(path).stat().st_size
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".partial")
        try:
            with Path(path).open("rb") as source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, destination)
            _sync_directory(destination.parent)
            return {"transferredBytes": size, "payloadBytes": size, "elapsedSeconds": time.monotonic() - started, "skipped": False}
        finally:
            temporary.unlink(missing_ok=True)

    def download(self, key: str, path: str | Path, sha256: str | None = None) -> dict[str, Any]:
        started = time.monotonic()
        source = self._path(key)
        if not source.is_file():
            raise FileNotFoundError(key)
        destination = Path(path)
        if destination.is_file() and sha256 and digest(destination) == sha256:
            return {"transferredBytes": 0, "payloadBytes": destination.stat().st_size, "elapsedSeconds": time.monotonic() - started, "skipped": True}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".partial")
        try:
            shutil.copyfile(source, temporary)
            if sha256 and digest(temporary) != sha256:
                raise StorageIntegrityError("Downloaded object checksum mismatch")
            os.replace(temporary, destination)
            return {"transferredBytes": destination.stat().st_size, "payloadBytes": destination.stat().st_size, "elapsedSeconds": time.monotonic() - started, "skipped": False}
        finally:
            temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def vessl_api() -> Any:
    """Verified SDK 0.1.199 setup; never return/log federation tokens outside adapter code."""
    os.environ.setdefault("VESSL_DATA_COLLECTION_OPT_OUT", "true")
    os.environ.setdefault("ENABLE_SENTRY", "false")
    os.environ.setdefault("VESSL_SAVE_CONFIG", "false")
    import vessl
    client = vessl.vessl_api.api_client
    client.configuration.verify_ssl = True
    client.rest_client.pool_manager.connection_pool_kw["cert_reqs"] = "CERT_REQUIRED"
    client.rest_client.pool_manager.clear()
    return vessl.vessl_api


def vessl_raw(method: str, **kwargs: Any) -> Any:
    response = getattr(vessl_api(), method)(_preload_content=False, _request_timeout=(15, 90), **kwargs)
    try:
        return _decode_json(response.data) if response.data else {}
    finally:
        response.release_conn()


class VesslStore(_Namespaced):
    """Federated S3 object access to one explicitly owned fresh VESSL namespace.

    Install the control/runtime pins: vessl==0.1.199, boto3==1.43.118,
    botocore==1.43.118. There is deliberately no remote deletion API.
    """
    def __init__(self, organization: str, volume: str, storage: str = "vessl-storage", *, namespace: str,
                 api_call: Callable[..., Any] = vessl_raw):
        super().__init__(namespace)
        self.organization, self.volume, self.storage = organization, volume, storage
        self._api_call = api_call
        self._clients: dict[str, tuple[float, Any, str, str]] = {}
        self.counters = {"uploadedBytes": 0, "downloadedBytes": 0, "uploadOperations": 0,
                         "downloadOperations": 0, "jsonReadBytes": 0, "jsonWrittenBytes": 0,
                         "federationRefreshes": 0, "expiredTokenRetries": 0}

    def _connection(self, write: bool = False) -> tuple[Any, str, str]:
        mode = "write" if write else "read"
        cached = self._clients.get(mode)
        if cached and time.time() < cached[0]:
            return cached[1:]
        import boto3
        from botocore.config import Config
        federation = self._api_call("volume_v2_federate_api", organization_name=self.organization,
            storage_name=self.storage, volume_name=self.volume,
            volume_v2_federate_api_input={"federation_type": mode})
        self.counters["federationRefreshes"] += 1
        if federation.get("type") != "s3":
            raise RuntimeError("This adapter requires VESSL S3 federation")
        token = federation.get("s3_token") or {}
        credentials = {} if not token else {
            "aws_access_key_id": token["access_key_id"], "aws_secret_access_key": token["secret_access_key"],
            "aws_session_token": token.get("session_token")}
        client = boto3.client("s3", endpoint_url=federation.get("endpoint"),
            region_name=federation.get("region") or "ap-northeast-2", **credentials,
            config=Config(connect_timeout=15, read_timeout=90, retries={"max_attempts": 5, "mode": "standard"},
                s3={"addressing_style": "path" if federation.get("force_path_style") else "auto"}))
        # Shorter than normal federation expiry; expired-token responses force immediate renewal.
        self._clients[mode] = (time.time() + 540, client, federation["bucket_name"], federation["path"].strip("/"))
        return self._clients[mode][1:]

    def _call(self, write: bool, operation: Callable[..., Any]) -> Any:
        from botocore.exceptions import ClientError
        for attempt in range(2):
            try:
                return operation(*self._connection(write))
            except Exception as error:
                # boto3.upload_file wraps ClientError in S3UploadFailedError;
                # download retry wrappers can retain a last_exception instead.
                # Inspect typed causes, never signed-URL-bearing error messages.
                pending = [error]; seen: set[int] = set(); client_error = None
                while pending:
                    current = pending.pop()
                    if id(current) in seen:
                        continue
                    seen.add(id(current))
                    if isinstance(current, ClientError):
                        client_error = current; break
                    pending.extend(value for value in (current.__cause__, current.__context__, getattr(current, "last_exception", None)) if isinstance(value, BaseException))
                if client_error is None:
                    raise
                code = client_error.response.get("Error", {}).get("Code", "")
                if code in {"NoSuchKey", "NotFound", "404"}:
                    raise FileNotFoundError("Remote object does not exist") from None
                if attempt or code not in {"ExpiredToken", "ExpiredTokenException", "InvalidToken", "RequestExpired"}:
                    raise
                self._clients.clear()
                self.counters["expiredTokenRetries"] += 1
        raise RuntimeError("Unreachable federation retry state")

    def list(self, prefix: str = "") -> list[dict[str, Any]]:
        prefix = self._key(prefix, listing=True)
        def operation(client: Any, bucket: str, base: str) -> list[dict[str, Any]]:
            result = []
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=posixpath.join(base, prefix)):
                for item in page.get("Contents", []):
                    key = item["Key"][len(base) + 1:] if base else item["Key"]
                    self._key(key)
                    result.append({"key": key, "size": item["Size"]})
            return result
        return self._call(False, operation)

    def get_json(self, key: str) -> Any:
        key = self._key(key)
        def operation(client: Any, bucket: str, base: str) -> Any:
            body = client.get_object(Bucket=bucket, Key=posixpath.join(base, key))["Body"]
            try:
                data = body.read(MAX_MANIFEST_BYTES + 1)
                if len(data) > MAX_MANIFEST_BYTES:
                    raise StorageIntegrityError("Remote JSON object exceeds its limit")
                self.counters["jsonReadBytes"] += len(data)
                return _decode_json(data)
            finally:
                body.close()
        return self._call(False, operation)

    def put_json(self, key: str, value: Any) -> None:
        key = self._key(key); data = _json_bytes(value)
        self._call(True, lambda client, bucket, base: client.put_object(Bucket=bucket,
            Key=posixpath.join(base, key), Body=data, ContentType="application/json"))
        self.counters["jsonWrittenBytes"] += len(data)

    def upload(self, path: str | Path, key: str) -> dict[str, Any]:
        key = self._key(key); started = time.monotonic(); size = Path(path).stat().st_size
        expired = self.counters["expiredTokenRetries"]
        self._call(True, lambda client, bucket, base: client.upload_file(str(path), bucket, posixpath.join(base, key)))
        self.counters["uploadedBytes"] += size; self.counters["uploadOperations"] += 1
        return {"transferredBytes": size, "payloadBytes": size, "elapsedSeconds": time.monotonic() - started,
                "skipped": False, "expiredTokenRetries": self.counters["expiredTokenRetries"] - expired}

    def download(self, key: str, path: str | Path, sha256: str | None = None) -> dict[str, Any]:
        key = self._key(key); path = Path(path); started = time.monotonic(); expired = self.counters["expiredTokenRetries"]
        if path.is_file() and sha256 and digest(path) == sha256:
            return {"transferredBytes": 0, "payloadBytes": path.stat().st_size, "elapsedSeconds": time.monotonic() - started, "skipped": True}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".partial")
        try:
            self._call(False, lambda client, bucket, base: client.download_file(bucket, posixpath.join(base, key), str(temporary)))
            if sha256 and digest(temporary) != sha256:
                raise StorageIntegrityError("Downloaded object checksum mismatch")
            os.replace(temporary, path)
            size = path.stat().st_size; self.counters["downloadedBytes"] += size; self.counters["downloadOperations"] += 1
            return {"transferredBytes": size, "payloadBytes": size, "elapsedSeconds": time.monotonic() - started,
                    "skipped": False, "expiredTokenRetries": self.counters["expiredTokenRetries"] - expired}
        finally:
            temporary.unlink(missing_ok=True)


def _remote_ready(store: ObjectStore, prefix: str) -> dict[str, Any] | None:
    try:
        ready = store.get_json(prefix.rstrip("/") + "/READY")
    except FileNotFoundError:
        return None
    if not isinstance(ready, dict) or ready.get("format") != READY_FORMAT or not isinstance(ready.get("manifestSha256"), str) or not _SHA256.fullmatch(ready["manifestSha256"]):
        raise StorageIntegrityError("Invalid remote READY marker")
    size = _nonnegative_integer(ready.get("manifestBytes"), "manifestBytes")
    if size > MAX_MANIFEST_BYTES:
        raise StorageIntegrityError("Remote manifest exceeds its configured limit")
    return ready


def publish_ready_artifact(store: ObjectStore, directory: str | Path, prefix: str, *, acknowledge: bool = False) -> dict[str, Any]:
    """Publish an immutable artifact; a retry never replaces an existing different identity.

    Returned byte counts describe confirmed application payload/metadata transfers,
    not TLS/S3 framing or opaque SDK-internal retry traffic. An ambiguous final READY
    response is resolved by reading the same bound checksum on the next retry.
    """
    artifact = validate_ready_artifact(directory)
    prefix = prefix.rstrip("/")
    if not prefix.startswith(store.namespace + "/"):
        raise StorageIntegrityError("Publish prefix is outside the owned namespace")
    started = time.monotonic(); previous = _remote_ready(store, prefix)
    if previous is not None and previous["manifestSha256"] != artifact.manifest_sha256:
        raise StorageConflictError("Remote artifact identity already has a different manifest")
    if previous is not None and previous["manifestBytes"] != artifact.manifest_bytes:
        raise StorageConflictError("Remote artifact identity has inconsistent manifest length")
    if previous is None:
        for item in artifact.manifest["files"]:
            store.upload(_local_file(artifact.directory, item["path"]), prefix + "/" + item["path"])
        store.upload(artifact.directory / "manifest.json", prefix + "/manifest.json")
        store.upload(artifact.directory / "READY", prefix + "/READY")
        # Do not acknowledge a merely attempted upload. Verify server commit binding.
        committed = _remote_ready(store, prefix)
        if committed is None or committed["manifestSha256"] != artifact.manifest_sha256 or committed["manifestBytes"] != artifact.manifest_bytes:
            raise StorageIntegrityError("Remote READY verification failed after publication")
    receipt = {"format": "swarmada.worker-upload-receipt.v1", "prefix": prefix,
               "manifestSha256": artifact.manifest_sha256, "payloadBytes": artifact.payload_bytes,
               "metadataBytes": artifact.metadata_bytes,
               "confirmedUploadedBytes": 0 if previous is not None else artifact.total_bytes,
               "alreadyPublished": previous is not None, "elapsedSeconds": time.monotonic() - started,
               "confirmedUtc": time.time()}
    if acknowledge:
        atomic_json(artifact.directory / ".uploaded.json", receipt)
        (artifact.directory / ".retry.json").unlink(missing_ok=True)
    return receipt


def fetch_ready_artifact(store: ObjectStore, prefix: str, directory: str | Path, *,
                         expected_versions: dict[str, str] | None = None,
                         expected_kind: str | None = None) -> dict[str, Any]:
    """Expose a local directory only after remote READY, manifest and all checksums pass."""
    prefix = prefix.rstrip("/"); directory = Path(directory); started = time.monotonic()
    ready = _remote_ready(store, prefix)
    if ready is None:
        raise FileNotFoundError("Artifact is not ready")
    if directory.exists():
        existing = validate_ready_artifact(directory, expected_versions=expected_versions)
        if expected_kind is not None and (existing.manifest.get("format") != BUNDLE_FORMAT or existing.manifest.get("kind") != expected_kind):
            raise StorageIntegrityError("Existing artifact kind is incompatible")
        if existing.manifest_sha256 != ready["manifestSha256"]:
            raise StorageConflictError("Download destination already contains different immutable data")
        return {"prefix": prefix, "payloadBytes": existing.payload_bytes, "metadataBytes": existing.metadata_bytes,
                "confirmedDownloadedBytes": 0, "alreadyPresent": True, "elapsedSeconds": time.monotonic() - started}
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name("." + directory.name + "." + uuid.uuid4().hex + ".partial")
    temporary.mkdir()
    try:
        store.download(prefix + "/manifest.json", temporary / "manifest.json", ready["manifestSha256"])
        manifest, data = _read_json(temporary / "manifest.json")
        if len(data) != ready["manifestBytes"]:
            raise StorageIntegrityError("Downloaded manifest length differs from READY")
        manifest = validate_manifest(manifest, expected_versions=expected_versions)
        if expected_kind is not None and (manifest.get("format") != BUNDLE_FORMAT or manifest.get("kind") != expected_kind):
            raise StorageIntegrityError("Remote artifact kind is incompatible")
        for item in manifest["files"]:
            path = _local_file(temporary, item["path"])
            store.download(prefix + "/" + item["path"], path, item["sha256"])
            if path.stat().st_size != item["bytes"]:
                raise StorageIntegrityError("Downloaded payload length differs from manifest")
        # Retain the exact marker bytes, rather than reserializing native JSON.
        store.download(prefix + "/READY", temporary / "READY")
        artifact = validate_ready_artifact(temporary, expected_versions=expected_versions)
        if artifact.manifest_sha256 != ready["manifestSha256"]:
            raise StorageIntegrityError("Remote artifact changed while downloading")
        os.replace(temporary, directory)
        _sync_directory(directory.parent)
        return {"prefix": prefix, "payloadBytes": artifact.payload_bytes, "metadataBytes": artifact.metadata_bytes,
                "confirmedDownloadedBytes": artifact.total_bytes, "alreadyPresent": False,
                "elapsedSeconds": time.monotonic() - started}
    finally:
        if temporary.exists():
            # This path was created immediately above, never derived from a manifest.
            shutil.rmtree(temporary)


def publish_bundle(store: ObjectStore, directory: str | Path, prefix: str, metadata: dict[str, Any]) -> dict[str, Any]:
    directory = Path(directory)
    if not (directory / "READY").exists():
        seal_directory(directory, {**metadata, "format": BUNDLE_FORMAT})
    artifact = validate_ready_artifact(directory)
    if artifact.manifest.get("format") != BUNDLE_FORMAT:
        raise StorageIntegrityError("Expected a worker artifact bundle")
    if metadata.get("kind") != artifact.manifest.get("kind"):
        raise StorageIntegrityError("Existing bundle kind differs from requested publication")
    return publish_ready_artifact(store, directory, prefix)


def fetch_bundle(store: ObjectStore, prefix: str, directory: str | Path, *, kind: str | None = None,
                 expected_versions: dict[str, str] | None = None) -> dict[str, Any]:
    result = fetch_ready_artifact(store, prefix, directory, expected_versions=expected_versions, expected_kind=kind)
    artifact = validate_ready_artifact(directory)
    if artifact.manifest.get("format") != BUNDLE_FORMAT or kind is not None and artifact.manifest.get("kind") != kind:
        raise StorageIntegrityError("Downloaded bundle kind is incompatible")
    return result


def validate_game_records(directory: str | Path, *, decode_inputs: bool = True) -> dict[str, int]:
    """Validate the native worker's real tensor stream, sparse targets and terminal labels.

    This is a storage integrity check, not a trainer. Decoding is optional on the
    uploader hot path, and uses the production standard-library codec on demand.
    """
    artifact = validate_ready_artifact(directory)
    if artifact.manifest["format"] != SHARD_FORMAT or len(artifact.manifest["games"]) != 1:
        raise StorageIntegrityError("This record validator expects one native game per shard")
    targets, _ = _read_json(artifact.directory / "targets.json", 256 * 1024 * 1024)
    decisions = targets.get("decisions") if isinstance(targets, dict) else None
    terminal = targets.get("terminalTargets") if isinstance(targets, dict) else None
    if not isinstance(decisions, list) or len(decisions) != artifact.manifest["games"][0]["decisions"] or not isinstance(terminal, dict):
        raise StorageIntegrityError("Targets and manifest decision counts/terminal labels disagree")
    winner = terminal.get("winner")
    if not isinstance(winner, int) or isinstance(winner, bool) or winner not in (0, 1, 2):
        raise StorageIntegrityError("Invalid genuine terminal winner")
    final_round = _nonnegative_integer(terminal.get("finalRound"), "terminal finalRound")
    if final_round < 1:
        raise StorageIntegrityError("Terminal round must be positive")
    for name, length in (("rawPoints", 2), ("finalShipHullFractions", None), ("finalSquadronHullFractions", None)):
        values = terminal.get(name)
        if not isinstance(values, list) or length is not None and len(values) != length or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise StorageIntegrityError("Invalid terminal auxiliary array: " + name)
    if decode_inputs:
        # Import remains optional so this module can bootstrap as a standalone ops file.
        from swarmada_headless.encoding import decode_observation
    payload_bytes = 0; candidates = 0
    with (artifact.directory / "inputs.bin").open("rb") as stream:
        if stream.read(8) != INPUT_MAGIC or stream.read(4) != struct.pack("<I", INPUT_VERSION):
            raise StorageIntegrityError("Unsupported native input stream header")
        for decision in decisions:
            if not isinstance(decision, dict) or decision.get("observer") not in (1, 2):
                raise StorageIntegrityError("Invalid conditional decision observer")
            prefix = decision.get("prefix")
            if not isinstance(prefix, list) or any(not isinstance(value, str) for value in prefix):
                raise StorageIntegrityError("Invalid conditional decision prefix")
            indices, visits = decision.get("candidateIndices"), decision.get("visits")
            if not isinstance(indices, list) or not isinstance(visits, list) or not indices or len(indices) != len(visits):
                raise StorageIntegrityError("Sparse visit target is empty or inconsistent")
            if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices) or len(set(indices)) != len(indices) or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in visits):
                raise StorageIntegrityError("Sparse visits must have unique indices and positive integer counts")
            length_bytes = stream.read(4)
            if len(length_bytes) != 4:
                raise StorageIntegrityError("Truncated conditional input stream")
            length = struct.unpack("<I", length_bytes)[0]
            if length < 1 or length > 65 * 1024 * 1024 or decision.get("inputOffset") != stream.tell() or decision.get("inputBytes") != length:
                raise StorageIntegrityError("Tensor stream length/target offsets disagree")
            payload = stream.read(length)
            if len(payload) != length:
                raise StorageIntegrityError("Truncated lossless tensor record")
            if decode_inputs:
                decoded = decode_observation(payload, expected_schema=artifact.manifest["versions"]["schemaVersion"])
                policy = decoded.tensors.get("policy.legal_mask")
                if policy is None or len(policy.shape) != 1 or max(indices) >= policy.shape[0]:
                    raise StorageIntegrityError("Sparse visits reference a missing policy candidate")
            payload_bytes += length; candidates += len(indices)
        if stream.read(1):
            raise StorageIntegrityError("Unreferenced input bytes after the last conditional record")
    return {"games": 1, "decisions": len(decisions), "nonzeroVisits": candidates, "inputPayloadBytes": payload_bytes}


@dataclass(frozen=True)
class SpoolLimits:
    maximum_bytes: int = 8 * 1024**3
    maximum_pending_shards: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.maximum_bytes, int) or self.maximum_bytes < 1 or not isinstance(self.maximum_pending_shards, int) or self.maximum_pending_shards < 1:
            raise ValueError("Spool limits must be positive integers")


class WorkerSpool:
    """Restart-safe uploader with persistent retry state and explicit backpressure.

    Local deletion is opt-in and only follows a verified remote READY acknowledgement.
    Remote retention/deletion is deliberately outside this worker-only module.
    """
    def __init__(self, directory: str | Path, store: ObjectStore, *, limits: SpoolLimits | None = None,
                 retry_initial_seconds: float = 2, retry_maximum_seconds: float = 300,
                 log: Callable[[dict[str, Any]], None] | None = None):
        self.directory = Path(directory).resolve(); self.directory.mkdir(parents=True, exist_ok=True)
        self.store, self.limits = store, limits or SpoolLimits()
        if retry_initial_seconds < 0 or retry_maximum_seconds < retry_initial_seconds:
            raise ValueError("Invalid persistent retry interval")
        self.retry_initial_seconds, self.retry_maximum_seconds = retry_initial_seconds, retry_maximum_seconds
        self.log = log or (lambda _: None)

    def _directories(self) -> list[Path]:
        return sorted(path for path in self.directory.iterdir() if path.is_dir() and not path.is_symlink() and (path / "READY").is_file())

    @staticmethod
    def _acknowledged(path: Path) -> bool:
        receipt = path / ".uploaded.json"
        if not receipt.is_file():
            return False
        try:
            acknowledgement, _ = _read_json(receipt)
            ready, _ = _read_json(path / "READY", MAX_READY_BYTES)
            return acknowledgement.get("format") == "swarmada.worker-upload-receipt.v1" and acknowledgement.get("manifestSha256") == ready.get("manifestSha256")
        except (StorageIntegrityError, AttributeError):
            return False

    def usage(self) -> dict[str, int]:
        # Include unfinished native directories and partials in the disk bound.
        total = sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file() and not path.is_symlink())
        pending = sum(not self._acknowledged(path) for path in self._directories())
        return {"bytes": total, "pendingShards": pending, "readyShards": len(self._directories())}

    def require_capacity(self, *, reserve_bytes: int = 0, reserve_shards: int = 1) -> dict[str, int]:
        usage = self.usage()
        if reserve_bytes < 0 or reserve_shards < 0:
            raise ValueError("Spool reservations cannot be negative")
        if usage["bytes"] + reserve_bytes >= self.limits.maximum_bytes or usage["pendingShards"] + reserve_shards > self.limits.maximum_pending_shards:
            raise SpoolFullError("Worker spool is full; retain data and retry upload before generating more")
        return usage

    def pump_once(self, *, now: float | None = None, delete_acknowledged: bool = False) -> dict[str, Any]:
        now = time.time() if now is None else now
        published: list[dict[str, Any]] = []; deferred = 0; failed: list[dict[str, str]] = []
        for directory in self._directories():
            if self._acknowledged(directory):
                if delete_acknowledged:
                    self.prune_acknowledged(directory)
                continue
            retry: dict[str, Any] = {}
            if (directory / ".retry.json").is_file():
                retry = _read_json(directory / ".retry.json")[0]
            if float(retry.get("nextRetryUtc", 0)) > now:
                deferred += 1; continue
            attempts = _nonnegative_integer(retry.get("attempts", 0), "upload retry attempts") + 1
            try:
                artifact = validate_ready_artifact(directory)
                if artifact.manifest["format"] != SHARD_FORMAT:
                    raise StorageIntegrityError("A worker replay spool contains a non-game bundle")
                manifest = artifact.manifest
                prefix = posixpath.join(self.store.namespace, "replay", manifest["runId"], manifest["workerId"], manifest["shardId"])
                receipt = publish_ready_artifact(self.store, directory, prefix, acknowledge=True)
                receipt["attempts"] = attempts; receipt["retryCount"] = attempts - 1
                atomic_json(directory / ".uploaded.json", receipt)
                published.append(receipt); self.log({"event": "shard-uploaded", **receipt})
                if delete_acknowledged:
                    self.prune_acknowledged(directory)
            except Exception as error:
                # Exception messages can contain signed URLs or SDK request headers.
                # Persist only the safe type, counts, and retry timing.
                delay = min(self.retry_maximum_seconds, self.retry_initial_seconds * 2 ** min(attempts - 1, 20))
                state = {"attempts": attempts, "nextRetryUtc": now + delay, "errorType": type(error).__name__}
                atomic_json(directory / ".retry.json", state)
                item = {"shard": directory.name, "errorType": type(error).__name__}
                failed.append(item); self.log({"event": "shard-upload-failed", **item, **state})
        return {"published": published, "deferred": deferred, "failed": failed, "usage": self.usage()}

    def prune_acknowledged(self, directory: str | Path) -> None:
        original = Path(directory)
        if original.is_symlink():
            raise StorageIntegrityError("A linked spool directory cannot be removed")
        directory = original.resolve()
        if directory.parent != self.directory or not self._acknowledged(directory):
            raise StorageIntegrityError("Only an acknowledged direct child of this spool may be removed")
        artifact = validate_ready_artifact(directory)
        acknowledgement, _ = _read_json(directory / ".uploaded.json")
        if acknowledgement.get("manifestSha256") != artifact.manifest_sha256:
            raise StorageIntegrityError("Local acknowledgement does not match immutable shard")
        remote = _remote_ready(self.store, acknowledgement.get("prefix", ""))
        if remote is None or remote.get("manifestSha256") != artifact.manifest_sha256:
            raise StorageIntegrityError("Remote acknowledgement must still exist before local pruning")
        # Preserve the durable receipt before removing only this verified owned directory.
        atomic_json(self.directory / ".receipts" / (directory.name + ".json"), acknowledgement)
        shutil.rmtree(directory)


def jsonl_logger(path: str | Path) -> Callable[[dict[str, Any]], None]:
    """Append bounded, structured events. Callers must not pass credentials or SDK errors."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def write(event: dict[str, Any]) -> None:
        with path.open("ab") as stream:
            stream.write(_json_bytes({"timeUtc": time.time(), **event}))
            stream.flush()
    return write


def write_mock_model(directory: str | Path, *, seed: int, model_version: str,
                     versions: dict[str, str]) -> ReadyArtifact:
    """Create actual tiny random-model configuration bytes, never a BigDeep-size surrogate."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Mock model seed must be a nonnegative integer")
    _safe_id(model_version, "modelVersion"); directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if (directory / "READY").exists():
        existing = validate_ready_artifact(directory, expected_versions=versions)
        model, _ = _read_json(directory / "model-seed.json")
        if existing.manifest.get("kind") != "mock-model" or model.get("seed") != seed or model.get("modelVersion") != model_version:
            raise StorageConflictError("An existing mock model identity cannot be replaced")
        return existing
    atomic_json(directory / "model-seed.json", {"format": "swarmada.mock-model.v1", "kind": "random-policy-value",
                "seed": seed, "modelVersion": model_version, "neuralWeights": False})
    return seal_directory(directory, {"format": BUNDLE_FORMAT, "kind": "mock-model", "versions": versions,
                                      "modelVersion": model_version, "createdUtc": time.time()})


def fetch_mock_model(store: ObjectStore, prefix: str, directory: str | Path, *,
                     expected_versions: dict[str, str] | None = None) -> dict[str, Any]:
    receipt = fetch_bundle(store, prefix, directory, kind="mock-model", expected_versions=expected_versions)
    model, data = _read_json(Path(directory) / "model-seed.json")
    if not isinstance(model, dict) or model.get("format") != "swarmada.mock-model.v1" or model.get("kind") != "random-policy-value" or model.get("neuralWeights") is not False:
        raise StorageIntegrityError("Only explicit random policy/value mock models are supported by this worker test")
    seed = _nonnegative_integer(model.get("seed"), "mock model seed")
    return {**receipt, "modelVersion": model.get("modelVersion"), "seed": seed,
            "modelConfigurationBytes": len(data), "neuralWeightBytes": 0}


def _cli_store(args: argparse.Namespace) -> ObjectStore:
    if args.local_store:
        return LocalStore(args.local_store, args.namespace)
    if not args.organization or not args.volume:
        raise ValueError("VESSL operations require explicit organization, fresh volume and namespace")
    return VesslStore(args.organization, args.volume, args.storage, namespace=args.namespace)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verified worker-only spool/artifact transfers; never launches cloud runs or a trainer")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--organization")
    parser.add_argument("--volume")
    parser.add_argument("--storage", default="vessl-storage")
    parser.add_argument("--local-store", type=Path)
    commands = parser.add_subparsers(dest="operation", required=True)
    upload = commands.add_parser("upload-spool")
    upload.add_argument("directory", type=Path); upload.add_argument("--max-gib", type=float, default=8)
    upload.add_argument("--max-pending", type=int, default=64); upload.add_argument("--watch-seconds", type=float, default=0)
    upload.add_argument("--delete-acknowledged-local", action="store_true", default=False)
    upload.add_argument("--events", type=Path)
    download = commands.add_parser("download-verify")
    download.add_argument("prefix"); download.add_argument("directory", type=Path)
    download.add_argument("--decode-game", action="store_true")
    logs = commands.add_parser("export-logs")
    logs.add_argument("directory", type=Path); logs.add_argument("prefix"); logs.add_argument("--code-version", required=True)
    model = commands.add_parser("model-download")
    model.add_argument("prefix"); model.add_argument("directory", type=Path)
    listing = commands.add_parser("list")
    listing.add_argument("prefix", nargs="?", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        store = _cli_store(args)
        if args.operation == "upload-spool":
            if not math.isfinite(args.max_gib) or args.max_gib <= 0 or args.watch_seconds < 0:
                raise ValueError("Invalid spool/watch limit")
            spool = WorkerSpool(args.directory, store, limits=SpoolLimits(int(args.max_gib * 1024**3), args.max_pending),
                                log=jsonl_logger(args.events) if args.events else None)
            while True:
                result = spool.pump_once(delete_acknowledged=args.delete_acknowledged_local)
                print(json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
                if args.watch_seconds == 0:
                    return 2 if result["failed"] else 0
                time.sleep(min(30, args.watch_seconds))
        elif args.operation == "download-verify":
            result = fetch_ready_artifact(store, args.prefix, args.directory)
            if args.decode_game:
                result["records"] = validate_game_records(args.directory)
        elif args.operation == "export-logs":
            result = publish_bundle(store, args.directory, args.prefix, {"kind": "logs", "versions": {"codeVersion": args.code_version}, "createdUtc": time.time()})
        elif args.operation == "model-download":
            result = fetch_mock_model(store, args.prefix, args.directory)
        else:
            result = store.list(args.prefix)
        print(json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
        return 0
    except Exception as error:
        # SDK messages may embed signed URLs. Local integrity errors are our own safe text.
        detail = str(error) if isinstance(error, (StorageIntegrityError, SpoolFullError)) else "Operation failed; inspect authenticated service diagnostics separately"
        print(json.dumps({"errorType": type(error).__name__, "detail": detail}, separators=(",", ":")), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
