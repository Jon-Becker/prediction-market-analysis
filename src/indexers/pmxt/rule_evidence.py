"""One-shot, immutable capture of hash-bound native rule evidence.

This module only captures three public Kalshi contract PDFs and five public
Polymarket Gamma event documents.  It has no PMXT, credential, account, book,
fee, position, order, or semantic-promotion surface.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit

import httpcore
import httpx

PROTOCOL_ID = "pmxt-native-rule-evidence-capture-v1"
IMPLEMENTATION_ID = "pmxt-rule-evidence-capture-v1"
SCHEMA_VERSION = 1
DEFAULT_PROTOCOL_PATH = Path("results/rule_evidence_v1/protocol_v1.json")
OUTPUT_PARENT = Path("data/pmxt/rule_evidence/runs")
MAX_RESOURCE_BYTES = 5_000_000
MAX_TOTAL_BYTES = 40_000_000
INACTIVITY_TIMEOUT_SECONDS = 20.0
RESOURCE_ELAPSED_CHECKPOINT_SECONDS = 30.0
RUN_ELAPSED_CHECKPOINT_SECONDS = 240.0
EXPECTED_RESOURCE_COUNT = 8
EXPECTED_HTTPX_VERSION = "0.28.1"
EXPECTED_HTTPCORE_VERSION = "1.0.9"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RESOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_CANDIDATE_ID_RE = re.compile(r"^pmxt_candidate_[0-9a-f]{20}$")
_ALLOWED_HOSTS = {"assets.kalshi.com", "gamma-api.polymarket.com"}
_ALLOWED_HEADERS = {
    "age",
    "cache-control",
    "content-encoding",
    "content-length",
    "content-type",
    "date",
    "etag",
    "last-modified",
    "request-id",
    "x-amz-id-2",
    "x-amz-request-id",
    "x-amz-version-id",
    "x-amzn-requestid",
    "x-correlation-id",
    "x-request-id",
}
_FORBIDDEN_REQUEST_HEADERS = {"authorization", "cookie", "proxy-authorization"}
_SAFETY_FIELDS = {
    "books_requested": False,
    "credentials_read": False,
    "economics_computed": False,
    "fees_requested": False,
    "live_eligible": False,
    "orders_submitted": 0,
    "pmxt_requests": 0,
    "positions_requested": False,
    "semantic_promotions": 0,
}
_EXACT_RESOURCES = (
    (
        1,
        "kalshi-title-terms",
        "kalshi",
        "pdf",
        "https://assets.kalshi.com/contract_terms/TITLE.pdf",
        None,
    ),
    (
        2,
        "kalshi-presnom-terms",
        "kalshi",
        "pdf",
        "https://assets.kalshi.com/contract_terms/PRESNOM.pdf",
        None,
    ),
    (
        3,
        "kalshi-gov-terms",
        "kalshi",
        "pdf",
        "https://assets.kalshi.com/contract_terms/GOV.pdf",
        None,
    ),
    (
        4,
        "polymarket-event-179312",
        "polymarket",
        "json",
        "https://gamma-api.polymarket.com/events/179312",
        "179312",
    ),
    (
        5,
        "polymarket-event-202857",
        "polymarket",
        "json",
        "https://gamma-api.polymarket.com/events/202857",
        "202857",
    ),
    (
        6,
        "polymarket-event-31875",
        "polymarket",
        "json",
        "https://gamma-api.polymarket.com/events/31875",
        "31875",
    ),
    (
        7,
        "polymarket-event-59234",
        "polymarket",
        "json",
        "https://gamma-api.polymarket.com/events/59234",
        "59234",
    ),
    (
        8,
        "polymarket-event-57096",
        "polymarket",
        "json",
        "https://gamma-api.polymarket.com/events/57096",
        "57096",
    ),
)


class RuleEvidenceCaptureError(RuntimeError):
    """Raised when a capture cannot satisfy its frozen fail-closed contract."""


class _ResourceCaptureFailure(RuleEvidenceCaptureError):
    def __init__(
        self,
        message: str,
        *,
        receipt: Mapping[str, Any] | None,
    ) -> None:
        super().__init__(message)
        self.receipt = dict(receipt) if receipt is not None else None


@dataclass(frozen=True)
class _InputSnapshot:
    path: Path
    payload: bytes
    binding: Mapping[str, Any]


def _is_reparse_status(status: os.stat_result) -> bool:
    attributes = int(getattr(status, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag)


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuleEvidenceCaptureError(f"cannot inspect path component: {path}") from exc


def _reject_absolute_reparse_chain(path: Path, *, label: str) -> os.stat_result:
    if not path.is_absolute():
        raise RuleEvidenceCaptureError(f"{label} must be absolute")
    current = Path(path.anchor)
    status = _lstat_or_none(current)
    if status is None:
        raise RuleEvidenceCaptureError(f"{label} is missing")
    if _is_reparse_status(status):
        raise RuleEvidenceCaptureError(f"{label} must not traverse a symlink or reparse point")
    for part in path.parts[1:]:
        current /= part
        status = _lstat_or_none(current)
        if status is None:
            raise RuleEvidenceCaptureError(f"{label} is missing")
        if _is_reparse_status(status):
            raise RuleEvidenceCaptureError(f"{label} must not traverse a symlink or reparse point")
    return status


def _reject_reparse_chain(
    root: Path,
    parts: Sequence[str],
    *,
    label: str,
    require_all: bool,
) -> Path:
    current = root
    root_status = _lstat_or_none(root)
    if root_status is None:
        raise RuleEvidenceCaptureError("repository root is missing")
    if _is_reparse_status(root_status):
        raise RuleEvidenceCaptureError("repository root must not be a symlink or reparse point")
    for index, part in enumerate(parts):
        current /= part
        status = _lstat_or_none(current)
        if status is None:
            if require_all:
                raise RuleEvidenceCaptureError(f"{label} is missing")
            return root.joinpath(*parts)
        if _is_reparse_status(status):
            raise RuleEvidenceCaptureError(f"{label} must not traverse a symlink or reparse point")
        if index < len(parts) - 1 and not stat.S_ISDIR(status.st_mode):
            raise RuleEvidenceCaptureError(f"{label} has a non-directory path component")
    return current


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    if left.st_dev != right.st_dev:
        return False
    if (left.st_ino or right.st_ino) and left.st_ino != right.st_ino:
        return False
    return (
        stat.S_IFMT(left.st_mode),
        left.st_size,
        getattr(left, "st_mtime_ns", None),
        getattr(left, "st_ctime_ns", None),
    ) == (
        stat.S_IFMT(right.st_mode),
        right.st_size,
        getattr(right, "st_mtime_ns", None),
        getattr(right, "st_ctime_ns", None),
    )


def _read_input(path: Path, *, root: Path, label: str) -> _InputSnapshot:
    """Read and hash one input through the same non-following file handle."""

    try:
        relative_parts = path.relative_to(root).parts
    except ValueError as exc:
        raise RuleEvidenceCaptureError(f"{label} is outside the repository") from exc
    _reject_reparse_chain(root, relative_parts, label=label, require_all=True)
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuleEvidenceCaptureError(f"cannot open {label} without following links") from exc
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _is_reparse_status(before):
            raise RuleEvidenceCaptureError(f"{label} is not a regular non-reparse file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not _same_file(before, after) or before.st_size != after.st_size:
        raise RuleEvidenceCaptureError(f"{label} changed while it was read")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise RuleEvidenceCaptureError(f"{label} size changed while it was read")
    path_status = _lstat_or_none(path)
    if path_status is None or _is_reparse_status(path_status) or not _same_file(after, path_status):
        raise RuleEvidenceCaptureError(f"{label} path changed while it was read")
    return _InputSnapshot(
        path=path,
        payload=payload,
        binding={"path": path.relative_to(root).as_posix(), "byte_size": len(payload), "sha256": digest.hexdigest()},
    )


def _ensure_real_directories(root: Path, parts: Sequence[str]) -> Path:
    current = root
    for part in parts:
        current /= part
        status = _lstat_or_none(current)
        if status is None:
            try:
                os.mkdir(current)
            except FileExistsError:
                pass
            status = _lstat_or_none(current)
        if status is None or _is_reparse_status(status) or not stat.S_ISDIR(status.st_mode):
            raise RuleEvidenceCaptureError("output path must not traverse a symlink or reparse point")
    return current


def _sync_directory(path: Path) -> None:
    # Python cannot portably open Windows directories for FlushFileBuffers.
    # File fsyncs remain mandatory; directory durability is best effort there.
    if os.name == "nt":
        return
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically promote a directory without replacing an existing target."""

    if _lstat_or_none(destination) is not None:
        raise RuleEvidenceCaptureError("final rule-evidence output already exists")
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise RuleEvidenceCaptureError("final rule-evidence output already exists") from exc
        except OSError as exc:
            raise RuleEvidenceCaptureError("exclusive rule-evidence finalization failed") from exc
        return

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuleEvidenceCaptureError("atomic no-replace finalization is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin":
        renamex_np = getattr(library, "renamex_np", None)
        if renamex_np is None:
            raise RuleEvidenceCaptureError("atomic no-replace finalization is unavailable")
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
    else:
        raise RuleEvidenceCaptureError("atomic no-replace finalization is unavailable")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise RuleEvidenceCaptureError("final rule-evidence output already exists")
    raise RuleEvidenceCaptureError(f"exclusive rule-evidence finalization failed: errno {error_number}")


class _DuplicateKey(ValueError):
    pass


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _strict_json(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, _DuplicateKey) as exc:
        raise RuleEvidenceCaptureError(f"{label} is not strict UTF-8 JSON") from exc


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuleEvidenceCaptureError("value is not canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _require_object(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuleEvidenceCaptureError(f"{label} must be a JSON object")
    return value


def _require_array(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuleEvidenceCaptureError(f"{label} must be a JSON array")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise RuleEvidenceCaptureError(f"{label} has unexpected or missing fields")


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuleEvidenceCaptureError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuleEvidenceCaptureError(f"{label} must be a non-negative integer")
    return value


def _relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuleEvidenceCaptureError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or path.drive or any(part in {"", ".", ".."} for part in path.parts):
        raise RuleEvidenceCaptureError(f"{label} must be a contained relative path")
    return path


def _bound_input(root: Path, binding: Any, *, label: str) -> tuple[Mapping[str, Any], bytes, Path]:
    item = _require_object(binding, label=f"{label} binding")
    _require_exact_keys(item, {"path", "sha256", "byte_size"}, label=f"{label} binding")
    relative = _relative_path(item["path"], label=f"{label} path")
    snapshot = _read_input(root / relative, root=root, label=label)
    expected_sha = _require_sha(item["sha256"], label=f"{label} SHA-256")
    expected_size = _require_nonnegative_int(item["byte_size"], label=f"{label} byte size")
    if snapshot.binding["sha256"] != expected_sha or snapshot.binding["byte_size"] != expected_size:
        raise RuleEvidenceCaptureError(f"{label} does not match its protocol binding")
    return item, snapshot.payload, snapshot.path


def _parse_jsonl(payload: bytes, *, label: str) -> tuple[tuple[bytes, Mapping[str, Any]], ...]:
    if payload and not payload.endswith(b"\n"):
        raise RuleEvidenceCaptureError(f"{label} must end with LF")
    rows: list[tuple[bytes, Mapping[str, Any]]] = []
    for line_number, raw in enumerate(payload.splitlines(keepends=True), 1):
        if not raw.strip():
            raise RuleEvidenceCaptureError(f"{label} contains a blank line")
        value = _strict_json(raw, label=f"{label} line {line_number}")
        rows.append((raw, _require_object(value, label=f"{label} line {line_number}")))
    return tuple(rows)


def _row_at(
    rows: Sequence[tuple[bytes, Mapping[str, Any]]],
    reference: Any,
    *,
    label: str,
    extra_keys: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    item = _require_object(reference, label=label)
    expected_keys = {"line", "row_sha256", *extra_keys}
    if set(item) != expected_keys:
        raise RuleEvidenceCaptureError(f"{label} has unexpected or missing binding fields")
    line = item["line"]
    if isinstance(line, bool) or not isinstance(line, int) or line < 1 or line > len(rows):
        raise RuleEvidenceCaptureError(f"{label} line is out of range")
    raw, row = rows[line - 1]
    if _sha256(raw) != _require_sha(item["row_sha256"], label=f"{label} row SHA-256"):
        raise RuleEvidenceCaptureError(f"{label} row hash does not match")
    return row


def _validate_safety(value: Any, *, label: str) -> None:
    authority = _require_object(value, label=label)
    if authority != _SAFETY_FIELDS:
        raise RuleEvidenceCaptureError(f"{label} is not the exact capture-only safety boundary")


def _validate_url(resource: Mapping[str, Any], expected: tuple[Any, ...]) -> None:
    ordinal, resource_id, venue, kind, url, event_id = expected
    if resource.get("ordinal") != ordinal or resource.get("resource_id") != resource_id:
        raise RuleEvidenceCaptureError("resource order or identifier differs from the frozen eight-resource plan")
    if resource.get("venue") != venue or resource.get("kind") != kind or resource.get("url") != url:
        raise RuleEvidenceCaptureError(f"resource {resource_id} differs from the frozen source")
    parsed = urlsplit(str(resource.get("url")))
    if (
        parsed.scheme != "https"
        or parsed.hostname != urlsplit(url).hostname
        or parsed.hostname not in _ALLOWED_HOSTS
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != urlsplit(url).path
    ):
        raise RuleEvidenceCaptureError(f"resource {resource_id} URL is not an exact approved HTTPS source")
    expected_media = "application/pdf" if kind == "pdf" else "application/json"
    if resource.get("expected_media_type") != expected_media:
        raise RuleEvidenceCaptureError(f"resource {resource_id} media type is not frozen")
    if resource.get("expected_event_id") != event_id:
        raise RuleEvidenceCaptureError(f"resource {resource_id} event binding is not frozen")


def _validate_implementation(root: Path, protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    implementation = _require_object(protocol.get("implementation"), label="implementation")
    _require_exact_keys(implementation, {"identifier", "path", "sha256", "byte_size"}, label="implementation")
    if implementation.get("identifier") != IMPLEMENTATION_ID:
        raise RuleEvidenceCaptureError("implementation identifier mismatch")
    if implementation.get("path") != "src/indexers/pmxt/rule_evidence.py":
        raise RuleEvidenceCaptureError("implementation path is not frozen")
    source_binding = {key: implementation[key] for key in ("path", "sha256", "byte_size")}
    _, repository_payload, _ = _bound_input(root, source_binding, label="implementation source")

    running_path = Path(os.path.abspath(__file__))
    status = _reject_absolute_reparse_chain(running_path, label="running implementation source")
    if not stat.S_ISREG(status.st_mode):
        raise RuleEvidenceCaptureError("running implementation source is not a regular file")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(running_path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    running_payload = b"".join(chunks)
    if not _same_file(before, after) or len(running_payload) != after.st_size:
        raise RuleEvidenceCaptureError("running implementation source changed while it was read")
    if running_payload != repository_payload or _sha256(running_payload) != implementation["sha256"]:
        raise RuleEvidenceCaptureError("protocol does not bind the running implementation source")
    return implementation


def _artifact_payloads(
    root: Path,
    run_binding: Any,
    *,
    label: str,
) -> tuple[Mapping[str, Any], bytes, dict[str, bytes]]:
    run = _require_object(run_binding, label=label)
    if not isinstance(run.get("run_id"), str) or not run["run_id"]:
        raise RuleEvidenceCaptureError(f"{label} run_id is invalid")
    _, manifest_payload, _ = _bound_input(root, run.get("manifest"), label=f"{label} manifest")
    manifest = _require_object(_strict_json(manifest_payload, label=f"{label} manifest"), label=f"{label} manifest")
    if manifest.get("run_id") != run["run_id"]:
        raise RuleEvidenceCaptureError(f"{label} manifest run_id mismatch")
    declared_artifacts = _require_object(manifest.get("artifacts"), label=f"{label} manifest artifacts")
    protocol_artifacts = _require_object(run.get("artifacts"), label=f"{label} protocol artifacts")
    payloads: dict[str, bytes] = {}
    for name, binding in protocol_artifacts.items():
        if not isinstance(name, str) or name not in declared_artifacts:
            raise RuleEvidenceCaptureError(f"{label} artifact {name!r} is not declared by the manifest")
        protocol_binding = _require_object(binding, label=f"{label} artifact {name}")
        manifest_binding = _require_object(declared_artifacts[name], label=f"{label} manifest artifact {name}")
        manifest_relative = _relative_path(manifest_binding.get("path"), label=f"{label} manifest artifact path")
        manifest_protocol_binding = _require_object(run.get("manifest"), label=f"{label} manifest binding")
        expected_path = (
            _relative_path(manifest_protocol_binding.get("path"), label=f"{label} manifest path").parent
            / manifest_relative
        ).as_posix()
        if (
            protocol_binding.get("path") != expected_path
            or protocol_binding.get("sha256") != manifest_binding.get("sha256")
            or protocol_binding.get("byte_size") != manifest_binding.get("byte_size")
        ):
            raise RuleEvidenceCaptureError(f"{label} artifact {name} disagrees with the manifest")
        _, payload, _ = _bound_input(root, binding, label=f"{label} artifact {name}")
        payloads[name] = payload
    return manifest, manifest_payload, payloads


def _validate_sources(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    adjudication_manifest, _, adjudication_artifacts = _artifact_payloads(
        root,
        protocol.get("source_adjudication_run"),
        label="source adjudication run",
    )
    monitor_manifest, monitor_manifest_payload, monitor_artifacts = _artifact_payloads(
        root,
        protocol.get("source_monitor_run"),
        label="source monitor run",
    )
    if adjudication_manifest.get("protocol_id") != "pmxt-offline-semantic-adjudication-v1":
        raise RuleEvidenceCaptureError("source adjudication protocol is not authoritative")
    counts = _require_object(adjudication_manifest.get("counts"), label="source adjudication counts")
    authority = _require_object(adjudication_manifest.get("authority"), label="source adjudication authority")
    if counts.get("needs_review") != 14 or counts.get("total") != 25:
        raise RuleEvidenceCaptureError("source adjudication does not contain exactly fourteen review candidates")
    if (
        authority.get("network_requests") != 0
        or authority.get("orders_submitted") != 0
        or authority.get("live_eligible") is not False
        or authority.get("economics_computed") is not False
    ):
        raise RuleEvidenceCaptureError("source adjudication safety boundary is invalid")
    if adjudication_manifest.get("source_run_id") != monitor_manifest.get("run_id"):
        raise RuleEvidenceCaptureError("adjudication and monitor run IDs are not bound")
    monitor_counts = _require_object(monitor_manifest.get("counts"), label="source monitor counts")
    if (
        monitor_manifest.get("live_eligible") is not False
        or monitor_manifest.get("no_order_actions") is not True
        or monitor_counts.get("native_book_attempts") != 0
        or monitor_counts.get("pmxt_network_requests") != 0
    ):
        raise RuleEvidenceCaptureError("source monitor safety boundary is invalid")
    if adjudication_manifest.get("source_manifest_sha256") != _sha256(monitor_manifest_payload):
        raise RuleEvidenceCaptureError("adjudication does not bind the monitor manifest")

    required_adjudication = {"decisions", "source_bindings", "summary"}
    required_monitor = {"candidates", "raw_native_metadata", "semantic_decisions"}
    if set(adjudication_artifacts) != required_adjudication or set(monitor_artifacts) != required_monitor:
        raise RuleEvidenceCaptureError("protocol must bind every required source artifact")

    adjudication_source_bindings = _require_object(
        _strict_json(adjudication_artifacts["source_bindings"], label="adjudication source bindings"),
        label="adjudication source bindings",
    )
    bound_monitor = _require_object(
        adjudication_source_bindings.get("source_monitor_run"),
        label="adjudication-bound monitor run",
    )
    protocol_monitor = _require_object(protocol.get("source_monitor_run"), label="protocol monitor run")
    if bound_monitor.get("run_id") != protocol_monitor.get("run_id") or _require_object(
        bound_monitor.get("manifest"), label="adjudication-bound monitor manifest"
    ).get("sha256") != _require_object(protocol_monitor.get("manifest"), label="protocol monitor manifest").get(
        "sha256"
    ):
        raise RuleEvidenceCaptureError("adjudication source bindings disagree with the monitor protocol binding")
    adjudication_summary = _require_object(
        _strict_json(adjudication_artifacts["summary"], label="adjudication summary"),
        label="adjudication summary",
    )
    summary_counts = _require_object(adjudication_summary.get("counts"), label="adjudication summary counts")
    if summary_counts.get("needs_review") != 14 or summary_counts.get("verified_equivalent") != 0:
        raise RuleEvidenceCaptureError("adjudication summary does not preserve the review-only result")

    decisions = _parse_jsonl(adjudication_artifacts["decisions"], label="adjudication decisions")
    candidates = _parse_jsonl(monitor_artifacts["candidates"], label="monitor candidates")
    metadata = _parse_jsonl(monitor_artifacts["raw_native_metadata"], label="monitor native metadata")
    semantic = _parse_jsonl(monitor_artifacts["semantic_decisions"], label="monitor semantic decisions")
    bindings = _require_array(protocol.get("candidate_bindings"), label="candidate bindings")
    if len(bindings) != 14:
        raise RuleEvidenceCaptureError("protocol must bind exactly fourteen candidates")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, raw_binding in enumerate(bindings, 1):
        binding = _require_object(raw_binding, label=f"candidate binding {index}")
        _require_exact_keys(
            binding,
            {
                "candidate_id",
                "adjudication_decision_row",
                "candidate_row",
                "automated_decision_row",
                "kalshi_metadata_row",
                "polymarket_metadata_row",
                "native_ids",
                "kalshi_contract_terms_url",
                "gap_axes",
                "resource_ids",
            },
            label=f"candidate binding {index}",
        )
        candidate_id = binding.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or _CANDIDATE_ID_RE.fullmatch(candidate_id) is None
            or candidate_id in seen
        ):
            raise RuleEvidenceCaptureError("candidate bindings must have unique frozen candidate IDs")
        seen.add(candidate_id)
        decision = _row_at(decisions, binding.get("adjudication_decision_row"), label=f"{candidate_id} decision")
        candidate = _row_at(candidates, binding.get("candidate_row"), label=f"{candidate_id} candidate")
        kalshi = _row_at(
            metadata,
            binding.get("kalshi_metadata_row"),
            label=f"{candidate_id} Kalshi metadata",
            extra_keys=frozenset({"raw_sha256", "rule_hash"}),
        )
        polymarket = _row_at(
            metadata,
            binding.get("polymarket_metadata_row"),
            label=f"{candidate_id} Polymarket metadata",
            extra_keys=frozenset({"raw_sha256", "rule_hash"}),
        )
        automated = _row_at(
            semantic,
            binding.get("automated_decision_row"),
            label=f"{candidate_id} automated decision",
        )
        if any(row.get("candidate_id") != candidate_id for row in (decision, candidate, kalshi, polymarket, automated)):
            raise RuleEvidenceCaptureError(f"{candidate_id} source row identity mismatch")
        if decision.get("decision") != "NEEDS_REVIEW" or decision.get("semantic_verified") is not False:
            raise RuleEvidenceCaptureError(f"{candidate_id} is not a NEEDS_REVIEW decision")
        if decision.get("live_eligible") is not False or decision.get("orders_submitted") != 0:
            raise RuleEvidenceCaptureError(f"{candidate_id} decision violates the safety boundary")
        source_rows = _require_object(decision.get("source_rows"), label=f"{candidate_id} source rows")
        for decision_key, protocol_key in (
            ("candidate", "candidate_row"),
            ("automated_decision", "automated_decision_row"),
            ("kalshi_metadata", "kalshi_metadata_row"),
            ("polymarket_metadata", "polymarket_metadata_row"),
        ):
            expected_ref = _require_object(binding.get(protocol_key), label=f"{candidate_id} {protocol_key}")
            actual_ref = _require_object(source_rows.get(decision_key), label=f"{candidate_id} decision {decision_key}")
            if actual_ref.get("line") != expected_ref.get("line") or actual_ref.get("row_sha256") != expected_ref.get(
                "row_sha256"
            ):
                raise RuleEvidenceCaptureError(f"{candidate_id} decision/source row binding mismatch")
        for venue, row, protocol_key in (
            ("kalshi", kalshi, "kalshi_metadata_row"),
            ("polymarket", polymarket, "polymarket_metadata_row"),
        ):
            if row.get("venue") != venue or row.get("live_eligible") is not False:
                raise RuleEvidenceCaptureError(f"{candidate_id} {venue} metadata is invalid")
            reference = _require_object(binding[protocol_key], label=f"{candidate_id} {venue} reference")
            decision_reference = _require_object(source_rows[venue + "_metadata"], label=f"{candidate_id} decision ref")
            for hash_key in ("raw_sha256", "rule_hash"):
                if row.get(hash_key) != reference.get(hash_key) or decision_reference.get(hash_key) != reference.get(
                    hash_key
                ):
                    raise RuleEvidenceCaptureError(f"{candidate_id} {venue} {hash_key} binding mismatch")
        gaps = _require_array(decision.get("evidence_gaps"), label=f"{candidate_id} evidence gaps")
        actual_axes = [gap.get("axis") for gap in gaps if isinstance(gap, Mapping)]
        expected_axes = binding.get("gap_axes")
        if (
            not isinstance(expected_axes, list)
            or any(not isinstance(axis, str) or not axis for axis in expected_axes)
            or actual_axes != expected_axes
        ):
            raise RuleEvidenceCaptureError(f"{candidate_id} gap axes do not match the adjudication")
        resources = binding.get("resource_ids")
        if not isinstance(resources, list) or len(resources) != 2 or len(set(resources)) != 2:
            raise RuleEvidenceCaptureError(f"{candidate_id} must map to exactly two rule resources")
        native = _require_object(binding.get("native_ids"), label=f"{candidate_id} native IDs")
        _require_exact_keys(
            native,
            {
                "kalshi_market_id",
                "kalshi_event_id",
                "kalshi_series_id",
                "polymarket_market_id",
                "polymarket_event_id",
            },
            label=f"{candidate_id} native IDs",
        )
        if (
            kalshi.get("native_market_id") != native.get("kalshi_market_id")
            or kalshi.get("native_event_id") != native.get("kalshi_event_id")
            or kalshi.get("native_series_id") != native.get("kalshi_series_id")
            or polymarket.get("native_market_id") != native.get("polymarket_market_id")
            or polymarket.get("native_event_id") != native.get("polymarket_event_id")
        ):
            raise RuleEvidenceCaptureError(f"{candidate_id} native ID binding mismatch")
        normalized = _require_object(kalshi.get("normalized_rules"), label=f"{candidate_id} Kalshi normalized rules")
        if normalized.get("series_contract_terms_url") != binding.get("kalshi_contract_terms_url"):
            raise RuleEvidenceCaptureError(f"{candidate_id} contract-terms URL binding mismatch")
        validated.append(dict(binding))

    review_ids = {row.get("candidate_id") for _, row in decisions if row.get("decision") == "NEEDS_REVIEW"}
    if review_ids != seen or len(review_ids) != 14:
        raise RuleEvidenceCaptureError("candidate bindings are not the exact adjudicated review set")
    return {
        "adjudication_manifest": adjudication_manifest,
        "monitor_manifest": monitor_manifest,
        "candidate_bindings": validated,
    }


def _validated_runtime_dependencies(protocol: Mapping[str, Any]) -> dict[str, str]:
    expected = {
        "httpcore": EXPECTED_HTTPCORE_VERSION,
        "httpx": EXPECTED_HTTPX_VERSION,
    }
    frozen = _require_object(protocol.get("runtime_dependencies"), label="runtime dependencies")
    if frozen != expected:
        raise RuleEvidenceCaptureError("protocol runtime dependency versions are not frozen")
    observed = {
        "httpcore": str(httpcore.__version__),
        "httpx": str(httpx.__version__),
    }
    if observed != expected:
        raise RuleEvidenceCaptureError(
            "runtime dependency version mismatch: "
            f"expected httpx={EXPECTED_HTTPX_VERSION}, httpcore={EXPECTED_HTTPCORE_VERSION}"
        )
    return observed


def _validate_protocol(root: Path, protocol: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    _require_exact_keys(
        protocol,
        {
            "schema_version",
            "protocol_id",
            "implementation",
            "resource_record_schema",
            "authority",
            "limits",
            "runtime_dependencies",
            "source_adjudication_run",
            "source_monitor_run",
            "candidate_bindings",
            "resources",
            "output",
        },
        label="protocol",
    )
    if protocol.get("schema_version") != SCHEMA_VERSION or protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuleEvidenceCaptureError("unsupported rule-evidence protocol")
    _validate_safety(protocol.get("authority"), label="protocol authority")
    runtime_dependencies = _validated_runtime_dependencies(protocol)
    limits = _require_object(protocol.get("limits"), label="protocol limits")
    if limits != {
        "inactivity_timeout_seconds": int(INACTIVITY_TIMEOUT_SECONDS),
        "max_resource_bytes": MAX_RESOURCE_BYTES,
        "resource_elapsed_checkpoint_seconds": int(RESOURCE_ELAPSED_CHECKPOINT_SECONDS),
        "run_elapsed_checkpoint_seconds": int(RUN_ELAPSED_CHECKPOINT_SECONDS),
        "max_total_bytes": MAX_TOTAL_BYTES,
        "request_count": EXPECTED_RESOURCE_COUNT,
        "retries": 0,
        "sequential": True,
    }:
        raise RuleEvidenceCaptureError("protocol limits differ from the frozen bounded capture")
    _validate_implementation(root, protocol)
    schema_binding, schema_payload, _ = _bound_input(
        root,
        protocol.get("resource_record_schema"),
        label="resource record schema",
    )
    schema = _require_object(
        _strict_json(schema_payload, label="resource record schema"), label="resource record schema"
    )
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema" or schema.get("type") != "object":
        raise RuleEvidenceCaptureError("resource record schema is not Draft 2020-12 object schema")
    resources = _require_array(protocol.get("resources"), label="resources")
    if len(resources) != EXPECTED_RESOURCE_COUNT:
        raise RuleEvidenceCaptureError("protocol must contain exactly eight resources")
    validated_resources: list[Mapping[str, Any]] = []
    for resource, expected in zip(resources, _EXACT_RESOURCES):
        item = _require_object(resource, label="resource")
        _require_exact_keys(
            item,
            {
                "ordinal",
                "resource_id",
                "venue",
                "kind",
                "url",
                "expected_media_type",
                "expected_event_id",
                "expected_market_ids",
                "candidate_ids",
                "gap_axes",
            },
            label="resource",
        )
        _validate_url(item, expected)
        candidate_ids = item.get("candidate_ids")
        market_ids = item.get("expected_market_ids")
        gap_axes = item.get("gap_axes")
        if (
            not isinstance(candidate_ids, list)
            or not candidate_ids
            or len(set(candidate_ids)) != len(candidate_ids)
            or not isinstance(market_ids, list)
            or len(set(market_ids)) != len(market_ids)
            or not isinstance(gap_axes, list)
            or not gap_axes
            or len(set(gap_axes)) != len(gap_axes)
        ):
            raise RuleEvidenceCaptureError(f"resource {item.get('resource_id')} mappings are invalid")
        if item.get("kind") == "pdf" and market_ids:
            raise RuleEvidenceCaptureError("PDF resources must not declare Gamma market IDs")
        if item.get("kind") == "json" and len(market_ids) != len(candidate_ids):
            raise RuleEvidenceCaptureError("Gamma resources must bind one market ID per candidate")
        validated_resources.append(item)
    source_state = _validate_sources(root, protocol)
    bindings = source_state["candidate_bindings"]
    by_candidate = {item["candidate_id"]: item for item in bindings}
    resource_ids = {item["resource_id"] for item in validated_resources}
    mapped: dict[str, list[str]] = {candidate_id: [] for candidate_id in by_candidate}
    for resource in validated_resources:
        for candidate_id in resource["candidate_ids"]:
            if candidate_id not in by_candidate:
                raise RuleEvidenceCaptureError("resource maps an unbound candidate")
            mapped[candidate_id].append(str(resource["resource_id"]))
        expected_markets = [
            by_candidate[candidate_id]["native_ids"]["polymarket_market_id"]
            for candidate_id in resource["candidate_ids"]
        ]
        if resource["kind"] == "json" and resource["expected_market_ids"] != expected_markets:
            raise RuleEvidenceCaptureError(f"resource {resource['resource_id']} market membership binding mismatch")
        if resource["kind"] == "json" and any(
            by_candidate[candidate_id]["native_ids"]["polymarket_event_id"] != resource["expected_event_id"]
            for candidate_id in resource["candidate_ids"]
        ):
            raise RuleEvidenceCaptureError(f"resource {resource['resource_id']} event ID binding mismatch")
        if resource["kind"] == "pdf" and any(
            by_candidate[candidate_id]["kalshi_contract_terms_url"] != resource["url"]
            for candidate_id in resource["candidate_ids"]
        ):
            raise RuleEvidenceCaptureError(f"resource {resource['resource_id']} contract URL binding mismatch")
        union_axes = sorted(
            {axis for candidate_id in resource["candidate_ids"] for axis in by_candidate[candidate_id]["gap_axes"]}
        )
        if sorted(resource["gap_axes"]) != union_axes:
            raise RuleEvidenceCaptureError(f"resource {resource['resource_id']} gap-axis mapping mismatch")
    for candidate_id, binding in by_candidate.items():
        if (
            set(binding["resource_ids"]) != set(mapped[candidate_id])
            or not set(binding["resource_ids"]) <= resource_ids
        ):
            raise RuleEvidenceCaptureError(f"{candidate_id} resource mapping mismatch")
    output = _require_object(protocol.get("output"), label="output")
    run_id = output.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise RuleEvidenceCaptureError("output run ID is invalid")
    expected_output = f"data/pmxt/rule_evidence/runs/{run_id}"
    if output.get("path") != expected_output:
        raise RuleEvidenceCaptureError(f"output path must be exactly {expected_output}")
    source_state["schema_binding"] = dict(schema_binding)
    source_state["runtime_dependencies"] = runtime_dependencies
    return validated_resources, source_state


def _assert_path_identity(path: Path, expected: os.stat_result, *, label: str) -> None:
    current = _lstat_or_none(path)
    if (
        current is None
        or _is_reparse_status(expected)
        or _is_reparse_status(current)
        or not stat.S_ISDIR(expected.st_mode)
        or not stat.S_ISDIR(current.st_mode)
    ):
        raise RuleEvidenceCaptureError(f"{label} identity changed")
    expected_inode = int(expected.st_ino)
    current_inode = int(current.st_ino)
    if expected_inode == 0 or current_inode == 0:
        raise RuleEvidenceCaptureError(f"{label} has no stable filesystem identity")
    if (
        int(expected.st_dev) != int(current.st_dev)
        or expected_inode != current_inode
        or stat.S_IFMT(expected.st_mode) != stat.S_IFMT(current.st_mode)
    ):
        raise RuleEvidenceCaptureError(f"{label} identity changed")


def _assert_file_path_identity(
    path: Path,
    opened: os.stat_result,
    *,
    expected_size: int,
    label: str,
) -> None:
    current = _lstat_or_none(path)
    if (
        current is None
        or _is_reparse_status(opened)
        or _is_reparse_status(current)
        or not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or int(opened.st_ino) == 0
        or int(current.st_ino) == 0
        or int(opened.st_dev) != int(current.st_dev)
        or int(opened.st_ino) != int(current.st_ino)
        or stat.S_IFMT(opened.st_mode) != stat.S_IFMT(current.st_mode)
        or opened.st_size != expected_size
        or current.st_size != expected_size
        or getattr(opened, "st_mtime_ns", None) != getattr(current, "st_mtime_ns", None)
        or getattr(opened, "st_ctime_ns", None) != getattr(current, "st_ctime_ns", None)
    ):
        raise RuleEvidenceCaptureError(f"{label} file identity or size changed")


def _open_exclusive(path: Path) -> BinaryIO:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "wb", buffering=0)


def _write_durable(path: Path, payload: bytes) -> dict[str, Any]:
    parent_status = _lstat_or_none(path.parent)
    if parent_status is None or _is_reparse_status(parent_status) or not stat.S_ISDIR(parent_status.st_mode):
        raise RuleEvidenceCaptureError("artifact parent is not a real directory")
    digest = hashlib.sha256()
    byte_size = 0
    with _open_exclusive(path) as handle:
        view = memoryview(payload)
        while view:
            written = handle.write(view)
            if written is None or written <= 0:
                raise OSError("short write")
            digest.update(view[:written])
            byte_size += written
            view = view[written:]
        handle.flush()
        os.fsync(handle.fileno())
        opened_status = os.fstat(handle.fileno())
    _assert_path_identity(path.parent, parent_status, label="artifact parent")
    _assert_file_path_identity(path, opened_status, expected_size=byte_size, label=path.name)
    _sync_directory(path.parent)
    if byte_size != len(payload):
        raise RuleEvidenceCaptureError("artifact byte count differs from its input payload")
    return {"path": path.as_posix(), "byte_size": byte_size, "sha256": digest.hexdigest()}


def _append_durable(handle: BinaryIO, payload: bytes, *, digest: Any | None = None) -> int:
    byte_size = 0
    view = memoryview(payload)
    while view:
        written = handle.write(view)
        if written is None or written <= 0:
            raise OSError("short append")
        if digest is not None:
            digest.update(view[:written])
        byte_size += written
        view = view[written:]
    handle.flush()
    os.fsync(handle.fileno())
    return byte_size


def _relative_written_metadata(staging: Path, path: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": path.relative_to(staging).as_posix(),
        "byte_size": metadata["byte_size"],
        "sha256": metadata["sha256"],
    }


def _revalidate_tracked_artifacts(staging: Path, tracked: Mapping[str, Mapping[str, Any]]) -> None:
    """Reopen and rehash every tracked artifact through its non-following path."""

    for name, metadata in sorted(tracked.items()):
        if set(metadata) != {"path", "byte_size", "sha256"}:
            raise RuleEvidenceCaptureError(f"tracked artifact {name} binding is invalid")
        relative = _relative_path(metadata["path"], label=f"tracked artifact {name} path")
        expected_size = _require_nonnegative_int(metadata["byte_size"], label=f"tracked artifact {name} byte size")
        expected_sha = _require_sha(metadata["sha256"], label=f"tracked artifact {name} SHA-256")
        snapshot = _read_input(staging / relative, root=staging, label=f"tracked artifact {name}")
        if snapshot.binding["byte_size"] != expected_size or snapshot.binding["sha256"] != expected_sha:
            raise RuleEvidenceCaptureError(f"tracked artifact {name} no longer matches its binding")


def _monotonic_ns() -> int:
    """Private deterministic-clock seam for bounded offline tests."""

    return time.monotonic_ns()


def _check_elapsed_checkpoint(*, now_ns: int, threshold_ns: int, label: str) -> None:
    if now_ns > threshold_ns:
        raise RuleEvidenceCaptureError(f"{label} monotonic elapsed checkpoint threshold exceeded")


def _safe_error(exc: Exception) -> str:
    text = str(exc).replace("\r", " ").replace("\n", " ")
    return f"{type(exc).__name__}: {text[:500]}"


def _retained_headers(response: httpx.Response) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for raw_name, raw_value in response.headers.raw:
        try:
            name = raw_name.decode("ascii").lower()
            value = raw_value.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise RuleEvidenceCaptureError("response header name is not ASCII") from exc
        if name in _ALLOWED_HEADERS:
            result.append({"name": name, "value": value})
    return result


def _single_header(response: httpx.Response, name: str) -> str | None:
    values = [value for key, value in response.headers.multi_items() if key.lower() == name]
    if len(values) > 1:
        raise RuleEvidenceCaptureError(f"response has duplicate {name} headers")
    return values[0] if values else None


def _validate_gamma_event(payload: bytes, resource: Mapping[str, Any]) -> None:
    event = _require_object(_strict_json(payload, label="Gamma event response"), label="Gamma event response")
    if str(event.get("id")) != resource["expected_event_id"]:
        raise RuleEvidenceCaptureError("Gamma event ID mismatch")
    markets = _require_array(event.get("markets"), label="Gamma event markets")
    observed: list[str] = []
    for market in markets:
        item = _require_object(market, label="Gamma event market")
        market_id = item.get("id")
        if not isinstance(market_id, (str, int)) or isinstance(market_id, bool):
            raise RuleEvidenceCaptureError("Gamma event contains an invalid market ID")
        observed.append(str(market_id))
    if len(observed) != len(set(observed)):
        raise RuleEvidenceCaptureError("Gamma event contains duplicate market IDs")
    missing = sorted(set(resource["expected_market_ids"]) - set(observed))
    if missing:
        raise RuleEvidenceCaptureError(f"Gamma event is missing bound markets: {','.join(missing)}")


def _transport() -> httpx.BaseTransport:
    transport = httpx.HTTPTransport(retries=0, trust_env=False)
    if not isinstance(transport, httpx.BaseTransport):
        raise RuleEvidenceCaptureError("HTTP transport constructor must return an httpx.BaseTransport")
    return transport


def _capture_resource(
    *,
    resource: Mapping[str, Any],
    staging: Path,
    total_before: int,
    pinned_directories: Mapping[str, tuple[Path, os.stat_result]],
    run_checkpoint_ns: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    ordinal = int(resource["ordinal"])
    resource_id = str(resource["resource_id"])
    stem = f"{ordinal:02d}_{resource_id}"
    raw_path = staging / "raw" / f"{stem}.bin"
    header_path = staging / "headers" / f"{stem}.json"
    intent_path = staging / "requests" / f"{stem}.intent.json"
    receipt_path = staging / "requests" / f"{stem}.receipt.json"
    accept = str(resource["expected_media_type"])
    request_headers = {"Accept": accept, "Accept-Encoding": "identity"}
    resource_started_ns = _monotonic_ns()
    resource_checkpoint_ns = resource_started_ns + int(RESOURCE_ELAPSED_CHECKPOINT_SECONDS * 1_000_000_000)

    def check_elapsed_checkpoints() -> None:
        now_ns = _monotonic_ns()
        _check_elapsed_checkpoint(now_ns=now_ns, threshold_ns=resource_checkpoint_ns, label=f"resource {ordinal}")
        _check_elapsed_checkpoint(now_ns=now_ns, threshold_ns=run_checkpoint_ns, label="whole capture run")

    def check_directories() -> None:
        for label, (path, status) in pinned_directories.items():
            _assert_path_identity(path, status, label=label)

    check_directories()
    check_elapsed_checkpoints()
    intent = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "pmxt_rule_evidence_request_intent",
        "ordinal": ordinal,
        "resource_id": resource_id,
        "attempt_count": 1,
        "method": "GET",
        "url": resource["url"],
        "headers": request_headers,
        "intent_persisted_before_request": True,
        "created_at_utc": _utc_now(),
        **_SAFETY_FIELDS,
    }
    intent_payload = _canonical_json(intent)
    intent_written = _write_durable(intent_path, intent_payload)
    artifacts = {"intent": _relative_written_metadata(staging, intent_path, intent_written)}
    check_directories()
    check_elapsed_checkpoints()

    started_utc = _utc_now()
    started_ns = _monotonic_ns()
    status_code: int | None = None
    final_url: str | None = None
    retained: list[dict[str, str]] = []
    raw_hash = hashlib.sha256()
    raw_buffer = bytearray()
    byte_count = 0
    content_type: str | None = None
    declared_length: int | None = None
    redirect_count: int | None = None
    response_received = False
    body_capture_status = "NOT_CAPTURED"
    body_not_captured_reason = "request_not_sent"
    validation_error: Exception | None = None
    try:
        with ExitStack() as stack:
            check_elapsed_checkpoints()
            transport = _transport()
            client = stack.enter_context(
                httpx.Client(
                    transport=transport,
                    timeout=httpx.Timeout(INACTIVITY_TIMEOUT_SECONDS),
                    follow_redirects=False,
                    trust_env=False,
                )
            )
            request = client.build_request("GET", str(resource["url"]), headers=request_headers)
            if any(name in request.headers for name in _FORBIDDEN_REQUEST_HEADERS):
                raise RuleEvidenceCaptureError("forbidden credential, cookie, or proxy header on request")
            if request.headers.get("accept-encoding") != "identity":
                raise RuleEvidenceCaptureError("request Accept-Encoding is not identity")
            response = client.send(request, stream=True)
            stack.callback(response.close)
            response_received = True
            body_not_captured_reason = "response_body_not_yet_read"
            status_code = response.status_code
            final_url = str(response.url)
            retained = _retained_headers(response)
            redirect_count = len(response.history)
            header_document = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "pmxt_rule_evidence_response_headers",
                "ordinal": ordinal,
                "resource_id": resource_id,
                "status_code": status_code,
                "final_url": final_url,
                "raw_header_pairs_in_order": retained,
                "retention": "explicit_allowlist_no_cookies",
                **_SAFETY_FIELDS,
            }
            header_payload = _canonical_json(header_document)
            check_directories()
            header_written = _write_durable(header_path, header_payload)
            artifacts["headers"] = _relative_written_metadata(staging, header_path, header_written)
            check_directories()

            def remember_error(error: Exception) -> None:
                nonlocal validation_error
                if validation_error is None:
                    validation_error = error

            if response.history:
                remember_error(RuleEvidenceCaptureError("redirect history is forbidden"))
            if status_code != 200:
                remember_error(RuleEvidenceCaptureError(f"expected HTTP 200, received {status_code}"))
            if final_url != resource["url"]:
                remember_error(RuleEvidenceCaptureError("final response URL differs from the frozen request URL"))
            try:
                encoding = _single_header(response, "content-encoding")
                if encoding is not None and encoding.strip().lower() not in {"", "identity"}:
                    remember_error(RuleEvidenceCaptureError("response content encoding is not identity"))
                content_type_header = _single_header(response, "content-type")
                if content_type_header is None:
                    remember_error(RuleEvidenceCaptureError("response content type is missing"))
                else:
                    content_type = content_type_header.split(";", 1)[0].strip().lower()
                    if content_type != resource["expected_media_type"]:
                        remember_error(
                            RuleEvidenceCaptureError("response content type differs from the frozen media type")
                        )
                length_header = _single_header(response, "content-length")
                if length_header is not None:
                    if not length_header.isascii() or not length_header.isdigit():
                        remember_error(RuleEvidenceCaptureError("response Content-Length is invalid"))
                    else:
                        declared_length = int(length_header)
            except RuleEvidenceCaptureError as exc:
                remember_error(exc)

            skip_body = False
            if declared_length is not None and (
                declared_length > MAX_RESOURCE_BYTES or total_before + declared_length > MAX_TOTAL_BYTES
            ):
                skip_body = True
                body_not_captured_reason = "declared_content_length_exceeds_frozen_bound"
                remember_error(RuleEvidenceCaptureError("declared response size exceeds the capture bound"))
            try:
                check_elapsed_checkpoints()
            except RuleEvidenceCaptureError as exc:
                skip_body = True
                body_not_captured_reason = "monotonic_elapsed_checkpoint_exceeded_before_body"
                remember_error(exc)

            if not skip_body:
                raw_error: Exception | None = None
                opened_status: os.stat_result | None = None
                raw_handle = _open_exclusive(raw_path)
                try:
                    try:
                        for chunk in response.iter_raw():
                            check_elapsed_checkpoints()
                            if not chunk:
                                continue
                            if byte_count + len(chunk) > MAX_RESOURCE_BYTES:
                                raise RuleEvidenceCaptureError("response exceeds the per-resource byte bound")
                            if total_before + byte_count + len(chunk) > MAX_TOTAL_BYTES:
                                raise RuleEvidenceCaptureError("response exceeds the cumulative byte bound")
                            view = memoryview(chunk)
                            while view:
                                written = raw_handle.write(view)
                                if written is None or written <= 0:
                                    raise OSError("short raw response write")
                                raw_hash.update(view[:written])
                                raw_buffer.extend(view[:written])
                                byte_count += written
                                view = view[written:]
                            raw_handle.flush()
                            os.fsync(raw_handle.fileno())
                            check_elapsed_checkpoints()
                    except Exception as exc:
                        raw_error = exc
                    try:
                        raw_handle.flush()
                        os.fsync(raw_handle.fileno())
                        opened_status = os.fstat(raw_handle.fileno())
                    except Exception as exc:
                        if raw_error is None:
                            raw_error = exc
                finally:
                    raw_handle.close()
                if opened_status is not None:
                    _assert_file_path_identity(
                        raw_path,
                        opened_status,
                        expected_size=byte_count,
                        label=f"resource {ordinal} raw evidence",
                    )
                    _sync_directory(raw_path.parent)
                    artifacts["raw"] = {
                        "path": raw_path.relative_to(staging).as_posix(),
                        "byte_size": byte_count,
                        "sha256": raw_hash.hexdigest(),
                    }
                    body_capture_status = "PARTIAL" if raw_error is not None else "FULL"
                    body_not_captured_reason = None
                else:
                    body_capture_status = "NOT_CAPTURED"
                    body_not_captured_reason = "raw_evidence_durability_or_identity_not_established"
                if raw_error is not None:
                    remember_error(raw_error)

            check_directories()
            if validation_error is not None:
                raise validation_error
        raw_payload = bytes(raw_buffer)
        if len(raw_payload) != byte_count or _sha256(raw_payload) != raw_hash.hexdigest():
            raise RuleEvidenceCaptureError("persisted response bytes differ from the streamed bytes")
        if declared_length is not None and declared_length != byte_count:
            raise RuleEvidenceCaptureError("response Content-Length does not equal captured bytes")
        if resource["kind"] == "pdf":
            if not raw_payload.startswith(b"%PDF-"):
                raise RuleEvidenceCaptureError("Kalshi contract terms do not have PDF magic")
        else:
            _validate_gamma_event(raw_payload, resource)
        check_elapsed_checkpoints()
        completed_ns = _monotonic_ns()
        completed_utc = _utc_now()
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "pmxt_rule_evidence_request_receipt",
            "ordinal": ordinal,
            "resource_id": resource_id,
            "attempt_count": 1,
            "request_started_at_utc": started_utc,
            "response_completed_at_utc": completed_utc,
            "duration_monotonic_ns": completed_ns - started_ns,
            "status": "VALIDATED",
            "status_code": status_code,
            "final_url": final_url,
            "redirect_count": 0,
            "byte_count": byte_count,
            "raw_sha256": raw_hash.hexdigest(),
            "raw": artifacts["raw"],
            "headers": artifacts["headers"],
            "body_capture_status": "FULL",
            "body_not_captured_reason": None,
            **_SAFETY_FIELDS,
        }
        receipt_payload = _canonical_json(receipt)
        check_directories()
        receipt_written = _write_durable(receipt_path, receipt_payload)
        artifacts["receipt"] = _relative_written_metadata(staging, receipt_path, receipt_written)
        check_directories()
        check_elapsed_checkpoints()
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "pmxt_rule_evidence_resource",
            "ordinal": ordinal,
            "resource_id": resource_id,
            "venue": resource["venue"],
            "kind": resource["kind"],
            "url": resource["url"],
            "candidate_ids": resource["candidate_ids"],
            "gap_axes": resource["gap_axes"],
            "expected_event_id": resource["expected_event_id"],
            "expected_market_ids": resource["expected_market_ids"],
            "request_started_at_utc": started_utc,
            "response_completed_at_utc": completed_utc,
            "duration_monotonic_ns": completed_ns - started_ns,
            "attempt_count": 1,
            "status_code": status_code,
            "final_url": final_url,
            "redirect_count": 0,
            "media_type": content_type,
            "content_length": declared_length,
            "content_length_agrees": declared_length is None or declared_length == byte_count,
            "byte_count": byte_count,
            "raw": artifacts["raw"],
            "headers": artifacts["headers"],
            "intent": artifacts["intent"],
            "receipt": artifacts["receipt"],
            "raw_byte_semantics": "HTTP response content octets after transfer framing, before content decoding",
            "validation_status": "VALIDATED",
            **_SAFETY_FIELDS,
        }
        return record, artifacts
    except Exception as exc:
        completed_ns = _monotonic_ns()
        if response_received and body_not_captured_reason == "response_body_not_yet_read":
            body_not_captured_reason = "response_body_not_captured_due_pre_body_failure"
        failed_receipt = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "pmxt_rule_evidence_request_receipt",
            "ordinal": ordinal,
            "resource_id": resource_id,
            "attempt_count": 1,
            "request_started_at_utc": started_utc,
            "response_completed_at_utc": _utc_now(),
            "duration_monotonic_ns": completed_ns - started_ns,
            "status": "FAILED_CLOSED",
            "status_code": status_code,
            "final_url": final_url,
            "redirect_count": redirect_count,
            "byte_count": byte_count,
            "raw_sha256": artifacts.get("raw", {}).get("sha256"),
            "raw": artifacts.get("raw"),
            "headers": artifacts.get("headers"),
            "body_capture_status": body_capture_status,
            "body_not_captured_reason": body_not_captured_reason,
            "response_received": response_received,
            "error": _safe_error(exc),
            **_SAFETY_FIELDS,
        }
        receipt_binding: dict[str, Any] | None = None
        try:
            receipt_payload = _canonical_json(failed_receipt)
            receipt_written = _write_durable(receipt_path, receipt_payload)
            receipt_binding = _relative_written_metadata(staging, receipt_path, receipt_written)
        except Exception as receipt_exc:
            raise _ResourceCaptureFailure(
                f"{_safe_error(exc)}; failed receipt was not durably bound: {_safe_error(receipt_exc)}",
                receipt=None,
            ) from exc
        raise _ResourceCaptureFailure(_safe_error(exc), receipt=receipt_binding) from exc


def _source_bindings_document(
    protocol_binding: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "pmxt_rule_evidence_source_bindings",
        "protocol": dict(protocol_binding),
        "implementation": dict(_require_object(protocol["implementation"], label="implementation")),
        "runtime_dependencies": dict(source_state["runtime_dependencies"]),
        "resource_record_schema": dict(source_state["schema_binding"]),
        "source_adjudication_run": protocol["source_adjudication_run"],
        "source_monitor_run": protocol["source_monitor_run"],
        "candidate_bindings": protocol["candidate_bindings"],
        **_SAFETY_FIELDS,
    }


def run_rule_evidence_capture(
    *,
    repository_root: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    network_authorized: bool,
) -> Path:
    """Run the exact eight-request capture once and atomically seal its directory."""

    if not isinstance(network_authorized, bool):
        raise RuleEvidenceCaptureError("network_authorized must be a boolean")
    if network_authorized is not True:
        raise RuleEvidenceCaptureError("network capture is not explicitly authorized")
    expected_protocol_sha256 = _require_sha(expected_protocol_sha256, label="expected protocol SHA-256")
    root_candidate = Path(os.path.abspath(repository_root))
    root_status = _reject_absolute_reparse_chain(root_candidate, label="repository root")
    if not stat.S_ISDIR(root_status.st_mode):
        raise RuleEvidenceCaptureError("repository root must be a real directory")
    root = root_candidate.resolve(strict=True)
    protocol_relative = _relative_path(Path(protocol_path).as_posix(), label="protocol path")
    protocol_snapshot = _read_input(root / protocol_relative, root=root, label="protocol")
    if protocol_snapshot.binding["sha256"] != expected_protocol_sha256:
        raise RuleEvidenceCaptureError("protocol SHA-256 mismatch")
    protocol = _require_object(_strict_json(protocol_snapshot.payload, label="protocol"), label="protocol")
    resources, source_state = _validate_protocol(root, protocol)
    output = _require_object(protocol["output"], label="output")
    run_id = str(output["run_id"])
    output_parts = OUTPUT_PARENT.parts
    _reject_reparse_chain(root, output_parts, label="output parent", require_all=False)
    final_path = root / OUTPUT_PARENT / run_id
    staging = final_path.with_name(f"{run_id}.inprogress")
    if _lstat_or_none(final_path) is not None or _lstat_or_none(staging) is not None:
        raise RuleEvidenceCaptureError("final or in-progress rule-evidence output already exists")

    _assert_path_identity(root_candidate, root_status, label="repository root")
    runs_root = _ensure_real_directories(root, output_parts)
    runs_status = _lstat_or_none(runs_root)
    if runs_status is None or _is_reparse_status(runs_status) or not stat.S_ISDIR(runs_status.st_mode):
        raise RuleEvidenceCaptureError("output parent is not a real directory")
    if _lstat_or_none(final_path) is not None or _lstat_or_none(staging) is not None:
        raise RuleEvidenceCaptureError("final or in-progress rule-evidence output already exists")
    try:
        os.mkdir(staging, 0o700)
    except FileExistsError as exc:
        raise RuleEvidenceCaptureError("in-progress rule-evidence output already exists") from exc
    staging_status = _lstat_or_none(staging)
    if staging_status is None or _is_reparse_status(staging_status) or not stat.S_ISDIR(staging_status.st_mode):
        raise RuleEvidenceCaptureError("failed to reserve a real in-progress directory")
    _sync_directory(staging)
    _sync_directory(runs_root)
    run_started_ns = _monotonic_ns()
    run_checkpoint_ns = run_started_ns + int(RUN_ELAPSED_CHECKPOINT_SECONDS * 1_000_000_000)

    tracked: dict[str, dict[str, Any]] = {}
    request_ordinal = 0
    failed_receipt: Mapping[str, Any] | None = None
    try:
        pinned_directories: dict[str, tuple[Path, os.stat_result]] = {
            "in-progress directory": (staging, staging_status)
        }
        for name in ("raw", "headers", "requests"):
            path = staging / name
            os.mkdir(path, 0o700)
            status = _lstat_or_none(path)
            if status is None or _is_reparse_status(status) or not stat.S_ISDIR(status.st_mode):
                raise RuleEvidenceCaptureError(f"{name} evidence directory is not a real directory")
            pinned_directories[f"{name} evidence directory"] = (path, status)
            _sync_directory(path)
        _sync_directory(staging)
        for label, (path, status) in pinned_directories.items():
            _assert_path_identity(path, status, label=label)
        protocol_binding = {
            "path": protocol_relative.as_posix(),
            "byte_size": protocol_snapshot.binding["byte_size"],
            "sha256": protocol_snapshot.binding["sha256"],
        }
        source_bindings = _source_bindings_document(protocol_binding, protocol, source_state)
        source_payload = _canonical_json(source_bindings)
        source_path = staging / "source_bindings.json"
        source_written = _write_durable(source_path, source_payload)
        tracked["source_bindings"] = _relative_written_metadata(staging, source_path, source_written)
        plan = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "pmxt_rule_evidence_capture_plan",
            "run_id": run_id,
            "resource_count": EXPECTED_RESOURCE_COUNT,
            "resources": resources,
            "transport": {
                "method": "GET",
                "https_only": True,
                "default_port_only": True,
                "allowed_hosts": sorted(_ALLOWED_HOSTS),
                "queries_allowed": False,
                "redirects_allowed": False,
                "trust_env": False,
                "retries": 0,
                "one_client_per_request": True,
                "accept_encoding": "identity",
                "inactivity_timeout_seconds": INACTIVITY_TIMEOUT_SECONDS,
                "resource_elapsed_checkpoint_seconds": RESOURCE_ELAPSED_CHECKPOINT_SECONDS,
                "run_elapsed_checkpoint_seconds": RUN_ELAPSED_CHECKPOINT_SECONDS,
                "hard_wall_clock_deadline_enforced": False,
                "elapsed_checkpoint_semantics": (
                    "thresholds are evaluated only at explicit monotonic checkpoints and do not preempt "
                    "blocking DNS, connect, send, or read calls"
                ),
                "raw_byte_semantics": "HTTP response content octets after transfer framing, before content decoding",
            },
            "runtime_dependencies": dict(source_state["runtime_dependencies"]),
            **_SAFETY_FIELDS,
        }
        plan_payload = _canonical_json(plan)
        plan_path = staging / "capture_plan.json"
        plan_written = _write_durable(plan_path, plan_payload)
        tracked["capture_plan"] = _relative_written_metadata(staging, plan_path, plan_written)
        index_rows = []
        for binding in source_state["candidate_bindings"]:
            index_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "pmxt_rule_evidence_candidate_gap_index",
                    "candidate_id": binding["candidate_id"],
                    "decision": "NEEDS_REVIEW",
                    "gap_axes": binding["gap_axes"],
                    "resource_ids": binding["resource_ids"],
                    "semantic_verified": False,
                    "profitability_evaluation_eligible": False,
                    **_SAFETY_FIELDS,
                }
            )
        index_payload = b"".join(_canonical_json(row) for row in index_rows)
        index_path = staging / "candidate_gap_index.jsonl"
        index_written = _write_durable(index_path, index_payload)
        tracked["candidate_gap_index"] = _relative_written_metadata(staging, index_path, index_written)
        records_path = staging / "resource_records.jsonl"
        total_bytes = 0
        records: list[dict[str, Any]] = []
        records_digest = hashlib.sha256()
        records_byte_size = 0
        with _open_exclusive(records_path) as records_handle:
            for resource in resources:
                request_ordinal = int(resource["ordinal"])
                _assert_path_identity(root_candidate, root_status, label="repository root")
                for label, (path, status) in pinned_directories.items():
                    _assert_path_identity(path, status, label=label)
                _revalidate_tracked_artifacts(staging, tracked)
                record, request_artifacts = _capture_resource(
                    resource=resource,
                    staging=staging,
                    total_before=total_bytes,
                    pinned_directories=pinned_directories,
                    run_checkpoint_ns=run_checkpoint_ns,
                )
                record_payload = _canonical_json(record)
                records_byte_size += _append_durable(records_handle, record_payload, digest=records_digest)
                records.append(record)
                total_bytes += int(record["byte_count"])
                for kind, metadata in request_artifacts.items():
                    tracked[f"resource_{request_ordinal:02d}_{kind}"] = metadata
                _revalidate_tracked_artifacts(staging, tracked)
                for label, (path, status) in pinned_directories.items():
                    _assert_path_identity(path, status, label=label)
            records_handle.flush()
            os.fsync(records_handle.fileno())
            records_status = os.fstat(records_handle.fileno())
        _assert_file_path_identity(
            records_path,
            records_status,
            expected_size=records_byte_size,
            label="resource_records.jsonl",
        )
        tracked["resource_records"] = {
            "path": records_path.relative_to(staging).as_posix(),
            "byte_size": records_byte_size,
            "sha256": records_digest.hexdigest(),
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "pmxt_rule_evidence_summary",
            "run_id": run_id,
            "status": "CAPTURE_COMPLETE_REVIEW_STATUS_UNCHANGED",
            "resource_count": len(records),
            "candidate_count": len(index_rows),
            "request_attempts": len(records),
            "total_raw_bytes": total_bytes,
            "semantic_result": "NO_SEMANTIC_PROMOTION_CAPTURE_ONLY",
            "process_crash_durability": "durable intents and completed artifacts remain in permanent in-progress residue",
            "power_loss_durability": "best effort subject to filesystem and hardware fsync guarantees",
            "hard_wall_clock_deadline_enforced": False,
            "elapsed_checkpoint_semantics": (
                "resource and run thresholds are evaluated only at explicit monotonic checkpoints and do not "
                "preempt blocking DNS, connect, send, or read calls"
            ),
            **_SAFETY_FIELDS,
        }
        summary_payload = _canonical_json(summary)
        summary_path = staging / "summary.json"
        summary_written = _write_durable(summary_path, summary_payload)
        tracked["summary"] = _relative_written_metadata(staging, summary_path, summary_written)
        _revalidate_tracked_artifacts(staging, tracked)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "status": "CAPTURE_COMPLETE_REVIEW_STATUS_UNCHANGED",
            "counts": {"candidates": len(index_rows), "resources": len(records), "request_attempts": len(records)},
            "total_raw_bytes": total_bytes,
            "artifacts": dict(sorted(tracked.items())),
            **_SAFETY_FIELDS,
        }
        manifest_payload = _canonical_json(manifest)
        manifest_path = staging / "manifest.json"
        manifest_written = _write_durable(manifest_path, manifest_payload)
        tracked["manifest"] = _relative_written_metadata(staging, manifest_path, manifest_written)
        sidecar_payload = f"{_sha256(manifest_payload)}  manifest.json\n".encode("ascii")
        sidecar_path = staging / "manifest.sha256"
        sidecar_written = _write_durable(sidecar_path, sidecar_payload)
        tracked["manifest_sidecar"] = _relative_written_metadata(staging, sidecar_path, sidecar_written)
        _sync_directory(staging)
        _revalidate_tracked_artifacts(staging, tracked)
        _check_elapsed_checkpoint(now_ns=_monotonic_ns(), threshold_ns=run_checkpoint_ns, label="whole capture run")
        _assert_path_identity(root_candidate, root_status, label="repository root")
        _assert_path_identity(runs_root, runs_status, label="output parent")
        _assert_path_identity(staging, staging_status, label="in-progress directory")
        if _lstat_or_none(final_path) is not None:
            raise RuleEvidenceCaptureError("final rule-evidence output appeared before sealing")
        _rename_directory_no_replace(staging, final_path)
        _sync_directory(runs_root)
        return final_path
    except Exception as exc:
        if isinstance(exc, _ResourceCaptureFailure):
            failed_receipt = exc.receipt
        failure_path = staging / "failure.json"
        if _lstat_or_none(failure_path) is None:
            failure = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "pmxt_rule_evidence_failure",
                "run_id": run_id,
                "status": "FAILED_CLOSED_STAGING_RETAINED",
                "failed_resource_ordinal": request_ordinal or None,
                "request_attempts_at_most": request_ordinal,
                "failed_receipt": failed_receipt,
                "error": _safe_error(exc),
                "retry_permitted": False,
                "staging_must_not_be_reused_or_deleted_automatically": True,
                **_SAFETY_FIELDS,
            }
            try:
                _write_durable(failure_path, _canonical_json(failure))
                _sync_directory(staging)
            except Exception:
                pass
        if isinstance(exc, RuleEvidenceCaptureError):
            raise
        raise RuleEvidenceCaptureError(f"rule-evidence capture failed closed: {_safe_error(exc)}") from exc


__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "IMPLEMENTATION_ID",
    "PROTOCOL_ID",
    "RuleEvidenceCaptureError",
    "run_rule_evidence_capture",
]
