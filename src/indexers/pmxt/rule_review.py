"""Seal explicit offline reviews of one immutable native rule-evidence run.

This module has no network, credential, book, fee, account, order, position, or
economics surface.  It validates a caller-hash-bound rule-evidence capture and
an independently prepared reviewed-decisions JSONL file, then copies the exact
accepted decision bytes into a new no-overwrite run.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

PROTOCOL_ID = "pmxt-offline-native-rule-review-v1"
SCHEMA_VERSION = 1
OUTPUT_PARENT = Path("data/pmxt/rule_review/runs")
DEFAULT_PROTOCOL_PATH = Path("results/rule_review_v1/protocol_v1.json")

_CANDIDATE_ID_RE = re.compile(r"^pmxt_candidate_[0-9a-f]{20}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REASON_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$")
_DECISIONS = {"REJECTED", "NEEDS_REVIEW"}
_FINDING_STATES = {"EQUIVALENT", "MISMATCH", "UNRESOLVED"}
_EXPECTED_CANDIDATES = 14
_EXPECTED_RESOURCES = 8
_EXPECTED_PRIOR_TOTAL = 25
_EXPECTED_PRIOR_REJECTED = 11
_EXPECTED_REVIEW_REJECTED = 14
_EXPECTED_REVIEW_NEEDS_REVIEW = 0
_CAPTURE_SAFETY = {
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
_OUTPUT_SAFETY = {
    "books_requested": False,
    "credentials_read": False,
    "economics_computed": False,
    "fees_requested": False,
    "live_eligible": False,
    "network_requests": 0,
    "orders_submitted": 0,
    "pmxt_requests": 0,
    "positions_requested": False,
    "profitability_established": False,
}


class RuleReviewError(RuntimeError):
    """Raised when an offline review cannot be validated and sealed."""


class _DuplicateKey(ValueError):
    pass


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    payload: bytes
    binding: Mapping[str, Any]


@dataclass(frozen=True)
class _JsonlRow:
    line: int
    raw: bytes
    sha256: str
    value: Mapping[str, Any]


@dataclass(frozen=True)
class _AdjudicationState:
    run_id: str
    manifest_snapshot: _Snapshot
    sidecar_snapshot: _Snapshot
    artifact_snapshots: Mapping[str, _Snapshot]
    decision_order: tuple[str, ...]
    decisions: Mapping[str, _JsonlRow]


@dataclass(frozen=True)
class _CaptureState:
    run_id: str
    run_dir: Path
    manifest: Mapping[str, Any]
    manifest_snapshot: _Snapshot
    sidecar_snapshot: _Snapshot
    artifact_snapshots: Mapping[str, _Snapshot]
    adjudication: _AdjudicationState
    candidate_order: tuple[str, ...]
    candidates: Mapping[str, _JsonlRow]
    resources: Mapping[str, _JsonlRow]


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
        raise RuleReviewError(f"{label} is not strict UTF-8 JSON") from exc


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuleReviewError("value is not canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuleReviewError(f"{label} must be an object")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuleReviewError(f"{label} must be an array")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise RuleReviewError(f"{label} has unexpected or missing fields")


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuleReviewError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuleReviewError(f"{label} must be a non-negative integer")
    return value


def _relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuleReviewError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or path.drive or any(part in {"", ".", ".."} for part in path.parts):
        raise RuleReviewError(f"{label} must be a contained relative path")
    return path


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
        raise RuleReviewError(f"cannot inspect path component: {path}") from exc


def _reject_absolute_reparse_chain(path: Path, *, label: str) -> os.stat_result:
    if not path.is_absolute():
        raise RuleReviewError(f"{label} must be absolute")
    current = Path(path.anchor)
    status = _lstat_or_none(current)
    if status is None:
        raise RuleReviewError(f"{label} is missing")
    if _is_reparse_status(status):
        raise RuleReviewError(f"{label} must not traverse a symlink or reparse point")
    for part in path.parts[1:]:
        current /= part
        status = _lstat_or_none(current)
        if status is None:
            raise RuleReviewError(f"{label} is missing")
        if _is_reparse_status(status):
            raise RuleReviewError(f"{label} must not traverse a symlink or reparse point")
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
    if root_status is None or _is_reparse_status(root_status) or not stat.S_ISDIR(root_status.st_mode):
        raise RuleReviewError("repository root must be a real non-reparse directory")
    for index, part in enumerate(parts):
        current /= part
        status = _lstat_or_none(current)
        if status is None:
            if require_all:
                raise RuleReviewError(f"{label} is missing")
            return root.joinpath(*parts)
        if _is_reparse_status(status):
            raise RuleReviewError(f"{label} must not traverse a symlink or reparse point")
        if index < len(parts) - 1 and not stat.S_ISDIR(status.st_mode):
            raise RuleReviewError(f"{label} has a non-directory path component")
    return current


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        int(left.st_dev),
        int(left.st_ino),
        stat.S_IFMT(left.st_mode),
        left.st_size,
        getattr(left, "st_mtime_ns", None),
        getattr(left, "st_ctime_ns", None),
    ) == (
        int(right.st_dev),
        int(right.st_ino),
        stat.S_IFMT(right.st_mode),
        right.st_size,
        getattr(right, "st_mtime_ns", None),
        getattr(right, "st_ctime_ns", None),
    )


def _read_input(path: Path, *, root: Path, label: str) -> _Snapshot:
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise RuleReviewError(f"{label} is outside the repository") from exc
    _reject_reparse_chain(root, parts, label=label, require_all=True)
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuleReviewError(f"cannot open {label} without following links") from exc
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _is_reparse_status(before):
            raise RuleReviewError(f"{label} must be a regular non-reparse file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not _same_file(before, after):
        raise RuleReviewError(f"{label} changed while it was read")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise RuleReviewError(f"{label} size changed while it was read")
    path_status = _lstat_or_none(path)
    if path_status is None or _is_reparse_status(path_status) or not _same_file(after, path_status):
        raise RuleReviewError(f"{label} path changed while it was read")
    return _Snapshot(
        path=path,
        payload=payload,
        binding={"path": path.relative_to(root).as_posix(), "byte_size": len(payload), "sha256": digest.hexdigest()},
    )


def _parse_jsonl(payload: bytes, *, label: str) -> tuple[_JsonlRow, ...]:
    if not payload or not payload.endswith(b"\n"):
        raise RuleReviewError(f"{label} must be non-empty and LF-terminated")
    rows: list[_JsonlRow] = []
    for line, raw in enumerate(payload.splitlines(keepends=True), 1):
        if not raw.strip():
            raise RuleReviewError(f"{label} contains a blank row")
        value = _require_mapping(_strict_json(raw, label=f"{label} line {line}"), label=f"{label} line {line}")
        rows.append(_JsonlRow(line=line, raw=raw, sha256=_sha256(raw), value=value))
    return tuple(rows)


def _validate_capture_safety(value: Mapping[str, Any], *, label: str) -> None:
    if any(
        value.get(key) != expected or type(value.get(key)) is not type(expected)
        for key, expected in _CAPTURE_SAFETY.items()
    ):
        raise RuleReviewError(f"{label} widens the capture-only safety boundary")


def _artifact_binding(value: Any, *, label: str) -> Mapping[str, Any]:
    binding = _require_mapping(value, label=label)
    _require_exact_keys(binding, {"path", "byte_size", "sha256"}, label=label)
    _relative_path(binding["path"], label=f"{label} path")
    _require_nonnegative_int(binding["byte_size"], label=f"{label} byte size")
    _require_sha(binding["sha256"], label=f"{label} SHA-256")
    return binding


def _load_bound_artifact(run_dir: Path, name: str, value: Any) -> _Snapshot:
    binding = _artifact_binding(value, label=f"capture artifact {name}")
    relative = _relative_path(binding["path"], label=f"capture artifact {name} path")
    snapshot = _read_input(run_dir / relative, root=run_dir, label=f"capture artifact {name}")
    if snapshot.binding["byte_size"] != binding["byte_size"] or snapshot.binding["sha256"] != binding["sha256"]:
        raise RuleReviewError(f"capture artifact {name} does not match its manifest binding")
    return snapshot


def _load_repository_binding(root: Path, value: Any, *, label: str) -> _Snapshot:
    binding = _artifact_binding(value, label=label)
    relative = _relative_path(binding["path"], label=f"{label} path")
    snapshot = _read_input(root / relative, root=root, label=label)
    if snapshot.binding["byte_size"] != binding["byte_size"] or snapshot.binding["sha256"] != binding["sha256"]:
        raise RuleReviewError(f"{label} does not match its binding")
    return snapshot


def _validate_adjudication_lineage(root: Path, source_bindings_snapshot: _Snapshot) -> _AdjudicationState:
    source_bindings = _require_mapping(
        _strict_json(source_bindings_snapshot.payload, label="capture source bindings"),
        label="capture source bindings",
    )
    _validate_capture_safety(source_bindings, label="capture source bindings")
    source = _require_mapping(source_bindings.get("source_adjudication_run"), label="source adjudication run")
    run_id = source.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise RuleReviewError("source adjudication run ID is invalid")
    manifest_snapshot = _load_repository_binding(root, source.get("manifest"), label="source adjudication manifest")
    expected_manifest_path = Path("data/pmxt/semantic_adjudication/runs") / run_id / "manifest.json"
    if manifest_snapshot.path.relative_to(root) != expected_manifest_path:
        raise RuleReviewError("source adjudication manifest path is invalid")
    manifest = _require_mapping(
        _strict_json(manifest_snapshot.payload, label="source adjudication manifest"),
        label="source adjudication manifest",
    )
    authority = _require_mapping(manifest.get("authority"), label="source adjudication authority")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != "pmxt-offline-semantic-adjudication-v1"
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "OFFLINE_ADJUDICATION_COMPLETE_NO_VERIFIED_CANDIDATES"
        or authority.get("offline_only") is not True
        or authority.get("network_requests") != 0
        or authority.get("credentials_read") is not False
        or authority.get("books_requested") is not False
        or authority.get("orders_submitted") != 0
        or authority.get("economics_computed") is not False
        or authority.get("profitability_established") is not False
        or authority.get("live_eligible") is not False
    ):
        raise RuleReviewError("source adjudication manifest is not the required offline result")
    counts = _require_mapping(manifest.get("counts"), label="source adjudication counts")
    if counts != {
        "needs_review": _EXPECTED_CANDIDATES,
        "rejected": _EXPECTED_PRIOR_REJECTED,
        "total": _EXPECTED_PRIOR_TOTAL,
        "verified_equivalent": 0,
    }:
        raise RuleReviewError("source adjudication counts are not the required 11/14/0 lineage")
    run_dir = manifest_snapshot.path.parent
    sidecar_snapshot = _read_input(
        run_dir / "manifest.sha256",
        root=root,
        label="source adjudication manifest sidecar",
    )
    expected_sidecar = f"{manifest_snapshot.binding['sha256']}  manifest.json\n".encode("ascii")
    if sidecar_snapshot.payload != expected_sidecar:
        raise RuleReviewError("source adjudication manifest sidecar mismatch")

    declared = _require_mapping(manifest.get("artifacts"), label="source adjudication manifest artifacts")
    source_artifacts = _require_mapping(source.get("artifacts"), label="source adjudication source artifacts")
    if set(source_artifacts) != {"decisions", "source_bindings", "summary"}:
        raise RuleReviewError("source adjudication binding must include its exact three artifacts")
    artifact_snapshots: dict[str, _Snapshot] = {}
    for name, binding in source_artifacts.items():
        snapshot = _load_repository_binding(root, binding, label=f"source adjudication artifact {name}")
        manifest_binding = _artifact_binding(declared.get(name), label=f"source adjudication manifest artifact {name}")
        expected_path = (
            (run_dir / _relative_path(manifest_binding["path"], label=f"source artifact {name} path"))
            .relative_to(root)
            .as_posix()
        )
        if (
            snapshot.binding["path"] != expected_path
            or snapshot.binding["byte_size"] != manifest_binding["byte_size"]
            or snapshot.binding["sha256"] != manifest_binding["sha256"]
        ):
            raise RuleReviewError(f"source adjudication artifact {name} disagrees with its manifest")
        artifact_snapshots[name] = snapshot

    decision_rows = _parse_jsonl(artifact_snapshots["decisions"].payload, label="source adjudication decisions")
    if len(decision_rows) != _EXPECTED_PRIOR_TOTAL:
        raise RuleReviewError("source adjudication decisions must contain exactly 25 rows")
    decisions: dict[str, _JsonlRow] = {}
    decision_order: list[str] = []
    observed_counts = {"REJECTED": 0, "NEEDS_REVIEW": 0}
    for row in decision_rows:
        value = row.value
        candidate_id = value.get("candidate_id")
        decision = value.get("decision")
        if (
            value.get("record_type") != "pmxt_offline_adjudication"
            or not isinstance(candidate_id, str)
            or _CANDIDATE_ID_RE.fullmatch(candidate_id) is None
            or candidate_id in decisions
            or decision not in observed_counts
            or value.get("reviewed") is not True
            or value.get("semantic_verified") is not False
            or value.get("profitability_evaluation_eligible") is not False
            or value.get("network_requests") != 0
            or value.get("orders_submitted") != 0
            or value.get("economics_computed") is not False
            or value.get("live_eligible") is not False
        ):
            raise RuleReviewError(f"source adjudication decision line {row.line} is invalid")
        decisions[candidate_id] = row
        decision_order.append(candidate_id)
        observed_counts[str(decision)] += 1
    if observed_counts != {"REJECTED": _EXPECTED_PRIOR_REJECTED, "NEEDS_REVIEW": _EXPECTED_CANDIDATES}:
        raise RuleReviewError("source adjudication decision rows do not preserve the 11/14 split")
    return _AdjudicationState(
        run_id=run_id,
        manifest_snapshot=manifest_snapshot,
        sidecar_snapshot=sidecar_snapshot,
        artifact_snapshots=artifact_snapshots,
        decision_order=tuple(decision_order),
        decisions=decisions,
    )


def _validate_capture(
    *,
    root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> _CaptureState:
    manifest_snapshot = _read_input(manifest_path, root=root, label="rule-evidence manifest")
    if manifest_snapshot.binding["sha256"] != expected_manifest_sha256:
        raise RuleReviewError("rule-evidence manifest SHA-256 mismatch")
    manifest = _require_mapping(
        _strict_json(manifest_snapshot.payload, label="rule-evidence manifest"), label="rule-evidence manifest"
    )
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise RuleReviewError("rule-evidence run ID is invalid")
    expected_manifest_path = Path("data/pmxt/rule_evidence/runs") / run_id / "manifest.json"
    if manifest_path.relative_to(root) != expected_manifest_path:
        raise RuleReviewError("rule-evidence manifest path is not the exact sealed-run location")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != "pmxt-native-rule-evidence-capture-v1"
        or manifest.get("status") != "CAPTURE_COMPLETE_REVIEW_STATUS_UNCHANGED"
    ):
        raise RuleReviewError("rule-evidence manifest is not a completed v1 capture")
    _validate_capture_safety(manifest, label="rule-evidence manifest")
    counts = _require_mapping(manifest.get("counts"), label="rule-evidence manifest counts")
    if counts != {
        "candidates": _EXPECTED_CANDIDATES,
        "request_attempts": _EXPECTED_RESOURCES,
        "resources": _EXPECTED_RESOURCES,
    }:
        raise RuleReviewError("rule-evidence manifest counts are not the exact 14-candidate capture")

    run_dir = manifest_path.parent
    sidecar_snapshot = _read_input(run_dir / "manifest.sha256", root=run_dir, label="rule-evidence manifest sidecar")
    expected_sidecar = f"{expected_manifest_sha256}  manifest.json\n".encode("ascii")
    if sidecar_snapshot.payload != expected_sidecar:
        raise RuleReviewError("rule-evidence manifest sidecar mismatch")
    manifest_artifacts = _require_mapping(manifest.get("artifacts"), label="rule-evidence manifest artifacts")
    required = {"candidate_gap_index", "resource_records", "source_bindings"} | {
        f"resource_{ordinal:02d}_raw" for ordinal in range(1, _EXPECTED_RESOURCES + 1)
    }
    if not required <= set(manifest_artifacts):
        raise RuleReviewError("rule-evidence manifest omits required review artifacts")
    artifact_snapshots: dict[str, _Snapshot] = {}
    seen_paths: set[str] = set()
    for name, binding in manifest_artifacts.items():
        if not isinstance(name, str) or not name:
            raise RuleReviewError("rule-evidence manifest contains an invalid artifact name")
        item = _artifact_binding(binding, label=f"capture artifact {name}")
        if item["path"] in seen_paths:
            raise RuleReviewError("rule-evidence manifest aliases two artifacts to one path")
        seen_paths.add(str(item["path"]))
        artifact_snapshots[name] = _load_bound_artifact(run_dir, name, item)

    adjudication = _validate_adjudication_lineage(root, artifact_snapshots["source_bindings"])
    candidate_rows = _parse_jsonl(artifact_snapshots["candidate_gap_index"].payload, label="candidate gap index")
    resource_rows = _parse_jsonl(artifact_snapshots["resource_records"].payload, label="resource records")
    if len(candidate_rows) != _EXPECTED_CANDIDATES or len(resource_rows) != _EXPECTED_RESOURCES:
        raise RuleReviewError("rule-evidence row counts do not match the sealed manifest")

    resources: dict[str, _JsonlRow] = {}
    ordinal_to_resource: dict[int, str] = {}
    for row in resource_rows:
        value = row.value
        _validate_capture_safety(value, label=f"resource record line {row.line}")
        ordinal = value.get("ordinal")
        resource_id = value.get("resource_id")
        candidates = value.get("candidate_ids")
        if (
            value.get("record_type") != "pmxt_rule_evidence_resource"
            or value.get("validation_status") != "VALIDATED"
            or value.get("status_code") != 200
            or value.get("redirect_count") != 0
            or value.get("attempt_count") != 1
            or value.get("content_length_agrees") is not True
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal not in range(1, _EXPECTED_RESOURCES + 1)
            or not isinstance(resource_id, str)
            or not resource_id
            or resource_id in resources
            or ordinal in ordinal_to_resource
            or not isinstance(candidates, list)
            or not candidates
            or any(not isinstance(candidate, str) for candidate in candidates)
            or len(set(candidates)) != len(candidates)
        ):
            raise RuleReviewError(f"resource record line {row.line} is outside the validated capture envelope")
        raw = _artifact_binding(value.get("raw"), label=f"resource record line {row.line} raw")
        manifest_raw = _artifact_binding(
            manifest_artifacts[f"resource_{ordinal:02d}_raw"],
            label=f"manifest resource {ordinal} raw",
        )
        if raw != manifest_raw:
            raise RuleReviewError(f"resource record line {row.line} raw binding disagrees with the manifest")
        resources[resource_id] = row
        ordinal_to_resource[ordinal] = resource_id
    if set(ordinal_to_resource) != set(range(1, _EXPECTED_RESOURCES + 1)):
        raise RuleReviewError("resource records do not contain exact ordinals 1 through 8")

    candidates: dict[str, _JsonlRow] = {}
    candidate_order: list[str] = []
    for row in candidate_rows:
        value = row.value
        _validate_capture_safety(value, label=f"candidate gap line {row.line}")
        candidate_id = value.get("candidate_id")
        gap_axes = value.get("gap_axes")
        resource_ids = value.get("resource_ids")
        if (
            value.get("record_type") != "pmxt_rule_evidence_candidate_gap_index"
            or value.get("decision") != "NEEDS_REVIEW"
            or value.get("semantic_verified") is not False
            or value.get("profitability_evaluation_eligible") is not False
            or not isinstance(candidate_id, str)
            or _CANDIDATE_ID_RE.fullmatch(candidate_id) is None
            or candidate_id in candidates
            or not isinstance(gap_axes, list)
            or not gap_axes
            or any(not isinstance(axis, str) or not axis for axis in gap_axes)
            or len(set(gap_axes)) != len(gap_axes)
            or not isinstance(resource_ids, list)
            or len(resource_ids) != 2
            or len(set(resource_ids)) != 2
            or any(resource_id not in resources for resource_id in resource_ids)
        ):
            raise RuleReviewError(f"candidate gap line {row.line} is invalid")
        if any(candidate_id not in resources[str(resource_id)].value["candidate_ids"] for resource_id in resource_ids):
            raise RuleReviewError(f"candidate {candidate_id} is not a member of each bound evidence resource")
        candidates[candidate_id] = row
        candidate_order.append(candidate_id)
    for resource_id, row in resources.items():
        for candidate_id in row.value["candidate_ids"]:
            if candidate_id not in candidates or resource_id not in candidates[candidate_id].value["resource_ids"]:
                raise RuleReviewError("resource/candidate membership is not symmetric")
    prior_review_order = tuple(
        candidate_id
        for candidate_id in adjudication.decision_order
        if adjudication.decisions[candidate_id].value["decision"] == "NEEDS_REVIEW"
    )
    if tuple(candidate_order) != prior_review_order:
        raise RuleReviewError("rule-evidence candidates are not the exact prior NEEDS_REVIEW set and order")
    return _CaptureState(
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        manifest_snapshot=manifest_snapshot,
        sidecar_snapshot=sidecar_snapshot,
        artifact_snapshots=artifact_snapshots,
        adjudication=adjudication,
        candidate_order=tuple(candidate_order),
        candidates=candidates,
        resources=resources,
    )


def _validate_utc(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise RuleReviewError(f"{label} must be a microsecond UTC Z timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise RuleReviewError(f"{label} is not a valid timestamp") from exc
    return value


def _validate_review_row(row: _JsonlRow, *, candidate: _JsonlRow, capture: _CaptureState) -> str:
    value = row.value
    required = {
        "schema_version",
        "record_type",
        "candidate_id",
        "decision",
        "reason_codes",
        "review_method",
        "review_version",
        "reviewed_at_utc",
        "source_candidate",
        "evidence_resources",
        "axis_findings",
        "reviewed",
        "semantic_verified",
        "profitability_evaluation_eligible",
        *_OUTPUT_SAFETY,
    }
    _require_exact_keys(value, required, label=f"review line {row.line}")
    candidate_id = candidate.value["candidate_id"]
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("record_type") != "pmxt_native_rule_review_decision"
        or value.get("candidate_id") != candidate_id
        or value.get("review_method") != "offline_native_rule_clause_review"
        or value.get("review_version") != 1
        or isinstance(value.get("review_version"), bool)
        or value.get("reviewed") is not True
        or value.get("profitability_evaluation_eligible") is not False
    ):
        raise RuleReviewError(f"review line {row.line} identity or review provenance is invalid")
    _validate_utc(value.get("reviewed_at_utc"), label=f"review line {row.line} reviewed_at_utc")
    if any(
        value.get(key) != expected or type(value.get(key)) is not type(expected)
        for key, expected in _OUTPUT_SAFETY.items()
    ):
        raise RuleReviewError(f"review line {row.line} widens the offline non-executable boundary")
    source_candidate = _require_mapping(value.get("source_candidate"), label=f"review line {row.line} source candidate")
    _require_exact_keys(source_candidate, {"line", "row_sha256"}, label=f"review line {row.line} source candidate")
    if source_candidate != {"line": candidate.line, "row_sha256": candidate.sha256}:
        raise RuleReviewError(f"review line {row.line} source candidate binding mismatch")

    reason_codes = _require_list(value.get("reason_codes"), label=f"review line {row.line} reason codes")
    if (
        not reason_codes
        or any(not isinstance(code, str) or _REASON_RE.fullmatch(code) is None for code in reason_codes)
        or len(set(reason_codes)) != len(reason_codes)
    ):
        raise RuleReviewError(f"review line {row.line} reason codes are invalid")

    expected_resource_ids = list(candidate.value["resource_ids"])
    evidence_resources = _require_list(
        value.get("evidence_resources"), label=f"review line {row.line} evidence resources"
    )
    if len(evidence_resources) != len(expected_resource_ids):
        raise RuleReviewError(f"review line {row.line} must bind both candidate evidence resources")
    expected_raw_by_resource: dict[str, Mapping[str, Any]] = {}
    for index, (raw_reference, expected_resource_id) in enumerate(zip(evidence_resources, expected_resource_ids)):
        reference = _require_mapping(raw_reference, label=f"review line {row.line} evidence resource {index}")
        _require_exact_keys(
            reference,
            {"resource_id", "record_line", "record_sha256", "raw"},
            label=f"review line {row.line} evidence resource {index}",
        )
        resource = capture.resources[expected_resource_id]
        expected_raw = _artifact_binding(resource.value["raw"], label=f"resource {expected_resource_id} raw")
        if (
            reference.get("resource_id") != expected_resource_id
            or reference.get("record_line") != resource.line
            or reference.get("record_sha256") != resource.sha256
            or reference.get("raw") != expected_raw
        ):
            raise RuleReviewError(f"review line {row.line} evidence resource binding mismatch")
        expected_raw_by_resource[expected_resource_id] = expected_raw

    findings = _require_list(value.get("axis_findings"), label=f"review line {row.line} axis findings")
    expected_axes = list(candidate.value["gap_axes"])
    if len(findings) != len(expected_axes):
        raise RuleReviewError(f"review line {row.line} must decide every captured gap axis")
    finding_states: list[str] = []
    for index, (raw_finding, expected_axis) in enumerate(zip(findings, expected_axes)):
        finding = _require_mapping(raw_finding, label=f"review line {row.line} axis finding {index}")
        _require_exact_keys(
            finding,
            {"axis", "status", "rationale", "evidence"},
            label=f"review line {row.line} axis finding {index}",
        )
        status = finding.get("status")
        if (
            finding.get("axis") != expected_axis
            or status not in _FINDING_STATES
            or not isinstance(finding.get("rationale"), str)
            or not str(finding["rationale"]).strip()
        ):
            raise RuleReviewError(f"review line {row.line} axis finding {index} is invalid")
        citations = _require_list(finding.get("evidence"), label=f"review line {row.line} axis evidence {index}")
        if len(citations) != len(expected_resource_ids):
            raise RuleReviewError(f"review line {row.line} axis finding {index} must cite both venues")
        for citation_index, (raw_citation, expected_resource_id) in enumerate(zip(citations, expected_resource_ids)):
            citation = _require_mapping(
                raw_citation,
                label=f"review line {row.line} axis finding {index} citation {citation_index}",
            )
            _require_exact_keys(
                citation,
                {"resource_id", "raw_sha256", "locator"},
                label=f"review line {row.line} axis finding {index} citation {citation_index}",
            )
            if (
                citation.get("resource_id") != expected_resource_id
                or citation.get("raw_sha256") != expected_raw_by_resource[expected_resource_id]["sha256"]
                or not isinstance(citation.get("locator"), str)
                or not str(citation["locator"]).strip()
            ):
                raise RuleReviewError(f"review line {row.line} axis evidence binding mismatch")
        finding_states.append(str(status))

    decision = value.get("decision")
    if decision not in _DECISIONS:
        raise RuleReviewError(f"review line {row.line} decision state is invalid")
    if value.get("semantic_verified") is not False:
        raise RuleReviewError(f"review line {row.line} semantic_verified disagrees with its decision")
    if decision == "REJECTED" and "MISMATCH" not in finding_states:
        raise RuleReviewError("REJECTED requires at least one material MISMATCH")
    if decision == "NEEDS_REVIEW" and ("MISMATCH" in finding_states or "UNRESOLVED" not in finding_states):
        raise RuleReviewError("NEEDS_REVIEW requires an UNRESOLVED axis and no MISMATCH")
    return str(decision)


def _validate_reviews(snapshot: _Snapshot, capture: _CaptureState) -> tuple[tuple[_JsonlRow, ...], dict[str, int]]:
    rows = _parse_jsonl(snapshot.payload, label="reviewed decisions")
    if len(rows) != _EXPECTED_CANDIDATES:
        raise RuleReviewError("reviewed decisions must contain exactly 14 rows")
    counts = dict.fromkeys(sorted(_DECISIONS), 0)
    seen: set[str] = set()
    for row, expected_candidate_id in zip(rows, capture.candidate_order):
        candidate_id = row.value.get("candidate_id")
        if candidate_id in seen:
            raise RuleReviewError("reviewed decisions contain a duplicate candidate")
        if candidate_id != expected_candidate_id:
            raise RuleReviewError("reviewed decisions are not the exact source candidate order")
        seen.add(str(candidate_id))
        decision = _validate_review_row(row, candidate=capture.candidates[expected_candidate_id], capture=capture)
        counts[decision] += 1
    if seen != set(capture.candidate_order):
        raise RuleReviewError("reviewed decisions are not the exact 14 source candidates")
    if counts != {"NEEDS_REVIEW": _EXPECTED_REVIEW_NEEDS_REVIEW, "REJECTED": _EXPECTED_REVIEW_REJECTED}:
        raise RuleReviewError(
            "reviewed decisions must preserve the evidence-supported 14 REJECTED / 0 NEEDS_REVIEW split"
        )
    return rows, counts


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
            raise RuleReviewError("output path must not traverse a symlink or reparse point")
    return current


def _assert_directory_identity(path: Path, expected: os.stat_result, *, label: str) -> None:
    current = _lstat_or_none(path)
    if (
        current is None
        or _is_reparse_status(expected)
        or _is_reparse_status(current)
        or not stat.S_ISDIR(expected.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or int(expected.st_dev) != int(current.st_dev)
        or int(expected.st_ino) == 0
        or int(expected.st_ino) != int(current.st_ino)
    ):
        raise RuleReviewError(f"{label} identity changed")


def _open_exclusive(path: Path) -> BinaryIO:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    return os.fdopen(os.open(path, flags, 0o600), "wb", buffering=0)


def _write_exclusive(path: Path, payload: bytes) -> Mapping[str, Any]:
    parent_status = _lstat_or_none(path.parent)
    if parent_status is None or _is_reparse_status(parent_status) or not stat.S_ISDIR(parent_status.st_mode):
        raise RuleReviewError("artifact parent is not a real directory")
    digest = hashlib.sha256()
    byte_size = 0
    with _open_exclusive(path) as handle:
        view = memoryview(payload)
        while view:
            written = handle.write(view)
            if written is None or written <= 0:
                raise OSError("short write while sealing rule review")
            digest.update(view[:written])
            byte_size += written
            view = view[written:]
        handle.flush()
        os.fsync(handle.fileno())
        opened_status = os.fstat(handle.fileno())
    _assert_directory_identity(path.parent, parent_status, label="artifact parent")
    path_status = _lstat_or_none(path)
    if path_status is None or _is_reparse_status(path_status) or not _same_file(opened_status, path_status):
        raise RuleReviewError("sealed artifact identity changed")
    return {"path": path.name, "byte_size": byte_size, "sha256": digest.hexdigest()}


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    if _lstat_or_none(destination) is not None:
        raise RuleReviewError("final rule-review output already exists")
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise RuleReviewError("final rule-review output already exists") from exc
        except OSError as exc:
            raise RuleReviewError("exclusive rule-review finalization failed") from exc
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise RuleReviewError("atomic no-replace finalization is unavailable")
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin":
        function = getattr(library, "renamex_np", None)
        if function is None:
            raise RuleReviewError("atomic no-replace finalization is unavailable")
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(os.fsencode(source), os.fsencode(destination), 0x00000004)
    else:
        raise RuleReviewError("atomic no-replace finalization is unavailable")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise RuleReviewError("final rule-review output already exists")
    raise RuleReviewError(f"exclusive rule-review finalization failed: errno {error_number}")


def _revalidate_snapshot(snapshot: _Snapshot, *, root: Path, label: str) -> None:
    current = _read_input(snapshot.path, root=root, label=label)
    if current.binding != snapshot.binding:
        raise RuleReviewError(f"{label} changed after validation")


def _revalidate_sources(capture: _CaptureState, review: _Snapshot, *, root: Path) -> None:
    _revalidate_snapshot(capture.manifest_snapshot, root=root, label="rule-evidence manifest")
    _revalidate_snapshot(capture.sidecar_snapshot, root=capture.run_dir, label="rule-evidence manifest sidecar")
    for name, snapshot in capture.artifact_snapshots.items():
        _revalidate_snapshot(snapshot, root=capture.run_dir, label=f"capture artifact {name}")
    _revalidate_snapshot(
        capture.adjudication.manifest_snapshot,
        root=root,
        label="source adjudication manifest",
    )
    _revalidate_snapshot(
        capture.adjudication.sidecar_snapshot,
        root=root,
        label="source adjudication manifest sidecar",
    )
    for name, snapshot in capture.adjudication.artifact_snapshots.items():
        _revalidate_snapshot(snapshot, root=root, label=f"source adjudication artifact {name}")
    _revalidate_snapshot(review, root=root, label="reviewed decisions")


def _revalidate_outputs(staging: Path, bindings: Mapping[str, Mapping[str, Any]]) -> None:
    for name, binding in bindings.items():
        snapshot = _read_input(staging / str(binding["path"]), root=staging, label=f"sealed artifact {name}")
        if snapshot.binding != binding:
            raise RuleReviewError(f"sealed artifact {name} changed before finalization")


def _seal_rule_review(
    *,
    repository_root: Path,
    rule_evidence_manifest_path: Path,
    expected_rule_evidence_manifest_sha256: str,
    reviewed_decisions_path: Path,
    expected_reviewed_decisions_sha256: str,
    run_id: str,
    protocol_snapshot: _Snapshot | None,
) -> Path:
    """Validate all source bytes, then seal one offline, non-executable review."""

    expected_manifest_sha = _require_sha(
        expected_rule_evidence_manifest_sha256,
        label="expected rule-evidence manifest SHA-256",
    )
    expected_reviews_sha = _require_sha(
        expected_reviewed_decisions_sha256,
        label="expected reviewed-decisions SHA-256",
    )
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise RuleReviewError("output run ID is invalid")
    root_candidate = Path(os.path.abspath(repository_root))
    root_status = _reject_absolute_reparse_chain(root_candidate, label="repository root")
    if not stat.S_ISDIR(root_status.st_mode):
        raise RuleReviewError("repository root must be a real directory")
    root = root_candidate.resolve(strict=True)
    manifest_relative = _relative_path(
        Path(rule_evidence_manifest_path).as_posix(), label="rule-evidence manifest path"
    )
    review_relative = _relative_path(Path(reviewed_decisions_path).as_posix(), label="reviewed-decisions path")
    capture = _validate_capture(
        root=root,
        manifest_path=root / manifest_relative,
        expected_manifest_sha256=expected_manifest_sha,
    )
    review_snapshot = _read_input(root / review_relative, root=root, label="reviewed decisions")
    if review_snapshot.binding["sha256"] != expected_reviews_sha:
        raise RuleReviewError("reviewed-decisions SHA-256 mismatch")
    review_rows, counts = _validate_reviews(review_snapshot, capture)

    final_path = root / OUTPUT_PARENT / run_id
    staging_path = final_path.with_name(f"{run_id}.inprogress")
    _reject_reparse_chain(root, OUTPUT_PARENT.parts, label="rule-review output parent", require_all=False)
    if _lstat_or_none(final_path) is not None or _lstat_or_none(staging_path) is not None:
        raise RuleReviewError("final or in-progress rule-review output already exists")
    _revalidate_sources(capture, review_snapshot, root=root)

    implementation_path = Path(os.path.abspath(__file__))
    implementation_snapshot = _read_input(
        implementation_path, root=implementation_path.parent, label="rule-review implementation"
    )
    source_bindings = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "pmxt_rule_review_source_bindings",
        "protocol": dict(protocol_snapshot.binding) if protocol_snapshot is not None else None,
        "implementation": {
            "identifier": "pmxt-offline-native-rule-review-v1",
            "path": "src/indexers/pmxt/rule_review.py",
            "byte_size": implementation_snapshot.binding["byte_size"],
            "sha256": implementation_snapshot.binding["sha256"],
        },
        "rule_evidence_run": {
            "run_id": capture.run_id,
            "manifest": dict(capture.manifest_snapshot.binding),
            "manifest_sidecar": {
                "path": capture.sidecar_snapshot.path.relative_to(root).as_posix(),
                "byte_size": capture.sidecar_snapshot.binding["byte_size"],
                "sha256": capture.sidecar_snapshot.binding["sha256"],
            },
            "capture_source_bindings": dict(capture.artifact_snapshots["source_bindings"].binding),
            "candidate_gap_index": dict(capture.artifact_snapshots["candidate_gap_index"].binding),
            "resource_records": dict(capture.artifact_snapshots["resource_records"].binding),
        },
        "source_adjudication_run": {
            "run_id": capture.adjudication.run_id,
            "manifest": dict(capture.adjudication.manifest_snapshot.binding),
            "manifest_sidecar": dict(capture.adjudication.sidecar_snapshot.binding),
            "decisions": dict(capture.adjudication.artifact_snapshots["decisions"].binding),
        },
        "reviewed_decisions_input": dict(review_snapshot.binding),
        "row_sha256_semantics": "exact source JSONL line bytes including the LF terminator",
        **_OUTPUT_SAFETY,
    }
    review_by_candidate = {str(row.value["candidate_id"]): row for row in review_rows}
    final_rows: list[tuple[Mapping[str, Any], bytes]] = []
    for candidate_id in capture.adjudication.decision_order:
        prior = capture.adjudication.decisions[candidate_id]
        if prior.value["decision"] == "REJECTED":
            source_artifact = "source_adjudication_decisions"
            source_row = prior
        else:
            source_artifact = "reviewed_decisions"
            source_row = review_by_candidate[candidate_id]
        final_row = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "pmxt_rule_review_final_decision",
            "candidate_id": candidate_id,
            "decision": source_row.value["decision"],
            "decision_source": {
                "artifact": source_artifact,
                "line": source_row.line,
                "row_sha256": source_row.sha256,
            },
            "decision_payload": source_row.value,
            "reviewed": True,
            "semantic_verified": False,
            "profitability_evaluation_eligible": False,
            **_OUTPUT_SAFETY,
        }
        final_rows.append((final_row, _canonical_json(final_row)))
    final_decisions_payload = b"".join(payload for _, payload in final_rows)
    rejections_payload = b"".join(payload for row, payload in final_rows if row["decision"] == "REJECTED")
    needs_review_payload = b"".join(payload for row, payload in final_rows if row["decision"] == "NEEDS_REVIEW")
    summary_counts = {
        "total": _EXPECTED_PRIOR_TOTAL,
        "carried_rejected": _EXPECTED_PRIOR_REJECTED,
        "reviewed_total": _EXPECTED_CANDIDATES,
        "newly_rejected": counts["REJECTED"],
        "rejected": _EXPECTED_PRIOR_REJECTED + counts["REJECTED"],
        "needs_review": counts["NEEDS_REVIEW"],
        "verified_equivalent": 0,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "pmxt_rule_review_summary",
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "source_rule_evidence_run_id": capture.run_id,
        "status": "OFFLINE_RULE_REVIEW_COMPLETE_ALL_CANDIDATES_REJECTED",
        "counts": summary_counts,
        "review_input_copied_without_reclassification": True,
        **_OUTPUT_SAFETY,
    }
    contents = {
        "source_bindings": _canonical_json(source_bindings),
        "reviewed_decisions": review_snapshot.payload,
        "final_decisions": final_decisions_payload,
        "rejections": rejections_payload,
        "needs_review": needs_review_payload,
        "summary": _canonical_json(summary),
    }
    filenames = {
        "source_bindings": "source_bindings.json",
        "reviewed_decisions": "reviewed_decisions.jsonl",
        "final_decisions": "final_decisions.jsonl",
        "rejections": "rejections.jsonl",
        "needs_review": "needs_review.jsonl",
        "summary": "summary.json",
    }
    artifact_bindings = {
        name: {"path": filenames[name], "byte_size": len(payload), "sha256": _sha256(payload)}
        for name, payload in contents.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "source_rule_evidence_run_id": capture.run_id,
        "source_rule_evidence_manifest_sha256": expected_manifest_sha,
        "status": summary["status"],
        "counts": summary_counts,
        "artifacts": artifact_bindings,
        "authority": {"offline_only": True, **_OUTPUT_SAFETY},
    }
    manifest_payload = _canonical_json(manifest)
    sidecar_payload = f"{_sha256(manifest_payload)}  manifest.json\n".encode("ascii")

    output_parent = _ensure_real_directories(root, OUTPUT_PARENT.parts)
    parent_status = _lstat_or_none(output_parent)
    if parent_status is None:
        raise RuleReviewError("rule-review output parent is missing")
    if _lstat_or_none(final_path) is not None or _lstat_or_none(staging_path) is not None:
        raise RuleReviewError("final or in-progress rule-review output already exists")
    os.mkdir(staging_path, 0o700)
    staging_status = _lstat_or_none(staging_path)
    if staging_status is None or _is_reparse_status(staging_status) or not stat.S_ISDIR(staging_status.st_mode):
        raise RuleReviewError("in-progress rule-review output is not a real directory")
    try:
        written: dict[str, Mapping[str, Any]] = {}
        for name in (
            "source_bindings",
            "reviewed_decisions",
            "final_decisions",
            "rejections",
            "needs_review",
            "summary",
        ):
            written[name] = _write_exclusive(staging_path / filenames[name], contents[name])
            if written[name] != artifact_bindings[name]:
                raise RuleReviewError(f"sealed artifact {name} differs from its precomputed binding")
        written["manifest"] = _write_exclusive(staging_path / "manifest.json", manifest_payload)
        written["manifest_sidecar"] = _write_exclusive(staging_path / "manifest.sha256", sidecar_payload)
        _sync_directory(staging_path)
        _revalidate_sources(capture, review_snapshot, root=root)
        if protocol_snapshot is not None:
            _revalidate_snapshot(protocol_snapshot, root=root, label="rule-review protocol")
        _revalidate_outputs(staging_path, written)
        _assert_directory_identity(output_parent, parent_status, label="rule-review output parent")
        _assert_directory_identity(staging_path, staging_status, label="in-progress rule-review output")
        if _lstat_or_none(final_path) is not None:
            raise RuleReviewError("final rule-review output appeared before sealing")
        _rename_no_replace(staging_path, final_path)
        _sync_directory(output_parent)
    except BaseException:
        failure_path = staging_path / "failure.json"
        if _lstat_or_none(failure_path) is None:
            try:
                _write_exclusive(
                    failure_path,
                    _canonical_json(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_type": "pmxt_rule_review_failure",
                            "run_id": run_id,
                            "status": "FAILED_CLOSED_STAGING_RETAINED",
                            "retry_permitted": False,
                            **_OUTPUT_SAFETY,
                        }
                    ),
                )
                _sync_directory(staging_path)
            except (OSError, RuleReviewError):
                pass
        raise
    return final_path


def seal_rule_review(
    *,
    repository_root: Path,
    rule_evidence_manifest_path: Path,
    expected_rule_evidence_manifest_sha256: str,
    reviewed_decisions_path: Path,
    expected_reviewed_decisions_sha256: str,
    run_id: str,
) -> Path:
    """Seal an explicitly hash-bound review without a protocol document."""

    return _seal_rule_review(
        repository_root=repository_root,
        rule_evidence_manifest_path=rule_evidence_manifest_path,
        expected_rule_evidence_manifest_sha256=expected_rule_evidence_manifest_sha256,
        reviewed_decisions_path=reviewed_decisions_path,
        expected_reviewed_decisions_sha256=expected_reviewed_decisions_sha256,
        run_id=run_id,
        protocol_snapshot=None,
    )


def seal_rule_review_protocol(
    *,
    repository_root: Path,
    expected_protocol_sha256: str,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> Path:
    """Validate a frozen v1 protocol, then seal its exact inputs and output."""

    expected_protocol_sha256 = _require_sha(expected_protocol_sha256, label="expected protocol SHA-256")
    root_candidate = Path(os.path.abspath(repository_root))
    status = _reject_absolute_reparse_chain(root_candidate, label="repository root")
    if not stat.S_ISDIR(status.st_mode):
        raise RuleReviewError("repository root must be a real directory")
    root = root_candidate.resolve(strict=True)
    protocol_relative = _relative_path(Path(protocol_path).as_posix(), label="rule-review protocol path")
    protocol_snapshot = _read_input(root / protocol_relative, root=root, label="rule-review protocol")
    if protocol_snapshot.binding["sha256"] != expected_protocol_sha256:
        raise RuleReviewError("rule-review protocol SHA-256 mismatch")
    protocol = _require_mapping(
        _strict_json(protocol_snapshot.payload, label="rule-review protocol"),
        label="rule-review protocol",
    )
    _require_exact_keys(
        protocol,
        {
            "schema_version",
            "protocol_id",
            "implementation",
            "source_rule_evidence_run",
            "reviewed_decisions",
            "expected_counts",
            "output",
            "authority",
        },
        label="rule-review protocol",
    )
    if protocol.get("schema_version") != SCHEMA_VERSION or protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuleReviewError("unsupported rule-review protocol")
    implementation = _require_mapping(protocol.get("implementation"), label="rule-review implementation")
    _require_exact_keys(
        implementation,
        {"identifier", "path", "byte_size", "sha256"},
        label="rule-review implementation",
    )
    if (
        implementation.get("identifier") != PROTOCOL_ID
        or implementation.get("path") != "src/indexers/pmxt/rule_review.py"
    ):
        raise RuleReviewError("rule-review implementation identity is invalid")
    repository_implementation = _load_repository_binding(
        root,
        {key: implementation[key] for key in ("path", "byte_size", "sha256")},
        label="repository rule-review implementation",
    )
    running_path = Path(os.path.abspath(__file__))
    running_snapshot = _read_input(running_path, root=running_path.parent, label="running rule-review implementation")
    if (
        running_snapshot.payload != repository_implementation.payload
        or running_snapshot.binding["byte_size"] != implementation["byte_size"]
        or running_snapshot.binding["sha256"] != implementation["sha256"]
    ):
        raise RuleReviewError("protocol does not bind the running rule-review implementation")
    authority = _require_mapping(protocol.get("authority"), label="rule-review protocol authority")
    if authority != {"offline_only": True, **_OUTPUT_SAFETY}:
        raise RuleReviewError("rule-review protocol authority is not the exact offline boundary")
    expected_counts = _require_mapping(protocol.get("expected_counts"), label="rule-review expected counts")
    if expected_counts != {
        "source_total": _EXPECTED_PRIOR_TOTAL,
        "source_rejected": _EXPECTED_PRIOR_REJECTED,
        "reviewed_total": _EXPECTED_CANDIDATES,
        "newly_rejected": _EXPECTED_REVIEW_REJECTED,
        "final_rejected": _EXPECTED_PRIOR_TOTAL,
        "needs_review": 0,
        "verified_equivalent": 0,
    }:
        raise RuleReviewError("rule-review expected counts differ from the frozen terminal result")
    source = _require_mapping(protocol.get("source_rule_evidence_run"), label="source rule-evidence run")
    _require_exact_keys(source, {"run_id", "manifest"}, label="source rule-evidence run")
    source_run_id = source.get("run_id")
    if not isinstance(source_run_id, str) or _RUN_ID_RE.fullmatch(source_run_id) is None:
        raise RuleReviewError("source rule-evidence run ID is invalid")
    manifest_binding = _artifact_binding(source.get("manifest"), label="source rule-evidence manifest")
    expected_manifest_path = f"data/pmxt/rule_evidence/runs/{source_run_id}/manifest.json"
    if manifest_binding["path"] != expected_manifest_path:
        raise RuleReviewError("source rule-evidence manifest path is invalid")
    review_binding = _artifact_binding(protocol.get("reviewed_decisions"), label="reviewed decisions")
    output = _require_mapping(protocol.get("output"), label="rule-review output")
    _require_exact_keys(output, {"run_id", "path"}, label="rule-review output")
    run_id = output.get("run_id")
    if (
        not isinstance(run_id, str)
        or _RUN_ID_RE.fullmatch(run_id) is None
        or output.get("path") != f"data/pmxt/rule_review/runs/{run_id}"
    ):
        raise RuleReviewError("rule-review output path is invalid")
    return _seal_rule_review(
        repository_root=root,
        rule_evidence_manifest_path=Path(str(manifest_binding["path"])),
        expected_rule_evidence_manifest_sha256=str(manifest_binding["sha256"]),
        reviewed_decisions_path=Path(str(review_binding["path"])),
        expected_reviewed_decisions_sha256=str(review_binding["sha256"]),
        run_id=run_id,
        protocol_snapshot=protocol_snapshot,
    )


__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "PROTOCOL_ID",
    "RuleReviewError",
    "seal_rule_review",
    "seal_rule_review_protocol",
]
