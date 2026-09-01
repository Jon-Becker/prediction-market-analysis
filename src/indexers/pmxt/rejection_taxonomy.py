"""Derive one immutable, offline taxonomy from the sealed PMXT 25-candidate cohort.

The PMXT catalog is candidate-discovery provenance only.  Taxonomy assignments
come exclusively from the already sealed, venue-native semantic decisions and
remain post-native/pre-book.  This module has no credential, environment,
network, book, fee, account, order, position, or economics surface.
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
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA_VERSION = 1
PROTOCOL_ID = "pmxt-offline-rejection-taxonomy-v1"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CANDIDATE_ID_RE = re.compile(r"^pmxt_candidate_[0-9a-f]{20}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_AUTHORITATIVE_MANIFESTS: Mapping[str, tuple[str, str, bool]] = {
    "monitor": (
        "data/pmxt/runs/20260829T222258601245Z_18398390/manifest.json",
        "25886c6e30598a99423047755ac2b7850945839d0909fb802e2762cc4c4c5cbf",
        False,
    ),
    "adjudication": (
        "data/pmxt/semantic_adjudication/runs/20260829T222258601245Z_18398390_adjudication_v1/manifest.json",
        "72f013cb68ba604083153c7e81b03cb2e977232f66e25f5a7c405440321ffad6",
        True,
    ),
    "rule_evidence": (
        "data/pmxt/rule_evidence/runs/20260829T235900000000Z_rule_evidence_v1/manifest.json",
        "90ae288640416381ca515e4cb30f8da4862497c4d2b9c32213a0155b8e2119f7",
        True,
    ),
    "rule_review": (
        "data/pmxt/rule_review/runs/20260830T002000000000Z_rule_review_v1/manifest.json",
        "790e8e37fe5bae2491f71ed4fe1613ce916747335a62534ef978e33b79c678e6",
        True,
    ),
}

_MONITOR_COUNTS = {
    "alerts": 0,
    "bounded_clusters": 25,
    "calculations": 0,
    "candidates": 25,
    "cluster_bound_rejections": 0,
    "fee_evidence_unavailable": 0,
    "native_book_attempts": 0,
    "native_book_errors": 0,
    "native_books": 0,
    "native_books_captured": 0,
    "needs_review_candidates": 25,
    "normalized_candidate_proposals": 25,
    "normalized_candidates": 25,
    "pmxt_clusters": 25,
    "pmxt_network_requests": 0,
    "pmxt_rejections": 0,
    "raw_native_metadata": 50,
    "raw_pmxt": 1,
    "rejected_candidates": 0,
    "rejections": 25,
    "semantic_decisions": 25,
    "shadow_calculations": 0,
    "verified_candidates": 0,
}
_ADJUDICATION_COUNTS = {"needs_review": 14, "rejected": 11, "total": 25, "verified_equivalent": 0}
_RULE_EVIDENCE_COUNTS = {"candidates": 14, "request_attempts": 8, "resources": 8}
_RULE_REVIEW_COUNTS = {
    "carried_rejected": 11,
    "needs_review": 0,
    "newly_rejected": 14,
    "rejected": 25,
    "reviewed_total": 14,
    "total": 25,
    "verified_equivalent": 0,
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

_TAXONOMY_BY_SIGNATURE: Mapping[frozenset[str], tuple[str, str, int, str]] = {
    frozenset({"TERMINAL_NO_WINNER_PAYOUT_MISMATCH"}): (
        "SPORTS_TERMINAL_NO_WINNER_PAYOUT_MISMATCH",
        "Terminal or no-winner payout treatment differs",
        9,
        "Sports",
    ),
    frozenset({"PRESIDENTIAL_CALL_VS_INAUGURATION_MISMATCH"}): (
        "PRESIDENTIAL_CALL_VS_INAUGURATION",
        "Race-call and inauguration predicates differ",
        4,
        "Elections",
    ),
    frozenset(
        {
            "STRICT_NATIVE_CLOSE_INSTANT_MISMATCH",
            "TERMINAL_RULEBOOK_CLAUSE_UNCAPTURED",
            "RESOLUTION_AUTHORITY_UNRESOLVED",
            "GROUP_SET_AND_OTHER_UNRESOLVED",
        }
    ): (
        "GOP_NOMINEE_STRICT_CLOSE_WITH_UNRESOLVED_AXES",
        "Strict native close instant differs while other settlement axes remain unresolved",
        3,
        "Elections",
    ),
    frozenset({"ALTERNATE_ELECTION_AND_CUTOFF_MISMATCH"}): (
        "ISRAEL_PM_ALTERNATE_ELECTION_AND_CUTOFF",
        "Alternate-election and terminal-cutoff treatment differs",
        2,
        "Elections",
    ),
    frozenset({"HOUSE_CONTROL_DETERMINATION_MISMATCH"}): (
        "HOUSE_CONTROL_DETERMINATION_PREDICATE",
        "House-control determination predicates differ",
        2,
        "Elections",
    ),
    frozenset({"MULTIPLE_WINNER_ALPHABETICAL_TIEBREAK_MISMATCH"}): (
        "MLB_AWARD_MULTIPLE_WINNER_TIEBREAK",
        "Multiple-winner award tie-break treatment differs",
        2,
        "Sports",
    ),
    frozenset({"RESOLUTION_TRIGGER_SOURCE_SET_MISMATCH", "DEADLINE_AND_TERMINAL_FALLBACK_MISMATCH"}): (
        "GOVERNOR_TRIGGER_DEADLINE_AND_FALLBACK",
        "Trigger-source, deadline, and terminal fallback treatment differs",
        2,
        "Elections",
    ),
    frozenset({"SETTLEMENT_SOURCE_FALLBACK_MISMATCH"}): (
        "BRAZIL_SETTLEMENT_SOURCE_FALLBACK",
        "Settlement source and fallback treatment differs",
        1,
        "Elections",
    ),
}

_EXPECTED_PAIR_FAMILIES = {
    "CONTROLH-2026|32225": 2,
    "KXBRPRES-26|45915": 1,
    "KXGOVAK-26|59234": 1,
    "KXGOVCA-26|57096": 1,
    "KXISRAELPM-26OCT27|81557": 2,
    "KXMLB-26|179312": 4,
    "KXMLBNLCY-26|215675": 1,
    "KXMLBNLMVP-26|215604": 1,
    "KXPRESPERSON-28|31552": 4,
    "KXPRESNOMR-28|31875": 3,
    "KXSB-27|202857": 5,
}

_EXPECTED_REASON_INCIDENCE = {
    "ALTERNATE_ELECTION_AND_CUTOFF_MISMATCH": 2,
    "DEADLINE_AND_TERMINAL_FALLBACK_MISMATCH": 2,
    "GROUP_SET_AND_OTHER_UNRESOLVED": 3,
    "HOUSE_CONTROL_DETERMINATION_MISMATCH": 2,
    "MULTIPLE_WINNER_ALPHABETICAL_TIEBREAK_MISMATCH": 2,
    "PRESIDENTIAL_CALL_VS_INAUGURATION_MISMATCH": 4,
    "RESOLUTION_AUTHORITY_UNRESOLVED": 3,
    "RESOLUTION_TRIGGER_SOURCE_SET_MISMATCH": 2,
    "SETTLEMENT_SOURCE_FALLBACK_MISMATCH": 1,
    "STRICT_NATIVE_CLOSE_INSTANT_MISMATCH": 3,
    "TERMINAL_NO_WINNER_PAYOUT_MISMATCH": 9,
    "TERMINAL_RULEBOOK_CLAUSE_UNCAPTURED": 3,
}


class RejectionTaxonomyError(RuntimeError):
    """Raised when the frozen source cohort cannot be safely derived."""


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
class _SourceRun:
    role: str
    run_dir: Path
    manifest: Mapping[str, Any]
    manifest_snapshot: _Snapshot
    sidecar_snapshot: _Snapshot | None
    artifacts: Mapping[str, _Snapshot]


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
        raise RejectionTaxonomyError(f"{label} is not strict UTF-8 JSON") from exc


def _canonical_value(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RejectionTaxonomyError("derived value is not canonical JSON") from exc


def _canonical_json(value: Any) -> bytes:
    return _canonical_value(value) + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RejectionTaxonomyError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RejectionTaxonomyError(f"{label} must be an array")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RejectionTaxonomyError(f"{label} must be a lowercase SHA-256")
    return value


def _relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RejectionTaxonomyError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or path.drive or any(part in {"", ".", ".."} for part in path.parts):
        raise RejectionTaxonomyError(f"{label} must be a contained relative path")
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
        raise RejectionTaxonomyError(f"cannot inspect path component: {path}") from exc


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


def _reject_absolute_reparse_chain(path: Path, *, label: str) -> os.stat_result:
    if not path.is_absolute():
        raise RejectionTaxonomyError(f"{label} must be absolute")
    current = Path(path.anchor)
    status = _lstat_or_none(current)
    if status is None:
        raise RejectionTaxonomyError(f"{label} is missing")
    for part in path.parts[1:]:
        if _is_reparse_status(status):
            raise RejectionTaxonomyError(f"{label} must not traverse a symlink or reparse point")
        current /= part
        status = _lstat_or_none(current)
        if status is None:
            raise RejectionTaxonomyError(f"{label} is missing")
    if _is_reparse_status(status):
        raise RejectionTaxonomyError(f"{label} must not be a symlink or reparse point")
    return status


def _reject_reparse_chain(root: Path, parts: Sequence[str], *, label: str) -> None:
    current = root
    root_status = _lstat_or_none(root)
    if root_status is None or _is_reparse_status(root_status) or not stat.S_ISDIR(root_status.st_mode):
        raise RejectionTaxonomyError("repository root must be a real non-reparse directory")
    for index, part in enumerate(parts):
        current /= part
        status = _lstat_or_none(current)
        if status is None:
            raise RejectionTaxonomyError(f"{label} is missing")
        if _is_reparse_status(status):
            raise RejectionTaxonomyError(f"{label} must not traverse a symlink or reparse point")
        if index < len(parts) - 1 and not stat.S_ISDIR(status.st_mode):
            raise RejectionTaxonomyError(f"{label} has a non-directory path component")


def _read_input(path: Path, *, root: Path, label: str) -> _Snapshot:
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise RejectionTaxonomyError(f"{label} is outside its allowed root") from exc
    _reject_reparse_chain(root, parts, label=label)
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RejectionTaxonomyError(f"cannot open {label} without following links") from exc
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _is_reparse_status(before):
            raise RejectionTaxonomyError(f"{label} must be a regular non-reparse file")
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
        raise RejectionTaxonomyError(f"{label} changed while it was read")
    payload = b"".join(chunks)
    path_status = _lstat_or_none(path)
    if len(payload) != after.st_size or path_status is None or not _same_file(after, path_status):
        raise RejectionTaxonomyError(f"{label} changed while it was read")
    return _Snapshot(
        path=path,
        payload=payload,
        binding={"path": path.relative_to(root).as_posix(), "byte_size": len(payload), "sha256": digest.hexdigest()},
    )


def _parse_jsonl(snapshot: _Snapshot, *, label: str) -> tuple[_JsonlRow, ...]:
    if not snapshot.payload or not snapshot.payload.endswith(b"\n"):
        raise RejectionTaxonomyError(f"{label} must be non-empty and LF-terminated")
    rows: list[_JsonlRow] = []
    for line_number, raw in enumerate(snapshot.payload.splitlines(keepends=True), 1):
        if not raw.strip():
            raise RejectionTaxonomyError(f"{label} contains a blank row")
        value = _mapping(_strict_json(raw, label=f"{label} line {line_number}"), label=f"{label} line {line_number}")
        rows.append(_JsonlRow(line_number, raw, _sha256(raw), value))
    return tuple(rows)


def _artifact_binding(value: Any, *, label: str) -> Mapping[str, Any]:
    binding = _mapping(value, label=label)
    if set(binding) != {"path", "byte_size", "sha256"}:
        raise RejectionTaxonomyError(f"{label} has unexpected or missing fields")
    if isinstance(binding["byte_size"], bool) or not isinstance(binding["byte_size"], int) or binding["byte_size"] < 0:
        raise RejectionTaxonomyError(f"{label} byte size is invalid")
    _relative_path(binding["path"], label=f"{label} path")
    _sha(binding["sha256"], label=f"{label} SHA-256")
    return binding


def _load_source_run(repository_root: Path, role: str) -> _SourceRun:
    relative, expected_sha, has_sidecar = _AUTHORITATIVE_MANIFESTS[role]
    manifest_path = repository_root / _relative_path(relative, label=f"{role} manifest path")
    manifest_snapshot = _read_input(manifest_path, root=repository_root, label=f"{role} manifest")
    if manifest_snapshot.binding["sha256"] != expected_sha:
        raise RejectionTaxonomyError(f"{role} manifest is not the exact authoritative manifest")
    manifest = _mapping(_strict_json(manifest_snapshot.payload, label=f"{role} manifest"), label=f"{role} manifest")
    artifacts_value = _mapping(manifest.get("artifacts"), label=f"{role} manifest artifacts")
    run_dir = manifest_path.parent
    artifacts: dict[str, _Snapshot] = {}
    seen_paths: set[str] = set()
    for name, raw_binding in artifacts_value.items():
        if not isinstance(name, str) or not name:
            raise RejectionTaxonomyError(f"{role} manifest contains an invalid artifact name")
        binding = _artifact_binding(raw_binding, label=f"{role} artifact {name}")
        artifact_relative = _relative_path(binding["path"], label=f"{role} artifact {name} path")
        normalized = artifact_relative.as_posix()
        if normalized in seen_paths:
            raise RejectionTaxonomyError(f"{role} manifest aliases artifact paths")
        seen_paths.add(normalized)
        snapshot = _read_input(run_dir / artifact_relative, root=repository_root, label=f"{role} artifact {name}")
        if (
            snapshot.path.relative_to(run_dir).as_posix() != normalized
            or snapshot.binding["byte_size"] != binding["byte_size"]
            or snapshot.binding["sha256"] != binding["sha256"]
        ):
            raise RejectionTaxonomyError(f"{role} artifact {name} disagrees with its manifest")
        artifacts[name] = snapshot
    sidecar_snapshot = None
    if has_sidecar:
        sidecar_snapshot = _read_input(run_dir / "manifest.sha256", root=repository_root, label=f"{role} sidecar")
        if sidecar_snapshot.payload != f"{expected_sha}  manifest.json\n".encode("ascii"):
            raise RejectionTaxonomyError(f"{role} manifest sidecar mismatch")
    return _SourceRun(role, run_dir, manifest, manifest_snapshot, sidecar_snapshot, artifacts)


def _binding_matches(value: Any, snapshot: _Snapshot, *, label: str) -> None:
    binding = _artifact_binding(value, label=label)
    if binding != snapshot.binding:
        raise RejectionTaxonomyError(f"{label} does not bind the authoritative source")


def _contains_legacy_reference(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.replace("\\", "/").lower()
        return "semantic_equivalence/runs/capture_001" in lowered or "capture_001" in lowered
    if isinstance(value, Mapping):
        return any(_contains_legacy_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_legacy_reference(item) for item in value)
    return False


def _validate_source_manifests(runs: Mapping[str, _SourceRun]) -> None:
    monitor = runs["monitor"]
    adjudication = runs["adjudication"]
    evidence = runs["rule_evidence"]
    review = runs["rule_review"]
    if (
        monitor.manifest.get("schema_version") != 1
        or monitor.manifest.get("run_id") != "20260829T222258601245Z_18398390"
        or monitor.manifest.get("status") != "NO_VERIFIED_CANDIDATES"
        or monitor.manifest.get("counts") != _MONITOR_COUNTS
        or monitor.manifest.get("live_eligible") is not False
        or monitor.manifest.get("no_order_actions") is not True
    ):
        raise RejectionTaxonomyError("monitor manifest does not preserve the exact safe terminal result")
    if (
        adjudication.manifest.get("schema_version") != 1
        or adjudication.manifest.get("protocol_id") != "pmxt-offline-semantic-adjudication-v1"
        or adjudication.manifest.get("run_id") != "20260829T222258601245Z_18398390_adjudication_v1"
        or adjudication.manifest.get("source_run_id") != monitor.manifest.get("run_id")
        or adjudication.manifest.get("source_manifest_sha256") != monitor.manifest_snapshot.binding["sha256"]
        or adjudication.manifest.get("status") != "OFFLINE_ADJUDICATION_COMPLETE_NO_VERIFIED_CANDIDATES"
        or adjudication.manifest.get("counts") != _ADJUDICATION_COUNTS
    ):
        raise RejectionTaxonomyError("adjudication manifest does not preserve the exact safe result")
    adjudication_authority = _mapping(adjudication.manifest.get("authority"), label="adjudication authority")
    if adjudication_authority != {
        "books_requested": False,
        "credentials_read": False,
        "economics_computed": False,
        "live_eligible": False,
        "network_requests": 0,
        "offline_only": True,
        "orders_submitted": 0,
        "profitability_established": False,
    }:
        raise RejectionTaxonomyError("adjudication authority is not the exact offline boundary")
    if (
        evidence.manifest.get("schema_version") != 1
        or evidence.manifest.get("protocol_id") != "pmxt-native-rule-evidence-capture-v1"
        or evidence.manifest.get("run_id") != "20260829T235900000000Z_rule_evidence_v1"
        or evidence.manifest.get("status") != "CAPTURE_COMPLETE_REVIEW_STATUS_UNCHANGED"
        or evidence.manifest.get("counts") != _RULE_EVIDENCE_COUNTS
        or any(
            evidence.manifest.get(key) != expected
            for key, expected in {
                "books_requested": False,
                "credentials_read": False,
                "economics_computed": False,
                "fees_requested": False,
                "live_eligible": False,
                "orders_submitted": 0,
                "pmxt_requests": 0,
                "positions_requested": False,
                "semantic_promotions": 0,
            }.items()
        )
    ):
        raise RejectionTaxonomyError("rule-evidence manifest does not preserve its bounded safety result")
    if (
        review.manifest.get("schema_version") != 1
        or review.manifest.get("protocol_id") != "pmxt-offline-native-rule-review-v1"
        or review.manifest.get("run_id") != "20260830T002000000000Z_rule_review_v1"
        or review.manifest.get("source_rule_evidence_run_id") != evidence.manifest.get("run_id")
        or review.manifest.get("source_rule_evidence_manifest_sha256") != evidence.manifest_snapshot.binding["sha256"]
        or review.manifest.get("status") != "OFFLINE_RULE_REVIEW_COMPLETE_ALL_CANDIDATES_REJECTED"
        or review.manifest.get("counts") != _RULE_REVIEW_COUNTS
        or review.manifest.get("authority") != {"offline_only": True, **_OUTPUT_SAFETY}
    ):
        raise RejectionTaxonomyError("rule-review manifest does not preserve the exact terminal result")

    adjudication_bindings = _mapping(
        _strict_json(adjudication.artifacts["source_bindings"].payload, label="adjudication source bindings"),
        label="adjudication source bindings",
    )
    evidence_bindings = _mapping(
        _strict_json(evidence.artifacts["source_bindings"].payload, label="rule-evidence source bindings"),
        label="rule-evidence source bindings",
    )
    review_bindings = _mapping(
        _strict_json(review.artifacts["source_bindings"].payload, label="rule-review source bindings"),
        label="rule-review source bindings",
    )
    if any(_contains_legacy_reference(value) for value in (adjudication_bindings, evidence_bindings, review_bindings)):
        raise RejectionTaxonomyError("legacy semantic-equivalence capture is excluded from authoritative lineage")
    _binding_matches(
        _mapping(adjudication_bindings.get("source_monitor_run"), label="adjudication monitor binding").get("manifest"),
        monitor.manifest_snapshot,
        label="adjudication monitor manifest binding",
    )
    _binding_matches(
        _mapping(evidence_bindings.get("source_monitor_run"), label="rule-evidence monitor binding").get("manifest"),
        monitor.manifest_snapshot,
        label="rule-evidence monitor manifest binding",
    )
    _binding_matches(
        _mapping(evidence_bindings.get("source_adjudication_run"), label="rule-evidence adjudication binding").get(
            "manifest"
        ),
        adjudication.manifest_snapshot,
        label="rule-evidence adjudication manifest binding",
    )
    _binding_matches(
        _mapping(review_bindings.get("rule_evidence_run"), label="rule-review evidence binding").get("manifest"),
        evidence.manifest_snapshot,
        label="rule-review evidence manifest binding",
    )
    _binding_matches(
        _mapping(review_bindings.get("source_adjudication_run"), label="rule-review adjudication binding").get(
            "manifest"
        ),
        adjudication.manifest_snapshot,
        label="rule-review adjudication manifest binding",
    )


def _row_reference(artifact: str, row: _JsonlRow) -> Mapping[str, Any]:
    return {"artifact": artifact, "line": row.line, "row_sha256": row.sha256}


def _indexed_rows(
    snapshot: _Snapshot,
    *,
    label: str,
    expected_count: int,
    ordered_ids: Sequence[str] | None = None,
) -> tuple[tuple[_JsonlRow, ...], dict[str, _JsonlRow]]:
    rows = _parse_jsonl(snapshot, label=label)
    if len(rows) != expected_count:
        raise RejectionTaxonomyError(f"{label} does not contain exactly {expected_count} rows")
    index: dict[str, _JsonlRow] = {}
    for row in rows:
        candidate_id = row.value.get("candidate_id")
        if not isinstance(candidate_id, str) or _CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
            raise RejectionTaxonomyError(f"{label} contains an invalid candidate ID")
        if candidate_id in index:
            raise RejectionTaxonomyError(f"{label} contains duplicate candidate IDs")
        index[candidate_id] = row
    if ordered_ids is not None and tuple(index) != tuple(ordered_ids):
        raise RejectionTaxonomyError(f"{label} does not preserve the exact candidate order")
    return rows, index


def _validate_source_row_reference(reference: Any, artifact: str, row: _JsonlRow, *, label: str) -> None:
    value = _mapping(reference, label=label)
    if value.get("artifact") != artifact or value.get("line") != row.line or value.get("row_sha256") != row.sha256:
        raise RejectionTaxonomyError(f"{label} does not bind the expected source row")
    if artifact == "raw_native_metadata" and (
        value.get("raw_sha256") != row.value.get("raw_sha256") or value.get("rule_hash") != row.value.get("rule_hash")
    ):
        raise RejectionTaxonomyError(f"{label} does not bind native raw and rule hashes")


def _native_identity(row: _JsonlRow) -> Mapping[str, Any]:
    value = row.value
    for field in ("native_market_id", "native_event_id"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise RejectionTaxonomyError(f"native metadata {field} is invalid")
    series = value.get("native_series_id")
    if series is not None and (not isinstance(series, str) or not series):
        raise RejectionTaxonomyError("native metadata series ID is invalid")
    return {
        "venue": value["venue"],
        "side": value["side"],
        "native_market_id": value["native_market_id"],
        "native_event_id": value["native_event_id"],
        "native_series_id": series,
        "native_outcome_ids": value.get("native_outcome_ids"),
        "raw_sha256": value["raw_sha256"],
        "rule_hash": value["rule_hash"],
        "source_row": _row_reference("raw_native_metadata", row),
    }


def _reason_codes(value: Any, *, candidate_id: str) -> list[str]:
    codes = _list(value, label=f"terminal reason codes for {candidate_id}")
    if not codes or any(not isinstance(code, str) or not code for code in codes) or len(set(codes)) != len(codes):
        raise RejectionTaxonomyError(f"terminal reason codes for {candidate_id} are invalid")
    return codes


def _classify_reason_codes(codes: Sequence[str]) -> tuple[str, str, int, str]:
    classification = _TAXONOMY_BY_SIGNATURE.get(frozenset(codes))
    if classification is None:
        raise RejectionTaxonomyError("terminal native decision has an unknown rejection taxonomy signature")
    return classification


def _derive_documents(runs: Mapping[str, _SourceRun], *, run_id: str) -> Mapping[str, bytes]:
    monitor = runs["monitor"]
    adjudication = runs["adjudication"]
    evidence = runs["rule_evidence"]
    review = runs["rule_review"]
    candidate_rows, candidates = _indexed_rows(
        monitor.artifacts["candidates"], label="monitor candidates", expected_count=25
    )
    candidate_order = tuple(candidates)
    for row in candidate_rows:
        value = row.value
        if (
            value.get("relation") != "identity"
            or value.get("live_eligible") is not False
            or {value.get("venue_a"), value.get("venue_b")} != {"kalshi", "polymarket"}
        ):
            raise RejectionTaxonomyError("candidate is outside the identity-only, live-ineligible boundary")
    semantic_rows, semantics = _indexed_rows(
        monitor.artifacts["semantic_decisions"],
        label="monitor semantic decisions",
        expected_count=25,
        ordered_ids=candidate_order,
    )
    for row in semantic_rows:
        if row.value.get("status") != "NEEDS_REVIEW" or row.value.get("live_eligible") is not False:
            raise RejectionTaxonomyError("monitor semantic decisions must all be live-ineligible NEEDS_REVIEW")

    native_rows = _parse_jsonl(monitor.artifacts["raw_native_metadata"], label="monitor native metadata")
    if len(native_rows) != 50:
        raise RejectionTaxonomyError("monitor native metadata must contain exactly 50 rows")
    native: dict[tuple[str, str], _JsonlRow] = {}
    for row in native_rows:
        value = row.value
        candidate_id = value.get("candidate_id")
        venue = value.get("venue")
        if candidate_id not in candidates or venue not in {"kalshi", "polymarket"}:
            raise RejectionTaxonomyError("native metadata is not bound to a cohort candidate and venue")
        key = (str(candidate_id), str(venue))
        if key in native:
            raise RejectionTaxonomyError("native metadata contains duplicate candidate/venue rows")
        side = value.get("side")
        if (
            side not in {"a", "b"}
            or candidates[str(candidate_id)].value.get(f"venue_{side}") != venue
            or value.get("status") != "RESOLVED"
            or value.get("live_eligible") is not False
        ):
            raise RejectionTaxonomyError("native metadata is outside the resolved live-ineligible boundary")
        raw_sha = _sha(value.get("raw_sha256"), label="native raw SHA-256")
        rule_sha = _sha(value.get("rule_hash"), label="native rule SHA-256")
        if _sha256(_canonical_value(value.get("raw_response"))) != raw_sha:
            raise RejectionTaxonomyError("native raw-response hash mismatch")
        if _sha256(_canonical_value(value.get("normalized_rules"))) != rule_sha:
            raise RejectionTaxonomyError("native normalized-rule hash mismatch")
        native[key] = row
    if set(native) != {(candidate_id, venue) for candidate_id in candidate_order for venue in ("kalshi", "polymarket")}:
        raise RejectionTaxonomyError("native metadata is not exactly two venues per candidate")

    adjudication_rows, adjudications = _indexed_rows(
        adjudication.artifacts["decisions"],
        label="adjudication decisions",
        expected_count=25,
        ordered_ids=candidate_order,
    )
    adjudication_counts = Counter(str(row.value.get("decision")) for row in adjudication_rows)
    if adjudication_counts != Counter({"REJECTED": 11, "NEEDS_REVIEW": 14}):
        raise RejectionTaxonomyError("adjudication decisions do not preserve the exact 11/14 split")
    for candidate_id, row in adjudications.items():
        value = row.value
        if any(
            value.get(key) != expected
            for key, expected in {
                "reviewed": True,
                "semantic_verified": False,
                "profitability_evaluation_eligible": False,
                "economics_computed": False,
                "network_requests": 0,
                "orders_submitted": 0,
                "live_eligible": False,
            }.items()
        ):
            raise RejectionTaxonomyError("adjudication decision violates the safe offline boundary")
        source_rows = _mapping(value.get("source_rows"), label="adjudication source rows")
        _validate_source_row_reference(
            source_rows.get("candidate"), "candidates", candidates[candidate_id], label="candidate row"
        )
        _validate_source_row_reference(
            source_rows.get("automated_decision"),
            "semantic_decisions",
            semantics[candidate_id],
            label="semantic-decision row",
        )
        for venue in ("kalshi", "polymarket"):
            _validate_source_row_reference(
                source_rows.get(f"{venue}_metadata"),
                "raw_native_metadata",
                native[(candidate_id, venue)],
                label=f"{venue} native row",
            )

    gap_rows, gap_index = _indexed_rows(
        evidence.artifacts["candidate_gap_index"],
        label="rule-evidence candidate gap index",
        expected_count=14,
    )
    expected_review_ids = tuple(
        candidate_id
        for candidate_id in candidate_order
        if adjudications[candidate_id].value["decision"] == "NEEDS_REVIEW"
    )
    if tuple(gap_index) != expected_review_ids:
        raise RejectionTaxonomyError("rule-evidence gap index does not match the exact 14 review candidates")
    for row in gap_rows:
        if (
            row.value.get("decision") != "NEEDS_REVIEW"
            or row.value.get("semantic_verified") is not False
            or row.value.get("profitability_evaluation_eligible") is not False
            or row.value.get("live_eligible") is not False
            or row.value.get("books_requested") is not False
            or row.value.get("economics_computed") is not False
            or row.value.get("orders_submitted") != 0
        ):
            raise RejectionTaxonomyError("rule-evidence gap row violates the pre-book boundary")

    reviewed_rows, reviewed = _indexed_rows(
        review.artifacts["reviewed_decisions"],
        label="reviewed native decisions",
        expected_count=14,
        ordered_ids=expected_review_ids,
    )
    for row in reviewed_rows:
        if row.value.get("decision") != "REJECTED" or row.value.get("live_eligible") is not False:
            raise RejectionTaxonomyError("reviewed native decisions must all be live-ineligible REJECTED")
    terminal_rows, terminals = _indexed_rows(
        review.artifacts["final_decisions"],
        label="terminal decisions",
        expected_count=25,
        ordered_ids=candidate_order,
    )
    for candidate_id, row in terminals.items():
        value = row.value
        if (
            value.get("decision") != "REJECTED"
            or value.get("reviewed") is not True
            or value.get("semantic_verified") is not False
            or value.get("profitability_evaluation_eligible") is not False
            or any(value.get(key) != expected for key, expected in _OUTPUT_SAFETY.items())
        ):
            raise RejectionTaxonomyError("terminal decision violates the all-rejected offline boundary")
        decision_source = _mapping(value.get("decision_source"), label="terminal decision source")
        artifact = decision_source.get("artifact")
        expected_row = (
            adjudications[candidate_id] if artifact == "source_adjudication_decisions" else reviewed.get(candidate_id)
        )
        if expected_row is None or artifact not in {"source_adjudication_decisions", "reviewed_decisions"}:
            raise RejectionTaxonomyError("terminal decision has an invalid decision source")
        if decision_source.get("line") != expected_row.line or decision_source.get("row_sha256") != expected_row.sha256:
            raise RejectionTaxonomyError("terminal decision source row binding mismatch")
        if value.get("decision_payload") != expected_row.value:
            raise RejectionTaxonomyError("terminal decision payload differs from its native decision source")

    taxonomy_counts: Counter[str] = Counter()
    reason_incidence: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    pair_families: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    output_rows: list[Mapping[str, Any]] = []
    for candidate_id in candidate_order:
        candidate = candidates[candidate_id]
        semantic = semantics[candidate_id]
        adjudication_row = adjudications[candidate_id]
        terminal = terminals[candidate_id]
        terminal_payload = _mapping(terminal.value["decision_payload"], label="terminal decision payload")
        codes = _reason_codes(terminal_payload.get("reason_codes"), candidate_id=candidate_id)
        taxonomy_id, taxonomy_label, _expected_count, expected_category = _classify_reason_codes(codes)
        category = candidate.value.get("category")
        if category != expected_category:
            raise RejectionTaxonomyError("taxonomy category disagrees with the source candidate category")
        taxonomy_counts[taxonomy_id] += 1
        category_counts[str(category)] += 1
        reason_incidence.update(codes)
        transition = (
            f"{semantic.value['status']} -> {adjudication_row.value['decision']} -> {terminal.value['decision']}"
        )
        transitions[transition] += 1
        kalshi = native[(candidate_id, "kalshi")]
        polymarket = native[(candidate_id, "polymarket")]
        pair_family = f"{kalshi.value['native_event_id']}|{polymarket.value['native_event_id']}"
        pair_families[pair_family] += 1
        source_artifact = str(terminal.value["decision_source"]["artifact"])
        evidence_field = (
            "material_mismatches" if source_artifact == "source_adjudication_decisions" else "axis_findings"
        )
        findings = _list(terminal_payload.get(evidence_field), label=f"native findings for {candidate_id}")
        secondary_codes = [code for code in codes if code.endswith("_UNRESOLVED") or code.endswith("_UNCAPTURED")]
        primary_codes = [code for code in codes if code not in secondary_codes]
        output_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "pmxt_rejection_taxonomy_row",
                "taxonomy_run_id": run_id,
                "candidate_id": candidate_id,
                "category": category,
                "terminal_decision": "REJECTED",
                "taxonomy_id": taxonomy_id,
                "taxonomy_label": taxonomy_label,
                "primary_reason_codes": primary_codes,
                "secondary_unresolved_reason_codes": secondary_codes,
                "review_provenance": (
                    "offline_native_adjudication"
                    if source_artifact == "source_adjudication_decisions"
                    else "official_native_rule_review"
                ),
                "strict_policy_dependency": taxonomy_id == "GOP_NOMINEE_STRICT_CLOSE_WITH_UNRESOLVED_AXES",
                "pair_family_id": pair_family,
                "pmxt_discovery": {
                    "source": candidate.value.get("source"),
                    "cluster_id": candidate.value.get("cluster_id"),
                    "snapshot_id": candidate.value.get("snapshot_id"),
                    "pmxt_market_ids": [
                        candidate.value.get("pmxt_market_id_a"),
                        candidate.value.get("pmxt_market_id_b"),
                    ],
                    "proposed_relation": candidate.value.get("relation"),
                    "relation_confidence": candidate.value.get("relation_confidence"),
                    "catalog_semantic_fields_authoritative": False,
                    "catalog_semantic_fields_can_hard_reject": False,
                    "role": "candidate_discovery_only",
                },
                "semantic_evidence_boundary": {
                    "stage": "POST_NATIVE_PRE_BOOK",
                    "authoritative_source": "venue_native_metadata_rules_and_official_rule_evidence",
                    "pmxt_catalog_semantics_authoritative": False,
                    "books_consulted": False,
                },
                "native_markets": [_native_identity(kalshi), _native_identity(polymarket)],
                "native_semantic_findings": {
                    "source_field": f"decision_payload.{evidence_field}",
                    "findings": findings,
                },
                "stage_history": [
                    {
                        "stage": "monitor_semantic",
                        "decision": "NEEDS_REVIEW",
                        "source_row": _row_reference("semantic_decisions", semantic),
                    },
                    {
                        "stage": "offline_adjudication",
                        "decision": adjudication_row.value["decision"],
                        "source_row": _row_reference("adjudication_decisions", adjudication_row),
                    },
                    {
                        "stage": "terminal_rule_review",
                        "decision": "REJECTED",
                        "source_row": _row_reference("final_decisions", terminal),
                    },
                ],
                "source_candidate": _row_reference("candidates", candidate),
                "profitability_evaluation_eligible": False,
                **_OUTPUT_SAFETY,
            }
        )

    expected_taxonomy_counts = {value[0]: value[2] for value in _TAXONOMY_BY_SIGNATURE.values()}
    if dict(taxonomy_counts) != expected_taxonomy_counts:
        raise RejectionTaxonomyError("derived taxonomy counts differ from the exact audited 25-candidate result")
    if dict(reason_incidence) != _EXPECTED_REASON_INCIDENCE:
        raise RejectionTaxonomyError("overlapping reason-code incidence differs from the audited cohort")
    if dict(transitions) != {
        "NEEDS_REVIEW -> REJECTED -> REJECTED": 11,
        "NEEDS_REVIEW -> NEEDS_REVIEW -> REJECTED": 14,
    }:
        raise RejectionTaxonomyError("stage transitions differ from the exact audited cohort")
    if dict(pair_families) != _EXPECTED_PAIR_FAMILIES:
        raise RejectionTaxonomyError("native pair-family counts differ from the audited cohort")
    if dict(category_counts) != {"Elections": 14, "Sports": 11}:
        raise RejectionTaxonomyError("source category counts differ from the audited cohort")

    source_bindings_runs: dict[str, Any] = {}
    for role, source in runs.items():
        source_bindings_runs[role] = {
            "run_id": source.manifest.get("run_id"),
            "manifest": dict(source.manifest_snapshot.binding),
            "manifest_sidecar": dict(source.sidecar_snapshot.binding) if source.sidecar_snapshot else None,
            "manifest_artifacts": {name: dict(snapshot.binding) for name, snapshot in source.artifacts.items()},
            "all_manifest_listed_artifacts_hash_validated": True,
        }
    implementation_path = Path(os.path.abspath(__file__))
    implementation = _read_input(
        implementation_path, root=implementation_path.parent, label="rejection-taxonomy implementation"
    )
    source_bindings = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "pmxt_rejection_taxonomy_source_bindings",
        "taxonomy_run_id": run_id,
        "protocol_id": PROTOCOL_ID,
        "implementation": {
            "identifier": PROTOCOL_ID,
            "path": "src/indexers/pmxt/rejection_taxonomy.py",
            "byte_size": implementation.binding["byte_size"],
            "sha256": implementation.binding["sha256"],
        },
        "authoritative_runs": source_bindings_runs,
        "source_manifest_count": 4,
        "all_listed_source_artifacts_hash_validated": True,
        "row_sha256_semantics": "exact source JSONL line bytes including the LF terminator",
        "legacy_capture": {
            "path": "data/pmxt/semantic_equivalence/runs/capture_001",
            "authoritative": False,
            "read_or_ingested": False,
            "exclusion_reason": "legacy capture is outside the exact four-manifest authoritative lineage",
        },
        "pmxt_catalog_role": "candidate_discovery_only_non_authoritative_for_semantic_rejection",
        "semantic_evidence_stage": "POST_NATIVE_PRE_BOOK",
        **_OUTPUT_SAFETY,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "pmxt_rejection_taxonomy_summary",
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "status": "OFFLINE_REJECTION_TAXONOMY_COMPLETE_ALL_CANDIDATES_REJECTED",
        "counts": {"total": 25, "rejected": 25, "verified_equivalent": 0, "needs_review": 0},
        "counts_by_taxonomy_id": dict(sorted(taxonomy_counts.items())),
        "overlapping_reason_code_incidence": dict(sorted(reason_incidence.items())),
        "stage_transition_counts": dict(sorted(transitions.items())),
        "pair_family_counts": dict(sorted(pair_families.items())),
        "source_category_counts": dict(sorted(category_counts.items())),
        "pmxt_catalog_semantic_fields_authoritative": False,
        "pmxt_catalog_semantic_fields_can_hard_reject": False,
        "native_semantic_axes_stage": "POST_NATIVE_PRE_BOOK",
        "terminal_result": "NO_VERIFIED_CANDIDATES",
        **_OUTPUT_SAFETY,
    }
    caveat_values = [
        (
            "MONITOR_REJECTIONS_ARE_FUNNEL_LOG",
            "monitor",
            "The monitor rejections artifact records 25 NEEDS_REVIEW funnel outcomes, not terminal semantic rejections.",
        ),
        (
            "TERMINAL_REJECTIONS_DUPLICATE_FINAL_DECISIONS",
            "rule_review",
            "Terminal rejections.jsonl is byte-identical to final_decisions.jsonl and is not independent corroboration.",
        ),
        (
            "DERIVED_REPORT_NOT_NEW_SEMANTIC_EVIDENCE",
            "taxonomy",
            "This taxonomy copies and hash-binds sealed decisions; it does not independently reproduce semantic judgment.",
        ),
        (
            "STRICT_CLOSE_REJECTION_HAS_UNRESOLVED_AXES",
            "three_nominee_candidates",
            "The strict close-instant mismatch is established, while terminal, authority, and group-set axes remain unresolved.",
        ),
        (
            "LEGACY_CAPTURE_EXCLUDED",
            "lineage",
            "The legacy semantic-equivalence capture is outside the authoritative lineage and was not read or ingested.",
        ),
        (
            "MONITOR_MANIFEST_HAS_NO_SIDECAR",
            "monitor",
            "The monitor manifest is exact-hash-bound here and downstream, but its source run has no adjacent sidecar.",
        ),
        (
            "HTTP_CONTENT_OCTETS_NOT_WIRE_TRANSCRIPT",
            "rule_evidence",
            "Captured response content octets are not a TCP or TLS wire transcript.",
        ),
        (
            "PMXT_CONFIDENCE_NOT_IDENTITY_PROOF",
            "pmxt_discovery",
            "PMXT relation confidence is diagnostic discovery metadata and cannot establish or reject semantic identity.",
        ),
        (
            "NO_BOOKS_ECONOMICS_OR_PROFIT_CLAIM",
            "terminal",
            "No native books, calculations, alerts, orders, executable edge, or profitability conclusion exist for this cohort.",
        ),
    ]
    caveats = [
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "pmxt_rejection_taxonomy_caveat",
            "taxonomy_run_id": run_id,
            "caveat_id": caveat_id,
            "scope": scope,
            "detail": detail,
            "live_eligible": False,
        }
        for caveat_id, scope, detail in caveat_values
    ]
    return {
        "taxonomy_rows": b"".join(_canonical_json(row) for row in output_rows),
        "taxonomy_summary": _canonical_json(summary),
        "source_bindings": _canonical_json(source_bindings),
        "unresolved_caveats": b"".join(_canonical_json(row) for row in caveats),
    }


def _ensure_real_directories(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    status = _lstat_or_none(current)
    if status is None or _is_reparse_status(status) or not stat.S_ISDIR(status.st_mode):
        raise RejectionTaxonomyError("output anchor must be a real directory")
    for part in path.parts[1:]:
        current /= part
        status = _lstat_or_none(current)
        if status is None:
            try:
                os.mkdir(current)
            except FileExistsError:
                pass
            status = _lstat_or_none(current)
        if status is None or _is_reparse_status(status) or not stat.S_ISDIR(status.st_mode):
            raise RejectionTaxonomyError("output path must not traverse a symlink or reparse point")
    return path


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
        raise RejectionTaxonomyError(f"{label} identity changed")


def _open_exclusive(path: Path) -> BinaryIO:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    return os.fdopen(os.open(path, flags, 0o600), "wb", buffering=0)


def _write_exclusive(path: Path, payload: bytes) -> Mapping[str, Any]:
    parent_status = _lstat_or_none(path.parent)
    if parent_status is None or _is_reparse_status(parent_status) or not stat.S_ISDIR(parent_status.st_mode):
        raise RejectionTaxonomyError("artifact parent is not a real directory")
    digest = hashlib.sha256()
    byte_size = 0
    with _open_exclusive(path) as handle:
        view = memoryview(payload)
        while view:
            written = handle.write(view)
            if written is None or written <= 0:
                raise OSError("short write while sealing rejection taxonomy")
            digest.update(view[:written])
            byte_size += written
            view = view[written:]
        handle.flush()
        os.fsync(handle.fileno())
        opened_status = os.fstat(handle.fileno())
    _assert_directory_identity(path.parent, parent_status, label="artifact parent")
    path_status = _lstat_or_none(path)
    if path_status is None or _is_reparse_status(path_status) or not _same_file(opened_status, path_status):
        raise RejectionTaxonomyError("sealed artifact identity changed")
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
        raise RejectionTaxonomyError("final rejection-taxonomy output already exists")
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise RejectionTaxonomyError("final rejection-taxonomy output already exists") from exc
        except OSError as exc:
            raise RejectionTaxonomyError("exclusive rejection-taxonomy finalization failed") from exc
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise RejectionTaxonomyError("atomic no-replace finalization is unavailable")
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin":
        function = getattr(library, "renamex_np", None)
        if function is None:
            raise RejectionTaxonomyError("atomic no-replace finalization is unavailable")
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(os.fsencode(source), os.fsencode(destination), 0x00000004)
    else:
        raise RejectionTaxonomyError("atomic no-replace finalization is unavailable")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise RejectionTaxonomyError("final rejection-taxonomy output already exists")
    raise RejectionTaxonomyError(f"exclusive rejection-taxonomy finalization failed: errno {error_number}")


def _revalidate_sources(runs: Mapping[str, _SourceRun], repository_root: Path) -> None:
    for role, source in runs.items():
        snapshots = [source.manifest_snapshot, *source.artifacts.values()]
        if source.sidecar_snapshot is not None:
            snapshots.append(source.sidecar_snapshot)
        for snapshot in snapshots:
            current = _read_input(snapshot.path, root=repository_root, label=f"{role} source revalidation")
            if current.binding != snapshot.binding:
                raise RejectionTaxonomyError(f"{role} source changed after validation")


def derive_rejection_taxonomy(*, repository_root: Path, output_root: Path, run_id: str) -> Path:
    """Validate the exact sealed cohort and publish one immutable taxonomy run."""

    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise RejectionTaxonomyError("output run ID is invalid")
    repository_candidate = Path(os.path.abspath(repository_root))
    repository_status = _reject_absolute_reparse_chain(repository_candidate, label="repository root")
    if not stat.S_ISDIR(repository_status.st_mode):
        raise RejectionTaxonomyError("repository root must be a real directory")
    repository = repository_candidate.resolve(strict=True)
    output = Path(os.path.abspath(output_root))
    final_path = output / "runs" / run_id
    staging_path = final_path.with_name(f"{run_id}.inprogress")
    if _lstat_or_none(final_path) is not None or _lstat_or_none(staging_path) is not None:
        raise RejectionTaxonomyError("final or in-progress rejection-taxonomy output already exists")

    runs = {role: _load_source_run(repository, role) for role in _AUTHORITATIVE_MANIFESTS}
    _validate_source_manifests(runs)
    contents = dict(_derive_documents(runs, run_id=run_id))
    filenames = {
        "taxonomy_rows": "taxonomy_rows.jsonl",
        "taxonomy_summary": "taxonomy_summary.json",
        "source_bindings": "source_bindings.json",
        "unresolved_caveats": "unresolved_caveats.jsonl",
    }
    artifact_bindings = {
        name: {"path": filenames[name], "byte_size": len(payload), "sha256": _sha256(payload)}
        for name, payload in contents.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "status": "OFFLINE_REJECTION_TAXONOMY_COMPLETE_ALL_CANDIDATES_REJECTED",
        "counts": {"total": 25, "rejected": 25, "verified_equivalent": 0, "needs_review": 0},
        "artifacts": artifact_bindings,
        "authority": {"offline_only": True, **_OUTPUT_SAFETY},
    }
    manifest_payload = _canonical_json(manifest)
    sidecar_payload = f"{_sha256(manifest_payload)}  manifest.json\n".encode("ascii")

    output_parent = _ensure_real_directories(output / "runs")
    parent_status = _lstat_or_none(output_parent)
    if parent_status is None:
        raise RejectionTaxonomyError("rejection-taxonomy output parent is missing")
    if _lstat_or_none(final_path) is not None or _lstat_or_none(staging_path) is not None:
        raise RejectionTaxonomyError("final or in-progress rejection-taxonomy output already exists")
    os.mkdir(staging_path, 0o700)
    staging_status = _lstat_or_none(staging_path)
    if staging_status is None or _is_reparse_status(staging_status) or not stat.S_ISDIR(staging_status.st_mode):
        raise RejectionTaxonomyError("in-progress rejection-taxonomy output is not a real directory")
    try:
        written: dict[str, Mapping[str, Any]] = {}
        for name in ("taxonomy_rows", "taxonomy_summary", "source_bindings", "unresolved_caveats"):
            written[name] = _write_exclusive(staging_path / filenames[name], contents[name])
            if written[name] != artifact_bindings[name]:
                raise RejectionTaxonomyError(f"sealed artifact {name} differs from its precomputed binding")
        written["manifest"] = _write_exclusive(staging_path / "manifest.json", manifest_payload)
        written["manifest_sidecar"] = _write_exclusive(staging_path / "manifest.sha256", sidecar_payload)
        _sync_directory(staging_path)
        _revalidate_sources(runs, repository)
        for name, binding in written.items():
            filename = "manifest.sha256" if name == "manifest_sidecar" else f"{name}.json"
            if name in filenames:
                filename = filenames[name]
            elif name == "manifest":
                filename = "manifest.json"
            snapshot = _read_input(staging_path / filename, root=staging_path, label=f"sealed {name}")
            if snapshot.binding["byte_size"] != binding["byte_size"] or snapshot.binding["sha256"] != binding["sha256"]:
                raise RejectionTaxonomyError(f"sealed artifact {name} changed before finalization")
        _assert_directory_identity(output_parent, parent_status, label="rejection-taxonomy output parent")
        _assert_directory_identity(staging_path, staging_status, label="in-progress rejection-taxonomy output")
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
                            "record_type": "pmxt_rejection_taxonomy_failure",
                            "run_id": run_id,
                            "status": "FAILED_CLOSED_STAGING_RETAINED",
                            "retry_permitted": False,
                            **_OUTPUT_SAFETY,
                        }
                    ),
                )
                _sync_directory(staging_path)
            except (OSError, RejectionTaxonomyError):
                pass
        raise
    return final_path


__all__ = ["PROTOCOL_ID", "RejectionTaxonomyError", "derive_rejection_taxonomy"]
