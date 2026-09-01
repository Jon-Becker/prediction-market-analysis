"""Immutable persistence for one complete PMXT monitor run."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_JSONL_FILENAMES = {
    "candidates": "candidates.jsonl",
    "raw_native_metadata": "raw_native_metadata.jsonl",
    "semantic_decisions": "semantic_decisions.jsonl",
    "rejections": "rejections.jsonl",
    "native_books": "native_books.jsonl",
    "calculations": "calculations.jsonl",
    "alerts": "alerts.jsonl",
}


@dataclass(frozen=True)
class MonitorRunArtifacts:
    """Paths and exact row counts for a persisted monitor run."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    artifact_paths: Mapping[str, Path]
    counts: Mapping[str, int]


@dataclass(frozen=True)
class VerifiedMonitorSource:
    """Hash-verified, bounded inputs loaded from one immutable monitor run."""

    source_run_id: str
    source_run_dir: Path
    manifest: Mapping[str, Any]
    raw_pmxt: Mapping[str, Any]
    raw_payload: Any
    candidates: tuple[Mapping[str, Any], ...]
    artifact_sha256: Mapping[str, str]
    provenance: Mapping[str, Any]


class SourceRunValidationError(ValueError):
    """Raised when an immutable source run fails structural or hash validation."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_run_id(run_id: str) -> None:
    """Reject path-like identifiers before creating the runs directory."""

    if not isinstance(run_id, str) or not run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty single path component")
    if Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be a non-empty single path component")


def _json_bytes(value: Any) -> bytes:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return content.encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> tuple[bytes, int]:
    encoded_rows: list[bytes] = []
    count = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("JSONL evidence rows must be mappings")
        encoded_rows.append(
            (json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        )
        count += 1
    return b"".join(encoded_rows), count


def _reject_json_constant(value: str) -> None:
    raise SourceRunValidationError(f"non-finite JSON constant is not allowed: {value}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceRunValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_json(content: bytes, *, label: str) -> Any:
    try:
        text = content.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except SourceRunValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceRunValidationError(f"invalid strict JSON in {label}: {exc}") from exc


def _strict_jsonl(content: bytes, *, label: str) -> list[Mapping[str, Any]]:
    if not content:
        return []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceRunValidationError(f"invalid UTF-8 in {label}: {exc}") from exc

    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise SourceRunValidationError(f"blank JSONL row in {label} at line {line_number}")
        row = _strict_json(line.encode("utf-8"), label=f"{label}:{line_number}")
        if not isinstance(row, Mapping):
            raise SourceRunValidationError(f"JSONL row must be an object in {label} at line {line_number}")
        rows.append(row)
    return rows


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_child(run_dir: Path, relative_name: Any, *, label: str) -> Path:
    if not isinstance(relative_name, str) or not relative_name or Path(relative_name).name != relative_name:
        raise SourceRunValidationError(f"{label} path must be one file name")
    if "/" in relative_name or "\\" in relative_name:
        raise SourceRunValidationError(f"{label} path must be one file name")

    path = run_dir / relative_name
    if path.is_symlink():
        raise SourceRunValidationError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise SourceRunValidationError(f"{label} is missing or inaccessible") from exc
    if resolved.parent != run_dir or not resolved.is_file():
        raise SourceRunValidationError(f"{label} is not a contained regular file")
    return resolved


def load_verified_monitor_source(output_dir: Path | str, source_run_id: str) -> VerifiedMonitorSource:
    """Load one prior run only after fail-closed integrity and scope checks.

    This function never contacts PMXT or a venue. It accepts only an immutable,
    identity-only source run of at most 25 candidates and verifies every
    manifest-listed artifact before returning any discovery input for reuse.
    """

    _validate_run_id(source_run_id)
    root_input = Path(output_dir)
    if root_input.is_symlink():
        raise SourceRunValidationError("output_dir must not be a symlink")
    try:
        root = root_input.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise SourceRunValidationError("output_dir is missing or inaccessible") from exc
    if not root.is_dir():
        raise SourceRunValidationError("output_dir must be a directory")

    runs_dir_input = root / "runs"
    if runs_dir_input.is_symlink():
        raise SourceRunValidationError("runs directory must not be a symlink")
    try:
        runs_dir = runs_dir_input.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise SourceRunValidationError("runs directory is missing or inaccessible") from exc
    if runs_dir.parent != root or not runs_dir.is_dir():
        raise SourceRunValidationError("runs directory is not contained by output_dir")

    run_dir_input = runs_dir / source_run_id
    if run_dir_input.is_symlink():
        raise SourceRunValidationError("source run directory must not be a symlink")
    try:
        run_dir = run_dir_input.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise SourceRunValidationError("source run directory is missing or inaccessible") from exc
    if run_dir.parent != runs_dir or not run_dir.is_dir():
        raise SourceRunValidationError("source run directory is not contained by runs directory")

    manifest_path = _source_child(run_dir, "manifest.json", label="manifest")
    manifest_content = manifest_path.read_bytes()
    manifest = _strict_json(manifest_content, label="manifest.json")
    if not isinstance(manifest, Mapping):
        raise SourceRunValidationError("manifest must be a JSON object")
    manifest_schema_version = manifest.get("schema_version")
    if (
        isinstance(manifest_schema_version, bool)
        or manifest_schema_version != 1
        or manifest.get("run_id") != source_run_id
    ):
        raise SourceRunValidationError("manifest schema or run_id does not match source run")
    if manifest.get("live_eligible") is not False or manifest.get("no_order_actions") is not True:
        raise SourceRunValidationError("source manifest does not preserve the read-only safety boundary")

    manifest_artifacts = manifest.get("artifacts")
    manifest_counts = manifest.get("counts")
    if not isinstance(manifest_artifacts, Mapping) or not isinstance(manifest_counts, Mapping):
        raise SourceRunValidationError("manifest artifacts and counts must be JSON objects")
    if "raw_pmxt" not in manifest_artifacts or "candidates" not in manifest_artifacts:
        raise SourceRunValidationError("source manifest must list raw_pmxt and candidates artifacts")

    parsed_artifacts: dict[str, Any] = {}
    artifact_sha256: dict[str, str] = {}
    artifact_paths: set[Path] = set()
    for artifact_name, metadata in manifest_artifacts.items():
        if not isinstance(artifact_name, str) or not artifact_name or not isinstance(metadata, Mapping):
            raise SourceRunValidationError("each manifest artifact must have a named metadata object")
        expected_size = metadata.get("byte_size")
        expected_sha256 = metadata.get("sha256")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise SourceRunValidationError(f"invalid size or SHA256 metadata for artifact {artifact_name}")
        artifact_path = _source_child(run_dir, metadata.get("path"), label=f"artifact {artifact_name}")
        if artifact_path in artifact_paths:
            raise SourceRunValidationError("multiple manifest artifacts reference the same file")
        artifact_paths.add(artifact_path)
        content = artifact_path.read_bytes()
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if len(content) != expected_size or actual_sha256 != expected_sha256:
            raise SourceRunValidationError(f"size or SHA256 mismatch for artifact {artifact_name}")

        if artifact_path.suffix == ".json":
            parsed = _strict_json(content, label=artifact_path.name)
            row_count = 1
        elif artifact_path.suffix == ".jsonl":
            parsed = _strict_jsonl(content, label=artifact_path.name)
            row_count = len(parsed)
        else:
            raise SourceRunValidationError(f"unsupported artifact format for {artifact_name}")
        declared_count = manifest_counts.get(artifact_name)
        if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count != row_count:
            raise SourceRunValidationError(f"manifest count mismatch for artifact {artifact_name}")
        parsed_artifacts[artifact_name] = parsed
        artifact_sha256[artifact_name] = actual_sha256

    raw_pmxt = parsed_artifacts["raw_pmxt"]
    candidates = parsed_artifacts["candidates"]
    if not isinstance(raw_pmxt, Mapping) or not isinstance(candidates, list):
        raise SourceRunValidationError("raw_pmxt and candidates artifacts have invalid structures")
    raw_schema_version = raw_pmxt.get("schema_version")
    if (
        isinstance(raw_schema_version, bool)
        or raw_schema_version != 1
        or raw_pmxt.get("source") != "pmxt_router"
        or raw_pmxt.get("run_id") != source_run_id
        or raw_pmxt.get("live_eligible") is not False
    ):
        raise SourceRunValidationError("raw PMXT document is not bound to the safe source run")
    if "payload" not in raw_pmxt:
        raise SourceRunValidationError("raw PMXT document is missing its payload")

    query = raw_pmxt.get("query")
    config = manifest.get("config")
    discovery = config.get("discovery") if isinstance(config, Mapping) else None
    if not isinstance(query, Mapping) or query != discovery:
        raise SourceRunValidationError("raw PMXT query does not match the source manifest")
    limit = query.get("limit")
    min_confidence = query.get("min_confidence")
    min_venues = query.get("min_venues")
    offset = query.get("offset")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 25
        or isinstance(min_confidence, bool)
        or not isinstance(min_confidence, (int, float))
        or not 0.8 <= min_confidence <= 1.0
        or isinstance(min_venues, bool)
        or min_venues != 2
        or isinstance(offset, bool)
        or offset != 0
        or query.get("include_raw_matches") is not True
        or query.get("relation") != "identity"
        or query.get("venues") != ["kalshi", "polymarket"]
    ):
        raise SourceRunValidationError("source query is not bounded identity-only Kalshi/Polymarket discovery")

    raw_payload = raw_pmxt["payload"]
    payload_sha256 = raw_pmxt.get("payload_sha256")
    if (
        not isinstance(payload_sha256, str)
        or _SHA256_RE.fullmatch(payload_sha256) is None
        or _canonical_json_sha256(raw_payload) != payload_sha256
    ):
        raise SourceRunValidationError("raw PMXT payload hash does not match its document")

    if len(candidates) > 25 or len(candidates) > limit:
        raise SourceRunValidationError("source run contains more candidates than its bounded query allows")
    candidate_ids: set[str] = set()
    verified_candidates: list[Mapping[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise SourceRunValidationError("candidate rows must be JSON objects")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidate_ids:
            raise SourceRunValidationError("candidate IDs must be non-empty and unique")
        candidate_ids.add(candidate_id)
        candidate_schema_version = candidate.get("schema_version")
        relation_confidence = candidate.get("relation_confidence")
        if (
            isinstance(candidate_schema_version, bool)
            or candidate_schema_version != 1
            or candidate.get("snapshot_id") != source_run_id
            or candidate.get("source") != "pmxt_router"
            or candidate.get("relation") != "identity"
            or isinstance(relation_confidence, bool)
            or not isinstance(relation_confidence, (int, float))
            or not min_confidence <= relation_confidence <= 1.0
            or candidate.get("raw_edge_present") is not True
            or candidate.get("live_eligible") is not False
            or {candidate.get("venue_a"), candidate.get("venue_b")} != {"kalshi", "polymarket"}
        ):
            raise SourceRunValidationError("candidate is not an identity-only row bound to the safe source run")
        verified_candidates.append(candidate)

    if manifest_counts.get("candidates") != len(verified_candidates):
        raise SourceRunValidationError("source manifest candidate count does not match candidate rows")

    manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
    provenance = {
        "source_run_id": source_run_id,
        "source_manifest_sha256": manifest_sha256,
        "source_artifact_sha256": dict(sorted(artifact_sha256.items())),
        "source_raw_payload_sha256": payload_sha256,
    }
    return VerifiedMonitorSource(
        source_run_id=source_run_id,
        source_run_dir=run_dir,
        manifest=manifest,
        raw_pmxt=raw_pmxt,
        raw_payload=raw_payload,
        candidates=tuple(verified_candidates),
        artifact_sha256=artifact_sha256,
        provenance=provenance,
    )


def _write_new(path: Path, content: bytes) -> None:
    """Write one file exclusively inside an already reserved run directory."""

    with path.open("xb") as handle:
        handle.write(content)


def _metadata(path: Path, content: bytes) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_size": len(content),
    }


def _remove_reserved_run(run_dir: Path, runs_dir: Path) -> None:
    """Remove only the exact run directory reserved by this invocation."""

    if run_dir.parent != runs_dir or run_dir == runs_dir:
        raise RuntimeError("refusing to clean an uncontained monitor run path")
    if run_dir.exists():
        shutil.rmtree(run_dir)


def persist_monitor_run(
    output_dir: Path | str,
    *,
    run_id: str,
    status: str,
    config: Mapping[str, Any] | None = None,
    counts: Mapping[str, int] | None = None,
    raw_pmxt: Any = None,
    candidates: Iterable[Mapping[str, Any]] | None = None,
    raw_native_metadata: Iterable[Mapping[str, Any]] | None = None,
    semantic_decisions: Iterable[Mapping[str, Any]] | None = None,
    rejections: Iterable[Mapping[str, Any]] | None = None,
    native_books: Iterable[Mapping[str, Any]] | None = None,
    calculations: Iterable[Mapping[str, Any]] | None = None,
    alerts: Iterable[Mapping[str, Any]] | None = None,
    supporting_evidence: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> MonitorRunArtifacts:
    """Persist a whole monitor run without ever reopening an existing run.

    ``None`` omits an evidence class. An explicitly supplied empty iterable
    creates the corresponding empty JSONL file so a completed zero-result
    stage remains distinguishable from a stage that was not run.
    """

    _validate_run_id(run_id)
    if not isinstance(status, str) or not status.strip():
        raise ValueError("status must be a non-empty string")

    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a JSON-safe mapping")
    provenance_document = dict(provenance) if provenance is not None else None
    if provenance_document is not None:
        _json_bytes(provenance_document)
    if supporting_evidence is not None and not isinstance(supporting_evidence, Mapping):
        raise TypeError("supporting_evidence must be a JSON-safe mapping")
    supporting_document = dict(supporting_evidence) if supporting_evidence is not None else None
    if supporting_document is not None:
        _json_bytes(supporting_document)

    root = Path(output_dir)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    run_dir.mkdir(exist_ok=False)

    artifact_paths: dict[str, Path] = {}
    artifact_metadata: dict[str, dict[str, Any]] = {}
    evidence_counts: dict[str, int] = {}

    rows_by_name = {
        "candidates": candidates,
        "raw_native_metadata": raw_native_metadata,
        "semantic_decisions": semantic_decisions,
        "rejections": rejections,
        "native_books": native_books,
        "calculations": calculations,
        "alerts": alerts,
    }

    try:
        if raw_pmxt is not None:
            raw_path = run_dir / "raw_pmxt.json"
            raw_content = _json_bytes(raw_pmxt)
            _write_new(raw_path, raw_content)
            artifact_paths["raw_pmxt"] = raw_path
            artifact_metadata["raw_pmxt"] = _metadata(raw_path, raw_content)
            evidence_counts["raw_pmxt"] = 1

        if supporting_document is not None:
            supporting_path = run_dir / "supporting_evidence.json"
            supporting_content = _json_bytes(supporting_document)
            _write_new(supporting_path, supporting_content)
            artifact_paths["supporting_evidence"] = supporting_path
            artifact_metadata["supporting_evidence"] = _metadata(supporting_path, supporting_content)
            evidence_counts["supporting_evidence"] = 1

        for name, rows in rows_by_name.items():
            if rows is None:
                continue
            path = run_dir / _JSONL_FILENAMES[name]
            content, row_count = _jsonl_bytes(rows)
            _write_new(path, content)
            artifact_paths[name] = path
            artifact_metadata[name] = _metadata(path, content)
            evidence_counts[name] = row_count

        manifest_counts = dict(counts or {})
        manifest_counts.update(evidence_counts)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": status,
            "config": dict(config or {}),
            "counts": manifest_counts,
            "live_eligible": False,
            "no_order_actions": True,
            "artifacts": artifact_metadata,
        }
        if provenance_document is not None:
            manifest["provenance"] = provenance_document
        manifest_path = run_dir / "manifest.json"
        _write_new(manifest_path, _json_bytes(manifest))
    except BaseException:
        _remove_reserved_run(run_dir, runs_dir)
        raise

    return MonitorRunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        artifact_paths=artifact_paths,
        counts=evidence_counts,
    )
