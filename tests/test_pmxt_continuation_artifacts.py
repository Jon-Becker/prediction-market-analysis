"""Tests for hash-verified reuse of an immutable PMXT monitor source run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.indexers.pmxt.artifacts import (
    SourceRunValidationError,
    load_verified_monitor_source,
    persist_monitor_run,
)


def _canonical_sha256(value: Any) -> str:
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _query(*, limit: int = 25) -> dict[str, Any]:
    return {
        "include_raw_matches": True,
        "limit": limit,
        "min_confidence": 0.8,
        "min_venues": 2,
        "offset": 0,
        "relation": "identity",
        "sort": "volume",
        "venues": ["kalshi", "polymarket"],
    }


def _candidate(run_id: str, number: int = 1, *, relation: str = "identity") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate_id": f"candidate-{number}",
        "snapshot_id": run_id,
        "source": "pmxt_router",
        "relation": relation,
        "relation_confidence": 0.95,
        "raw_edge_present": True,
        "venue_a": "kalshi",
        "venue_b": "polymarket",
        "live_eligible": False,
    }


def _persist_source(
    output_dir: Path,
    *,
    run_id: str = "source-run",
    candidates: list[dict[str, Any]] | None = None,
    query: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_query = query or _query()
    source_payload = {"data": [{"clusterId": "cluster-1"}], "pagination": {"limit": 25, "offset": 0}}
    raw_pmxt = {
        "schema_version": 1,
        "source": "pmxt_router",
        "run_id": run_id,
        "query": source_query,
        "payload_sha256": _canonical_sha256(source_payload),
        "payload": source_payload,
        "live_eligible": False,
    }
    source_candidates = candidates if candidates is not None else [_candidate(run_id)]
    persist_monitor_run(
        output_dir,
        run_id=run_id,
        status="NO_VERIFIED_CANDIDATES",
        config={"discovery": source_query, "identity_only": True},
        raw_pmxt=raw_pmxt,
        candidates=source_candidates,
    )
    return raw_pmxt, source_candidates


def _rewrite_manifest_artifact_metadata(run_dir: Path, artifact_name: str) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = run_dir / manifest["artifacts"][artifact_name]["path"]
    content = artifact_path.read_bytes()
    manifest["artifacts"][artifact_name]["byte_size"] = len(content)
    manifest["artifacts"][artifact_name]["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_load_verified_monitor_source_returns_exact_input_and_provenance(tmp_path: Path) -> None:
    raw_pmxt, candidates = _persist_source(tmp_path)

    result = load_verified_monitor_source(tmp_path, "source-run")

    assert result.raw_pmxt == raw_pmxt
    assert result.raw_payload is result.raw_pmxt["payload"]
    assert result.raw_payload == raw_pmxt["payload"]
    assert result.candidates == tuple(candidates)
    assert result.source_run_dir == (tmp_path / "runs" / "source-run").resolve()
    assert result.provenance == {
        "source_run_id": "source-run",
        "source_manifest_sha256": hashlib.sha256(
            (tmp_path / "runs" / "source-run" / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_artifact_sha256": dict(sorted(result.artifact_sha256.items())),
        "source_raw_payload_sha256": raw_pmxt["payload_sha256"],
    }


@pytest.mark.parametrize("source_run_id", ["", ".", "..", "../source-run", "..\\source-run", "a/b", "a\\b"])
def test_load_verified_monitor_source_rejects_path_like_run_ids(tmp_path: Path, source_run_id: str) -> None:
    tmp_path.mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="single path component"):
        load_verified_monitor_source(tmp_path, source_run_id)


def test_load_verified_monitor_source_rejects_symlinked_run_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "runs").mkdir()
    external_dir = tmp_path / "external"
    _persist_source(external_dir)
    source_target = external_dir / "runs" / "source-run"
    source_link = output_dir / "runs" / "source-run"
    try:
        source_link.symlink_to(source_target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    assert os.path.islink(source_link)
    with pytest.raises(SourceRunValidationError, match="must not be a symlink"):
        load_verified_monitor_source(output_dir, "source-run")


def test_load_verified_monitor_source_rejects_manifest_artifact_traversal(tmp_path: Path) -> None:
    _persist_source(tmp_path)
    run_dir = tmp_path / "runs" / "source-run"
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["candidates"]["path"] = "../candidates.jsonl"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(SourceRunValidationError, match="one file name"):
        load_verified_monitor_source(tmp_path, "source-run")


def test_load_verified_monitor_source_rejects_symlinked_artifact(tmp_path: Path) -> None:
    _persist_source(tmp_path)
    run_dir = tmp_path / "runs" / "source-run"
    candidate_path = run_dir / "candidates.jsonl"
    original_path = run_dir / "candidates-original.jsonl"
    candidate_path.replace(original_path)
    try:
        candidate_path.symlink_to(original_path)
    except OSError as exc:
        original_path.replace(candidate_path)
        pytest.skip(f"file symlinks unavailable: {exc}")

    assert os.path.islink(candidate_path)
    with pytest.raises(SourceRunValidationError, match="must not be a symlink"):
        load_verified_monitor_source(tmp_path, "source-run")


def test_load_verified_monitor_source_rejects_artifact_tamper(tmp_path: Path) -> None:
    _persist_source(tmp_path)
    candidate_path = tmp_path / "runs" / "source-run" / "candidates.jsonl"
    candidate_path.write_bytes(candidate_path.read_bytes() + b" ")

    with pytest.raises(SourceRunValidationError, match="size or SHA256 mismatch"):
        load_verified_monitor_source(tmp_path, "source-run")


def test_load_verified_monitor_source_rejects_strict_jsonl_duplicate_keys(tmp_path: Path) -> None:
    _persist_source(tmp_path)
    run_dir = tmp_path / "runs" / "source-run"
    candidate_path = run_dir / "candidates.jsonl"
    candidate_path.write_text(
        '{"candidate_id":"candidate-1","candidate_id":"candidate-2"}\n',
        encoding="utf-8",
    )
    _rewrite_manifest_artifact_metadata(run_dir, "candidates")

    with pytest.raises(SourceRunValidationError, match="duplicate JSON object key"):
        load_verified_monitor_source(tmp_path, "source-run")


def test_load_verified_monitor_source_rejects_raw_payload_hash_mismatch(tmp_path: Path) -> None:
    _persist_source(tmp_path)
    run_dir = tmp_path / "runs" / "source-run"
    raw_path = run_dir / "raw_pmxt.json"
    raw_pmxt = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_pmxt["payload"]["data"].append({"clusterId": "unhashed-cluster"})
    raw_path.write_text(
        json.dumps(raw_pmxt, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest_artifact_metadata(run_dir, "raw_pmxt")

    with pytest.raises(SourceRunValidationError, match="payload hash"):
        load_verified_monitor_source(tmp_path, "source-run")


def test_load_verified_monitor_source_rejects_duplicate_candidate_ids(tmp_path: Path) -> None:
    duplicate_candidates = [_candidate("source-run", 1), _candidate("source-run", 1)]
    _persist_source(tmp_path, candidates=duplicate_candidates)

    with pytest.raises(SourceRunValidationError, match="unique"):
        load_verified_monitor_source(tmp_path, "source-run")


def test_load_verified_monitor_source_rejects_non_identity_candidate(tmp_path: Path) -> None:
    _persist_source(tmp_path, candidates=[_candidate("source-run", relation="overlap")])

    with pytest.raises(SourceRunValidationError, match="identity-only"):
        load_verified_monitor_source(tmp_path, "source-run")


def test_load_verified_monitor_source_rejects_more_than_twenty_five_candidates(tmp_path: Path) -> None:
    candidates = [_candidate("source-run", number) for number in range(26)]
    _persist_source(tmp_path, candidates=candidates)

    with pytest.raises(SourceRunValidationError, match="more candidates"):
        load_verified_monitor_source(tmp_path, "source-run")


def test_persist_monitor_run_adds_optional_top_level_provenance(tmp_path: Path) -> None:
    provenance = {
        "source_run_id": "source-run",
        "source_manifest_sha256": "a" * 64,
        "source_artifact_sha256": {"raw_pmxt": "b" * 64},
    }

    result = persist_monitor_run(
        tmp_path,
        run_id="continuation-run",
        status="NO_VERIFIED_CANDIDATES",
        provenance=provenance,
        candidates=[],
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["provenance"] == provenance


def test_persist_monitor_run_rejects_non_json_provenance_before_reserving_run(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        persist_monitor_run(
            tmp_path,
            run_id="continuation-run",
            status="NO_VERIFIED_CANDIDATES",
            provenance={"not_json": object()},
        )

    assert not (tmp_path / "runs").exists()
