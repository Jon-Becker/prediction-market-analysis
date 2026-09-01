"""Tests for immutable whole-run PMXT monitor artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

import src.indexers.pmxt.artifacts as artifact_module
from src.indexers.pmxt.artifacts import persist_monitor_run


def _byte_inventory(path: Path) -> dict[str, bytes]:
    return {item.relative_to(path).as_posix(): item.read_bytes() for item in sorted(path.rglob("*")) if item.is_file()}


def test_supporting_evidence_is_exactly_persisted_and_manifested(tmp_path: Path) -> None:
    raw_body_base64 = base64.b64encode(b"\x00fixture-fee-schedule").decode("ascii")
    supporting_evidence = {
        "schema_version": 1,
        "kalshi_official_fee_schedule": {
            "raw_body_base64": raw_body_base64,
            "raw_body_sha256": hashlib.sha256(b"\x00fixture-fee-schedule").hexdigest(),
        },
        "live_eligible": False,
    }

    result = persist_monitor_run(
        tmp_path,
        run_id="supporting-run",
        status="NO_EXECUTABLE_SHADOW_EDGE",
        supporting_evidence=supporting_evidence,
    )

    supporting_path = result.artifact_paths["supporting_evidence"]
    content = supporting_path.read_bytes()
    persisted = json.loads(content)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["artifacts"]["supporting_evidence"]
    assert supporting_path == result.run_dir / "supporting_evidence.json"
    assert persisted == supporting_evidence
    assert persisted["kalshi_official_fee_schedule"]["raw_body_base64"] == raw_body_base64
    assert result.counts["supporting_evidence"] == 1
    assert manifest["counts"]["supporting_evidence"] == 1
    assert metadata == {
        "path": "supporting_evidence.json",
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_size": len(content),
    }


def test_persist_monitor_run_writes_content_counts_and_exact_digests(tmp_path: Path) -> None:
    raw_pmxt = {"clusters": [{"clusterId": "cluster-1", "raw": {"title": "Café"}}]}
    rows = {
        "candidates": [{"candidate_id": "candidate-1", "live_eligible": False}],
        "raw_native_metadata": [
            {
                "candidate_id": "candidate-1",
                "raw_response": {"market_id": "KX-1", "rules": "Native rules"},
            }
        ],
        "semantic_decisions": [{"candidate_id": "candidate-1", "classification": "VERIFIED_EQUIVALENT"}],
        "rejections": [],
        "native_books": [{"candidate_id": "candidate-1", "venue": "kalshi", "bids": [[52, 4]]}],
        "calculations": [{"candidate_id": "candidate-1", "net_residual": 0.04}],
        "alerts": [],
    }

    result = persist_monitor_run(
        tmp_path,
        run_id="run-001",
        status="NO_EXECUTABLE_SHADOW_EDGE",
        config={"threshold": 0.05, "venues": ["kalshi", "polymarket"]},
        counts={"pmxt_clusters": 1},
        raw_pmxt=raw_pmxt,
        **rows,
    )

    expected_filenames = {
        "raw_pmxt.json",
        "candidates.jsonl",
        "raw_native_metadata.jsonl",
        "semantic_decisions.jsonl",
        "rejections.jsonl",
        "native_books.jsonl",
        "calculations.jsonl",
        "alerts.jsonl",
        "manifest.json",
    }
    assert {path.name for path in result.run_dir.iterdir()} == expected_filenames
    assert result.run_dir == tmp_path / "runs" / "run-001"
    assert json.loads(result.artifact_paths["raw_pmxt"].read_text(encoding="utf-8")) == raw_pmxt
    assert (
        json.loads(result.artifact_paths["raw_native_metadata"].read_text(encoding="utf-8"))
        == rows["raw_native_metadata"][0]
    )
    assert result.artifact_paths["rejections"].read_bytes() == b""
    assert result.counts == {
        "raw_pmxt": 1,
        "candidates": 1,
        "raw_native_metadata": 1,
        "semantic_decisions": 1,
        "rejections": 0,
        "native_books": 1,
        "calculations": 1,
        "alerts": 0,
    }

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "NO_EXECUTABLE_SHADOW_EDGE"
    assert manifest["config"] == {"threshold": 0.05, "venues": ["kalshi", "polymarket"]}
    assert manifest["counts"]["pmxt_clusters"] == 1
    assert manifest["counts"]["rejections"] == 0
    assert manifest["live_eligible"] is False
    assert manifest["no_order_actions"] is True
    assert "manifest" not in manifest["artifacts"]

    for name, path in result.artifact_paths.items():
        content = path.read_bytes()
        metadata = manifest["artifacts"][name]
        assert metadata["path"] == path.name
        assert metadata["byte_size"] == len(content)
        assert metadata["sha256"] == hashlib.sha256(content).hexdigest()


def test_existing_run_is_never_touched(tmp_path: Path) -> None:
    original_supporting_evidence = {
        "schema_version": 1,
        "raw_body_base64": base64.b64encode(b"original").decode("ascii"),
        "live_eligible": False,
    }
    first = persist_monitor_run(
        tmp_path,
        run_id="fixed-run",
        status="NO_VERIFIED_CANDIDATES",
        raw_pmxt={"clusters": []},
        candidates=[],
        rejections=[{"reason": "semantic_mismatch"}],
        supporting_evidence=original_supporting_evidence,
    )
    before = _byte_inventory(first.run_dir)

    with pytest.raises(FileExistsError):
        persist_monitor_run(
            tmp_path,
            run_id="fixed-run",
            status="COMPLETED",
            raw_pmxt={"clusters": [{"clusterId": "must-not-appear"}]},
            candidates=[{"candidate_id": "must-not-appear"}],
            supporting_evidence={"raw_body_base64": base64.b64encode(b"replacement").decode("ascii")},
        )

    assert _byte_inventory(first.run_dir) == before
    assert json.loads(first.artifact_paths["supporting_evidence"].read_text(encoding="utf-8")) == (
        original_supporting_evidence
    )


def test_later_serialization_failure_removes_only_newly_reserved_run(tmp_path: Path) -> None:
    sibling = tmp_path / "runs" / "preserved-run"
    sibling.mkdir(parents=True)
    marker = sibling / "marker.json"
    marker.write_bytes(b'{"preserve":true}\n')

    with pytest.raises(TypeError):
        persist_monitor_run(
            tmp_path,
            run_id="failed-run",
            status="COMPLETED",
            raw_pmxt={"clusters": []},
            candidates=[{"candidate_id": "written-before-failure"}],
            calculations=[{"not_json_serializable": object()}],
        )

    assert not (tmp_path / "runs" / "failed-run").exists()
    assert marker.read_bytes() == b'{"preserve":true}\n'


def test_later_write_failure_removes_only_newly_reserved_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sibling = tmp_path / "runs" / "preserved-run"
    sibling.mkdir(parents=True)
    marker = sibling / "marker.json"
    marker.write_bytes(b'{"preserve":true}\n')
    real_write_new = artifact_module._write_new

    def fail_on_calculations(path: Path, content: bytes) -> None:
        if path.name == "calculations.jsonl":
            raise OSError("simulated later write failure")
        real_write_new(path, content)

    monkeypatch.setattr(artifact_module, "_write_new", fail_on_calculations)

    with pytest.raises(OSError, match="simulated later write failure"):
        persist_monitor_run(
            tmp_path,
            run_id="failed-run",
            status="COMPLETED",
            raw_pmxt={"clusters": []},
            candidates=[{"candidate_id": "written-before-failure"}],
            calculations=[{"net_residual": 0.0}],
        )

    assert not (tmp_path / "runs" / "failed-run").exists()
    assert marker.read_bytes() == b'{"preserve":true}\n'
