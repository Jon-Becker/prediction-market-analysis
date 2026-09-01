from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from src.indexers.pmxt import rule_review as rule_review_module
from src.indexers.pmxt.rule_review import RuleReviewError, seal_rule_review_protocol

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


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _binding(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "byte_size": len(payload), "sha256": _sha(payload)}


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _make_repository(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "repo"
    root.mkdir()
    implementation_path = root / "src/indexers/pmxt/rule_review.py"
    _write(implementation_path, Path(rule_review_module.__file__).read_bytes())
    prior_ids = [f"pmxt_candidate_{index:020x}" for index in range(1, 26)]
    review_ids = prior_ids[11:]

    adjudication_id = "fixture_adjudication_v1"
    adjudication_dir = root / "data/pmxt/semantic_adjudication/runs" / adjudication_id
    adjudication_dir.mkdir(parents=True)
    prior_rows = []
    for index, candidate_id in enumerate(prior_ids):
        decision = "REJECTED" if index < 11 else "NEEDS_REVIEW"
        prior_rows.append(
            {
                "schema_version": 1,
                "record_type": "pmxt_offline_adjudication",
                "candidate_id": candidate_id,
                "decision": decision,
                "reviewed": True,
                "semantic_verified": False,
                "profitability_evaluation_eligible": False,
                "network_requests": 0,
                "orders_submitted": 0,
                "economics_computed": False,
                "live_eligible": False,
            }
        )
    decisions_path = adjudication_dir / "decisions.jsonl"
    source_path = adjudication_dir / "source_bindings.json"
    summary_path = adjudication_dir / "summary.json"
    _write(decisions_path, b"".join(_json(row) for row in prior_rows))
    _write(source_path, _json({"fixture": True}))
    _write(summary_path, _json({"fixture": True}))
    adjudication_artifacts = {
        "decisions": {**_binding(decisions_path, adjudication_dir)},
        "source_bindings": {**_binding(source_path, adjudication_dir)},
        "summary": {**_binding(summary_path, adjudication_dir)},
    }
    adjudication_manifest = {
        "schema_version": 1,
        "protocol_id": "pmxt-offline-semantic-adjudication-v1",
        "run_id": adjudication_id,
        "status": "OFFLINE_ADJUDICATION_COMPLETE_NO_VERIFIED_CANDIDATES",
        "counts": {"total": 25, "rejected": 11, "needs_review": 14, "verified_equivalent": 0},
        "artifacts": adjudication_artifacts,
        "authority": {"offline_only": True, **_OUTPUT_SAFETY},
    }
    adjudication_manifest_path = adjudication_dir / "manifest.json"
    adjudication_manifest_payload = _json(adjudication_manifest)
    _write(adjudication_manifest_path, adjudication_manifest_payload)
    _write(adjudication_dir / "manifest.sha256", f"{_sha(adjudication_manifest_payload)}  manifest.json\n".encode())

    capture_id = "fixture_rule_evidence_v1"
    capture_dir = root / "data/pmxt/rule_evidence/runs" / capture_id
    (capture_dir / "raw").mkdir(parents=True)
    resource_ids = ["kalshi-a", "kalshi-b", "kalshi-c", "poly-a", "poly-b", "poly-c", "poly-d", "poly-e"]
    members: dict[str, list[str]] = {resource_id: [] for resource_id in resource_ids}
    candidate_specs = []
    for index, candidate_id in enumerate(review_ids):
        bound_resources = [resource_ids[index % 3], resource_ids[3 + (index % 5)]]
        for resource_id in bound_resources:
            members[resource_id].append(candidate_id)
        candidate_specs.append((candidate_id, bound_resources, [f"axis_{index:02d}"]))

    resource_values = []
    raw_bindings: dict[str, dict[str, Any]] = {}
    for ordinal, resource_id in enumerate(resource_ids, 1):
        raw_path = capture_dir / "raw" / f"{ordinal:02d}_{resource_id}.bin"
        _write(raw_path, f"native evidence {resource_id}\n".encode())
        raw_binding = _binding(raw_path, capture_dir)
        raw_bindings[resource_id] = raw_binding
        resource_values.append(
            {
                "schema_version": 1,
                "record_type": "pmxt_rule_evidence_resource",
                "ordinal": ordinal,
                "resource_id": resource_id,
                "candidate_ids": members[resource_id],
                "attempt_count": 1,
                "status_code": 200,
                "redirect_count": 0,
                "content_length_agrees": True,
                "validation_status": "VALIDATED",
                "raw": raw_binding,
                **_CAPTURE_SAFETY,
            }
        )
    resource_records_path = capture_dir / "resource_records.jsonl"
    resource_payloads = [_json(value) for value in resource_values]
    _write(resource_records_path, b"".join(resource_payloads))
    resource_rows = {
        value["resource_id"]: {
            "line": index,
            "row_sha256": _sha(resource_payloads[index - 1]),
            "raw": value["raw"],
        }
        for index, value in enumerate(resource_values, 1)
    }

    candidate_values = [
        {
            "schema_version": 1,
            "record_type": "pmxt_rule_evidence_candidate_gap_index",
            "candidate_id": candidate_id,
            "decision": "NEEDS_REVIEW",
            "gap_axes": gap_axes,
            "resource_ids": bound_resources,
            "semantic_verified": False,
            "profitability_evaluation_eligible": False,
            **_CAPTURE_SAFETY,
        }
        for candidate_id, bound_resources, gap_axes in candidate_specs
    ]
    candidate_payloads = [_json(value) for value in candidate_values]
    candidate_path = capture_dir / "candidate_gap_index.jsonl"
    _write(candidate_path, b"".join(candidate_payloads))

    capture_source = {
        "schema_version": 1,
        "record_type": "pmxt_rule_evidence_source_bindings",
        "source_adjudication_run": {
            "run_id": adjudication_id,
            "manifest": _binding(adjudication_manifest_path, root),
            "artifacts": {
                "decisions": _binding(decisions_path, root),
                "source_bindings": _binding(source_path, root),
                "summary": _binding(summary_path, root),
            },
        },
        **_CAPTURE_SAFETY,
    }
    capture_source_path = capture_dir / "source_bindings.json"
    _write(capture_source_path, _json(capture_source))
    capture_artifacts = {
        "candidate_gap_index": _binding(candidate_path, capture_dir),
        "resource_records": _binding(resource_records_path, capture_dir),
        "source_bindings": _binding(capture_source_path, capture_dir),
    }
    for ordinal, resource_id in enumerate(resource_ids, 1):
        capture_artifacts[f"resource_{ordinal:02d}_raw"] = raw_bindings[resource_id]
    capture_manifest = {
        "schema_version": 1,
        "protocol_id": "pmxt-native-rule-evidence-capture-v1",
        "run_id": capture_id,
        "status": "CAPTURE_COMPLETE_REVIEW_STATUS_UNCHANGED",
        "counts": {"candidates": 14, "request_attempts": 8, "resources": 8},
        "artifacts": capture_artifacts,
        **_CAPTURE_SAFETY,
    }
    capture_manifest_path = capture_dir / "manifest.json"
    capture_manifest_payload = _json(capture_manifest)
    _write(capture_manifest_path, capture_manifest_payload)
    _write(capture_dir / "manifest.sha256", f"{_sha(capture_manifest_payload)}  manifest.json\n".encode())

    review_values = []
    for line, (candidate_id, bound_resources, gap_axes) in enumerate(candidate_specs, 1):
        evidence_resources = [
            {
                "resource_id": resource_id,
                "record_line": resource_rows[resource_id]["line"],
                "record_sha256": resource_rows[resource_id]["row_sha256"],
                "raw": resource_rows[resource_id]["raw"],
            }
            for resource_id in bound_resources
        ]
        review_values.append(
            {
                "schema_version": 1,
                "record_type": "pmxt_native_rule_review_decision",
                "candidate_id": candidate_id,
                "decision": "REJECTED",
                "reason_codes": ["FIXTURE_MATERIAL_MISMATCH"],
                "review_method": "offline_native_rule_clause_review",
                "review_version": 1,
                "reviewed_at_utc": "2026-08-30T00:20:00.000000Z",
                "source_candidate": {"line": line, "row_sha256": _sha(candidate_payloads[line - 1])},
                "evidence_resources": evidence_resources,
                "axis_findings": [
                    {
                        "axis": axis,
                        "status": "MISMATCH",
                        "rationale": "Fixture-native clauses are materially different.",
                        "evidence": [
                            {
                                "resource_id": resource_id,
                                "raw_sha256": resource_rows[resource_id]["raw"]["sha256"],
                                "locator": f"fixture locator {resource_id}",
                            }
                            for resource_id in bound_resources
                        ],
                    }
                    for axis in gap_axes
                ],
                "reviewed": True,
                "semantic_verified": False,
                "profitability_evaluation_eligible": False,
                **_OUTPUT_SAFETY,
            }
        )
    review_path = root / "results/rule_review_v1/reviewed_decisions_v1.jsonl"
    _write(review_path, b"".join(_json(value) for value in review_values))
    output_id = "fixture_rule_review_v1"
    protocol = {
        "schema_version": 1,
        "protocol_id": "pmxt-offline-native-rule-review-v1",
        "implementation": {
            "identifier": "pmxt-offline-native-rule-review-v1",
            **_binding(implementation_path, root),
        },
        "source_rule_evidence_run": {
            "run_id": capture_id,
            "manifest": _binding(capture_manifest_path, root),
        },
        "reviewed_decisions": _binding(review_path, root),
        "expected_counts": {
            "source_total": 25,
            "source_rejected": 11,
            "reviewed_total": 14,
            "newly_rejected": 14,
            "final_rejected": 25,
            "needs_review": 0,
            "verified_equivalent": 0,
        },
        "output": {
            "run_id": output_id,
            "path": f"data/pmxt/rule_review/runs/{output_id}",
        },
        "authority": {"offline_only": True, **_OUTPUT_SAFETY},
    }
    protocol_path = root / "results/rule_review_v1/protocol_v1.json"
    _write(protocol_path, _json(protocol))
    return {
        "root": root,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "capture_manifest_path": capture_manifest_path,
        "raw_path": capture_dir / raw_bindings[resource_ids[0]]["path"],
        "review_path": review_path,
        "prior_decisions_path": decisions_path,
    }


def _run(fixture: dict[str, Any]) -> Path:
    protocol_path = fixture["protocol_path"]
    return seal_rule_review_protocol(
        repository_root=fixture["root"],
        protocol_path=protocol_path.relative_to(fixture["root"]),
        expected_protocol_sha256=_sha(protocol_path.read_bytes()),
    )


def _rewrite_reviews(fixture: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    review_path = fixture["review_path"]
    _write(review_path, b"".join(_json(row) for row in rows))
    protocol = fixture["protocol"]
    protocol["reviewed_decisions"] = _binding(review_path, fixture["root"])
    _write(fixture["protocol_path"], _json(protocol))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tree_snapshot(path: Path) -> dict[str, tuple[str, bytes | None]]:
    if not path.exists():
        return {}
    snapshot: dict[str, tuple[str, bytes | None]] = {}
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path).as_posix()
        snapshot[relative] = ("directory", None) if item.is_dir() else ("file", item.read_bytes())
    return snapshot


def test_fixture_seals_terminal_25_rejections_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_repository(tmp_path)

    def forbid_socket(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"network use forbidden: {args!r} {kwargs!r}")

    monkeypatch.setattr(socket, "socket", forbid_socket)
    output = _run(fixture)

    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == {
        "final_decisions.jsonl",
        "manifest.json",
        "manifest.sha256",
        "needs_review.jsonl",
        "rejections.jsonl",
        "reviewed_decisions.jsonl",
        "source_bindings.json",
        "summary.json",
    }
    summary = _read_json(output / "summary.json")
    expected_status = "OFFLINE_RULE_REVIEW_COMPLETE_ALL_CANDIDATES_REJECTED"
    excluded_monitor_statuses = {"NO_EXECUTABLE_SHADOW_EDGE", "FEE_EVIDENCE_UNAVAILABLE"}
    assert summary["status"] == expected_status
    assert summary["status"] not in excluded_monitor_statuses
    for key, value in _OUTPUT_SAFETY.items():
        assert summary[key] == value
    assert summary["counts"] == {
        "carried_rejected": 11,
        "needs_review": 0,
        "newly_rejected": 14,
        "rejected": 25,
        "reviewed_total": 14,
        "total": 25,
        "verified_equivalent": 0,
    }
    final = _read_jsonl(output / "final_decisions.jsonl")
    assert len(final) == len(_read_jsonl(output / "rejections.jsonl")) == 25
    assert (output / "needs_review.jsonl").read_bytes() == b""
    assert (output / "reviewed_decisions.jsonl").read_bytes() == fixture["review_path"].read_bytes()
    assert [row["decision_source"]["artifact"] for row in final[:11]] == ["source_adjudication_decisions"] * 11
    assert [row["decision_source"]["artifact"] for row in final[11:]] == ["reviewed_decisions"] * 14
    assert all(
        row["decision"] == "REJECTED"
        and row["live_eligible"] is False
        and row["orders_submitted"] == 0
        and row["books_requested"] is False
        and row["economics_computed"] is False
        and row["profitability_evaluation_eligible"] is False
        for row in final
    )
    manifest_payload = (output / "manifest.json").read_bytes()
    assert (output / "manifest.sha256").read_bytes() == f"{_sha(manifest_payload)}  manifest.json\n".encode()
    manifest = json.loads(manifest_payload)
    assert manifest["status"] == expected_status
    assert manifest["status"] not in excluded_monitor_statuses
    assert manifest["authority"] == {"offline_only": True, **_OUTPUT_SAFETY}
    assert {"native_books", "calculations", "alerts"}.isdisjoint(manifest["artifacts"])
    for binding in manifest["artifacts"].values():
        payload = (output / binding["path"]).read_bytes()
        assert binding["byte_size"] == len(payload)
        assert binding["sha256"] == _sha(payload)


def test_capture_raw_tamper_fails_before_output(tmp_path: Path) -> None:
    fixture = _make_repository(tmp_path)
    fixture["raw_path"].write_bytes(b"tampered native evidence\n")

    with pytest.raises(RuleReviewError, match="does not match its manifest binding"):
        _run(fixture)

    assert not (fixture["root"] / fixture["protocol"]["output"]["path"]).exists()


def test_review_input_hash_tamper_fails_before_output(tmp_path: Path) -> None:
    fixture = _make_repository(tmp_path)
    with fixture["review_path"].open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(RuleReviewError, match="reviewed-decisions SHA-256 mismatch"):
        _run(fixture)

    assert not (fixture["root"] / fixture["protocol"]["output"]["path"]).exists()


def test_rejected_requires_material_mismatch(tmp_path: Path) -> None:
    fixture = _make_repository(tmp_path)
    reviews = _read_jsonl(fixture["review_path"])
    reviews[0]["axis_findings"][0]["status"] = "UNRESOLVED"
    _rewrite_reviews(fixture, reviews)

    with pytest.raises(RuleReviewError, match="REJECTED requires at least one material MISMATCH"):
        _run(fixture)


def test_evidence_raw_binding_tamper_is_rejected(tmp_path: Path) -> None:
    fixture = _make_repository(tmp_path)
    reviews = _read_jsonl(fixture["review_path"])
    reviews[0]["evidence_resources"][0]["raw"]["sha256"] = "0" * 64
    _rewrite_reviews(fixture, reviews)

    with pytest.raises(RuleReviewError, match="evidence resource binding mismatch"):
        _run(fixture)


def test_duplicate_review_candidate_is_rejected(tmp_path: Path) -> None:
    fixture = _make_repository(tmp_path)
    reviews = _read_jsonl(fixture["review_path"])
    reviews[1]["candidate_id"] = reviews[0]["candidate_id"]
    _rewrite_reviews(fixture, reviews)

    with pytest.raises(RuleReviewError, match="duplicate candidate|exact source candidate order"):
        _run(fixture)


def test_prior_adjudication_tamper_is_rejected(tmp_path: Path) -> None:
    fixture = _make_repository(tmp_path)
    with fixture["prior_decisions_path"].open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(RuleReviewError, match="does not match its binding"):
        _run(fixture)


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    fixture = _make_repository(tmp_path)
    output = _run(fixture)
    manifest_before = (output / "manifest.json").read_bytes()

    with pytest.raises(RuleReviewError, match="already exists"):
        _run(fixture)

    assert (output / "manifest.json").read_bytes() == manifest_before


def test_competing_final_directory_at_publish_boundary_is_preserved_and_retains_failed_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_repository(tmp_path)
    final_path = fixture["root"] / fixture["protocol"]["output"]["path"]
    staging_path = final_path.with_name(f"{final_path.name}.inprogress")
    original_rename = rule_review_module._rename_no_replace
    sentinel_payload = b"competing writer owns this final directory\n"

    def inject_competing_final(source: Path, destination: Path) -> None:
        assert source == staging_path
        assert destination == final_path
        destination.mkdir()
        _write(destination / "sentinel.bin", sentinel_payload)
        original_rename(source, destination)

    monkeypatch.setattr(rule_review_module, "_rename_no_replace", inject_competing_final)

    with pytest.raises(RuleReviewError, match="already exists"):
        _run(fixture)

    assert _tree_snapshot(final_path) == {"sentinel.bin": ("file", sentinel_payload)}
    assert staging_path.is_dir()
    failure = _read_json(staging_path / "failure.json")
    assert failure["status"] == "FAILED_CLOSED_STAGING_RETAINED"
    assert failure["retry_permitted"] is False
    assert failure["orders_submitted"] == 0
    assert failure["live_eligible"] is False


@pytest.mark.parametrize("preexisting_kind", ["final", "inprogress"])
def test_preexisting_final_or_inprogress_tree_is_never_overwritten(
    tmp_path: Path,
    preexisting_kind: str,
) -> None:
    fixture = _make_repository(tmp_path)
    final_path = fixture["root"] / fixture["protocol"]["output"]["path"]
    staging_path = final_path.with_name(f"{final_path.name}.inprogress")
    preexisting_path = final_path if preexisting_kind == "final" else staging_path
    _write(preexisting_path / "nested" / "sentinel.bin", f"preserve-{preexisting_kind}\n".encode())
    output_parent = final_path.parent
    before = _tree_snapshot(output_parent)

    with pytest.raises(RuleReviewError, match="final or in-progress rule-review output already exists"):
        _run(fixture)

    assert _tree_snapshot(output_parent) == before


def test_repository_review_packet_matches_sealed_capture_without_writing() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "results/rule_review_v1/protocol_v1.json"
    protocol = _read_json(protocol_path)
    manifest_binding = protocol["source_rule_evidence_run"]["manifest"]
    review_binding = protocol["reviewed_decisions"]
    manifest_path = root / manifest_binding["path"]
    review_path = root / review_binding["path"]
    output_path = root / protocol["output"]["path"]
    output_existed_before = output_path.exists()

    assert _binding(manifest_path, root) == manifest_binding
    assert _binding(review_path, root) == review_binding
    capture = rule_review_module._validate_capture(
        root=root,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_binding["sha256"],
    )
    snapshot = rule_review_module._read_input(review_path, root=root, label="reviewed decisions")
    rows, counts = rule_review_module._validate_reviews(snapshot, capture)

    assert len(rows) == 14
    assert counts == {"NEEDS_REVIEW": 0, "REJECTED": 14}
    assert output_path.exists() is output_existed_before
