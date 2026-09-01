"""Offline tests for hash-bound, immutable PMXT adjudication."""

from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import src.indexers.pmxt.offline_adjudication as offline_adjudication_module
from src.indexers.pmxt.offline_adjudication import (
    OfflineAdjudicationError,
    run_offline_adjudication,
)

_SCHEMA_SOURCE = (
    Path(__file__).resolve().parents[1] / "results" / "semantic_adjudication_v1" / "decision_schema_v1.json"
)


def _value_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return _value_bytes(value) + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict]) -> list[dict]:
    payloads = [_json_bytes(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(payloads))
    return [{"line": index, "row_sha256": _sha256(payload)} for index, payload in enumerate(payloads, start=1)]


def _pmxt_market_id(candidate_id: str, venue: str) -> str:
    return f"pmxt_{venue}_{candidate_id}"


def _acquisition(path: str, payload: object, *, request_id: str) -> dict:
    raw_body = _value_bytes(payload)
    return {
        "method": "GET",
        "path": path,
        "params": {},
        "status_code": 200,
        "request_wall_utc": "2026-08-29T12:00:00.000000Z",
        "request_monotonic_ns": 1_000_000_000,
        "response_wall_utc": "2026-08-29T12:00:00.005000Z",
        "response_monotonic_ns": 1_005_000_000,
        "rtt_ms": 5.0,
        "http_date": "Sat, 29 Aug 2026 12:00:00 GMT",
        "age_header": None,
        "cache_control": "no-store",
        "request_id": request_id,
        "request_id_header": "x-request-id",
        "raw_body_hash": _sha256(raw_body),
        "raw_body_base64": base64.b64encode(raw_body).decode("ascii"),
        "response_headers": {
            "date": "Sat, 29 Aug 2026 12:00:00 GMT",
            "cache-control": "no-store",
            "x-request-id": request_id,
        },
        "response_header_items": [
            ["date", "Sat, 29 Aug 2026 12:00:00 GMT"],
            ["cache-control", "no-store"],
            ["x-request-id", request_id],
        ],
        "body_complete": True,
        "freshness_basis": "LOCAL_RECEIPT_BOUNDED",
        "requested_at": "2026-08-29T12:00:00.000000Z",
        "received_at": "2026-08-29T12:00:00.005000Z",
    }


def _native_row(candidate_id: str, venue: str, side: str) -> dict:
    slug = candidate_id.upper()
    proposition = f"Will {candidate_id} occur?"
    authority = "Official fixture authority"
    source_url = "https://authority.example/result"
    open_time = "2026-01-01T00:00:00Z"
    close_time = "2026-11-04T02:00:00Z"
    expiration_time = "2026-11-04T04:00:00Z"
    void_cancel = "Void only if the event is permanently cancelled."
    if venue == "kalshi":
        native_market_id = f"KX-{slug}"
        native_event_id = f"KXEVENT-{slug}"
        native_series_id = f"KXSERIES-{slug}"
        resolution_criteria = "Resolves by official certification."
        market = {
            "ticker": native_market_id,
            "event_ticker": native_event_id,
            "series_ticker": native_series_id,
            "title": proposition,
            "yes_sub_title": "Yes",
            "no_sub_title": "No",
            "rules_primary": resolution_criteria,
            "rules_secondary": None,
            "open_time": open_time,
            "close_time": close_time,
            "expiration_time": expiration_time,
            "expected_expiration_time": None,
            "latest_expiration_time": None,
            "settlement_timer_seconds": 3600,
            "void_cancel_rules": void_cancel,
            "market_type": "binary",
            "mve_collection_ticker": None,
            "mve_selected_legs": [],
        }
        settlement_source = {"name": authority, "url": source_url}
        raw_response = {
            "market": {"market": market},
            "event_metadata": {"settlement_sources": [settlement_source]},
            "event": {
                "event": {
                    "event_ticker": native_event_id,
                    "series_ticker": native_series_id,
                    "mutually_exclusive": False,
                    "settlement_sources": [settlement_source],
                }
            },
            "series": {
                "series": {
                    "ticker": native_series_id,
                    "settlement_sources": [settlement_source],
                }
            },
        }
        normalized_rules = {
            "rules_primary": resolution_criteria,
            "rules_secondary": None,
            "open_time": open_time,
            "close_time": close_time,
            "expiration_time": expiration_time,
            "expected_expiration_time": None,
            "latest_expiration_time": None,
            "settlement_timer_seconds": 3600,
            "settlement_sources": [settlement_source],
            "void_cancel_rules": void_cancel,
            "market_type": "binary",
            "mve_collection_ticker": None,
            "mve_selected_legs": [],
            "event_mutually_exclusive": False,
            "outcome_polarity": {"YES": "YES", "NO": "NO"},
        }
        requests = [
            _acquisition(
                f"/markets/{native_market_id}",
                raw_response["market"],
                request_id=f"{candidate_id}-kalshi-market",
            ),
            _acquisition(
                f"/events/{native_event_id}/metadata",
                raw_response["event_metadata"],
                request_id=f"{candidate_id}-kalshi-event-metadata",
            ),
            _acquisition(
                f"/events/{native_event_id}",
                raw_response["event"],
                request_id=f"{candidate_id}-kalshi-event",
            ),
            _acquisition(
                f"/series/{native_series_id}",
                raw_response["series"],
                request_id=f"{candidate_id}-kalshi-series",
            ),
        ]
        native_condition_id = None
        native_outcome_ids = {"YES": native_market_id, "NO": native_market_id}
        native_outcome_labels = {"YES": "Yes", "NO": "No"}
        event_mutually_exclusive = False
        negative_risk = None
    else:
        native_market_id = f"POLY-{slug}"
        native_event_id = f"POLYEVENT-{slug}"
        native_series_id = None
        native_condition_id = f"0x{_sha256(candidate_id.encode('utf-8'))}"
        resolution_criteria = "Resolves when three named media sources call the race."
        selected_market = {
            "id": native_market_id,
            "conditionId": native_condition_id,
            "question": proposition,
            "description": resolution_criteria,
            "resolutionSource": source_url,
            "resolvedBy": authority,
            "startDate": open_time,
            "endDate": close_time,
            "umaEndDate": expiration_time,
            "voidCancelRules": void_cancel,
            "settlementDelaySeconds": 3600,
            "negRisk": False,
            "outcomes": ["Yes", "No"],
            "clobTokenIds": [f"{native_market_id}-YES", f"{native_market_id}-NO"],
            "events": [{"id": native_event_id}],
        }
        markets = [selected_market]
        raw_response = {"markets": markets, "selected_market": selected_market}
        normalized_rules = {
            "question": proposition,
            "description": resolution_criteria,
            "resolution_source": source_url,
            "resolved_by": authority,
            "start_date": open_time,
            "end_date": close_time,
            "uma_end_date": expiration_time,
            "void_cancel_rules": void_cancel,
            "settlement_delay_seconds": 3600.0,
            "neg_risk": False,
            "outcome_polarity": {"YES": "YES", "NO": "NO"},
        }
        requests = [
            _acquisition(
                "/markets",
                markets,
                request_id=f"{candidate_id}-polymarket-markets",
            )
        ]
        native_outcome_ids = {
            "YES": f"{native_market_id}-YES",
            "NO": f"{native_market_id}-NO",
        }
        native_outcome_labels = {"YES": "Yes", "NO": "No"}
        event_mutually_exclusive = None
        negative_risk = False
    return {
        "candidate_id": candidate_id,
        "venue": venue,
        "side": side,
        "pmxt_market_id": _pmxt_market_id(candidate_id, venue),
        "native_market_id": native_market_id,
        "native_event_id": native_event_id,
        "native_series_id": native_series_id,
        "native_condition_id": native_condition_id,
        "native_outcome_ids": native_outcome_ids,
        "native_outcome_labels": native_outcome_labels,
        "status": "RESOLVED",
        "requests": requests,
        "raw_response": raw_response,
        "raw_sha256": _sha256(_value_bytes(raw_response)),
        "normalized_rules": normalized_rules,
        "rule_hash": _sha256(_value_bytes(normalized_rules)),
        "proposition": proposition,
        "outcome_polarity": {"YES": "YES", "NO": "NO"},
        "open_time": open_time,
        "close_time": close_time,
        "expiration_time": expiration_time,
        "expected_expiration_time": None,
        "latest_expiration_time": None,
        "settlement_authority": authority,
        "resolution_source": source_url,
        "resolution_criteria": resolution_criteria,
        "void_cancel": void_cancel,
        "material_edge_cases": {},
        "settlement_delay_seconds": 3600,
        "market_type": "binary",
        "mve_collection_ticker": None,
        "mve_selected_legs": [] if venue == "kalshi" else None,
        "event_mutually_exclusive": event_mutually_exclusive,
        "negative_risk": negative_risk,
        "live_eligible": False,
    }


def _source_ref(artifact: str, row_ref: dict, native_row: dict | None = None) -> dict:
    result = {
        "artifact": artifact,
        "line": row_ref["line"],
        "row_sha256": row_ref["row_sha256"],
    }
    if native_row is not None:
        result.update(
            {
                "raw_sha256": native_row["raw_sha256"],
                "rule_hash": native_row["rule_hash"],
            }
        )
    return result


def _review_row(
    *,
    candidate_id: str,
    decision: str,
    candidate_ref: dict,
    decision_ref: dict,
    kalshi_ref: dict,
    polymarket_ref: dict,
    kalshi_row: dict,
    polymarket_row: dict,
) -> dict:
    common = {
        "schema_version": 1,
        "record_type": "pmxt_offline_adjudication",
        "candidate_id": candidate_id,
        "decision": decision,
        "review_group": "fixture review",
        "review_summary": "Fixture-only semantic adjudication.",
        "reason_codes": ["EXPLICIT_RULE_MISMATCH"] if decision == "REJECTED" else ["NATIVE_RULE_EVIDENCE_INCOMPLETE"],
        "source_rows": {
            "candidate": _source_ref("candidates", candidate_ref),
            "automated_decision": _source_ref("semantic_decisions", decision_ref),
            "kalshi_metadata": _source_ref("raw_native_metadata", kalshi_ref, kalshi_row),
            "polymarket_metadata": _source_ref("raw_native_metadata", polymarket_ref, polymarket_row),
        },
        "automated_decision_evidence": [
            {"source_row": "automated_decision", "json_pointer": "/status"},
            {"source_row": "automated_decision", "json_pointer": "/reason_codes"},
        ],
        "reviewed": True,
        "semantic_verified": False,
        "profitability_evaluation_eligible": False,
        "network_requests": 0,
        "orders_submitted": 0,
        "economics_computed": False,
        "live_eligible": False,
    }
    if decision == "REJECTED":
        kalshi_clause = kalshi_row["normalized_rules"]["rules_primary"]
        polymarket_clause = polymarket_row["normalized_rules"]["description"]
        common.update(
            {
                "material_mismatches": [
                    {
                        "axis": "resolution_criteria",
                        "summary": "Certification and media-call triggers are not identical.",
                        "kalshi_clause": {
                            "source_row": "kalshi_metadata",
                            "json_pointer": "/normalized_rules/rules_primary",
                            "clause_sha256": _sha256(_value_bytes(kalshi_clause)),
                        },
                        "polymarket_clause": {
                            "source_row": "polymarket_metadata",
                            "json_pointer": "/normalized_rules/description",
                            "clause_sha256": _sha256(_value_bytes(polymarket_clause)),
                        },
                    }
                ],
                "evidence_gaps": [],
            }
        )
    else:
        common.update(
            {
                "material_mismatches": [],
                "evidence_gaps": [
                    {
                        "axis": "void_cancel",
                        "summary": "Neither captured row provides a terminal void clause.",
                        "evidence": [
                            {
                                "source_row": "kalshi_metadata",
                                "json_pointer": "/normalized_rules/void_cancel_rules",
                            },
                            {
                                "source_row": "polymarket_metadata",
                                "json_pointer": "/normalized_rules/void_cancel_rules",
                            },
                        ],
                    }
                ],
            }
        )
    return common


def _fixture_repository(tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    root = tmp_path / "repo"
    run_dir = root / "data/pmxt/runs/source_run"
    candidate_ids = ["pmxt_candidate_rejected", "pmxt_candidate_needs_review"]
    candidates = [
        {
            "candidate_id": candidate_id,
            "cluster_id": f"cluster_{candidate_id}",
            "relation": "identity",
            "raw_edge_present": True,
            "venue_a": "kalshi",
            "venue_b": "polymarket",
            "pmxt_market_id_a": _pmxt_market_id(candidate_id, "kalshi"),
            "pmxt_market_id_b": _pmxt_market_id(candidate_id, "polymarket"),
            "live_eligible": False,
        }
        for candidate_id in candidate_ids
    ]
    automated_decisions = [
        {
            "candidate_id": candidate_id,
            "status": "NEEDS_REVIEW",
            "reason_codes": ["SOURCE_AUTOMATION_FAIL_CLOSED"],
            "live_eligible": False,
        }
        for candidate_id in candidate_ids
    ]
    native_rows = [
        _native_row(candidate_id, venue, side)
        for candidate_id in candidate_ids
        for venue, side in (("kalshi", "a"), ("polymarket", "b"))
    ]
    candidate_refs = _write_jsonl(run_dir / "candidates.jsonl", candidates)
    native_refs = _write_jsonl(run_dir / "raw_native_metadata.jsonl", native_rows)
    decision_refs = _write_jsonl(run_dir / "semantic_decisions.jsonl", automated_decisions)

    manifest_artifacts = {}
    for name in ("candidates", "raw_native_metadata", "semantic_decisions"):
        path = run_dir / f"{name}.jsonl"
        manifest_artifacts[name] = {
            "path": path.name,
            "byte_size": path.stat().st_size,
            "sha256": _sha256(path.read_bytes()),
        }
    source_manifest = {
        "schema_version": 1,
        "run_id": "source_run",
        "status": "NO_VERIFIED_CANDIDATES",
        "counts": {name: len(_read_jsonl(run_dir / f"{name}.jsonl")) for name in manifest_artifacts},
        "artifacts": manifest_artifacts,
        "no_order_actions": True,
        "live_eligible": False,
    }
    _write_json(run_dir / "manifest.json", source_manifest)

    reviews = []
    for index, candidate_id in enumerate(candidate_ids):
        native_index = index * 2
        reviews.append(
            _review_row(
                candidate_id=candidate_id,
                decision="REJECTED" if index == 0 else "NEEDS_REVIEW",
                candidate_ref=candidate_refs[index],
                decision_ref=decision_refs[index],
                kalshi_ref=native_refs[native_index],
                polymarket_ref=native_refs[native_index + 1],
                kalshi_row=native_rows[native_index],
                polymarket_row=native_rows[native_index + 1],
            )
        )
    review_path = root / "results/semantic_adjudication_v1/reviewed_decisions_v1.jsonl"
    _write_jsonl(review_path, reviews)
    schema = json.loads(_SCHEMA_SOURCE.read_text(encoding="utf-8"))
    schema_path = root / "results/semantic_adjudication_v1/decision_schema_v1.json"
    _write_json(schema_path, schema)

    protocol = {
        "schema_version": 1,
        "protocol_id": "pmxt-offline-semantic-adjudication-v1",
        "authority": {
            "offline_only": True,
            "network_requests": 0,
            "credentials_read": False,
            "books_requested": False,
            "orders_submitted": 0,
            "economics_computed": False,
            "live_eligible": False,
        },
        "adjudication_provenance": {
            "method": "offline_native_rule_clause_review",
            "version": 1,
            "review_completed_at_utc": "2026-08-29T12:30:00Z",
        },
        "expected_counts": {"total": 2, "rejected": 1, "needs_review": 1},
        "source_monitor_run": {
            "run_id": "source_run",
            "manifest": {
                "path": "data/pmxt/runs/source_run/manifest.json",
                "sha256": _sha256((run_dir / "manifest.json").read_bytes()),
            },
            "artifacts": {
                name: {
                    "path": f"data/pmxt/runs/source_run/{name}.jsonl",
                    "sha256": manifest_artifacts[name]["sha256"],
                }
                for name in manifest_artifacts
            },
        },
        "decision_schema": {
            "path": "results/semantic_adjudication_v1/decision_schema_v1.json",
            "sha256": _sha256(schema_path.read_bytes()),
        },
        "review_input": {
            "path": "results/semantic_adjudication_v1/reviewed_decisions_v1.jsonl",
            "sha256": _sha256(review_path.read_bytes()),
        },
        "output": {
            "run_id": "adjudication_fixture",
            "path": "data/pmxt/semantic_adjudication/runs/adjudication_fixture",
        },
    }
    protocol_path = root / "results/semantic_adjudication_v1/protocol_v1.json"
    _write_json(protocol_path, protocol)
    return root, protocol, reviews


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite_protocol(root: Path, protocol: dict) -> str:
    protocol_path = root / "results/semantic_adjudication_v1/protocol_v1.json"
    _write_json(protocol_path, protocol)
    return _sha256(protocol_path.read_bytes())


def _reseal_review_input(root: Path, protocol: dict, reviews: list[dict]) -> str:
    review_path = root / protocol["review_input"]["path"]
    _write_jsonl(review_path, reviews)
    protocol["review_input"]["sha256"] = _sha256(review_path.read_bytes())
    schema_path = root / protocol["decision_schema"]["path"]
    protocol["decision_schema"]["sha256"] = _sha256(schema_path.read_bytes())
    return _rewrite_protocol(root, protocol)


def _reseal_source_inputs(root: Path, protocol: dict, reviews: list[dict]) -> str:
    run_dir = root / "data/pmxt/runs/source_run"
    artifact_rows = {
        name: _read_jsonl(run_dir / f"{name}.jsonl")
        for name in ("candidates", "raw_native_metadata", "semantic_decisions")
    }
    artifact_refs = {name: _write_jsonl(run_dir / f"{name}.jsonl", rows) for name, rows in artifact_rows.items()}

    source_manifest_path = run_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    for name, rows in artifact_rows.items():
        path = run_dir / f"{name}.jsonl"
        source_manifest["counts"][name] = len(rows)
        source_manifest["artifacts"][name].update(
            {
                "byte_size": path.stat().st_size,
                "sha256": _sha256(path.read_bytes()),
            }
        )
        protocol["source_monitor_run"]["artifacts"][name]["sha256"] = _sha256(path.read_bytes())
    _write_json(source_manifest_path, source_manifest)
    protocol["source_monitor_run"]["manifest"]["sha256"] = _sha256(source_manifest_path.read_bytes())

    candidates = {
        row["candidate_id"]: (row, artifact_refs["candidates"][index])
        for index, row in enumerate(artifact_rows["candidates"])
    }
    decisions = {
        row["candidate_id"]: (row, artifact_refs["semantic_decisions"][index])
        for index, row in enumerate(artifact_rows["semantic_decisions"])
    }
    native = {
        (row["candidate_id"], row["venue"]): (
            row,
            artifact_refs["raw_native_metadata"][index],
        )
        for index, row in enumerate(artifact_rows["raw_native_metadata"])
    }
    for review in reviews:
        candidate_id = review["candidate_id"]
        candidate_row, candidate_ref = candidates[candidate_id]
        decision_row, decision_ref = decisions[candidate_id]
        kalshi_row, kalshi_ref = native[(candidate_id, "kalshi")]
        polymarket_row, polymarket_ref = native[(candidate_id, "polymarket")]
        del candidate_row, decision_row
        review["source_rows"] = {
            "candidate": _source_ref("candidates", candidate_ref),
            "automated_decision": _source_ref("semantic_decisions", decision_ref),
            "kalshi_metadata": _source_ref("raw_native_metadata", kalshi_ref, kalshi_row),
            "polymarket_metadata": _source_ref("raw_native_metadata", polymarket_ref, polymarket_row),
        }
    return _reseal_review_input(root, protocol, reviews)


def _run(root: Path) -> Path:
    protocol_path = root / "results/semantic_adjudication_v1/protocol_v1.json"
    return run_offline_adjudication(
        repository_root=root,
        protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
        expected_protocol_sha256=_sha256(protocol_path.read_bytes()),
    )


def _output_path(root: Path, protocol: dict) -> Path:
    return root / protocol["output"]["path"]


def test_seals_offline_decisions_and_refuses_overwrite(tmp_path: Path) -> None:
    root, _, _ = _fixture_repository(tmp_path)

    output = _run(root)

    decisions = _read_jsonl(output / "decisions.jsonl")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [row["decision"] for row in decisions] == ["REJECTED", "NEEDS_REVIEW"]
    assert all(row["live_eligible"] is False for row in decisions)
    assert summary["counts"] == {
        "total": 2,
        "rejected": 1,
        "needs_review": 1,
        "verified_equivalent": 0,
    }
    assert summary["profitability_established"] is False
    assert manifest["authority"]["orders_submitted"] == 0
    assert manifest["authority"]["live_eligible"] is False
    sidecar = (output / "manifest.sha256").read_text(encoding="ascii")
    assert sidecar == f"{_sha256((output / 'manifest.json').read_bytes())}  manifest.json\n"
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(OfflineAdjudicationError, match="already exists"):
        _run(root)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_source_tamper_fails_before_output_reservation(tmp_path: Path) -> None:
    root, _, _ = _fixture_repository(tmp_path)
    source = root / "data/pmxt/runs/source_run/candidates.jsonl"
    source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(OfflineAdjudicationError, match="size mismatch"):
        _run(root)

    assert not (root / "data/pmxt/semantic_adjudication").exists()


def test_rejected_requires_nonempty_clauses_from_both_venues(tmp_path: Path) -> None:
    root, protocol, reviews = _fixture_repository(tmp_path)
    rejected = reviews[0]
    rejected["material_mismatches"][0]["kalshi_clause"] = {
        "source_row": "kalshi_metadata",
        "json_pointer": "/normalized_rules/rules_secondary",
        "clause_sha256": _sha256(_value_bytes(None)),
    }
    review_path = root / protocol["review_input"]["path"]
    _write_jsonl(review_path, reviews)
    protocol["review_input"]["sha256"] = _sha256(review_path.read_bytes())
    expected_hash = _rewrite_protocol(root, protocol)

    with pytest.raises(OfflineAdjudicationError, match="non-empty two-sided clause"):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )

    assert not (root / protocol["output"]["path"]).exists()


def test_rejected_clause_hash_is_bound_to_resolved_text(tmp_path: Path) -> None:
    root, protocol, reviews = _fixture_repository(tmp_path)
    reviews[0]["material_mismatches"][0]["polymarket_clause"]["clause_sha256"] = "0" * 64
    review_path = root / protocol["review_input"]["path"]
    _write_jsonl(review_path, reviews)
    protocol["review_input"]["sha256"] = _sha256(review_path.read_bytes())
    expected_hash = _rewrite_protocol(root, protocol)

    with pytest.raises(OfflineAdjudicationError, match="clause SHA-256 mismatch"):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )


def test_rejected_mismatch_may_retain_separately_validated_evidence_gaps(
    tmp_path: Path,
) -> None:
    root, protocol, reviews = _fixture_repository(tmp_path)
    reviews[0]["evidence_gaps"] = deepcopy(reviews[1]["evidence_gaps"])
    expected_hash = _reseal_review_input(root, protocol, reviews)

    output = run_offline_adjudication(
        repository_root=root,
        protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
        expected_protocol_sha256=expected_hash,
    )

    sealed_rows = _read_jsonl(output / "decisions.jsonl")
    assert sealed_rows[0]["decision"] == "REJECTED"
    assert sealed_rows[0]["material_mismatches"]
    assert sealed_rows[0]["evidence_gaps"] == reviews[0]["evidence_gaps"]


def test_expected_counts_are_configurable_and_enforced(tmp_path: Path) -> None:
    root, protocol, _ = _fixture_repository(tmp_path)
    protocol["expected_counts"] = {"total": 2, "rejected": 2, "needs_review": 0}
    expected_hash = _rewrite_protocol(root, protocol)

    with pytest.raises(OfflineAdjudicationError, match="decision counts"):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )


def test_review_safety_flags_fail_closed(tmp_path: Path) -> None:
    root, protocol, reviews = _fixture_repository(tmp_path)
    unsafe = deepcopy(reviews)
    unsafe[1]["live_eligible"] = True
    review_path = root / protocol["review_input"]["path"]
    _write_jsonl(review_path, unsafe)
    protocol["review_input"]["sha256"] = _sha256(review_path.read_bytes())
    expected_hash = _rewrite_protocol(root, protocol)

    with pytest.raises(
        OfflineAdjudicationError,
        match="(?:safety boundary|does not satisfy decision schema)",
    ):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )


def test_verified_equivalent_is_not_an_allowed_offline_decision(tmp_path: Path) -> None:
    root, protocol, reviews = _fixture_repository(tmp_path)
    reviews[1]["decision"] = "VERIFIED_EQUIVALENT"
    review_path = root / protocol["review_input"]["path"]
    _write_jsonl(review_path, reviews)
    protocol["review_input"]["sha256"] = _sha256(review_path.read_bytes())
    expected_hash = _rewrite_protocol(root, protocol)

    with pytest.raises(
        OfflineAdjudicationError,
        match="(?:REJECTED or NEEDS_REVIEW|does not satisfy decision schema)",
    ):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )


def test_bound_draft_2020_12_schema_is_applied_to_each_review_row(
    tmp_path: Path,
) -> None:
    root, protocol, reviews = _fixture_repository(tmp_path)
    schema_path = root / protocol["decision_schema"]["path"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["properties"]["review_group"]["const"] = "schema-required-group"
    _write_json(schema_path, schema)
    expected_hash = _reseal_review_input(root, protocol, reviews)

    with pytest.raises(OfflineAdjudicationError, match="does not satisfy decision schema"):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )

    assert not _output_path(root, protocol).exists()


def test_unsupported_draft_schema_keyword_fails_closed(tmp_path: Path) -> None:
    root, protocol, reviews = _fixture_repository(tmp_path)
    schema_path = root / protocol["decision_schema"]["path"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["properties"]["review_group"]["maxLength"] = 16
    _write_json(schema_path, schema)
    expected_hash = _reseal_review_input(root, protocol, reviews)

    with pytest.raises(
        OfflineAdjudicationError,
        match=r"unsupported Draft 2020-12 keyword: maxLength",
    ):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )

    assert not _output_path(root, protocol).exists()


@pytest.mark.parametrize("field", ["network_requests", "orders_submitted"])
def test_boolean_false_does_not_satisfy_review_row_integer_zero(
    tmp_path: Path,
    field: str,
) -> None:
    root, protocol, reviews = _fixture_repository(tmp_path)
    reviews[0][field] = False
    expected_hash = _reseal_review_input(root, protocol, reviews)

    with pytest.raises(OfflineAdjudicationError):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )

    assert not _output_path(root, protocol).exists()


def test_boolean_false_does_not_satisfy_protocol_integer_zero(tmp_path: Path) -> None:
    root, protocol, _ = _fixture_repository(tmp_path)
    protocol["authority"]["network_requests"] = False
    expected_hash = _rewrite_protocol(root, protocol)

    with pytest.raises(OfflineAdjudicationError, match="offline-only boundary"):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )

    assert not _output_path(root, protocol).exists()


def test_candidate_to_native_pmxt_id_binding_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    root, protocol, reviews = _fixture_repository(tmp_path)
    native_path = root / "data/pmxt/runs/source_run/raw_native_metadata.jsonl"
    native_rows = _read_jsonl(native_path)
    native_rows[0]["pmxt_market_id"] = "different_pmxt_discovery_id"
    _write_jsonl(native_path, native_rows)
    expected_hash = _reseal_source_inputs(root, protocol, reviews)

    with pytest.raises(
        OfflineAdjudicationError,
        match="KALSHI_PMXT_MARKET_ID_BINDING_MISMATCH",
    ):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )

    assert not _output_path(root, protocol).exists()


def test_top_level_native_id_must_match_retained_native_response(
    tmp_path: Path,
) -> None:
    root, protocol, reviews = _fixture_repository(tmp_path)
    native_path = root / "data/pmxt/runs/source_run/raw_native_metadata.jsonl"
    native_rows = _read_jsonl(native_path)
    native_rows[0]["native_market_id"] = "KX-DIFFERENT-NATIVE-ID"
    _write_jsonl(native_path, native_rows)
    expected_hash = _reseal_source_inputs(root, protocol, reviews)

    with pytest.raises(
        OfflineAdjudicationError,
        match="KALSHI_TOP_LEVEL_NATIVE_MARKET_ID_RAW_RESPONSE_MISMATCH",
    ):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )

    assert not _output_path(root, protocol).exists()


def test_rehashed_acquisition_body_must_still_match_retained_raw_response(
    tmp_path: Path,
) -> None:
    root, protocol, reviews = _fixture_repository(tmp_path)
    native_path = root / "data/pmxt/runs/source_run/raw_native_metadata.jsonl"
    native_rows = _read_jsonl(native_path)
    request = native_rows[0]["requests"][0]
    replacement_body = _value_bytes({"market": {"ticker": "KX-OTHER"}})
    request["raw_body_base64"] = base64.b64encode(replacement_body).decode("ascii")
    request["raw_body_hash"] = _sha256(replacement_body)
    _write_jsonl(native_path, native_rows)
    expected_hash = _reseal_source_inputs(root, protocol, reviews)

    with pytest.raises(
        OfflineAdjudicationError,
        match="KALSHI_MARKET_RAW_BODY_RESPONSE_MISMATCH",
    ):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )

    assert not _output_path(root, protocol).exists()


@pytest.mark.parametrize(
    "untrusted_path",
    [
        "results/adjudication_fixture",
        "data/pmxt/semantic_adjudication/other/adjudication_fixture",
        "data/pmxt/semantic_adjudication/runs/../adjudication_fixture",
        "../outside/adjudication_fixture",
    ],
)
def test_output_path_requires_the_exact_dedicated_parent(
    tmp_path: Path,
    untrusted_path: str,
) -> None:
    root, protocol, _ = _fixture_repository(tmp_path)
    protocol["output"]["path"] = untrusted_path
    expected_hash = _rewrite_protocol(root, protocol)

    with pytest.raises(OfflineAdjudicationError, match="output path"):
        run_offline_adjudication(
            repository_root=root,
            protocol_path=Path("results/semantic_adjudication_v1/protocol_v1.json"),
            expected_protocol_sha256=expected_hash,
        )

    assert not (root / "data/pmxt/semantic_adjudication/runs").exists()


@pytest.mark.parametrize("reserved_kind", ["final", "staging"])
def test_preexisting_output_reservation_is_never_overwritten(
    tmp_path: Path,
    reserved_kind: str,
) -> None:
    root, protocol, _ = _fixture_repository(tmp_path)
    final_path = _output_path(root, protocol)
    reserved_path = final_path if reserved_kind == "final" else final_path.with_name(f"{final_path.name}.inprogress")
    reserved_path.mkdir(parents=True)
    sentinel = reserved_path / "owner-data.txt"
    sentinel.write_bytes(b"preserve-owner-data")
    before = sentinel.read_bytes()

    with pytest.raises(OfflineAdjudicationError, match="already exists"):
        _run(root)

    assert sentinel.read_bytes() == before
    assert sorted(path.name for path in reserved_path.iterdir()) == [sentinel.name]


def _symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")


def test_source_file_symlink_is_rejected_even_when_bytes_and_hash_match(
    tmp_path: Path,
) -> None:
    root, protocol, _ = _fixture_repository(tmp_path)
    review_path = root / protocol["review_input"]["path"]
    outside = tmp_path / "outside-review.jsonl"
    outside.write_bytes(review_path.read_bytes())
    review_path.unlink()
    _symlink_or_skip(review_path, outside, target_is_directory=False)

    with pytest.raises(OfflineAdjudicationError, match="symlink or reparse point"):
        _run(root)

    assert not _output_path(root, protocol).exists()


def test_dangling_output_parent_symlink_is_rejected(tmp_path: Path) -> None:
    root, protocol, _ = _fixture_repository(tmp_path)
    linked_parent = root / "data/pmxt/semantic_adjudication"
    missing_target = tmp_path / "missing-output-root"
    _symlink_or_skip(linked_parent, missing_target, target_is_directory=True)

    with pytest.raises(OfflineAdjudicationError, match="symlink or reparse point"):
        _run(root)

    assert not missing_target.exists()


def test_input_swap_after_single_read_cannot_change_validated_or_bound_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, protocol, _ = _fixture_repository(tmp_path)
    candidate_path = root / "data/pmxt/runs/source_run/candidates.jsonl"
    accepted_payload = candidate_path.read_bytes()
    accepted_hash = _sha256(accepted_payload)
    swapped_payload = accepted_payload.replace(b'"identity"', b'"identitx"', 1)
    assert swapped_payload != accepted_payload
    assert len(swapped_payload) == len(accepted_payload)
    real_read_input = offline_adjudication_module._read_input
    did_swap = False

    def swap_after_read(
        path: Path,
        *,
        root: Path,
        label: str,
    ) -> object:
        nonlocal did_swap
        snapshot = real_read_input(path, root=root, label=label)
        if label == "source artifact candidates":
            path.write_bytes(swapped_payload)
            did_swap = True
        return snapshot

    monkeypatch.setattr(
        offline_adjudication_module,
        "_read_input",
        swap_after_read,
    )

    output = _run(root)

    assert did_swap is True
    assert _sha256(candidate_path.read_bytes()) != accepted_hash
    bindings = json.loads((output / "source_bindings.json").read_text(encoding="utf-8"))
    assert bindings["source_monitor_run"]["artifacts"]["candidates"]["sha256"] == accepted_hash
    sealed = _read_jsonl(output / "decisions.jsonl")
    assert [row["candidate_id"] for row in sealed] == [
        "pmxt_candidate_rejected",
        "pmxt_candidate_needs_review",
    ]
