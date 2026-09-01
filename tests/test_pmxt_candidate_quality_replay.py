"""Offline replay contract for PMXT candidate quality.

The replay deliberately stops after candidate normalization and local semantic
verification.  It never constructs a PMXT or venue client and never evaluates
books, fees, shadow economics, alerts, or live eligibility.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from src.indexers.pmxt.artifacts import persist_monitor_run
from src.indexers.pmxt.candidates import normalize_clusters
from src.indexers.pmxt.models import PmxtQuery
from src.indexers.pmxt.semantic import NEEDS_REVIEW, REJECTED, VERIFIED_EQUIVALENT, verify_semantics

_PROPOSITION = "Will Candidate X win the 2026 test election?"
_YES = "Candidate X wins the 2026 test election"
_NO = "Candidate X does not win the 2026 test election"
_OPEN_TIME = "2026-01-01T14:00:00Z"
_CLOSE_TIME = "2026-11-04T02:00:00Z"
_EXPIRATION_TIME = "2026-11-04T04:00:00Z"
_AUTHORITY = "Certified state election authority"
_SOURCE = "Official certified election result"
_CRITERIA = "Resolves YES only if Candidate X is certified as the winner."
_VOID = "Void only if the election is permanently cancelled."
_EDGE_CASES: dict[str, object] = {}
_OBSERVED_AT = "2026-08-29T12:00:00Z"

_CASE_ORDER = (
    "valid_identity_kalshi_a",
    "valid_identity_polymarket_a",
    "reject_relation_overlap",
    "review_inverted_polarity",
    "reject_deadline",
    "reject_authority",
    "reject_rule_source",
)

_EXPECTED_SEMANTIC_RESULTS = {
    "valid_identity_kalshi_a": (VERIFIED_EQUIVALENT, ["ALL_REQUIRED_SEMANTICS_EQUIVALENT"]),
    "valid_identity_polymarket_a": (VERIFIED_EQUIVALENT, ["ALL_REQUIRED_SEMANTICS_EQUIVALENT"]),
    "review_inverted_polarity": (
        NEEDS_REVIEW,
        [
            "OUTCOME_POLARITY_INVERTED_NON_IDENTITY",
            "POLYMARKET_NORMALIZED_RULES_RAW_OUTCOME_POLARITY_MISMATCH",
        ],
    ),
    "reject_deadline": (REJECTED, ["DEADLINE_MISMATCH"]),
    "reject_authority": (REJECTED, ["SETTLEMENT_AUTHORITY_MISMATCH"]),
    "reject_rule_source": (REJECTED, ["RESOLUTION_SOURCE_MISMATCH"]),
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _acquisition(venue: str, path: str, payload: object) -> dict[str, Any]:
    raw_body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    http_date = "Sat, 29 Aug 2026 12:00:00 GMT"
    cache_control = "no-store"
    request_id = f"fixture-{venue}-{hashlib.sha256(path.encode()).hexdigest()[:8]}"
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
        "http_date": http_date,
        "age_header": None,
        "cache_control": cache_control,
        "request_id": request_id,
        "request_id_header": "x-request-id",
        "raw_body_hash": hashlib.sha256(raw_body).hexdigest(),
        "raw_body_base64": base64.b64encode(raw_body).decode("ascii"),
        "response_headers": {
            "date": http_date,
            "cache-control": cache_control,
            "x-request-id": request_id,
        },
        "response_header_items": [
            ["date", http_date],
            ["cache-control", cache_control],
            ["x-request-id", request_id],
        ],
        "body_complete": True,
        "freshness_basis": "LOCAL_RECEIPT_BOUNDED",
        "requested_at": "2026-08-29T12:00:00.000000Z",
        "received_at": "2026-08-29T12:00:00.005000Z",
    }


def _cluster(case_id: str) -> dict[str, Any]:
    venues = ("polymarket", "kalshi") if case_id == "valid_identity_polymarket_a" else ("kalshi", "polymarket")
    relation = "overlap" if case_id == "reject_relation_overlap" else "identity"

    markets = []
    for venue in venues:
        pmxt_market_id = f"pmxt_{venue}_{case_id}"
        markets.append(
            {
                "marketId": pmxt_market_id,
                "sourceExchange": venue,
                "title": _PROPOSITION,
                "description": _CRITERIA,
                "resolutionDate": _CLOSE_TIME,
                "outcomes": [
                    {"outcomeId": f"{pmxt_market_id}_yes", "label": "Yes", "price": 0.52},
                    {"outcomeId": f"{pmxt_market_id}_no", "label": "No", "price": 0.48},
                ],
            }
        )

    return {
        "clusterId": f"mcl_{case_id}",
        "canonicalTitle": _PROPOSITION,
        "category": "Offline fixture",
        "relations": [relation],
        "confidence": 0.95,
        "markets": markets,
        "rawMatches": [
            {
                "marketAId": markets[0]["marketId"],
                "marketBId": markets[1]["marketId"],
                "relation": relation,
                "confidence": 0.95,
                "reasoning": "Synthetic offline candidate-quality fixture.",
            }
        ],
    }


def _candidate_side(candidate: dict[str, Any], venue: str) -> str:
    for side in ("a", "b"):
        if candidate[f"venue_{side}"] == venue:
            return side
    raise AssertionError(f"candidate has no {venue} side")


def _native(candidate: dict[str, Any], venue: str) -> dict[str, Any]:
    case_id = str(candidate["case_id"])
    side = _candidate_side(candidate, venue)
    pmxt_market_id = candidate[f"pmxt_market_id_{side}"]

    if venue == "kalshi":
        native_market_id = f"KX_{case_id}"
        native_event_id = f"KXE_{case_id}"
        native_series_id = f"KXS_{case_id}"
        market = {
            "ticker": native_market_id,
            "event_ticker": native_event_id,
            "series_ticker": native_series_id,
            "title": _PROPOSITION,
            "yes_sub_title": _YES,
            "no_sub_title": _NO,
            "rules_primary": _CRITERIA,
            "rules_secondary": None,
            "open_time": _OPEN_TIME,
            "close_time": _CLOSE_TIME,
            "expiration_time": _EXPIRATION_TIME,
            "expected_expiration_time": None,
            "latest_expiration_time": None,
            "settlement_timer_seconds": 3600,
            "void_cancel_rules": _VOID,
            "market_type": "binary",
            "mve_collection_ticker": None,
            "mve_selected_legs": [],
        }
        source = {"name": _AUTHORITY, "url": _SOURCE}
        raw_response = {
            "market": {"market": market},
            "event_metadata": {"settlement_sources": [source]},
            "event": {
                "event": {
                    "event_ticker": native_event_id,
                    "series_ticker": native_series_id,
                    "mutually_exclusive": False,
                    "settlement_sources": [source],
                }
            },
            "series": {"series": {"ticker": native_series_id, "settlement_sources": [source]}},
        }
        normalized_rules = {
            "rules_primary": _CRITERIA,
            "rules_secondary": None,
            "open_time": _OPEN_TIME,
            "close_time": _CLOSE_TIME,
            "expiration_time": _EXPIRATION_TIME,
            "expected_expiration_time": None,
            "latest_expiration_time": None,
            "settlement_timer_seconds": 3600,
            "settlement_sources": [source],
            "void_cancel_rules": _VOID,
            "market_type": "binary",
            "mve_collection_ticker": None,
            "mve_selected_legs": [],
            "event_mutually_exclusive": False,
            "outcome_polarity": {"YES": "YES", "NO": "NO"},
        }
        requests = [
            _acquisition(venue, f"/markets/{native_market_id}", raw_response["market"]),
            _acquisition(venue, f"/events/{native_event_id}/metadata", raw_response["event_metadata"]),
            _acquisition(venue, f"/events/{native_event_id}", raw_response["event"]),
            _acquisition(venue, f"/series/{native_series_id}", raw_response["series"]),
        ]
        native_condition_id = None
        native_outcome_ids = {"YES": native_market_id, "NO": native_market_id}
        native_outcome_labels = {"YES": _YES, "NO": _NO}
        event_mutually_exclusive = False
        negative_risk = None
    else:
        native_market_id = f"PM_{case_id}"
        native_event_id = f"PME_{case_id}"
        native_condition_id = f"PMC_{case_id}"
        selected_market = {
            "id": native_market_id,
            "conditionId": native_condition_id,
            "question": _PROPOSITION,
            "description": _CRITERIA,
            "resolutionSource": _SOURCE,
            "resolvedBy": _AUTHORITY,
            "startDate": _OPEN_TIME,
            "endDate": _CLOSE_TIME,
            "umaEndDate": _EXPIRATION_TIME,
            "voidCancelRules": _VOID,
            "settlementDelaySeconds": 3600,
            "negRisk": False,
            "outcomes": ["Yes", "No"],
            "clobTokenIds": [f"{native_market_id}_yes", f"{native_market_id}_no"],
            "events": [{"id": native_event_id}],
        }
        markets = [selected_market]
        raw_response = {"markets": markets, "selected_market": selected_market}
        normalized_rules = {
            "question": _PROPOSITION,
            "description": _CRITERIA,
            "resolution_source": _SOURCE,
            "resolved_by": _AUTHORITY,
            "start_date": _OPEN_TIME,
            "end_date": _CLOSE_TIME,
            "uma_end_date": _EXPIRATION_TIME,
            "void_cancel_rules": _VOID,
            "settlement_delay_seconds": 3600.0,
            "neg_risk": False,
            "outcome_polarity": {"YES": "YES", "NO": "NO"},
        }
        requests = [_acquisition(venue, "/markets", markets)]
        native_series_id = None
        native_outcome_ids = {
            "YES": f"{native_market_id}_yes",
            "NO": f"{native_market_id}_no",
        }
        native_outcome_labels = {"YES": "Yes", "NO": "No"}
        event_mutually_exclusive = None
        negative_risk = False

    evidence = {
        "candidate_id": candidate["candidate_id"],
        "case_id": case_id,
        "side": side,
        "venue": venue,
        "pmxt_market_id": pmxt_market_id,
        "native_market_id": native_market_id,
        "native_event_id": native_event_id,
        "native_series_id": native_series_id,
        "native_condition_id": native_condition_id,
        "native_outcome_ids": native_outcome_ids,
        "native_outcome_labels": native_outcome_labels,
        "raw_sha256": _canonical_sha256(raw_response),
        "rule_hash": _canonical_sha256(normalized_rules),
        "proposition": _PROPOSITION,
        "outcome_polarity": {"YES": "YES", "NO": "NO"},
        "open_time": _OPEN_TIME,
        "close_time": _CLOSE_TIME,
        "expiration_time": _EXPIRATION_TIME,
        "settlement_authority": _AUTHORITY,
        "resolution_source": _SOURCE,
        "resolution_criteria": _CRITERIA,
        "void_cancel": _VOID,
        "material_edge_cases": _EDGE_CASES,
        "settlement_delay_seconds": 3600,
        "market_type": "binary",
        "mve_collection_ticker": None,
        "mve_selected_legs": [],
        "event_mutually_exclusive": event_mutually_exclusive,
        "negative_risk": negative_risk,
        "status": "RESOLVED",
        "requests": requests,
        "raw_response": raw_response,
        "normalized_rules": normalized_rules,
        "live_eligible": False,
    }
    return evidence


def _refresh_provenance(evidence: dict[str, Any]) -> None:
    if evidence["venue"] == "kalshi":
        payloads = [evidence["raw_response"][name] for name in ("market", "event_metadata", "event", "series")]
    else:
        payloads = [evidence["raw_response"]["markets"]]
    assert len(evidence["requests"]) == len(payloads)
    for request, payload in zip(evidence["requests"], payloads):
        raw_body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        request["raw_body_hash"] = hashlib.sha256(raw_body).hexdigest()
        request["raw_body_base64"] = base64.b64encode(raw_body).decode("ascii")
    evidence["raw_sha256"] = _canonical_sha256(evidence["raw_response"])
    evidence["rule_hash"] = _canonical_sha256(evidence["normalized_rules"])


def _mutate_polymarket(case_id: str, evidence: dict[str, Any]) -> None:
    selected = evidence["raw_response"]["selected_market"]
    if case_id == "review_inverted_polarity":
        evidence["outcome_polarity"] = {"YES": "NO", "NO": "YES"}
        evidence["normalized_rules"]["outcome_polarity"] = evidence["outcome_polarity"]
    elif case_id == "reject_deadline":
        deadline = "2026-11-04T03:00:00Z"
        evidence["close_time"] = deadline
        evidence["normalized_rules"]["end_date"] = deadline
        selected["endDate"] = deadline
    elif case_id == "reject_authority":
        authority = "A different certified authority"
        evidence["settlement_authority"] = authority
        evidence["normalized_rules"]["resolved_by"] = authority
        selected["resolvedBy"] = authority
    elif case_id == "reject_rule_source":
        source = "A news-network projection"
        evidence["resolution_source"] = source
        evidence["normalized_rules"]["resolution_source"] = source
        selected["resolutionSource"] = source
    _refresh_provenance(evidence)


def _byte_inventory(path: Path) -> dict[str, bytes]:
    return {item.relative_to(path).as_posix(): item.read_bytes() for item in sorted(path.rglob("*")) if item.is_file()}


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_offline_candidate_quality_replay_is_exact_immutable_and_non_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PMXT_API_KEY", raising=False)
    network_attempts: list[tuple[object, ...]] = []

    def forbid_network(*args: object, **kwargs: object) -> None:
        network_attempts.append((*args, kwargs))
        raise AssertionError("offline candidate-quality replay attempted network access")

    monkeypatch.setattr(socket, "socket", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)

    clusters = [_cluster(case_id) for case_id in _CASE_ORDER]
    result = normalize_clusters(
        clusters,
        query=PmxtQuery(),
        snapshot_id="offline_candidate_quality_fixture",
        observed_at=_OBSERVED_AT,
    )
    candidates = []
    for candidate in result.rows:
        candidate = dict(candidate)
        candidate["case_id"] = str(candidate["cluster_id"]).removeprefix("mcl_")
        candidates.append(candidate)
    candidate_rejections = []
    for rejection in result.rejected:
        rejection = dict(rejection)
        rejection["case_id"] = str(rejection["cluster_id"]).removeprefix("mcl_")
        candidate_rejections.append(rejection)

    assert [candidate["case_id"] for candidate in candidates] == [
        case_id for case_id in _CASE_ORDER if case_id != "reject_relation_overlap"
    ]
    assert len(candidate_rejections) == 1
    assert {key: candidate_rejections[0][key] for key in ("case_id", "reason", "relation", "live_eligible")} == {
        "case_id": "reject_relation_overlap",
        "reason": "non_identity_relation",
        "relation": "overlap",
        "live_eligible": False,
    }
    assert len({candidate["candidate_id"] for candidate in candidates}) == 6
    assert candidates[0]["venue_a"] == "kalshi"
    assert candidates[1]["venue_a"] == "polymarket"

    native_records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for candidate in candidates:
        case_id = str(candidate["case_id"])
        kalshi = _native(candidate, "kalshi")
        polymarket = _native(candidate, "polymarket")
        _mutate_polymarket(case_id, polymarket)
        native_records.extend((kalshi, polymarket))
        decision = verify_semantics(candidate, polymarket, kalshi)
        decisions.append({**decision, "case_id": case_id})

    assert {record["case_id"] for record in native_records} == set(_EXPECTED_SEMANTIC_RESULTS)
    assert "reject_relation_overlap" not in {record["case_id"] for record in native_records}
    assert {
        decision["case_id"]: (decision["status"], decision["reason_codes"]) for decision in decisions
    } == _EXPECTED_SEMANTIC_RESULTS
    assert sum(decision["status"] == VERIFIED_EQUIVALENT for decision in decisions) == 2
    assert sum(decision["status"] == REJECTED for decision in decisions) == 3
    assert sum(decision["status"] == NEEDS_REVIEW for decision in decisions) == 1

    assert len(candidates) == len(decisions)
    for candidate, decision in zip(candidates, decisions):
        native_by_venue = {item["venue"]: item for item in decision["evidence"]["native_markets"]}
        for venue in ("kalshi", "polymarket"):
            record = next(
                item for item in native_records if item["case_id"] == candidate["case_id"] and item["venue"] == venue
            )
            side = _candidate_side(candidate, venue)
            assert native_by_venue[venue]["side"] == side
            assert native_by_venue[venue]["pmxt_market_id"] == candidate[f"pmxt_market_id_{side}"]
            assert native_by_venue[venue]["native_market_id"] == record["native_market_id"]
            assert native_by_venue[venue]["raw_sha256"] == record["raw_sha256"]
            assert native_by_venue[venue]["rule_hash"] == record["rule_hash"]

    semantic_rejections = [decision for decision in decisions if decision["status"] == REJECTED]
    rejections = [*candidate_rejections, *semantic_rejections]
    native_books: list[dict[str, Any]] = []
    calculations: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    raw_pmxt = {
        "schema_version": 1,
        "source": "offline_fixture",
        "mode": "OFFLINE_REPLAY_FIXTURE",
        "observed_at": _OBSERVED_AT,
        "clusters": clusters,
        "live_eligible": False,
    }

    output_dir = tmp_path / "pmxt"
    run_id = "candidate-quality-replay"
    artifacts = persist_monitor_run(
        output_dir,
        run_id=run_id,
        status="OFFLINE_CANDIDATE_QUALITY_REPLAY_COMPLETE",
        config={
            "mode": "OFFLINE_REPLAY_FIXTURE",
            "relations": ["identity"],
            "venues": ["kalshi", "polymarket"],
            "stops_after": "semantic_verification",
        },
        counts={
            "fixture_proposals": 7,
            "candidate_gate_rejections": 1,
            "semantic_evaluations": 6,
            "verified_equivalent": 2,
            "semantic_rejections": 3,
            "needs_review": 1,
            "total_rejected": 4,
            "runtime_network_requests": 0,
            "native_book_attempts": 0,
            "shadow_calculation_attempts": 0,
            "orders_submitted": 0,
        },
        raw_pmxt=raw_pmxt,
        candidates=candidates,
        raw_native_metadata=native_records,
        semantic_decisions=decisions,
        rejections=rejections,
        native_books=native_books,
        calculations=calculations,
        alerts=alerts,
        provenance={
            "source": "OFFLINE_REPLAY_FIXTURE",
            "fixture_sha256": _canonical_sha256(clusters),
            "credentials_read": False,
            "live_eligible": False,
        },
    )

    assert network_attempts == []
    assert len(clusters) == 7
    assert len(candidates) == 6
    assert len(candidate_rejections) == 1
    assert len(native_records) == 12
    assert len(decisions) == 6
    assert len(rejections) == 4
    assert native_books == calculations == alerts == []

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "OFFLINE_CANDIDATE_QUALITY_REPLAY_COMPLETE"
    assert manifest["counts"] == {
        "fixture_proposals": 7,
        "candidate_gate_rejections": 1,
        "semantic_evaluations": 6,
        "verified_equivalent": 2,
        "semantic_rejections": 3,
        "needs_review": 1,
        "total_rejected": 4,
        "runtime_network_requests": 0,
        "native_book_attempts": 0,
        "shadow_calculation_attempts": 0,
        "orders_submitted": 0,
        "raw_pmxt": 1,
        "candidates": 6,
        "raw_native_metadata": 12,
        "semantic_decisions": 6,
        "rejections": 4,
        "native_books": 0,
        "calculations": 0,
        "alerts": 0,
    }
    assert manifest["live_eligible"] is False
    assert manifest["no_order_actions"] is True
    assert manifest["provenance"]["credentials_read"] is False
    assert artifacts.artifact_paths["native_books"].read_bytes() == b""
    assert artifacts.artifact_paths["calculations"].read_bytes() == b""
    assert artifacts.artifact_paths["alerts"].read_bytes() == b""

    for name, path in artifacts.artifact_paths.items():
        content = path.read_bytes()
        assert manifest["artifacts"][name] == {
            "path": path.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "byte_size": len(content),
        }
        rows = [json.loads(content)] if name == "raw_pmxt" else _jsonl_rows(path)
        assert all(row["live_eligible"] is False for row in rows)

    before = _byte_inventory(artifacts.run_dir)
    with pytest.raises(FileExistsError):
        persist_monitor_run(
            output_dir,
            run_id=run_id,
            status="MUST_NOT_OVERWRITE",
            raw_pmxt={"replacement": True, "live_eligible": False},
        )
    assert _byte_inventory(artifacts.run_dir) == before
