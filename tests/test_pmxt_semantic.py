"""Fixture-first tests for deterministic PMXT semantic verification."""

from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy

import pytest

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


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate(**overrides: object) -> dict:
    candidate = {
        "candidate_id": "pmxt_candidate_fixture",
        "cluster_id": "mcl_fixture",
        "relation": "identity",
        "raw_edge_present": True,
        "venue_a": "kalshi",
        "venue_b": "polymarket",
        "pmxt_market_id_a": "pmxt_kalshi_discovery_id",
        "pmxt_market_id_b": "pmxt_polymarket_discovery_id",
    }
    candidate.update(overrides)
    return candidate


def _acquisition(venue: str, path: str, payload: object) -> dict:
    raw_body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    http_date = "Sat, 29 Aug 2026 12:00:00 GMT"
    cache_control = "no-store"
    request_id = f"fixture-{venue}"
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


def _native(venue: str) -> dict:
    if venue == "kalshi":
        market = {
            "ticker": "kalshi_native_001",
            "event_ticker": "kalshi_event_001",
            "series_ticker": "kalshi_series_001",
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
                    "event_ticker": "kalshi_event_001",
                    "series_ticker": "kalshi_series_001",
                    "mutually_exclusive": False,
                    "settlement_sources": [source],
                }
            },
            "series": {"series": {"ticker": "kalshi_series_001", "settlement_sources": [source]}},
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
            _acquisition(venue, "/markets/kalshi_native_001", raw_response["market"]),
            _acquisition(venue, "/events/kalshi_event_001/metadata", raw_response["event_metadata"]),
            _acquisition(venue, "/events/kalshi_event_001", raw_response["event"]),
            _acquisition(venue, "/series/kalshi_series_001", raw_response["series"]),
        ]
        side = "a"
        pmxt_market_id = "pmxt_kalshi_discovery_id"
    else:
        selected_market = {
            "id": "polymarket_native_001",
            "conditionId": "polymarket_condition_001",
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
            "clobTokenIds": ["polymarket_yes", "polymarket_no"],
            "events": [{"id": "polymarket_event_001"}],
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
        side = "b"
        pmxt_market_id = "pmxt_polymarket_discovery_id"
    return {
        "candidate_id": "pmxt_candidate_fixture",
        "side": side,
        "venue": venue,
        "pmxt_market_id": pmxt_market_id,
        "native_market_id": f"{venue}_native_001",
        "native_event_id": f"{venue}_event_001",
        "native_series_id": "kalshi_series_001" if venue == "kalshi" else None,
        "native_condition_id": "polymarket_condition_001" if venue == "polymarket" else None,
        "native_outcome_ids": (
            {"YES": "kalshi_native_001", "NO": "kalshi_native_001"}
            if venue == "kalshi"
            else {"YES": "polymarket_yes", "NO": "polymarket_no"}
        ),
        "native_outcome_labels": {
            "YES": _YES if venue == "kalshi" else "Yes",
            "NO": _NO if venue == "kalshi" else "No",
        },
        "raw_sha256": _canonical_sha256(raw_response),
        "rule_hash": _canonical_sha256(normalized_rules),
        "proposition": _PROPOSITION,
        "outcome_polarity": {
            "YES": "YES",
            "NO": "NO",
        },
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
        "mve_collection_ticker": None if venue == "kalshi" else None,
        "mve_selected_legs": [] if venue == "kalshi" else None,
        "event_mutually_exclusive": False if venue == "kalshi" else None,
        "negative_risk": False if venue == "polymarket" else None,
        "status": "RESOLVED",
        "requests": requests,
        "raw_response": raw_response,
        "normalized_rules": normalized_rules,
    }


def _refresh_provenance(evidence: dict) -> None:
    if evidence["venue"] == "kalshi":
        payloads = [evidence["raw_response"][name] for name in ("market", "event_metadata", "event", "series")]
    else:
        payloads = [evidence["raw_response"]["markets"]]
    for request, payload in zip(evidence["requests"], payloads):
        raw_body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        request["raw_body_hash"] = hashlib.sha256(raw_body).hexdigest()
        request["raw_body_base64"] = base64.b64encode(raw_body).decode("ascii")
    evidence["raw_sha256"] = _canonical_sha256(evidence["raw_response"])
    evidence["rule_hash"] = _canonical_sha256(evidence["normalized_rules"])


def test_exact_native_semantics_are_verified_but_never_live_eligible() -> None:
    polymarket = _native("polymarket")
    kalshi = _native("kalshi")
    decision = verify_semantics(_candidate(), polymarket, kalshi)

    assert decision["status"] == VERIFIED_EQUIVALENT
    assert decision["live_eligible"] is False
    assert decision["reason_codes"] == ["ALL_REQUIRED_SEMANTICS_EQUIVALENT"]
    assert [item["venue"] for item in decision["evidence"]["native_markets"]] == ["kalshi", "polymarket"]
    assert [item["native_market_id"] for item in decision["evidence"]["native_markets"]] == [
        "kalshi_native_001",
        "polymarket_native_001",
    ]
    assert decision["evidence"]["native_markets"][0]["raw_sha256"] == kalshi["raw_sha256"]
    assert decision["evidence"]["native_markets"][0]["rule_hash"] == kalshi["rule_hash"]
    assert decision["evidence"]["native_markets"][0]["side"] == "a"
    assert decision["evidence"]["native_markets"][0]["pmxt_market_id"] == "pmxt_kalshi_discovery_id"
    assert decision["evidence"]["native_markets"][0]["native_outcome_ids"] == {
        "no": "kalshi_native_001",
        "yes": "kalshi_native_001",
    }
    assert decision["evidence"]["pmxt_discovery"]["pmxt_ids_are_native_authority"] is False


@pytest.mark.parametrize("venue", ["kalshi", "polymarket"])
def test_inverted_derived_polarity_without_matching_raw_evidence_needs_review(venue: str) -> None:
    evidence = _native(venue)
    polarity = evidence["outcome_polarity"]
    evidence["outcome_polarity"] = {"YES": polarity["NO"], "NO": polarity["YES"]}
    evidence["normalized_rules"]["outcome_polarity"] = evidence["outcome_polarity"]
    _refresh_provenance(evidence)
    other = _native("polymarket" if venue == "kalshi" else "kalshi")

    decision = verify_semantics(_candidate(), evidence, other)

    assert decision["status"] == NEEDS_REVIEW
    assert decision["reason_codes"] == sorted(
        [
            f"{venue.upper()}_NORMALIZED_RULES_RAW_OUTCOME_POLARITY_MISMATCH",
            "OUTCOME_POLARITY_INVERTED_NON_IDENTITY",
        ]
    )
    assert decision["live_eligible"] is False


def test_deadline_mismatch_is_rejected() -> None:
    polymarket = _native("polymarket")
    polymarket["close_time"] = "2026-11-04T03:00:00Z"
    polymarket["normalized_rules"]["end_date"] = polymarket["close_time"]
    polymarket["raw_response"]["selected_market"]["endDate"] = polymarket["close_time"]
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == REJECTED
    assert "DEADLINE_MISMATCH" in decision["reason_codes"]
    assert decision["live_eligible"] is False


def test_timezone_offset_mismatch_is_rejected_even_for_same_instant() -> None:
    polymarket = _native("polymarket")
    polymarket["close_time"] = "2026-11-03T21:00:00-05:00"
    polymarket["normalized_rules"]["end_date"] = polymarket["close_time"]
    polymarket["raw_response"]["selected_market"]["endDate"] = polymarket["close_time"]
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == REJECTED
    assert "TIMEZONE_MISMATCH" in decision["reason_codes"]
    assert "DEADLINE_MISMATCH" not in decision["reason_codes"]


def test_differing_provenance_bound_expiration_times_are_rejected() -> None:
    polymarket = _native("polymarket")
    polymarket["expiration_time"] = "2026-11-04T05:00:00Z"
    polymarket["normalized_rules"]["uma_end_date"] = polymarket["expiration_time"]
    polymarket["raw_response"]["selected_market"]["umaEndDate"] = polymarket["expiration_time"]
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == REJECTED
    assert "EXPIRATION_TIME_MISMATCH" in decision["reason_codes"]
    assert decision["evidence"]["comparisons"]["expiration_time"]["equivalent"] is False
    assert decision["live_eligible"] is False


def test_top_level_expiration_cannot_escape_native_rule_provenance() -> None:
    polymarket = _native("polymarket")
    polymarket["expiration_time"] = "2026-11-04T05:00:00Z"

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert "POLYMARKET_TOP_LEVEL_EXPIRATION_TIME_NORMALIZED_RULES_MISMATCH" in decision["reason_codes"]
    assert decision["live_eligible"] is False


def test_close_time_alone_cannot_prove_resolution_horizon_equivalence() -> None:
    kalshi = _native("kalshi")
    polymarket = _native("polymarket")
    del kalshi["expiration_time"]
    del kalshi["normalized_rules"]["expiration_time"]
    del kalshi["raw_response"]["market"]["market"]["expiration_time"]
    del polymarket["expiration_time"]
    del polymarket["normalized_rules"]["uma_end_date"]
    del polymarket["raw_response"]["selected_market"]["umaEndDate"]
    _refresh_provenance(kalshi)
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), kalshi, polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert "EXPIRATION_TIME_EVIDENCE_MISSING" in decision["reason_codes"]
    assert decision["evidence"]["comparisons"]["close_time"]["equivalent"] is True
    assert decision["live_eligible"] is False


def test_one_sided_expiration_horizon_needs_review() -> None:
    polymarket = _native("polymarket")
    del polymarket["expiration_time"]
    del polymarket["normalized_rules"]["uma_end_date"]
    del polymarket["raw_response"]["selected_market"]["umaEndDate"]
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert "EXPIRATION_TIME_EVIDENCE_ONE_SIDED" in decision["reason_codes"]
    assert decision["evidence"]["comparisons"]["expiration_time"]["evidence_complete"] is False
    assert decision["live_eligible"] is False


def test_one_sided_expected_expiration_horizon_needs_review() -> None:
    kalshi = _native("kalshi")
    kalshi["expected_expiration_time"] = "2026-11-04T03:00:00Z"
    kalshi["normalized_rules"]["expected_expiration_time"] = kalshi["expected_expiration_time"]
    kalshi["raw_response"]["market"]["market"]["expected_expiration_time"] = kalshi["expected_expiration_time"]
    _refresh_provenance(kalshi)

    decision = verify_semantics(_candidate(), kalshi, _native("polymarket"))

    assert decision["status"] == NEEDS_REVIEW
    assert "EXPECTED_EXPIRATION_TIME_EVIDENCE_ONE_SIDED" in decision["reason_codes"]
    assert decision["live_eligible"] is False


def test_differing_resolution_sources_are_rejected() -> None:
    polymarket = _native("polymarket")
    polymarket["resolution_source"] = "A news-network projection"
    polymarket["normalized_rules"]["resolution_source"] = polymarket["resolution_source"]
    polymarket["raw_response"]["selected_market"]["resolutionSource"] = polymarket["resolution_source"]
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == REJECTED
    assert "RESOLUTION_SOURCE_MISMATCH" in decision["reason_codes"]


def test_multiple_provenance_bound_native_mismatches_accumulate_reason_codes() -> None:
    polymarket = _native("polymarket")
    polymarket["close_time"] = "2026-11-04T03:00:00Z"
    polymarket["settlement_authority"] = "A different certified authority"
    polymarket["resolution_source"] = "A different official result"
    polymarket["normalized_rules"].update(
        {
            "end_date": polymarket["close_time"],
            "resolved_by": polymarket["settlement_authority"],
            "resolution_source": polymarket["resolution_source"],
        }
    )
    polymarket["raw_response"]["selected_market"].update(
        {
            "endDate": polymarket["close_time"],
            "resolvedBy": polymarket["settlement_authority"],
            "resolutionSource": polymarket["resolution_source"],
        }
    )
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == REJECTED
    assert {
        "DEADLINE_MISMATCH",
        "SETTLEMENT_AUTHORITY_MISMATCH",
        "RESOLUTION_SOURCE_MISMATCH",
    } <= set(decision["reason_codes"])
    assert decision["live_eligible"] is False


def test_apparent_mismatch_with_tampered_provenance_needs_review() -> None:
    polymarket = _native("polymarket")
    polymarket["resolution_source"] = "A different official result"
    polymarket["normalized_rules"]["resolution_source"] = polymarket["resolution_source"]
    polymarket["raw_response"]["selected_market"]["resolutionSource"] = polymarket["resolution_source"]
    _refresh_provenance(polymarket)
    polymarket["raw_sha256"] = "0" * 64

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert decision["status"] != REJECTED
    assert "POLYMARKET_RAW_SHA256_MISMATCH" in decision["reason_codes"]
    assert "RESOLUTION_SOURCE_MISMATCH" in decision["reason_codes"]
    assert decision["live_eligible"] is False


def test_missing_native_fields_need_review_without_pmxt_id_fallback() -> None:
    kalshi = _native("kalshi")
    del kalshi["native_market_id"]
    del kalshi["rule_hash"]

    decision = verify_semantics(_candidate(), kalshi, _native("polymarket"))

    assert decision["status"] == NEEDS_REVIEW
    assert "KALSHI_MISSING_NATIVE_MARKET_ID" in decision["reason_codes"]
    assert "KALSHI_MISSING_RULE_HASH" in decision["reason_codes"]
    assert decision["evidence"]["native_markets"][0]["native_market_id"] is None
    assert decision["evidence"]["native_markets"][0]["native_market_id"] != "pmxt_kalshi_discovery_id"


def test_tampered_native_raw_response_hash_needs_review() -> None:
    kalshi = _native("kalshi")
    kalshi["raw_response"]["fixture"] = False

    decision = verify_semantics(_candidate(), kalshi, _native("polymarket"))

    assert decision["status"] == NEEDS_REVIEW
    assert "KALSHI_RAW_SHA256_MISMATCH" in decision["reason_codes"]
    assert decision["live_eligible"] is False


def test_tampered_normalized_rule_hash_needs_review() -> None:
    polymarket = _native("polymarket")
    polymarket["normalized_rules"]["resolution_source"] = "tampered source"

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert "POLYMARKET_RULE_HASH_MISMATCH" in decision["reason_codes"]
    assert decision["live_eligible"] is False


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("candidate_id", "another_candidate", "KALSHI_CANDIDATE_ID_BINDING_MISMATCH"),
        ("side", "b", "KALSHI_CANDIDATE_SIDE_VENUE_BINDING_MISMATCH"),
        ("pmxt_market_id", "another_pmxt_id", "KALSHI_PMXT_MARKET_ID_BINDING_MISMATCH"),
    ],
)
def test_tampered_candidate_side_binding_needs_review(field: str, value: object, reason: str) -> None:
    kalshi = _native("kalshi")
    kalshi[field] = value

    decision = verify_semantics(_candidate(), kalshi, _native("polymarket"))

    assert decision["status"] == NEEDS_REVIEW
    assert reason in decision["reason_codes"]
    assert decision["live_eligible"] is False


def test_decoded_acquisition_bytes_must_equal_the_retained_raw_response() -> None:
    kalshi = _native("kalshi")
    different_payload = {"market": {"ticker": "DIFFERENT-NATIVE-MARKET"}}
    raw_body = json.dumps(different_payload, separators=(",", ":")).encode()
    kalshi["requests"][0]["raw_body_base64"] = base64.b64encode(raw_body).decode("ascii")
    kalshi["requests"][0]["raw_body_hash"] = hashlib.sha256(raw_body).hexdigest()

    decision = verify_semantics(_candidate(), kalshi, _native("polymarket"))

    assert decision["status"] == NEEDS_REVIEW
    assert "KALSHI_MARKET_RAW_BODY_RESPONSE_MISMATCH" in decision["reason_codes"]
    assert "KALSHI_ACQUISITION_0_RAW_BODY_HASH_MISMATCH" not in decision["reason_codes"]


def test_polymarket_selected_market_must_be_an_exact_returned_record() -> None:
    polymarket = _native("polymarket")
    polymarket["raw_response"]["selected_market"] = {
        **polymarket["raw_response"]["selected_market"],
        "id": "not-in-returned-records",
    }
    polymarket["raw_sha256"] = _canonical_sha256(polymarket["raw_response"])

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert "POLYMARKET_SELECTED_MARKET_NOT_IN_RAW_RESPONSE" in decision["reason_codes"]


def test_top_level_semantics_must_match_the_normalized_rules() -> None:
    polymarket = _native("polymarket")
    polymarket["settlement_authority"] = "Unbound top-level authority"

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert "POLYMARKET_TOP_LEVEL_SETTLEMENT_AUTHORITY_NORMALIZED_RULES_MISMATCH" in decision["reason_codes"]


@pytest.mark.parametrize(
    ("top_field", "rule_field", "raw_field", "value", "reason"),
    [
        (
            "settlement_authority",
            "resolved_by",
            "resolvedBy",
            "A different settlement authority",
            "SETTLEMENT_AUTHORITY_MISMATCH",
        ),
        (
            "resolution_criteria",
            "description",
            "description",
            "A materially different criterion",
            "RESOLUTION_CRITERIA_MISMATCH",
        ),
        (
            "void_cancel",
            "void_cancel_rules",
            "voidCancelRules",
            "Never void under any circumstances",
            "VOID_CANCEL_MISMATCH",
        ),
        (
            "settlement_delay_seconds",
            "settlement_delay_seconds",
            "settlementDelaySeconds",
            7200,
            "SETTLEMENT_DELAY_MISMATCH",
        ),
    ],
)
def test_internally_bound_material_semantic_mismatches_are_rejected(
    top_field: str,
    rule_field: str,
    raw_field: str,
    value: object,
    reason: str,
) -> None:
    polymarket = _native("polymarket")
    polymarket[top_field] = value
    polymarket["normalized_rules"][rule_field] = float(value) if top_field == "settlement_delay_seconds" else value
    polymarket["raw_response"]["selected_market"][raw_field] = value
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == REJECTED
    assert reason in decision["reason_codes"]


def test_unbound_material_edge_case_mismatch_needs_review() -> None:
    polymarket = _native("polymarket")
    polymarket["material_edge_cases"] = {"venue_specific_edge_case": True}

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert decision["reason_codes"] == [
        "MATERIAL_EDGE_CASES_EQUIVALENCE_UNPROVEN",
        "POLYMARKET_TOP_LEVEL_MATERIAL_EDGE_CASES_NORMALIZED_RULES_MISMATCH",
    ]
    assert decision["live_eligible"] is False


@pytest.mark.parametrize(
    ("venue", "rule_field", "value", "reason"),
    [
        (
            "kalshi",
            "can_close_early",
            True,
            "KALSHI_NORMALIZED_RULES_RAW_CAN_CLOSE_EARLY_MISMATCH",
        ),
        (
            "polymarket",
            "uma_resolution_status",
            "disputed",
            "POLYMARKET_NORMALIZED_RULES_RAW_UMA_RESOLUTION_STATUS_MISMATCH",
        ),
    ],
)
def test_material_edge_case_normalization_cannot_escape_retained_raw_rules(
    venue: str,
    rule_field: str,
    value: object,
    reason: str,
) -> None:
    evidence = _native(venue)
    evidence["normalized_rules"][rule_field] = value
    evidence["material_edge_cases"] = {rule_field: value}
    _refresh_provenance(evidence)
    other = _native("polymarket" if venue == "kalshi" else "kalshi")

    decision = verify_semantics(_candidate(), evidence, other)

    assert decision["status"] == NEEDS_REVIEW
    assert decision["reason_codes"] == sorted([reason, "MATERIAL_EDGE_CASES_EQUIVALENCE_UNPROVEN"])
    assert decision["live_eligible"] is False


def test_raw_bound_material_edge_case_difference_needs_review_not_rejection() -> None:
    polymarket = _native("polymarket")
    polymarket["normalized_rules"]["uma_resolution_status"] = "disputed"
    polymarket["raw_response"]["selected_market"]["umaResolutionStatus"] = "disputed"
    polymarket["material_edge_cases"] = {"uma_resolution_status": "disputed"}
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert decision["reason_codes"] == ["MATERIAL_EDGE_CASES_EQUIVALENCE_UNPROVEN"]
    assert decision["live_eligible"] is False


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("market_type", "scalar", "KALSHI_MARKET_TYPE_NOT_BINARY"),
        ("mve_collection_ticker", "KXMVE-FIXTURE", "KALSHI_MULTIVARIATE_MARKET_UNSUPPORTED"),
    ],
)
def test_non_binary_or_multivariate_kalshi_market_is_rejected(field: str, value: object, reason: str) -> None:
    kalshi = _native("kalshi")
    kalshi[field] = value
    kalshi["normalized_rules"][field] = value
    kalshi["raw_response"]["market"]["market"][field] = value
    _refresh_provenance(kalshi)

    decision = verify_semantics(_candidate(), kalshi, _native("polymarket"))

    assert decision["status"] == REJECTED
    assert reason in decision["reason_codes"]
    assert decision["live_eligible"] is False


def test_mismatched_native_group_structure_is_rejected() -> None:
    polymarket = _native("polymarket")
    polymarket["negative_risk"] = True
    polymarket["normalized_rules"]["neg_risk"] = True
    polymarket["raw_response"]["selected_market"]["negRisk"] = True
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == REJECTED
    assert "MARKET_GROUP_STRUCTURE_MISMATCH" in decision["reason_codes"]
    assert decision["live_eligible"] is False


def test_matching_group_flags_do_not_prove_candidate_set_or_other_path_equivalence() -> None:
    kalshi = _native("kalshi")
    kalshi["event_mutually_exclusive"] = True
    kalshi["normalized_rules"]["event_mutually_exclusive"] = True
    kalshi["raw_response"]["event"]["event"]["mutually_exclusive"] = True
    _refresh_provenance(kalshi)
    polymarket = _native("polymarket")
    polymarket["negative_risk"] = True
    polymarket["normalized_rules"]["neg_risk"] = True
    polymarket["raw_response"]["selected_market"]["negRisk"] = True
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), kalshi, polymarket)

    comparison = decision["evidence"]["comparisons"]["market_group_structure"]
    assert decision["status"] == NEEDS_REVIEW
    assert "MARKET_GROUP_CANDIDATE_SET_EQUIVALENCE_UNPROVEN" in decision["reason_codes"]
    assert comparison["flags_match"] is True
    assert comparison["candidate_set_equivalence_proven"] is False
    assert comparison["equivalent"] is False
    assert decision["live_eligible"] is False


def test_missing_native_group_structure_needs_review() -> None:
    polymarket = _native("polymarket")
    del polymarket["negative_risk"]

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert "POLYMARKET_MISSING_NEGATIVE_RISK" in decision["reason_codes"]
    assert decision["live_eligible"] is False


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"relation": "overlap"}, "PMXT_RELATION_NOT_IDENTITY"),
        ({"raw_edge_present": False}, "PMXT_RAW_EDGE_MISSING"),
        ({"venue_b": "manifold"}, "PMXT_VENUE_PAIR_UNSUPPORTED"),
    ],
)
def test_nonidentity_missing_edge_or_wrong_venues_are_rejected(overrides: dict, reason: str) -> None:
    decision = verify_semantics(_candidate(**overrides), _native("kalshi"), _native("polymarket"))

    assert decision["status"] == REJECTED
    assert reason in decision["reason_codes"]
    assert decision["live_eligible"] is False


def test_proposition_mismatch_and_non_yes_no_polarity_need_review() -> None:
    polymarket = deepcopy(_native("polymarket"))
    polymarket["proposition"] = "Will Candidate Y win the 2026 test election?"
    polymarket["outcome_polarity"] = {"UP": "Candidate Y wins", "DOWN": "Candidate Y loses"}
    polymarket["normalized_rules"]["question"] = polymarket["proposition"]
    polymarket["normalized_rules"]["outcome_polarity"] = polymarket["outcome_polarity"]
    polymarket["raw_response"]["selected_market"]["question"] = polymarket["proposition"]
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert "PROPOSITION_MISMATCH" in decision["reason_codes"]
    assert "POLYMARKET_OUTCOME_POLARITY_NOT_EXPLICIT_YES_NO" in decision["reason_codes"]


def test_naive_close_time_needs_review() -> None:
    polymarket = _native("polymarket")
    polymarket["close_time"] = "2026-11-04T02:00:00"
    polymarket["normalized_rules"]["end_date"] = polymarket["close_time"]
    polymarket["raw_response"]["selected_market"]["endDate"] = polymarket["close_time"]
    _refresh_provenance(polymarket)

    decision = verify_semantics(_candidate(), _native("kalshi"), polymarket)

    assert decision["status"] == NEEDS_REVIEW
    assert "POLYMARKET_CLOSE_TIME_NOT_TIMEZONE_AWARE" in decision["reason_codes"]


def test_whole_unresolved_native_leg_cannot_be_filled_from_pmxt_catalog_fields() -> None:
    candidate = _candidate(
        title_a=_PROPOSITION,
        description_a=_CRITERIA,
        resolution_date_a=_EXPIRATION_TIME,
        source_metadata_a={
            "ticker": "pmxt_kalshi_discovery_id",
            "close_time": _CLOSE_TIME,
            "expiration_time": _EXPIRATION_TIME,
        },
    )
    unresolved_leg = {
        "candidate_id": candidate["candidate_id"],
        "side": "a",
        "venue": "kalshi",
        "pmxt_market_id": candidate["pmxt_market_id_a"],
        "native_market_id": None,
        "raw_sha256": None,
        "rule_hash": None,
        "requests": [],
        "raw_response": None,
        "proposition": None,
        "outcome_polarity": None,
        "close_time": None,
        "expiration_time": None,
        "settlement_authority": None,
        "resolution_source": None,
        "resolution_criteria": None,
        "void_cancel": None,
        "material_edge_cases": None,
        "settlement_delay_seconds": None,
        "status": "ERROR",
        "reason_code": "NATIVE_ID_UNRESOLVED",
        "live_eligible": False,
    }

    decision = verify_semantics(candidate, unresolved_leg, _native("polymarket"))

    kalshi_evidence = decision["evidence"]["native_markets"][0]
    assert decision["status"] == NEEDS_REVIEW
    assert "KALSHI_NATIVE_STATUS_NOT_RESOLVED" in decision["reason_codes"]
    assert kalshi_evidence["status"] == "ERROR"
    assert kalshi_evidence["reason_code"] == "NATIVE_ID_UNRESOLVED"
    assert kalshi_evidence["native_market_id"] is None
    assert kalshi_evidence["proposition"] is None
    assert kalshi_evidence["close_time"] is None
    assert kalshi_evidence["expiration_time"] is None
    assert decision["evidence"]["pmxt_discovery"]["pmxt_ids_are_native_authority"] is False
    assert decision["live_eligible"] is False
