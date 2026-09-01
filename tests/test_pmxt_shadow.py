"""Fixture-first tests for the fail-closed PMXT shadow calculator."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from src.indexers.pmxt import shadow as shadow_module
from src.indexers.pmxt.shadow import calculate_shadow

EVALUATED_AT = datetime(2026, 8, 27, 15, 0, 0, tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fee_evidence(venue: str) -> dict:
    common = {
        "status": "VALID",
        "venue": venue,
        "liquidity_role": "TAKER",
        "passive_fills_assumed": False,
        "schedule_observed_at": _iso(EVALUATED_AT - timedelta(seconds=3)),
    }
    if venue == "kalshi":
        schedule = {
            "official_fee_schedule": {
                "status": "REVIEWED",
                "raw_body_sha256": "1" * 64,
                "formula_binding": {
                    "binding_id": "kalshi-prediction-fees-2026-07-07",
                    "taker_base_coefficient": "0.07",
                    "trade_fee_rounding_quantum": "0.000001",
                    "balance_precision_upper_bound": "0.01",
                },
            },
            "effective": {
                "fee_type": "quadratic",
                "fee_multiplier": 1,
                "taker_base_coefficient": "0.07",
                "trade_fee_rounding_quantum": "0.000001",
                "balance_precision_upper_bound": "0.01",
            },
        }
        raw_response = {
            "series": {"fee_type": "quadratic", "fee_multiplier": 1},
            "series_fee_changes": [],
            "event_fee_changes": [],
        }
        return {
            **common,
            "model": "KALSHI_QUADRATIC_TAKER",
            "effective_from": "2026-07-07T00:00:00Z",
            "official_fee_schedule_sha256": "1" * 64,
            "formula_binding_id": "kalshi-prediction-fees-2026-07-07",
            "calculation_class": "CONSERVATIVE_UPPER_BOUND",
            "exact_fee_claimed": False,
            "account_fee_accumulator_inputs": "NOT_AVAILABLE",
            "account_class_inputs": "NOT_AVAILABLE",
            "schedule_sha256": _canonical_sha256(schedule),
            "raw_sha256": _canonical_sha256(raw_response),
            "schedule": schedule,
            "raw_response": raw_response,
        }
    schedule = {
        "native_condition_id": "0x" + "2" * 64,
        "native_outcome_ids": {"YES": "1001", "NO": "1002"},
        "fees_enabled": True,
        "base_fee_bps_estimate_by_outcome": {"YES": 500, "NO": 500},
        "condition_fee_details": None,
    }
    raw_response = {"YES": {"base_fee": 500}, "NO": {"base_fee": 500}}
    return {
        **common,
        "status": "FAIL_CLOSED",
        "model": "POLYMARKET_FEE_ESTIMATE_ONLY",
        "estimate_formula": "shares * (base_fee_bps / 10000) * price * (1 - price)",
        "estimate_only": True,
        "effective_from": None,
        "effective_basis": "UNAVAILABLE",
        "reason_codes": [
            "POLYMARKET_CONDITION_FEE_PARAMETERS_NOT_CAPTURED",
            "POLYMARKET_FEE_EFFECTIVE_TIMESTAMP_UNAVAILABLE",
            "POLYMARKET_TOKEN_BASE_FEE_IS_ESTIMATE_ONLY",
        ],
        "schedule_sha256": _canonical_sha256(schedule),
        "raw_sha256": _canonical_sha256(raw_response),
        "schedule": schedule,
        "raw_response": raw_response,
    }


def _book(
    venue: str,
    market_id: str,
    *,
    source_offset_seconds: int = -2,
    yes_asks: list[dict[str, float]] | None = None,
    no_asks: list[dict[str, float]] | None = None,
) -> dict:
    return {
        "candidate_id": "pmxt_candidate_test",
        "venue": venue,
        "native_market_id": market_id,
        "market_status": "active",
        "book_eligible": True,
        "minimum_order_size": 1.0,
        "size_increment": 1.0,
        "request_started_at": _iso(EVALUATED_AT - timedelta(seconds=1)),
        "received_at": _iso(EVALUATED_AT - timedelta(milliseconds=100)),
        "request_monotonic_ns": 1_000_000_000,
        "response_monotonic_ns": 1_900_000_000,
        "rtt_ms": 900.0,
        "source_timestamp": _iso(EVALUATED_AT + timedelta(seconds=source_offset_seconds)),
        "as_of": _iso(EVALUATED_AT + timedelta(seconds=source_offset_seconds)),
        "freshness_basis": "VENUE_SOURCE_TIMESTAMP",
        "raw_sha256": ("a" if venue == "kalshi" else "b") * 64,
        "fee_evidence": _fee_evidence(venue),
        "sides": {
            "YES": {
                "bids": [{"price": 0.40, "size": 50.0}],
                "asks": yes_asks or [{"price": 0.42, "size": 50.0}],
            },
            "NO": {
                "bids": [{"price": 0.40, "size": 50.0}],
                "asks": no_asks or [{"price": 0.60, "size": 50.0}],
            },
        },
    }


def _decision(status: str = "VERIFIED_EQUIVALENT") -> dict:
    return {
        "schema_version": 1,
        "candidate_id": "pmxt_candidate_test",
        "status": status,
        "live_eligible": False,
        "reason_codes": [],
        "evidence": {
            "native_markets": [
                {"venue": "kalshi", "native_market_id": "KX-TEST"},
                {"venue": "polymarket", "native_market_id": "PM-TEST"},
            ]
        },
    }


def _policy(**overrides: object) -> dict:
    policy = {
        "requested_size": 10.0,
        "max_book_age_seconds": 30.0,
        "max_cross_venue_skew_seconds": 3.0,
        "annual_capital_rate": 0.05,
        "capital_lock_days": 2.0,
        "explicit_slippage_buffer_per_unit": 0.0,
        "timestamp_skew_buffer_per_unit": 0.0,
        "settlement_divergence_buffer_per_unit": 0.0,
        "collateral_basis_buffer_per_unit": 0.0,
        "rebalancing_withdrawal_allowance_per_unit": 0.0,
        "net_residual_threshold": 0.05,
    }
    policy.update(overrides)
    return policy


def _calculate(book_a: dict, book_b: dict, policy: dict | None = None) -> dict:
    return calculate_shadow(
        _decision(),
        book_a,
        book_b,
        policy or _policy(),
        evaluated_at=_iso(EVALUATED_AT),
    )


def _kalshi_book_validation(
    book: dict,
    *,
    label: str = "A",
    policy: dict | None = None,
) -> tuple[object | None, list[str], dict]:
    policy_values, policy_reasons = shadow_module._policy_values(policy or _policy())
    assert policy_values is not None
    assert policy_reasons == []
    return shadow_module._book_values(
        book,
        label=label,
        evaluated_at=EVALUATED_AT,
        policy=policy_values,
    )


def _unvalidated_arithmetic_book(book: dict, *, label: str) -> object:
    """Build the private value object solely to isolate fee/cost arithmetic."""

    return shadow_module._BookValues(
        label=label,
        candidate_id=str(book["candidate_id"]),
        venue=str(book["venue"]),
        native_market_id=str(book["native_market_id"]),
        source_at=EVALUATED_AT - timedelta(seconds=2),
        fee_evidence=deepcopy(book["fee_evidence"]),
        sides=deepcopy(book["sides"]),
        evidence={"test_scope": "ARITHMETIC_ONLY_UNVALIDATED"},
    )


def _kalshi_direction(
    book_a: dict,
    book_b: dict,
    policy: dict | None = None,
    *,
    outcome_a: str = "YES",
    outcome_b: str = "NO",
) -> dict:
    """Exercise fee/cost arithmetic without inventing executable Polymarket evidence."""

    policy_values, policy_reasons = shadow_module._policy_values(policy or _policy())
    assert policy_values is not None
    assert policy_reasons == []
    parsed_a = _unvalidated_arithmetic_book(book_a, label="A")
    parsed_b = _unvalidated_arithmetic_book(book_b, label="B")
    return shadow_module._calculate_direction(
        "A_YES_B_NO",
        outcome_a,
        outcome_b,
        parsed_a,
        parsed_b,
        policy_values,
    )


def test_stale_snapshot_fails_closed_before_arithmetic() -> None:
    parsed, reasons, _ = _kalshi_book_validation(_book("kalshi", "KX-TEST", source_offset_seconds=-31))

    assert parsed is None
    assert "BOOK_A_STALE_SOURCE_TIMESTAMP" in reasons


def test_slow_capture_window_and_outcome_book_skew_fail_closed() -> None:
    book = _book("kalshi", "KX-TEST")
    book["request_started_at"] = _iso(EVALUATED_AT - timedelta(seconds=31))
    book["request_monotonic_ns"] = 1_000_000_000
    book["response_monotonic_ns"] = 32_000_000_000
    book["rtt_ms"] = 31_000.0
    book["source_timestamps"] = {
        "YES": _iso(EVALUATED_AT - timedelta(seconds=2)),
        "NO": _iso(EVALUATED_AT - timedelta(seconds=8)),
    }

    parsed, reasons, _ = _kalshi_book_validation(book)

    assert parsed is None
    assert "BOOK_A_CAPTURE_WINDOW_EXCEEDS_AGE_LIMIT" in reasons
    assert "BOOK_A_OUTCOME_SOURCE_SKEW_EXCEEDS_LIMIT" in reasons


def test_local_receipt_basis_never_satisfies_strict_book_freshness() -> None:
    book = _book("kalshi", "KX-TEST")
    book["freshness_basis"] = "LOCAL_RECEIPT_BOUNDED"

    parsed, reasons, _ = _kalshi_book_validation(book)

    assert parsed is None
    assert "BOOK_A_FRESHNESS_BASIS_NOT_VENUE_SOURCE_TIMESTAMP" in reasons


def test_shallow_depth_never_produces_a_partial_size_alert() -> None:
    book_a = _book("kalshi", "KX-A", yes_asks=[{"price": 0.42, "size": 5.0}])
    book_b = _book("kalshi", "KX-B")

    direction = _kalshi_direction(book_a, book_b)

    assert direction["status"] == "FAIL_CLOSED"
    assert direction["alert"] is False
    assert direction["available_size"] == pytest.approx(5.0)
    assert direction["reasons"] == ["BOOK_A_YES_INSUFFICIENT_ASK_DEPTH:5<10"]


def test_fees_slippage_and_capital_lock_can_remove_the_displayed_edge() -> None:
    book_a = _book(
        "kalshi",
        "KX-A",
        yes_asks=[{"price": 0.43, "size": 5.0}, {"price": 0.47, "size": 5.0}],
    )
    book_b = _book(
        "kalshi",
        "KX-B",
        no_asks=[{"price": 0.46, "size": 5.0}, {"price": 0.50, "size": 5.0}],
    )
    policy = _policy(annual_capital_rate=0.25, capital_lock_days=30.0)

    direction = _kalshi_direction(book_a, book_b, policy)

    assert direction["legs"][0]["best_ask"] == pytest.approx(0.43)
    assert direction["legs"][0]["vwap"] == pytest.approx(0.45)
    assert direction["legs"][0]["slippage_per_unit"] == pytest.approx(0.02)
    assert direction["gross_residual_per_unit"] == pytest.approx(0.07)
    assert direction["costs"]["fees"] > 0
    assert direction["costs"]["capital_lock_cost"] > 0
    assert direction["net_residual_per_unit"] < 0.05
    assert direction["status"] == "NO_EXECUTABLE_SHADOW_EDGE"
    assert direction["alert"] is False
    assert direction["reasons"] == ["NET_RESIDUAL_BELOW_THRESHOLD"]


def test_explicit_per_unit_buffers_are_visible_and_subtracted_from_residual() -> None:
    book_a = _book("kalshi", "KX-A", yes_asks=[{"price": 0.40, "size": 20.0}])
    book_b = _book("kalshi", "KX-B", no_asks=[{"price": 0.42, "size": 20.0}])
    unbuffered = _kalshi_direction(book_a, book_b, _policy(annual_capital_rate=0.0))
    buffered = _kalshi_direction(
        book_a,
        book_b,
        _policy(
            annual_capital_rate=0.0,
            explicit_slippage_buffer_per_unit=0.01,
            timestamp_skew_buffer_per_unit=0.02,
            settlement_divergence_buffer_per_unit=0.03,
            collateral_basis_buffer_per_unit=0.04,
            rebalancing_withdrawal_allowance_per_unit=0.05,
        ),
    )
    costs = buffered["costs"]

    assert costs["explicit_slippage_buffer"] == pytest.approx(0.1)
    assert costs["timestamp_skew_buffer"] == pytest.approx(0.2)
    assert costs["settlement_divergence_buffer"] == pytest.approx(0.3)
    assert costs["collateral_basis_buffer"] == pytest.approx(0.4)
    assert costs["rebalancing_withdrawal_allowance"] == pytest.approx(0.5)
    assert buffered["net_residual"] == pytest.approx(unbuffered["net_residual"] - 1.5)
    assert costs["total_cost_including_all_deductions"] == pytest.approx(
        costs["aggressive_fill_cost"] + costs["fees"] + costs["capital_lock_cost"] + 1.5
    )
    assert buffered["reasons"] == ["NET_RESIDUAL_BELOW_THRESHOLD"]
    assert buffered["alert"] is False


def test_kalshi_taker_fee_is_a_conservative_upper_bound_not_an_exact_fee() -> None:
    book_a = _book(
        "kalshi",
        "KX-A",
        yes_asks=[{"price": 0.43, "size": 5.0}, {"price": 0.47, "size": 5.0}],
    )
    book_b = _book("kalshi", "KX-B")

    direction = _kalshi_direction(book_a, book_b, _policy(annual_capital_rate=0.0))
    leg = direction["legs"][0]
    fee = leg["fee_evidence"]

    assert leg["fill_cost"] == pytest.approx(4.5)
    assert fee["model"] == "KALSHI_QUADRATIC_TAKER"
    assert fee["liquidity_role"] == "TAKER"
    assert fee["raw_fee"] == pytest.approx(0.17297)
    assert fee["trade_fee_venue_quantum_ceiling"] == pytest.approx(0.17297)
    assert fee["trade_fee_rounding_quantum"] == pytest.approx(0.000001)
    assert fee["balance_precision_upper_bound"] == pytest.approx(0.01)
    assert fee["balance_rounding_upper_bound"] == pytest.approx(0.009999)
    assert leg["fee"] == pytest.approx(0.182969)
    assert fee["calculation_class"] == "CONSERVATIVE_UPPER_BOUND"
    assert fee["exact_fee_claimed"] is False
    assert fee["account_fee_accumulator_inputs"] == "NOT_AVAILABLE"
    assert fee["account_class_inputs"] == "NOT_AVAILABLE"
    assert fee["official_fee_schedule_sha256"] == "1" * 64
    assert fee["formula_binding_id"] == "kalshi-prediction-fees-2026-07-07"
    assert fee["raw_fee"] != pytest.approx(leg["fill_cost"] * 0.07)
    assert fee["passive_or_maker_credit_assumed"] is False


def test_polymarket_fee_rate_only_is_unavailable_never_no_edge_or_alert() -> None:
    book_a = _book("kalshi", "KX-TEST", yes_asks=[{"price": 0.40, "size": 20.0}])
    book_b = _book("polymarket", "PM-TEST", no_asks=[{"price": 0.42, "size": 20.0}])

    result = _calculate(book_a, book_b)
    fee_evidence = result["book_evidence"][1]["fee_evidence"]

    assert result["status"] == "FEE_EVIDENCE_UNAVAILABLE"
    assert result["status"] != "NO_EXECUTABLE_SHADOW_EDGE"
    assert result["alert"] is result["live_eligible"] is False
    assert "BOOK_B_NATIVE_FEE_EVIDENCE_NOT_VALID" in result["reasons"]
    assert all(direction["status"] == "FAIL_CLOSED" for direction in result["directions"])
    assert all(direction["alert"] is False for direction in result["directions"])
    assert all(direction["legs"] == [] for direction in result["directions"])
    assert fee_evidence["status"] == "FAIL_CLOSED"
    assert fee_evidence["model"] == "POLYMARKET_FEE_ESTIMATE_ONLY"
    assert fee_evidence["estimate_only"] is True
    assert fee_evidence["effective_from"] is None
    assert fee_evidence["effective_basis"] == "UNAVAILABLE"
    assert fee_evidence["schedule"]["condition_fee_details"] is None
    assert set(fee_evidence["reason_codes"]) == {
        "POLYMARKET_CONDITION_FEE_PARAMETERS_NOT_CAPTURED",
        "POLYMARKET_FEE_EFFECTIVE_TIMESTAMP_UNAVAILABLE",
        "POLYMARKET_TOKEN_BASE_FEE_IS_ESTIMATE_ONLY",
    }
    assert result["execution_assumptions"]["passive_fills_assumed"] is False
    assert result["execution_assumptions"]["queue_priority_assumed"] is False


def test_kalshi_only_arithmetic_threshold_is_inclusive_but_not_cross_venue_evidence() -> None:
    book_a = _book("kalshi", "KX-A", yes_asks=[{"price": 0.40, "size": 20.0}])
    book_b = _book("kalshi", "KX-B", no_asks=[{"price": 0.42, "size": 20.0}])
    policy = _policy(annual_capital_rate=0.0)

    direction = _kalshi_direction(book_a, book_b, policy)
    boundary = _kalshi_direction(
        book_a,
        book_b,
        _policy(
            annual_capital_rate=0.0,
            net_residual_threshold=direction["net_residual_per_unit"],
        ),
    )
    below = _kalshi_direction(
        book_a,
        book_b,
        _policy(
            annual_capital_rate=0.0,
            net_residual_threshold=direction["net_residual_per_unit"] + 0.000001,
        ),
    )

    assert direction["alert"] is True
    assert boundary["alert"] is True
    assert below["alert"] is False
    assert below["reasons"] == ["NET_RESIDUAL_BELOW_THRESHOLD"]


def test_missing_native_fee_evidence_and_unverified_semantics_fail_closed() -> None:
    book_a = _book("kalshi", "KX-TEST")
    book_b = _book("polymarket", "PM-TEST")
    del book_a["fee_evidence"]

    fee_result = _calculate(book_a, book_b)
    semantic_result = calculate_shadow(
        _decision("NEEDS_REVIEW"),
        deepcopy(book_a),
        deepcopy(book_b),
        _policy(),
        evaluated_at=_iso(EVALUATED_AT),
    )

    assert fee_result["status"] == "FEE_EVIDENCE_UNAVAILABLE"
    assert semantic_result["status"] == "FEE_EVIDENCE_UNAVAILABLE"
    assert "BOOK_A_MISSING_NATIVE_FEE_EVIDENCE" in fee_result["reasons"]
    assert "SEMANTIC_DECISION_NOT_VERIFIED_EQUIVALENT" in semantic_result["reasons"]
    assert fee_result["alert"] is semantic_result["alert"] is False
    assert fee_result["live_eligible"] is semantic_result["live_eligible"] is False


@pytest.mark.parametrize(
    ("missing_field", "reason"),
    [
        ("official_fee_schedule_sha256", "BOOK_A_NATIVE_FEE_OFFICIAL_SCHEDULE_SHA256_INVALID"),
        ("formula_binding_id", "BOOK_A_NATIVE_FEE_FORMULA_BINDING_ID_INVALID"),
    ],
)
def test_missing_kalshi_official_formula_binding_fails_closed(missing_field: str, reason: str) -> None:
    book_a = _book("kalshi", "KX-TEST")
    del book_a["fee_evidence"][missing_field]
    book_b = _book("polymarket", "PM-TEST")

    result = _calculate(book_a, book_b)

    assert reason in result["reasons"]
    assert result["status"] == "FEE_EVIDENCE_UNAVAILABLE"
    assert result["alert"] is result["live_eligible"] is False
    assert all(direction["status"] == "FAIL_CLOSED" for direction in result["directions"])


def test_tampered_kalshi_official_formula_binding_chain_fails_closed() -> None:
    book_a = _book("kalshi", "KX-TEST")
    official_schedule = book_a["fee_evidence"]["schedule"]["official_fee_schedule"]
    official_schedule["formula_binding"]["taker_base_coefficient"] = "0.08"
    book_b = _book("polymarket", "PM-TEST")

    result = _calculate(book_a, book_b)

    assert "BOOK_A_NATIVE_FEE_OFFICIAL_BINDING_CHAIN_INVALID" in result["reasons"]
    assert result["status"] == "FEE_EVIDENCE_UNAVAILABLE"
    assert result["alert"] is result["live_eligible"] is False
    assert all(direction["status"] == "FAIL_CLOSED" for direction in result["directions"])


def test_closed_market_and_invalid_lot_size_fail_closed_before_arithmetic() -> None:
    book = _book("kalshi", "KX-TEST")
    book["book_eligible"] = False
    book["market_status"] = "closed"
    policy = _policy(requested_size=10.5)

    parsed, reasons, _ = _kalshi_book_validation(book, policy=policy)

    assert parsed is None
    assert "BOOK_A_MARKET_STATUS_NOT_ACTIVE" in reasons
    assert "BOOK_A_MARKET_NOT_BOOK_ELIGIBLE" in reasons
    assert "BOOK_A_REQUESTED_SIZE_VIOLATES_INCREMENT" in reasons
