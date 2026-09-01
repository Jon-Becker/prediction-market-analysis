"""Pure, fail-closed shadow calculations for verified cross-venue pairs.

The functions in this module do not perform I/O.  They consume independently
captured venue-native evidence and model only immediate aggressive purchases of
the complementary outcomes.  They never model passive fills, queue priority,
or live eligibility.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

DEFAULT_NET_RESIDUAL_THRESHOLD = 0.05
_OUTCOMES = ("YES", "NO")
_DIRECTIONS = (
    ("A_YES_B_NO", "YES", "NO"),
    ("A_NO_B_YES", "NO", "YES"),
)
_EPSILON = 1e-12


@dataclass(frozen=True)
class ShadowPolicy:
    """Explicit assumptions required for an aggressive-fill shadow check."""

    requested_size: float
    max_book_age_seconds: float
    max_cross_venue_skew_seconds: float
    annual_capital_rate: float
    capital_lock_days: float
    explicit_slippage_buffer_per_unit: float
    timestamp_skew_buffer_per_unit: float
    settlement_divergence_buffer_per_unit: float
    collateral_basis_buffer_per_unit: float
    rebalancing_withdrawal_allowance_per_unit: float
    net_residual_threshold: float = DEFAULT_NET_RESIDUAL_THRESHOLD

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe policy mapping."""

        return {
            "requested_size": self.requested_size,
            "max_book_age_seconds": self.max_book_age_seconds,
            "max_cross_venue_skew_seconds": self.max_cross_venue_skew_seconds,
            "annual_capital_rate": self.annual_capital_rate,
            "capital_lock_days": self.capital_lock_days,
            "explicit_slippage_buffer_per_unit": self.explicit_slippage_buffer_per_unit,
            "timestamp_skew_buffer_per_unit": self.timestamp_skew_buffer_per_unit,
            "settlement_divergence_buffer_per_unit": self.settlement_divergence_buffer_per_unit,
            "collateral_basis_buffer_per_unit": self.collateral_basis_buffer_per_unit,
            "rebalancing_withdrawal_allowance_per_unit": self.rebalancing_withdrawal_allowance_per_unit,
            "net_residual_threshold": self.net_residual_threshold,
        }


@dataclass(frozen=True)
class _PolicyValues:
    requested_size: float
    max_book_age_seconds: float
    max_cross_venue_skew_seconds: float
    annual_capital_rate: float
    capital_lock_days: float
    explicit_slippage_buffer_per_unit: float
    timestamp_skew_buffer_per_unit: float
    settlement_divergence_buffer_per_unit: float
    collateral_basis_buffer_per_unit: float
    rebalancing_withdrawal_allowance_per_unit: float
    net_residual_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_size": self.requested_size,
            "max_book_age_seconds": self.max_book_age_seconds,
            "max_cross_venue_skew_seconds": self.max_cross_venue_skew_seconds,
            "annual_capital_rate": self.annual_capital_rate,
            "capital_lock_days": self.capital_lock_days,
            "explicit_slippage_buffer_per_unit": self.explicit_slippage_buffer_per_unit,
            "timestamp_skew_buffer_per_unit": self.timestamp_skew_buffer_per_unit,
            "settlement_divergence_buffer_per_unit": self.settlement_divergence_buffer_per_unit,
            "collateral_basis_buffer_per_unit": self.collateral_basis_buffer_per_unit,
            "rebalancing_withdrawal_allowance_per_unit": self.rebalancing_withdrawal_allowance_per_unit,
            "net_residual_threshold": self.net_residual_threshold,
            "threshold_basis": "net_residual_per_unit",
        }


@dataclass(frozen=True)
class _BookValues:
    label: str
    candidate_id: str
    venue: str
    native_market_id: str
    source_at: datetime
    fee_evidence: dict[str, Any]
    sides: dict[str, dict[str, list[dict[str, float]]]]
    evidence: dict[str, Any]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical_json_sha256(value: Any) -> str | None:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _utc_datetime(value: Any) -> datetime | None:
    parsed: datetime
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _reason(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def _has_fee_evidence_failure(reasons: Sequence[str]) -> bool:
    return any("_FEE_" in reason or reason.endswith("_FEE") for reason in reasons)


def _empty_costs() -> dict[str, float | None]:
    return {
        "aggressive_fill_cost": None,
        "fees": None,
        "capital_lock_cost": None,
        "explicit_slippage_buffer": None,
        "timestamp_skew_buffer": None,
        "settlement_divergence_buffer": None,
        "collateral_basis_buffer": None,
        "rebalancing_withdrawal_allowance": None,
        "total_cost_including_all_deductions": None,
        "slippage_within_aggressive_fill_cost": None,
        "observed_spread_for_requested_size": None,
    }


def _empty_direction(name: str, outcome_a: str, outcome_b: str, reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "direction": name,
        "outcome_a": outcome_a,
        "outcome_b": outcome_b,
        "status": "FAIL_CLOSED",
        "alert": False,
        "reasons": sorted(set(reasons)),
        "requested_size": None,
        "available_size": None,
        "legs": [],
        "gross_payout": None,
        "gross_residual": None,
        "gross_residual_per_unit": None,
        "net_residual": None,
        "net_residual_per_unit": None,
        "costs": _empty_costs(),
    }


def _base_result(decision: Mapping[str, Any] | Any, evaluated_at: Any) -> dict[str, Any]:
    candidate_id = decision.get("candidate_id") if isinstance(decision, Mapping) else None
    semantic_status = decision.get("status") if isinstance(decision, Mapping) else None
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "evaluated_at": evaluated_at if isinstance(evaluated_at, str) else None,
        "semantic_status": semantic_status,
        "status": "NO_EXECUTABLE_SHADOW_EDGE",
        "alert": False,
        "live_eligible": False,
        "reasons": [],
        "execution_assumptions": {
            "fill_style": "AGGRESSIVE_ASK_DEPTH_ONLY",
            "passive_fills_assumed": False,
            "queue_priority_assumed": False,
            "partial_size_alerts_allowed": False,
            "fee_model": "VENUE_NATIVE_TAKER_LEVELWISE",
            "fee_rebates_assumed": False,
            "capital_lock_cost_basis": "AGGRESSIVE_FILL_COST",
            "threshold_comparison": "GREATER_THAN_OR_EQUAL",
        },
        "policy": None,
        "book_evidence": [],
        "cross_venue_source_skew_seconds": None,
        "directions": [],
    }


def _policy_values(policy: Mapping[str, Any] | ShadowPolicy | Any) -> tuple[_PolicyValues | None, list[str]]:
    if isinstance(policy, ShadowPolicy):
        supplied: Mapping[str, Any] = policy.as_dict()
    elif isinstance(policy, Mapping):
        supplied = policy
    else:
        return None, ["MISSING_OR_INVALID_POLICY"]

    reasons: list[str] = []

    def required_nonnegative(key: str) -> float | None:
        value = _number(supplied.get(key))
        if value is None:
            reasons.append(f"MISSING_OR_INVALID_{key.upper()}")
        elif value < 0:
            reasons.append(f"NEGATIVE_{key.upper()}")
        return value

    requested_size = _number(supplied.get("requested_size"))
    if requested_size is None or requested_size <= 0:
        reasons.append("MISSING_OR_INVALID_REQUESTED_SIZE")

    max_age = required_nonnegative("max_book_age_seconds")
    max_skew = required_nonnegative("max_cross_venue_skew_seconds")
    annual_rate = required_nonnegative("annual_capital_rate")
    explicit_slippage_buffer = required_nonnegative("explicit_slippage_buffer_per_unit")
    timestamp_skew_buffer = required_nonnegative("timestamp_skew_buffer_per_unit")
    settlement_divergence_buffer = required_nonnegative("settlement_divergence_buffer_per_unit")
    collateral_basis_buffer = required_nonnegative("collateral_basis_buffer_per_unit")
    rebalancing_allowance = required_nonnegative("rebalancing_withdrawal_allowance_per_unit")

    lock_value = supplied.get("capital_lock_days")
    if lock_value is None:
        lock_value = supplied.get("settlement_duration_days")
    capital_lock_days = _number(lock_value)
    if capital_lock_days is None:
        reasons.append("MISSING_SETTLEMENT_OR_CAPITAL_LOCK_DURATION")
    elif capital_lock_days < 0:
        reasons.append("NEGATIVE_CAPITAL_LOCK_DAYS")

    threshold_raw = supplied.get("net_residual_threshold", DEFAULT_NET_RESIDUAL_THRESHOLD)
    threshold = _number(threshold_raw)
    if threshold is None:
        reasons.append("INVALID_NET_RESIDUAL_THRESHOLD")
    elif threshold < 0:
        reasons.append("NEGATIVE_NET_RESIDUAL_THRESHOLD")

    if reasons:
        return None, sorted(set(reasons))

    assert requested_size is not None
    assert max_age is not None
    assert max_skew is not None
    assert annual_rate is not None
    assert explicit_slippage_buffer is not None
    assert timestamp_skew_buffer is not None
    assert settlement_divergence_buffer is not None
    assert collateral_basis_buffer is not None
    assert rebalancing_allowance is not None
    assert capital_lock_days is not None
    assert threshold is not None
    return (
        _PolicyValues(
            requested_size=requested_size,
            max_book_age_seconds=max_age,
            max_cross_venue_skew_seconds=max_skew,
            annual_capital_rate=annual_rate,
            capital_lock_days=capital_lock_days,
            explicit_slippage_buffer_per_unit=explicit_slippage_buffer,
            timestamp_skew_buffer_per_unit=timestamp_skew_buffer,
            settlement_divergence_buffer_per_unit=settlement_divergence_buffer,
            collateral_basis_buffer_per_unit=collateral_basis_buffer,
            rebalancing_withdrawal_allowance_per_unit=rebalancing_allowance,
            net_residual_threshold=threshold,
        ),
        [],
    )


def _levels(value: Any, prefix: str, kind: str) -> tuple[list[dict[str, float]], list[str]]:
    reason_prefix = f"{prefix}_{kind.upper()}"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return [], [_reason(reason_prefix, "MISSING_OR_EMPTY")]

    levels: list[dict[str, float]] = []
    reasons: list[str] = []
    for level in value:
        if not isinstance(level, Mapping):
            reasons.append(_reason(reason_prefix, "INVALID_LEVEL"))
            continue
        price = _number(level.get("price"))
        size = _number(level.get("size"))
        if price is None or price < 0 or price > 1:
            reasons.append(_reason(reason_prefix, "INVALID_PRICE"))
        if size is None or size <= 0:
            reasons.append(_reason(reason_prefix, "INVALID_SIZE"))
        if price is not None and 0 <= price <= 1 and size is not None and size > 0:
            levels.append({"price": price, "size": size})

    reverse = kind == "bids"
    levels.sort(key=lambda item: item["price"], reverse=reverse)
    return levels, sorted(set(reasons))


def _raw_levels(value: Any, *, cents: bool = False) -> tuple[list[dict[str, float]], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [], False
    levels: list[dict[str, float]] = []
    valid = True
    for raw_level in value:
        price: float | None = None
        size: float | None = None
        if isinstance(raw_level, Mapping):
            price = _number(raw_level.get("price"))
            size = _number(raw_level.get("size"))
        elif isinstance(raw_level, Sequence) and not isinstance(raw_level, (str, bytes)) and len(raw_level) >= 2:
            price = _number(raw_level[0])
            size = _number(raw_level[1])
        if price is None or size is None or size <= 0:
            valid = False
            continue
        if cents:
            price /= 100
        if not 0 <= price <= 1:
            valid = False
            continue
        levels.append({"price": price, "size": size})
    return levels, valid


def _sides_from_raw_response(
    venue: str,
    raw_response: Mapping[str, Any],
    depth: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    if venue == "kalshi":
        wrapper = raw_response.get("orderbook")
        wrapper = dict(wrapper) if isinstance(wrapper, Mapping) else {}
        book = wrapper.get("orderbook_fp")
        cents = False
        if not isinstance(book, Mapping) or not book:
            book = wrapper.get("orderbook")
            cents = True
        if not isinstance(book, Mapping) or not book:
            return None, ["RAW_ORDERBOOK_SHAPE_INVALID"]
        yes_bids, yes_valid = _raw_levels(book.get("yes" if cents else "yes_dollars"), cents=cents)
        no_bids, no_valid = _raw_levels(book.get("no" if cents else "no_dollars"), cents=cents)
        if not yes_valid or not no_valid:
            reasons.append("RAW_ORDERBOOK_LEVEL_INVALID")
        yes_bids.sort(key=lambda level: level["price"], reverse=True)
        no_bids.sort(key=lambda level: level["price"], reverse=True)
        yes_bids = yes_bids[:depth]
        no_bids = no_bids[:depth]
        return (
            {
                "YES": {
                    "bids": yes_bids,
                    "asks": sorted(
                        ({"price": round(1 - level["price"], 10), "size": level["size"]} for level in no_bids),
                        key=lambda level: level["price"],
                    ),
                },
                "NO": {
                    "bids": no_bids,
                    "asks": sorted(
                        ({"price": round(1 - level["price"], 10), "size": level["size"]} for level in yes_bids),
                        key=lambda level: level["price"],
                    ),
                },
            },
            reasons,
        )

    if venue == "polymarket":
        raw_books = raw_response.get("books")
        if not isinstance(raw_books, Mapping):
            return None, ["RAW_BOOKS_SHAPE_INVALID"]
        sides: dict[str, Any] = {}
        for outcome in _OUTCOMES:
            raw_book = raw_books.get(outcome)
            if not isinstance(raw_book, Mapping):
                reasons.append(f"RAW_{outcome}_BOOK_MISSING")
                continue
            bids, bids_valid = _raw_levels(raw_book.get("bids"))
            asks, asks_valid = _raw_levels(raw_book.get("asks"))
            if not bids_valid or not asks_valid:
                reasons.append(f"RAW_{outcome}_BOOK_LEVEL_INVALID")
            bids.sort(key=lambda level: level["price"], reverse=True)
            asks.sort(key=lambda level: level["price"])
            sides[outcome] = {"bids": bids[:depth], "asks": asks[:depth]}
        return (sides if set(sides) == set(_OUTCOMES) else None), reasons

    return None, ["RAW_BOOK_VENUE_UNSUPPORTED"]


def _decoded_request_payload(request: Any) -> tuple[Any | None, list[str]]:
    if not isinstance(request, Mapping):
        return None, ["REQUEST_INVALID"]
    reasons: list[str] = []
    if request.get("method") != "GET":
        reasons.append("REQUEST_METHOD_NOT_GET")
    if request.get("body_complete") is not True:
        reasons.append("REQUEST_BODY_INCOMPLETE")
    if request.get("request_accept_encoding") != "identity":
        reasons.append("REQUEST_IDENTITY_ENCODING_NOT_BOUND")
    if request.get("raw_body_representation") != "HTTP_ENTITY_BYTES_ACCEPT_ENCODING_IDENTITY":
        reasons.append("REQUEST_RAW_BODY_REPRESENTATION_INVALID")
    if request.get("response_header_retention") != "EXPLICIT_ALLOWLIST":
        reasons.append("REQUEST_HEADER_RETENTION_INVALID")
    status_code = request.get("status_code")
    if isinstance(status_code, bool) or not isinstance(status_code, int) or not 200 <= status_code < 300:
        reasons.append("REQUEST_HTTP_STATUS_INVALID")
    raw_hash = request.get("raw_body_hash")
    raw_base64 = request.get("raw_body_base64")
    try:
        raw_bytes = base64.b64decode(raw_base64, validate=True) if isinstance(raw_base64, str) else None
    except (binascii.Error, ValueError):
        raw_bytes = None
    if raw_bytes is None:
        return None, [*reasons, "REQUEST_RAW_BODY_INVALID"]
    if not _is_sha256(raw_hash) or hashlib.sha256(raw_bytes).hexdigest() != str(raw_hash).lower():
        reasons.append("REQUEST_RAW_BODY_HASH_MISMATCH")
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, ValueError):
        return None, [*reasons, "REQUEST_RAW_BODY_NOT_JSON"]
    return payload, reasons


def _book_provenance_reasons(book: Mapping[str, Any], venue: str, prefix: str) -> list[str]:
    reasons: list[str] = []
    raw_response = book.get("raw_response")
    raw_digest = book.get("raw_sha256")
    if not isinstance(raw_response, Mapping):
        return [_reason(prefix, "RAW_RESPONSE_MISSING_OR_INVALID")]
    computed_digest = _canonical_json_sha256(raw_response)
    if not _is_sha256(raw_digest) or computed_digest is None or str(raw_digest).lower() != computed_digest:
        reasons.append(_reason(prefix, "RAW_SHA256_MISMATCH"))

    depth = book.get("normalized_depth_limit")
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 100:
        reasons.append(_reason(prefix, "NORMALIZED_DEPTH_LIMIT_INVALID"))
    else:
        expected_sides, raw_reasons = _sides_from_raw_response(venue, raw_response, depth)
        reasons.extend(_reason(prefix, reason) for reason in raw_reasons)
        if expected_sides is None or _canonical_json_sha256(expected_sides) != _canonical_json_sha256(
            book.get("sides")
        ):
            reasons.append(_reason(prefix, "NORMALIZED_SIDES_RAW_BINDING_MISMATCH"))

    raw_requests = book.get("requests")
    if not isinstance(raw_requests, Sequence) or isinstance(raw_requests, (str, bytes)) or not raw_requests:
        reasons.append(_reason(prefix, "REQUEST_EVIDENCE_MISSING"))
        return reasons
    decoded_requests: list[tuple[Mapping[str, Any], Any]] = []
    for index, request in enumerate(raw_requests):
        payload, request_reasons = _decoded_request_payload(request)
        reasons.extend(_reason(prefix, f"REQUEST_{index}_{reason}") for reason in request_reasons)
        if isinstance(request, Mapping) and payload is not None:
            decoded_requests.append((request, payload))

    valid_requests = [request for request, _payload in decoded_requests]
    if valid_requests:
        request_walls = [request.get("request_wall_utc") for request in valid_requests]
        response_walls = [request.get("response_wall_utc") for request in valid_requests]
        request_ns = [request.get("request_monotonic_ns") for request in valid_requests]
        response_ns = [request.get("response_monotonic_ns") for request in valid_requests]
        if not all(
            isinstance(value, str) and _utc_datetime(value) is not None for value in request_walls + response_walls
        ) or not all(isinstance(value, int) and not isinstance(value, bool) for value in request_ns + response_ns):
            reasons.append(_reason(prefix, "REQUEST_WINDOW_INVALID"))
        else:
            expected_start_wall = min(request_walls)
            expected_end_wall = max(response_walls)
            expected_start_ns = min(request_ns)
            expected_end_ns = max(response_ns)
            expected_rtt_ms = (expected_end_ns - expected_start_ns) / 1_000_000
            if (
                book.get("request_started_at") != expected_start_wall
                or book.get("received_at") != expected_end_wall
                or book.get("request_monotonic_ns") != expected_start_ns
                or book.get("response_monotonic_ns") != expected_end_ns
                or _number(book.get("rtt_ms")) != expected_rtt_ms
            ):
                reasons.append(_reason(prefix, "REQUEST_WINDOW_BINDING_MISMATCH"))

    if venue == "kalshi":
        orderbook_payloads = [
            payload for request, payload in decoded_requests if str(request.get("path", "")).endswith("/orderbook")
        ]
        if len(orderbook_payloads) != 1 or orderbook_payloads[0] != raw_response.get("orderbook"):
            reasons.append(_reason(prefix, "RAW_ORDERBOOK_REQUEST_BINDING_MISMATCH"))
    elif venue == "polymarket":
        outcome_ids = book.get("native_outcome_ids")
        token_to_outcome = (
            {str(token): str(outcome).upper() for outcome, token in outcome_ids.items()}
            if isinstance(outcome_ids, Mapping)
            else {}
        )
        raw_books = raw_response.get("books")
        raw_fee_rates = raw_response.get("fee_rates")
        for request, payload in decoded_requests:
            params = request.get("params")
            token_id = str(params.get("token_id")) if isinstance(params, Mapping) and params.get("token_id") else None
            outcome = token_to_outcome.get(token_id or "")
            path = request.get("path")
            expected = None
            if outcome is not None and isinstance(raw_books, Mapping) and path == "/book":
                expected = raw_books.get(outcome)
            elif outcome is not None and isinstance(raw_fee_rates, Mapping) and path == "/fee-rate":
                expected = raw_fee_rates.get(outcome)
            if expected is None or payload != expected:
                reasons.append(_reason(prefix, "RAW_REQUEST_BINDING_MISMATCH"))
        if len(decoded_requests) != 4:
            reasons.append(_reason(prefix, "RAW_REQUEST_SET_INCOMPLETE"))
    return sorted(set(reasons))


def _book_values(
    book: Mapping[str, Any] | Any,
    *,
    label: str,
    evaluated_at: datetime,
    policy: _PolicyValues,
) -> tuple[_BookValues | None, list[str], dict[str, Any]]:
    prefix = f"BOOK_{label}"
    if not isinstance(book, Mapping):
        return None, [_reason(prefix, "MISSING_OR_INVALID")], {"label": label}

    candidate_id_raw = book.get("candidate_id")
    candidate_id = str(candidate_id_raw).strip() if candidate_id_raw is not None else ""
    venue_raw = book.get("venue")
    venue = str(venue_raw).strip().lower() if venue_raw is not None else ""
    native_id_raw = book.get("native_market_id")
    native_market_id = str(native_id_raw).strip() if native_id_raw is not None else ""
    evidence = _book_evidence(book, label)
    reasons: list[str] = []
    if not candidate_id:
        reasons.append(_reason(prefix, "MISSING_CANDIDATE_ID"))
    if not venue:
        reasons.append(_reason(prefix, "MISSING_VENUE"))
    if not native_market_id:
        reasons.append(_reason(prefix, "MISSING_NATIVE_MARKET_ID"))

    reasons.extend(_book_provenance_reasons(book, venue, prefix))

    market_status = book.get("market_status")
    if not isinstance(market_status, str) or not market_status.strip():
        reasons.append(_reason(prefix, "MISSING_MARKET_STATUS"))
    elif market_status.strip().lower() not in {"active", "open"}:
        reasons.append(_reason(prefix, "MARKET_STATUS_NOT_ACTIVE"))
    if book.get("book_eligible") is not True:
        reasons.append(_reason(prefix, "MARKET_NOT_BOOK_ELIGIBLE"))

    minimum_order_size = _number(book.get("minimum_order_size"))
    if minimum_order_size is None or minimum_order_size <= 0:
        reasons.append(_reason(prefix, "MISSING_OR_INVALID_MINIMUM_ORDER_SIZE"))
    elif policy.requested_size + _EPSILON < minimum_order_size:
        reasons.append(_reason(prefix, "REQUESTED_SIZE_BELOW_MINIMUM"))
    size_increment_raw = book.get("size_increment")
    if size_increment_raw is not None:
        size_increment = _number(size_increment_raw)
        if size_increment is None or size_increment <= 0:
            reasons.append(_reason(prefix, "INVALID_SIZE_INCREMENT"))
        else:
            increments = policy.requested_size / size_increment
            if abs(increments - round(increments)) > _EPSILON:
                reasons.append(_reason(prefix, "REQUESTED_SIZE_VIOLATES_INCREMENT"))

    timestamps: dict[str, datetime | None] = {}
    for key in ("request_started_at", "received_at"):
        raw_value = book.get(key)
        parsed = _utc_datetime(raw_value)
        timestamps[key] = parsed
        if raw_value is None:
            reasons.append(_reason(prefix, f"MISSING_{key.upper()}"))
        elif parsed is None:
            reasons.append(_reason(prefix, f"INVALID_{key.upper()}"))

    source_raw = book.get("source_timestamp")
    source_at = _utc_datetime(source_raw)
    freshness_basis = book.get("freshness_basis")
    if freshness_basis != "VENUE_SOURCE_TIMESTAMP":
        reasons.append(_reason(prefix, "FRESHNESS_BASIS_NOT_VENUE_SOURCE_TIMESTAMP"))
    if source_raw is None:
        reasons.append(_reason(prefix, "MISSING_SOURCE_TIMESTAMP"))
    elif source_at is None:
        reasons.append(_reason(prefix, "INVALID_SOURCE_TIMESTAMP"))

    request_started_at = timestamps["request_started_at"]
    received_at = timestamps["received_at"]
    if request_started_at is not None and received_at is not None and received_at < request_started_at:
        reasons.append(_reason(prefix, "RECEIVED_BEFORE_REQUEST_STARTED"))
    request_monotonic_ns = book.get("request_monotonic_ns")
    response_monotonic_ns = book.get("response_monotonic_ns")
    rtt_ms = _number(book.get("rtt_ms"))
    if isinstance(request_monotonic_ns, bool) or not isinstance(request_monotonic_ns, int):
        reasons.append(_reason(prefix, "MISSING_OR_INVALID_REQUEST_MONOTONIC_NS"))
    if isinstance(response_monotonic_ns, bool) or not isinstance(response_monotonic_ns, int):
        reasons.append(_reason(prefix, "MISSING_OR_INVALID_RESPONSE_MONOTONIC_NS"))
    if rtt_ms is None or rtt_ms < 0:
        reasons.append(_reason(prefix, "MISSING_OR_INVALID_RTT_MS"))
    if (
        isinstance(request_monotonic_ns, int)
        and not isinstance(request_monotonic_ns, bool)
        and isinstance(response_monotonic_ns, int)
        and not isinstance(response_monotonic_ns, bool)
    ):
        if response_monotonic_ns < request_monotonic_ns:
            reasons.append(_reason(prefix, "RESPONSE_MONOTONIC_BEFORE_REQUEST"))
        else:
            capture_window_seconds = (response_monotonic_ns - request_monotonic_ns) / 1_000_000_000
            evidence["capture_window_seconds"] = capture_window_seconds
            if capture_window_seconds > policy.max_book_age_seconds:
                reasons.append(_reason(prefix, "CAPTURE_WINDOW_EXCEEDS_AGE_LIMIT"))
            if rtt_ms is not None and abs(rtt_ms - (capture_window_seconds * 1000)) > 0.001:
                reasons.append(_reason(prefix, "RTT_DOES_NOT_MATCH_MONOTONIC_WINDOW"))
    if received_at is not None and received_at > evaluated_at:
        reasons.append(_reason(prefix, "RECEIVED_AFTER_EVALUATION"))
    if source_at is not None:
        evidence["effective_source_timestamp"] = _utc_iso(source_at)
        age_seconds = (evaluated_at - source_at).total_seconds()
        evidence["source_age_seconds"] = age_seconds
        if age_seconds < 0:
            reasons.append(_reason(prefix, "SOURCE_TIMESTAMP_AFTER_EVALUATION"))
        elif age_seconds > policy.max_book_age_seconds:
            reasons.append(_reason(prefix, "STALE_SOURCE_TIMESTAMP"))

    raw_source_timestamps = book.get("source_timestamps")
    if isinstance(raw_source_timestamps, Mapping):
        parsed_source_timestamps = {
            str(outcome): _utc_datetime(value) for outcome, value in raw_source_timestamps.items()
        }
        if any(value is None for value in parsed_source_timestamps.values()):
            reasons.append(_reason(prefix, "INVALID_OUTCOME_SOURCE_TIMESTAMP"))
        else:
            source_values = [value for value in parsed_source_timestamps.values() if value is not None]
            for outcome, outcome_at in parsed_source_timestamps.items():
                assert outcome_at is not None
                outcome_age_seconds = (evaluated_at - outcome_at).total_seconds()
                if outcome_age_seconds < 0:
                    reasons.append(_reason(prefix, f"{str(outcome).upper()}_SOURCE_TIMESTAMP_AFTER_EVALUATION"))
                elif outcome_age_seconds > policy.max_book_age_seconds:
                    reasons.append(_reason(prefix, f"{str(outcome).upper()}_STALE_SOURCE_TIMESTAMP"))
            if len(source_values) >= 2:
                outcome_skew_seconds = (max(source_values) - min(source_values)).total_seconds()
                evidence["outcome_source_skew_seconds"] = outcome_skew_seconds
                if outcome_skew_seconds > policy.max_cross_venue_skew_seconds:
                    reasons.append(_reason(prefix, "OUTCOME_SOURCE_SKEW_EXCEEDS_LIMIT"))

    fee_evidence_raw = book.get("fee_evidence")
    fee_evidence = dict(fee_evidence_raw) if isinstance(fee_evidence_raw, Mapping) else {}
    if not fee_evidence:
        reasons.append(_reason(prefix, "MISSING_NATIVE_FEE_EVIDENCE"))
    else:
        if fee_evidence.get("status") != "VALID":
            reasons.append(_reason(prefix, "NATIVE_FEE_EVIDENCE_NOT_VALID"))
        if str(fee_evidence.get("venue", "")).strip().lower() != venue:
            reasons.append(_reason(prefix, "NATIVE_FEE_VENUE_MISMATCH"))
        if fee_evidence.get("liquidity_role") != "TAKER":
            reasons.append(_reason(prefix, "NATIVE_FEE_ROLE_NOT_TAKER"))
        if fee_evidence.get("passive_fills_assumed") is not False:
            reasons.append(_reason(prefix, "NATIVE_FEE_PASSIVE_ASSUMPTION_INVALID"))
        expected_model = {
            "kalshi": "KALSHI_QUADRATIC_TAKER",
            "polymarket": "POLYMARKET_FEE_ESTIMATE_ONLY",
        }.get(venue)
        if expected_model is None or fee_evidence.get("model") != expected_model:
            reasons.append(_reason(prefix, "NATIVE_FEE_MODEL_UNSUPPORTED"))
        for digest_key, value_key in (("schedule_sha256", "schedule"), ("raw_sha256", "raw_response")):
            digest = fee_evidence.get(digest_key)
            computed_digest = _canonical_json_sha256(fee_evidence.get(value_key))
            if not _is_sha256(digest):
                reasons.append(_reason(prefix, f"NATIVE_FEE_{digest_key.upper()}_INVALID"))
            elif computed_digest is None or str(digest).lower() != computed_digest:
                reasons.append(_reason(prefix, f"NATIVE_FEE_{digest_key.upper()}_MISMATCH"))
        if venue == "kalshi":
            official_digest = fee_evidence.get("official_fee_schedule_sha256")
            if (
                not isinstance(official_digest, str)
                or len(official_digest) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in official_digest)
            ):
                reasons.append(_reason(prefix, "NATIVE_FEE_OFFICIAL_SCHEDULE_SHA256_INVALID"))
            if (
                not isinstance(fee_evidence.get("formula_binding_id"), str)
                or not fee_evidence["formula_binding_id"].strip()
            ):
                reasons.append(_reason(prefix, "NATIVE_FEE_FORMULA_BINDING_ID_INVALID"))
            schedule = fee_evidence.get("schedule")
            official_schedule = schedule.get("official_fee_schedule") if isinstance(schedule, Mapping) else None
            formula_binding = (
                official_schedule.get("formula_binding") if isinstance(official_schedule, Mapping) else None
            )
            effective_schedule = schedule.get("effective") if isinstance(schedule, Mapping) else None
            if (
                not isinstance(official_schedule, Mapping)
                or official_schedule.get("status") != "REVIEWED"
                or official_schedule.get("raw_body_sha256") != official_digest
                or not isinstance(formula_binding, Mapping)
                or formula_binding.get("binding_id") != fee_evidence.get("formula_binding_id")
                or not isinstance(effective_schedule, Mapping)
                or str(effective_schedule.get("taker_base_coefficient"))
                != str(formula_binding.get("taker_base_coefficient"))
                or str(effective_schedule.get("trade_fee_rounding_quantum"))
                != str(formula_binding.get("trade_fee_rounding_quantum"))
                or str(effective_schedule.get("balance_precision_upper_bound"))
                != str(formula_binding.get("balance_precision_upper_bound"))
            ):
                reasons.append(_reason(prefix, "NATIVE_FEE_OFFICIAL_BINDING_CHAIN_INVALID"))
            if (
                fee_evidence.get("calculation_class") != "CONSERVATIVE_UPPER_BOUND"
                or fee_evidence.get("exact_fee_claimed") is not False
                or fee_evidence.get("account_fee_accumulator_inputs") != "NOT_AVAILABLE"
                or fee_evidence.get("account_class_inputs") != "NOT_AVAILABLE"
            ):
                reasons.append(_reason(prefix, "NATIVE_FEE_CONSERVATIVE_BOUND_DISCLOSURE_INVALID"))
        schedule_observed_at = _utc_datetime(fee_evidence.get("schedule_observed_at"))
        if schedule_observed_at is None:
            reasons.append(_reason(prefix, "NATIVE_FEE_OBSERVED_AT_INVALID"))
        elif schedule_observed_at > evaluated_at:
            reasons.append(_reason(prefix, "NATIVE_FEE_OBSERVED_AFTER_EVALUATION"))
        if venue == "kalshi" and _utc_datetime(fee_evidence.get("effective_from")) is None:
            reasons.append(_reason(prefix, "NATIVE_FEE_EFFECTIVE_FROM_INVALID"))

    sides_raw = book.get("sides")
    parsed_sides: dict[str, dict[str, list[dict[str, float]]]] = {}
    if not isinstance(sides_raw, Mapping):
        reasons.append(_reason(prefix, "MISSING_OR_INVALID_SIDES"))
    else:
        side_lookup = {str(key).upper(): value for key, value in sides_raw.items()}
        for outcome in _OUTCOMES:
            side = side_lookup.get(outcome)
            if not isinstance(side, Mapping):
                reasons.append(_reason(prefix, f"{outcome}_MISSING_OR_INVALID"))
                continue
            bids, bid_reasons = _levels(side.get("bids"), f"{prefix}_{outcome}", "bids")
            asks, ask_reasons = _levels(side.get("asks"), f"{prefix}_{outcome}", "asks")
            reasons.extend(bid_reasons)
            reasons.extend(ask_reasons)
            if bids and asks and bids[0]["price"] > asks[0]["price"]:
                reasons.append(_reason(prefix, f"{outcome}_CROSSED_BOOK"))
            parsed_sides[outcome] = {"bids": bids, "asks": asks}

    if reasons or source_at is None or not fee_evidence:
        return None, sorted(set(reasons)), evidence
    return (
        _BookValues(
            label=label,
            candidate_id=candidate_id,
            venue=venue,
            native_market_id=native_market_id,
            source_at=source_at,
            fee_evidence=fee_evidence,
            sides=parsed_sides,
            evidence=evidence,
        ),
        [],
        evidence,
    )


def _book_evidence(book: Mapping[str, Any] | Any, label: str) -> dict[str, Any]:
    if not isinstance(book, Mapping):
        return {"label": label}
    venue_raw = book.get("venue")
    native_id_raw = book.get("native_market_id")
    return {
        "label": label,
        "candidate_id": book.get("candidate_id"),
        "venue": str(venue_raw).strip().lower() if venue_raw is not None else None,
        "native_market_id": str(native_id_raw).strip() if native_id_raw is not None else None,
        "market_status": book.get("market_status"),
        "book_eligible": book.get("book_eligible"),
        "minimum_order_size": book.get("minimum_order_size"),
        "size_increment": book.get("size_increment"),
        "request_started_at": book.get("request_started_at"),
        "received_at": book.get("received_at"),
        "request_monotonic_ns": book.get("request_monotonic_ns"),
        "response_monotonic_ns": book.get("response_monotonic_ns"),
        "rtt_ms": book.get("rtt_ms"),
        "source_timestamp": book.get("source_timestamp"),
        "as_of": book.get("as_of"),
        "freshness_basis": book.get("freshness_basis"),
        "raw_sha256": book.get("raw_sha256"),
        "fee_evidence": copy.deepcopy(book.get("fee_evidence")),
        "sides": copy.deepcopy(book.get("sides")),
    }


def _walk_asks(asks: Sequence[Mapping[str, float]], requested_size: float) -> dict[str, Any] | None:
    available_size = sum(level["size"] for level in asks)
    if available_size + _EPSILON < requested_size:
        return None

    remaining = requested_size
    fill_cost = 0.0
    fills: list[dict[str, float]] = []
    for level in asks:
        fill_size = min(remaining, level["size"])
        fill_cost += fill_size * level["price"]
        if fill_size > 0:
            fills.append({"price": level["price"], "size": fill_size, "cost": fill_size * level["price"]})
        remaining -= fill_size
        if remaining <= _EPSILON:
            break
    if remaining > _EPSILON:
        return None

    best_ask = asks[0]["price"]
    vwap = fill_cost / requested_size
    return {
        "available_size": available_size,
        "best_ask": best_ask,
        "vwap": vwap,
        "slippage_per_unit": vwap - best_ask,
        "slippage": fill_cost - (best_ask * requested_size),
        "fill_cost": fill_cost,
        "fills": fills,
    }


def _decimal(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _ceil_decimal(value: Decimal, quantum: Decimal) -> Decimal:
    return (value / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum


def _fee_for_fills(
    book: _BookValues,
    outcome: str,
    fills: Sequence[Mapping[str, float]],
) -> tuple[dict[str, Any] | None, list[str]]:
    prefix = f"BOOK_{book.label}_{outcome}_FEE"
    evidence = book.fee_evidence
    raw_fee = Decimal("0")

    if book.venue == "kalshi":
        effective = (
            evidence.get("schedule", {}).get("effective") if isinstance(evidence.get("schedule"), Mapping) else None
        )
        if not isinstance(effective, Mapping):
            return None, [f"{prefix}_EFFECTIVE_SCHEDULE_MISSING"]
        fee_type = effective.get("fee_type")
        multiplier = _decimal(effective.get("fee_multiplier"))
        base_coefficient = _decimal(effective.get("taker_base_coefficient"))
        trade_fee_quantum = _decimal(effective.get("trade_fee_rounding_quantum"))
        balance_precision = _decimal(effective.get("balance_precision_upper_bound"))
        if (
            fee_type not in {"quadratic", "quadratic_with_maker_fees"}
            or multiplier is None
            or multiplier < 0
            or base_coefficient is None
            or base_coefficient <= 0
            or trade_fee_quantum is None
            or trade_fee_quantum <= 0
            or balance_precision is None
            or balance_precision <= trade_fee_quantum
        ):
            return None, [f"{prefix}_EFFECTIVE_SCHEDULE_INVALID"]
        coefficient = base_coefficient * multiplier
        for fill in fills:
            price = _decimal(fill.get("price"))
            size = _decimal(fill.get("size"))
            if price is None or size is None:
                return None, [f"{prefix}_FILL_INVALID"]
            raw_fee += coefficient * size * price * (Decimal("1") - price)
        trade_fee = _ceil_decimal(raw_fee, trade_fee_quantum)
        rounding_buffer = balance_precision - trade_fee_quantum
        charged_upper_bound = trade_fee + rounding_buffer
        return (
            {
                "model": "KALSHI_QUADRATIC_TAKER",
                "liquidity_role": "TAKER",
                "fee_type": fee_type,
                "fee_multiplier": float(multiplier),
                "base_coefficient": float(base_coefficient),
                "effective_coefficient": float(coefficient),
                "raw_fee": float(raw_fee),
                "trade_fee_venue_quantum_ceiling": float(trade_fee),
                "trade_fee_rounding_quantum": float(trade_fee_quantum),
                "balance_precision_upper_bound": float(balance_precision),
                "balance_rounding_upper_bound": float(rounding_buffer),
                "fee_upper_bound": float(charged_upper_bound),
                "calculation_class": "CONSERVATIVE_UPPER_BOUND",
                "exact_fee_claimed": False,
                "account_fee_accumulator_inputs": "NOT_AVAILABLE",
                "account_class_inputs": "NOT_AVAILABLE",
                "schedule_sha256": evidence.get("schedule_sha256"),
                "official_fee_schedule_sha256": evidence.get("official_fee_schedule_sha256"),
                "formula_binding_id": evidence.get("formula_binding_id"),
                "passive_or_maker_credit_assumed": False,
            },
            [],
        )

    if book.venue == "polymarket":
        return None, [f"{prefix}_PARAMETERS_NOT_EXECUTION_BOUND"]

    return None, [f"{prefix}_VENUE_UNSUPPORTED"]


def _leg(book: _BookValues, outcome: str, requested_size: float) -> tuple[dict[str, Any] | None, list[str]]:
    side = book.sides[outcome]
    walk = _walk_asks(side["asks"], requested_size)
    if walk is None:
        available = sum(level["size"] for level in side["asks"])
        return None, [f"BOOK_{book.label}_{outcome}_INSUFFICIENT_ASK_DEPTH:{available:g}<{requested_size:g}"]

    fee_detail, fee_reasons = _fee_for_fills(book, outcome, walk["fills"])
    if fee_detail is None:
        return None, fee_reasons
    best_bid = side["bids"][0]["price"]
    fee = fee_detail["fee_upper_bound"]
    return (
        {
            "book": book.label,
            "venue": book.venue,
            "native_market_id": book.native_market_id,
            "outcome": outcome,
            "fill_style": "AGGRESSIVE_ASK_DEPTH",
            "requested_size": requested_size,
            "available_size": walk["available_size"],
            "best_bid": best_bid,
            "best_ask": walk["best_ask"],
            "spread": walk["best_ask"] - best_bid,
            "vwap": walk["vwap"],
            "slippage_per_unit": walk["slippage_per_unit"],
            "slippage": walk["slippage"],
            "fill_cost": walk["fill_cost"],
            "fee": fee,
            "fee_evidence": fee_detail,
            "fills": walk["fills"],
        },
        [],
    )


def _calculate_direction(
    name: str,
    outcome_a: str,
    outcome_b: str,
    book_a: _BookValues,
    book_b: _BookValues,
    policy: _PolicyValues,
) -> dict[str, Any]:
    leg_a, reasons_a = _leg(book_a, outcome_a, policy.requested_size)
    leg_b, reasons_b = _leg(book_b, outcome_b, policy.requested_size)
    reasons = reasons_a + reasons_b
    if reasons or leg_a is None or leg_b is None:
        direction = _empty_direction(name, outcome_a, outcome_b, reasons)
        direction["requested_size"] = policy.requested_size
        available_sizes = []
        for book, outcome in ((book_a, outcome_a), (book_b, outcome_b)):
            available_sizes.append(sum(level["size"] for level in book.sides[outcome]["asks"]))
        direction["available_size"] = min(available_sizes)
        return direction

    legs = [leg_a, leg_b]
    requested_size = policy.requested_size
    aggressive_fill_cost = sum(leg["fill_cost"] for leg in legs)
    fee_total = sum(leg["fee"] for leg in legs)
    slippage_total = sum(leg["slippage"] for leg in legs)
    spread_observation = sum(leg["spread"] * requested_size for leg in legs)
    capital_lock_cost = aggressive_fill_cost * policy.annual_capital_rate * policy.capital_lock_days / 365.0
    explicit_slippage_buffer = requested_size * policy.explicit_slippage_buffer_per_unit
    timestamp_skew_buffer = requested_size * policy.timestamp_skew_buffer_per_unit
    settlement_divergence_buffer = requested_size * policy.settlement_divergence_buffer_per_unit
    collateral_basis_buffer = requested_size * policy.collateral_basis_buffer_per_unit
    rebalancing_withdrawal_allowance = requested_size * policy.rebalancing_withdrawal_allowance_per_unit
    explicit_buffers = (
        explicit_slippage_buffer
        + timestamp_skew_buffer
        + settlement_divergence_buffer
        + collateral_basis_buffer
        + rebalancing_withdrawal_allowance
    )
    gross_payout = requested_size
    gross_residual = gross_payout - aggressive_fill_cost
    net_residual = gross_residual - fee_total - capital_lock_cost - explicit_buffers
    net_residual_per_unit = net_residual / requested_size
    alert = net_residual_per_unit >= policy.net_residual_threshold
    reasons = [] if alert else ["NET_RESIDUAL_BELOW_THRESHOLD"]

    return {
        "direction": name,
        "outcome_a": outcome_a,
        "outcome_b": outcome_b,
        "status": "ALERT" if alert else "NO_EXECUTABLE_SHADOW_EDGE",
        "alert": alert,
        "reasons": reasons,
        "requested_size": requested_size,
        "available_size": min(leg["available_size"] for leg in legs),
        "legs": legs,
        "gross_payout": gross_payout,
        "gross_residual": gross_residual,
        "gross_residual_per_unit": gross_residual / requested_size,
        "net_residual": net_residual,
        "net_residual_per_unit": net_residual_per_unit,
        "costs": {
            "aggressive_fill_cost": aggressive_fill_cost,
            "fees": fee_total,
            "capital_lock_cost": capital_lock_cost,
            "explicit_slippage_buffer": explicit_slippage_buffer,
            "timestamp_skew_buffer": timestamp_skew_buffer,
            "settlement_divergence_buffer": settlement_divergence_buffer,
            "collateral_basis_buffer": collateral_basis_buffer,
            "rebalancing_withdrawal_allowance": rebalancing_withdrawal_allowance,
            "total_cost_including_all_deductions": (
                aggressive_fill_cost + fee_total + capital_lock_cost + explicit_buffers
            ),
            "slippage_within_aggressive_fill_cost": slippage_total,
            "observed_spread_for_requested_size": spread_observation,
        },
    }


def _semantic_binding(decision: Mapping[str, Any]) -> tuple[str | None, dict[str, str], list[str]]:
    candidate_raw = decision.get("candidate_id")
    candidate_id = str(candidate_raw).strip() if candidate_raw is not None else ""
    reasons: list[str] = []
    if not candidate_id:
        reasons.append("SEMANTIC_DECISION_MISSING_CANDIDATE_ID")

    evidence = decision.get("evidence")
    native_rows = evidence.get("native_markets") if isinstance(evidence, Mapping) else None
    expected: dict[str, str] = {}
    if not isinstance(native_rows, Sequence) or isinstance(native_rows, (str, bytes)):
        reasons.append("SEMANTIC_DECISION_NATIVE_EVIDENCE_MISSING")
    else:
        for row in native_rows:
            if not isinstance(row, Mapping):
                reasons.append("SEMANTIC_DECISION_NATIVE_EVIDENCE_INVALID")
                continue
            venue = str(row.get("venue", "")).strip().lower()
            native_market_id = str(row.get("native_market_id", "")).strip()
            if not venue or not native_market_id or venue in expected:
                reasons.append("SEMANTIC_DECISION_NATIVE_EVIDENCE_INVALID")
                continue
            expected[venue] = native_market_id
    if set(expected) != {"kalshi", "polymarket"}:
        reasons.append("SEMANTIC_DECISION_NATIVE_VENUE_PAIR_INVALID")
    return candidate_id or None, expected, sorted(set(reasons))


def calculate_shadow(
    semantic_decision: Mapping[str, Any] | Any,
    book_a: Mapping[str, Any] | Any,
    book_b: Mapping[str, Any] | Any,
    policy: Mapping[str, Any] | ShadowPolicy | Any,
    *,
    evaluated_at: str | datetime,
) -> dict[str, Any]:
    """Calculate two aggressive complementary-outcome directions.

    All validation failures are returned as evidence-bearing, fail-closed
    results.  An alert is possible only for an independently verified semantic
    decision and two fresh, complete venue-native books with sufficient depth.
    The returned ``live_eligible`` value is unconditionally ``False``.
    """

    result = _base_result(semantic_decision, evaluated_at)
    result["book_evidence"] = [_book_evidence(book_a, "A"), _book_evidence(book_b, "B")]
    evaluated = _utc_datetime(evaluated_at)
    global_reasons: list[str] = []
    if evaluated is None:
        global_reasons.append("MISSING_OR_INVALID_EVALUATED_AT")
    else:
        result["evaluated_at"] = _utc_iso(evaluated)

    expected_candidate_id: str | None = None
    expected_native_ids: dict[str, str] = {}
    if not isinstance(semantic_decision, Mapping):
        global_reasons.append("MISSING_OR_INVALID_SEMANTIC_DECISION")
    else:
        if semantic_decision.get("status") != "VERIFIED_EQUIVALENT":
            global_reasons.append("SEMANTIC_DECISION_NOT_VERIFIED_EQUIVALENT")
        expected_candidate_id, expected_native_ids, binding_reasons = _semantic_binding(semantic_decision)
        global_reasons.extend(binding_reasons)

    policy_values, policy_reasons = _policy_values(policy)
    global_reasons.extend(policy_reasons)
    if policy_values is not None:
        result["policy"] = policy_values.as_dict()

    parsed_a: _BookValues | None = None
    parsed_b: _BookValues | None = None
    if evaluated is not None and policy_values is not None:
        parsed_a, reasons_a, evidence_a = _book_values(
            book_a,
            label="A",
            evaluated_at=evaluated,
            policy=policy_values,
        )
        parsed_b, reasons_b, evidence_b = _book_values(
            book_b,
            label="B",
            evaluated_at=evaluated,
            policy=policy_values,
        )
        global_reasons.extend(reasons_a)
        global_reasons.extend(reasons_b)
        result["book_evidence"] = [evidence_a, evidence_b]
        if parsed_a is not None and parsed_b is not None:
            for parsed in (parsed_a, parsed_b):
                if expected_candidate_id is not None and parsed.candidate_id != expected_candidate_id:
                    global_reasons.append(f"BOOK_{parsed.label}_CANDIDATE_ID_MISMATCH")
                expected_native_id = expected_native_ids.get(parsed.venue)
                if expected_native_id is None or parsed.native_market_id != expected_native_id:
                    global_reasons.append(f"BOOK_{parsed.label}_NATIVE_MARKET_ID_MISMATCH")
            if parsed_a.venue == parsed_b.venue:
                global_reasons.append("BOOKS_ARE_NOT_CROSS_VENUE")
            skew_seconds = abs((parsed_a.source_at - parsed_b.source_at).total_seconds())
            result["cross_venue_source_skew_seconds"] = skew_seconds
            if skew_seconds > policy_values.max_cross_venue_skew_seconds:
                global_reasons.append("CROSS_VENUE_SOURCE_SKEW_EXCEEDS_LIMIT")

    global_reasons = sorted(set(global_reasons))
    if global_reasons or parsed_a is None or parsed_b is None or policy_values is None:
        result["reasons"] = global_reasons or ["INCOMPLETE_VALIDATED_INPUTS"]
        if _has_fee_evidence_failure(result["reasons"]):
            result["status"] = "FEE_EVIDENCE_UNAVAILABLE"
        result["directions"] = [
            _empty_direction(name, outcome_a, outcome_b, result["reasons"])
            for name, outcome_a, outcome_b in _DIRECTIONS
        ]
        if policy_values is not None:
            for direction in result["directions"]:
                direction["requested_size"] = policy_values.requested_size
        return result

    directions = [
        _calculate_direction(name, outcome_a, outcome_b, parsed_a, parsed_b, policy_values)
        for name, outcome_a, outcome_b in _DIRECTIONS
    ]
    result["directions"] = directions
    result["alert"] = any(direction["alert"] for direction in directions)
    direction_reasons = sorted({reason for direction in directions for reason in direction["reasons"]})
    if result["alert"]:
        result["status"] = "SHADOW_ALERT"
    elif _has_fee_evidence_failure(direction_reasons):
        result["status"] = "FEE_EVIDENCE_UNAVAILABLE"
    else:
        result["status"] = "NO_EXECUTABLE_SHADOW_EDGE"
    if not result["alert"]:
        result["reasons"] = direction_reasons
    return result
