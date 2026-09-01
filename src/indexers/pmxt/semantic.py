"""Deterministic semantic verification for PMXT-discovered market pairs.

PMXT supplies candidate relationships only.  This module deliberately accepts
separately collected venue-native metadata and never promotes PMXT market IDs
to native identifiers.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

VERIFIED_EQUIVALENT = "VERIFIED_EQUIVALENT"
REJECTED = "REJECTED"
NEEDS_REVIEW = "NEEDS_REVIEW"

_EXPECTED_VENUES = ("kalshi", "polymarket")
_REQUIRED_NATIVE_FIELDS = (
    "candidate_id",
    "side",
    "venue",
    "pmxt_market_id",
    "native_market_id",
    "native_event_id",
    "native_outcome_ids",
    "native_outcome_labels",
    "raw_sha256",
    "rule_hash",
    "normalized_rules",
    "proposition",
    "outcome_polarity",
    "close_time",
    "expiration_time",
    "settlement_authority",
    "resolution_source",
    "resolution_criteria",
    "void_cancel",
    "material_edge_cases",
    "settlement_delay_seconds",
    "market_type",
    "status",
    "requests",
    "raw_response",
)
_VENUE_REQUIRED_NATIVE_FIELDS = {
    "kalshi": ("native_series_id", "event_mutually_exclusive"),
    "polymarket": ("native_condition_id", "negative_risk"),
}
_STRICT_SEMANTIC_FIELDS = (
    "settlement_authority",
    "resolution_source",
    "resolution_criteria",
    "void_cancel",
)
_MISMATCH_CODES = {
    "settlement_authority": "SETTLEMENT_AUTHORITY_MISMATCH",
    "resolution_source": "RESOLUTION_SOURCE_MISMATCH",
    "resolution_criteria": "RESOLUTION_CRITERIA_MISMATCH",
    "void_cancel": "VOID_CANCEL_MISMATCH",
}
_MATERIAL_HORIZON_FIELDS = (
    "expiration_time",
    "expected_expiration_time",
    "latest_expiration_time",
)
_HORIZON_MISMATCH_CODES = {
    "expiration_time": "EXPIRATION_TIME_MISMATCH",
    "expected_expiration_time": "EXPECTED_EXPIRATION_TIME_MISMATCH",
    "latest_expiration_time": "LATEST_EXPIRATION_TIME_MISMATCH",
}


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    return normalized or None


def _canonical_text(value: Any) -> str | None:
    normalized = _text(value)
    return normalized.casefold() if normalized is not None else None


def _venue(value: Any) -> str | None:
    return _canonical_text(value)


def _has_explicit_value(field: str, evidence: Mapping[str, Any]) -> bool:
    if field not in evidence or evidence[field] is None:
        return False
    value = evidence[field]
    if isinstance(value, str):
        return _text(value) is not None
    if field == "material_edge_cases":
        # An empty collection is an explicit statement that no edge cases were
        # identified; absence/None is not.
        return True
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return len(value) > 0
    return True


def _canonical_value(value: Any) -> Any:
    """Return a JSON-like canonical form suitable for strict comparison."""

    if isinstance(value, str):
        return _canonical_text(value)
    if isinstance(value, Mapping):
        items = sorted(
            ((_canonical_text(str(key)) or "", _canonical_value(item)) for key, item in value.items()),
            key=lambda item: item[0],
        )
        return dict(items)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        normalized = [_canonical_value(item) for item in value]
        return sorted(normalized, key=repr)
    return value


def normalized_material_edge_cases(venue: str, rules: Mapping[str, Any]) -> dict[str, Any]:
    """Return raw-derived edge-case fields not covered by dedicated axes.

    Resolution horizons and grouped-market flags are intentionally excluded:
    the verifier compares those fields separately with stricter semantics.
    An empty mapping is an explicit statement that no additional modeled edge
    cases were present in the retained native rule fields.
    """

    normalized_venue = _venue(venue)
    if normalized_venue == "kalshi":
        fields = (
            "can_close_early",
            "early_close_condition",
            "strike_type",
            "floor_strike",
            "cap_strike",
            "functional_strike",
            "custom_strike",
            "event_collateral_return_type",
        )
    elif normalized_venue == "polymarket":
        fields = ("uma_resolution_status",)
    else:
        raise ValueError(f"unsupported venue for material edge cases: {venue!r}")
    return {field: rules[field] for field in fields if field in rules and rules[field] is not None}


def _parse_close_time(value: Any) -> tuple[datetime | None, str | None, int | None]:
    text = _text(value)
    if text is None:
        return None, None, None
    iso_value = f"{text[:-1]}+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError:
        return None, None, None
    offset = parsed.utcoffset() if parsed.tzinfo is not None else None
    if offset is None:
        return None, None, None
    utc_value = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return parsed, utc_value, int(offset.total_seconds())


def _normalize_polarity(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    normalized = {(_canonical_text(str(key)) or ""): _canonical_value(item) for key, item in value.items()}
    if set(normalized) != {"yes", "no"}:
        return None
    if normalized["yes"] in (None, "") or normalized["no"] in (None, ""):
        return None
    if normalized["yes"] == normalized["no"]:
        return None
    return {"YES": normalized["yes"], "NO": normalized["no"]}


def _settlement_delay(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError):
        return None
    if delay < 0 or delay != delay or delay in (float("inf"), float("-inf")):
        return None
    return delay


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


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


_INVALID_JSON = object()


def _decoded_request_json(raw_request: Mapping[str, Any]) -> Any:
    """Decode one retained response body without trusting its parsed sibling."""

    encoded = raw_request.get("raw_body_base64")
    if not isinstance(encoded, str):
        return _INVALID_JSON
    try:
        raw_body = base64.b64decode(encoded, validate=True)
        return json.loads(
            raw_body.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _INVALID_JSON


def _same_json(left: Any, right: Any) -> bool:
    left_hash = _canonical_json_sha256(left)
    return left_hash is not None and left_hash == _canonical_json_sha256(right)


def _native_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _native_timestamp(value: Any) -> str | None:
    """Mirror the native capture's timestamp normalization for provenance checks."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000
        try:
            return datetime.fromtimestamp(number, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except (OSError, OverflowError, ValueError):
            return None
    text = _native_string(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.isoformat().replace("+00:00", "Z")


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _metadata_integrity_reasons(venue: str, evidence: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    raw_digest = evidence.get("raw_sha256")
    computed_raw_digest = _canonical_json_sha256(evidence.get("raw_response"))
    if not _is_sha256(raw_digest):
        reasons.append(f"{venue.upper()}_RAW_SHA256_INVALID")
    elif computed_raw_digest is None or str(raw_digest).lower() != computed_raw_digest:
        reasons.append(f"{venue.upper()}_RAW_SHA256_MISMATCH")
    rule_digest = evidence.get("rule_hash")
    computed_rule_digest = _canonical_json_sha256(evidence.get("normalized_rules"))
    if not _is_sha256(rule_digest):
        reasons.append(f"{venue.upper()}_RULE_HASH_INVALID")
    elif computed_rule_digest is None or str(rule_digest).lower() != computed_rule_digest:
        reasons.append(f"{venue.upper()}_RULE_HASH_MISMATCH")
    return reasons


def _acquisition_reasons(venue: str, evidence: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if evidence.get("status") != "RESOLVED":
        reasons.append(f"{venue.upper()}_NATIVE_STATUS_NOT_RESOLVED")
    raw_requests = evidence.get("requests")
    if not isinstance(raw_requests, Sequence) or isinstance(raw_requests, (str, bytes)) or not raw_requests:
        return [*reasons, f"{venue.upper()}_ACQUISITION_EVIDENCE_MISSING"]
    for index, raw_request in enumerate(raw_requests):
        prefix = f"{venue.upper()}_ACQUISITION_{index}"
        if not isinstance(raw_request, Mapping):
            reasons.append(f"{prefix}_INVALID")
            continue
        if raw_request.get("method") != "GET":
            reasons.append(f"{prefix}_METHOD_NOT_GET")
        if raw_request.get("freshness_basis") != "LOCAL_RECEIPT_BOUNDED":
            reasons.append(f"{prefix}_FRESHNESS_BASIS_INVALID")
        if raw_request.get("body_complete") is not True:
            reasons.append(f"{prefix}_BODY_INCOMPLETE")
        status_code = raw_request.get("status_code")
        if isinstance(status_code, bool) or not isinstance(status_code, int) or not 200 <= status_code < 300:
            reasons.append(f"{prefix}_HTTP_STATUS_INVALID")
        raw_body_hash = raw_request.get("raw_body_hash")
        if not _is_sha256(raw_body_hash):
            reasons.append(f"{prefix}_RAW_BODY_HASH_INVALID")
        raw_body_base64 = raw_request.get("raw_body_base64")
        try:
            raw_body = base64.b64decode(raw_body_base64, validate=True) if isinstance(raw_body_base64, str) else None
        except (binascii.Error, ValueError):
            raw_body = None
        if raw_body is None:
            reasons.append(f"{prefix}_RAW_BODY_BYTES_INVALID")
        elif _is_sha256(raw_body_hash) and hashlib.sha256(raw_body).hexdigest() != str(raw_body_hash).lower():
            reasons.append(f"{prefix}_RAW_BODY_HASH_MISMATCH")
        if raw_body is not None and _decoded_request_json(raw_request) is _INVALID_JSON:
            reasons.append(f"{prefix}_RAW_BODY_JSON_INVALID")
        response_headers = raw_request.get("response_headers")
        response_header_items = raw_request.get("response_header_items")
        valid_header_items = (
            isinstance(response_header_items, Sequence)
            and not isinstance(response_header_items, (str, bytes))
            and all(
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and len(item) == 2
                and all(isinstance(part, str) for part in item)
                for item in response_header_items
            )
        )
        if (
            not isinstance(response_headers, Mapping)
            or not all(isinstance(key, str) and isinstance(value, str) for key, value in response_headers.items())
            or not valid_header_items
        ):
            reasons.append(f"{prefix}_RESPONSE_HEADERS_INVALID")
        for envelope_key in ("http_date", "age_header", "cache_control", "request_id"):
            if envelope_key not in raw_request:
                reasons.append(f"{prefix}_{envelope_key.upper()}_MISSING")
        request_ns = raw_request.get("request_monotonic_ns")
        response_ns = raw_request.get("response_monotonic_ns")
        if (
            isinstance(request_ns, bool)
            or not isinstance(request_ns, int)
            or isinstance(response_ns, bool)
            or not isinstance(response_ns, int)
            or response_ns < request_ns
        ):
            reasons.append(f"{prefix}_MONOTONIC_WINDOW_INVALID")
        rtt = raw_request.get("rtt_ms")
        if isinstance(rtt, bool) or not isinstance(rtt, (int, float)) or not math.isfinite(float(rtt)) or rtt < 0:
            reasons.append(f"{prefix}_RTT_INVALID")
        elif (
            isinstance(request_ns, int)
            and not isinstance(request_ns, bool)
            and isinstance(response_ns, int)
            and not isinstance(response_ns, bool)
            and response_ns >= request_ns
            and abs(float(rtt) - ((response_ns - request_ns) / 1_000_000)) > 0.001
        ):
            reasons.append(f"{prefix}_RTT_WINDOW_MISMATCH")
        request_wall, _, _ = _parse_close_time(raw_request.get("request_wall_utc"))
        response_wall, _, _ = _parse_close_time(raw_request.get("response_wall_utc"))
        if request_wall is None:
            reasons.append(f"{prefix}_REQUEST_WALL_UTC_INVALID")
        if response_wall is None:
            reasons.append(f"{prefix}_RESPONSE_WALL_UTC_INVALID")
        if request_wall is not None and response_wall is not None and response_wall < request_wall:
            reasons.append(f"{prefix}_RESPONSE_WALL_BEFORE_REQUEST")
    return reasons


def _candidate_binding_reasons(
    candidate: Mapping[str, Any],
    by_venue: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Bind each native row to one exact candidate side, never by venue inference alone."""

    reasons: list[str] = []
    candidate_id = candidate.get("candidate_id")
    if _text(candidate_id) is None:
        reasons.append("PMXT_CANDIDATE_ID_INVALID")
    observed_sides: list[str] = []
    for venue in _EXPECTED_VENUES:
        evidence = by_venue[venue]
        prefix = venue.upper()
        if evidence.get("candidate_id") != candidate_id:
            reasons.append(f"{prefix}_CANDIDATE_ID_BINDING_MISMATCH")
        side = evidence.get("side")
        if side not in {"a", "b"}:
            reasons.append(f"{prefix}_CANDIDATE_SIDE_BINDING_INVALID")
            continue
        observed_sides.append(side)
        if _venue(candidate.get(f"venue_{side}")) != venue:
            reasons.append(f"{prefix}_CANDIDATE_SIDE_VENUE_BINDING_MISMATCH")
        if evidence.get("pmxt_market_id") != candidate.get(f"pmxt_market_id_{side}"):
            reasons.append(f"{prefix}_PMXT_MARKET_ID_BINDING_MISMATCH")
    if sorted(observed_sides) != ["a", "b"]:
        reasons.append("NATIVE_CANDIDATE_SIDE_BINDING_NOT_ONE_TO_ONE")
    return reasons


def _mapping_record(payload: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    nested = payload.get(key)
    return nested if isinstance(nested, Mapping) and nested else payload


def _polymarket_records(payload: Any) -> list[Mapping[str, Any]]:
    records = payload
    if isinstance(payload, Mapping):
        records = payload.get("markets", payload.get("data", payload.get("results", [])))
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []
    return [item for item in records if isinstance(item, Mapping)]


def _raw_response_provenance_reasons(venue: str, evidence: Mapping[str, Any]) -> list[str]:
    """Prove parsed metadata is backed by the retained response bytes."""

    reasons: list[str] = []
    raw_requests = evidence.get("requests")
    requests = (
        list(raw_requests) if isinstance(raw_requests, Sequence) and not isinstance(raw_requests, (str, bytes)) else []
    )
    raw_response = evidence.get("raw_response")
    if not isinstance(raw_response, Mapping):
        return [f"{venue.upper()}_RAW_RESPONSE_PROVENANCE_INVALID"]

    if venue == "kalshi":
        entry_names = ("market", "event_metadata", "event", "series")
        expected_paths = (
            f"/markets/{evidence.get('native_market_id')}",
            f"/events/{evidence.get('native_event_id')}/metadata",
            f"/events/{evidence.get('native_event_id')}",
            f"/series/{evidence.get('native_series_id')}",
        )
        if len(requests) != len(entry_names):
            reasons.append("KALSHI_ACQUISITION_REQUEST_SEQUENCE_INVALID")
        for index, (entry_name, expected_path) in enumerate(zip(entry_names, expected_paths)):
            if index >= len(requests) or not isinstance(requests[index], Mapping):
                reasons.append(f"KALSHI_{entry_name.upper()}_RAW_BODY_PROVENANCE_MISSING")
                continue
            request = requests[index]
            if request.get("path") != expected_path:
                reasons.append(f"KALSHI_{entry_name.upper()}_REQUEST_PATH_MISMATCH")
            decoded = _decoded_request_json(request)
            if decoded is _INVALID_JSON:
                continue
            if entry_name not in raw_response or not _same_json(decoded, raw_response.get(entry_name)):
                reasons.append(f"KALSHI_{entry_name.upper()}_RAW_BODY_RESPONSE_MISMATCH")
        return reasons

    if len(requests) != 1:
        reasons.append("POLYMARKET_ACQUISITION_REQUEST_SEQUENCE_INVALID")
    if not requests or not isinstance(requests[0], Mapping):
        return [*reasons, "POLYMARKET_MARKETS_RAW_BODY_PROVENANCE_MISSING"]
    request = requests[0]
    if request.get("path") != "/markets":
        reasons.append("POLYMARKET_MARKETS_REQUEST_PATH_MISMATCH")
    decoded = _decoded_request_json(request)
    if decoded is not _INVALID_JSON:
        if "markets" not in raw_response or not _same_json(decoded, raw_response.get("markets")):
            reasons.append("POLYMARKET_MARKETS_RAW_BODY_RESPONSE_MISMATCH")
        selected_market = raw_response.get("selected_market")
        records = _polymarket_records(decoded)
        if not isinstance(selected_market, Mapping) or not any(
            _same_json(selected_market, record) for record in records
        ):
            reasons.append("POLYMARKET_SELECTED_MARKET_NOT_IN_RAW_RESPONSE")
    return reasons


def _append_consistency_reason(
    reasons: list[str],
    venue: str,
    field: str,
    actual: Any,
    expected: Any,
    *,
    source: str = "NORMALIZED_RULES",
) -> None:
    if _canonical_value(actual) != _canonical_value(expected):
        reasons.append(f"{venue.upper()}_TOP_LEVEL_{field.upper()}_{source}_MISMATCH")


def _kalshi_consistency_reasons(evidence: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    rules = evidence.get("normalized_rules")
    raw_response = evidence.get("raw_response")
    if not isinstance(rules, Mapping) or not isinstance(raw_response, Mapping):
        return ["KALSHI_NORMALIZED_RULES_PROVENANCE_INVALID"]
    market = _mapping_record(raw_response.get("market"), "market")
    event_metadata = raw_response.get("event_metadata")
    event = _mapping_record(raw_response.get("event"), "event")
    series = _mapping_record(raw_response.get("series"), "series")
    event_metadata = event_metadata if isinstance(event_metadata, Mapping) else {}

    settlement_sources: list[Any] = []
    for source_record in (
        *_json_list(event_metadata.get("settlement_sources")),
        *_json_list(event.get("settlement_sources")),
        *_json_list(series.get("settlement_sources")),
    ):
        if source_record not in settlement_sources:
            settlement_sources.append(source_record)
    expected_rule_fields = {
        "rules_primary": _native_string(market.get("rules_primary")),
        "rules_secondary": _native_string(market.get("rules_secondary")),
        "open_time": _native_timestamp(market.get("open_time")),
        "close_time": _native_timestamp(market.get("close_time")),
        "expiration_time": _native_timestamp(market.get("expiration_time")),
        "expected_expiration_time": _native_timestamp(market.get("expected_expiration_time")),
        "latest_expiration_time": _native_timestamp(market.get("latest_expiration_time")),
        "settlement_timer_seconds": market.get("settlement_timer_seconds"),
        "settlement_sources": settlement_sources,
        "can_close_early": market.get("can_close_early"),
        "early_close_condition": _native_string(market.get("early_close_condition")),
        "void_cancel_rules": _native_string(market.get("void_cancel_rules")),
        "strike_type": _native_string(market.get("strike_type")),
        "floor_strike": market.get("floor_strike"),
        "cap_strike": market.get("cap_strike"),
        "functional_strike": market.get("functional_strike"),
        "custom_strike": market.get("custom_strike"),
        "market_type": _native_string(market.get("market_type")),
        "mve_collection_ticker": _native_string(market.get("mve_collection_ticker")),
        "mve_selected_legs": _json_list(market.get("mve_selected_legs")),
        "event_mutually_exclusive": event.get("mutually_exclusive"),
        "event_collateral_return_type": _native_string(event.get("collateral_return_type")),
        "outcome_polarity": {"YES": "YES", "NO": "NO"},
    }
    for field, expected in expected_rule_fields.items():
        if not _same_json(rules.get(field), expected):
            reasons.append(f"KALSHI_NORMALIZED_RULES_RAW_{field.upper()}_MISMATCH")

    source_names = sorted(
        name
        for item in settlement_sources
        if isinstance(item, Mapping) and (name := _native_string(item.get("name"))) is not None
    )
    source_urls = sorted(
        url
        for item in settlement_sources
        if isinstance(item, Mapping) and (url := _native_string(item.get("url"))) is not None
    )
    criteria_parts = [
        value
        for value in (_native_string(rules.get("rules_primary")), _native_string(rules.get("rules_secondary")))
        if value
    ]
    comparisons = {
        "proposition": (_native_string(market.get("title")), "RAW_RESPONSE"),
        "open_time": (rules.get("open_time"), "NORMALIZED_RULES"),
        "close_time": (rules.get("close_time"), "NORMALIZED_RULES"),
        "expiration_time": (rules.get("expiration_time"), "NORMALIZED_RULES"),
        "expected_expiration_time": (rules.get("expected_expiration_time"), "NORMALIZED_RULES"),
        "latest_expiration_time": (rules.get("latest_expiration_time"), "NORMALIZED_RULES"),
        "settlement_authority": (" | ".join(source_names) if source_names else None, "NORMALIZED_RULES"),
        "resolution_source": (" | ".join(source_urls) if source_urls else None, "NORMALIZED_RULES"),
        "resolution_criteria": ("\n".join(criteria_parts) if criteria_parts else None, "NORMALIZED_RULES"),
        "void_cancel": (rules.get("void_cancel_rules"), "NORMALIZED_RULES"),
        "material_edge_cases": (normalized_material_edge_cases("kalshi", rules), "NORMALIZED_RULES"),
        "settlement_delay_seconds": (_settlement_delay(rules.get("settlement_timer_seconds")), "NORMALIZED_RULES"),
        "outcome_polarity": (_normalize_polarity(rules.get("outcome_polarity")), "NORMALIZED_RULES"),
        "market_type": (rules.get("market_type"), "NORMALIZED_RULES"),
        "event_mutually_exclusive": (rules.get("event_mutually_exclusive"), "NORMALIZED_RULES"),
        "mve_collection_ticker": (rules.get("mve_collection_ticker"), "NORMALIZED_RULES"),
        "mve_selected_legs": (rules.get("mve_selected_legs"), "NORMALIZED_RULES"),
    }
    for field, (expected, source) in comparisons.items():
        actual = evidence.get(field)
        if field == "settlement_delay_seconds":
            actual = _settlement_delay(actual)
        elif field == "outcome_polarity":
            actual = _normalize_polarity(actual)
        _append_consistency_reason(reasons, "kalshi", field, actual, expected, source=source)
    native_market_id = _native_string(market.get("ticker"))
    native_event_id = _native_string(market.get("event_ticker"))
    native_series_id = _native_string(event.get("series_ticker")) or _native_string(market.get("series_ticker"))
    raw_identity_fields = {
        "native_market_id": native_market_id,
        "native_event_id": native_event_id,
        "native_series_id": native_series_id,
        "native_outcome_ids": {"YES": native_market_id, "NO": native_market_id},
        "native_outcome_labels": {
            "YES": _native_string(market.get("yes_sub_title")) or "YES",
            "NO": _native_string(market.get("no_sub_title")) or "NO",
        },
    }
    for field, expected in raw_identity_fields.items():
        _append_consistency_reason(
            reasons,
            "kalshi",
            field,
            evidence.get(field),
            expected,
            source="RAW_RESPONSE",
        )
    return reasons


def _polymarket_consistency_reasons(evidence: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    rules = evidence.get("normalized_rules")
    raw_response = evidence.get("raw_response")
    if not isinstance(rules, Mapping) or not isinstance(raw_response, Mapping):
        return ["POLYMARKET_NORMALIZED_RULES_PROVENANCE_INVALID"]
    selected = raw_response.get("selected_market")
    if not isinstance(selected, Mapping):
        return ["POLYMARKET_SELECTED_MARKET_PROVENANCE_INVALID"]

    labels = [_native_string(value) for value in _json_list(selected.get("outcomes"))]
    token_ids = [_native_string(value) for value in _json_list(selected.get("clobTokenIds"))]
    if len(labels) != len(token_ids):
        token_ids = []
    expected_outcome_ids = {
        label.upper(): token
        for label, token in zip(labels, token_ids)
        if label is not None and token is not None and label.upper() in {"YES", "NO"}
    }
    expected_outcome_labels = {
        label.upper(): label for label in labels if label is not None and label.upper() in {"YES", "NO"}
    }
    expected_outcome_polarity = {
        label.upper(): label.upper() for label in labels if label is not None and label.upper() in {"YES", "NO"}
    }
    expected_rule_fields = {
        "question": _native_string(selected.get("question")),
        "description": _native_string(selected.get("description")),
        "resolution_source": _native_string(selected.get("resolutionSource")),
        "resolved_by": _native_string(selected.get("resolvedBy")),
        "start_date": _native_timestamp(selected.get("startDate") or selected.get("startDateIso")),
        "end_date": _native_timestamp(selected.get("endDate") or selected.get("endDateIso")),
        "uma_end_date": _native_timestamp(selected.get("umaEndDate")),
        "uma_resolution_status": selected.get("umaResolutionStatus"),
        "void_cancel_rules": _native_string(selected.get("voidCancelRules"))
        or _native_string(selected.get("void_cancel_rules")),
        "settlement_delay_seconds": _settlement_delay(selected.get("settlementDelaySeconds")),
        "neg_risk": selected.get("negRisk"),
        "outcome_polarity": expected_outcome_polarity,
    }
    for field, expected in expected_rule_fields.items():
        if not _same_json(rules.get(field), expected):
            reasons.append(f"POLYMARKET_NORMALIZED_RULES_RAW_{field.upper()}_MISMATCH")

    comparisons = {
        "proposition": rules.get("question"),
        "open_time": rules.get("start_date"),
        "close_time": rules.get("end_date"),
        "expiration_time": rules.get("uma_end_date"),
        "settlement_authority": rules.get("resolved_by"),
        "resolution_source": rules.get("resolution_source"),
        "resolution_criteria": rules.get("description"),
        "void_cancel": rules.get("void_cancel_rules"),
        "material_edge_cases": normalized_material_edge_cases("polymarket", rules),
        "settlement_delay_seconds": _settlement_delay(rules.get("settlement_delay_seconds")),
        "outcome_polarity": _normalize_polarity(rules.get("outcome_polarity")),
        "negative_risk": rules.get("neg_risk"),
        "market_type": "binary" if set(expected_outcome_ids) == {"YES", "NO"} else None,
        "native_outcome_ids": expected_outcome_ids,
        "native_outcome_labels": expected_outcome_labels,
    }
    for field, expected in comparisons.items():
        actual = evidence.get(field)
        if field == "settlement_delay_seconds":
            actual = _settlement_delay(actual)
        elif field == "outcome_polarity":
            actual = _normalize_polarity(actual)
        _append_consistency_reason(reasons, "polymarket", field, actual, expected)

    # Gamma exposes no separate expected/latest expiration fields today.  Do
    # not allow unproven top-level values to become comparable native facts.
    for field in ("expected_expiration_time", "latest_expiration_time"):
        _append_consistency_reason(
            reasons,
            "polymarket",
            field,
            evidence.get(field),
            None,
            source="NATIVE_SCHEMA",
        )

    expected_market_id = _native_string(selected.get("id"))
    expected_condition_id = _native_string(selected.get("conditionId")) or _native_string(selected.get("condition_id"))
    _append_consistency_reason(
        reasons,
        "polymarket",
        "native_market_id",
        evidence.get("native_market_id"),
        expected_market_id,
        source="RAW_RESPONSE",
    )
    _append_consistency_reason(
        reasons,
        "polymarket",
        "native_condition_id",
        evidence.get("native_condition_id"),
        expected_condition_id,
        source="RAW_RESPONSE",
    )
    parent_events = [item for item in _json_list(selected.get("events")) if isinstance(item, Mapping)]
    native_event_ids = sorted(
        {event_id for event in parent_events if (event_id := _native_string(event.get("id"))) is not None}
    )
    _append_consistency_reason(
        reasons,
        "polymarket",
        "native_event_id",
        evidence.get("native_event_id"),
        native_event_ids[0] if len(native_event_ids) == 1 else None,
        source="RAW_RESPONSE",
    )
    return reasons


def _metadata_provenance_reasons(
    candidate: Mapping[str, Any],
    by_venue: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    reasons = _candidate_binding_reasons(candidate, by_venue)
    for venue in _EXPECTED_VENUES:
        evidence = by_venue[venue]
        reasons.extend(_acquisition_reasons(venue, evidence))
        reasons.extend(_metadata_integrity_reasons(venue, evidence))
        reasons.extend(_raw_response_provenance_reasons(venue, evidence))
    reasons.extend(_kalshi_consistency_reasons(by_venue["kalshi"]))
    reasons.extend(_polymarket_consistency_reasons(by_venue["polymarket"]))
    return reasons


def _native_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    _, close_time_utc, timezone_offset_seconds = _parse_close_time(evidence.get("close_time"))
    raw_requests = evidence.get("requests")
    requests = raw_requests if isinstance(raw_requests, Sequence) and not isinstance(raw_requests, (str, bytes)) else []
    return {
        "candidate_id": evidence.get("candidate_id"),
        "side": evidence.get("side"),
        "venue": _venue(evidence.get("venue")),
        "pmxt_market_id": evidence.get("pmxt_market_id"),
        # These three values come exclusively from venue-native evidence.
        "native_market_id": evidence.get("native_market_id"),
        "native_event_id": evidence.get("native_event_id"),
        "native_series_id": evidence.get("native_series_id"),
        "native_condition_id": evidence.get("native_condition_id"),
        "native_outcome_ids": _canonical_value(evidence.get("native_outcome_ids")),
        "native_outcome_labels": _canonical_value(evidence.get("native_outcome_labels")),
        "raw_sha256": evidence.get("raw_sha256"),
        "rule_hash": evidence.get("rule_hash"),
        "status": evidence.get("status"),
        "reason_code": evidence.get("reason_code"),
        "request_raw_body_hashes": [item.get("raw_body_hash") for item in requests if isinstance(item, Mapping)],
        "proposition": _canonical_value(evidence.get("proposition")),
        "outcome_polarity": _normalize_polarity(evidence.get("outcome_polarity")),
        "close_time": evidence.get("close_time"),
        "close_time_utc": close_time_utc,
        "timezone_offset_seconds": timezone_offset_seconds,
        "open_time": evidence.get("open_time"),
        "expiration_time": evidence.get("expiration_time"),
        "expected_expiration_time": evidence.get("expected_expiration_time"),
        "latest_expiration_time": evidence.get("latest_expiration_time"),
        "market_type": evidence.get("market_type"),
        "negative_risk": evidence.get("negative_risk"),
        "event_mutually_exclusive": evidence.get("event_mutually_exclusive"),
        "mve_collection_ticker": evidence.get("mve_collection_ticker"),
        "mve_selected_legs": _canonical_value(evidence.get("mve_selected_legs")),
        "clarification_or_revision_timestamps": _canonical_value(evidence.get("clarification_or_revision_timestamps")),
        "settlement_authority": _canonical_value(evidence.get("settlement_authority")),
        "resolution_source": _canonical_value(evidence.get("resolution_source")),
        "resolution_criteria": _canonical_value(evidence.get("resolution_criteria")),
        "void_cancel": _canonical_value(evidence.get("void_cancel")),
        "material_edge_cases": _canonical_value(evidence.get("material_edge_cases")),
        "settlement_delay_seconds": _settlement_delay(evidence.get("settlement_delay_seconds")),
    }


def _candidate_venues(candidate: Mapping[str, Any]) -> list[str | None]:
    return [_venue(candidate.get("venue_a")), _venue(candidate.get("venue_b"))]


def _decision(
    candidate: Mapping[str, Any],
    native_evidence: list[Mapping[str, Any]],
    *,
    status: str,
    reason_codes: list[str],
    comparisons: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summaries = [_native_summary(item) for item in native_evidence]
    summaries.sort(key=lambda item: (str(item.get("venue")), str(item.get("native_market_id"))))
    return {
        "schema_version": 1,
        "candidate_id": candidate.get("candidate_id"),
        "status": status,
        "live_eligible": False,
        "reason_codes": sorted(set(reason_codes)),
        "evidence": {
            "pmxt_discovery": {
                "cluster_id": candidate.get("cluster_id"),
                "relation": candidate.get("relation"),
                "raw_edge_present": candidate.get("raw_edge_present"),
                "venues": _candidate_venues(candidate),
                "pmxt_market_ids": [candidate.get("pmxt_market_id_a"), candidate.get("pmxt_market_id_b")],
                "pmxt_ids_are_native_authority": False,
            },
            "native_markets": summaries,
            "comparisons": comparisons or {},
        },
    }


def verify_semantics(
    candidate: Mapping[str, Any],
    native_a: Mapping[str, Any],
    native_b: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare one PMXT proposal against two authoritative native records.

    The function is pure and order-independent with respect to ``native_a`` and
    ``native_b``.  It performs no I/O, and every result remains ineligible for
    live execution even when the semantic evidence is equivalent.
    """

    native_evidence = [native_a, native_b]
    gate_reasons: list[str] = []
    relation = _canonical_text(candidate.get("relation"))
    if relation != "identity":
        gate_reasons.append("PMXT_RELATION_NOT_IDENTITY")
    if candidate.get("raw_edge_present") is not True:
        gate_reasons.append("PMXT_RAW_EDGE_MISSING")
    if sorted(_candidate_venues(candidate), key=lambda item: str(item)) != list(_EXPECTED_VENUES):
        gate_reasons.append("PMXT_VENUE_PAIR_UNSUPPORTED")
    if gate_reasons:
        return _decision(candidate, native_evidence, status=REJECTED, reason_codes=gate_reasons)

    native_venues = [_venue(item.get("venue")) for item in native_evidence]
    if sorted(native_venues, key=lambda item: str(item)) != list(_EXPECTED_VENUES):
        return _decision(
            candidate,
            native_evidence,
            status=NEEDS_REVIEW,
            reason_codes=["NATIVE_EVIDENCE_VENUE_PAIR_INVALID"],
        )

    by_venue = {str(_venue(item.get("venue"))): item for item in native_evidence}
    provenance_reasons: list[str] = []
    review_reasons: list[str] = []
    reject_reasons: list[str] = []
    comparisons: dict[str, Any] = {}

    for venue in _EXPECTED_VENUES:
        evidence = by_venue[venue]
        for field in _REQUIRED_NATIVE_FIELDS:
            if not _has_explicit_value(field, evidence):
                provenance_reasons.append(f"{venue.upper()}_MISSING_{field.upper()}")
        for field in _VENUE_REQUIRED_NATIVE_FIELDS[venue]:
            if not _has_explicit_value(field, evidence):
                provenance_reasons.append(f"{venue.upper()}_MISSING_{field.upper()}")
    provenance_reasons.extend(_metadata_provenance_reasons(candidate, by_venue))

    left = by_venue["kalshi"]
    right = by_venue["polymarket"]

    market_types = {
        "kalshi": _canonical_text(left.get("market_type")),
        "polymarket": _canonical_text(right.get("market_type")),
    }
    comparisons["market_type"] = {
        **market_types,
        "equivalent": market_types["kalshi"] == market_types["polymarket"] == "binary",
    }
    for venue, market_type in market_types.items():
        if market_type is not None and market_type != "binary":
            reject_reasons.append(f"{venue.upper()}_MARKET_TYPE_NOT_BINARY")
    if left.get("mve_collection_ticker") or left.get("mve_selected_legs"):
        reject_reasons.append("KALSHI_MULTIVARIATE_MARKET_UNSUPPORTED")

    kalshi_grouped = left.get("event_mutually_exclusive")
    polymarket_grouped = right.get("negative_risk")
    kalshi_group_flag_valid = isinstance(kalshi_grouped, bool)
    polymarket_group_flag_valid = isinstance(polymarket_grouped, bool)
    grouped_structure_present = kalshi_grouped is True or polymarket_grouped is True
    if kalshi_group_flag_valid and polymarket_group_flag_valid:
        flags_match = kalshi_grouped == polymarket_grouped
        comparisons["market_group_structure"] = {
            "kalshi_event_mutually_exclusive": kalshi_grouped,
            "polymarket_negative_risk": polymarket_grouped,
            "flags_match": flags_match,
            "candidate_set_equivalence_proven": not grouped_structure_present,
            "equivalent": flags_match and not grouped_structure_present,
        }
        if not flags_match:
            reject_reasons.append("MARKET_GROUP_STRUCTURE_MISMATCH")
        elif grouped_structure_present:
            review_reasons.append("MARKET_GROUP_CANDIDATE_SET_EQUIVALENCE_UNPROVEN")
    else:
        comparisons["market_group_structure"] = {
            "kalshi_event_mutually_exclusive": kalshi_grouped,
            "polymarket_negative_risk": polymarket_grouped,
            "flags_match": None,
            "candidate_set_equivalence_proven": False,
            "equivalent": False,
        }
        if not kalshi_group_flag_valid and _has_explicit_value("event_mutually_exclusive", left):
            review_reasons.append("KALSHI_EVENT_MUTUALLY_EXCLUSIVE_NOT_BOOLEAN")
        if not polymarket_group_flag_valid and _has_explicit_value("negative_risk", right):
            review_reasons.append("POLYMARKET_NEGATIVE_RISK_NOT_BOOLEAN")
        if grouped_structure_present:
            review_reasons.append("MARKET_GROUP_CANDIDATE_SET_EQUIVALENCE_UNPROVEN")

    native_labels: dict[str, dict[str, Any] | None] = {
        "kalshi": _normalize_polarity(left.get("native_outcome_labels")),
        "polymarket": _normalize_polarity(right.get("native_outcome_labels")),
    }
    for venue, labels in native_labels.items():
        if labels is None and _has_explicit_value("native_outcome_labels", by_venue[venue]):
            review_reasons.append(f"{venue.upper()}_NATIVE_OUTCOME_LABELS_NOT_EXPLICIT_YES_NO")
    if native_labels["kalshi"] is not None and native_labels["polymarket"] is not None:
        comparisons["native_outcome_labels"] = {
            "kalshi": native_labels["kalshi"],
            "polymarket": native_labels["polymarket"],
            "equivalence_basis": "OUTCOME_POLARITY_COMPARISON",
        }

    for venue, evidence in (("kalshi", left), ("polymarket", right)):
        outcome_ids = evidence.get("native_outcome_ids")
        normalized_ids = (
            {str(key).upper(): _text(value) for key, value in outcome_ids.items()}
            if isinstance(outcome_ids, Mapping)
            else {}
        )
        if set(normalized_ids) != {"YES", "NO"} or any(value is None for value in normalized_ids.values()):
            review_reasons.append(f"{venue.upper()}_NATIVE_OUTCOME_IDS_NOT_EXPLICIT_YES_NO")
        elif venue == "polymarket" and normalized_ids["YES"] == normalized_ids["NO"]:
            reject_reasons.append("POLYMARKET_NATIVE_OUTCOME_IDS_AMBIGUOUS")

    left_proposition = _canonical_text(left.get("proposition"))
    right_proposition = _canonical_text(right.get("proposition"))
    if left_proposition is not None and right_proposition is not None:
        equivalent = left_proposition == right_proposition
        comparisons["proposition"] = {
            "kalshi": left_proposition,
            "polymarket": right_proposition,
            "equivalent": equivalent,
        }
        if not equivalent:
            review_reasons.append("PROPOSITION_MISMATCH")

    left_polarity = _normalize_polarity(left.get("outcome_polarity"))
    right_polarity = _normalize_polarity(right.get("outcome_polarity"))
    for venue, polarity in (("kalshi", left_polarity), ("polymarket", right_polarity)):
        if polarity is None and _has_explicit_value("outcome_polarity", by_venue[venue]):
            review_reasons.append(f"{venue.upper()}_OUTCOME_POLARITY_NOT_EXPLICIT_YES_NO")
    if left_polarity is not None and right_polarity is not None:
        equivalent = left_polarity == right_polarity
        comparisons["outcome_polarity"] = {
            "kalshi": left_polarity,
            "polymarket": right_polarity,
            "equivalent": equivalent,
        }
        if not equivalent:
            if left_polarity["YES"] == right_polarity["NO"] and left_polarity["NO"] == right_polarity["YES"]:
                reject_reasons.append("OUTCOME_POLARITY_INVERTED_NON_IDENTITY")
            else:
                review_reasons.append("OUTCOME_POLARITY_MISMATCH")

    left_close, left_close_utc, left_offset = _parse_close_time(left.get("close_time"))
    right_close, right_close_utc, right_offset = _parse_close_time(right.get("close_time"))
    for venue, evidence, parsed in (
        ("kalshi", left, left_close),
        ("polymarket", right, right_close),
    ):
        if parsed is None and _has_explicit_value("close_time", evidence):
            review_reasons.append(f"{venue.upper()}_CLOSE_TIME_NOT_TIMEZONE_AWARE")
    if left_close is not None and right_close is not None:
        same_deadline = left_close_utc == right_close_utc
        same_timezone_offset = left_offset == right_offset
        comparisons["close_time"] = {
            "kalshi_utc": left_close_utc,
            "polymarket_utc": right_close_utc,
            "kalshi_timezone_offset_seconds": left_offset,
            "polymarket_timezone_offset_seconds": right_offset,
            "same_deadline": same_deadline,
            "same_timezone_offset": same_timezone_offset,
            "equivalent": same_deadline and same_timezone_offset,
        }
        if not same_deadline:
            reject_reasons.append("DEADLINE_MISMATCH")
        if not same_timezone_offset:
            reject_reasons.append("TIMEZONE_MISMATCH")

    for field in _MATERIAL_HORIZON_FIELDS:
        left_present = _has_explicit_value(field, left)
        right_present = _has_explicit_value(field, right)
        left_horizon, left_horizon_utc, left_horizon_offset = _parse_close_time(left.get(field))
        right_horizon, right_horizon_utc, right_horizon_offset = _parse_close_time(right.get(field))
        comparisons[field] = {
            "kalshi": left.get(field),
            "polymarket": right.get(field),
            "kalshi_utc": left_horizon_utc,
            "polymarket_utc": right_horizon_utc,
            "kalshi_timezone_offset_seconds": left_horizon_offset,
            "polymarket_timezone_offset_seconds": right_horizon_offset,
            "evidence_complete": left_present and right_present,
            "equivalent": (
                left_horizon is not None and right_horizon is not None and left_horizon_utc == right_horizon_utc
            ),
        }

        if not left_present and not right_present:
            if field == "expiration_time":
                review_reasons.append("EXPIRATION_TIME_EVIDENCE_MISSING")
            continue
        if left_present != right_present:
            review_reasons.append(f"{field.upper()}_EVIDENCE_ONE_SIDED")
            continue
        for venue, parsed in (("kalshi", left_horizon), ("polymarket", right_horizon)):
            if parsed is None:
                review_reasons.append(f"{venue.upper()}_{field.upper()}_NOT_TIMEZONE_AWARE")
        if left_horizon is not None and right_horizon is not None and left_horizon_utc != right_horizon_utc:
            reject_reasons.append(_HORIZON_MISMATCH_CODES[field])

    for field in _STRICT_SEMANTIC_FIELDS:
        if not (_has_explicit_value(field, left) and _has_explicit_value(field, right)):
            continue
        left_value = _canonical_value(left.get(field))
        right_value = _canonical_value(right.get(field))
        equivalent = left_value == right_value
        comparisons[field] = {
            "kalshi": left_value,
            "polymarket": right_value,
            "equivalent": equivalent,
        }
        if not equivalent:
            reject_reasons.append(_MISMATCH_CODES[field])

    if _has_explicit_value("material_edge_cases", left) and _has_explicit_value("material_edge_cases", right):
        left_edge_cases = _canonical_value(left.get("material_edge_cases"))
        right_edge_cases = _canonical_value(right.get("material_edge_cases"))
        equivalent = left_edge_cases == right_edge_cases
        comparisons["material_edge_cases"] = {
            "kalshi": left_edge_cases,
            "polymarket": right_edge_cases,
            "equivalent": equivalent,
        }
        if not equivalent:
            review_reasons.append("MATERIAL_EDGE_CASES_EQUIVALENCE_UNPROVEN")

    left_delay = _settlement_delay(left.get("settlement_delay_seconds"))
    right_delay = _settlement_delay(right.get("settlement_delay_seconds"))
    for venue, evidence, delay in (
        ("kalshi", left, left_delay),
        ("polymarket", right, right_delay),
    ):
        if delay is None and _has_explicit_value("settlement_delay_seconds", evidence):
            review_reasons.append(f"{venue.upper()}_SETTLEMENT_DELAY_INVALID")
    if left_delay is not None and right_delay is not None:
        equivalent = left_delay == right_delay
        comparisons["settlement_delay_seconds"] = {
            "kalshi": left_delay,
            "polymarket": right_delay,
            "equivalent": equivalent,
        }
        if not equivalent:
            reject_reasons.append("SETTLEMENT_DELAY_MISMATCH")

    if provenance_reasons:
        return _decision(
            candidate,
            native_evidence,
            status=NEEDS_REVIEW,
            reason_codes=provenance_reasons + reject_reasons + review_reasons,
            comparisons=comparisons,
        )
    if reject_reasons:
        return _decision(
            candidate,
            native_evidence,
            status=REJECTED,
            reason_codes=reject_reasons + review_reasons,
            comparisons=comparisons,
        )
    if review_reasons:
        return _decision(
            candidate,
            native_evidence,
            status=NEEDS_REVIEW,
            reason_codes=review_reasons,
            comparisons=comparisons,
        )
    return _decision(
        candidate,
        native_evidence,
        status=VERIFIED_EQUIVALENT,
        reason_codes=["ALL_REQUIRED_SEMANTICS_EQUIVALENT"],
        comparisons=comparisons,
    )
