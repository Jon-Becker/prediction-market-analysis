"""Fixture-first tests for GET-only venue-native PMXT evidence capture."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from src.indexers.pmxt.native import NativeEvidenceClient, NativeEvidenceError, canonical_json_sha256
from src.indexers.pmxt.shadow import calculate_shadow

_REVIEWED_KALSHI_FEE_PDF = b"%PDF-1.7\nfixture reviewed Kalshi fee schedule\n%%EOF\n"
_REVIEWED_KALSHI_FEE_PDF_SHA256 = hashlib.sha256(_REVIEWED_KALSHI_FEE_PDF).hexdigest()
_REVIEWED_KALSHI_FEE_BINDING = {
    "binding_id": "fixture-reviewed-kalshi-fee-pdf-v1",
    "document_effective_date": "2025-01-01",
    "supported_taker_fee_types": ["quadratic", "quadratic_with_maker_fees"],
    "taker_base_coefficient": "0.07",
    "trade_fee_rounding_quantum": "0.0001",
    "balance_precision_upper_bound": "0.0099",
}


def _json_response(payload: Any, *, request_id: str, status_code: int = 200) -> httpx.Response:
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return httpx.Response(
        status_code,
        stream=httpx.ByteStream(content),
        headers={
            "content-length": str(len(content)),
            "content-type": "application/json",
            "date": "Sat, 29 Aug 2026 21:00:00 GMT",
            "age": "0",
            "cache-control": "no-store",
            "x-request-id": request_id,
            "set-cookie": "must-not-be-retained=fixture-secret; HttpOnly",
            "authorization": "Bearer fixture-response-secret",
            "x-api-key": "fixture-response-api-key",
        },
    )


def _raw_body_hash(payload: Any) -> str:
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _fee_pdf_handler(seen_requests: list[httpx.Request], *, body: bytes = _REVIEWED_KALSHI_FEE_PDF):
    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        assert request.method == "GET"
        assert request.url == httpx.URL("https://kalshi.com/docs/kalshi-fee-schedule.pdf")
        assert request.headers["accept"] == "application/pdf"
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200,
            stream=httpx.ByteStream(body),
            headers={
                "content-length": str(len(body)),
                "content-type": "application/pdf",
                "date": "Sat, 29 Aug 2026 21:00:00 GMT",
                "age": "0",
                "cache-control": "no-store",
                "x-request-id": "kalshi-fee-pdf",
                "set-cookie": "must-not-be-retained=fixture-secret; HttpOnly",
            },
        )

    return handler


def _reviewed_fee_pdf_client_kwargs(seen_requests: list[httpx.Request]) -> dict[str, Any]:
    return {
        "kalshi_fee_schedule_transport": httpx.MockTransport(_fee_pdf_handler(seen_requests)),
        "kalshi_fee_schedule_bindings": {
            _REVIEWED_KALSHI_FEE_PDF_SHA256: _REVIEWED_KALSHI_FEE_BINDING,
        },
    }


def _assert_acquisition_envelope(
    evidence: Mapping[str, Any],
    *,
    path: str,
    payload: Any,
    params: Mapping[str, Any] | None = None,
) -> None:
    assert evidence["method"] == "GET"
    assert evidence["path"] == path
    assert evidence["params"] == dict(params or {})
    assert evidence["request_wall_utc"].endswith("Z")
    assert evidence["response_wall_utc"].endswith("Z")
    assert isinstance(evidence["request_monotonic_ns"], int)
    assert isinstance(evidence["response_monotonic_ns"], int)
    assert evidence["response_monotonic_ns"] >= evidence["request_monotonic_ns"]
    assert evidence["rtt_ms"] == pytest.approx(
        (evidence["response_monotonic_ns"] - evidence["request_monotonic_ns"]) / 1_000_000
    )
    assert evidence["http_date"] == "Sat, 29 Aug 2026 21:00:00 GMT"
    assert evidence["age_header"] == "0"
    assert evidence["cache_control"] == "no-store"
    assert evidence["request_id"]
    assert evidence["request_id_header"] == "x-request-id"
    assert evidence["raw_body_hash"] == _raw_body_hash(payload)
    expected_body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert base64.b64decode(evidence["raw_body_base64"], validate=True) == expected_body
    assert evidence["response_headers"]["content-type"] == "application/json"
    assert evidence["response_headers"]["date"] == "Sat, 29 Aug 2026 21:00:00 GMT"
    assert evidence["response_headers"]["age"] == "0"
    assert evidence["response_headers"]["cache-control"] == "no-store"
    assert evidence["response_headers"]["x-request-id"] == evidence["request_id"]
    assert set(evidence["response_headers"]) == {
        "age",
        "cache-control",
        "content-length",
        "content-type",
        "date",
        "x-request-id",
    }
    assert "set-cookie" not in evidence["response_headers"]
    assert {"set-cookie", "authorization", "x-api-key"}.isdisjoint(evidence["response_headers"])
    assert all(
        item[0] not in {"set-cookie", "authorization", "x-api-key"} for item in evidence["response_header_items"]
    )
    assert ["x-request-id", evidence["request_id"]] in evidence["response_header_items"]
    assert evidence["response_header_retention"] == "EXPLICIT_ALLOWLIST"
    assert evidence["request_accept_encoding"] == "identity"
    assert evidence["raw_body_representation"] == "HTTP_ENTITY_BYTES_ACCEPT_ENCODING_IDENTITY"
    assert evidence["body_complete"] is True
    assert evidence["freshness_basis"] == "LOCAL_RECEIPT_BOUNDED"
    assert evidence["requested_at"] == evidence["request_wall_utc"]
    assert evidence["received_at"] == evidence["response_wall_utc"]
    assert evidence["status_code"] == 200


def _assert_fee_pdf_acquisition_envelope(evidence: Mapping[str, Any], body: bytes) -> None:
    assert evidence["method"] == "GET"
    assert evidence["path"] == "/docs/kalshi-fee-schedule.pdf"
    assert evidence["final_url"] == "https://kalshi.com/docs/kalshi-fee-schedule.pdf"
    assert evidence["params"] == {}
    assert evidence["request_wall_utc"].endswith("Z")
    assert evidence["response_wall_utc"].endswith("Z")
    assert evidence["response_monotonic_ns"] >= evidence["request_monotonic_ns"]
    assert evidence["rtt_ms"] == pytest.approx(
        (evidence["response_monotonic_ns"] - evidence["request_monotonic_ns"]) / 1_000_000
    )
    assert evidence["raw_body_hash"] == hashlib.sha256(body).hexdigest()
    assert base64.b64decode(evidence["raw_body_base64"], validate=True) == body
    assert evidence["response_headers"]["content-type"] == "application/pdf"
    assert set(evidence["response_headers"]) == {
        "age",
        "cache-control",
        "content-length",
        "content-type",
        "date",
        "x-request-id",
    }
    assert "set-cookie" not in evidence["response_headers"]
    assert all(item[0] != "set-cookie" for item in evidence["response_header_items"])
    assert evidence["response_header_retention"] == "EXPLICIT_ALLOWLIST"
    assert evidence["request_accept_encoding"] == "identity"
    assert evidence["raw_body_representation"] == "HTTP_ENTITY_BYTES_ACCEPT_ENCODING_IDENTITY"
    assert evidence["request_id"] == "kalshi-fee-pdf"
    assert evidence["body_complete"] is True
    assert evidence["freshness_basis"] == "LOCAL_RECEIPT_BOUNDED"
    assert evidence["status_code"] == 200


def _candidate() -> dict:
    return {
        "candidate_id": "pmxt_candidate_fixture",
        "venue_a": "kalshi",
        "pmxt_market_id_a": "pmxt_catalog_uuid_kalshi",
        "title_a": "Will Candidate X win?",
        "source_metadata_a": {"ticker": "KXTEST-26-X"},
        "outcomes_a": [],
        "venue_b": "polymarket",
        "pmxt_market_id_b": "pmxt_catalog_uuid_polymarket",
        "title_b": "Will Candidate X win?",
        "slug_b": "candidate-x-win",
        "source_metadata_b": {},
        "outcomes_b": [],
    }


def test_native_clients_ignore_ambient_httpx_environment() -> None:
    with NativeEvidenceClient() as client:
        for native_client in (client._kalshi, client._gamma, client._clob, client._fee_schedule):
            assert native_client._trust_env is False


def test_native_response_cookie_is_never_replayed_between_requests() -> None:
    requests: list[httpx.Request] = []

    def cookie_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert "cookie" not in request.headers
        body = b"{}"
        return httpx.Response(
            200,
            stream=httpx.ByteStream(body),
            headers={
                "content-length": str(len(body)),
                "content-type": "application/json",
                "set-cookie": "must-not-replay=fixture-secret; Path=/; HttpOnly",
            },
        )

    with NativeEvidenceClient(kalshi_transport=httpx.MockTransport(cookie_handler)) as client:
        first_body, first_evidence = client._get_response_bytes(client._kalshi, "/first")
        assert len(client._kalshi.cookies) == 0
        second_body, second_evidence = client._get_response_bytes(client._kalshi, "/second")
        assert len(client._kalshi.cookies) == 0

    assert first_body == second_body == b"{}"
    assert len(requests) == 2
    for evidence in (first_evidence, second_evidence):
        assert "set-cookie" not in evidence["response_headers"]
        assert all(item[0] != "set-cookie" for item in evidence["response_header_items"])


def test_kalshi_fee_schedule_redirect_is_one_request_and_fails_closed() -> None:
    requests: list[httpx.Request] = []

    def redirect_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert "cookie" not in request.headers
        return httpx.Response(
            302,
            stream=httpx.ByteStream(b""),
            headers={
                "content-length": "0",
                "content-type": "text/html",
                "location": "/docs/redirected-fee-schedule.pdf",
                "set-cookie": "must-not-follow=fixture-secret; Path=/; HttpOnly",
            },
        )

    with NativeEvidenceClient(
        kalshi_fee_schedule_transport=httpx.MockTransport(redirect_handler),
    ) as client:
        capture = client._capture_kalshi_official_fee_schedule()
        assert client._fee_schedule.follow_redirects is False
        assert len(client._fee_schedule.cookies) == 0

    assert len(requests) == 1
    assert requests[0].url == httpx.URL("https://kalshi.com/docs/kalshi-fee-schedule.pdf")
    assert capture["status"] == "FAIL_CLOSED"
    assert "KALSHI_OFFICIAL_FEE_SCHEDULE_HTTP_STATUS_INVALID" in capture["reason_codes"]
    assert capture["request"]["status_code"] == 302
    assert "set-cookie" not in capture["request"]["response_headers"]


def test_compressed_response_is_retained_as_raw_bytes_and_fails_closed() -> None:
    decoded = b'{"unexpected":"decoded body"}'
    encoded = gzip.compress(decoded, mtime=0)

    def compressed_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200,
            stream=httpx.ByteStream(encoded),
            headers={
                "content-encoding": "gzip",
                "content-length": str(len(encoded)),
                "content-type": "application/json",
                "x-request-id": "compressed-response",
            },
        )

    with NativeEvidenceClient(gamma_transport=httpx.MockTransport(compressed_handler)) as client:
        with pytest.raises(NativeEvidenceError) as exc_info:
            client.fetch_metadata(_candidate(), "b")

    error = exc_info.value
    assert error.reason_code == "NATIVE_RESPONSE_CONTENT_ENCODING_UNEXPECTED"
    assert error.evidence["raw_body_hash"] == hashlib.sha256(encoded).hexdigest()
    assert base64.b64decode(error.evidence["raw_body_base64"], validate=True) == encoded
    assert error.evidence["response_headers"] == {
        "content-encoding": "gzip",
        "content-length": str(len(encoded)),
        "content-type": "application/json",
        "x-request-id": "compressed-response",
    }
    assert error.evidence["body_complete"] is True


def _kalshi_handler(seen_paths: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        seen_paths.append(request.url.path)
        if request.url.path.endswith("/markets/KXTEST-26-X"):
            payload = {
                "market": {
                    "ticker": "KXTEST-26-X",
                    "event_ticker": "KXTEST-26",
                    "series_ticker": "KXTEST",
                    "title": "Will Candidate X win?",
                    "yes_sub_title": "Candidate X wins",
                    "no_sub_title": "Candidate X does not win",
                    "open_time": "2026-01-01T14:00:00Z",
                    "close_time": "2026-11-04T02:00:00Z",
                    "expiration_time": "2026-11-04T04:00:00Z",
                    "expected_expiration_time": "2026-11-04T03:00:00Z",
                    "latest_expiration_time": "2026-11-05T03:00:00Z",
                    "settlement_timer_seconds": 3600,
                    "status": "active",
                    "fractional_trading_enabled": False,
                    "rules_primary": "Resolves Yes if Candidate X is certified as winner.",
                    "rules_secondary": "Certification controls.",
                    "can_close_early": False,
                    "market_type": "binary",
                    "notional_value_dollars": "1.0000",
                    "price_level_structure": "linear_cent",
                    "price_ranges": [{"start": "0.01", "end": "0.99", "step": "0.01"}],
                    "strike_type": "custom",
                    "custom_strike": {"candidate": "Candidate X"},
                    "updated_time": "2026-08-28T12:00:00Z",
                    "fee_waiver_expiration_time": "2026-02-01T00:00:00Z",
                    "mve_collection_ticker": None,
                    "mve_selected_legs": [],
                }
            }
            return _json_response(payload, request_id="kalshi-market")
        if request.url.path.endswith("/events/KXTEST-26/metadata"):
            payload = {"settlement_sources": [{"name": "Official authority", "url": "https://example.test/official"}]}
            return _json_response(payload, request_id="kalshi-event-metadata")
        if request.url.path.endswith("/events/KXTEST-26"):
            payload = {
                "event": {
                    "event_ticker": "KXTEST-26",
                    "series_ticker": "KXTEST",
                    "fee_type_override": None,
                    "fee_multiplier_override": None,
                    "mutually_exclusive": True,
                    "collateral_return_type": "binary",
                    "settlement_sources": [{"name": "Official authority", "url": "https://example.test/official"}],
                    "last_updated_ts": "2026-01-01T00:00:00Z",
                }
            }
            return _json_response(payload, request_id="kalshi-event")
        if request.url.path.endswith("/series/KXTEST"):
            payload = {
                "series": {
                    "ticker": "KXTEST",
                    "fee_type": "quadratic",
                    "fee_multiplier": 1.0,
                    "contract_url": "https://example.test/contract",
                    "contract_terms_url": "https://example.test/contract-terms",
                    "settlement_sources": [{"name": "Official authority", "url": "https://example.test/official"}],
                    "last_updated_ts": "2026-01-01T00:00:00Z",
                }
            }
            return _json_response(payload, request_id="kalshi-series")
        if request.url.path.endswith("/series/fee_changes"):
            assert dict(request.url.params) == {"series_ticker": "KXTEST", "show_historical": "true"}
            payload = {
                "series_fee_change_arr": [
                    {
                        "id": "series-change-1",
                        "series_ticker": "KXTEST",
                        "fee_type": "quadratic",
                        "fee_multiplier": 1.0,
                        "scheduled_ts": "2026-01-01T00:00:00Z",
                    }
                ]
            }
            return _json_response(payload, request_id="kalshi-series-fees")
        if request.url.path.endswith("/events/fee_changes"):
            assert dict(request.url.params) == {"event_ticker": "KXTEST-26", "limit": "1000"}
            payload = {"event_fee_changes": [], "cursor": ""}
            return _json_response(payload, request_id="kalshi-event-fees")
        if request.url.path.endswith("/markets/KXTEST-26-X/orderbook"):
            assert request.url.params["depth"] in {"1", "25"}
            payload = {
                "orderbook_fp": {
                    "yes_dollars": [["0.40", "12.0"], ["0.39", "4.0"]],
                    "no_dollars": [["0.55", "8.0"], ["0.54", "5.0"]],
                }
            }
            return _json_response(payload, request_id="kalshi-orderbook")
        raise AssertionError(f"unexpected Kalshi URL: {request.url}")

    return handler


def _gamma_handler(request: httpx.Request) -> httpx.Response:
    assert request.headers["accept-encoding"] == "identity"
    assert request.url.path == "/markets"
    assert request.url.params["slug"] == "candidate-x-win"
    return _json_response(
        [
            {
                "id": "12345",
                "conditionId": "0xcondition",
                "slug": "candidate-x-win",
                "question": "Will Candidate X win?",
                "description": "Resolves Yes if Candidate X is certified as winner.",
                "resolutionSource": "https://example.test/official",
                "resolvedBy": "Official authority",
                "startDate": "2026-01-02T15:00:00Z",
                "endDate": "2026-11-04T02:00:00Z",
                "eventStartTime": "2026-11-03T14:00:00Z",
                "umaEndDate": "2026-11-04T04:00:00Z",
                "outcomes": json.dumps(["Yes", "No"]),
                "clobTokenIds": json.dumps(["token_yes", "token_no"]),
                "negRisk": False,
                "active": True,
                "closed": False,
                "archived": False,
                "acceptingOrders": True,
                "settlementDelaySeconds": 3600,
                "feesEnabled": True,
                "createdAt": "2026-01-01T12:00:00Z",
                "updatedAt": "2026-08-28T12:00:00Z",
                "events": [
                    {
                        "id": "event-123",
                        "slug": "candidate-x-event",
                        "title": "Candidate X election",
                        "startDate": "2026-01-02T15:00:00Z",
                        "endDate": "2026-11-04T02:00:00Z",
                    }
                ],
            }
        ],
        request_id="polymarket-gamma",
    )


def _clob_handler(request: httpx.Request) -> httpx.Response:
    assert request.headers["accept-encoding"] == "identity"
    token_id = request.url.params["token_id"]
    assert token_id in {"token_yes", "token_no"}
    if request.url.path == "/fee-rate":
        return _json_response({"base_fee": 100}, request_id=f"polymarket-fee-{token_id}")
    assert request.url.path == "/book"
    return _json_response(
        {
            "market": "0xcondition",
            "asset_id": token_id,
            "timestamp": "1780000000000",
            "bids": [{"price": "0.40", "size": "20"}],
            "asks": [{"price": "0.42", "size": "15"}],
            "min_order_size": "5",
            "tick_size": "0.01",
            "hash": f"hash_{token_id}",
        },
        request_id=f"polymarket-book-{token_id}",
    )


def test_native_metadata_reverse_resolves_ids_and_retains_raw_rules() -> None:
    seen_paths: list[str] = []
    with NativeEvidenceClient(
        kalshi_transport=httpx.MockTransport(_kalshi_handler(seen_paths)),
        gamma_transport=httpx.MockTransport(_gamma_handler),
        clob_transport=httpx.MockTransport(_clob_handler),
    ) as client:
        kalshi = client.fetch_metadata(_candidate(), "a")
        polymarket = client.fetch_metadata(_candidate(), "b")

    assert kalshi["native_market_id"] == "KXTEST-26-X"
    assert kalshi["native_event_id"] == "KXTEST-26"
    assert kalshi["native_series_id"] == "KXTEST"
    assert kalshi["native_outcome_labels"] == {
        "YES": "Candidate X wins",
        "NO": "Candidate X does not win",
    }
    assert kalshi["pmxt_market_id"] == "pmxt_catalog_uuid_kalshi"
    assert kalshi["raw_response"]["market"]["market"]["rules_primary"].startswith("Resolves Yes")
    assert kalshi["open_time"] == "2026-01-01T14:00:00Z"
    assert kalshi["close_time"] == "2026-11-04T02:00:00Z"
    assert kalshi["expiration_time"] == "2026-11-04T04:00:00Z"
    assert kalshi["expected_expiration_time"] == "2026-11-04T03:00:00Z"
    assert kalshi["latest_expiration_time"] == "2026-11-05T03:00:00Z"
    assert kalshi["market_type"] == "binary"
    assert kalshi["notional_value_dollars"] == "1.0000"
    assert kalshi["event_mutually_exclusive"] is True
    assert kalshi["event_collateral_return_type"] == "binary"
    assert kalshi["outcome_polarity"] == {"YES": "YES", "NO": "NO"}
    assert kalshi["material_edge_cases"] == {
        "can_close_early": False,
        "strike_type": "custom",
        "custom_strike": {"candidate": "Candidate X"},
        "event_collateral_return_type": "binary",
    }
    assert kalshi["clarification_or_revision_timestamps"] == {
        "market_updated_time": "2026-08-28T12:00:00Z",
        "event_last_updated_ts": "2026-01-01T00:00:00Z",
        "series_last_updated_ts": "2026-01-01T00:00:00Z",
    }
    assert polymarket["native_market_id"] == "12345"
    assert polymarket["native_event_id"] == "event-123"
    assert polymarket["native_condition_id"] == "0xcondition"
    assert polymarket["native_outcome_ids"] == {"YES": "token_yes", "NO": "token_no"}
    assert polymarket["native_outcome_labels"] == {"YES": "Yes", "NO": "No"}
    assert polymarket["open_time"] == "2026-01-02T15:00:00Z"
    assert polymarket["close_time"] == "2026-11-04T02:00:00Z"
    assert polymarket["expiration_time"] == "2026-11-04T04:00:00Z"
    assert polymarket["market_type"] == "binary"
    assert polymarket["negative_risk"] is False
    assert polymarket["outcome_polarity"] == {"YES": "YES", "NO": "NO"}
    assert polymarket["material_edge_cases"] == {}
    assert polymarket["clarification_or_revision_timestamps"] == {
        "created_at": "2026-01-01T12:00:00Z",
        "updated_at": "2026-08-28T12:00:00Z",
        "uma_end_date": "2026-11-04T04:00:00Z",
    }
    assert polymarket["raw_response"]["selected_market"]["events"][0]["id"] == "event-123"
    assert len(kalshi["raw_sha256"]) == len(kalshi["rule_hash"]) == 64
    assert len(polymarket["raw_sha256"]) == len(polymarket["rule_hash"]) == 64
    assert kalshi["raw_sha256"] == canonical_json_sha256(kalshi["raw_response"])
    assert polymarket["raw_sha256"] == canonical_json_sha256(polymarket["raw_response"])
    assert kalshi["rule_hash"] != kalshi["raw_sha256"]
    assert polymarket["rule_hash"] != polymarket["raw_sha256"]

    kalshi_market_payload = kalshi["raw_response"]["market"]
    kalshi_event_metadata_payload = kalshi["raw_response"]["event_metadata"]
    kalshi_event_payload = kalshi["raw_response"]["event"]
    kalshi_series_payload = kalshi["raw_response"]["series"]
    _assert_acquisition_envelope(
        kalshi["requests"][0],
        path="/markets/KXTEST-26-X",
        payload=kalshi_market_payload,
    )
    _assert_acquisition_envelope(
        kalshi["requests"][1],
        path="/events/KXTEST-26/metadata",
        payload=kalshi_event_metadata_payload,
    )
    _assert_acquisition_envelope(
        kalshi["requests"][2],
        path="/events/KXTEST-26",
        payload=kalshi_event_payload,
    )
    _assert_acquisition_envelope(
        kalshi["requests"][3],
        path="/series/KXTEST",
        payload=kalshi_series_payload,
    )
    _assert_acquisition_envelope(
        polymarket["requests"][0],
        path="/markets",
        params={"slug": "candidate-x-win"},
        payload=polymarket["raw_response"]["markets"],
    )
    assert seen_paths == [
        "/trade-api/v2/markets/KXTEST-26-X",
        "/trade-api/v2/events/KXTEST-26/metadata",
        "/trade-api/v2/events/KXTEST-26",
        "/trade-api/v2/series/KXTEST",
    ]
    assert all("pmxt_catalog_uuid" not in path for path in seen_paths)
    assert kalshi["live_eligible"] is polymarket["live_eligible"] is False


def test_pmxt_catalog_id_is_never_used_as_kalshi_native_fallback() -> None:
    candidate = _candidate()
    candidate["source_metadata_a"] = {}
    calls = 0

    def forbidden(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError(f"unexpected request: {request.url}")

    with NativeEvidenceClient(kalshi_transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(candidate, "a")

    assert error.value.reason_code == "NATIVE_ID_UNRESOLVED"
    assert calls == 0


def test_plain_uuid_pmxt_catalog_id_is_never_sent_as_a_kalshi_ticker() -> None:
    candidate = _candidate()
    catalog_id = "95c75c42-7b64-4bd5-96a2-37dd73ab984d"
    candidate["pmxt_market_id_a"] = catalog_id
    candidate["source_metadata_a"] = {"original_id": catalog_id}
    calls = 0

    def forbidden(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError(f"unexpected request: {request.url}")

    with NativeEvidenceClient(kalshi_transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(candidate, "a")

    assert error.value.reason_code == "NATIVE_ID_UNRESOLVED"
    assert calls == 0


def test_kalshi_market_slug_alone_resolves_and_is_checked_against_native_ticker() -> None:
    candidate = _candidate()
    catalog_id = "95c75c42-7b64-4bd5-96a2-37dd73ab984d"
    candidate.update(
        {
            "pmxt_market_id_a": catalog_id,
            "slug_a": "KXPRESPERSON-28-NHAL",
            "url_a": None,
            "source_metadata_a": {},
        }
    )
    seen_paths: list[str] = []

    def observed_shape(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        seen_paths.append(request.url.path)
        if request.url.path.endswith("/markets/KXPRESPERSON-28-NHAL"):
            return _json_response(
                {
                    "market": {
                        "ticker": "KXPRESPERSON-28-NHAL",
                        "event_ticker": "KXPRESPERSON-28",
                        "series_ticker": "KXPRESPERSON",
                    }
                },
                request_id="kalshi-slug-market",
            )
        if request.url.path.endswith("/events/KXPRESPERSON-28/metadata"):
            return _json_response({"settlement_sources": []}, request_id="kalshi-slug-event-metadata")
        if request.url.path.endswith("/events/KXPRESPERSON-28"):
            return _json_response(
                {"event": {"event_ticker": "KXPRESPERSON-28", "series_ticker": "KXPRESPERSON"}},
                request_id="kalshi-slug-event",
            )
        if request.url.path.endswith("/series/KXPRESPERSON"):
            return _json_response(
                {"series": {"ticker": "KXPRESPERSON"}},
                request_id="kalshi-slug-series",
            )
        raise AssertionError(f"unexpected Kalshi URL: {request.url}")

    with NativeEvidenceClient(kalshi_transport=httpx.MockTransport(observed_shape)) as client:
        metadata = client.fetch_metadata(candidate, "a")

    assert metadata["native_market_id"] == "KXPRESPERSON-28-NHAL"
    assert seen_paths == [
        "/trade-api/v2/markets/KXPRESPERSON-28-NHAL",
        "/trade-api/v2/events/KXPRESPERSON-28/metadata",
        "/trade-api/v2/events/KXPRESPERSON-28",
        "/trade-api/v2/series/KXPRESPERSON",
    ]


@pytest.mark.parametrize("market_ticker", ["CONTROLH-2026-R", "CONTROLH-2026-D"])
def test_replay_non_kx_kalshi_market_slugs_resolve_from_observed_shape(market_ticker: str) -> None:
    candidate = _candidate()
    catalog_id = "95c75c42-7b64-4bd5-96a2-37dd73ab984d"
    candidate.update(
        {
            "pmxt_market_id_a": catalog_id,
            "slug_a": market_ticker,
            "url_a": "https://pmxt.test/events/CONTROLH-2026",
            "source_metadata_a": {
                "sourceExchange": "kalshi",
                "id": catalog_id,
                "marketId": catalog_id,
            },
        }
    )
    seen_paths: list[str] = []

    def replay_shape(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        seen_paths.append(request.url.path)
        if request.url.path.endswith(f"/markets/{market_ticker}"):
            return _json_response(
                {
                    "market": {
                        "ticker": market_ticker,
                        "event_ticker": "CONTROLH-2026",
                        "series_ticker": "CONTROLH",
                    }
                },
                request_id="kalshi-replay-market",
            )
        if request.url.path.endswith("/events/CONTROLH-2026/metadata"):
            return _json_response({"settlement_sources": []}, request_id="kalshi-replay-event-metadata")
        if request.url.path.endswith("/events/CONTROLH-2026"):
            return _json_response(
                {"event": {"event_ticker": "CONTROLH-2026", "series_ticker": "CONTROLH"}},
                request_id="kalshi-replay-event",
            )
        if request.url.path.endswith("/series/CONTROLH"):
            return _json_response(
                {"series": {"ticker": "CONTROLH"}},
                request_id="kalshi-replay-series",
            )
        raise AssertionError(f"unexpected Kalshi URL: {request.url}")

    with NativeEvidenceClient(kalshi_transport=httpx.MockTransport(replay_shape)) as client:
        metadata = client.fetch_metadata(candidate, "a")

    assert metadata["native_market_id"] == market_ticker
    assert seen_paths == [
        f"/trade-api/v2/markets/{market_ticker}",
        "/trade-api/v2/events/CONTROLH-2026/metadata",
        "/trade-api/v2/events/CONTROLH-2026",
        "/trade-api/v2/series/CONTROLH",
    ]


@pytest.mark.parametrize(
    ("slug", "url"),
    [
        ("KXPRESPERSON-28", None),
        (None, "https://pmxt.test/events/KXPRESPERSON-28"),
    ],
)
def test_event_only_kalshi_slug_or_url_never_resolves_as_market_ticker(
    slug: str | None,
    url: str | None,
) -> None:
    candidate = _candidate()
    candidate["source_metadata_a"] = {}
    candidate["slug_a"] = slug
    candidate["url_a"] = url
    calls = 0

    def forbidden(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError(f"unexpected request: {request.url}")

    with NativeEvidenceClient(kalshi_transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(candidate, "a")

    assert error.value.reason_code == "NATIVE_ID_UNRESOLVED"
    assert calls == 0


def test_conflicting_kalshi_market_ticker_hints_fail_before_request() -> None:
    candidate = _candidate()
    candidate["slug_a"] = "KXPRESPERSON-28-NHAL"
    candidate["source_metadata_a"] = {"ticker": "KXPRESPERSON-28-DTRU"}
    calls = 0

    def forbidden(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError(f"unexpected request: {request.url}")

    with NativeEvidenceClient(kalshi_transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(candidate, "a")

    assert error.value.reason_code == "NATIVE_ID_AMBIGUOUS"
    assert error.value.evidence == {"hint_count": 2}
    assert calls == 0


@pytest.mark.parametrize(
    ("pmxt_market_id", "slug"),
    [
        (
            "pmxt_catalog_identifier",
            "95c75c42-7b64-4bd5-96a2-37dd73ab984d",
        ),
        ("KXPRESPERSON-28-NHAL", "KXPRESPERSON-28-NHAL"),
        ("pmxt_catalog_identifier", "pmxt_catalog_identifier"),
    ],
)
def test_uuid_and_pmxt_alias_slugs_fail_before_request(
    pmxt_market_id: str,
    slug: str,
) -> None:
    candidate = _candidate()
    candidate["pmxt_market_id_a"] = pmxt_market_id
    candidate["slug_a"] = slug
    candidate["source_metadata_a"] = {}
    calls = 0

    def forbidden(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError(f"unexpected request: {request.url}")

    with NativeEvidenceClient(kalshi_transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(candidate, "a")

    assert error.value.reason_code == "NATIVE_ID_UNRESOLVED"
    assert calls == 0


def test_pmxt_catalog_id_is_never_sent_as_polymarket_condition_fallback() -> None:
    candidate = _candidate()
    candidate["slug_b"] = None
    candidate["contract_address_b"] = candidate["pmxt_market_id_b"]
    calls = 0

    def forbidden(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError(f"unexpected request: {request.url}")

    with NativeEvidenceClient(gamma_transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(candidate, "b")

    assert error.value.reason_code == "NATIVE_ID_UNRESOLVED"
    assert calls == 0


def test_native_transport_failure_is_returned_as_evidence_bearing_fail_closed_error() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("fixture unavailable", request=request)

    with NativeEvidenceClient(kalshi_transport=httpx.MockTransport(unavailable)) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(_candidate(), "a")

    assert error.value.reason_code == "NATIVE_REQUEST_ERROR"
    assert error.value.evidence["method"] == "GET"
    assert error.value.evidence["path"] == "/markets/KXTEST-26-X"
    assert "requested_at" in error.value.evidence
    assert "received_at" in error.value.evidence


def test_native_response_body_is_hard_bounded() -> None:
    calls = 0

    def oversized(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _json_response({"padding": "x" * 200}, request_id="oversized")

    with NativeEvidenceClient(
        kalshi_transport=httpx.MockTransport(oversized),
        max_response_bytes=100,
    ) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(_candidate(), "a")

    assert error.value.reason_code == "NATIVE_RESPONSE_TOO_LARGE"
    assert error.value.evidence["max_response_bytes"] == 100
    assert calls == 1


def test_native_redirect_response_is_not_accepted_as_evidence() -> None:
    def redirect(request: httpx.Request) -> httpx.Response:
        return _json_response({"market": {}}, request_id="redirect", status_code=302)

    with NativeEvidenceClient(kalshi_transport=httpx.MockTransport(redirect)) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(_candidate(), "a")

    assert error.value.reason_code == "NATIVE_HTTP_ERROR"
    assert error.value.evidence["status_code"] == 302


def test_later_kalshi_metadata_failure_preserves_successful_native_response() -> None:
    seen_paths: list[str] = []

    def fail_event_after_market(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events/KXTEST-26/metadata"):
            return _json_response({"error": "fixture unavailable"}, request_id="unavailable", status_code=503)
        return _kalshi_handler(seen_paths)(request)

    with NativeEvidenceClient(kalshi_transport=httpx.MockTransport(fail_event_after_market)) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(_candidate(), "a")

    assert error.value.reason_code == "NATIVE_HTTP_ERROR"
    assert len(error.value.evidence["completed_requests"]) == 1
    assert [request["path"] for request in error.value.evidence["completed_requests"]] == ["/markets/KXTEST-26-X"]
    assert error.value.evidence["metadata_capture_stage"] == "event_metadata"
    assert error.value.evidence["partial_raw_response"]["market"]["market"]["ticker"] == "KXTEST-26-X"


def test_polymarket_slug_lookup_must_match_explicit_condition_hint() -> None:
    candidate = _candidate()
    candidate["contract_address_b"] = "0xdifferent-condition"

    with NativeEvidenceClient(gamma_transport=httpx.MockTransport(_gamma_handler)) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(candidate, "b")

    assert error.value.reason_code == "NATIVE_ID_MISMATCH"
    assert error.value.evidence["returned_condition_id"] == "0xcondition"


def test_polymarket_duplicate_yes_no_token_ids_fail_closed() -> None:
    def duplicate_token_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(_gamma_handler(request).read())
        payload[0]["clobTokenIds"] = json.dumps(["same_token", "same_token"])
        return _json_response(payload, request_id="polymarket-gamma-duplicate-tokens")

    with NativeEvidenceClient(gamma_transport=httpx.MockTransport(duplicate_token_handler)) as client:
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_metadata(_candidate(), "b")

    assert error.value.reason_code == "NATIVE_OUTCOME_ID_AMBIGUOUS"


def test_polymarket_rule_hash_changes_with_settlement_delay() -> None:
    def with_delay(delay_seconds: int):
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(_gamma_handler(request).read())
            payload[0]["settlementDelaySeconds"] = delay_seconds
            return _json_response(payload, request_id=f"polymarket-gamma-delay-{delay_seconds}")

        return handler

    with NativeEvidenceClient(gamma_transport=httpx.MockTransport(with_delay(3600))) as client:
        one_hour = client.fetch_metadata(_candidate(), "b")
    with NativeEvidenceClient(gamma_transport=httpx.MockTransport(with_delay(7200))) as client:
        two_hours = client.fetch_metadata(_candidate(), "b")

    assert one_hour["settlement_delay_seconds"] == 3600
    assert two_hours["settlement_delay_seconds"] == 7200
    assert one_hour["rule_hash"] != two_hours["rule_hash"]


def test_native_books_preserve_raw_depth_and_use_only_aggressive_sides() -> None:
    seen_paths: list[str] = []
    fee_pdf_requests: list[httpx.Request] = []
    with NativeEvidenceClient(
        kalshi_transport=httpx.MockTransport(_kalshi_handler(seen_paths)),
        gamma_transport=httpx.MockTransport(_gamma_handler),
        clob_transport=httpx.MockTransport(_clob_handler),
        **_reviewed_fee_pdf_client_kwargs(fee_pdf_requests),
    ) as client:
        kalshi_metadata = client.fetch_metadata(_candidate(), "a")
        polymarket_metadata = client.fetch_metadata(_candidate(), "b")
        assert client.supporting_evidence() is None
        assert fee_pdf_requests == []
        kalshi_book = client.fetch_book(kalshi_metadata, depth=25)
        supporting_evidence = client.supporting_evidence()
        polymarket_book = client.fetch_book(polymarket_metadata, depth=25)

    assert kalshi_book["sides"]["YES"]["asks"][0] == {"price": 0.45, "size": 8.0}
    assert kalshi_book["sides"]["NO"]["asks"][0] == {"price": 0.6, "size": 12.0}
    assert kalshi_book["timestamp_basis"] == "no_venue_source_timestamp"
    assert kalshi_book["freshness_basis"] == "LOCAL_RECEIPT_BOUNDED"
    assert kalshi_book["source_timestamp"] is kalshi_book["as_of"] is None
    assert kalshi_book["request_started_at"] == kalshi_book["requests"][0]["request_wall_utc"]
    assert kalshi_book["received_at"] == kalshi_book["requests"][0]["response_wall_utc"]
    assert kalshi_book["raw_sha256"] == canonical_json_sha256(kalshi_book["raw_response"])
    _assert_acquisition_envelope(
        kalshi_book["requests"][0],
        path="/markets/KXTEST-26-X/orderbook",
        params={"depth": 25},
        payload=kalshi_book["raw_response"]["orderbook"],
    )

    kalshi_fees = kalshi_book["fee_evidence"]
    assert kalshi_fees["status"] == "VALID"
    assert kalshi_fees["venue"] == "kalshi"
    assert kalshi_fees["model"] == "KALSHI_QUADRATIC_TAKER"
    assert kalshi_fees["liquidity_role"] == "TAKER"
    assert kalshi_fees["passive_fills_assumed"] is False
    assert kalshi_fees["effective_from"] == "2026-01-01T00:00:00Z"
    assert kalshi_fees["effective_basis"] == "VENUE_SCHEDULED_CHANGE"
    assert kalshi_fees["reason_codes"] == []
    assert kalshi_fees["official_fee_schedule_sha256"] == _REVIEWED_KALSHI_FEE_PDF_SHA256
    assert kalshi_fees["formula_binding_id"] == "fixture-reviewed-kalshi-fee-pdf-v1"
    assert kalshi_fees["coefficient"] == "0.07"
    assert kalshi_fees["trade_fee_rounding_quantum"] == "0.0001"
    assert kalshi_fees["balance_precision_upper_bound"] == "0.0099"
    assert kalshi_fees["schedule"]["series_current"] == {
        "fee_type": "quadratic",
        "fee_multiplier": 1.0,
        "last_updated_ts": "2026-01-01T00:00:00Z",
    }
    assert kalshi_fees["schedule"]["effective"] == {
        "fee_type": "quadratic",
        "fee_multiplier": 1.0,
        "taker_base_coefficient": "0.07",
        "trade_fee_rounding_quantum": "0.0001",
        "balance_precision_upper_bound": "0.0099",
    }
    assert kalshi_fees["schedule"]["official_fee_schedule"]["status"] == "REVIEWED"
    assert kalshi_fees["schedule"]["official_fee_schedule"]["formula_binding"] == _REVIEWED_KALSHI_FEE_BINDING
    assert kalshi_fees["schedule_sha256"] == canonical_json_sha256(kalshi_fees["schedule"])
    assert kalshi_fees["raw_sha256"] == canonical_json_sha256(kalshi_fees["raw_response"])
    assert (
        kalshi_fees["capture_window"]["request_monotonic_ns"] <= kalshi_fees["capture_window"]["response_monotonic_ns"]
    )
    assert kalshi_fees["capture_window"]["rtt_ms"] >= 0
    _assert_fee_pdf_acquisition_envelope(kalshi_fees["requests"][0], _REVIEWED_KALSHI_FEE_PDF)
    for request, (name, path, params) in zip(
        kalshi_fees["requests"][1:],
        [
            ("event", "/events/KXTEST-26", {}),
            ("series", "/series/KXTEST", {}),
            ("series_fee_changes", "/series/fee_changes", {"series_ticker": "KXTEST", "show_historical": "true"}),
            ("event_fee_changes", "/events/fee_changes", {"event_ticker": "KXTEST-26", "limit": 1000}),
        ],
    ):
        _assert_acquisition_envelope(
            request,
            path=path,
            params=params,
            payload=kalshi_fees["raw_response"][name],
        )

    assert len(fee_pdf_requests) == 1
    assert supporting_evidence is not None
    official_support = supporting_evidence["kalshi_official_fee_schedule"]
    assert official_support["status"] == "REVIEWED"
    assert official_support["reason_codes"] == []
    assert official_support["source_url"] == "https://kalshi.com/docs/kalshi-fee-schedule.pdf"
    assert official_support["final_url"] == "https://kalshi.com/docs/kalshi-fee-schedule.pdf"
    assert official_support["raw_body_sha256"] == _REVIEWED_KALSHI_FEE_PDF_SHA256
    assert official_support["byte_size"] == len(_REVIEWED_KALSHI_FEE_PDF)
    assert official_support["formula_binding"] == _REVIEWED_KALSHI_FEE_BINDING
    _assert_fee_pdf_acquisition_envelope(official_support["request"], _REVIEWED_KALSHI_FEE_PDF)
    assert len(fee_pdf_requests) == 1

    assert polymarket_book["source_timestamps"] == {
        "YES": "2026-05-28T20:26:40Z",
        "NO": "2026-05-28T20:26:40Z",
    }
    assert polymarket_book["freshness_basis"] == "VENUE_SOURCE_TIMESTAMP"
    assert polymarket_book["raw_response"]["books"]["YES"]["asset_id"] == "token_yes"
    assert polymarket_book["raw_sha256"] == canonical_json_sha256(polymarket_book["raw_response"])

    polymarket_fees = polymarket_book["fee_evidence"]
    assert polymarket_fees["status"] == "FAIL_CLOSED"
    assert polymarket_fees["venue"] == "polymarket"
    assert polymarket_fees["model"] == "POLYMARKET_FEE_ESTIMATE_ONLY"
    assert polymarket_fees["liquidity_role"] == "TAKER"
    assert polymarket_fees["passive_fills_assumed"] is False
    assert polymarket_fees["estimate_formula"] == "shares * (base_fee_bps / 10000) * price * (1 - price)"
    assert polymarket_fees["estimate_only"] is True
    assert "formula" not in polymarket_fees
    assert polymarket_fees["effective_from"] is None
    assert polymarket_fees["effective_basis"] == "UNAVAILABLE"
    assert polymarket_fees["reason_codes"] == [
        "POLYMARKET_CONDITION_FEE_PARAMETERS_NOT_CAPTURED",
        "POLYMARKET_FEE_EFFECTIVE_TIMESTAMP_UNAVAILABLE",
        "POLYMARKET_TOKEN_BASE_FEE_IS_ESTIMATE_ONLY",
    ]
    assert polymarket_fees["schedule"]["fees_enabled"] is True
    assert polymarket_fees["schedule"]["base_fee_bps_estimate_by_outcome"] == {"YES": 100, "NO": 100}
    assert polymarket_fees["schedule"]["condition_fee_details"] is None
    assert "base_fee_bps_by_outcome" not in polymarket_fees["schedule"]
    assert polymarket_fees["schedule_sha256"] == canonical_json_sha256(polymarket_fees["schedule"])
    assert polymarket_fees["raw_sha256"] == canonical_json_sha256(polymarket_fees["raw_response"])
    assert [request["path"] for request in polymarket_book["requests"]] == [
        "/fee-rate",
        "/book",
        "/fee-rate",
        "/book",
    ]
    for polarity, token_id in (("YES", "token_yes"), ("NO", "token_no")):
        fee_request = next(
            request for request in polymarket_fees["requests"] if request["params"] == {"token_id": token_id}
        )
        _assert_acquisition_envelope(
            fee_request,
            path="/fee-rate",
            params={"token_id": token_id},
            payload=polymarket_fees["raw_response"][polarity],
        )
        book_request = next(
            request
            for request in polymarket_book["requests"]
            if request["path"] == "/book" and request["params"] == {"token_id": token_id}
        )
        _assert_acquisition_envelope(
            book_request,
            path="/book",
            params={"token_id": token_id},
            payload=polymarket_book["raw_response"]["books"][polarity],
        )

    assert seen_paths == [
        "/trade-api/v2/markets/KXTEST-26-X",
        "/trade-api/v2/events/KXTEST-26/metadata",
        "/trade-api/v2/events/KXTEST-26",
        "/trade-api/v2/series/KXTEST",
        "/trade-api/v2/events/KXTEST-26",
        "/trade-api/v2/series/KXTEST",
        "/trade-api/v2/series/fee_changes",
        "/trade-api/v2/events/fee_changes",
        "/trade-api/v2/markets/KXTEST-26-X/orderbook",
    ]
    assert len(kalshi_book["raw_sha256"]) == len(polymarket_book["raw_sha256"]) == 64
    assert kalshi_book["live_eligible"] is polymarket_book["live_eligible"] is False


def test_changed_unreviewed_fee_pdf_fails_fee_evidence_closed_but_preserves_l2_book() -> None:
    seen_paths: list[str] = []
    fee_pdf_requests: list[httpx.Request] = []
    changed_pdf = _REVIEWED_KALSHI_FEE_PDF.replace(b"reviewed", b"changed")

    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected non-Kalshi request: {request.url}")

    with NativeEvidenceClient(
        kalshi_transport=httpx.MockTransport(_kalshi_handler(seen_paths)),
        gamma_transport=httpx.MockTransport(forbidden),
        clob_transport=httpx.MockTransport(forbidden),
        kalshi_fee_schedule_transport=httpx.MockTransport(_fee_pdf_handler(fee_pdf_requests, body=changed_pdf)),
        kalshi_fee_schedule_bindings={
            _REVIEWED_KALSHI_FEE_PDF_SHA256: _REVIEWED_KALSHI_FEE_BINDING,
        },
    ) as client:
        metadata = client.fetch_metadata(_candidate(), "a")
        book = client.fetch_book(metadata, depth=25)
        supporting_evidence = client.supporting_evidence()

    assert book["sides"]["YES"]["asks"][0] == {"price": 0.45, "size": 8.0}
    assert book["raw_response"]["orderbook"]["orderbook_fp"]["yes_dollars"]
    assert book["raw_sha256"] == canonical_json_sha256(book["raw_response"])
    fee_evidence = book["fee_evidence"]
    assert fee_evidence["status"] == "FAIL_CLOSED"
    assert fee_evidence["model"] is None
    assert fee_evidence["formula_binding_id"] is None
    assert "KALSHI_OFFICIAL_FEE_SCHEDULE_UNRECOGNIZED" in fee_evidence["reason_codes"]
    assert "KALSHI_OFFICIAL_FEE_SCHEDULE_NOT_REVIEWED" in fee_evidence["reason_codes"]
    changed_digest = hashlib.sha256(changed_pdf).hexdigest()
    assert changed_digest != _REVIEWED_KALSHI_FEE_PDF_SHA256
    assert fee_evidence["official_fee_schedule_sha256"] == changed_digest
    assert len(fee_pdf_requests) == 1
    assert supporting_evidence is not None
    official_support = supporting_evidence["kalshi_official_fee_schedule"]
    assert official_support["status"] == "FAIL_CLOSED"
    assert official_support["reason_codes"] == ["KALSHI_OFFICIAL_FEE_SCHEDULE_UNRECOGNIZED"]
    assert official_support["raw_body_sha256"] == changed_digest
    assert official_support["formula_binding"] is None
    _assert_fee_pdf_acquisition_envelope(official_support["request"], changed_pdf)


def test_native_book_normalization_honors_depth_bound() -> None:
    seen_paths: list[str] = []
    fee_pdf_requests: list[httpx.Request] = []

    def multi_level_clob(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fee-rate":
            return _clob_handler(request)
        response = json.loads(_clob_handler(request).read())
        response["bids"].append({"price": "0.39", "size": "50"})
        response["asks"].append({"price": "0.43", "size": "50"})
        return _json_response(response, request_id=f"polymarket-book-{request.url.params['token_id']}")

    with NativeEvidenceClient(
        kalshi_transport=httpx.MockTransport(_kalshi_handler(seen_paths)),
        gamma_transport=httpx.MockTransport(_gamma_handler),
        clob_transport=httpx.MockTransport(multi_level_clob),
        **_reviewed_fee_pdf_client_kwargs(fee_pdf_requests),
    ) as client:
        kalshi_metadata = client.fetch_metadata(_candidate(), "a")
        polymarket_metadata = client.fetch_metadata(_candidate(), "b")
        kalshi_book = client.fetch_book(kalshi_metadata, depth=1)
        polymarket_book = client.fetch_book(polymarket_metadata, depth=1)

    for book in (kalshi_book, polymarket_book):
        assert book["normalized_depth_limit"] == 1
        assert book["normalized_depth_truncated"] is True
        assert all(len(side[level_type]) <= 1 for side in book["sides"].values() for level_type in ("bids", "asks"))


@pytest.mark.parametrize("missing_field", ["asset_id", "market"])
def test_polymarket_book_requires_response_native_identifiers(missing_field: str) -> None:
    def missing_identifier(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fee-rate":
            return _clob_handler(request)
        payload = json.loads(_clob_handler(request).read())
        del payload[missing_field]
        return _json_response(payload, request_id=f"polymarket-book-{request.url.params['token_id']}")

    with NativeEvidenceClient(
        gamma_transport=httpx.MockTransport(_gamma_handler),
        clob_transport=httpx.MockTransport(missing_identifier),
    ) as client:
        metadata = client.fetch_metadata(_candidate(), "b")
        with pytest.raises(NativeEvidenceError) as error:
            client.fetch_book(metadata, depth=25)

    assert error.value.reason_code == "NATIVE_ID_MISMATCH"
    assert error.value.evidence["polarity"] == "YES"


def test_native_kalshi_book_without_venue_timestamp_cannot_reach_shadow_alert() -> None:
    seen_paths: list[str] = []
    fee_pdf_requests: list[httpx.Request] = []
    with NativeEvidenceClient(
        kalshi_transport=httpx.MockTransport(_kalshi_handler(seen_paths)),
        gamma_transport=httpx.MockTransport(_gamma_handler),
        clob_transport=httpx.MockTransport(_clob_handler),
        **_reviewed_fee_pdf_client_kwargs(fee_pdf_requests),
    ) as client:
        kalshi_metadata = client.fetch_metadata(_candidate(), "a")
        polymarket_metadata = client.fetch_metadata(_candidate(), "b")
        kalshi_book = client.fetch_book(kalshi_metadata, depth=25)
        polymarket_book = client.fetch_book(polymarket_metadata, depth=25)

    decision = {
        "candidate_id": "pmxt_candidate_fixture",
        "status": "VERIFIED_EQUIVALENT",
        "evidence": {
            "native_markets": [
                {"venue": "kalshi", "native_market_id": "KXTEST-26-X"},
                {"venue": "polymarket", "native_market_id": "12345"},
            ]
        },
        "live_eligible": False,
    }
    result = calculate_shadow(
        decision,
        kalshi_book,
        polymarket_book,
        {
            "requested_size": 5.0,
            "max_book_age_seconds": 30.0,
            "max_cross_venue_skew_seconds": 3.0,
            "venue_fee_rates": {"kalshi": 0.01, "polymarket": 0.01},
            "annual_capital_rate": 0.05,
            "capital_lock_days": 2.0,
            "net_residual_threshold": 0.05,
            "explicit_slippage_buffer_per_unit": 0.0,
            "timestamp_skew_buffer_per_unit": 0.0,
            "settlement_divergence_buffer_per_unit": 0.0,
            "collateral_basis_buffer_per_unit": 0.0,
            "rebalancing_withdrawal_allowance_per_unit": 0.0,
        },
        evaluated_at=kalshi_book["received_at"],
    )

    assert "BOOK_A_MISSING_SOURCE_TIMESTAMP" in result["reasons"]
    assert "BOOK_B_NATIVE_FEE_EVIDENCE_NOT_VALID" in result["reasons"]
    assert result["status"] == "FEE_EVIDENCE_UNAVAILABLE"
    assert result["alert"] is result["live_eligible"] is False
