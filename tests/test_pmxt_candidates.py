"""Tests for the read-only PMXT candidate-sync milestone."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

import src.indexers.pmxt.candidates as candidates_module
import src.indexers.pmxt.market_clusters as market_clusters_module
from src.indexers.pmxt.candidates import normalize_clusters, persist_sync
from src.indexers.pmxt.client import PmxtRouterClient, PmxtRouterError, extract_clusters
from src.indexers.pmxt.market_clusters import PmxtMarketClustersIndexer
from src.indexers.pmxt.models import PmxtQuery


def _cluster_fixture() -> dict:
    return {
        "clusterId": "mcl_test_001",
        "canonicalTitle": "Will the test candidate win?",
        "category": "Test",
        "relations": ["identity"],
        "confidence": 0.96,
        "markets": [
            {
                "marketId": "pm_test_market",
                "sourceExchange": "polymarket",
                "title": "Will the test candidate win?",
                "slug": "test-candidate-win",
                "outcomes": [
                    {"outcomeId": "pm_yes", "label": "Yes", "price": 0.54},
                    {"outcomeId": "pm_no", "label": "No", "price": 0.46},
                ],
            },
            {
                "marketId": "kx_test_market",
                "sourceExchange": "kalshi",
                "title": "Test candidate wins",
                "outcomes": [
                    {"outcomeId": "kx_yes", "label": "Yes", "price": 0.57},
                    {"outcomeId": "kx_no", "label": "No", "price": 0.43},
                ],
            },
        ],
        "rawMatches": [
            {
                "marketAId": "pm_test_market",
                "marketBId": "kx_test_market",
                "relation": "identity",
                "confidence": 0.94,
                "reasoning": "The resolution conditions appear equivalent.",
            }
        ],
    }


def _stream_response(
    status_code: int,
    body: bytes,
    *,
    headers: dict[str, str] | list[tuple[str, str]] | None = None,
) -> httpx.Response:
    return httpx.Response(status_code, headers=headers, stream=httpx.ByteStream(body))


def test_router_client_sends_only_query_metadata_and_auth_header() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["authorization"]
        seen["accept_encoding"] = request.headers["accept-encoding"]
        return _stream_response(200, json.dumps({"clusters": [_cluster_fixture()]}).encode())

    query = PmxtQuery()
    with PmxtRouterClient("pmxt_test_secret", transport=httpx.MockTransport(handler)) as client:
        assert client._client._trust_env is False
        payload = client.fetch_market_clusters(query)

    assert extract_clusters(payload)[0]["clusterId"] == "mcl_test_001"
    assert seen["authorization"] == "Bearer pmxt_test_secret"
    assert seen["accept_encoding"] == "identity"
    assert "minConfidence=0.8" in str(seen["url"])
    assert "venues=kalshi%2Cpolymarket" in str(seen["url"])


def test_router_response_body_is_hard_bounded() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return _stream_response(200, json.dumps({"clusters": [], "padding": "x" * 200}).encode())

    with PmxtRouterClient(
        "pmxt_test_secret",
        transport=httpx.MockTransport(handler),
        max_response_bytes=100,
    ) as client:
        with pytest.raises(PmxtRouterError, match="response exceeded"):
            client.fetch_market_clusters(PmxtQuery())

    assert requests == 1


def test_router_rejects_nonfinite_json_constants() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(200, b'{"clusters": [], "confidence": NaN}')

    with PmxtRouterClient("pmxt_test_secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError, match="invalid JSON"):
            client.fetch_market_clusters(PmxtQuery())


def test_router_rejects_redirect_response_even_if_body_looks_valid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(302, b'{"clusters": []}')

    with PmxtRouterClient("pmxt_test_secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError, match="HTTP 302"):
            client.fetch_market_clusters(PmxtQuery())


def test_router_429_minute_body_and_safe_headers_are_exact_evidence_without_retry() -> None:
    requests: list[httpx.Request] = []
    raw_body = (
        b'{\n  "error": "rate_limit_exceeded", "plan": "free", '
        b'"limit": 60, "used": 60, "window": "1 minute"\n}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _stream_response(
            429,
            raw_body,
            headers=[
                ("Content-Type", "application/json"),
                ("Retry-After", "37"),
                ("X-Request-ID", "fixture-request-id"),
                ("Set-Cookie", "session=unsafe-cookie"),
                ("Authorization", "unsafe-response-authorization"),
                ("X-Api-Key", "unsafe-response-key"),
            ],
        )

    with PmxtRouterClient("pmxt_test_secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError, match="rate limited") as raised:
            client.fetch_market_clusters(PmxtQuery())
        assert len(client._client.cookies) == 0

    error = raised.value
    assert len(requests) == 1
    assert requests[0].headers["accept-encoding"] == "identity"
    assert error.reason_code == "PMXT_HTTP_RATE_LIMITED"
    response = error.evidence["response"]
    body = response["body"]
    assert response["status_code"] == 429
    assert base64.b64decode(body["raw_body_base64"], validate=True) == raw_body
    assert body["raw_body_sha256"] == hashlib.sha256(raw_body).hexdigest()
    assert body["raw_body_byte_size"] == len(raw_body)
    assert body["capture_status"] == "EXACT_COMPLETE"
    assert response["response_headers"]["retry-after"] == "37"
    assert response["response_headers"]["x-request-id"] == "fixture-request-id"
    serialized = json.dumps(error.evidence, sort_keys=True)
    for unsafe in ("set-cookie", "unsafe-cookie", "authorization", "unsafe-response-key"):
        assert unsafe not in serialized.lower()
    assert response["rate_limit"] == {
        "classification": "PER_MINUTE",
        "retry_policy": "NO_RETRY_THIS_RUN",
        "requests_attempted": 1,
        "retries_performed": 0,
        "retry_after": {"kind": "DELTA_SECONDS", "raw": "37", "seconds": 37},
        "body_parse_status": "VALID_JSON_OBJECT",
        "server_error_code": "rate_limit_exceeded",
        "quota": {"plan": "free", "limit": 60, "used": 60, "window": "1 minute"},
    }


def test_router_429_monthly_quota_is_distinct_and_not_retried() -> None:
    requests = 0
    raw_body = b'{"error":"monthly_quota_exceeded","plan":"free","limit":25000,"used":25000}'

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return _stream_response(429, raw_body, headers={"Content-Type": "application/json"})

    with PmxtRouterClient("pmxt_test_secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError) as raised:
            client.fetch_market_clusters(PmxtQuery())

    rate_limit = raised.value.evidence["response"]["rate_limit"]
    assert requests == 1
    assert rate_limit["classification"] == "MONTHLY_QUOTA"
    assert rate_limit["server_error_code"] == "monthly_quota_exceeded"
    assert rate_limit["quota"] == {"plan": "free", "limit": 25000, "used": 25000}
    assert rate_limit["retry_after"] == {"kind": "ABSENT"}
    assert rate_limit["retries_performed"] == 0


@pytest.mark.parametrize(
    ("raw_body", "parse_status"),
    [
        (b'{"error":', "INVALID_JSON"),
    ],
)
def test_router_429_invalid_body_is_withheld_when_credential_scan_is_indeterminate(
    raw_body: bytes,
    parse_status: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(429, raw_body)

    with PmxtRouterClient("pmxt_test_secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError) as raised:
            client.fetch_market_clusters(PmxtQuery())

    response = raised.value.evidence["response"]
    assert response["rate_limit"]["classification"] == "UNKNOWN_429"
    assert response["rate_limit"]["body_parse_status"] == parse_status
    assert response["body"]["capture_status"] == "WITHHELD_CREDENTIAL_SCAN_INDETERMINATE"
    assert response["body"]["credential_scan_status"] == "UNKNOWN_INVALID_JSON"
    assert "raw_body_base64" not in response["body"]
    assert "raw_body_sha256" not in response["body"]


def test_router_429_duplicate_error_codes_are_unknown_but_credential_scan_remains_conclusive() -> None:
    raw_body = b'{"error":"rate_limit_exceeded","error":"monthly_quota_exceeded"}'

    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(429, raw_body)

    with PmxtRouterClient("pmxt_test_secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError) as raised:
            client.fetch_market_clusters(PmxtQuery())

    response = raised.value.evidence["response"]
    assert response["rate_limit"]["classification"] == "UNKNOWN_429"
    assert response["rate_limit"]["body_parse_status"] == "INVALID_JSON"
    assert response["body"]["capture_status"] == "EXACT_COMPLETE"
    assert base64.b64decode(response["body"]["raw_body_base64"], validate=True) == raw_body


def test_router_non_utf8_body_is_withheld_when_credential_scan_is_indeterminate() -> None:
    api_key = "fixture_utf16_secret"
    raw_body = json.dumps({"error": "rate_limit_exceeded", "echo": api_key}).encode("utf-16le")
    assert api_key.encode() not in raw_body

    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(429, raw_body, headers={"Content-Type": "application/json; charset=utf-16"})

    with PmxtRouterClient(api_key, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError, match="credential safety scan") as raised:
            client.fetch_market_clusters(PmxtQuery())

    response = raised.value.evidence["response"]
    assert raised.value.reason_code == "PMXT_RESPONSE_CREDENTIAL_SCAN_INDETERMINATE"
    assert response["body"]["capture_status"] == "WITHHELD_CREDENTIAL_SCAN_INDETERMINATE"
    assert response["body"]["credential_scan_status"] == "UNKNOWN_INVALID_JSON"
    assert "raw_body_base64" not in response["body"]
    assert api_key not in json.dumps(raised.value.evidence, sort_keys=True)


def test_router_oversized_429_withholds_incomplete_body_and_does_not_retry() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return _stream_response(429, b"x" * 101)

    with PmxtRouterClient(
        "pmxt_test_secret",
        transport=httpx.MockTransport(handler),
        max_response_bytes=100,
    ) as client:
        with pytest.raises(PmxtRouterError, match="response exceeded") as raised:
            client.fetch_market_clusters(PmxtQuery())

    body = raised.value.evidence["response"]["body"]
    assert requests == 1
    assert raised.value.reason_code == "PMXT_RESPONSE_TOO_LARGE"
    assert body == {
        "capture_status": "WITHHELD_INCOMPLETE_RESPONSE",
        "retained_byte_size": 100,
        "observed_byte_size_lower_bound": 101,
        "complete": False,
        "max_response_bytes": 100,
        "representation": "HTTP_ENTITY_BYTES_ACCEPT_ENCODING_IDENTITY",
        "credential_scan_status": "UNKNOWN_INCOMPLETE",
    }
    assert raised.value.evidence["response"]["rate_limit"]["body_parse_status"] == "NOT_PARSED_INCOMPLETE"


def test_router_error_evidence_withholds_reflected_api_key_from_body_and_headers() -> None:
    api_key = "fixture_secret_must_never_persist"
    raw_body = json.dumps(
        {
            "error": "rate_limit_exceeded",
            "plan": api_key,
            "limit": 60,
            "used": 60,
            "window": "1 minute",
            "message": f"Bearer {api_key}",
        }
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(
            429,
            raw_body,
            headers={"Retry-After": api_key, "X-Request-ID": api_key, "X-Api-Key": api_key},
        )

    with PmxtRouterClient(api_key, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError) as raised:
            client.fetch_market_clusters(PmxtQuery())

    serialized = json.dumps(raised.value.evidence, sort_keys=True)
    response = raised.value.evidence["response"]
    assert api_key not in serialized
    assert api_key not in str(raised.value)
    assert api_key not in repr(raised.value)
    assert response["body"]["capture_status"] == "WITHHELD_API_KEY_ECHO"
    assert "raw_body_base64" not in response["body"]
    assert "raw_body_sha256" not in response["body"]
    assert response["response_headers"] == {}
    assert response["response_header_capture"]["omitted_for_credential_echo"] == 2
    assert response["rate_limit"]["retry_after"] == {"kind": "WITHHELD_CREDENTIAL_ECHO"}
    assert response["rate_limit"]["credential_fields_withheld"] == ["plan"]


def test_router_success_body_reflecting_api_key_fails_closed_before_payload_return() -> None:
    api_key = "fixture_secret_must_never_return"
    raw_body = json.dumps({"clusters": [], "reflected": api_key}).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(200, raw_body)

    with PmxtRouterClient(api_key, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError, match="credential material") as raised:
            client.fetch_market_clusters(PmxtQuery())

    assert raised.value.reason_code == "PMXT_RESPONSE_CREDENTIAL_ECHO"
    assert raised.value.evidence["response"]["body"]["capture_status"] == "WITHHELD_API_KEY_ECHO"
    assert api_key not in json.dumps(raised.value.evidence, sort_keys=True)


def test_router_success_body_unicode_escaping_api_key_fails_credential_scan() -> None:
    api_key = "fixture_escaped_secret"
    escaped_key = "".join(f"\\u{ord(character):04x}" for character in api_key)
    raw_body = f'{{"clusters":[],"echo":"{escaped_key}"}}'.encode()
    assert api_key.encode() not in raw_body

    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(200, raw_body)

    with PmxtRouterClient(api_key, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError, match="credential material") as raised:
            client.fetch_market_clusters(PmxtQuery())

    body = raised.value.evidence["response"]["body"]
    assert body["capture_status"] == "WITHHELD_API_KEY_ECHO"
    assert body["credential_scan_status"] == "PRESENT"
    assert "raw_body_base64" not in body


def test_router_unexpected_content_encoding_withholds_raw_body_before_error_persistence() -> None:
    api_key = "fixture_secret_hidden_inside_gzip"
    compressed = gzip.compress(json.dumps({"error": "rate_limit_exceeded", "echo": api_key}).encode())
    assert api_key.encode() not in compressed

    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(429, compressed, headers={"Content-Encoding": "gzip"})

    with PmxtRouterClient(api_key, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError, match="unexpected content encoding") as raised:
            client.fetch_market_clusters(PmxtQuery())

    response = raised.value.evidence["response"]
    assert raised.value.reason_code == "PMXT_UNEXPECTED_CONTENT_ENCODING"
    assert response["body"]["capture_status"] == "WITHHELD_UNEXPECTED_CONTENT_ENCODING"
    assert "raw_body_base64" not in response["body"]
    assert "raw_body_sha256" not in response["body"]
    assert api_key not in json.dumps(raised.value.evidence, sort_keys=True)


def test_router_retry_after_truncation_is_invalid_evidence() -> None:
    raw_body = b'{"error":"rate_limit_exceeded"}'

    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(429, raw_body, headers={"Retry-After": "9" * 1_025})

    with PmxtRouterClient("pmxt_test_secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError) as raised:
            client.fetch_market_clusters(PmxtQuery())

    response = raised.value.evidence["response"]
    assert response["response_header_capture"]["retry_after_value_truncated"] is True
    assert response["rate_limit"]["retry_after"] == {"kind": "INVALID", "reason": "value_truncated"}


def test_router_retry_after_omitted_by_header_bound_is_incomplete_evidence() -> None:
    headers = [("X-Request-ID", f"request-{index}") for index in range(32)]
    headers.append(("Retry-After", "12"))

    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(429, b'{"error":"rate_limit_exceeded"}', headers=headers)

    with PmxtRouterClient("pmxt_test_secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError) as raised:
            client.fetch_market_clusters(PmxtQuery())

    response = raised.value.evidence["response"]
    assert response["response_header_capture"]["retry_after_omitted_for_bound"] is True
    assert response["rate_limit"]["retry_after"] == {"kind": "INCOMPLETE", "reason": "header_item_bound"}


def test_router_extreme_retry_after_date_is_invalid_instead_of_escaping_error_wrapper() -> None:
    extreme_date = "Fri, 31 Dec 9999 23:59:59 -2359"

    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(
            429,
            b'{"error":"rate_limit_exceeded"}',
            headers={"Retry-After": extreme_date},
        )

    with PmxtRouterClient("pmxt_test_secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(PmxtRouterError, match="rate limited") as raised:
            client.fetch_market_clusters(PmxtQuery())

    retry_after = raised.value.evidence["response"]["rate_limit"]["retry_after"]
    assert retry_after == {"kind": "INVALID", "raw": extreme_date}


def test_custom_pmxt_base_url_requires_an_injected_test_transport() -> None:
    with pytest.raises(ValueError, match="custom PMXT base URL"):
        PmxtRouterClient("pmxt_test_secret", base_url="https://not-pmxt.example")


def test_normalization_keeps_candidates_unverified_and_execution_disabled() -> None:
    query = PmxtQuery()
    result = normalize_clusters(
        [_cluster_fixture()],
        query=query,
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["relation"] == "identity"
    assert row["raw_edge_present"] is True
    assert row["verification_status"] == "UNVERIFIED"
    assert row["semantic_verification"] == "PENDING_REVIEW"
    assert row["book_status"] == "NOT_REFRESHED"
    assert row["price_source"] == "PMXT_CATALOG_ONLY"
    assert row["catalog_price_conflicts_with_displayed_spread"] is False
    assert row["live_eligible"] is False


def test_conflicting_pmxt_catalog_semantics_remain_unverified_candidates() -> None:
    cluster = _cluster_fixture()
    polymarket, kalshi = cluster["markets"]
    polymarket.update(
        {
            "title": "Will Candidate X win?",
            "description": "Resolves YES if Candidate X wins.",
            "resolutionDate": "2026-11-04T02:00:00Z",
            "sourceMetadata": {
                "settlement_authority": "Authority A",
                "resolution_source": "Official source A",
            },
            "outcomes": [
                {"outcomeId": "pm_yes", "label": "Candidate X wins", "price": 0.54},
                {"outcomeId": "pm_no", "label": "Candidate X loses", "price": 0.46},
            ],
        }
    )
    kalshi.update(
        {
            "title": "Will Candidate X lose?",
            "description": "Resolves YES if Candidate X loses.",
            "resolutionDate": "2026-11-05T02:00:00Z",
            "sourceMetadata": {
                "settlement_authority": "Authority B",
                "resolution_source": "Official source B",
            },
            "outcomes": [
                {"outcomeId": "kx_yes", "label": "Candidate X loses", "price": 0.57},
                {"outcomeId": "kx_no", "label": "Candidate X wins", "price": 0.43},
            ],
        }
    )

    result = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    assert result.rejected == []
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["title_a"] != row["title_b"]
    assert row["description_a"] != row["description_b"]
    assert row["resolution_date_a"] != row["resolution_date_b"]
    assert row["source_metadata_a"] != row["source_metadata_b"]
    assert [outcome["label"] for outcome in row["outcomes_a"]] != [outcome["label"] for outcome in row["outcomes_b"]]
    assert row["verification_status"] == "UNVERIFIED"
    assert row["semantic_verification"] == "PENDING_REVIEW"
    assert row["live_eligible"] is False


def test_catalog_price_within_displayed_spread_is_diagnostic_only() -> None:
    cluster = _cluster_fixture()
    cluster["markets"][0]["outcomes"][0].update({"bestBid": 0.52, "bestAsk": 0.55})

    result = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    row = result.rows[0]
    outcome = row["outcomes_a"][0]
    assert outcome["best_bid"] == 0.52
    assert outcome["best_ask"] == 0.55
    assert outcome["catalog_quote_coherence_status"] == "WITHIN_DISPLAYED_SPREAD"
    assert row["catalog_price_conflicts_with_displayed_spread"] is False
    assert row["price_source"] == "PMXT_CATALOG_ONLY"
    assert row["live_eligible"] is False


def test_catalog_price_outside_displayed_spread_is_flagged_but_not_rejected() -> None:
    cluster = _cluster_fixture()
    cluster["markets"][0]["outcomes"][0].update({"bestBid": 0.40, "bestAsk": 0.45})

    result = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    assert len(result.rows) == 1
    assert result.rejected == []
    row = result.rows[0]
    assert row["outcomes_a"][0]["catalog_quote_coherence_status"] == "OUTSIDE_DISPLAYED_SPREAD"
    assert row["catalog_price_conflicts_with_displayed_spread"] is True
    assert row["fair_value_status"] == "NOT_CALCULATED"


@pytest.mark.parametrize(
    ("best_bid", "best_ask"),
    [
        ("not-a-number", 0.55),
        (float("inf"), 0.55),
        (-0.01, 0.55),
        (0.60, 0.55),
        (0.52, 1.01),
    ],
)
def test_malformed_displayed_spread_is_invalid_not_a_conflict(best_bid: object, best_ask: object) -> None:
    cluster = _cluster_fixture()
    cluster["markets"][0]["outcomes"][0].update({"bestBid": best_bid, "bestAsk": best_ask})

    result = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    row = result.rows[0]
    assert row["outcomes_a"][0]["catalog_quote_coherence_status"] == "INVALID_DISPLAYED_SPREAD"
    assert row["catalog_price_conflicts_with_displayed_spread"] is False


def test_missing_displayed_quote_is_not_evaluable() -> None:
    result = normalize_clusters(
        [_cluster_fixture()],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    outcome = result.rows[0]["outcomes_a"][0]
    assert outcome["best_bid"] is None
    assert outcome["best_ask"] is None
    assert outcome["catalog_quote_coherence_status"] == "NOT_EVALUABLE"
    assert result.rows[0]["catalog_price_conflicts_with_displayed_spread"] is False


def test_single_venue_clusters_are_rejected() -> None:
    cluster = _cluster_fixture()
    cluster["markets"] = [cluster["markets"][0]]
    result = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    assert result.rows == []
    assert result.rejected[0]["reason"] == "fewer_than_two_usable_markets"


def test_missing_direct_edge_is_rejected_instead_of_promoting_cluster_relation() -> None:
    cluster = _cluster_fixture()
    cluster["rawMatches"] = []
    result = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    assert result.rows == []
    assert result.rejected[0]["reason"] == "missing_direct_raw_match"


def test_non_identity_direct_edge_is_rejected_until_separately_modeled() -> None:
    cluster = _cluster_fixture()
    cluster["rawMatches"][0]["relation"] = "overlap"
    result = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    assert result.rows == []
    assert result.rejected[0]["reason"] == "non_identity_relation"


@pytest.mark.parametrize(
    ("missing_field", "reason"),
    [
        ("relation", "missing_direct_edge_relation"),
        ("confidence", "missing_direct_edge_confidence"),
    ],
)
def test_direct_edge_requires_its_own_identity_relation_and_confidence(missing_field: str, reason: str) -> None:
    cluster = _cluster_fixture()
    del cluster["rawMatches"][0][missing_field]
    result = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    assert result.rows == []
    assert result.rejected[0]["reason"] == reason
    assert result.rejected[0]["live_eligible"] is False


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01])
def test_invalid_direct_edge_confidence_fails_closed(confidence: float) -> None:
    cluster = _cluster_fixture()
    cluster["rawMatches"][0]["confidence"] = confidence

    result = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    assert result.rows == []
    assert result.rejected[0]["reason"] == "invalid_direct_edge_confidence"
    assert result.rejected[0]["live_eligible"] is False


def test_duplicate_direct_edges_are_order_independently_rejected_as_ambiguous() -> None:
    cluster = _cluster_fixture()
    contradictory = deepcopy(cluster["rawMatches"][0])
    contradictory["relation"] = "overlap"
    cluster["rawMatches"].append(contradictory)

    forward = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )
    cluster["rawMatches"].reverse()
    reverse = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    assert forward.rows == reverse.rows == []
    assert forward.rejected == reverse.rejected
    assert forward.rejected[0]["reason"] == "ambiguous_duplicate_direct_raw_matches"


def test_duplicate_normalized_market_record_rejects_cluster_before_candidate_emission() -> None:
    cluster = _cluster_fixture()
    cluster["markets"].append(deepcopy(cluster["markets"][1]))

    result = normalize_clusters(
        [cluster],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    assert result.rows == []
    assert result.rejected[0]["reason"] == "duplicate_normalized_market_key"
    assert result.rejected[0]["duplicate_market_keys"] == [{"venue": "kalshi", "pmxt_market_id": "kx_test_market"}]


def test_repeated_cluster_id_is_rejected_once_without_duplicate_candidates() -> None:
    cluster = _cluster_fixture()

    result = normalize_clusters(
        [cluster, deepcopy(cluster)],
        query=PmxtQuery(),
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )

    assert result.rows == []
    assert len(result.rejected) == 1
    assert result.rejected[0]["reason"] == "duplicate_cluster_id"
    assert result.rejected[0]["duplicate_count"] == 2


def test_missing_key_fails_before_client_or_artifact_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PMXT_API_KEY", raising=False)
    calls = 0

    class ForbiddenClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            raise AssertionError("network client must not be constructed without a key")

    monkeypatch.setattr(market_clusters_module, "PmxtRouterClient", ForbiddenClient)
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(RuntimeError, match="no network request was made"):
        PmxtMarketClustersIndexer(output_dir=output_dir).sync_once()

    assert calls == 0
    assert not output_dir.exists()


def test_persistence_is_run_immutable_and_retains_raw_payload(tmp_path: Path) -> None:
    query = PmxtQuery()
    result = normalize_clusters(
        [_cluster_fixture()],
        query=query,
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
    )
    artifacts = persist_sync(
        tmp_path,
        snapshot_id="snapshot_test",
        observed_at="2026-08-25T12:00:00Z",
        query=query,
        raw_payload={"clusters": [_cluster_fixture()]},
        result=result,
        cluster_count=1,
    )

    raw = json.loads(artifacts.raw_path.read_text(encoding="utf-8"))
    assert raw["payload"]["clusters"][0]["clusterId"] == "mcl_test_001"
    mapping = json.loads(artifacts.mapping_path.read_text(encoding="utf-8").splitlines()[0])
    assert mapping["snapshot_id"] == "snapshot_test"
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["live_eligible"] is False

    with pytest.raises(FileExistsError):
        persist_sync(
            tmp_path,
            snapshot_id="snapshot_test",
            observed_at="2026-08-25T12:00:00Z",
            query=query,
            raw_payload={"clusters": []},
            result=result,
            cluster_count=0,
        )


def test_legacy_sync_rolls_back_new_files_if_a_later_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = normalize_clusters(
        [_cluster_fixture()],
        query=PmxtQuery(),
        snapshot_id="snapshot_rollback",
        observed_at="2026-08-25T12:00:00Z",
    )
    real_write_new = candidates_module._write_new

    def fail_on_rejections(path: Path, content: str) -> None:
        if path.name.startswith("rejected_clusters_"):
            raise OSError("simulated later legacy write failure")
        real_write_new(path, content)

    monkeypatch.setattr(candidates_module, "_write_new", fail_on_rejections)

    with pytest.raises(OSError, match="simulated later legacy write failure"):
        persist_sync(
            tmp_path,
            snapshot_id="snapshot_rollback",
            observed_at="2026-08-25T12:00:00Z",
            query=PmxtQuery(),
            raw_payload={"clusters": [_cluster_fixture()]},
            result=result,
            cluster_count=1,
        )

    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []
