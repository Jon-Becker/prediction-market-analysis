from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from src.indexers.pmxt import semantic_equivalence as semantic


def _candidate() -> dict:
    return {
        "candidate_id": "pmxt_candidate_fixture",
        "snapshot_id": "fixture_snapshot",
        "cluster_id": "fixture_cluster",
        "relation": "identity",
        "relation_confidence": 0.99,
        "venue_a": "kalshi",
        "venue_b": "polymarket",
        "pmxt_market_id_a": "pmxt_kalshi",
        "pmxt_market_id_b": "pmxt_polymarket",
        "slug_a": "KXTEST-26-YES",
        "slug_b": "will-test-outcome-happen",
        "url_a": "https://kalshi.com/events/KXTEST-26",
        "url_b": "https://polymarket.com/event/test-event",
        "contract_address_a": None,
        "contract_address_b": "0x" + "1" * 64,
        "title_a": "Will the test outcome happen?",
        "title_b": "Will the test outcome happen?",
        "description_a": "Fixture",
        "description_b": "Fixture",
        "event_id_a": "pmxt_event_a",
        "event_id_b": "pmxt_event_b",
        "outcomes_a": [],
        "outcomes_b": [],
    }


def _kalshi_payload(path: str) -> dict:
    if path.endswith("/markets/KXTEST-26-YES"):
        return {
            "market": {
                "ticker": "KXTEST-26-YES",
                "event_ticker": "KXTEST-26",
                "series_ticker": "KXTEST",
                "title": "Will the test outcome happen?",
                "subtitle": "Fixture subtitle",
                "yes_sub_title": "Test outcome happens",
                "no_sub_title": "Test outcome does not happen",
                "rules_primary": "Resolves Yes from https://authority.example/result.",
                "rules_secondary": "If cancelled, the market is void and refunded.",
                "open_time": "2026-01-01T08:00:00-05:00",
                "close_time": "2026-12-31T23:00:00-05:00",
                "expiration_time": "2027-01-02T04:00:00Z",
                "earliest_resolution_time": "2026-12-31T23:00:01-05:00",
                "can_close_early": True,
                "early_close_condition": "May resolve immediately after official publication.",
                "settlement_timer_seconds": 3600,
                "strike_type": "binary",
                "fee_type": "quadratic",
                "updated_time": "2026-08-29T20:00:00Z",
            }
        }
    if path.endswith("/events/KXTEST-26/metadata"):
        return {
            "metadata": {
                "settlement_sources": [
                    {"name": "Official fixture authority", "url": "https://authority.example/result"}
                ]
            }
        }
    if path.endswith("/events/KXTEST-26"):
        return {
            "event": {
                "event_ticker": "KXTEST-26",
                "series_ticker": "KXTEST",
                "title": "Fixture event",
                "sub_title": "Fixture event subtitle",
                "mutually_exclusive": False,
                "markets": [{"ticker": "KXTEST-26-YES"}],
                "updated_time": "2026-08-29T20:00:00Z",
            }
        }
    if path.endswith("/series/KXTEST"):
        return {
            "series": {
                "ticker": "KXTEST",
                "title": "Fixture series",
                "fee_type": "quadratic",
                "fee_multiplier": 1.0,
                "updated_time": "2026-08-29T20:00:00Z",
            }
        }
    raise AssertionError(path)


def _polymarket_payload(path: str) -> list[dict]:
    event = {
        "id": "200",
        "slug": "test-event",
        "title": "Fixture event",
        "description": "If cancelled this resolves Other; it may resolve immediately.",
        "startDate": "2026-01-01T13:00:00Z",
        "endDate": "2026-12-31T23:00:00-05:00",
        "negRisk": False,
        "mutuallyExclusive": False,
        "updatedAt": "2026-08-29T20:00:00Z",
        "markets": [{"id": "100"}],
        "series": [{"id": "300", "slug": "fixture-series", "updatedAt": "2026-08-29T20:00:00Z"}],
    }
    if path.endswith("/events"):
        return [event]
    if path.endswith("/markets"):
        return [
            {
                "id": "100",
                "slug": "will-test-outcome-happen",
                "title": "Fixture market",
                "question": "Will the test outcome happen?",
                "description": "Resolves Yes from https://authority.example/result; a refund applies if void.",
                "resolutionSource": "https://authority.example/result",
                "resolvedBy": "0x" + "2" * 40,
                "conditionId": "0x" + "1" * 64,
                "questionID": "0x" + "3" * 64,
                "outcomes": '["Yes","No"]',
                "clobTokenIds": '["yes-token","no-token"]',
                "startDate": "2026-01-01T08:00:00-05:00",
                "endDate": "2026-12-31T23:00:00-05:00",
                "umaEndDate": "2027-01-02T04:00:00Z",
                "earliestResolutionTime": "2026-12-31T23:00:01-05:00",
                "umaBond": "500",
                "umaReward": "5",
                "umaResolutionStatuses": "[]",
                "negRisk": False,
                "canCloseEarly": True,
                "feeType": "fixture_fee",
                "feesEnabled": True,
                "updatedAt": "2026-08-29T20:00:00Z",
                "events": [event],
            }
        ]
    raise AssertionError(path)


def _transport(venue: str, calls: list[tuple[str, str]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        assert request.headers["accept-encoding"] == "identity"
        calls.append((venue, request.url.path))
        payload = _kalshi_payload(request.url.path) if venue == "kalshi" else _polymarket_payload(request.url.path)
        return httpx.Response(
            200,
            json=payload,
            headers=[
                ("Cache-Control", venue),
                ("Cache-Control", "second"),
                ("Set-Cookie", "must-not-be-retained=fixture-secret; HttpOnly"),
                ("X-Untrusted", "must-not-be-retained"),
            ],
        )

    return httpx.MockTransport(handler)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pair_capture_retains_raw_bytes_headers_and_full_native_fields(tmp_path: Path) -> None:
    writer = semantic.CaptureWriter(
        tmp_path / "capture",
        _fixture_permit=semantic._FIXTURE_PARSER_PERMIT,  # noqa: SLF001
    )
    calls: list[tuple[str, str]] = []
    with semantic.PublicMetadataClient(
        writer=writer,
        kalshi_transport=_transport("kalshi", calls),
        polymarket_transport=_transport("polymarket", calls),
        _fixture_permit=semantic._FIXTURE_PARSER_PERMIT,  # noqa: SLF001
    ) as client:
        assert all(native_client._trust_env is False for native_client in client._clients.values())  # noqa: SLF001
        pair = semantic.capture_pair(_candidate(), client)

    assert len(calls) == 6
    assert pair["coverage"]["complete"] is True
    assert pair["semantic_decision"] == "PENDING_SEMANTIC_REVIEW"
    assert pair["live_eligible"] is False
    kalshi = pair["venues"]["kalshi"]
    polymarket = pair["venues"]["polymarket"]
    assert kalshi["identifiers"]["event"][0]["value"] == "KXTEST-26"
    assert kalshi["identifiers"]["series"][0]["value"] == "KXTEST"
    assert kalshi["polarity"]["yes"]["label"] == "Test outcome happens"
    assert kalshi["rules"]["primary"].startswith("Resolves Yes")
    assert kalshi["times"]["opening"][0]["raw"] == "2026-01-01T08:00:00-05:00"
    assert kalshi["times"]["opening"][0]["normalized_utc"] == "2026-01-01T13:00:00Z"
    assert kalshi["resolution"]["source_urls"] == ["https://authority.example/result"]
    assert kalshi["exceptional_outcomes"]["void"]
    assert kalshi["exceptional_outcomes"]["refund"]
    assert kalshi["immutable_raw_hash"] != kalshi["normalized_rule_hash"]
    assert polymarket["identifiers"]["token"][0]["value"] == "yes-token"
    assert polymarket["identifiers"]["series"][0]["value"] == "300"
    assert polymarket["polarity"]["explicit"] is True
    assert polymarket["market_structure"]["negative_risk"] is False
    assert polymarket["resolution"]["oracle_process"]["classification"] == "UMA_METADATA_FIELDS_PRESENT"
    assert polymarket["exceptional_outcomes"]["indeterminate"]
    assert all(item["status"] == "OK" for item in kalshi["response_evidence"] + polymarket["response_evidence"])
    first = writer.request_records[0]
    body_path = writer.staging_path / first["body_path"]
    headers_path = writer.staging_path / first["headers_path"]
    assert _hash(body_path) == first["body_sha256"]
    assert _hash(headers_path) == first["headers_sha256"]
    header_document = json.loads(headers_path.read_text(encoding="utf-8"))
    assert (
        len(
            [
                item
                for item in header_document["raw_header_pairs_in_order"]
                if item["name_latin1"].lower() == "cache-control"
            ]
        )
        == 2
    )
    retained_names = {item["name_latin1"].lower() for item in header_document["raw_header_pairs_in_order"]}
    assert "set-cookie" not in retained_names
    assert "x-untrusted" not in retained_names


def test_missing_native_fields_stay_explicitly_partial(tmp_path: Path) -> None:
    writer = semantic.CaptureWriter(
        tmp_path / "capture",
        _fixture_permit=semantic._FIXTURE_PARSER_PERMIT,  # noqa: SLF001
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if "kalshi" in request.url.host:
            return httpx.Response(200, json={"market": {"ticker": "KXTEST-26-YES"}})
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    with semantic.PublicMetadataClient(
        writer=writer,
        kalshi_transport=transport,
        polymarket_transport=transport,
        _fixture_permit=semantic._FIXTURE_PARSER_PERMIT,  # noqa: SLF001
    ) as client:
        pair = semantic.capture_pair(_candidate(), client)

    assert pair["coverage"]["complete"] is False
    assert "event_id" in pair["venues"]["kalshi"]["missing_requirements"]
    assert "market_id" in pair["venues"]["polymarket"]["missing_requirements"]
    assert pair["semantic_decision"] == "PENDING_SEMANTIC_REVIEW"


def test_legacy_capture_is_explicitly_quarantined_and_not_publicly_exported() -> None:
    assert semantic.LEGACY_CAPTURE_QUARANTINED is True
    assert semantic.PRODUCTION_READY is False
    assert semantic.REAL_NETWORK_CAPTURE_ENABLED is False
    assert {
        "CaptureWriter",
        "PublicMetadataClient",
        "capture_pair",
        "run_capture",
    }.isdisjoint(semantic.__all__)


def test_writer_requires_private_fixture_permit_before_creating_artifacts(tmp_path: Path) -> None:
    output_path = tmp_path / "must-not-exist"

    with pytest.raises(semantic.SemanticCaptureError, match="quarantined"):
        semantic.CaptureWriter(output_path)

    assert not output_path.exists()
    assert not output_path.with_name("must-not-exist.inprogress").exists()


def test_parser_client_requires_private_permit_and_exact_mock_transports(tmp_path: Path) -> None:
    writer = semantic.CaptureWriter(
        tmp_path / "fixture-capture",
        _fixture_permit=semantic._FIXTURE_PARSER_PERMIT,  # noqa: SLF001
    )
    calls: list[tuple[str, str]] = []

    with pytest.raises(semantic.SemanticCaptureError, match="quarantined"):
        semantic.PublicMetadataClient(
            writer=writer,
            kalshi_transport=_transport("kalshi", calls),
            polymarket_transport=_transport("polymarket", calls),
        )

    with pytest.raises(semantic.SemanticCaptureError, match="exact httpx.MockTransport"):
        semantic.PublicMetadataClient(
            writer=writer,
            kalshi_transport=_transport("kalshi", calls),
            polymarket_transport=None,
            _fixture_permit=semantic._FIXTURE_PARSER_PERMIT,  # noqa: SLF001
        )

    assert calls == []


@pytest.mark.parametrize("network_authorized", [False, True])
def test_run_capture_fails_before_transport_or_artifacts_even_when_authorized(
    tmp_path: Path,
    network_authorized: bool,
) -> None:
    calls: list[tuple[str, str]] = []
    output_path = Path("data/output/must-not-exist")

    with pytest.raises(semantic.SemanticCaptureError, match="quarantined"):
        semantic.run_capture(
            repository_root=tmp_path,
            expected_freeze_sha256="0" * 64,
            output_path=output_path,
            network_authorized=network_authorized,
            kalshi_transport=_transport("kalshi", calls),
            polymarket_transport=_transport("polymarket", calls),
        )

    assert calls == []
    assert not (tmp_path / output_path).exists()
    assert not (tmp_path / output_path).with_name("must-not-exist.inprogress").exists()


def test_module_cli_fails_immediately(capsys: pytest.CaptureFixture[str]) -> None:
    assert semantic.main(["--capture-public-metadata"]) == 2
    assert "quarantined" in capsys.readouterr().err


def test_frozen_protocol_and_schema_match_implementation_contract() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    protocol = json.loads((repository_root / semantic.DEFAULT_PROTOCOL_PATH).read_text(encoding="utf-8"))
    schema = json.loads((repository_root / semantic.DEFAULT_SCHEMA_PATH).read_text(encoding="utf-8"))

    assert protocol["protocol_id"] == semantic.PROTOCOL_ID
    assert protocol["schema_version"] == semantic.SCHEMA_VERSION
    assert protocol["input"]["sha256"] == semantic.DEFAULT_CANDIDATE_SHA256
    assert protocol["input"]["pair_count"] == semantic.EXPECTED_PAIR_COUNT
    assert protocol["network_authority"]["maximum_requests_per_pair"] == semantic.MAX_REQUESTS_PER_PAIR
    assert protocol["network_authority"]["maximum_requests_total"] == (
        semantic.EXPECTED_PAIR_COUNT * semantic.MAX_REQUESTS_PER_PAIR
    )
    assert protocol["run"]["final_path"] == semantic.DEFAULT_OUTPUT_PATH.as_posix()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["semantic_decision"]["const"] == "PENDING_SEMANTIC_REVIEW"
    assert schema["properties"]["live_eligible"]["const"] is False
    venue_required = set(schema["$defs"]["venue_record"]["required"])
    assert {
        "identifiers",
        "exact_text",
        "polarity",
        "market_structure",
        "times",
        "rules",
        "resolution",
        "early_close",
        "exceptional_outcomes",
        "revisions",
        "response_evidence",
        "immutable_raw_hash",
        "normalized_rule_hash",
        "missing_requirements",
    } <= venue_required
