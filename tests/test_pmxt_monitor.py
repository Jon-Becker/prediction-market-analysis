"""End-to-end fixture tests for the bounded read-only PMXT monitor."""

from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

import src.indexers.pmxt.monitor as monitor_module
from src.indexers.pmxt.candidates import utc_now_iso
from src.indexers.pmxt.monitor import MonitorConfig, PmxtReadOnlyMonitor


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pmxt_cluster() -> dict:
    return {
        "clusterId": "mcl_monitor_fixture",
        "canonicalTitle": "Will Candidate X win?",
        "relations": ["identity"],
        "confidence": 0.99,
        "markets": [
            {
                "marketId": "pmxt_catalog_kalshi_uuid",
                "sourceExchange": "kalshi",
                "title": "Will Candidate X win?",
                "sourceMetadata": {"ticker": "KXTEST-26-X"},
                "outcomes": [
                    {"outcomeId": "pmxt_k_yes", "label": "Yes", "price": 0.40},
                    {"outcomeId": "pmxt_k_no", "label": "No", "price": 0.60},
                ],
            },
            {
                "marketId": "pmxt_catalog_polymarket_uuid",
                "sourceExchange": "polymarket",
                "title": "Will Candidate X win?",
                "slug": "candidate-x-win",
                "outcomes": [
                    {"outcomeId": "pmxt_p_yes", "label": "Yes", "price": 0.42},
                    {"outcomeId": "pmxt_p_no", "label": "No", "price": 0.58},
                ],
            },
        ],
        "rawMatches": [
            {
                "marketAId": "pmxt_catalog_kalshi_uuid",
                "marketBId": "pmxt_catalog_polymarket_uuid",
                "relation": "identity",
                "confidence": 0.98,
            }
        ],
    }


def _pmxt_transport(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.dumps({"clusters": [_pmxt_cluster()]}).encode()
        return httpx.Response(200, stream=httpx.ByteStream(body))

    return httpx.MockTransport(handler)


def _acquisition(path: str, payload: object, *, params: dict[str, str] | None = None) -> dict:
    raw_body = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    http_date = "Sat, 29 Aug 2026 12:00:00 GMT"
    cache_control = "no-store"
    request_id = f"fixture-{path.replace('/', '-').strip('-')}"
    return {
        "method": "GET",
        "path": path,
        "params": dict(params or {}),
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
        "final_url": f"https://fixture.invalid{path}",
        "requested_at": "2026-08-29T12:00:00.000000Z",
        "received_at": "2026-08-29T12:00:00.005000Z",
    }


def _fee_evidence(venue: str, observed_at: str) -> dict:
    common = {
        "schema_version": 1,
        "status": "VALID",
        "venue": venue,
        "liquidity_role": "TAKER",
        "passive_fills_assumed": False,
        "schedule_observed_at": observed_at,
        "live_eligible": False,
    }
    if venue == "kalshi":
        schedule = {
            "official_fee_schedule": {
                "status": "REVIEWED",
                "raw_body_sha256": "7" * 64,
                "formula_binding": {
                    "binding_id": "fixture-reviewed-kalshi-fee-formula-v1",
                    "taker_base_coefficient": "0.07",
                    "trade_fee_rounding_quantum": "0.000001",
                    "balance_precision_upper_bound": "0.01",
                },
            },
            "effective": {
                "fee_type": "quadratic",
                "fee_multiplier": 1.0,
                "taker_base_coefficient": "0.07",
                "trade_fee_rounding_quantum": "0.000001",
                "balance_precision_upper_bound": "0.01",
            },
        }
        raw_response = {
            "event": {"event": {"event_ticker": "KXTEST-26", "series_ticker": "KXTEST"}},
            "series": {"series": {"ticker": "KXTEST", "fee_type": "quadratic", "fee_multiplier": 1}},
            "series_fee_changes": {"series_fee_change_arr": []},
            "event_fee_changes": {"event_fee_changes": [], "cursor": ""},
        }
        return {
            **common,
            "model": "KALSHI_QUADRATIC_TAKER",
            "effective_from": "2026-01-01T00:00:00Z",
            "official_fee_schedule_sha256": "7" * 64,
            "formula_binding_id": "fixture-reviewed-kalshi-fee-formula-v1",
            "schedule_sha256": _canonical_sha256(schedule),
            "raw_sha256": _canonical_sha256(raw_response),
            "schedule": schedule,
            "raw_response": raw_response,
        }
    raw_fee_rates = {"YES": {"base_fee": 100}, "NO": {"base_fee": 100}}
    fee_schedule = {
        "native_condition_id": "condition-12345",
        "native_outcome_ids": {"YES": "polymarket_yes", "NO": "polymarket_no"},
        "fees_enabled": True,
        "base_fee_bps_estimate_by_outcome": {"YES": 100, "NO": 100},
        "condition_fee_details": None,
    }
    return {
        **common,
        "status": "FAIL_CLOSED",
        "model": "POLYMARKET_FEE_ESTIMATE_ONLY",
        "estimate_formula": "shares * (base_fee_bps / 10000) * price * (1 - price)",
        "estimate_only": True,
        "rounding_quantum": "0.00001",
        "rounding_basis": "CONSERVATIVE_CEILING",
        "effective_from": None,
        "effective_basis": "UNAVAILABLE",
        "schedule_sha256": _canonical_sha256(fee_schedule),
        "raw_sha256": _canonical_sha256(raw_fee_rates),
        "reason_codes": [
            "POLYMARKET_CONDITION_FEE_PARAMETERS_NOT_CAPTURED",
            "POLYMARKET_FEE_EFFECTIVE_TIMESTAMP_UNAVAILABLE",
            "POLYMARKET_TOKEN_BASE_FEE_IS_ESTIMATE_ONLY",
        ],
        "schedule": fee_schedule,
        "raw_response": raw_fee_rates,
    }


def _byte_inventory(path: Path) -> dict[str, bytes]:
    return {item.relative_to(path).as_posix(): item.read_bytes() for item in sorted(path.rglob("*")) if item.is_file()}


def _assert_all_persisted_rows_not_live(run_dir: Path) -> None:
    for artifact_path in sorted(run_dir.iterdir()):
        if artifact_path.suffix == ".json":
            rows = [json.loads(artifact_path.read_text(encoding="utf-8"))]
        elif artifact_path.suffix == ".jsonl":
            rows = [json.loads(line) for line in artifact_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            continue
        assert all(row["live_eligible"] is False for row in rows), artifact_path.name


class FixtureNativeClient:
    def __init__(
        self,
        *,
        mismatch_resolution_source: bool = False,
        stale_books: bool = False,
        past_deadline: bool = False,
    ) -> None:
        deadline_delta = timedelta(days=-1 if past_deadline else 1)
        close_time = datetime.now(timezone.utc) + deadline_delta
        self.close_time = close_time.isoformat().replace("+00:00", "Z")
        self.expiration_time = (close_time + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        self.mismatch_resolution_source = mismatch_resolution_source
        self.stale_books = stale_books
        self.metadata_calls: list[tuple[str, str]] = []
        self.book_calls: list[str] = []

    def supporting_evidence(self) -> dict:
        raw_body = b"fixture-reviewed-fee-schedule"
        return {
            "schema_version": 1,
            "kalshi_official_fee_schedule": {
                "status": "REVIEWED",
                "raw_body_base64": base64.b64encode(raw_body).decode("ascii"),
                "raw_body_sha256": hashlib.sha256(raw_body).hexdigest(),
            },
            "live_eligible": False,
        }

    def fetch_metadata(self, candidate: dict, side: str) -> dict:
        venue = candidate[f"venue_{side}"]
        self.metadata_calls.append((candidate["candidate_id"], side))
        resolution_source = "official certified result"
        if self.mismatch_resolution_source and venue == "polymarket":
            resolution_source = "news projection"
        proposition = "Will Candidate X win?"
        authority = "official authority"
        criteria = "candidate x must be certified as winner"
        void_cancel = "void only if permanently cancelled"
        polarity = {"YES": "YES", "NO": "NO"}
        if venue == "kalshi":
            market_payload = {
                "market": {
                    "ticker": "KXTEST-26-X",
                    "event_ticker": "KXTEST-26",
                    "series_ticker": "KXTEST",
                    "title": proposition,
                    "yes_sub_title": "candidate wins",
                    "no_sub_title": "candidate does not win",
                    "rules_primary": criteria,
                    "rules_secondary": None,
                    "close_time": self.close_time,
                    "expiration_time": self.expiration_time,
                    "settlement_timer_seconds": 0,
                    "void_cancel_rules": void_cancel,
                    "market_type": "binary",
                    "notional_value_dollars": "1.00",
                    "fractional_trading_enabled": False,
                    "mve_collection_ticker": None,
                    "mve_selected_legs": [],
                    "status": "active",
                }
            }
            event_metadata_payload = {
                "settlement_sources": [{"name": authority, "url": resolution_source}],
            }
            event_payload = {
                "event": {
                    "event_ticker": "KXTEST-26",
                    "series_ticker": "KXTEST",
                    "mutually_exclusive": False,
                    "settlement_sources": [{"name": authority, "url": resolution_source}],
                }
            }
            series_payload = {
                "series": {
                    "ticker": "KXTEST",
                    "settlement_sources": [{"name": authority, "url": resolution_source}],
                }
            }
            raw_response = {
                "market": market_payload,
                "event_metadata": event_metadata_payload,
                "event": event_payload,
                "series": series_payload,
            }
            requests = [
                _acquisition("/markets/KXTEST-26-X", market_payload),
                _acquisition("/events/KXTEST-26/metadata", event_metadata_payload),
                _acquisition("/events/KXTEST-26", event_payload),
                _acquisition("/series/KXTEST", series_payload),
            ]
            normalized_rules = {
                "rules_primary": criteria,
                "rules_secondary": None,
                "open_time": None,
                "close_time": self.close_time,
                "expiration_time": self.expiration_time,
                "expected_expiration_time": None,
                "latest_expiration_time": None,
                "settlement_timer_seconds": 0,
                "settlement_sources": [{"name": authority, "url": resolution_source}],
                "status": "active",
                "fractional_trading_enabled": False,
                "can_close_early": None,
                "early_close_condition": None,
                "void_cancel_rules": void_cancel,
                "strike_type": None,
                "floor_strike": None,
                "cap_strike": None,
                "functional_strike": None,
                "custom_strike": None,
                "market_type": "binary",
                "notional_value_dollars": "1.00",
                "price_level_structure": None,
                "price_ranges": [],
                "fee_waiver_expiration_time": None,
                "mve_collection_ticker": None,
                "mve_selected_legs": [],
                "event_mutually_exclusive": False,
                "event_collateral_return_type": None,
                "event_last_updated_ts": None,
                "series_contract_url": None,
                "series_contract_terms_url": None,
                "series_last_updated_ts": None,
                "outcome_polarity": polarity,
            }
            native_market_id = "KXTEST-26-X"
            native_event_id = "KXTEST-26"
            native_series_id = "KXTEST"
            native_condition_id = None
            native_outcome_ids = {"YES": native_market_id, "NO": native_market_id}
            native_outcome_labels = {"YES": "candidate wins", "NO": "candidate does not win"}
            event_mutually_exclusive = False
            negative_risk = None
        else:
            selected_market = {
                "id": "12345",
                "slug": "candidate-x-win",
                "conditionId": "condition-12345",
                "question": proposition,
                "description": criteria,
                "resolutionSource": resolution_source,
                "resolvedBy": authority,
                "outcomes": ["Yes", "No"],
                "clobTokenIds": ["polymarket_yes", "polymarket_no"],
                "endDate": self.close_time,
                "umaEndDate": self.expiration_time,
                "negRisk": False,
                "active": True,
                "closed": False,
                "archived": False,
                "acceptingOrders": True,
                "voidCancelRules": void_cancel,
                "settlementDelaySeconds": 0,
                "feesEnabled": True,
                "events": [{"id": "event-12345"}],
            }
            markets_payload = [selected_market]
            raw_response = {"markets": markets_payload, "selected_market": selected_market}
            requests = [
                _acquisition("/markets", markets_payload, params={"slug": "candidate-x-win"}),
            ]
            normalized_rules = {
                "question": proposition,
                "description": criteria,
                "resolution_source": resolution_source,
                "resolved_by": authority,
                "start_date": None,
                "end_date": self.close_time,
                "event_start_time": None,
                "uma_end_date": self.expiration_time,
                "uma_resolution_status": None,
                "neg_risk": False,
                "active": True,
                "closed": False,
                "archived": False,
                "accepting_orders": True,
                "void_cancel_rules": void_cancel,
                "settlement_delay_seconds": 0.0,
                "fees_enabled": True,
                "created_at": None,
                "updated_at": None,
                "parent_events": [{"id": "event-12345"}],
                "outcome_polarity": polarity,
            }
            native_market_id = "12345"
            native_event_id = "event-12345"
            native_series_id = None
            native_condition_id = "condition-12345"
            native_outcome_ids = {"YES": "polymarket_yes", "NO": "polymarket_no"}
            native_outcome_labels = {"YES": "Yes", "NO": "No"}
            event_mutually_exclusive = None
            negative_risk = False
        result = {
            "schema_version": 1,
            "candidate_id": candidate["candidate_id"],
            "side": side,
            "venue": venue,
            "pmxt_market_id": candidate[f"pmxt_market_id_{side}"],
            "native_market_id": native_market_id,
            "native_event_id": native_event_id,
            "native_series_id": native_series_id,
            "native_condition_id": native_condition_id,
            "native_outcome_ids": native_outcome_ids,
            "native_outcome_labels": native_outcome_labels,
            "outcome_polarity": polarity,
            "market_status": "active",
            "book_eligible": True,
            "minimum_order_size": 1.0,
            "size_increment": 1.0,
            "proposition": proposition,
            "close_time": self.close_time,
            "expiration_time": self.expiration_time,
            "settlement_authority": authority,
            "resolution_source": resolution_source,
            "resolution_criteria": criteria,
            "void_cancel": void_cancel,
            "material_edge_cases": {},
            "settlement_delay_seconds": 0,
            "market_type": "binary",
            "mve_collection_ticker": None,
            "mve_selected_legs": [],
            "event_mutually_exclusive": event_mutually_exclusive,
            "negative_risk": negative_risk,
            "raw_sha256": _canonical_sha256(raw_response),
            "rule_hash": _canonical_sha256(normalized_rules),
            "requested_at": utc_now_iso(),
            "received_at": utc_now_iso(),
            "requests": requests,
            "raw_response": raw_response,
            "normalized_rules": normalized_rules,
            "status": "RESOLVED",
            "live_eligible": False,
        }
        return result

    def fetch_book(self, metadata: dict, *, depth: int) -> dict:
        assert depth == 25
        venue = metadata["venue"]
        self.book_calls.append(venue)
        as_of = "2020-01-01T00:00:00Z" if self.stale_books else utc_now_iso()
        if venue == "kalshi":
            orderbook_payload = {
                "orderbook_fp": {
                    "yes_dollars": [[0.20, 20.0]],
                    "no_dollars": [[0.60, 20.0]],
                }
            }
            raw_response = {"orderbook": orderbook_payload}
            sides = {
                "YES": {
                    "bids": [{"price": 0.20, "size": 20.0}],
                    "asks": [{"price": 0.40, "size": 20.0}],
                },
                "NO": {
                    "bids": [{"price": 0.60, "size": 20.0}],
                    "asks": [{"price": 0.80, "size": 20.0}],
                },
            }
            requests = [_acquisition("/markets/KXTEST-26-X/orderbook", orderbook_payload)]
        else:
            raw_books = {
                "YES": {
                    "asset_id": "polymarket_yes",
                    "market": "condition-12345",
                    "timestamp": as_of,
                    "bids": [{"price": 0.78, "size": 20.0}],
                    "asks": [{"price": 0.80, "size": 20.0}],
                },
                "NO": {
                    "asset_id": "polymarket_no",
                    "market": "condition-12345",
                    "timestamp": as_of,
                    "bids": [{"price": 0.40, "size": 20.0}],
                    "asks": [{"price": 0.42, "size": 20.0}],
                },
            }
            raw_fee_rates = {"YES": {"base_fee": 100}, "NO": {"base_fee": 100}}
            raw_response = {"books": raw_books, "fee_rates": raw_fee_rates}
            requests = []
            for outcome in ("YES", "NO"):
                token_id = metadata["native_outcome_ids"][outcome]
                params = {"token_id": token_id}
                requests.append(_acquisition("/fee-rate", raw_fee_rates[outcome], params=params))
                requests.append(_acquisition("/book", raw_books[outcome], params=params))
            sides = {
                outcome: {
                    "bids": [dict(level) for level in raw_books[outcome]["bids"]],
                    "asks": [dict(level) for level in raw_books[outcome]["asks"]],
                }
                for outcome in ("YES", "NO")
            }
        return {
            "schema_version": 1,
            "candidate_id": metadata["candidate_id"],
            "venue": venue,
            "native_market_id": metadata["native_market_id"],
            "market_status": metadata["market_status"],
            "book_eligible": metadata["book_eligible"],
            "minimum_order_size": metadata["minimum_order_size"],
            "size_increment": metadata["size_increment"],
            "request_started_at": as_of,
            "received_at": as_of,
            "request_monotonic_ns": 2_000_000_000,
            "response_monotonic_ns": 2_005_000_000,
            "rtt_ms": 5.0,
            "source_timestamp": as_of,
            "as_of": as_of,
            "freshness_basis": "VENUE_SOURCE_TIMESTAMP",
            "normalized_depth_limit": depth,
            "normalized_depth_truncated": False,
            "native_outcome_ids": metadata["native_outcome_ids"],
            "raw_sha256": _canonical_sha256(raw_response),
            "fee_evidence": _fee_evidence(venue, as_of),
            "sides": sides,
            "requests": requests,
            "raw_response": raw_response,
            "live_eligible": False,
        }


def _config() -> MonitorConfig:
    return MonitorConfig(
        limit=25,
        min_confidence=0.80,
        net_residual_threshold=0.05,
        requested_size=10.0,
        max_book_age_seconds=60.0,
        max_cross_venue_skew_seconds=3.0,
        annual_capital_rate=0.0,
        native_book_depth=25,
    )


def test_config_declares_venue_native_taker_fees_without_manual_rates() -> None:
    config = _config()

    assert config.as_dict()["fee_authority"] == "VENUE_NATIVE_CAPTURE_ONLY"
    assert config.as_dict()["fee_liquidity_role"] == "TAKER"
    assert "venue_fee_rates" not in config.as_dict()


def test_missing_key_stops_before_network_and_creates_no_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PMXT_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    output_dir = tmp_path / "must-not-exist"
    monitor = PmxtReadOnlyMonitor(
        output_dir=output_dir,
        config=_config(),
        pmxt_transport=_pmxt_transport(requests),
        native_client=FixtureNativeClient(),
    )

    with pytest.raises(RuntimeError, match="no artifacts were created"):
        monitor.sync_once()

    assert requests == []
    assert not output_dir.exists()


def test_pmxt_connect_error_is_attempted_once_and_persists_only_sanitized_failure(
    tmp_path: Path,
) -> None:
    attempts: list[httpx.Request] = []
    api_key = "fixture_secret_must_not_be_persisted"
    transport_detail = "fixture-sensitive-upstream-diagnostic"

    def fail_connect(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectError(transport_detail, request=request)

    with pytest.raises(RuntimeError, match="PMXT sync failed; evidence manifest:"):
        PmxtReadOnlyMonitor(
            api_key=api_key,
            output_dir=tmp_path,
            config=_config(),
            pmxt_transport=httpx.MockTransport(fail_connect),
            native_client=FixtureNativeClient(),
        ).sync_once()

    assert len(attempts) == 1
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    raw_pmxt = json.loads((run_dir / "raw_pmxt.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PMXT_SYNC_FAILED"
    assert manifest["live_eligible"] is False
    assert manifest["no_order_actions"] is True
    assert raw_pmxt["status"] == "ERROR"
    assert raw_pmxt["error"] == {
        "type": "PmxtRouterError",
        "message": "PMXT Router request failed before a response was received",
    }
    persisted = b"".join(path.read_bytes() for path in sorted(run_dir.iterdir()) if path.is_file())
    assert api_key.encode() not in persisted
    assert transport_detail.encode() not in persisted
    for name in (
        "candidates.jsonl",
        "raw_native_metadata.jsonl",
        "semantic_decisions.jsonl",
        "rejections.jsonl",
        "native_books.jsonl",
        "calculations.jsonl",
        "alerts.jsonl",
    ):
        assert (run_dir / name).read_bytes() == b""


def test_pmxt_429_is_persisted_once_with_exact_bounded_evidence_and_no_native_reads(
    tmp_path: Path,
) -> None:
    attempts: list[httpx.Request] = []
    api_key = "fixture_secret_must_not_be_persisted"
    raw_body = (
        b'{"error":"rate_limit_exceeded","plan":"free",'
        b'"limit":60,"used":60,"window":"1 minute"}'
    )

    def rate_limited(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(
            429,
            headers={
                "Content-Type": "application/json",
                "Retry-After": "17",
                "X-Request-ID": "fixture-429-id",
                "Set-Cookie": f"session={api_key}",
            },
            stream=httpx.ByteStream(raw_body),
        )

    native = FixtureNativeClient()
    with pytest.raises(RuntimeError, match="PMXT sync failed; evidence manifest:"):
        PmxtReadOnlyMonitor(
            api_key=api_key,
            output_dir=tmp_path,
            config=_config(),
            pmxt_transport=httpx.MockTransport(rate_limited),
            native_client=native,
        ).sync_once()

    assert len(attempts) == 1
    assert native.metadata_calls == []
    assert native.book_calls == []
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    manifest_bytes = (run_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    raw_pmxt_bytes = (run_dir / "raw_pmxt.json").read_bytes()
    raw_pmxt = json.loads(raw_pmxt_bytes)
    assert manifest["status"] == "PMXT_SYNC_FAILED"
    assert manifest["counts"]["pmxt_network_requests"] == 1
    assert manifest["counts"]["pmxt_retry_attempts"] == 0
    assert manifest["live_eligible"] is False
    assert manifest["no_order_actions"] is True
    assert raw_pmxt["pmxt_network_requests"] == 1
    assert raw_pmxt["pmxt_retry_attempts"] == 0
    assert raw_pmxt["router_error"]["reason_code"] == "PMXT_HTTP_RATE_LIMITED"
    response = raw_pmxt["router_error"]["evidence"]["response"]
    assert response["status_code"] == 429
    assert base64.b64decode(response["body"]["raw_body_base64"], validate=True) == raw_body
    assert response["body"]["raw_body_sha256"] == hashlib.sha256(raw_body).hexdigest()
    assert response["rate_limit"]["classification"] == "PER_MINUTE"
    assert response["rate_limit"]["retries_performed"] == 0
    assert response["rate_limit"]["retry_after"] == {
        "kind": "DELTA_SECONDS",
        "raw": "17",
        "seconds": 17,
    }
    assert manifest["artifacts"]["raw_pmxt"]["sha256"] == hashlib.sha256(raw_pmxt_bytes).hexdigest()
    persisted = b"".join(path.read_bytes() for path in sorted(run_dir.iterdir()) if path.is_file())
    assert api_key.encode() not in persisted
    for name in (
        "candidates.jsonl",
        "raw_native_metadata.jsonl",
        "semantic_decisions.jsonl",
        "rejections.jsonl",
        "native_books.jsonl",
        "calculations.jsonl",
        "alerts.jsonl",
    ):
        assert (run_dir / name).read_bytes() == b""


def test_pmxt_success_response_reflecting_key_is_withheld_and_never_persisted_as_payload(
    tmp_path: Path,
) -> None:
    attempts = 0
    api_key = "fixture_success_echo_must_not_persist"
    raw_body = json.dumps({"clusters": [], "echo": api_key}).encode()

    def reflected_key(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, stream=httpx.ByteStream(raw_body))

    native = FixtureNativeClient()
    with pytest.raises(RuntimeError, match="PMXT sync failed; evidence manifest:"):
        PmxtReadOnlyMonitor(
            api_key=api_key,
            output_dir=tmp_path,
            config=_config(),
            pmxt_transport=httpx.MockTransport(reflected_key),
            native_client=native,
        ).sync_once()

    assert attempts == 1
    assert native.metadata_calls == []
    run_dir = next((tmp_path / "runs").iterdir())
    raw_pmxt = json.loads((run_dir / "raw_pmxt.json").read_text(encoding="utf-8"))
    assert "payload" not in raw_pmxt
    assert raw_pmxt["router_error"]["reason_code"] == "PMXT_RESPONSE_CREDENTIAL_ECHO"
    assert (
        raw_pmxt["router_error"]["evidence"]["response"]["body"]["capture_status"]
        == "WITHHELD_API_KEY_ECHO"
    )
    persisted = b"".join(path.read_bytes() for path in run_dir.iterdir() if path.is_file())
    assert api_key.encode() not in persisted


def test_deeply_nested_429_still_persists_a_bounded_failure_artifact(tmp_path: Path) -> None:
    attempts = 0
    raw_body = (b"[" * 1_100) + b"0" + (b"]" * 1_100)

    def deeply_nested(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, stream=httpx.ByteStream(raw_body))

    with pytest.raises(RuntimeError, match="PMXT sync failed; evidence manifest:"):
        PmxtReadOnlyMonitor(
            api_key="fixture_key",
            output_dir=tmp_path,
            config=_config(),
            pmxt_transport=httpx.MockTransport(deeply_nested),
            native_client=FixtureNativeClient(),
        ).sync_once()

    assert attempts == 1
    run_dir = next((tmp_path / "runs").iterdir())
    raw_pmxt = json.loads((run_dir / "raw_pmxt.json").read_text(encoding="utf-8"))
    router_error = raw_pmxt["router_error"]
    assert router_error["reason_code"] == "PMXT_RESPONSE_CREDENTIAL_SCAN_INDETERMINATE"
    response = router_error["evidence"]["response"]
    assert response["body"]["capture_status"] == "WITHHELD_CREDENTIAL_SCAN_INDETERMINATE"
    assert response["body"]["credential_scan_status"] == "UNKNOWN_PARSER_LIMIT"
    assert "raw_body_base64" not in response["body"]
    assert response["rate_limit"]["body_parse_status"] == "PARSER_LIMIT_EXCEEDED"
    assert raw_pmxt["pmxt_network_requests"] == 1
    assert raw_pmxt["pmxt_retry_attempts"] == 0


def test_one_identity_sync_fails_closed_on_estimate_only_fees_with_immutable_evidence(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    native = FixtureNativeClient()
    outcome = PmxtReadOnlyMonitor(
        api_key="fixture_key_not_persisted",
        output_dir=tmp_path,
        config=_config(),
        pmxt_transport=_pmxt_transport(requests),
        native_client=native,
    ).sync_once()

    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/v0/matched-market-clusters"
    assert request.url.params["relation"] == "identity"
    assert request.url.params["limit"] == "25"
    assert request.url.params["minConfidence"] == "0.8"
    assert request.url.params["venues"] == "kalshi,polymarket"
    assert outcome.status == "FEE_EVIDENCE_UNAVAILABLE"
    assert outcome.cluster_count == outcome.candidate_count == outcome.verified_count == 1
    assert outcome.alert_count == 0
    assert outcome.rejected_count == outcome.needs_review_count == 0
    assert len(native.metadata_calls) == len(native.book_calls) == 2

    manifest = json.loads(outcome.artifacts.manifest_path.read_text(encoding="utf-8"))
    raw_pmxt = json.loads(outcome.artifacts.artifact_paths["raw_pmxt"].read_text(encoding="utf-8"))
    calculation = json.loads(outcome.artifacts.artifact_paths["calculations"].read_text(encoding="utf-8"))
    native_books = [
        json.loads(row)
        for row in outcome.artifacts.artifact_paths["native_books"].read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["live_eligible"] is False
    assert manifest["no_order_actions"] is True
    assert manifest["status"] == "FEE_EVIDENCE_UNAVAILABLE"
    assert manifest["counts"]["fee_evidence_unavailable"] == 1
    for artifact_path in outcome.artifacts.run_dir.iterdir():
        assert "fixture_key_not_persisted" not in artifact_path.read_text(encoding="utf-8")
    assert "fixture_key_not_persisted" not in json.dumps(raw_pmxt)
    assert outcome.artifacts.artifact_paths["alerts"].read_bytes() == b""
    assert calculation["status"] == "FEE_EVIDENCE_UNAVAILABLE"
    assert calculation["alert"] is calculation["live_eligible"] is False
    assert "BOOK_B_NATIVE_FEE_EVIDENCE_NOT_VALID" in calculation["reasons"]
    assert {book["fee_evidence"]["model"] for book in native_books} == {
        "KALSHI_QUADRATIC_TAKER",
        "POLYMARKET_FEE_ESTIMATE_ONLY",
    }
    polymarket_fees = next(book["fee_evidence"] for book in native_books if book["venue"] == "polymarket")
    assert polymarket_fees["status"] == "FAIL_CLOSED"
    assert polymarket_fees["reason_codes"] == [
        "POLYMARKET_CONDITION_FEE_PARAMETERS_NOT_CAPTURED",
        "POLYMARKET_FEE_EFFECTIVE_TIMESTAMP_UNAVAILABLE",
        "POLYMARKET_TOKEN_BASE_FEE_IS_ESTIMATE_ONLY",
    ]
    _assert_all_persisted_rows_not_live(outcome.artifacts.run_dir)


def test_limit_one_still_allows_one_two_market_candidate(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    config = replace(_config(), limit=1)
    outcome = PmxtReadOnlyMonitor(
        api_key="fixture_key",
        output_dir=tmp_path,
        config=config,
        pmxt_transport=_pmxt_transport(requests),
        native_client=FixtureNativeClient(),
    ).sync_once()

    assert requests[0].url.params["limit"] == "1"
    assert outcome.candidate_count == outcome.verified_count == 1
    assert outcome.bounded_rejection_count == 0


def test_semantic_rejection_produces_no_books_and_no_verified_candidate_status(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    native = FixtureNativeClient(mismatch_resolution_source=True)
    outcome = PmxtReadOnlyMonitor(
        api_key="fixture_key",
        output_dir=tmp_path,
        config=_config(),
        pmxt_transport=_pmxt_transport(requests),
        native_client=native,
    ).sync_once()

    assert outcome.status == "NO_VERIFIED_CANDIDATES"
    assert outcome.verified_count == outcome.alert_count == 0
    assert outcome.rejected_count == 1
    assert native.book_calls == []
    assert outcome.artifacts.artifact_paths["native_books"].read_bytes() == b""
    decision = json.loads(outcome.artifacts.artifact_paths["semantic_decisions"].read_text(encoding="utf-8"))
    assert decision["status"] == "REJECTED"
    assert "RESOLUTION_SOURCE_MISMATCH" in decision["reason_codes"]
    assert decision["live_eligible"] is False


def test_needs_review_skips_books_calculations_and_alerts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    native = FixtureNativeClient()

    def needs_review(candidate: dict, _left: dict, _right: dict) -> dict:
        return {
            "schema_version": 1,
            "candidate_id": candidate["candidate_id"],
            "status": "NEEDS_REVIEW",
            "reason_codes": ["FIXTURE_EVIDENCE_GAP"],
            "evidence": {"fixture": "insufficient semantic evidence"},
            "live_eligible": False,
        }

    def forbid_shadow(*_args: object, **_kwargs: object) -> dict:
        raise AssertionError("NEEDS_REVIEW candidate reached shadow calculation")

    monkeypatch.setattr(monitor_module, "verify_semantics", needs_review)
    monkeypatch.setattr(monitor_module, "calculate_shadow", forbid_shadow)
    outcome = PmxtReadOnlyMonitor(
        api_key="fixture_key",
        output_dir=tmp_path,
        config=_config(),
        pmxt_transport=_pmxt_transport(requests),
        native_client=native,
    ).sync_once()

    assert len(requests) == 1
    assert len(native.metadata_calls) == 2
    assert native.book_calls == []
    assert outcome.status == "NO_VERIFIED_CANDIDATES"
    assert outcome.verified_count == outcome.rejected_count == outcome.alert_count == 0
    assert outcome.needs_review_count == 1
    manifest = json.loads(outcome.artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "NO_VERIFIED_CANDIDATES"
    assert manifest["counts"]["needs_review_candidates"] == 1
    assert manifest["counts"]["native_book_attempts"] == 0
    assert manifest["counts"]["shadow_calculations"] == 0
    assert manifest["counts"]["alerts"] == 0
    for name in ("native_books", "calculations", "alerts"):
        assert outcome.artifacts.artifact_paths[name].read_bytes() == b""
    decision = json.loads(outcome.artifacts.artifact_paths["semantic_decisions"].read_text(encoding="utf-8"))
    rejection = json.loads(outcome.artifacts.artifact_paths["rejections"].read_text(encoding="utf-8"))
    assert decision["status"] == rejection["status"] == "NEEDS_REVIEW"
    assert decision["reason_codes"] == rejection["reason_codes"] == ["FIXTURE_EVIDENCE_GAP"]
    assert decision["live_eligible"] is rejection["live_eligible"] is False


def test_stale_native_books_produce_no_executable_shadow_edge_status(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    native = FixtureNativeClient(stale_books=True)
    outcome = PmxtReadOnlyMonitor(
        api_key="fixture_key",
        output_dir=tmp_path,
        config=_config(),
        pmxt_transport=_pmxt_transport(requests),
        native_client=native,
    ).sync_once()

    assert outcome.status == "FEE_EVIDENCE_UNAVAILABLE"
    assert outcome.verified_count == 1
    assert outcome.alert_count == 0
    calculation = json.loads(outcome.artifacts.artifact_paths["calculations"].read_text(encoding="utf-8"))
    assert calculation["status"] == "FEE_EVIDENCE_UNAVAILABLE"
    assert any(reason.endswith("STALE_SOURCE_TIMESTAMP") for reason in calculation["reasons"])
    assert calculation["live_eligible"] is False


def test_active_fixture_with_past_settlement_horizon_fails_closed(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    outcome = PmxtReadOnlyMonitor(
        api_key="fixture_key",
        output_dir=tmp_path,
        config=_config(),
        pmxt_transport=_pmxt_transport(requests),
        native_client=FixtureNativeClient(past_deadline=True),
    ).sync_once()

    calculation = json.loads(outcome.artifacts.artifact_paths["calculations"].read_text(encoding="utf-8"))
    assert outcome.status == "NO_EXECUTABLE_SHADOW_EDGE"
    assert calculation["status"] == "NO_EXECUTABLE_SHADOW_EDGE"
    assert "MISSING_SETTLEMENT_OR_CAPITAL_LOCK_DURATION" in calculation["reasons"]
    assert calculation["alert"] is calculation["live_eligible"] is False


def test_server_overdelivery_is_hard_capped_before_native_requests(tmp_path: Path) -> None:
    clusters = []
    for index in range(26):
        cluster = deepcopy(_pmxt_cluster())
        cluster["clusterId"] = f"mcl_bound_{index:02d}"
        for market in cluster["markets"]:
            old_id = market["marketId"]
            new_id = f"{old_id}_{index:02d}"
            market["marketId"] = new_id
            for edge in cluster["rawMatches"]:
                if edge["marketAId"] == old_id:
                    edge["marketAId"] = new_id
                if edge["marketBId"] == old_id:
                    edge["marketBId"] = new_id
        clusters.append(cluster)

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.dumps({"clusters": clusters}).encode()
        return httpx.Response(200, stream=httpx.ByteStream(body))

    native = FixtureNativeClient()
    with pytest.raises(RuntimeError, match="PMXT sync failed; evidence manifest:"):
        PmxtReadOnlyMonitor(
            api_key="fixture_key",
            output_dir=tmp_path,
            config=_config(),
            pmxt_transport=httpx.MockTransport(handler),
            native_client=native,
        ).sync_once()

    assert len(requests) == 1
    assert native.metadata_calls == []
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    raw_pmxt = json.loads((run_dir / "raw_pmxt.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PMXT_SYNC_FAILED"
    assert manifest["live_eligible"] is False
    assert manifest["no_order_actions"] is True
    assert len(raw_pmxt["payload"]["clusters"]) == 26
    assert (run_dir / "candidates.jsonl").read_bytes() == b""


def test_native_only_continuation_needs_no_key_and_never_constructs_pmxt_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_requests: list[httpx.Request] = []
    source = PmxtReadOnlyMonitor(
        api_key="source_fixture_key",
        output_dir=tmp_path,
        config=_config(),
        pmxt_transport=_pmxt_transport(initial_requests),
        native_client=FixtureNativeClient(),
    ).sync_once()
    source_before = _byte_inventory(source.artifacts.run_dir)
    source_manifest = json.loads(source.artifacts.manifest_path.read_text(encoding="utf-8"))
    assert source_manifest["counts"]["supporting_evidence"] == 1
    assert source_manifest["artifacts"]["supporting_evidence"]["path"] == "supporting_evidence.json"

    class ForbiddenPmxtClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("native-only continuation constructed a PMXT client")

    forbidden_requests: list[httpx.Request] = []

    def forbidden_handler(request: httpx.Request) -> httpx.Response:
        forbidden_requests.append(request)
        raise AssertionError("native-only continuation made a PMXT request")

    monkeypatch.delenv("PMXT_API_KEY", raising=False)
    monkeypatch.setattr(monitor_module, "PmxtRouterClient", ForbiddenPmxtClient)
    continuation = PmxtReadOnlyMonitor(
        output_dir=tmp_path,
        config=_config(),
        pmxt_transport=httpx.MockTransport(forbidden_handler),
        native_client=FixtureNativeClient(),
    ).continue_once(source.run_id)

    assert len(initial_requests) == 1
    assert forbidden_requests == []
    assert continuation.run_id != source.run_id
    assert continuation.artifacts.run_dir != source.artifacts.run_dir
    assert continuation.artifacts.run_dir.exists()
    assert _byte_inventory(source.artifacts.run_dir) == source_before

    manifest = json.loads(continuation.artifacts.manifest_path.read_text(encoding="utf-8"))
    raw_pmxt = json.loads(continuation.artifacts.artifact_paths["raw_pmxt"].read_text(encoding="utf-8"))
    assert manifest["counts"]["pmxt_network_requests"] == 0
    assert manifest["provenance"]["pmxt_network_requests"] == 0
    assert manifest["provenance"]["source_run_id"] == source.run_id
    assert (
        manifest["provenance"]["source_artifact_sha256"]["supporting_evidence"]
        == (source_manifest["artifacts"]["supporting_evidence"]["sha256"])
    )
    assert raw_pmxt["pmxt_network_requests"] == 0
    assert raw_pmxt["mode"] == "NATIVE_ONLY_CONTINUATION"
    assert raw_pmxt["source_run_id"] == source.run_id
    assert manifest["live_eligible"] is raw_pmxt["live_eligible"] is False
