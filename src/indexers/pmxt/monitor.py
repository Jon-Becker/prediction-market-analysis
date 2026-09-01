"""Bounded read-only PMXT discovery, native verification, and shadow alerts."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from src.indexers.pmxt.artifacts import (
    MonitorRunArtifacts,
    SourceRunValidationError,
    load_verified_monitor_source,
    persist_monitor_run,
)
from src.indexers.pmxt.candidates import new_snapshot_id, normalize_clusters, utc_now_iso
from src.indexers.pmxt.client import PmxtRouterClient, PmxtRouterError, extract_clusters
from src.indexers.pmxt.models import PmxtQuery
from src.indexers.pmxt.native import NativeEvidenceClient, NativeEvidenceError, canonical_json_sha256
from src.indexers.pmxt.semantic import NEEDS_REVIEW, REJECTED, VERIFIED_EQUIVALENT, verify_semantics
from src.indexers.pmxt.shadow import DEFAULT_NET_RESIDUAL_THRESHOLD, calculate_shadow

DEFAULT_OUTPUT_DIR = Path("data/pmxt")


@dataclass(frozen=True)
class MonitorConfig:
    """All bounded research assumptions; none enables live execution."""

    limit: int = 25
    min_confidence: float = 0.80
    net_residual_threshold: float = DEFAULT_NET_RESIDUAL_THRESHOLD
    requested_size: float = 10.0
    max_book_age_seconds: float = 15.0
    max_cross_venue_skew_seconds: float = 3.0
    annual_capital_rate: float = 0.05
    explicit_slippage_buffer_per_unit: float = 0.0
    timestamp_skew_buffer_per_unit: float = 0.0
    settlement_divergence_buffer_per_unit: float = 0.0
    collateral_basis_buffer_per_unit: float = 0.0
    rebalancing_withdrawal_allowance_per_unit: float = 0.0
    native_book_depth: int = 100
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        numeric_values = (
            ("min_confidence", self.min_confidence),
            ("net_residual_threshold", self.net_residual_threshold),
            ("requested_size", self.requested_size),
            ("max_book_age_seconds", self.max_book_age_seconds),
            ("max_cross_venue_skew_seconds", self.max_cross_venue_skew_seconds),
            ("annual_capital_rate", self.annual_capital_rate),
            ("explicit_slippage_buffer_per_unit", self.explicit_slippage_buffer_per_unit),
            ("timestamp_skew_buffer_per_unit", self.timestamp_skew_buffer_per_unit),
            ("settlement_divergence_buffer_per_unit", self.settlement_divergence_buffer_per_unit),
            ("collateral_basis_buffer_per_unit", self.collateral_basis_buffer_per_unit),
            (
                "rebalancing_withdrawal_allowance_per_unit",
                self.rebalancing_withdrawal_allowance_per_unit,
            ),
            ("timeout_seconds", self.timeout_seconds),
        )
        for name, value in numeric_values:
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("monitor limit must be an integer")
        if isinstance(self.native_book_depth, bool) or not isinstance(self.native_book_depth, int):
            raise ValueError("native_book_depth must be an integer")
        if not 1 <= self.limit <= 25:
            raise ValueError("monitor limit must be between 1 and 25")
        if not 0.80 <= self.min_confidence <= 1:
            raise ValueError("monitor min_confidence must be between 0.80 and 1")
        if self.net_residual_threshold < 0:
            raise ValueError("net_residual_threshold must be non-negative")
        if self.requested_size <= 0:
            raise ValueError("requested_size must be positive")
        if self.max_book_age_seconds < 0 or self.max_cross_venue_skew_seconds < 0:
            raise ValueError("book age and skew limits must be non-negative")
        if self.annual_capital_rate < 0:
            raise ValueError("annual_capital_rate must be non-negative")
        if any(
            value < 0
            for value in (
                self.explicit_slippage_buffer_per_unit,
                self.timestamp_skew_buffer_per_unit,
                self.settlement_divergence_buffer_per_unit,
                self.collateral_basis_buffer_per_unit,
                self.rebalancing_withdrawal_allowance_per_unit,
            )
        ):
            raise ValueError("per-unit shadow cost buffers must be non-negative")
        if not 1 <= self.native_book_depth <= 100:
            raise ValueError("native_book_depth must be between 1 and 100")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def query(self) -> PmxtQuery:
        """Return the fixed identity-only Kalshi/Polymarket discovery query."""

        return PmxtQuery(
            relation="identity",
            min_confidence=self.min_confidence,
            min_venues=2,
            limit=self.limit,
            offset=0,
            sort="volume",
            venues=("kalshi", "polymarket"),
            include_raw_matches=True,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "discovery": self.query().as_dict(),
            "net_residual_threshold": self.net_residual_threshold,
            "requested_size": self.requested_size,
            "max_book_age_seconds": self.max_book_age_seconds,
            "max_cross_venue_skew_seconds": self.max_cross_venue_skew_seconds,
            "fee_authority": "VENUE_NATIVE_CAPTURE_ONLY",
            "fee_liquidity_role": "TAKER",
            "annual_capital_rate": self.annual_capital_rate,
            "explicit_slippage_buffer_per_unit": self.explicit_slippage_buffer_per_unit,
            "timestamp_skew_buffer_per_unit": self.timestamp_skew_buffer_per_unit,
            "settlement_divergence_buffer_per_unit": self.settlement_divergence_buffer_per_unit,
            "collateral_basis_buffer_per_unit": self.collateral_basis_buffer_per_unit,
            "rebalancing_withdrawal_allowance_per_unit": self.rebalancing_withdrawal_allowance_per_unit,
            "native_book_depth": self.native_book_depth,
            "timeout_seconds": self.timeout_seconds,
            "identity_only": True,
            "alerts_only": True,
            "live_eligible": False,
        }


@dataclass(frozen=True)
class MonitorOutcome:
    run_id: str
    status: str
    artifacts: MonitorRunArtifacts
    cluster_count: int
    candidate_count: int
    verified_count: int
    rejected_count: int
    needs_review_count: int
    pmxt_rejection_count: int
    bounded_rejection_count: int
    total_rejection_rows: int
    alert_count: int


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _capital_lock_days(native_records: list[Mapping[str, Any]], evaluated_at: str) -> float | None:
    evaluated = _parse_utc(evaluated_at)
    if evaluated is None:
        return None
    locked_until: list[datetime] = []
    for record in native_records:
        close_time = _parse_utc(record.get("close_time"))
        delay = record.get("settlement_delay_seconds")
        if close_time is None or isinstance(delay, bool):
            return None
        try:
            delay_seconds = float(delay)
        except (TypeError, ValueError):
            return None
        if delay_seconds < 0:
            return None
        venue_locked_until = close_time + timedelta(seconds=delay_seconds)
        if venue_locked_until <= evaluated:
            return None
        locked_until.append(venue_locked_until)
    if len(locked_until) != 2:
        return None
    return (max(locked_until) - evaluated).total_seconds() / 86_400


def _captured_error_evidence(error: NativeEvidenceError) -> tuple[list[dict[str, Any]], Any, str | None]:
    evidence = error.evidence
    requests: list[dict[str, Any]] = []
    completed = evidence.get("completed_requests")
    if isinstance(completed, Sequence) and not isinstance(completed, (str, bytes)):
        requests.extend(dict(item) for item in completed if isinstance(item, Mapping))
    nested_request = evidence.get("request")
    if isinstance(nested_request, Mapping):
        requests.append(dict(nested_request))
    if evidence.get("method") == "GET":
        requests.append(
            {
                key: value
                for key, value in evidence.items()
                if key
                not in {
                    "completed_requests",
                    "partial_raw_response",
                    "raw_response",
                    "response",
                }
            }
        )
    unique_requests: list[dict[str, Any]] = []
    for request in requests:
        if request not in unique_requests:
            unique_requests.append(request)

    raw_response: Any = evidence.get("partial_raw_response")
    if raw_response is None:
        raw_response = evidence.get("raw_response")
    if raw_response is None and "response" in evidence:
        raw_response = {"failed_response": evidence.get("response")}
    raw_sha256 = canonical_json_sha256(raw_response) if raw_response is not None else None
    return unique_requests, raw_response, raw_sha256


def _native_error_record(
    candidate: Mapping[str, Any],
    side: str,
    error: NativeEvidenceError,
) -> dict[str, Any]:
    venue = candidate.get(f"venue_{side}")
    requests, raw_response, raw_sha256 = _captured_error_evidence(error)
    return {
        "schema_version": 1,
        "candidate_id": candidate.get("candidate_id"),
        "side": side,
        "venue": venue,
        "pmxt_market_id": candidate.get(f"pmxt_market_id_{side}"),
        "native_market_id": None,
        "raw_sha256": raw_sha256,
        "rule_hash": None,
        "requests": requests,
        "raw_response": raw_response,
        "proposition": None,
        "outcome_polarity": None,
        "close_time": None,
        "settlement_authority": None,
        "resolution_source": None,
        "resolution_criteria": None,
        "void_cancel": None,
        "material_edge_cases": None,
        "settlement_delay_seconds": None,
        "status": "ERROR",
        "reason_code": error.reason_code,
        "error": str(error),
        "error_evidence": error.evidence,
        "live_eligible": False,
    }


def _bounded_cluster_shapes(
    clusters: list[dict[str, Any]],
    *,
    per_cluster_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject pathological cluster fan-out before pair normalization."""

    bounded: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for cluster in clusters:
        markets = cluster.get("markets", [])
        raw_matches = cluster.get("rawMatches", cluster.get("raw_matches", []))
        market_count = len(markets) if isinstance(markets, Sequence) and not isinstance(markets, (str, bytes)) else 0
        raw_match_count = (
            len(raw_matches) if isinstance(raw_matches, Sequence) and not isinstance(raw_matches, (str, bytes)) else 0
        )
        if market_count > per_cluster_limit or raw_match_count > per_cluster_limit:
            rejected.append(
                {
                    "stage": "pmxt_bounds",
                    "cluster_id": cluster.get("clusterId", cluster.get("cluster_id")),
                    "reason": "cluster_shape_exceeds_monitor_bound",
                    "market_count": market_count,
                    "raw_match_count": raw_match_count,
                    "per_cluster_limit": per_cluster_limit,
                    "live_eligible": False,
                }
            )
            continue
        bounded.append(cluster)
    return bounded, rejected


def _book_error_record(metadata: Mapping[str, Any], error: NativeEvidenceError) -> dict[str, Any]:
    requests, raw_response, raw_sha256 = _captured_error_evidence(error)
    return {
        "schema_version": 1,
        "candidate_id": metadata.get("candidate_id"),
        "venue": metadata.get("venue"),
        "native_market_id": metadata.get("native_market_id"),
        "request_started_at": None,
        "received_at": None,
        "request_monotonic_ns": None,
        "response_monotonic_ns": None,
        "rtt_ms": None,
        "source_timestamp": None,
        "as_of": None,
        "freshness_basis": None,
        "raw_sha256": raw_sha256,
        "fee_evidence": None,
        "sides": None,
        "requests": requests,
        "raw_response": raw_response,
        "status": "ERROR",
        "reason_code": error.reason_code,
        "error": str(error),
        "error_evidence": error.evidence,
        "live_eligible": False,
    }


class PmxtReadOnlyMonitor:
    """Exactly one PMXT catalog read followed by public native evidence reads."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        pmxt_base_url: str | None = None,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        config: MonitorConfig | None = None,
        pmxt_transport: httpx.BaseTransport | None = None,
        native_client: NativeEvidenceClient | Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._pmxt_base_url = pmxt_base_url
        self._output_dir = Path(output_dir)
        self._config = config or MonitorConfig()
        self._pmxt_transport = pmxt_transport
        self._native_client = native_client

    def _persist_pmxt_failure(
        self,
        *,
        run_id: str,
        query: PmxtQuery,
        requested_at: str,
        received_at: str,
        error: Exception,
        raw_payload: Any = None,
    ) -> MonitorRunArtifacts:
        router_error: dict[str, Any] | None = None
        if isinstance(error, PmxtRouterError):
            router_error = {"reason_code": error.reason_code}
            if error.evidence:
                router_error["evidence"] = error.evidence
        raw_pmxt = {
            "schema_version": 1,
            "source": "pmxt_router",
            "run_id": run_id,
            "requested_at": requested_at,
            "received_at": received_at,
            "query": query.as_dict(),
            "status": "ERROR",
            "error": {"type": type(error).__name__, "message": str(error)},
            "pmxt_network_requests": 1,
            "pmxt_retry_attempts": 0,
            "live_eligible": False,
        }
        if router_error is not None:
            raw_pmxt["router_error"] = router_error
        if raw_payload is not None:
            raw_pmxt["payload_sha256"] = canonical_json_sha256(raw_payload)
            raw_pmxt["payload"] = raw_payload
        return persist_monitor_run(
            self._output_dir,
            run_id=run_id,
            status="PMXT_SYNC_FAILED",
            config=self._config.as_dict(),
            counts={
                "pmxt_clusters": 0,
                "pmxt_network_requests": 1,
                "pmxt_retry_attempts": 0,
                "verified_candidates": 0,
                "alerts": 0,
            },
            raw_pmxt=raw_pmxt,
            candidates=[],
            raw_native_metadata=[],
            semantic_decisions=[],
            rejections=[],
            native_books=[],
            calculations=[],
            alerts=[],
        )

    def _evaluate_candidates(
        self,
        *,
        run_id: str,
        raw_pmxt: Mapping[str, Any],
        all_candidates: list[Mapping[str, Any]],
        rejections: list[dict[str, Any]],
        cluster_count: int,
        bounded_cluster_count: int,
        cluster_bound_rejection_count: int,
        pmxt_rejection_count: int,
        pmxt_network_requests: int,
        provenance: Mapping[str, Any] | None = None,
    ) -> MonitorOutcome:
        """Evaluate already-bounded discovery rows without any PMXT access."""

        eligible_candidates: list[Mapping[str, Any]] = []
        for candidate in all_candidates:
            confidence = candidate.get("relation_confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or confidence < self._config.min_confidence
            ):
                rejections.append(
                    {
                        "stage": "candidate_filter",
                        "candidate_id": candidate.get("candidate_id"),
                        "reason": "below_or_invalid_current_minimum_confidence",
                        "minimum_confidence": self._config.min_confidence,
                        "live_eligible": False,
                    }
                )
                continue
            eligible_candidates.append(candidate)
        candidates = eligible_candidates[: self._config.limit]
        for candidate in eligible_candidates[self._config.limit :]:
            rejections.append(
                {
                    "stage": "candidate_bound",
                    "candidate_id": candidate.get("candidate_id"),
                    "reason": "monitor_candidate_limit_exceeded",
                    "limit": self._config.limit,
                    "live_eligible": False,
                }
            )

        raw_native_metadata: list[dict[str, Any]] = []
        semantic_decisions: list[dict[str, Any]] = []
        native_books: list[dict[str, Any]] = []
        calculations: list[dict[str, Any]] = []
        alerts: list[dict[str, Any]] = []

        native_client = self._native_client
        owns_native_client = native_client is None
        if native_client is None:
            native_client = NativeEvidenceClient(timeout=self._config.timeout_seconds)
        try:
            for candidate in candidates:
                metadata_records: list[dict[str, Any]] = []
                for side in ("a", "b"):
                    try:
                        metadata = native_client.fetch_metadata(candidate, side)
                    except NativeEvidenceError as exc:
                        metadata = _native_error_record(candidate, side, exc)
                    metadata_records.append(metadata)
                    raw_native_metadata.append(metadata)

                decision = verify_semantics(candidate, metadata_records[0], metadata_records[1])
                semantic_decisions.append(decision)
                if decision["status"] != VERIFIED_EQUIVALENT:
                    rejections.append(
                        {
                            "stage": "semantic_verification",
                            "candidate_id": candidate.get("candidate_id"),
                            "status": decision["status"],
                            "reason_codes": decision["reason_codes"],
                            "live_eligible": False,
                        }
                    )
                    continue

                def capture_book(metadata: Mapping[str, Any]) -> dict[str, Any]:
                    try:
                        return native_client.fetch_book(metadata, depth=self._config.native_book_depth)
                    except NativeEvidenceError as exc:
                        return _book_error_record(metadata, exc)

                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pmxt-native-book") as executor:
                    books = list(executor.map(capture_book, metadata_records))
                native_books.extend(books)

                evaluated_at = utc_now_iso()
                policy = {
                    "requested_size": self._config.requested_size,
                    "max_book_age_seconds": self._config.max_book_age_seconds,
                    "max_cross_venue_skew_seconds": self._config.max_cross_venue_skew_seconds,
                    "annual_capital_rate": self._config.annual_capital_rate,
                    "capital_lock_days": _capital_lock_days(metadata_records, evaluated_at),
                    "explicit_slippage_buffer_per_unit": self._config.explicit_slippage_buffer_per_unit,
                    "timestamp_skew_buffer_per_unit": self._config.timestamp_skew_buffer_per_unit,
                    "settlement_divergence_buffer_per_unit": self._config.settlement_divergence_buffer_per_unit,
                    "collateral_basis_buffer_per_unit": self._config.collateral_basis_buffer_per_unit,
                    "rebalancing_withdrawal_allowance_per_unit": (
                        self._config.rebalancing_withdrawal_allowance_per_unit
                    ),
                    "net_residual_threshold": self._config.net_residual_threshold,
                }
                calculation = calculate_shadow(
                    decision,
                    books[0],
                    books[1],
                    policy,
                    evaluated_at=evaluated_at,
                )
                calculations.append(calculation)
                if calculation["alert"]:
                    alerts.append(calculation)
        finally:
            if owns_native_client:
                native_client.close()

        supporting_evidence = None
        supporting_evidence_reader = getattr(native_client, "supporting_evidence", None)
        if callable(supporting_evidence_reader):
            supporting_evidence = supporting_evidence_reader()

        verified_count = sum(decision["status"] == VERIFIED_EQUIVALENT for decision in semantic_decisions)
        rejected_count = sum(decision["status"] == REJECTED for decision in semantic_decisions)
        needs_review_count = sum(decision["status"] == NEEDS_REVIEW for decision in semantic_decisions)
        fee_evidence_unavailable_count = sum(
            calculation.get("status") == "FEE_EVIDENCE_UNAVAILABLE" for calculation in calculations
        )
        if verified_count == 0:
            status = "NO_VERIFIED_CANDIDATES"
        elif alerts:
            status = "SHADOW_ALERTS"
        elif fee_evidence_unavailable_count:
            status = "FEE_EVIDENCE_UNAVAILABLE"
        else:
            status = "NO_EXECUTABLE_SHADOW_EDGE"

        candidate_filter_rejections = len(all_candidates) - len(candidates)
        native_book_error_count = sum(book.get("status") == "ERROR" for book in native_books)
        native_book_success_count = len(native_books) - native_book_error_count
        counts = {
            "pmxt_network_requests": pmxt_network_requests,
            "pmxt_clusters": cluster_count,
            "bounded_clusters": bounded_cluster_count,
            "cluster_bound_rejections": cluster_bound_rejection_count,
            "normalized_candidate_proposals": len(all_candidates),
            "normalized_candidates": len(candidates),
            "pmxt_rejections": pmxt_rejection_count,
            "verified_candidates": verified_count,
            "rejected_candidates": rejected_count,
            "needs_review_candidates": needs_review_count,
            "native_book_attempts": len(native_books),
            "native_books_captured": native_book_success_count,
            "native_book_errors": native_book_error_count,
            "shadow_calculations": len(calculations),
            "fee_evidence_unavailable": fee_evidence_unavailable_count,
            "alerts": len(alerts),
        }
        artifacts = persist_monitor_run(
            self._output_dir,
            run_id=run_id,
            status=status,
            config=self._config.as_dict(),
            counts=counts,
            raw_pmxt=raw_pmxt,
            candidates=all_candidates,
            raw_native_metadata=raw_native_metadata,
            semantic_decisions=semantic_decisions,
            rejections=rejections,
            native_books=native_books,
            calculations=calculations,
            alerts=alerts,
            supporting_evidence=supporting_evidence,
            provenance=provenance,
        )
        return MonitorOutcome(
            run_id=run_id,
            status=status,
            artifacts=artifacts,
            cluster_count=cluster_count,
            candidate_count=len(candidates),
            verified_count=verified_count,
            rejected_count=rejected_count,
            needs_review_count=needs_review_count,
            pmxt_rejection_count=pmxt_rejection_count,
            bounded_rejection_count=cluster_bound_rejection_count + candidate_filter_rejections,
            total_rejection_rows=len(rejections),
            alert_count=len(alerts),
        )

    def sync_once(self) -> MonitorOutcome:
        """Run once; missing PMXT credentials fail before I/O or artifact creation."""

        api_key = (self._api_key or os.environ.get("PMXT_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError("PMXT_API_KEY is required; no network request was made and no artifacts were created")

        run_id = new_snapshot_id()
        query = self._config.query()
        requested_at = utc_now_iso()
        client_kwargs: dict[str, Any] = {
            "timeout": self._config.timeout_seconds,
            "transport": self._pmxt_transport,
        }
        if self._pmxt_base_url:
            client_kwargs["base_url"] = self._pmxt_base_url
        raw_payload: Any = None
        try:
            with PmxtRouterClient(api_key, **client_kwargs) as pmxt_client:
                raw_payload = pmxt_client.fetch_market_clusters(query)
            received_at = utc_now_iso()
            clusters = extract_clusters(raw_payload, max_clusters=query.limit)
        except (PmxtRouterError, httpx.HTTPError) as exc:
            received_at = utc_now_iso()
            artifacts = self._persist_pmxt_failure(
                run_id=run_id,
                query=query,
                requested_at=requested_at,
                received_at=received_at,
                error=exc,
                raw_payload=raw_payload,
            )
            raise RuntimeError(f"PMXT sync failed; evidence manifest: {artifacts.manifest_path}") from exc

        bounded_clusters, bound_rejections = _bounded_cluster_shapes(
            clusters,
            per_cluster_limit=max(2, self._config.limit),
        )
        normalized = normalize_clusters(
            bounded_clusters,
            query=query,
            snapshot_id=run_id,
            observed_at=received_at,
        )
        all_candidates = normalized.rows
        rejections = [*bound_rejections]
        rejections.extend({"stage": "pmxt_normalization", **row, "live_eligible": False} for row in normalized.rejected)
        raw_pmxt = {
            "schema_version": 1,
            "source": "pmxt_router",
            "run_id": run_id,
            "requested_at": requested_at,
            "received_at": received_at,
            "query": query.as_dict(),
            "payload_sha256": canonical_json_sha256(raw_payload),
            "payload": raw_payload,
            "live_eligible": False,
        }
        return self._evaluate_candidates(
            run_id=run_id,
            raw_pmxt=raw_pmxt,
            rejections=rejections,
            all_candidates=list(all_candidates),
            cluster_count=len(clusters),
            bounded_cluster_count=len(bounded_clusters),
            cluster_bound_rejection_count=len(bound_rejections),
            pmxt_rejection_count=len(normalized.rejected),
            pmxt_network_requests=1,
        )

    def continue_once(self, source_run_id: str) -> MonitorOutcome:
        """Refresh only public native evidence from one hash-verified source run."""

        source = load_verified_monitor_source(self._output_dir, source_run_id)
        run_id = new_snapshot_id()
        continued_at = utc_now_iso()
        raw_pmxt = {
            **dict(source.raw_pmxt),
            "run_id": run_id,
            "mode": "NATIVE_ONLY_CONTINUATION",
            "source_run_id": source_run_id,
            "continued_at": continued_at,
            "pmxt_network_requests": 0,
            "live_eligible": False,
        }
        provenance = {
            **dict(source.provenance),
            "mode": "NATIVE_ONLY_CONTINUATION",
            "continued_at": continued_at,
            "pmxt_network_requests": 0,
        }
        source_counts = source.manifest.get("counts")
        cluster_count = source_counts.get("pmxt_clusters", 0) if isinstance(source_counts, Mapping) else 0
        if isinstance(cluster_count, bool) or not isinstance(cluster_count, int) or cluster_count < 0:
            raise SourceRunValidationError("source PMXT cluster count is invalid")
        return self._evaluate_candidates(
            run_id=run_id,
            raw_pmxt=raw_pmxt,
            all_candidates=[dict(candidate) for candidate in source.candidates],
            rejections=[],
            cluster_count=cluster_count,
            bounded_cluster_count=cluster_count,
            cluster_bound_rejection_count=0,
            pmxt_rejection_count=0,
            pmxt_network_requests=0,
            provenance=provenance,
        )

    def run(self, *, source_run_id: str | None = None) -> None:
        outcome = self.continue_once(source_run_id) if source_run_id else self.sync_once()
        print(f"PMXT monitor complete: {outcome.status}")
        print(f"  clusters: {outcome.cluster_count}")
        print(f"  candidates: {outcome.candidate_count}")
        print(f"  verified: {outcome.verified_count}")
        print(f"  semantic_rejected: {outcome.rejected_count}")
        print(f"  needs_review: {outcome.needs_review_count}")
        print(f"  pmxt_rejections: {outcome.pmxt_rejection_count}")
        print(f"  bounded_rejections: {outcome.bounded_rejection_count}")
        print(f"  total_rejection_rows: {outcome.total_rejection_rows}")
        print(f"  alerts: {outcome.alert_count}")
        print(f"  manifest: {outcome.artifacts.manifest_path}")
        print("  execution: disabled; live_eligible=false; no orders sent")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one identity-only PMXT/native shadow monitor with immutable research artifacts."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--source-run-id",
        default=None,
        help="Hash-verified prior run to continue with public native reads only; makes zero PMXT requests",
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--threshold", type=float, default=DEFAULT_NET_RESIDUAL_THRESHOLD)
    parser.add_argument("--requested-size", type=float, default=10.0)
    parser.add_argument("--max-book-age-seconds", type=float, default=15.0)
    parser.add_argument("--max-book-skew-seconds", type=float, default=3.0)
    parser.add_argument("--annual-capital-rate", type=float, default=0.05)
    parser.add_argument("--explicit-slippage-buffer-per-unit", type=float, default=0.0)
    parser.add_argument("--timestamp-skew-buffer-per-unit", type=float, default=0.0)
    parser.add_argument("--settlement-divergence-buffer-per-unit", type=float, default=0.0)
    parser.add_argument("--collateral-basis-buffer-per-unit", type=float, default=0.0)
    parser.add_argument("--rebalancing-withdrawal-allowance-per-unit", type=float, default=0.0)
    parser.add_argument("--native-book-depth", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = MonitorConfig(
            limit=args.limit,
            min_confidence=args.min_confidence,
            net_residual_threshold=args.threshold,
            requested_size=args.requested_size,
            max_book_age_seconds=args.max_book_age_seconds,
            max_cross_venue_skew_seconds=args.max_book_skew_seconds,
            annual_capital_rate=args.annual_capital_rate,
            explicit_slippage_buffer_per_unit=args.explicit_slippage_buffer_per_unit,
            timestamp_skew_buffer_per_unit=args.timestamp_skew_buffer_per_unit,
            settlement_divergence_buffer_per_unit=args.settlement_divergence_buffer_per_unit,
            collateral_basis_buffer_per_unit=args.collateral_basis_buffer_per_unit,
            rebalancing_withdrawal_allowance_per_unit=args.rebalancing_withdrawal_allowance_per_unit,
            native_book_depth=args.native_book_depth,
            timeout_seconds=args.timeout_seconds,
        )
        PmxtReadOnlyMonitor(
            output_dir=args.output_dir,
            config=config,
        ).run(source_run_id=args.source_run_id)
        return 0
    except (PmxtRouterError, RuntimeError, ValueError) as exc:
        print(f"PMXT monitor failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
