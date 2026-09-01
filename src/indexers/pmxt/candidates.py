"""Normalize PMXT matches into append-only, reviewable candidate mappings."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from src.indexers.pmxt.models import PmxtQuery


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for manifests and evidence rows."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_snapshot_id() -> str:
    """Create an immutable run identifier that is safe to use in filenames."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class NormalizationResult:
    rows: list[dict[str, Any]]
    rejected: list[dict[str, Any]]


@dataclass(frozen=True)
class SyncArtifacts:
    snapshot_id: str
    raw_path: Path
    mapping_path: Path
    rejection_path: Path
    manifest_path: Path
    cluster_count: int
    candidate_count: int
    rejected_count: int


def _value(record: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _displayed_quote(value: Mapping[str, Any], *keys: str) -> tuple[float | None, str]:
    """Normalize one displayed quote while retaining missing-vs-invalid state."""

    for key in keys:
        if key not in value or value[key] is None:
            continue
        quote = _number(value[key])
        if quote is None or not 0.0 <= quote <= 1.0:
            return None, "INVALID"
        return quote, "VALID"
    return None, "MISSING"


def _catalog_quote_coherence(
    *,
    price: float | None,
    best_bid: float | None,
    best_ask: float | None,
    best_bid_state: str,
    best_ask_state: str,
) -> str:
    """Classify catalog-price coherence without treating it as executable evidence."""

    if "INVALID" in {best_bid_state, best_ask_state}:
        return "INVALID_DISPLAYED_SPREAD"
    if best_bid_state != "VALID" or best_ask_state != "VALID":
        return "NOT_EVALUABLE"
    if best_bid is None or best_ask is None or best_bid > best_ask:
        return "INVALID_DISPLAYED_SPREAD"
    if price is None or not 0.0 <= price <= 1.0:
        return "NOT_EVALUABLE"
    if best_bid <= price <= best_ask:
        return "WITHIN_DISPLAYED_SPREAD"
    return "OUTSIDE_DISPLAYED_SPREAD"


def _normalize_outcomes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    outcomes: list[dict[str, Any]] = []
    for outcome in value:
        if not isinstance(outcome, Mapping):
            continue
        price = _number(_value(outcome, "price"))
        best_bid, best_bid_state = _displayed_quote(outcome, "bestBid", "best_bid")
        best_ask, best_ask_state = _displayed_quote(outcome, "bestAsk", "best_ask")
        outcomes.append(
            {
                "outcome_id": _string(_value(outcome, "outcomeId", "outcome_id", "id")),
                "label": _string(_value(outcome, "label", "name", "title")),
                "price": price,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "catalog_quote_coherence_status": _catalog_quote_coherence(
                    price=price,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    best_bid_state=best_bid_state,
                    best_ask_state=best_ask_state,
                ),
                "metadata": (
                    dict(metadata) if isinstance((metadata := _value(outcome, "metadata", default={})), Mapping) else {}
                ),
            }
        )
    return outcomes


def _normalize_market(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None

    market_id = _string(_value(value, "marketId", "market_id", "id"))
    venue = _string(_value(value, "sourceExchange", "source_exchange", "venue", "exchange"))
    if not market_id or not venue:
        return None

    return {
        "pmxt_market_id": market_id,
        "venue": venue.lower(),
        "title": _string(_value(value, "title", "question")),
        "description": _string(_value(value, "description")),
        "slug": _string(_value(value, "slug")),
        "url": _string(_value(value, "url")),
        "event_id": _string(_value(value, "eventId", "event_id")),
        "resolution_date": _string(_value(value, "resolutionDate", "resolution_date")),
        "contract_address": _string(_value(value, "contractAddress", "contract_address")),
        "source_metadata": (
            dict(source_metadata)
            if isinstance((source_metadata := _value(value, "sourceMetadata", "source_metadata", default={})), Mapping)
            else {}
        ),
        "outcomes": _normalize_outcomes(_value(value, "outcomes", default=[])),
    }


def _pair_key(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[str, str, str, str]:
    values = (
        str(left.get("venue", "")),
        str(left.get("pmxt_market_id", "")),
        str(right.get("venue", "")),
        str(right.get("pmxt_market_id", "")),
    )
    first, second = sorted((values[:2], values[2:]))
    return first[0], first[1], second[0], second[1]


def _edge_lookup(
    raw_matches: Any,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], set[tuple[str, str]]]:
    if not isinstance(raw_matches, Sequence) or isinstance(raw_matches, (str, bytes)):
        return {}, set()

    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    ambiguous: set[tuple[str, str]] = set()
    for edge in raw_matches:
        if not isinstance(edge, Mapping):
            continue
        left = _string(_value(edge, "marketAId", "market_a_id", "marketA"))
        right = _string(_value(edge, "marketBId", "market_b_id", "marketB"))
        if left and right:
            key = tuple(sorted((left, right)))
            if key in lookup or key in ambiguous:
                lookup.pop(key, None)
                ambiguous.add(key)
            else:
                lookup[key] = edge
    return lookup, ambiguous


def _candidate_id(cluster_id: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    identity = "|".join((cluster_id, *_pair_key(left, right)))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"pmxt_candidate_{digest}"


def normalize_clusters(
    clusters: list[dict[str, Any]],
    *,
    query: PmxtQuery,
    snapshot_id: str,
    observed_at: str,
) -> NormalizationResult:
    """Create candidate rows while refusing to certify semantic equivalence."""

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    cluster_id_counts: dict[str, int] = {}
    for cluster in clusters:
        cluster_id = _string(_value(cluster, "clusterId", "cluster_id"))
        if cluster_id:
            cluster_id_counts[cluster_id] = cluster_id_counts.get(cluster_id, 0) + 1
    duplicate_cluster_ids = {cluster_id for cluster_id, count in cluster_id_counts.items() if count > 1}
    reported_duplicate_cluster_ids: set[str] = set()
    seen_candidate_ids: set[str] = set()

    for cluster in clusters:
        cluster_id = _string(_value(cluster, "clusterId", "cluster_id"))
        title = _string(_value(cluster, "canonicalTitle", "canonical_title"))
        if not cluster_id:
            rejected.append(
                {
                    "snapshot_id": snapshot_id,
                    "observed_at": observed_at,
                    "reason": "missing_cluster_id",
                    "live_eligible": False,
                }
            )
            continue
        if cluster_id in duplicate_cluster_ids:
            if cluster_id not in reported_duplicate_cluster_ids:
                rejected.append(
                    {
                        "snapshot_id": snapshot_id,
                        "observed_at": observed_at,
                        "cluster_id": cluster_id,
                        "canonical_title": title,
                        "reason": "duplicate_cluster_id",
                        "duplicate_count": cluster_id_counts[cluster_id],
                        "live_eligible": False,
                    }
                )
                reported_duplicate_cluster_ids.add(cluster_id)
            continue

        raw_markets = _value(cluster, "markets", default=[])
        if not isinstance(raw_markets, Sequence) or isinstance(raw_markets, (str, bytes)):
            raw_markets = []
        markets = [normalized for item in raw_markets if (normalized := _normalize_market(item)) is not None]
        market_keys = [(market["venue"], market["pmxt_market_id"]) for market in markets]
        duplicate_market_keys = sorted({key for key in market_keys if market_keys.count(key) > 1})
        if duplicate_market_keys:
            rejected.append(
                {
                    "snapshot_id": snapshot_id,
                    "observed_at": observed_at,
                    "cluster_id": cluster_id,
                    "canonical_title": title,
                    "reason": "duplicate_normalized_market_key",
                    "duplicate_market_keys": [
                        {"venue": venue, "pmxt_market_id": market_id} for venue, market_id in duplicate_market_keys
                    ],
                    "live_eligible": False,
                }
            )
            continue
        distinct_venues = {market["venue"] for market in markets}
        if len(markets) < 2:
            rejected.append(
                {
                    "snapshot_id": snapshot_id,
                    "observed_at": observed_at,
                    "cluster_id": cluster_id,
                    "canonical_title": title,
                    "reason": "fewer_than_two_usable_markets",
                    "live_eligible": False,
                }
            )
            continue
        if len(distinct_venues) < query.min_venues:
            rejected.append(
                {
                    "snapshot_id": snapshot_id,
                    "observed_at": observed_at,
                    "cluster_id": cluster_id,
                    "canonical_title": title,
                    "reason": "insufficient_distinct_venues",
                    "venue_count": len(distinct_venues),
                    "live_eligible": False,
                }
            )
            continue

        cluster_relations = _value(cluster, "relations", default=[])
        if not isinstance(cluster_relations, Sequence) or isinstance(cluster_relations, (str, bytes)):
            cluster_relations = [query.relation]
        cluster_relations = [str(relation) for relation in cluster_relations]
        cluster_relation = cluster_relations[0] if cluster_relations else query.relation
        cluster_confidence = _number(_value(cluster, "confidence"))
        edge_lookup, ambiguous_edges = _edge_lookup(_value(cluster, "rawMatches", "raw_matches", default=[]))

        for left, right in combinations(markets, 2):
            if left["venue"] == right["venue"]:
                continue

            edge_key = tuple(sorted((left["pmxt_market_id"], right["pmxt_market_id"])))
            edge = edge_lookup.get(edge_key)
            edge_relation = _string(_value(edge or {}, "relation"))
            edge_confidence_raw = _value(edge or {}, "confidence")
            edge_confidence = _number(edge_confidence_raw)
            reasoning = _string(_value(edge or {}, "reasoning"))
            relation = edge_relation or cluster_relation
            confidence = edge_confidence if edge_confidence is not None else cluster_confidence
            candidate_id = _candidate_id(cluster_id, left, right)

            pair_context = {
                "snapshot_id": snapshot_id,
                "observed_at": observed_at,
                "cluster_id": cluster_id,
                "candidate_id": candidate_id,
                "venue_a": left["venue"],
                "pmxt_market_id_a": left["pmxt_market_id"],
                "venue_b": right["venue"],
                "pmxt_market_id_b": right["pmxt_market_id"],
                "relation": relation,
                "relation_confidence": confidence,
                "live_eligible": False,
            }
            if {left["venue"], right["venue"]} != {"kalshi", "polymarket"}:
                rejected.append({**pair_context, "reason": "unsupported_venue_pair"})
                continue
            if edge_key in ambiguous_edges:
                rejected.append({**pair_context, "reason": "ambiguous_duplicate_direct_raw_matches"})
                continue
            if edge is None:
                rejected.append({**pair_context, "reason": "missing_direct_raw_match"})
                continue
            if edge_relation is None:
                rejected.append({**pair_context, "reason": "missing_direct_edge_relation"})
                continue
            if edge_relation != "identity":
                rejected.append({**pair_context, "reason": "non_identity_relation"})
                continue
            if edge_confidence_raw is None:
                rejected.append({**pair_context, "reason": "missing_direct_edge_confidence"})
                continue
            if edge_confidence is None or not 0.0 <= edge_confidence <= 1.0:
                rejected.append({**pair_context, "reason": "invalid_direct_edge_confidence"})
                continue
            if edge_confidence < query.min_confidence:
                rejected.append({**pair_context, "reason": "below_minimum_relation_confidence"})
                continue
            if candidate_id in seen_candidate_ids:
                rejected.append({**pair_context, "reason": "duplicate_candidate_id"})
                continue
            seen_candidate_ids.add(candidate_id)

            outcomes = [*left["outcomes"], *right["outcomes"]]

            rows.append(
                {
                    "schema_version": 1,
                    "candidate_id": candidate_id,
                    "source": "pmxt_router",
                    "snapshot_id": snapshot_id,
                    "observed_at": observed_at,
                    "cluster_id": cluster_id,
                    "canonical_title": title,
                    "category": _string(_value(cluster, "category")),
                    "cluster_confidence": cluster_confidence,
                    "relation": relation,
                    "relation_confidence": confidence,
                    "relation_reasoning": reasoning,
                    "raw_edge_present": edge is not None,
                    "venue_count": len(distinct_venues),
                    "venue_a": left["venue"],
                    "pmxt_market_id_a": left["pmxt_market_id"],
                    "title_a": left["title"],
                    "description_a": left["description"],
                    "slug_a": left["slug"],
                    "url_a": left["url"],
                    "event_id_a": left["event_id"],
                    "resolution_date_a": left["resolution_date"],
                    "contract_address_a": left["contract_address"],
                    "source_metadata_a": left["source_metadata"],
                    "outcomes_a": left["outcomes"],
                    "venue_b": right["venue"],
                    "pmxt_market_id_b": right["pmxt_market_id"],
                    "title_b": right["title"],
                    "description_b": right["description"],
                    "slug_b": right["slug"],
                    "url_b": right["url"],
                    "event_id_b": right["event_id"],
                    "resolution_date_b": right["resolution_date"],
                    "contract_address_b": right["contract_address"],
                    "source_metadata_b": right["source_metadata"],
                    "outcomes_b": right["outcomes"],
                    "verification_status": "UNVERIFIED",
                    "semantic_verification": "PENDING_REVIEW",
                    "book_status": "NOT_REFRESHED",
                    "price_source": "PMXT_CATALOG_ONLY",
                    "catalog_price_conflicts_with_displayed_spread": any(
                        outcome["catalog_quote_coherence_status"] == "OUTSIDE_DISPLAYED_SPREAD" for outcome in outcomes
                    ),
                    "fair_value_status": "NOT_CALCULATED",
                    "live_eligible": False,
                }
            )

    return NormalizationResult(rows=rows, rejected=rejected)


def _write_new(path: Path, content: str) -> None:
    """Create an evidence file once; never silently overwrite an existing run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _write_batch_new(items: list[tuple[Path, str]]) -> None:
    """Preflight and roll back only files created by this batch on failure."""

    for path, _ in items:
        if path.exists():
            raise FileExistsError(f"immutable PMXT artifact already exists: {path}")

    written: list[Path] = []
    try:
        for path, content in items:
            _write_new(path, content)
            written.append(path)
    except BaseException:
        for path in reversed(written):
            if path.exists() and path.is_file():
                path.unlink()
        raise


def persist_sync(
    output_dir: Path | str,
    *,
    snapshot_id: str,
    observed_at: str,
    query: PmxtQuery,
    raw_payload: Any,
    result: NormalizationResult,
    cluster_count: int,
) -> SyncArtifacts:
    """Persist raw response, normalized candidates, rejections, and a run manifest."""

    root = Path(output_dir)
    raw_path = root / "raw" / f"matched_market_clusters_{snapshot_id}.json"
    mapping_path = root / "mappings" / f"candidate_mappings_{snapshot_id}.jsonl"
    rejection_path = root / "rejections" / f"rejected_clusters_{snapshot_id}.jsonl"
    manifest_path = root / "runs" / f"sync_{snapshot_id}.json"

    raw_document = {
        "schema_version": 1,
        "source": "pmxt_router",
        "snapshot_id": snapshot_id,
        "observed_at": observed_at,
        "query": query.as_dict(),
        "payload": raw_payload,
    }
    raw_content = json.dumps(raw_document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"

    mapping_content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in result.rows
    )
    rejection_content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in result.rejected
    )

    manifest = {
        "schema_version": 1,
        "source": "pmxt_router",
        "snapshot_id": snapshot_id,
        "observed_at": observed_at,
        "query": query.as_dict(),
        "cluster_count": cluster_count,
        "candidate_count": len(result.rows),
        "rejected_count": len(result.rejected),
        "verification_status": "UNVERIFIED",
        "live_eligible": False,
        "artifacts": {
            "raw": str(raw_path.relative_to(root)),
            "mappings": str(mapping_path.relative_to(root)),
            "rejections": str(rejection_path.relative_to(root)),
        },
    }
    manifest_content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_batch_new(
        [
            (raw_path, raw_content),
            (mapping_path, mapping_content),
            (rejection_path, rejection_content),
            (manifest_path, manifest_content),
        ]
    )

    return SyncArtifacts(
        snapshot_id=snapshot_id,
        raw_path=raw_path,
        mapping_path=mapping_path,
        rejection_path=rejection_path,
        manifest_path=manifest_path,
        cluster_count=cluster_count,
        candidate_count=len(result.rows),
        rejected_count=len(result.rejected),
    )
