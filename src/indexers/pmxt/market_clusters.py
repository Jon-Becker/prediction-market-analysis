"""Indexer and CLI for the first PMXT cross-venue research milestone."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.common.indexer import Indexer
from src.indexers.pmxt.candidates import (
    SyncArtifacts,
    new_snapshot_id,
    normalize_clusters,
    persist_sync,
    utc_now_iso,
)
from src.indexers.pmxt.client import PmxtRouterClient, PmxtRouterError, extract_clusters
from src.indexers.pmxt.models import MATCH_RELATIONS, SORT_ORDERS, PmxtQuery

DEFAULT_OUTPUT_DIR = Path("data/pmxt")


class PmxtMarketClustersIndexer(Indexer):
    """Fetch PMXT cluster candidates and persist shadow-only mapping evidence."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        query: PmxtQuery | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            name="pmxt_market_clusters",
            description="Fetches read-only PMXT cross-venue candidates for shadow analysis",
        )
        self._api_key = api_key
        self._base_url = base_url
        self._output_dir = Path(output_dir)
        self._query = query or PmxtQuery()
        self._timeout = timeout

    def sync_once(self) -> SyncArtifacts:
        api_key = (self._api_key or os.environ.get("PMXT_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError("PMXT_API_KEY is required; no network request was made")

        snapshot_id = new_snapshot_id()
        observed_at = utc_now_iso()
        client_kwargs = {"base_url": self._base_url} if self._base_url else {}
        with PmxtRouterClient(api_key, timeout=self._timeout, **client_kwargs) as client:
            raw_payload = client.fetch_market_clusters(self._query)

        clusters = extract_clusters(raw_payload)
        result = normalize_clusters(
            clusters,
            query=self._query,
            snapshot_id=snapshot_id,
            observed_at=observed_at,
        )
        return persist_sync(
            self._output_dir,
            snapshot_id=snapshot_id,
            observed_at=observed_at,
            query=self._query,
            raw_payload=raw_payload,
            result=result,
            cluster_count=len(clusters),
        )

    def run(self) -> None:
        artifacts = self.sync_once()
        print(f"PMXT sync complete: {artifacts.cluster_count} clusters")
        print(f"  candidates: {artifacts.candidate_count}")
        print(f"  rejected: {artifacts.rejected_count}")
        print(f"  manifest: {artifacts.manifest_path}")
        print("  execution: disabled; all mappings remain UNVERIFIED")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch PMXT Router market clusters into shadow-only research artifacts."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--relation", choices=sorted(MATCH_RELATIONS), default="identity")
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--min-venues", type=int, default=2)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sort", choices=sorted(SORT_ORDERS), default="volume")
    parser.add_argument(
        "--venues",
        default="kalshi,polymarket",
        help="Comma-separated venue filter; default: kalshi,polymarket",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    venues = tuple(venue.strip() for venue in args.venues.split(",") if venue.strip())
    try:
        query = PmxtQuery(
            relation=args.relation,
            min_confidence=args.min_confidence,
            min_venues=args.min_venues,
            limit=args.limit,
            offset=args.offset,
            sort=args.sort,
            venues=venues,
        )
        PmxtMarketClustersIndexer(
            output_dir=args.output_dir,
            query=query,
        ).run()
        return 0
    except (PmxtRouterError, RuntimeError, ValueError) as exc:
        print(f"PMXT sync failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
