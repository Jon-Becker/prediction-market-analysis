"""Indexer for Kalshi events data.

Events (and their API-supplied category) aren't split into recent and historical
the way markets and trades are, so one shared events dataset serves both. Event
tickers are collected from every markets directory that has data.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/events")
MARKETS_DIRS = [Path("data/kalshi/markets"), Path("data/kalshi/historical/markets")]

BATCH_SIZE = 10000
# The /events endpoint accepts up to 100 event tickers per request
TICKERS_PER_REQUEST = 100


class KalshiEventsIndexer(Indexer):
    """Fetches and stores Kalshi events for every saved market (recent and historical)."""

    def __init__(self, max_workers: int = 10):
        super().__init__(
            name="kalshi_events",
            description="Backfills Kalshi events (incl. category) for all saved markets to parquet files",
        )
        self._max_workers = max_workers

        # unauth requests to /events endpoint is rate-limited to 4 requests/sec discovered through testing
        self.client = KalshiClient(rate_limit=4)
    
    @staticmethod
    def _load_existing_tickers() -> set[str]:
        if not any(DATA_DIR.glob("events_*.parquet")):
            return set()

        print("Loading existing events for deduplication...")
        try:
            result = duckdb.sql(f"SELECT DISTINCT event_ticker FROM '{DATA_DIR}/events_*.parquet'").fetchall()
        except duckdb.Error:
            return set()

        existing_tickers = {row[0] for row in result}
        print(f"Found {len(existing_tickers)} existing events")
        return existing_tickers

    @staticmethod
    def _load_market_event_tickers() -> list[str]:
        """Distinct event tickers across every markets directory that has data."""
        sources = [f"'{d}/markets_*_*.parquet'" for d in MARKETS_DIRS if any(d.glob("markets_*_*.parquet"))]
        if not sources:
            return []

        union = " UNION ".join(f"SELECT DISTINCT event_ticker FROM {s} WHERE volume >= 100" for s in sources)
        result = duckdb.sql(f"SELECT DISTINCT event_ticker FROM ({union}) ORDER BY event_ticker").fetchall()
        return [row[0] for row in result]

    @staticmethod
    def _next_chunk_index() -> int:
        indices = []
        for f in DATA_DIR.glob("events_*.parquet"):
            parts = f.stem.split("_")
            if len(parts) >= 2:
                try:
                    indices.append(int(parts[1]))
                except ValueError:
                    pass
        return max(indices) + BATCH_SIZE if indices else 0

    def run(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        existing_tickers = self._load_existing_tickers()

        all_tickers = self._load_market_event_tickers()
        print(f"Found {len(all_tickers)} unique events")

        # Filter to tickers not fully processed
        tickers_to_process = [t for t in all_tickers if t not in existing_tickers]
        print(
            f"Skipped {len(all_tickers) - len(tickers_to_process)} already processed, {len(tickers_to_process)} to fetch"
        )

        if not tickers_to_process:
            print("Nothing to process")
            return

        all_events: list[dict] = []
        total_events_saved = 0
        next_chunk_idx = self._next_chunk_index()

        def save_batch(events_batch: list[dict]) -> int:
            nonlocal next_chunk_idx
            if not events_batch:
                return 0
            chunk_path = DATA_DIR / f"events_{next_chunk_idx}_{next_chunk_idx + BATCH_SIZE}.parquet"
            df = pd.DataFrame(events_batch)
            df.to_parquet(chunk_path)
            next_chunk_idx += BATCH_SIZE
            return len(events_batch)

        def fetch_events(tickers: list[str]) -> list[dict]:
            """Fetch events for a list of event tickers."""
            events = self.client.list_events(tickers=",".join(tickers))
            if not events:
                return []
            fetched_at = datetime.utcnow()
            return [
                {**asdict(event), "_fetched_at": fetched_at}
                for event in events
                if event.event_ticker not in existing_tickers
            ]

        # Concurrent fetching, one request per chunk of tickers
        chunks = [
            tickers_to_process[i : i + TICKERS_PER_REQUEST]
            for i in range(0, len(tickers_to_process), TICKERS_PER_REQUEST)
        ]

        pbar = tqdm(total=len(tickers_to_process), desc="Fetching events")
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(fetch_events, chunk): chunk for chunk in chunks}

            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    events_data = future.result()
                    if events_data:
                        all_events.extend(events_data)

                    pbar.update(len(chunk))
                    pbar.set_postfix(buffer=len(all_events), saved=total_events_saved, last=chunk[-1] if chunk else "")

                    # Save in batches
                    while len(all_events) >= BATCH_SIZE:
                        saved = save_batch(all_events[:BATCH_SIZE])
                        total_events_saved += saved
                        all_events = all_events[BATCH_SIZE:]

                except Exception as e:
                    pbar.update(len(chunk))
                    tqdm.write(f"Error fetching {chunk[0]}..{chunk[-1]}: {e}")
                futures.pop(future, None)
        pbar.close()

        self.client.close()
        # Save remaining
        if all_events:
            total_events_saved += save_batch(all_events)

        print(
            f"\nBackfill events complete: {len(tickers_to_process)} events processed, {total_events_saved} events saved"
        )
