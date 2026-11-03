"""Indexer for Kalshi trades data."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/historical/events")
MARKETS_DIR = Path("data/kalshi/historical/markets")
CURSOR_FILE = Path("data/kalshi/.backfill_historical_events_cursor")


class KalshiHistoricalEventsIndexer(Indexer):
    """Fetches and stores Kalshi historical events data."""

    def __init__(
        self,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        max_workers: int = 10,
    ):
        super().__init__(
            name="kalshi_historical_events",
            description="Backfills Kalshi historical events data to parquet files",
        )
        self._min_ts = min_ts
        self._max_ts = max_ts
        self._max_workers = max_workers

        self.client = KalshiClient()

    def run(self) -> None:
        BATCH_SIZE = 10000
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load existing trade IDs for deduplication
        existing_tickers: set[str] = set()
        parquet_files = list(DATA_DIR.glob("events_*.parquet"))
        if parquet_files:
            print("Loading existing events for deduplication...")
            try:
                result = duckdb.sql(f"SELECT DISTINCT event_ticker FROM '{DATA_DIR}/events_*.parquet'").fetchall()
                for event_ticker in result:
                    existing_tickers.add(event_ticker[0])
                print(f"Found {len(existing_tickers)} existing events")
            except Exception:
                pass

        all_tickers = duckdb.sql(f"""
            SELECT DISTINCT event_ticker FROM '{MARKETS_DIR}/markets_*_*.parquet'
            WHERE volume >= 100
            ORDER BY event_ticker
        """).fetchall()
        all_tickers = [row[0] for row in all_tickers]
        print(f"Found {len(all_tickers)} unique events")

        # Filter to tickers not fully processed
        tickers_to_process = [t for t in all_tickers if t not in existing_tickers]
        print(
            f"Skipped {len(all_tickers) - len(tickers_to_process)} already processed, "
            f"{len(tickers_to_process)} to fetch"
        )

        if not tickers_to_process:
            print("Nothing to process")
            return

        all_events: list[dict] = []
        total_events_saved = 0
        next_chunk_idx = 0

        # Calculate next chunk index
        if parquet_files:
            indices = []
            for f in parquet_files:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    try:
                        indices.append(int(parts[1]))
                    except ValueError:
                        pass
            if indices:
                next_chunk_idx = max(indices) + BATCH_SIZE

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

            ticker_str = ",".join(tickers)

            try:
                events = self.client.list_events(tickers=ticker_str)
                if not events:
                    return []
                fetched_at = datetime.utcnow()
                return [
                    {**asdict(event), "_fetched_at": fetched_at}
                    for event in events
                    if event.event_ticker not in existing_tickers
                ]
            finally:
                pass

        # Sequential fetching (for debugging)
        # pbar = tqdm(total=len(tickers_to_process), desc="Fetching trades")
        # for ticker in tickers_to_process:
        #     try:
        #         trades_data = fetch_ticker_trades(ticker)
        #         if trades_data:
        #             all_trades.extend(trades_data)

        #         pbar.update(1)
        #         pbar.set_postfix(buffer=len(all_trades), saved=total_trades_saved, last=ticker[-20:])

        #         # Save in batches
        #         if len(all_trades) >= BATCH_SIZE:
        #             saved = save_batch(all_trades[:BATCH_SIZE])
        #             total_trades_saved += saved
        #             all_trades = all_trades[BATCH_SIZE:]

        #     except Exception as e:
        #         pbar.update(1)
        #         tqdm.write(f"Error fetching {ticker}: {e}")

        # Concurrent fetching in chunks of tickers (up to 100 per request)
        CHUNK_TICKERS = 100
        chunks = [tickers_to_process[i : i + CHUNK_TICKERS] for i in range(0, len(tickers_to_process), CHUNK_TICKERS)]

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
                    tqdm.write(f"Error fetching {chunk}: {e}")
                futures.pop(future, None)
        pbar.close()

        self.client.close()
        # Save remaining
        if all_events:
            total_events_saved += save_batch(all_events)

        print(
            f"\nBackfill events complete: {len(tickers_to_process)} events processed, {total_events_saved} events saved"
        )
