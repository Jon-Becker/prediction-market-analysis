"""Indexer for Kalshi trades data."""

import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.kalshi.client import KalshiClient

DATA_DIR = Path("data/kalshi/trades")
MARKETS_DIR = Path("data/kalshi/markets")
CURSOR_FILE = Path("data/kalshi/.backfill_trades_cursor")


class KalshiTradesIndexer(Indexer):
    """Fetches and stores Kalshi trades data."""

    def __init__(
        self,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        max_workers: int = 10,
    ):
        super().__init__(
            name="kalshi_trades",
            description="Backfills Kalshi trades data to parquet files",
        )
        self._min_ts = min_ts
        self._max_ts = max_ts
        self._max_workers = max_workers

    def run(self) -> None:
        BATCH_SIZE = 10000
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load existing tickers for deduplication (small, fits OK into memory)
        existing_tickers: set[str] = set()
        parquet_files = list(DATA_DIR.glob("trades_*.parquet"))
        if parquet_files:
            print("Loading existing tickers for deduplication...")
            try:
                existing_tickers = {
                    row[0]
                    for row in duckdb.sql(f"SELECT DISTINCT ticker FROM '{DATA_DIR}/trades_*.parquet'").fetchall()
                }
                print(f"Found {len(existing_tickers)} existing tickers")
            except Exception:
                traceback.print_exc()

        all_tickers = duckdb.sql(f"""
            SELECT DISTINCT ticker FROM '{MARKETS_DIR}/markets_*_*.parquet'
            WHERE volume >= 100
            ORDER BY ticker
        """).fetchall()
        all_tickers = [row[0] for row in all_tickers]
        print(f"Found {len(all_tickers)} unique markets")

        # Filter to tickers not fully processed
        tickers_to_process = [t for t in all_tickers if t not in existing_tickers]
        del existing_tickers  # free some RAM

        print(
            f"Skipped {len(all_tickers) - len(tickers_to_process)} already processed, "
            f"{len(tickers_to_process)} to fetch"
        )

        all_trades: list[dict] = []
        total_trades_saved = 0
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

        def save_batch(trades_batch: list[dict]) -> int:
            nonlocal next_chunk_idx
            if not trades_batch:
                return 0
            chunk_path = DATA_DIR / f"trades_{next_chunk_idx}_{next_chunk_idx + BATCH_SIZE}.parquet"
            df = pd.DataFrame(trades_batch)
            df.to_parquet(chunk_path)
            next_chunk_idx += BATCH_SIZE
            return len(trades_batch)

        def fetch_ticker_trades(ticker: str) -> tuple[str, Optional[list[dict]]]:
            """Fetch trades for a single ticker."""
            client = KalshiClient()
            try:
                trades = client.get_market_trades(
                    ticker,
                    verbose=False,
                    min_ts=self._min_ts,
                    max_ts=self._max_ts,
                )
                if not trades:
                    return ticker, []
                fetched_at = datetime.utcnow()
                return ticker, [{**asdict(t), "_fetched_at": fetched_at} for t in trades]
            except Exception as e:
                tqdm.write(f"Error fetching {ticker}: {e}")
                return ticker, None
            finally:
                client.close()

        MAX_PENDING = self._max_workers * 2  # Tune as needed
        pending = set()
        tickers_iter = iter(tickers_to_process)
        pbar = tqdm(total=len(tickers_to_process), desc="Fetching trades")

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit initial futures
            for _ in range(min(MAX_PENDING, len(tickers_to_process))):
                ticker = next(tickers_iter)
                future = executor.submit(fetch_ticker_trades, ticker)
                pending.add(future)

            while pending:
                # Wait for at least one future to complete
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    ticker, trades_data = future.result()
                    if trades_data:  # Handles both error and empty result
                        all_trades.extend(trades_data)

                    pbar.set_postfix(buffer=len(all_trades), saved=total_trades_saved, last=ticker[-20:])

                    # Save in batches
                    while len(all_trades) >= BATCH_SIZE:
                        saved = save_batch(all_trades[:BATCH_SIZE])
                        total_trades_saved += saved
                        all_trades = list(all_trades[BATCH_SIZE:])
                    pbar.update(1)

                # Submit new futures to replace the completed ones
                for _ in range(len(done)):
                    try:
                        ticker = next(tickers_iter)
                        pending.add(executor.submit(fetch_ticker_trades, ticker))
                    except StopIteration:
                        break

        pbar.close()

        # Save remaining
        if all_trades:
            total_trades_saved += save_batch(all_trades)

        print(
            f"\nBackfill trades complete: {len(tickers_to_process)} markets processed, "
            f"{total_trades_saved} trades saved"
        )
        self._deduplicate_trades()

    def _deduplicate_trades(self) -> None:
        parquet_files = list(DATA_DIR.glob("trades_*.parquet"))
        # It can be either empty or contain one file which should not contain duplicates
        if len(parquet_files) <= 1:
            return

        print("Deduplicating all trade data...")
        temp_file = DATA_DIR / "trades_dedup_temp.parquet"
        try:
            duckdb.sql(f"""
                COPY (
                    SELECT DISTINCT ON (trade_id) *
                    FROM '{DATA_DIR}/trades_*.parquet'
                ) TO '{temp_file}' (FORMAT 'parquet')
            """)

            temp_file.rename(DATA_DIR / "trades_all.parquet")

            for f in parquet_files:
                f.unlink()

            print(f"Deduplicated trades saved to {DATA_DIR}/trades_all.parquet")
        except BaseException as e:
            print(f"Error during deduplication: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
