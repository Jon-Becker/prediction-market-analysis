"""Indexers for Kalshi trades data.

All three indexers share the same fetch-and-save loop and differ only in:
- which market tickers they fetch trades for (`_select_tickers`)
- which API endpoint they call (`_fetch_trades`)
- which data directories they read from and write to

Trades are buffered in memory and flushed to parquet every BATCH_SIZE trades,
so memory use stays bounded no matter how many markets are processed.
"""

from abc import abstractmethod
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
from src.indexers.kalshi.models import Trade

BATCH_SIZE = 10000

TRADES_DIR = Path("data/kalshi/trades")
MARKETS_DIR = Path("data/kalshi/markets")
HISTORICAL_TRADES_DIR = Path("data/kalshi/historical/trades")
HISTORICAL_MARKETS_DIR = Path("data/kalshi/historical/markets")


class KalshiTradesIndexerBase(Indexer):
    """Shared fetch-and-save loop for the Kalshi trades indexers."""

    trades_dir: Path = TRADES_DIR
    markets_dir: Path = MARKETS_DIR

    def __init__(
        self,
        name: str,
        description: str,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        max_workers: int = 10,
    ):
        super().__init__(name=name, description=description)
        self._min_ts = min_ts
        self._max_ts = max_ts
        self._max_workers = max_workers

        self.client = KalshiClient(rate_limit=5)

    @abstractmethod
    def _fetch_trades(self, ticker: str) -> list[Trade]:
        """Fetch all trades for one market ticker from the API."""

    def _has_existing_trades(self) -> bool:
        return any(self.trades_dir.glob("trades_*.parquet"))

    def _load_existing_tickers(self) -> set[str]:
        """Tickers that already have trades saved.

        Only tickers are loaded (not every trade_id): a market's trades are fetched
        in full, so skipping already-saved tickers is enough to avoid duplicates.
        """
        if not self._has_existing_trades():
            return set()

        print("Loading existing tickers for deduplication...")
        try:
            result = duckdb.sql(f"SELECT DISTINCT ticker FROM '{self.trades_dir}/trades_*.parquet'").fetchall()
        except duckdb.Error:
            return set()

        existing_tickers = {row[0] for row in result}
        print(f"Found {len(existing_tickers)} markets with existing trades")
        return existing_tickers

    def _select_tickers(self) -> list[str]:
        """Markets with volume >= 100 that don't have trades saved yet."""
        existing_tickers = self._load_existing_tickers()

        all_tickers = duckdb.sql(f"""
            SELECT DISTINCT ticker FROM '{self.markets_dir}/markets_*_*.parquet'
            WHERE volume >= 100
            ORDER BY ticker
        """).fetchall()
        all_tickers = [row[0] for row in all_tickers]
        print(f"Found {len(all_tickers)} unique markets")

        tickers_to_process = [t for t in all_tickers if t not in existing_tickers]
        print(
            f"Skipped {len(all_tickers) - len(tickers_to_process)} already processed, {len(tickers_to_process)} to fetch"
        )
        return tickers_to_process

    def _next_chunk_index(self) -> int:
        """Start index for the next parquet file, based on files already saved."""
        indices = []
        for f in self.trades_dir.glob("trades_*.parquet"):
            parts = f.stem.split("_")
            if len(parts) >= 2:
                try:
                    indices.append(int(parts[1]))
                except ValueError:
                    pass
        return max(indices) + BATCH_SIZE if indices else 0

    def run(self) -> None:
        self.trades_dir.mkdir(parents=True, exist_ok=True)

        tickers_to_process = self._select_tickers()
        if not tickers_to_process:
            print("Nothing to process")
            return

        all_trades: list[dict] = []
        total_trades_saved = 0
        next_chunk_idx = self._next_chunk_index()

        def save_batch(trades_batch: list[dict]) -> int:
            nonlocal next_chunk_idx
            if not trades_batch:
                return 0
            chunk_path = self.trades_dir / f"trades_{next_chunk_idx}_{next_chunk_idx + BATCH_SIZE}.parquet"
            df = pd.DataFrame(trades_batch)
            df.to_parquet(chunk_path)
            next_chunk_idx += BATCH_SIZE
            return len(trades_batch)

        def fetch_ticker_trades(ticker: str) -> list[dict]:
            """Fetch trades for a single ticker."""
            trades = self._fetch_trades(ticker)
            if not trades:
                return []
            fetched_at = datetime.utcnow()
            return [{**asdict(t), "_fetched_at": fetched_at} for t in trades]

        # Concurrent fetching
        pbar = tqdm(total=len(tickers_to_process), desc="Fetching trades")
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(fetch_ticker_trades, ticker): ticker for ticker in tickers_to_process}

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    trades_data = future.result()
                    if trades_data:
                        all_trades.extend(trades_data)

                    pbar.update(1)
                    pbar.set_postfix(buffer=len(all_trades), saved=total_trades_saved, last=ticker[-20:])

                    # Save in batches so the buffer never grows past BATCH_SIZE
                    while len(all_trades) >= BATCH_SIZE:
                        saved = save_batch(all_trades[:BATCH_SIZE])
                        total_trades_saved += saved
                        all_trades = all_trades[BATCH_SIZE:]

                except Exception as e:
                    pbar.update(1)
                    tqdm.write(f"Error fetching {ticker}: {e}")
                # Drop the finished future so its result can be freed
                futures.pop(future, None)
        pbar.close()

        self.client.close()
        # Save remaining
        if all_trades:
            total_trades_saved += save_batch(all_trades)

        print(
            f"\nBackfill trades complete: {len(tickers_to_process)} markets processed, {total_trades_saved} trades saved"
        )


class KalshiTradesIndexer(KalshiTradesIndexerBase):
    """Fetches trades for every recent market that doesn't have trades saved yet."""

    def __init__(self, min_ts: Optional[int] = None, max_ts: Optional[int] = None, max_workers: int = 10):
        super().__init__(
            name="kalshi_trades",
            description="Backfills Kalshi trades data to parquet files",
            min_ts=min_ts,
            max_ts=max_ts,
            max_workers=max_workers,
        )

    def _fetch_trades(self, ticker: str) -> list[Trade]:
        return self.client.get_market_trades(ticker, verbose=False, min_ts=self._min_ts, max_ts=self._max_ts)


class KalshiFinalizedTradesIndexer(KalshiTradesIndexer):
    """Fetches trades only for finalized (resolved yes/no) markets missing trades.

    Useful for topping up after new markets have been indexed, since these are
    the only markets the analyses use.
    """

    def __init__(self, min_ts: Optional[int] = None, max_ts: Optional[int] = None, max_workers: int = 10):
        KalshiTradesIndexerBase.__init__(
            self,
            name="kalshi_trades_finalized",
            description="Backfills Kalshi trades for finalized markets that have no trades saved yet",
            min_ts=min_ts,
            max_ts=max_ts,
            max_workers=max_workers,
        )

    def _select_tickers(self) -> list[str]:
        missing_trades_filter = ""
        if self._has_existing_trades():
            missing_trades_filter = (
                f"AND ticker NOT IN (SELECT DISTINCT ticker FROM '{self.trades_dir}/trades_*.parquet')"
            )

        tickers = duckdb.sql(f"""
            SELECT DISTINCT ticker
            FROM '{self.markets_dir}/markets_*.parquet'
            WHERE volume > 100
            AND status = 'finalized'
            AND result IN ('yes', 'no')
            {missing_trades_filter}
        """).fetchall()

        tickers = [row[0] for row in tickers]
        print(f"Found {len(tickers)} finalized markets without trades")
        return tickers


class KalshiHistoricalTradesIndexer(KalshiTradesIndexerBase):
    """Fetches trades for historical markets from Kalshi's /historical/trades endpoint."""

    trades_dir = HISTORICAL_TRADES_DIR
    markets_dir = HISTORICAL_MARKETS_DIR

    def __init__(self, min_ts: Optional[int] = None, max_ts: Optional[int] = None, max_workers: int = 10):
        super().__init__(
            name="kalshi_historical_trades",
            description="Backfills historical Kalshi trades data to parquet files",
            min_ts=min_ts,
            max_ts=max_ts,
            max_workers=max_workers,
        )

    def _fetch_trades(self, ticker: str) -> list[Trade]:
        return self.client.get_historical_trades(ticker, verbose=False, min_ts=self._min_ts, max_ts=self._max_ts)
