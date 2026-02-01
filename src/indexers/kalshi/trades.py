"""Indexer for Kalshi trades data."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

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

    def __init__(self):
        super().__init__(
            name="kalshi_trades",
            description="Backfills Kalshi trades data to parquet files",
        )

    def run(self) -> None:
        BATCH_SIZE = 10000
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        existing_tickers = set()
        for f in DATA_DIR.glob("trades_*.parquet"):
            try:
                tickers_in_file = duckdb.sql(f"SELECT DISTINCT ticker FROM '{f}'").fetchall()
                existing_tickers.update(row[0] for row in tickers_in_file)
            except Exception:
                pass

        all_tickers = duckdb.sql(f"""
            SELECT DISTINCT ticker FROM '{MARKETS_DIR}/markets_*_*.parquet'
            WHERE volume >= 100
            ORDER BY ticker
        """).fetchall()
        all_tickers = [row[0] for row in all_tickers]
        print(f"Found {len(all_tickers)} unique markets")

        last_ticker = None
        if CURSOR_FILE.exists():
            last_ticker = CURSOR_FILE.read_text().strip() or None
            if last_ticker:
                print(f"Resuming from ticker: {last_ticker}")

        start_idx = 0
        if last_ticker and last_ticker in all_tickers:
            start_idx = all_tickers.index(last_ticker) + 1

        tickers_to_process = []
        skipped = 0
        for ticker in all_tickers[start_idx:]:
            if ticker in existing_tickers:
                skipped += 1
            else:
                tickers_to_process.append(ticker)

        print(f"Skipped {skipped} already processed, {len(tickers_to_process)} to fetch")

        all_trades = []
        total_trades_saved = 0

        def get_next_chunk_idx():
            existing = list(DATA_DIR.glob("trades_*.parquet"))
            if not existing:
                return 0
            indices = []
            for f in existing:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    try:
                        indices.append(int(parts[1]))
                    except ValueError:
                        pass
            return max(indices) + BATCH_SIZE if indices else 0

        def save_batch(trades_batch):
            nonlocal total_trades_saved
            if not trades_batch:
                return
            chunk_idx = get_next_chunk_idx()
            chunk_path = DATA_DIR / f"trades_{chunk_idx}_{chunk_idx + BATCH_SIZE}.parquet"
            df = pd.DataFrame(trades_batch)
            df.to_parquet(chunk_path)
            total_trades_saved += len(trades_batch)

        client = KalshiClient()
        pbar = tqdm(tickers_to_process, desc="Fetching trades")
        for ticker in pbar:
            try:
                trades = client.get_market_trades(ticker, verbose=False)
                if trades:
                    trades_data = [asdict(t) for t in trades]
                    fetched_at = datetime.utcnow()
                    for t in trades_data:
                        t["_fetched_at"] = fetched_at
                    all_trades.extend(trades_data)

                pbar.set_postfix(buffer=len(all_trades), saved=total_trades_saved, last=ticker[-20:])

                while len(all_trades) >= BATCH_SIZE:
                    save_batch(all_trades[:BATCH_SIZE])
                    all_trades = all_trades[BATCH_SIZE:]
                    pbar.set_postfix(buffer=len(all_trades), saved=total_trades_saved, last=ticker[-20:])

                CURSOR_FILE.write_text(ticker)
            except Exception as e:
                tqdm.write(f"Error fetching {ticker}: {e}")

        client.close()

        if all_trades:
            save_batch(all_trades)

        if CURSOR_FILE.exists():
            CURSOR_FILE.unlink()

        print(
            f"\nBackfill trades complete: {len(tickers_to_process)} markets processed, "
            f"{skipped} skipped, {total_trades_saved} trades saved"
        )
