from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Union

import pandas as pd


class ParquetStorage:
    CHUNK_SIZE = 10000

    def __init__(self, data_dir: Union[Path, str] = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_market_chunks(self) -> List[Path]:
        """Get all market chunk files sorted by start index."""
        chunks = list(self.data_dir.glob("markets_*_*.parquet"))
        chunks.sort(key=lambda p: int(p.stem.split("_")[1]))
        return chunks

    def _chunk_path(self, start: int, end: int) -> Path:
        return self.data_dir / f"markets_{start}_{end}.parquet"

    def save_market(self, market, trades=None):
        ticker = market.ticker
        market_dict = asdict(market)
        market_dict["_fetched_at"] = datetime.utcnow()

        new_row = pd.json_normalize(market_dict)
        chunks = self._get_market_chunks()

        if not chunks:
            chunk_path = self._chunk_path(0, self.CHUNK_SIZE)
            new_row.to_parquet(chunk_path)
        else:
            for chunk_path in chunks:
                df = pd.read_parquet(chunk_path)
                if ticker in df["ticker"].values:
                    df = df[df["ticker"] != ticker]
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_parquet(chunk_path)
                    break
            else:
                last_chunk = chunks[-1]
                last_df = pd.read_parquet(last_chunk)
                if len(last_df) < self.CHUNK_SIZE:
                    last_df = pd.concat([last_df, new_row], ignore_index=True)
                    last_df.to_parquet(last_chunk)
                else:
                    start = int(last_chunk.stem.split("_")[1]) + self.CHUNK_SIZE
                    new_chunk = self._chunk_path(start, start + self.CHUNK_SIZE)
                    new_row.to_parquet(new_chunk)
        print(f"Saved market {ticker}")

        if trades:
            trades_data = []
            for trade in trades:
                trade_dict = asdict(trade)
                trade_dict["_fetched_at"] = datetime.utcnow()
                trades_data.append(trade_dict)

            if trades_data:
                trades_df = pd.DataFrame(trades_data)
                trades_path = self.data_dir / f"{ticker}_trades.parquet"
                trades_df.to_parquet(trades_path)
                print(f"Saved {len(trades_data)} trades to {trades_path}")

        return ticker

    def load_markets(self) -> pd.DataFrame:
        chunks = self._get_market_chunks()
        if not chunks:
            return pd.DataFrame()
        dfs = [pd.read_parquet(chunk) for chunk in chunks]
        return pd.concat(dfs, ignore_index=True)

    def load_market(self, ticker: str) -> pd.DataFrame:
        markets = self.load_markets()
        return markets[markets["ticker"] == ticker]

    def load_trades(self, ticker: str) -> pd.DataFrame:
        trades_path = self.data_dir / f"{ticker}_trades.parquet"
        return pd.read_parquet(trades_path)

    def list_tickers(self) -> List[str]:
        chunks = self._get_market_chunks()
        if not chunks:
            return []
        tickers = []
        for chunk in chunks:
            df = pd.read_parquet(chunk, columns=["ticker"])
            tickers.extend(df["ticker"].tolist())
        return tickers

    def save_markets(self, markets: list) -> None:
        fetched_at = datetime.utcnow()
        records = []
        for market in markets:
            record = asdict(market)
            record["_fetched_at"] = fetched_at
            records.append(record)

        df = pd.DataFrame(records)
        for i in range(0, len(df), self.CHUNK_SIZE):
            chunk = df.iloc[i:i + self.CHUNK_SIZE]
            chunk_path = self._chunk_path(i, i + self.CHUNK_SIZE)
            chunk.to_parquet(chunk_path)
        print(f"Saved {len(markets)} markets across {(len(df) - 1) // self.CHUNK_SIZE + 1} chunks")

    def append_markets(self, markets: list) -> int:
        fetched_at = datetime.utcnow()
        records = []
        for market in markets:
            record = asdict(market)
            record["_fetched_at"] = fetched_at
            records.append(record)

        new_df = pd.DataFrame(records)
        chunks = self._get_market_chunks()

        if not chunks:
            chunk_path = self._chunk_path(0, self.CHUNK_SIZE)
            new_df.to_parquet(chunk_path)
            return len(new_df)

        last_chunk = chunks[-1]
        last_df = pd.read_parquet(last_chunk)
        new_tickers = set(new_df["ticker"])
        last_df = last_df[~last_df["ticker"].isin(new_tickers)]
        combined = pd.concat([last_df, new_df], ignore_index=True)

        start = int(last_chunk.stem.split("_")[1])
        if len(combined) <= self.CHUNK_SIZE:
            combined.to_parquet(last_chunk)
        else:
            first_part = combined.iloc[:self.CHUNK_SIZE]
            first_part.to_parquet(last_chunk)
            remaining = combined.iloc[self.CHUNK_SIZE:]
            new_start = start + self.CHUNK_SIZE
            new_chunk_path = self._chunk_path(new_start, new_start + self.CHUNK_SIZE)
            remaining.to_parquet(new_chunk_path)

        total = sum(len(pd.read_parquet(c, columns=["ticker"])) for c in self._get_market_chunks())
        return total
