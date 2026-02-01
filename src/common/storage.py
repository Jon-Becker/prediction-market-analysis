from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd


class ParquetStorage:
    CHUNK_SIZE = 10000

    def __init__(self, data_dir: Union[Path, str] = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_market_chunks(self) -> list[Path]:
        """Get all market chunk files sorted by start index."""
        chunks = list(self.data_dir.glob("markets_*_*.parquet"))
        chunks.sort(key=lambda p: int(p.stem.split("_")[1]))
        return chunks

    def _chunk_path(self, start: int, end: int) -> Path:
        return self.data_dir / f"markets_{start}_{end}.parquet"

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
            first_part = combined.iloc[: self.CHUNK_SIZE]
            first_part.to_parquet(last_chunk)
            remaining = combined.iloc[self.CHUNK_SIZE :]
            new_start = start + self.CHUNK_SIZE
            new_chunk_path = self._chunk_path(new_start, new_start + self.CHUNK_SIZE)
            remaining.to_parquet(new_chunk_path)

        total = sum(len(pd.read_parquet(c, columns=["ticker"])) for c in self._get_market_chunks())
        return total
