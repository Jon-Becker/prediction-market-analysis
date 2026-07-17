"""Live recorder for the Polymarket CLOB market WebSocket channel.

Polymarket exposes no historical order-book endpoint: depth data cannot be
fetched retroactively and only exists for the periods this recorder was
running. It subscribes to the market channel by CLOB token IDs, persists
full `book` snapshots and `price_change` deltas with their raw string
levels and integrity hashes (millisecond exchange timestamps preserved),
sends the required `PING` heartbeat every 10 seconds, and reconnects with
backoff on disconnect. Reconnects are safe for replay because the server
re-sends a full `book` snapshot for every subscribed token on subscribe,
superseding any deltas missed while offline.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import websockets

from src.common.indexer import Indexer
from src.indexers.polymarket.client import PolymarketClient
from src.indexers.polymarket.models import OrderBookDelta

WS_MARKET_URL = os.getenv("POLYMARKET_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market")
DATA_DIR = Path("data/polymarket/orderbook")
CHUNK_SIZE = 10000
PING_INTERVAL_SECONDS = 10.0
INITIAL_RECONNECT_DELAY_SECONDS = 1.0
MAX_RECONNECT_DELAY_SECONDS = 60.0
DISCOVER_MARKET_COUNT = 100


def discover_token_ids(client: PolymarketClient, max_markets: int = DISCOVER_MARKET_COUNT) -> list[str]:
    """Resolve CLOB token IDs for the highest-volume open markets on Gamma."""
    markets = client.get_markets(limit=max_markets, active=True, closed=False, order="volume", ascending=False)
    token_ids: list[str] = []
    for market in markets:
        try:
            ids = json.loads(market.clob_token_ids)
        except (TypeError, ValueError):
            continue
        token_ids.extend(str(token_id) for token_id in ids if token_id)
    return list(dict.fromkeys(token_ids))


class OrderBookWriter:
    """Buffers recorded rows and flushes them to timestamp-ranged parquet chunks."""

    def __init__(self, data_dir: Path = DATA_DIR, chunk_size: int = CHUNK_SIZE):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self._buffers: dict[str, list[dict]] = {"books": [], "price_changes": []}

    def add_book(self, row: dict) -> None:
        self._append("books", [row])

    def add_price_changes(self, rows: list[dict]) -> None:
        self._append("price_changes", rows)

    def flush(self) -> None:
        for table in self._buffers:
            self._flush_table(table)

    def _append(self, table: str, rows: list[dict]) -> None:
        self._buffers[table].extend(rows)
        if len(self._buffers[table]) >= self.chunk_size:
            self._flush_table(table)

    def _flush_table(self, table: str) -> None:
        buffer = self._buffers[table]
        if not buffer:
            return
        directory = self.data_dir / table
        directory.mkdir(parents=True, exist_ok=True)
        start, end = buffer[0]["timestamp"], buffer[-1]["timestamp"]
        path = directory / f"{table}_{start}_{end}.parquet"
        suffix = 1
        while path.exists():
            path = directory / f"{table}_{start}_{end}_{suffix}.parquet"
            suffix += 1
        pd.DataFrame(buffer).to_parquet(path)
        self._buffers[table] = []
        print(f"Flushed {len(buffer)} rows to {path}")


class OrderBookRecorder:
    """Consumes the CLOB market channel and persists book snapshots and deltas."""

    def __init__(
        self,
        token_ids: list[str],
        writer: OrderBookWriter,
        ws_url: str = WS_MARKET_URL,
        ping_interval: float = PING_INTERVAL_SECONDS,
        initial_reconnect_delay: float = INITIAL_RECONNECT_DELAY_SECONDS,
        connect: Callable | None = None,
    ):
        self.token_ids = list(token_ids)
        self.writer = writer
        self.ws_url = ws_url
        self.ping_interval = ping_interval
        self.initial_reconnect_delay = initial_reconnect_delay
        self._connect = connect or (lambda: websockets.connect(self.ws_url))
        self._stop: asyncio.Event | None = None

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()

    async def record(self) -> None:
        """Connect, subscribe, and record until stopped; reconnects on disconnect."""
        self._stop = asyncio.Event()
        delay = self.initial_reconnect_delay
        try:
            while not self._stop.is_set():
                try:
                    async with self._connect() as ws:
                        await self._subscribe(ws)
                        ping_task = asyncio.create_task(self._ping_loop(ws))
                        try:
                            async for raw in ws:
                                self.handle_message(raw)
                                delay = self.initial_reconnect_delay
                                if self._stop.is_set():
                                    break
                        finally:
                            ping_task.cancel()
                            with contextlib.suppress(Exception, asyncio.CancelledError):
                                await ping_task
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"WebSocket error: {exc}")
                if self._stop.is_set():
                    break
                print(f"Disconnected; reconnecting in {delay:.0f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY_SECONDS)
        finally:
            self.writer.flush()

    def handle_message(self, raw: str) -> None:
        """Parse one WebSocket frame and persist any book/price_change events."""
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return  # PONG heartbeat replies and other non-JSON frames
        messages = payload if isinstance(payload, list) else [payload]
        for message in messages:
            if not isinstance(message, dict):
                continue
            event_type = message.get("event_type")
            if event_type == "book":
                self.writer.add_book(self._book_row(message))
            elif event_type == "price_change":
                deltas = OrderBookDelta.list_from_message(message)
                if deltas:
                    recorded_at = datetime.now(timezone.utc)
                    self.writer.add_price_changes([{**asdict(delta), "_recorded_at": recorded_at} for delta in deltas])

    @staticmethod
    def _book_row(message: dict) -> dict:
        return {
            "token_id": message.get("asset_id", ""),
            "condition_id": message.get("market", ""),
            "timestamp": int(message.get("timestamp", 0) or 0),
            "hash": message.get("hash", ""),
            "bids": json.dumps(message.get("bids") or []),
            "asks": json.dumps(message.get("asks") or []),
            "_recorded_at": datetime.now(timezone.utc),
        }

    async def _subscribe(self, ws) -> None:
        await ws.send(json.dumps({"assets_ids": self.token_ids, "type": "market"}))

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(self.ping_interval)
            await ws.send("PING")


class PolymarketOrderbookRecorder(Indexer):
    """Records live Polymarket order-book snapshots and price changes."""

    def __init__(self, token_ids: list[str] | None = None, max_markets: int = DISCOVER_MARKET_COUNT):
        super().__init__(
            name="polymarket_orderbook",
            description="Records live CLOB order-book snapshots and price changes to parquet files",
        )
        self.token_ids = list(token_ids) if token_ids else None
        self.max_markets = max_markets

    def run(self) -> None:
        token_ids = self.token_ids
        if not token_ids:
            with PolymarketClient() as client:
                token_ids = discover_token_ids(client, self.max_markets)
        if not token_ids:
            print("No CLOB token IDs to record")
            return

        print(f"Recording order books for {len(token_ids)} tokens (Ctrl+C to stop)")
        writer = OrderBookWriter()
        recorder = OrderBookRecorder(token_ids, writer)
        try:
            asyncio.run(recorder.record())
        except KeyboardInterrupt:
            writer.flush()
            print("\nRecording stopped")
