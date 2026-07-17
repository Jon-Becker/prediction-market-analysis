"""Tests for the live orderbook recorder against a mocked market WebSocket."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from src.indexers.polymarket.models import OrderBookDelta
from src.indexers.polymarket.orderbook import OrderBookRecorder, OrderBookWriter

CONDITION_ID = "0x" + "ab" * 32

BOOK_MSG = {
    "event_type": "book",
    "asset_id": "111",
    "market": CONDITION_ID,
    "bids": [{"price": "0.48", "size": "30"}, {"price": "0.49", "size": "20"}],
    "asks": [{"price": "0.52", "size": "25"}],
    "timestamp": "1784254990511",
    "hash": "0x1234",
}

PRICE_CHANGE_MSG = {
    "event_type": "price_change",
    "market": CONDITION_ID,
    "price_changes": [
        {
            "asset_id": "111",
            "price": "0.5",
            "size": "200",
            "side": "BUY",
            "hash": "56621a121a47ed93",
            "best_bid": "0.5",
            "best_ask": "0.52",
        },
        {
            "asset_id": "111",
            "price": "0.52",
            "size": "0",
            "side": "SELL",
            "hash": "77731b232b58fe04",
            "best_bid": "0.5",
            "best_ask": "0.53",
        },
    ],
    "timestamp": "1784254991000",
}


class FakeSocket:
    """Mocked market-channel socket: yields scripted frames and records sends."""

    def __init__(self, frames: list[str] = (), error: Exception | None = None, on_drained=None):
        self.frames = list(frames)
        self.error = error
        self.on_drained = on_drained
        self.sent: list[str] = []
        self._hold = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.frames:
            return self.frames.pop(0)
        if self.error is not None:
            raise self.error
        if self.on_drained is not None:
            self.on_drained()
            raise StopAsyncIteration
        await self._hold.wait()
        raise StopAsyncIteration

    async def send(self, data: str):
        self.sent.append(data)

    def release(self):
        self._hold.set()


def read_table(data_dir: Path, table: str) -> pd.DataFrame:
    files = sorted((data_dir / table).glob("*.parquet"))
    assert files, f"no parquet chunks written for {table}"
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# -- Model parsing -------------------------------------------------------------


def test_delta_list_from_message_preserves_raw_strings():
    deltas = OrderBookDelta.list_from_message(PRICE_CHANGE_MSG)

    assert len(deltas) == 2
    assert deltas[0].condition_id == CONDITION_ID
    assert deltas[0].token_id == "111"
    assert deltas[0].timestamp == 1784254991000
    assert deltas[0].hash == "56621a121a47ed93"
    assert (deltas[0].side, deltas[0].price, deltas[0].size) == ("BUY", "0.5", "200")
    assert (deltas[0].best_bid, deltas[0].best_ask) == ("0.5", "0.52")
    assert (deltas[1].side, deltas[1].price, deltas[1].size) == ("SELL", "0.52", "0")


def test_delta_list_from_legacy_changes_shape():
    message = {
        "event_type": "price_change",
        "market": CONDITION_ID,
        "asset_id": "222",
        "hash": "abcd",
        "changes": [{"price": "0.4", "side": "SELL", "size": "3300"}],
        "timestamp": "1729084877448",
    }

    deltas = OrderBookDelta.list_from_message(message)

    assert len(deltas) == 1
    assert deltas[0].token_id == "222"
    assert deltas[0].hash == "abcd"
    assert (deltas[0].side, deltas[0].price, deltas[0].size) == ("SELL", "0.4", "3300")
    assert deltas[0].best_bid == ""


# -- Message handling & persistence ---------------------------------------------


def test_book_snapshot_persists_raw_levels_and_hash(tmp_path: Path):
    writer = OrderBookWriter(data_dir=tmp_path)
    recorder = OrderBookRecorder(["111"], writer)

    recorder.handle_message(json.dumps(BOOK_MSG))
    writer.flush()

    books = read_table(tmp_path, "books")
    assert len(books) == 1
    row = books.iloc[0]
    assert row["token_id"] == "111"
    assert row["condition_id"] == CONDITION_ID
    assert row["timestamp"] == 1784254990511  # exchange milliseconds
    assert row["hash"] == "0x1234"
    assert json.loads(row["bids"]) == BOOK_MSG["bids"]  # raw levels, string prices intact
    assert json.loads(row["asks"]) == BOOK_MSG["asks"]
    assert row["_recorded_at"] is not None


def test_price_change_deltas_persist_including_size_zero_removal(tmp_path: Path):
    writer = OrderBookWriter(data_dir=tmp_path)
    recorder = OrderBookRecorder(["111"], writer)

    recorder.handle_message(json.dumps(PRICE_CHANGE_MSG))
    writer.flush()

    changes = read_table(tmp_path, "price_changes")
    assert len(changes) == 2
    assert list(changes["side"]) == ["BUY", "SELL"]
    assert list(changes["size"]) == ["200", "0"]  # "0" = level removed
    assert list(changes["price"]) == ["0.5", "0.52"]
    assert list(changes["best_bid"]) == ["0.5", "0.5"]
    assert list(changes["best_ask"]) == ["0.52", "0.53"]
    assert set(changes["timestamp"]) == {1784254991000}


def test_non_json_and_unknown_events_are_ignored(tmp_path: Path):
    writer = OrderBookWriter(data_dir=tmp_path)
    recorder = OrderBookRecorder(["111"], writer)

    recorder.handle_message("PONG")
    recorder.handle_message(json.dumps({"event_type": "tick_size_change", "asset_id": "111"}))
    writer.flush()

    assert not (tmp_path / "books").exists()
    assert not (tmp_path / "price_changes").exists()


def test_writer_auto_flushes_at_chunk_size(tmp_path: Path):
    writer = OrderBookWriter(data_dir=tmp_path, chunk_size=2)
    recorder = OrderBookRecorder(["111"], writer)

    recorder.handle_message(json.dumps(BOOK_MSG))
    assert not (tmp_path / "books").exists()
    recorder.handle_message(json.dumps(BOOK_MSG))

    files = list((tmp_path / "books").glob("*.parquet"))
    assert len(files) == 1
    assert files[0].name == "books_1784254990511_1784254990511.parquet"
    assert len(pd.read_parquet(files[0])) == 2


# -- Live recording over a mocked socket ------------------------------------------


def test_record_subscribes_and_persists_stream(tmp_path: Path):
    writer = OrderBookWriter(data_dir=tmp_path)

    async def main() -> FakeSocket:
        # a real feed wraps some frames in a JSON array; both shapes must record
        socket = FakeSocket(
            frames=[json.dumps([BOOK_MSG]), json.dumps(PRICE_CHANGE_MSG)],
            on_drained=lambda: recorder.stop(),
        )
        recorder = OrderBookRecorder(["111", "222"], writer, connect=lambda: socket)
        await recorder.record()
        return socket

    socket = asyncio.run(main())

    assert json.loads(socket.sent[0]) == {"assets_ids": ["111", "222"], "type": "market"}
    assert len(read_table(tmp_path, "books")) == 1
    assert len(read_table(tmp_path, "price_changes")) == 2


def test_record_sends_ping_heartbeats(tmp_path: Path):
    writer = OrderBookWriter(data_dir=tmp_path)

    async def main() -> FakeSocket:
        socket = FakeSocket()
        recorder = OrderBookRecorder(["111"], writer, ping_interval=0.01, connect=lambda: socket)
        task = asyncio.create_task(recorder.record())
        await asyncio.sleep(0.06)
        recorder.stop()
        socket.release()
        await task
        return socket

    socket = asyncio.run(main())

    assert json.loads(socket.sent[0])["type"] == "market"
    assert socket.sent.count("PING") >= 2


def test_record_reconnects_and_resubscribes_after_drop(tmp_path: Path):
    writer = OrderBookWriter(data_dir=tmp_path)

    async def main() -> list[FakeSocket]:
        sockets: list[FakeSocket] = []
        recorder = OrderBookRecorder(
            ["111"],
            writer,
            initial_reconnect_delay=0,
            connect=lambda: sockets.pop(0),
        )
        sockets.append(FakeSocket(frames=[json.dumps(BOOK_MSG)], error=RuntimeError("connection dropped")))
        second = FakeSocket(frames=[json.dumps(BOOK_MSG)], on_drained=lambda: recorder.stop())
        sockets.append(second)
        first = sockets[0]
        await recorder.record()
        return [first, second]

    first, second = asyncio.run(main())

    # both connections must (re)subscribe; the server answers each with a fresh snapshot
    assert json.loads(first.sent[0]) == {"assets_ids": ["111"], "type": "market"}
    assert json.loads(second.sent[0]) == {"assets_ids": ["111"], "type": "market"}
    assert len(read_table(tmp_path, "books")) == 2
