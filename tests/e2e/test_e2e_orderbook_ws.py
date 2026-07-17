"""Orderbook scenario: Gamma token discovery, then the real websockets client
against the mock's scripted market channel - subscribe, book snapshots,
price_change deltas (including size-"0" removals and the legacy `changes`
shape), PING/PONG heartbeats, and a forced disconnect/reconnect/resubscribe."""

import asyncio
import json
import os
import threading
from pathlib import Path

import pandas as pd


def test_recorder_survives_disconnect_and_persists_raw_stream(workdir, admin, fx, wait_until):
    from src.indexers.polymarket.client import PolymarketClient
    from src.indexers.polymarket.orderbook import OrderBookRecorder, OrderBookWriter, discover_token_ids

    # Token discovery through the real Gamma client (order=volume descending).
    with PolymarketClient() as client:
        tokens = discover_token_ids(client)
    assert tokens == fx.all_featured_token_ids()

    writer = OrderBookWriter(data_dir=Path("data/polymarket/orderbook"))
    recorder = OrderBookRecorder(
        tokens,
        writer,
        ws_url=os.environ["POLYMARKET_WS_URL"],
        ping_interval=0.3,
        initial_reconnect_delay=0.2,
    )
    thread = threading.Thread(target=lambda: asyncio.run(recorder.record()), daemon=True)
    thread.start()
    try:
        # The mock closes the first connection mid-stream; wait until the
        # recorder has reconnected and heartbeated on the second one.
        wait_until(
            lambda: admin.ws_state()["connections"] >= 2 and admin.ws_state()["pings"] >= 2,
            timeout=30,
            message="reconnect plus two PING heartbeats",
        )
    finally:
        recorder.stop()
        thread.join(timeout=30)
    assert not thread.is_alive()

    state = admin.ws_state()
    assert state["connections"] == 2
    assert state["invalid_subscribes"] == 0
    # The recorder re-subscribed with the same asset IDs after the forced drop.
    assert state["subscriptions"] == [{"assets_ids": tokens, "type": "market"}] * 2

    books_dir = workdir / "data/polymarket/orderbook/books"
    books = pd.concat(pd.read_parquet(f) for f in books_dir.glob("books_*.parquet"))
    # One snapshot per token per connection: the server re-sends the full book
    # on every subscribe, which is what makes reconnects replay-safe.
    assert len(books) == 2 * len(tokens)
    assert books["token_id"].value_counts().eq(2).all()
    assert set(books["timestamp"]) == {int(fx.WS_BOOK_TS_MS)}
    assert set(books["bids"]) == {json.dumps(fx.WS_BOOK_BIDS)}
    assert set(books["asks"]) == {json.dumps(fx.WS_BOOK_ASKS)}

    deltas_dir = workdir / "data/polymarket/orderbook/price_changes"
    deltas = pd.concat(pd.read_parquet(f) for f in deltas_dir.glob("price_changes_*.parquet"))
    assert len(deltas) == fx.WS_CONN1_DELTA_ROWS + fx.WS_CONN2_DELTA_ROWS
    # Raw string prices and sizes are preserved, including "0" removals.
    assert sorted(zip(deltas["price"], deltas["size"])) == [
        ("0.44", "0"),
        ("0.46", "25"),
        ("0.47", "40"),
        ("0.55", "10"),
        ("0.56", "0"),
    ]
    legacy = deltas[deltas["hash"] == "legacy1"]
    assert len(legacy) == 1
    assert legacy.iloc[0]["token_id"] == tokens[0]
    assert legacy.iloc[0]["side"] == "SELL"
