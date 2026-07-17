"""Tests for the Data API trades indexer: market fan-out, splitting, resume, dedup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.indexers.polymarket.client import FULL_HISTORY_START
from src.indexers.polymarket.models import DataApiTrade


def make_trade(timestamp: int, tx: str = "0xabc", asset: str = "111", **kwargs) -> DataApiTrade:
    fields = {
        "proxy_wallet": "0x01",
        "side": "BUY",
        "asset": asset,
        "condition_id": "0xcc",
        "size": 1.0,
        "price": 0.5,
        "timestamp": timestamp,
        "transaction_hash": tx,
    }
    fields.update(kwargs)
    return DataApiTrade(**fields)


class FakeClient:
    """Stub for PolymarketClient serving a per-market tape of trades."""

    def __init__(
        self,
        tapes: dict[str, list[DataApiTrade]],
        *,
        overflow_threshold: int | None = None,
        interrupt_at_market: str | None = None,
        interrupt_at_window: int | None = None,
    ):
        self.tapes = {cid: sorted(tape, key=lambda t: t.timestamp) for cid, tape in tapes.items()}
        self.overflow_threshold = overflow_threshold
        self.interrupt_at_market = interrupt_at_market
        self.interrupt_at_window = interrupt_at_window
        self.requested: list[tuple[str, int, int]] = []
        self.taker_only_args: list[bool] = []

    def get_data_trades_window(self, start, end, limit=1000, taker_only=True, market=None):
        cids = market.split(",") if market else []
        if self.interrupt_at_market is not None and self.interrupt_at_market in cids:
            raise KeyboardInterrupt
        if self.interrupt_at_window is not None and start >= self.interrupt_at_window:
            raise KeyboardInterrupt
        self.requested.append((market, start, end))
        self.taker_only_args.append(taker_only)
        rows = sorted(
            (t for cid in cids for t in self.tapes.get(cid, []) if start <= t.timestamp <= end),
            key=lambda t: t.timestamp,
        )
        if self.overflow_threshold is not None and len(rows) > self.overflow_threshold:
            return rows[: self.overflow_threshold], True
        return rows, False

    def close(self):
        pass


@pytest.fixture()
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point DATA_DIR, MARKETS_DIR, and CURSOR_FILE at temp directories."""
    data_dir = tmp_path / "data_api_trades"
    markets_dir = tmp_path / "markets"
    cursor_file = tmp_path / ".data_api_trades_cursor"
    import src.indexers.polymarket.data_api_trades as mod

    monkeypatch.setattr(mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(mod, "MARKETS_DIR", markets_dir)
    monkeypatch.setattr(mod, "CURSOR_FILE", cursor_file)
    return data_dir, markets_dir, cursor_file


def write_markets(markets_dir: Path, condition_ids: list[str]):
    markets_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"condition_id": condition_ids})
    df.to_parquet(markets_dir / f"markets_0_{len(condition_ids)}.parquet")


def read_all(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("trades_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run_indexer(client: FakeClient, **kwargs):
    from src.indexers.polymarket.data_api_trades import PolymarketDataApiTradesIndexer

    with patch("src.indexers.polymarket.data_api_trades.PolymarketClient", return_value=client):
        PolymarketDataApiTradesIndexer(**kwargs).run()


def test_markets_are_batched_and_all_trades_saved(isolated_dirs, monkeypatch):
    import src.indexers.polymarket.data_api_trades as mod

    monkeypatch.setattr(mod, "BATCH_SIZE", 2)
    data_dir, markets_dir, cursor_file = isolated_dirs
    # Duplicate and empty condition IDs in the dataset must be ignored
    write_markets(markets_dir, ["0xa", "0xb", "0xa", "", "0xc"])
    client = FakeClient(
        {
            "0xa": [make_trade(100, tx="0x100"), make_trade(150, tx="0x150")],
            "0xb": [make_trade(199, tx="0x199"), make_trade(200, tx="0x200")],
            "0xc": [make_trade(299, tx="0x299")],
        }
    )

    run_indexer(client, start=100, end=299, markets_per_request=2)

    assert client.requested == [("0xa,0xb", 100, 299), ("0xc", 100, 299)]
    assert all(client.taker_only_args), "takerOnly must be sent explicitly as True"

    df = read_all(data_dir)
    assert sorted(df["timestamp"]) == [100, 150, 199, 200, 299]
    assert "_fetched_at" in df.columns
    assert df["asset"].map(type).eq(str).all(), "token IDs must be stored as strings"

    # Chunked in BATCH_SIZE files: 2 + 2 + 1 rows
    files = sorted(data_dir.glob("trades_*.parquet"))
    assert [f.name for f in files] == ["trades_0_2.parquet", "trades_2_4.parquet", "trades_4_6.parquet"]
    assert not cursor_file.exists(), "Cursor should be removed after clean completion"


def test_default_start_is_full_history(isolated_dirs):
    _, markets_dir, _ = isolated_dirs
    write_markets(markets_dir, ["0xa"])
    client = FakeClient({"0xa": [make_trade(50)]})

    run_indexer(client, end=200)

    assert client.requested == [("0xa", FULL_HISTORY_START, 200)]


def test_missing_markets_dataset_is_graceful(isolated_dirs, capsys):
    data_dir, _, _ = isolated_dirs
    client = FakeClient({})

    run_indexer(client)

    assert "run the polymarket_markets indexer first" in capsys.readouterr().out
    assert not data_dir.exists()
    assert client.requested == []


def test_overflowing_window_is_split_until_it_fits(isolated_dirs):
    data_dir, markets_dir, _ = isolated_dirs
    write_markets(markets_dir, ["0xa"])
    trades = [make_trade(ts, tx=f"0x{ts}") for ts in [100, 101, 102, 103, 104, 105]]
    client = FakeClient({"0xa": trades}, overflow_threshold=3)

    run_indexer(client, start=100, end=199)

    # The full window overflows the offset cap and must be split recursively
    assert ("0xa", 100, 199) in client.requested
    assert len(client.requested) > 1

    df = read_all(data_dir)
    assert sorted(df["timestamp"]) == [100, 101, 102, 103, 104, 105], "every trade must be recovered via splits"


def test_overflowing_single_second_window_keeps_partial_rows(isolated_dirs, capsys):
    data_dir, markets_dir, _ = isolated_dirs
    write_markets(markets_dir, ["0xa"])
    # 5 trades in the same second cannot be separated by timestamp windows
    trades = [make_trade(100, tx=f"0x{i}") for i in range(5)]
    client = FakeClient({"0xa": trades}, overflow_threshold=3)

    run_indexer(client, start=100, end=100)

    df = read_all(data_dir)
    assert len(df) == 3, "rows up to the cap are kept for an unsplittable window"
    assert "overflows the offset cap" in capsys.readouterr().out


def test_interrupt_then_resume_skips_completed_markets(isolated_dirs):
    data_dir, markets_dir, cursor_file = isolated_dirs
    write_markets(markets_dir, ["0xa", "0xb", "0xc"])
    tapes = {
        "0xa": [make_trade(100, tx="0x100"), make_trade(150, tx="0x150")],
        "0xb": [make_trade(250, tx="0x250")],
        "0xc": [make_trade(350, tx="0x350")],
    }

    # Run 1: interrupt when market 0xb is requested
    run_indexer(FakeClient(tapes, interrupt_at_market="0xb"), start=100, end=399, markets_per_request=1)

    assert cursor_file.exists(), "Cursor file should survive Ctrl+C"
    assert cursor_file.read_text().strip() == "1", "Cursor should point at the next unfetched market"

    # Run 2: resume from the cursor, no interrupt
    client2 = FakeClient(tapes)
    run_indexer(client2, start=100, end=399, markets_per_request=1)

    assert client2.requested == [("0xb", 100, 399), ("0xc", 100, 399)], "Resume should skip completed markets"
    assert not cursor_file.exists(), "Cursor file should be deleted after successful completion"

    df = read_all(data_dir)
    assert sorted(df["timestamp"]) == [100, 150, 250, 350]
    assert not df.duplicated(subset=["transaction_hash", "timestamp"]).any()


def test_interrupt_inside_market_resumes_from_window_cursor(isolated_dirs):
    data_dir, markets_dir, cursor_file = isolated_dirs
    write_markets(markets_dir, ["0xa"])
    trades = [make_trade(ts, tx=f"0x{ts}") for ts in [100, 101, 102, 103, 104, 105]]

    # Run 1: overflow forces splitting into leaf windows; interrupt once the
    # window walk reaches timestamp 104, after leaves [100,101] and [102,103].
    run_indexer(
        FakeClient({"0xa": trades}, overflow_threshold=3, interrupt_at_window=104),
        start=100,
        end=199,
    )

    idx, ts = cursor_file.read_text().strip().split(",")
    assert (idx, int(ts)) == ("0", 104), "Cursor should point inside the interrupted market"
    assert sorted(read_all(data_dir)["timestamp"]) == [100, 101, 102, 103]

    # Run 2: resume mid-market without refetching completed windows
    client2 = FakeClient({"0xa": trades})
    run_indexer(client2, start=100, end=199)

    assert client2.requested == [("0xa", 104, 199)]
    df = read_all(data_dir)
    assert sorted(df["timestamp"]) == [100, 101, 102, 103, 104, 105]
    assert not df.duplicated(subset=["transaction_hash", "timestamp"]).any()


def test_duplicate_rows_within_and_across_windows_are_deduped(isolated_dirs):
    data_dir, markets_dir, _ = isolated_dirs
    write_markets(markets_dir, ["0xa"])
    boundary = make_trade(199, tx="0xdup")

    class SloppyClient(FakeClient):
        """Splits once, returns a duplicated row in-window and a boundary row in both leaves."""

        def get_data_trades_window(self, start, end, limit=1000, taker_only=True, market=None):
            self.requested.append((market, start, end))
            if (start, end) == (100, 299):
                return [], True
            if end <= 199:
                return [make_trade(150, tx="0x150"), make_trade(150, tx="0x150"), boundary], False
            return [boundary, make_trade(250, tx="0x250")], False

    client = SloppyClient({})
    run_indexer(client, start=100, end=299)

    df = read_all(data_dir)
    assert sorted(df["timestamp"]) == [150, 199, 250]
