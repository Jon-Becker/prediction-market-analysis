"""Tests for the Data API trades indexer: windowing, splitting, resume, dedup."""

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
    """Stub for PolymarketClient serving a fixed tape of trades by timestamp."""

    def __init__(
        self,
        tape: list[DataApiTrade],
        *,
        overflow_threshold: int | None = None,
        interrupt_at: int | None = None,
    ):
        self.tape = sorted(tape, key=lambda t: t.timestamp)
        self.overflow_threshold = overflow_threshold
        self.interrupt_at = interrupt_at
        self.requested_windows: list[tuple[int, int]] = []
        self.taker_only_args: list[bool] = []

    def get_data_trades_window(self, start, end, limit=1000, taker_only=True):
        if self.interrupt_at is not None and start >= self.interrupt_at:
            raise KeyboardInterrupt
        self.requested_windows.append((start, end))
        self.taker_only_args.append(taker_only)
        rows = [t for t in self.tape if start <= t.timestamp <= end]
        if self.overflow_threshold is not None and len(rows) > self.overflow_threshold:
            return rows[: self.overflow_threshold], True
        return rows, False

    def close(self):
        pass


@pytest.fixture()
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point DATA_DIR and CURSOR_FILE at temp directories."""
    data_dir = tmp_path / "data_api_trades"
    cursor_file = tmp_path / ".data_api_trades_cursor"
    import src.indexers.polymarket.data_api_trades as mod

    monkeypatch.setattr(mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(mod, "CURSOR_FILE", cursor_file)
    return data_dir, cursor_file


def read_all(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("trades_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run_indexer(client: FakeClient, **kwargs):
    from src.indexers.polymarket.data_api_trades import PolymarketDataApiTradesIndexer

    with patch("src.indexers.polymarket.data_api_trades.PolymarketClient", return_value=client):
        PolymarketDataApiTradesIndexer(**kwargs).run()


def test_windows_cover_range_and_all_trades_saved(isolated_dirs, monkeypatch):
    import src.indexers.polymarket.data_api_trades as mod

    monkeypatch.setattr(mod, "BATCH_SIZE", 2)
    data_dir, cursor_file = isolated_dirs
    client = FakeClient([make_trade(ts, tx=f"0x{ts}") for ts in [100, 150, 199, 200, 299]])

    run_indexer(client, start=100, end=299, window_size=100)

    assert client.requested_windows == [(100, 199), (200, 299)]
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
    _, _ = isolated_dirs
    client = FakeClient([make_trade(50)])

    run_indexer(client, end=200, window_size=1000)

    assert client.requested_windows == [(FULL_HISTORY_START, 200)]


def test_explicit_start_overrides_cursor(isolated_dirs):
    data_dir, cursor_file = isolated_dirs
    cursor_file.parent.mkdir(parents=True, exist_ok=True)
    cursor_file.write_text("150")
    client = FakeClient([make_trade(60)])

    run_indexer(client, start=50, end=99, window_size=100)

    assert client.requested_windows == [(50, 99)]
    assert read_all(data_dir)["timestamp"].tolist() == [60]


def test_overflowing_window_is_split_until_it_fits(isolated_dirs):
    data_dir, _ = isolated_dirs
    trades = [make_trade(ts, tx=f"0x{ts}") for ts in [100, 101, 102, 103, 104, 105]]
    client = FakeClient(trades, overflow_threshold=3)

    run_indexer(client, start=100, end=199, window_size=100)

    # The full window overflows the offset cap and must be split recursively
    assert (100, 199) in client.requested_windows
    assert len(client.requested_windows) > 1

    df = read_all(data_dir)
    assert sorted(df["timestamp"]) == [100, 101, 102, 103, 104, 105], "every trade must be recovered via splits"


def test_overflowing_single_second_window_keeps_partial_rows(isolated_dirs, capsys):
    data_dir, _ = isolated_dirs
    # 5 trades in the same second cannot be separated by timestamp windows
    trades = [make_trade(100, tx=f"0x{i}") for i in range(5)]
    client = FakeClient(trades, overflow_threshold=3)

    run_indexer(client, start=100, end=100, window_size=100)

    df = read_all(data_dir)
    assert len(df) == 3, "rows up to the cap are kept for an unsplittable window"
    assert "overflows the offset cap" in capsys.readouterr().out


def test_interrupt_then_resume_completes_without_duplicates(isolated_dirs):
    data_dir, cursor_file = isolated_dirs
    tape = [make_trade(ts, tx=f"0x{ts}") for ts in [100, 150, 250, 350]]

    # Run 1: interrupt when the second window (starting at 200) is requested
    run_indexer(FakeClient(tape, interrupt_at=200), start=100, end=399, window_size=100)

    assert cursor_file.exists(), "Cursor file should survive Ctrl+C"
    assert int(cursor_file.read_text().strip()) == 200, "Cursor should point at the next unfetched timestamp"

    # Run 2: resume with no start (reads cursor), no interrupt
    client2 = FakeClient(tape)
    run_indexer(client2, end=399, window_size=100)

    assert client2.requested_windows == [(200, 299), (300, 399)], "Resume should start from the cursor"
    assert not cursor_file.exists(), "Cursor file should be deleted after successful completion"

    df = read_all(data_dir)
    assert sorted(df["timestamp"]) == [100, 150, 250, 350]
    assert not df.duplicated(subset=["transaction_hash", "timestamp"]).any()


def test_duplicate_rows_within_and_across_windows_are_deduped(isolated_dirs):
    data_dir, _ = isolated_dirs
    boundary = make_trade(199, tx="0xdup")

    class SloppyClient(FakeClient):
        """Returns a duplicated row in-window and a boundary row in both windows."""

        def get_data_trades_window(self, start, end, limit=1000, taker_only=True):
            self.requested_windows.append((start, end))
            if start == 100:
                return [make_trade(150, tx="0x150"), make_trade(150, tx="0x150"), boundary], False
            return [boundary, make_trade(250, tx="0x250")], False

    client = SloppyClient([])
    run_indexer(client, start=100, end=299, window_size=100)

    df = read_all(data_dir)
    assert sorted(df["timestamp"]) == [150, 199, 250]
