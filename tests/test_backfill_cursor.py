"""Test that backfill can be interrupted with Ctrl+C and resumed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest


@dataclass
class FakeTrade:
    order_hash: str = "0x00"
    maker: str = "0x01"
    taker: str = "0x02"
    maker_asset_id: int = 0
    taker_asset_id: int = 1
    maker_amount: int = 100
    taker_amount: int = 200
    block_number: int = 1000
    transaction_hash: str = "0xabc"
    log_index: int = 0


class FakeClient:
    """Stub for PolygonClient that tracks which blocks were requested."""

    def __init__(self, *, interrupt_at_block: int | None = None):
        self._interrupt_at = interrupt_at_block
        self.requested_ranges: list[tuple[int, int]] = []

    def get_block_number(self) -> int:
        return 5000

    def get_trades(self, from_block: int, to_block: int, contract_address: str) -> list:
        if self._interrupt_at is not None and from_block >= self._interrupt_at:
            raise KeyboardInterrupt
        self.requested_ranges.append((from_block, to_block))
        return [FakeTrade(block_number=from_block)]


@pytest.fixture()
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point DATA_DIR and CURSOR_FILE at temp directories."""
    data_dir = tmp_path / "trades"
    cursor_file = tmp_path / ".backfill_block_cursor"
    import src.indexers.polymarket.trades as mod

    monkeypatch.setattr(mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(mod, "CURSOR_FILE", cursor_file)
    return data_dir, cursor_file


def test_interrupt_then_resume_completes(isolated_dirs):
    """Interrupt a backfill with Ctrl+C, then resume and finish the job."""
    from src.indexers.polymarket.trades import PolymarketTradesIndexer

    data_dir, cursor_file = isolated_dirs

    # Run 1: interrupt at block 3000
    client1 = FakeClient(interrupt_at_block=3000)
    with patch("src.indexers.polymarket.trades.PolygonClient", return_value=client1):
        indexer = PolymarketTradesIndexer(from_block=1000, to_block=5000, chunk_size=1000)
        indexer.run()

    # Cursor should exist and point to the last completed block
    assert cursor_file.exists(), "Cursor file should survive Ctrl+C"
    saved_block = int(cursor_file.read_text().strip())
    assert saved_block == 2999, f"Cursor should be at last completed chunk end, got {saved_block}"

    # Run 2: resume with no from_block (reads cursor), no interrupt
    client2 = FakeClient()
    with patch("src.indexers.polymarket.trades.PolygonClient", return_value=client2):
        indexer = PolymarketTradesIndexer(to_block=5000, chunk_size=1000)
        indexer.run()

    # Should have resumed from the saved cursor, not from the beginning
    first_requested = min(start for start, _ in client2.requested_ranges)
    assert first_requested == saved_block, f"Resume should start from cursor ({saved_block}), not {first_requested}"

    # Cursor should be cleaned up after successful completion
    assert not cursor_file.exists(), "Cursor file should be deleted after successful completion"


@dataclass
class FakeMarket:
    id: str = "0"
    question: str = "test?"


class FakeMarketsClient:
    """Stub for PolymarketClient that serves paginated markets."""

    def __init__(self, total: int = 20, page_size: int = 5, *, interrupt_at_offset: int | None = None):
        self._total = total
        self._page_size = page_size
        self._interrupt_at = interrupt_at_offset
        self.requested_offsets: list[int] = []

    def iter_markets(self, offset: int = 0):
        current_offset = offset
        while True:
            if self._interrupt_at is not None and current_offset >= self._interrupt_at:
                raise KeyboardInterrupt
            self.requested_offsets.append(current_offset)
            markets = [
                FakeMarket(id=str(i)) for i in range(current_offset, min(current_offset + self._page_size, self._total))
            ]
            if not markets:
                yield [], -1
                break
            next_offset = current_offset + len(markets)
            yield markets, next_offset
            if len(markets) < self._page_size:
                break
            current_offset = next_offset

    def close(self):
        pass


@pytest.fixture()
def isolated_markets_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the markets indexer's DATA_DIR and OFFSET_FILE at temp directories."""
    data_dir = tmp_path / "markets"
    offset_file = tmp_path / ".backfill_offset"
    import src.indexers.polymarket.markets as mod

    monkeypatch.setattr(mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(mod, "OFFSET_FILE", offset_file)
    return data_dir, offset_file


def test_markets_interrupt_then_resume_completes(isolated_markets_dirs):
    """Interrupt a markets backfill with Ctrl+C: buffered rows must be flushed, then resume finishes the job."""
    from src.indexers.polymarket.markets import PolymarketMarketsIndexer

    data_dir, offset_file = isolated_markets_dirs

    # Run 1: interrupt after two pages (10 markets fetched, buffer below CHUNK_SIZE)
    client1 = FakeMarketsClient(interrupt_at_offset=10)
    with patch("src.indexers.polymarket.markets.PolymarketClient", return_value=client1):
        PolymarketMarketsIndexer().run()

    # Offset file should exist and point past the fetched pages
    assert offset_file.exists(), "Offset file should survive Ctrl+C"
    saved_offset = int(offset_file.read_text().strip())
    assert saved_offset == 10, f"Offset should be at last completed page, got {saved_offset}"

    # Every market the offset claims was fetched must be on disk
    persisted = pd.concat([pd.read_parquet(f) for f in data_dir.glob("markets_*.parquet")])
    assert len(persisted) == saved_offset, "Markets fetched before the interrupt must be flushed to parquet"

    # Run 2: resume (reads offset file), no interrupt
    client2 = FakeMarketsClient()
    with patch("src.indexers.polymarket.markets.PolymarketClient", return_value=client2):
        PolymarketMarketsIndexer().run()

    # Should have resumed from the saved offset, not from the beginning
    assert client2.requested_offsets[0] == saved_offset, (
        f"Resume should start from offset ({saved_offset}), not {client2.requested_offsets[0]}"
    )

    # Offset file should be cleaned up after successful completion
    assert not offset_file.exists(), "Offset file should be deleted after successful completion"

    # All markets present exactly once across both runs
    persisted = pd.concat([pd.read_parquet(f) for f in data_dir.glob("markets_*.parquet")])
    assert sorted(persisted["id"].astype(int).tolist()) == list(range(20))


class FakeLegacyClient:
    """Stub for PolygonClient used by the legacy FPMM indexer."""

    def __init__(self, *, interrupt_at_block: int | None = None):
        self._interrupt_at = interrupt_at_block
        self.requested_ranges: list[tuple[int, int]] = []
        self.w3 = SimpleNamespace(eth=SimpleNamespace(get_logs=self._get_logs))

    def get_block_number(self) -> int:
        return 5000

    def _get_logs(self, params: dict) -> list:
        from_block = params["fromBlock"]
        if self._interrupt_at is not None and from_block >= self._interrupt_at:
            raise KeyboardInterrupt
        self.requested_ranges.append((from_block, params["toBlock"]))
        return []


@pytest.fixture()
def isolated_legacy_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the legacy indexer's DATA_DIR and CURSOR_FILE at temp directories."""
    data_dir = tmp_path / "legacy_trades"
    cursor_file = tmp_path / ".legacy_backfill_block_cursor"
    import src.indexers.polymarket.fpmm_trades as mod

    monkeypatch.setattr(mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(mod, "CURSOR_FILE", cursor_file)
    return data_dir, cursor_file


def test_legacy_interrupt_then_resume_completes(isolated_legacy_dirs):
    """Interrupt a legacy FPMM backfill with Ctrl+C: cursor must survive, then resume finishes the job."""
    from src.indexers.polymarket.fpmm_trades import PolymarketLegacyTradesIndexer

    data_dir, cursor_file = isolated_legacy_dirs

    # Run 1: interrupt at block 3000
    client1 = FakeLegacyClient(interrupt_at_block=3000)
    with patch("src.indexers.polymarket.fpmm_trades.PolygonClient", return_value=client1):
        indexer = PolymarketLegacyTradesIndexer(from_block=1000, to_block=5000, chunk_size=1000, max_workers=1)
        indexer.run()

    # Cursor should exist and point to the last completed block
    assert cursor_file.exists(), "Cursor file should survive Ctrl+C"
    saved_block = int(cursor_file.read_text().strip())
    assert saved_block == 2999, f"Cursor should be at last completed chunk end, got {saved_block}"

    # Run 2: resume with no from_block (reads cursor), no interrupt
    client2 = FakeLegacyClient()
    with patch("src.indexers.polymarket.fpmm_trades.PolygonClient", return_value=client2):
        indexer = PolymarketLegacyTradesIndexer(to_block=5000, chunk_size=1000, max_workers=1)
        indexer.run()

    # Should have resumed from the saved cursor, not from the beginning
    first_requested = min(start for start, _ in client2.requested_ranges)
    assert first_requested == saved_block, f"Resume should start from cursor ({saved_block}), not {first_requested}"

    # Cursor should be cleaned up after successful completion
    assert not cursor_file.exists(), "Cursor file should be deleted after successful completion"
