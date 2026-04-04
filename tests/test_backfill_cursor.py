"""Test that backfill can be interrupted with Ctrl+C and resumed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

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
