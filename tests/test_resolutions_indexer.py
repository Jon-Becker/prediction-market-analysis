"""Test that the resolutions backfill can be interrupted and resumed without losing data."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.indexers.polymarket.blockchain import ConditionResolution


def make_resolution(block_number: int, payout_numerators: list[int] | None = None) -> ConditionResolution:
    return ConditionResolution(
        block_number=block_number,
        transaction_hash="0xabc",
        log_index=0,
        condition_id="0x" + "11" * 32,
        oracle="0x" + "22" * 20,
        question_id="0x" + "33" * 32,
        outcome_slot_count=2,
        payout_numerators=payout_numerators if payout_numerators is not None else [1, 0],
    )


class FakeClient:
    """Stub for PolygonClient that tracks which blocks were requested."""

    def __init__(self, *, interrupt_at_block: int | None = None, error_at_block: int | None = None):
        self._interrupt_at = interrupt_at_block
        self._error_at = error_at_block
        self.requested_ranges: list[tuple[int, int]] = []

    def get_block_number(self) -> int:
        return 5000

    def get_condition_resolutions(self, from_block: int, to_block: int) -> list[ConditionResolution]:
        if self._interrupt_at is not None and from_block >= self._interrupt_at:
            raise KeyboardInterrupt
        if self._error_at is not None and from_block >= self._error_at:
            raise RuntimeError("rpc exploded")
        self.requested_ranges.append((from_block, to_block))
        return [make_resolution(block_number=from_block)]


@pytest.fixture()
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point DATA_DIR and CURSOR_FILE at temp directories."""
    data_dir = tmp_path / "resolutions"
    cursor_file = tmp_path / ".resolutions_block_cursor"
    import src.indexers.polymarket.resolutions as mod

    monkeypatch.setattr(mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(mod, "CURSOR_FILE", cursor_file)
    return data_dir, cursor_file


def test_interrupt_then_resume_completes(isolated_dirs):
    """Interrupt a backfill with Ctrl+C, then resume and finish the job."""
    from src.indexers.polymarket.resolutions import PolymarketResolutionsIndexer

    data_dir, cursor_file = isolated_dirs

    # Run 1: interrupt at block 3000
    client1 = FakeClient(interrupt_at_block=3000)
    with patch("src.indexers.polymarket.resolutions.PolygonClient", return_value=client1):
        indexer = PolymarketResolutionsIndexer(from_block=1000, to_block=5000, chunk_size=1000)
        indexer.run()

    # Cursor should exist and point to the last completed block
    assert cursor_file.exists(), "Cursor file should survive Ctrl+C"
    saved_block = int(cursor_file.read_text().strip())
    assert saved_block == 2999, f"Cursor should be at last completed chunk end, got {saved_block}"

    # Buffered rows fetched before the interrupt should have been flushed
    flushed = pd.concat([pd.read_parquet(f) for f in data_dir.glob("resolutions_*.parquet")])
    assert len(flushed) == 2, "Resolutions fetched before the interrupt should be flushed to parquet"

    # Run 2: resume with no from_block (reads cursor), no interrupt
    client2 = FakeClient()
    with patch("src.indexers.polymarket.resolutions.PolygonClient", return_value=client2):
        indexer = PolymarketResolutionsIndexer(to_block=5000, chunk_size=1000)
        indexer.run()

    # Should have resumed from the block after the cursor: the cursor block's
    # rows were already flushed, so re-fetching it would duplicate them
    first_requested = min(start for start, _ in client2.requested_ranges)
    assert first_requested == saved_block + 1, (
        f"Resume should start after cursor ({saved_block + 1}), not {first_requested}"
    )

    # Cursor should be cleaned up after successful completion
    assert not cursor_file.exists(), "Cursor file should be deleted after successful completion"


def test_exception_preserves_cursor_and_flushes_buffer(isolated_dirs):
    """Any exception (not just Ctrl+C) preserves the cursor and flushes buffered rows."""
    from src.indexers.polymarket.resolutions import PolymarketResolutionsIndexer

    data_dir, cursor_file = isolated_dirs

    client = FakeClient(error_at_block=3000)
    with patch("src.indexers.polymarket.resolutions.PolygonClient", return_value=client):
        indexer = PolymarketResolutionsIndexer(from_block=1000, to_block=5000, chunk_size=1000)
        with pytest.raises(RuntimeError):
            indexer.run()

    assert cursor_file.exists(), "Cursor file should survive an unexpected exception"
    assert int(cursor_file.read_text().strip()) == 2999

    flushed = pd.concat([pd.read_parquet(f) for f in data_dir.glob("resolutions_*.parquet")])
    assert len(flushed) == 2, "Resolutions fetched before the error should be flushed to parquet"


def test_parquet_schema_and_winning_outcome_derivable(isolated_dirs):
    """Saved rows serialize payout_numerators as a JSON string and keep the winning outcome derivable."""
    from src.indexers.polymarket.resolutions import PolymarketResolutionsIndexer

    data_dir, cursor_file = isolated_dirs

    client = FakeClient()
    with patch("src.indexers.polymarket.resolutions.PolygonClient", return_value=client):
        indexer = PolymarketResolutionsIndexer(from_block=1000, to_block=1999, chunk_size=1000)
        indexer.run()

    files = list(data_dir.glob("resolutions_*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert list(df.columns) == [
        "block_number",
        "transaction_hash",
        "log_index",
        "condition_id",
        "oracle",
        "question_id",
        "outcome_slot_count",
        "payout_numerators",
        "_fetched_at",
    ]

    row = df.iloc[0]
    assert row["block_number"] == 1000
    assert row["condition_id"] == "0x" + "11" * 32
    assert row["outcome_slot_count"] == 2

    # payout_numerators is a JSON string from which the winning outcome is derivable
    numerators = json.loads(row["payout_numerators"])
    assert numerators == [1, 0]
    nonzero = [i for i, numerator in enumerate(numerators) if numerator != 0]
    assert nonzero == [0]
