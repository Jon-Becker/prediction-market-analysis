"""Block number to timestamp/date conversion utilities."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

# Fallback anchor points when blocks table is unavailable
FALLBACK_ANCHORS = [
    (40_000_176, 1_678_036_019),
    (50_000_000, 1_700_108_689),
    (60_000_000, 1_722_368_382),
    (70_000_000, 1_744_013_119),
    (78_756_659, 1_762_618_199),
]

BLOCKS_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "polymarket" / "blocks"


@lru_cache(maxsize=1)
def _load_blocks_data() -> tuple[np.ndarray, np.ndarray] | None:
    """Load blocks table data for interpolation. Returns None if unavailable."""
    if not BLOCKS_DIR.exists():
        return None

    parquet_files = list(BLOCKS_DIR.glob("*.parquet"))
    if not parquet_files:
        return None

    try:
        import duckdb

        result = duckdb.execute(
            f"""
            SELECT block_number, timestamp
            FROM '{BLOCKS_DIR}/*.parquet'
            ORDER BY block_number
            """
        ).fetchall()

        if not result:
            return None

        block_numbers = np.array([r[0] for r in result])
        timestamps = np.array([r[1] for r in result])
        return block_numbers, timestamps
    except Exception:
        return None


def _get_interpolation_data() -> tuple[np.ndarray, np.ndarray]:
    """Get block/timestamp arrays for interpolation, using blocks table or fallback."""
    blocks_data = _load_blocks_data()
    if blocks_data is not None:
        return blocks_data

    # Use fallback anchors
    block_numbers = np.array([b for b, _ in FALLBACK_ANCHORS])
    timestamps = np.array([t for _, t in FALLBACK_ANCHORS])
    return block_numbers, timestamps


def block_to_timestamp(block_number: int) -> int:
    """Convert a block number to a Unix timestamp via interpolation."""
    block_numbers, timestamps = _get_interpolation_data()
    return int(np.interp(block_number, block_numbers, timestamps))


def block_to_date(block_number: int) -> datetime:
    """Convert a block number to a datetime via interpolation."""
    return datetime.fromtimestamp(block_to_timestamp(block_number))
