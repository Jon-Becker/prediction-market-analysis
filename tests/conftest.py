"""Shared fixtures for analysis tests."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

# Use non-interactive backend for headless CI
matplotlib.use("Agg")


# -- Helper functions to build fixture DataFrames --


def _make_kalshi_trades() -> pd.DataFrame:
    """Build Kalshi trades DataFrame (2100 rows).

    7 price levels x 3 variants x 100 copies.  The 100-copy multiplier
    ensures analyses with minimum-sample thresholds (e.g. StatisticalTestsAnalysis
    requires >= 100 per price bin) don't hit empty-DataFrame edge cases.
    """
    rows = []
    prices = [10, 20, 30, 50, 60, 70, 90]
    variants = [
        ("MKT-A", "yes", lambda p: p, lambda p: 100 - p, 10),
        ("MKT-B", "no", lambda p: 100 - p, lambda p: p, 5),
        ("MKT-B", "yes", lambda p: p, lambda p: 100 - p, 3),
    ]
    trade_id = 0
    base_time = pd.Timestamp("2024-06-01 12:00:00")

    for _ in range(100):
        for price in prices:
            for ticker, taker_side, yes_fn, no_fn, count in variants:
                trade_id += 1
                rows.append(
                    {
                        "trade_id": str(trade_id),
                        "ticker": ticker,
                        "count": count,
                        "yes_price": yes_fn(price),
                        "no_price": no_fn(price),
                        "taker_side": taker_side,
                        "created_time": base_time + pd.Timedelta(minutes=trade_id),
                        "_fetched_at": base_time,
                    }
                )

    return pd.DataFrame(rows)


def _make_kalshi_markets() -> pd.DataFrame:
    """Build minimal Kalshi markets DataFrame (2 rows)."""
    return pd.DataFrame(
        [
            {
                "ticker": "MKT-A",
                "status": "finalized",
                "result": "yes",
                "volume": 1000,
                "event_ticker": "INXD-24JAN01",
                "close_time": pd.Timestamp("2024-07-01 12:00:00"),
            },
            {
                "ticker": "MKT-B",
                "status": "finalized",
                "result": "no",
                "volume": 2000,
                "event_ticker": "NFLGAME-25FEB01",
                "close_time": pd.Timestamp("2024-08-01 12:00:00"),
            },
        ]
    )


def _make_polymarket_ctf_trades() -> pd.DataFrame:
    """Build minimal Polymarket CTF trades DataFrame (~10 rows).

    5 price levels x 2 sides (buying token_yes_a and token_yes_b).
    maker_asset_id='0' means buyer pays USDC.
    Price = 100 * maker_amount / taker_amount.
    """
    rows = []
    prices = [20, 40, 50, 60, 80]

    for i, price in enumerate(prices):
        rows.append(
            {
                "block_number": 100 + i * 2,
                "maker_asset_id": "0",
                "taker_asset_id": "token_yes_a",
                "maker_amount": price * 10000,
                "taker_amount": 1000000,
            }
        )
        rows.append(
            {
                "block_number": 100 + i * 2 + 1,
                "maker_asset_id": "0",
                "taker_asset_id": "token_yes_b",
                "maker_amount": price * 10000,
                "taker_amount": 1000000,
            }
        )

    return pd.DataFrame(rows)


def _make_polymarket_legacy_trades() -> pd.DataFrame:
    """Build minimal Polymarket legacy FPMM trades DataFrame (~10 rows).

    5 price levels x 2 outcome indices, all for the same FPMM address.
    """
    rows = []
    prices = [20, 40, 50, 60, 80]

    for i, price in enumerate(prices):
        rows.append(
            {
                "block_number": 200 + i * 2,
                "fpmm_address": "0xfpmm_address_a",
                "amount": str(price * 10000),
                "outcome_tokens": str(1000000),
                "outcome_index": 0,
            }
        )
        rows.append(
            {
                "block_number": 200 + i * 2 + 1,
                "fpmm_address": "0xfpmm_address_a",
                "amount": str(price * 10000),
                "outcome_tokens": str(1000000),
                "outcome_index": 1,
            }
        )

    return pd.DataFrame(rows)


def _make_polymarket_markets() -> pd.DataFrame:
    """Build minimal Polymarket markets DataFrame (2 rows).

    Market A resolved YES (outcome_prices=[1.0, 0.0]), has FPMM address.
    Market B resolved NO (outcome_prices=[0.0, 1.0]).
    """
    return pd.DataFrame(
        [
            {
                "id": "market_a",
                "clob_token_ids": json.dumps(["token_yes_a", "token_no_a"]),
                "outcome_prices": json.dumps([1.0, 0.0]),
                "market_maker_address": "0xfpmm_address_a",
                "closed": True,
            },
            {
                "id": "market_b",
                "clob_token_ids": json.dumps(["token_yes_b", "token_no_b"]),
                "outcome_prices": json.dumps([0.0, 1.0]),
                "market_maker_address": None,
                "closed": True,
            },
        ]
    )


def _make_polymarket_blocks(ctf_trades: pd.DataFrame, legacy_trades: pd.DataFrame) -> pd.DataFrame:
    """Build blocks DataFrame with an entry for every exact block_number in trades.

    Needed by PolymarketTradesOverTimeAnalysis (direct join) and volume/animated
    analyses (bucketed join — all blocks fall in bucket 0 since block_number < 10800).
    """
    block_numbers = sorted(set(ctf_trades["block_number"].tolist()) | set(legacy_trades["block_number"].tolist()))
    base_time = pd.Timestamp("2024-06-01 12:00:00", tz="UTC")

    return pd.DataFrame(
        [
            {
                "block_number": bn,
                "timestamp": (base_time + pd.Timedelta(seconds=bn * 2)).isoformat(),
            }
            for bn in block_numbers
        ]
    )


# -- Session-scoped fixtures --


@pytest.fixture(scope="session")
def kalshi_trades_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("kalshi_trades")
    _make_kalshi_trades().to_parquet(d / "trades.parquet")
    return d


@pytest.fixture(scope="session")
def kalshi_markets_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("kalshi_markets")
    _make_kalshi_markets().to_parquet(d / "markets.parquet")
    return d


@pytest.fixture(scope="session")
def polymarket_trades_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("polymarket_trades")
    _make_polymarket_ctf_trades().to_parquet(d / "trades.parquet")
    return d


@pytest.fixture(scope="session")
def polymarket_legacy_trades_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("polymarket_legacy_trades")
    _make_polymarket_legacy_trades().to_parquet(d / "legacy_trades.parquet")
    return d


@pytest.fixture(scope="session")
def polymarket_markets_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("polymarket_markets")
    _make_polymarket_markets().to_parquet(d / "markets.parquet")
    return d


@pytest.fixture(scope="session")
def polymarket_blocks_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ctf = _make_polymarket_ctf_trades()
    legacy = _make_polymarket_legacy_trades()
    d = tmp_path_factory.mktemp("polymarket_blocks")
    _make_polymarket_blocks(ctf, legacy).to_parquet(d / "blocks.parquet")
    return d


@pytest.fixture(scope="session")
def collateral_lookup_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("polymarket_collateral")
    p = d / "fpmm_collateral_lookup.json"
    p.write_text(
        json.dumps(
            {
                "0xfpmm_address_a": {"collateral_symbol": "USDC", "collateral_decimals": 6},
            }
        )
    )
    return p


@pytest.fixture(scope="session")
def all_fixture_dirs(
    kalshi_trades_dir: Path,
    kalshi_markets_dir: Path,
    polymarket_trades_dir: Path,
    polymarket_legacy_trades_dir: Path,
    polymarket_markets_dir: Path,
    polymarket_blocks_dir: Path,
    collateral_lookup_path: Path,
) -> dict[str, Path]:
    """Bundle all fixture directories for easy access."""
    return {
        "kalshi_trades_dir": kalshi_trades_dir,
        "kalshi_markets_dir": kalshi_markets_dir,
        "polymarket_trades_dir": polymarket_trades_dir,
        "polymarket_legacy_trades_dir": polymarket_legacy_trades_dir,
        "polymarket_markets_dir": polymarket_markets_dir,
        "polymarket_blocks_dir": polymarket_blocks_dir,
        "collateral_lookup_path": collateral_lookup_path,
    }
