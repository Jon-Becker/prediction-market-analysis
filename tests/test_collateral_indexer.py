"""Test the FPMM collateral lookup indexer with a stubbed Polygon client."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
DAI = "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063"

FPMM_A = "0x1111111111111111111111111111111111111111"
FPMM_B = "0x2222222222222222222222222222222222222222"
FPMM_C = "0x3333333333333333333333333333333333333333"


class FakeContractFunction:
    def __init__(self, result):
        self._result = result

    def call(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeEth:
    """Stub for w3.eth that serves contract calls from canned maps."""

    def __init__(self, client: FakeClient):
        self._client = client

    def contract(self, address: str, abi: list[dict]):
        client = self._client
        if abi[0]["name"] == "collateralToken":
            client.collateral_calls.append(address)
            result = client.collateral_map[address]
            functions = SimpleNamespace(collateralToken=lambda: FakeContractFunction(result))
        else:
            client.symbol_calls.append(address)
            result = client.symbol_map[address]
            functions = SimpleNamespace(symbol=lambda: FakeContractFunction(result))
        return SimpleNamespace(functions=functions)


class FakeClient:
    """Stub for PolygonClient that tracks which contracts were queried."""

    def __init__(self, collateral_map: dict, symbol_map: dict):
        self.collateral_map = collateral_map
        self.symbol_map = symbol_map
        self.collateral_calls: list[str] = []
        self.symbol_calls: list[str] = []
        self.w3 = SimpleNamespace(eth=FakeEth(self))


@pytest.fixture()
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point LEGACY_TRADES_DIR and LOOKUP_FILE at temp locations."""
    trades_dir = tmp_path / "legacy_trades"
    lookup_file = tmp_path / "fpmm_collateral_lookup.json"
    import src.indexers.polymarket.collateral as mod

    monkeypatch.setattr(mod, "LEGACY_TRADES_DIR", trades_dir)
    monkeypatch.setattr(mod, "LOOKUP_FILE", lookup_file)
    return trades_dir, lookup_file


def _write_legacy_trades(trades_dir: Path, addresses: list[str]) -> None:
    trades_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"fpmm_address": addresses * 2})  # duplicates to exercise DISTINCT
    df.to_parquet(trades_dir / "trades_0_10000.parquet")


def _run_indexer(client: FakeClient, max_workers: int = 10) -> None:
    from src.indexers.polymarket.collateral import PolymarketCollateralIndexer

    with patch("src.indexers.polymarket.collateral.PolygonClient", return_value=client):
        PolymarketCollateralIndexer(max_workers=max_workers).run()


def test_generates_lookup(isolated_paths):
    """A fresh run resolves every distinct FPMM address to the documented JSON shape."""
    trades_dir, lookup_file = isolated_paths
    _write_legacy_trades(trades_dir, [FPMM_A, FPMM_B, FPMM_C])

    client = FakeClient(
        collateral_map={FPMM_A: USDC, FPMM_B: USDC, FPMM_C: DAI},
        symbol_map={USDC: "USDC", DAI: "DAI"},
    )
    _run_indexer(client)

    lookup = json.loads(lookup_file.read_text())
    assert lookup == {
        FPMM_A: {"collateral_address": USDC, "collateral_symbol": "USDC"},
        FPMM_B: {"collateral_address": USDC, "collateral_symbol": "USDC"},
        FPMM_C: {"collateral_address": DAI, "collateral_symbol": "DAI"},
    }
    assert sorted(client.collateral_calls) == sorted([FPMM_A, FPMM_B, FPMM_C])

    # Round-trips with the consumers' USDC filter (cf. polymarket_win_rate_by_price.py)
    usdc_markets = {addr.lower() for addr, info in lookup.items() if info["collateral_symbol"] == "USDC"}
    assert usdc_markets == {FPMM_A.lower(), FPMM_B.lower()}


def test_idempotent_resume(isolated_paths):
    """Addresses already in the lookup are skipped; only new ones hit the chain."""
    trades_dir, lookup_file = isolated_paths
    _write_legacy_trades(trades_dir, [FPMM_A, FPMM_B, FPMM_C])
    lookup_file.write_text(json.dumps({FPMM_A: {"collateral_address": USDC, "collateral_symbol": "USDC"}}))

    client = FakeClient(
        collateral_map={FPMM_B: USDC, FPMM_C: USDC},
        symbol_map={USDC: "USDC"},
    )
    _run_indexer(client)

    assert len(client.collateral_calls) == 2
    assert FPMM_A not in client.collateral_calls
    lookup = json.loads(lookup_file.read_text())
    assert set(lookup) == {FPMM_A, FPMM_B, FPMM_C}


def test_symbol_failure_falls_back_to_unknown(isolated_paths):
    """A non-standard ERC-20 symbol() failure records UNKNOWN instead of crashing."""
    trades_dir, lookup_file = isolated_paths
    _write_legacy_trades(trades_dir, [FPMM_A])

    client = FakeClient(
        collateral_map={FPMM_A: DAI},
        symbol_map={DAI: Exception("execution reverted")},
    )
    _run_indexer(client)

    lookup = json.loads(lookup_file.read_text())
    assert lookup == {FPMM_A: {"collateral_address": DAI, "collateral_symbol": "UNKNOWN"}}


def test_collateral_failure_skips_address(isolated_paths):
    """An FPMM whose collateralToken() call fails is skipped, not recorded."""
    trades_dir, lookup_file = isolated_paths
    _write_legacy_trades(trades_dir, [FPMM_A, FPMM_B])

    client = FakeClient(
        collateral_map={FPMM_A: Exception("execution reverted"), FPMM_B: USDC},
        symbol_map={USDC: "USDC"},
    )
    _run_indexer(client)

    lookup = json.loads(lookup_file.read_text())
    assert lookup == {FPMM_B: {"collateral_address": USDC, "collateral_symbol": "USDC"}}


def test_symbol_cache_deduplicates_token_calls(isolated_paths):
    """FPMMs sharing a collateral token trigger a single symbol() call."""
    trades_dir, lookup_file = isolated_paths
    _write_legacy_trades(trades_dir, [FPMM_A, FPMM_B])

    client = FakeClient(
        collateral_map={FPMM_A: USDC, FPMM_B: USDC},
        symbol_map={USDC: "USDC"},
    )
    _run_indexer(client, max_workers=1)

    assert client.symbol_calls == [USDC]


def test_empty_data_dir_is_a_noop(isolated_paths):
    """With no legacy trades parquet files, the indexer returns without touching the lookup."""
    trades_dir, lookup_file = isolated_paths
    trades_dir.mkdir(parents=True, exist_ok=True)

    client = FakeClient(collateral_map={}, symbol_map={})
    _run_indexer(client)

    assert not lookup_file.exists()
    assert client.collateral_calls == []
