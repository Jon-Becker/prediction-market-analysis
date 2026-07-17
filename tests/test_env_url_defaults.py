"""The API URL constants must resolve to the production endpoints whenever no
environment overrides are set (the overrides exist for the e2e mock harness)."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CHECK = """
from src.indexers.polymarket import client, orderbook

assert client.GAMMA_API_URL == "https://gamma-api.polymarket.com", client.GAMMA_API_URL
assert client.CLOB_API_URL == "https://clob.polymarket.com", client.CLOB_API_URL
assert client.DATA_API_URL == "https://data-api.polymarket.com", client.DATA_API_URL
assert orderbook.WS_MARKET_URL == "wss://ws-subscriptions-clob.polymarket.com/ws/market", orderbook.WS_MARKET_URL
"""


def test_url_constants_default_to_production_endpoints(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("POLYMARKET_")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", CHECK],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
