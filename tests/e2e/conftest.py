"""Fixtures for the Compose end-to-end suite.

These tests drive the real production classes (real httpx, web3, and
websockets clients) against the mock service container; the only test-side
injection is environment-based URL configuration, done by compose.e2e.yaml.
"""

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

E2E_DIR = Path(__file__).resolve().parent
REPO_ROOT = E2E_DIR.parents[1]

REQUIRED_ENV = [
    "E2E_MOCK_ADMIN_URL",
    "POLYMARKET_GAMMA_URL",
    "POLYMARKET_CLOB_URL",
    "POLYMARKET_DATA_API_URL",
    "POLYMARKET_WS_URL",
    "POLYGON_RPC",
]


def pytest_collection_modifyitems(items):
    for item in items:
        if E2E_DIR in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session", autouse=True)
def _require_harness():
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        pytest.skip(f"e2e harness not configured; missing env vars: {', '.join(missing)}")


@pytest.fixture(scope="session")
def fx():
    """The mock's fixture corpus, loaded from the exact module the mock serves."""
    spec = importlib.util.spec_from_file_location("e2e_mock_fixtures", E2E_DIR / "mock" / "fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MockAdmin:
    """Client for the mock's admin endpoints (request accounting, fault injection)."""

    def __init__(self, base_url: str):
        self.http = httpx.Client(base_url=base_url, timeout=10.0)

    def close(self):
        self.http.close()

    def reset(self):
        self.http.post("/admin/reset").raise_for_status()

    def counts(self) -> dict:
        return self.http.get("/admin/requests").json()["counts"]

    def count(self, key: str) -> int:
        return self.counts().get(key, 0)

    def events(self, key: str) -> list:
        events = self.http.get("/admin/requests").json()["events"]
        return [e for e in events if e["key"] == key]

    def fail_next(self, key: str, status: int = 500, times: int = 1):
        self.http.post("/admin/fail-next", json={"key": key, "status": status, "times": times}).raise_for_status()

    def set_latency(self, key: str, seconds: float):
        self.http.post("/admin/latency", json={"key": key, "seconds": seconds}).raise_for_status()

    def ws_state(self) -> dict:
        return self.http.get("/admin/ws").json()


@pytest.fixture(scope="session")
def admin(_require_harness):
    client = MockAdmin(os.environ["E2E_MOCK_ADMIN_URL"])
    yield client
    client.close()


@pytest.fixture(autouse=True)
def _reset_mock(_require_harness, admin):
    admin.reset()


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """Fresh working directory; the indexers' data/cursor paths are CWD-relative."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def seeded_markets_dir(tmp_path_factory, _require_harness):
    """A workspace where a real markets-indexer run has already completed,
    for scenarios that consume the markets dataset (price history, Data API)."""
    from src.indexers.polymarket.markets import PolymarketMarketsIndexer

    path = tmp_path_factory.mktemp("seeded-markets")
    cwd = os.getcwd()
    os.chdir(path)
    try:
        PolymarketMarketsIndexer().run()
    finally:
        os.chdir(cwd)
    return path


@pytest.fixture()
def seeded_workdir(seeded_markets_dir, monkeypatch):
    monkeypatch.chdir(seeded_markets_dir)
    return seeded_markets_dir


@pytest.fixture(scope="session")
def wait_until():
    def _wait(predicate, timeout: float = 60.0, interval: float = 0.01, message: str = "condition"):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(interval)
        raise AssertionError(f"timed out after {timeout}s waiting for {message}")

    return _wait


@pytest.fixture(scope="session")
def run_indexer():
    """Launch an indexer in a real subprocess so interrupts are delivered as
    actual SIGINT signals, not exceptions raised from inside a fake."""

    def _run(code: str, cwd: Path) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=str(cwd),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return _run


@pytest.fixture(scope="session")
def interrupt_and_wait():
    def _interrupt(proc: subprocess.Popen, timeout: float = 120.0) -> str:
        proc.send_signal(signal.SIGINT)
        output, _ = proc.communicate(timeout=timeout)
        return output

    return _interrupt
