"""Tests for the Polymarket price history indexer (no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import src.indexers.polymarket.price_history as mod
from src.indexers.polymarket.models import PricePoint
from src.indexers.polymarket.price_history import PolymarketPriceHistoryIndexer

CREATED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 1, 15, tzinfo=timezone.utc)
BASE_TS = int(CREATED_AT.timestamp())


class FakeClient:
    """Stub for PolymarketClient that returns canned histories and records calls."""

    def __init__(self, histories: dict[str, list[tuple[int, float]]] | None = None, errors: set[str] | None = None):
        self.histories = histories or {}
        self.errors = errors or set()
        self.calls: list[tuple[str, str | None, int | None, int | None, int | None]] = []

    def get_price_history(self, token_id, interval="max", fidelity=None, start_ts=None, end_ts=None):
        self.calls.append((token_id, interval, fidelity, start_ts, end_ts))
        if token_id in self.errors:
            raise RuntimeError("API unavailable")
        return [
            PricePoint(token_id=token_id, timestamp=t, price=p)
            for t, p in self.histories.get(token_id, [])
            if start_ts <= t <= end_ts
        ]

    def close(self):
        pass


@pytest.fixture()
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point DATA_DIR and MARKETS_DIR at temp directories."""
    data_dir = tmp_path / "price_history"
    markets_dir = tmp_path / "markets"
    markets_dir.mkdir()
    monkeypatch.setattr(mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(mod, "MARKETS_DIR", markets_dir)
    return data_dir, markets_dir


def write_markets(markets_dir: Path, rows: list[dict]) -> None:
    records = [
        {
            "id": str(i),
            "clob_token_ids": row.get("clob_token_ids", "[]"),
            "volume": row.get("volume", 0.0),
            "created_at": row.get("created_at", CREATED_AT),
            "end_date": row.get("end_date", END_DATE),
        }
        for i, row in enumerate(rows)
    ]
    pd.DataFrame(records).to_parquet(markets_dir / f"markets_0_{len(records)}.parquet")


def run_indexer(fake: FakeClient, max_workers: int = 2) -> None:
    with patch.object(mod, "PolymarketClient", return_value=fake):
        PolymarketPriceHistoryIndexer(max_workers=max_workers).run()


def load_prices(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("prices_*.parquet"))
    assert files, "expected at least one prices parquet chunk"
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def test_fetches_both_outcome_tokens(isolated_dirs):
    data_dir, markets_dir = isolated_dirs
    write_markets(markets_dir, [{"clob_token_ids": '["111", "222"]'}])
    fake = FakeClient(histories={"111": [(BASE_TS + 60, 0.65)], "222": [(BASE_TS + 60, 0.35)]})

    run_indexer(fake)

    assert {call[0] for call in fake.calls} == {"111", "222"}
    df = load_prices(data_dir)
    assert sorted(df["token_id"].unique()) == ["111", "222"]


def test_covers_all_markets_regardless_of_volume(isolated_dirs):
    data_dir, markets_dir = isolated_dirs
    write_markets(
        markets_dir,
        [
            {"clob_token_ids": '["111", "222"]', "volume": 0.0},
            {"clob_token_ids": '["333", "444"]', "volume": 250000.0},
        ],
    )
    histories = {t: [(BASE_TS + 60, 0.5)] for t in ["111", "222", "333", "444"]}
    fake = FakeClient(histories=histories)

    run_indexer(fake)

    assert {call[0] for call in fake.calls} == {"111", "222", "333", "444"}
    df = load_prices(data_dir)
    assert sorted(df["token_id"].unique()) == ["111", "222", "333", "444"]


def test_malformed_token_arrays_are_skipped(isolated_dirs):
    data_dir, markets_dir = isolated_dirs
    write_markets(
        markets_dir,
        [
            {"clob_token_ids": "not json"},
            {"clob_token_ids": '"111"'},  # valid JSON but not a list
            {"clob_token_ids": '[123, null, ""]'},  # non-string and empty entries
            {"clob_token_ids": None},
            {"clob_token_ids": '["555"]'},
        ],
    )
    fake = FakeClient(histories={"555": [(BASE_TS + 60, 0.9)]})

    run_indexer(fake)

    assert {call[0] for call in fake.calls} == {"555"}
    df = load_prices(data_dir)
    assert df["token_id"].unique().tolist() == ["555"]


def test_resume_skips_already_stored_tokens(isolated_dirs):
    data_dir, markets_dir = isolated_dirs
    data_dir.mkdir(parents=True)
    existing = pd.DataFrame([{"token_id": "111", "timestamp": BASE_TS, "price": 0.5, "_fetched_at": datetime.utcnow()}])
    existing.to_parquet(data_dir / "prices_0_10000.parquet")
    write_markets(markets_dir, [{"clob_token_ids": '["111", "222"]'}])
    fake = FakeClient(histories={"222": [(BASE_TS + 60, 0.7)]})

    run_indexer(fake)

    assert {call[0] for call in fake.calls} == {"222"}
    files = sorted(f.name for f in data_dir.glob("prices_*.parquet"))
    assert files == ["prices_0_10000.parquet", "prices_10000_20000.parquet"]
    new_chunk = pd.read_parquet(data_dir / "prices_10000_20000.parquet")
    assert new_chunk["token_id"].unique().tolist() == ["222"]


def test_rerun_after_completion_is_noop(isolated_dirs):
    data_dir, markets_dir = isolated_dirs
    write_markets(markets_dir, [{"clob_token_ids": '["111", "222"]'}])
    histories = {"111": [(BASE_TS + 60, 0.65)], "222": [(BASE_TS + 60, 0.35)]}

    run_indexer(FakeClient(histories=histories))
    rerun = FakeClient(histories=histories)
    run_indexer(rerun)

    assert rerun.calls == []
    assert len(list(data_dir.glob("prices_*.parquet"))) == 1


def test_api_failure_skips_token_and_retries_next_run(isolated_dirs):
    data_dir, markets_dir = isolated_dirs
    write_markets(markets_dir, [{"clob_token_ids": '["111", "222"]'}])
    histories = {"111": [(BASE_TS + 60, 0.65)], "222": [(BASE_TS + 60, 0.35)]}
    failing = FakeClient(histories=histories, errors={"111"})

    run_indexer(failing)

    df = load_prices(data_dir)
    assert df["token_id"].unique().tolist() == ["222"]

    recovered = FakeClient(histories=histories)
    run_indexer(recovered)

    assert {call[0] for call in recovered.calls} == {"111"}
    df = load_prices(data_dir)
    assert sorted(df["token_id"].unique()) == ["111", "222"]


def test_parquet_schema_and_values(isolated_dirs):
    data_dir, markets_dir = isolated_dirs
    write_markets(markets_dir, [{"clob_token_ids": '["111"]'}])
    fake = FakeClient(histories={"111": [(BASE_TS + 60, 0.65), (BASE_TS + 120, 0.66)]})

    run_indexer(fake, max_workers=1)

    df = load_prices(data_dir)
    assert list(df.columns) == ["token_id", "timestamp", "price", "_fetched_at"]
    assert df["token_id"].dtype == object
    assert df["timestamp"].dtype == "int64"
    assert df["price"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(df["_fetched_at"])
    assert df.sort_values("timestamp")[["timestamp", "price"]].values.tolist() == [
        [BASE_TS + 60, 0.65],
        [BASE_TS + 120, 0.66],
    ]


def test_long_histories_are_windowed_and_deduplicated(isolated_dirs):
    data_dir, markets_dir = isolated_dirs
    end_date = CREATED_AT + timedelta(days=40)  # spans two request windows
    write_markets(markets_dir, [{"clob_token_ids": '["111"]', "created_at": CREATED_AT, "end_date": end_date}])
    boundary_ts = BASE_TS + mod.WINDOW_SECONDS
    fake = FakeClient(histories={"111": [(BASE_TS + 60, 0.5), (boundary_ts, 0.6), (boundary_ts + 60, 0.7)]})

    run_indexer(fake, max_workers=1)

    expected_end = int(end_date.timestamp()) + mod.END_PADDING_SECONDS
    assert fake.calls == [
        ("111", None, mod.FIDELITY_MINUTES, BASE_TS, boundary_ts),
        ("111", None, mod.FIDELITY_MINUTES, boundary_ts, expected_end),
    ]
    df = load_prices(data_dir)
    assert len(df) == 3, "boundary point returned by both windows should be stored once"
    assert sorted(df["timestamp"].tolist()) == [BASE_TS + 60, boundary_ts, boundary_ts + 60]
