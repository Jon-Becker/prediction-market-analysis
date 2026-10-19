"""Regression coverage for the binary-settlement sample audit (issue #74)."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.analysis.kalshi.maker_taker_returns_by_category import MakerTakerReturnsByCategoryAnalysis


def _analysis(tmp_path, markets, trades):
    markets_dir = tmp_path / "markets"
    trades_dir = tmp_path / "trades"
    markets_dir.mkdir(exist_ok=True)
    trades_dir.mkdir(exist_ok=True)
    pd.DataFrame(markets, columns=["ticker", "event_ticker", "status", "result"]).to_parquet(
        markets_dir / "markets.parquet"
    )
    pd.DataFrame(trades, columns=["ticker", "taker_side", "yes_price", "no_price", "count"]).to_parquet(
        trades_dir / "trades.parquet"
    )
    return MakerTakerReturnsByCategoryAnalysis(trades_dir, markets_dir)


def test_exclusions_are_counted_without_changing_returns(tmp_path):
    markets = [
        ("yes", "NFLGAME-1", "finalized", "yes"),
        ("no", "NFLGAME-2", "finalized", "no"),
    ]
    trades = [("yes", "yes", 40, 60, 2), ("no", "no", 70, 30, 5)]
    baseline = _analysis(tmp_path, markets, trades).run()
    assert baseline.metadata["settlement_coverage"]["totals"] == {
        "included_markets": 2,
        "excluded_markets": 0,
        "included_trades": 2,
        "excluded_trades": 0,
    }

    markets += [
        ("tie", "KXNFLGAME-3", "finalized", "scalar"),
        ("weather", "HIGHNY-1", "finalized", "scalar"),
        ("no-trades", "HIGHNY-2", "finalized", "scalar"),
        ("unknown", "NFLGAME-4", "finalized", None),
        ("empty", "NFLGAME-5", "finalized", ""),
        ("future-result", "NFLGAME-6", "finalized", "other"),
        ("open", "NFLGAME-7", "active", "scalar"),
    ]
    trades += [
        ("tie", "yes", 30, 70, 100),
        ("tie", "no", 80, 20, 200),
        ("weather", "yes", 50, 50, 300),
        ("unknown", "yes", 50, 50, 400),
        ("empty", "yes", 50, 50, 500),
        ("future-result", "yes", 50, 50, 600),
        ("open", "yes", 50, 50, 700),
        ("unmatched", "yes", 50, 50, 800),
    ]
    analysis = _analysis(tmp_path, markets, trades)
    output = analysis.run()
    pd.testing.assert_frame_equal(output.data, baseline.data)
    assert output.chart.to_dict() == baseline.chart.to_dict()
    audit = output.metadata["settlement_coverage"]
    assert audit["totals"] == {"included_markets": 2, "excluded_markets": 6, "included_trades": 2, "excluded_trades": 6}
    assert audit["by_group"] == [
        {"group": "Sports", "included_markets": 2, "excluded_markets": 4, "included_trades": 2, "excluded_trades": 5},
        {"group": "Weather", "included_markets": 0, "excluded_markets": 2, "included_trades": 0, "excluded_trades": 1},
    ]
    # The ordinary save path must expose the audit, not just the in-memory result.
    saved = analysis.save(tmp_path / "output")
    assert json.loads(saved["metadata"].read_text()) == output.metadata
    plt.close(baseline.figure)
    plt.close(output.figure)


@pytest.mark.parametrize("status,excluded", [("finalized", 1), ("active", 0)])
def test_audit_survives_no_binary_settlements(tmp_path, status, excluded):
    analysis = _analysis(
        tmp_path,
        [("tie", "NFLGAME-1", status, "scalar")],
        [("tie", "yes", 40, 60, 10)],
    )
    output = analysis.run()
    assert output.data.empty
    assert output.chart.data == []
    assert output.metadata["settlement_coverage"]["totals"] == {
        "included_markets": 0,
        "excluded_markets": excluded,
        "included_trades": 0,
        "excluded_trades": excluded,
    }
    plt.close(output.figure)
