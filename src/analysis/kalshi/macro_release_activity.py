"""Measure Kalshi trading activity around confirmed macro releases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.fxmacrodata import (
    fetch_release_calendar,
    load_release_calendar,
    parse_release_calendar,
)

_OUTPUT_COLUMNS = [
    "release",
    "event_time_utc",
    "pre_window_start_utc",
    "post_window_end_utc",
    "pre_trade_count",
    "post_trade_count",
    "pre_contract_count",
    "post_contract_count",
    "pre_notional_usd",
    "post_notional_usd",
    "pre_unique_markets",
    "post_unique_markets",
]


class MacroReleaseActivityAnalysis(Analysis):
    """Compare Kalshi activity immediately before and after macro releases."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        release_calendar_path: Path | str | None = None,
        trades: pd.DataFrame | None = None,
        markets: pd.DataFrame | None = None,
        releases: list[dict[str, Any]] | None = None,
        currency: str = "USD",
        pre_window_minutes: int = 60,
        post_window_minutes: int = 60,
    ):
        super().__init__(
            name="macro_release_activity",
            description="Kalshi trading activity around confirmed macro releases",
        )
        if pre_window_minutes <= 0 or post_window_minutes <= 0:
            raise ValueError("release windows must be positive")

        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.release_calendar_path = Path(release_calendar_path) if release_calendar_path is not None else None
        self._trades = trades.copy() if trades is not None else None
        self._markets = markets.copy() if markets is not None else None
        self._releases = list(releases) if releases is not None else None
        self.currency = currency
        self.pre_window = pd.Timedelta(minutes=pre_window_minutes)
        self.post_window = pd.Timedelta(minutes=post_window_minutes)

    def run(self) -> AnalysisOutput:
        """Execute the event-window activity analysis."""
        trades = self._load_trades()
        markets = self._load_markets()
        releases = self._load_releases()

        rows = self._summarize_windows(trades, markets, releases)
        data = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
        figure = self._create_figure(data)
        return AnalysisOutput(
            figure=figure,
            data=data,
            metadata={
                "currency": self.currency.upper(),
                "pre_window_minutes": int(self.pre_window.total_seconds() / 60),
                "post_window_minutes": int(self.post_window.total_seconds() / 60),
                "point_in_time_rule": "confirmed scheduled timestamps only",
            },
        )

    def _load_trades(self) -> pd.DataFrame:
        if self._trades is not None:
            return self._trades.copy()
        files = sorted(self.trades_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Kalshi trade parquet files found in {self.trades_dir}")
        return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)

    def _load_markets(self) -> pd.DataFrame:
        if self._markets is not None:
            return self._markets.copy()
        files = sorted(self.markets_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Kalshi market parquet files found in {self.markets_dir}")
        return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)

    def _load_releases(self) -> list[dict[str, Any]]:
        if self._releases is not None:
            return parse_release_calendar({"data": self._releases})
        if self.release_calendar_path is not None:
            return load_release_calendar(self.release_calendar_path)
        return fetch_release_calendar(self.currency)

    def _summarize_windows(
        self,
        trades: pd.DataFrame,
        markets: pd.DataFrame,
        releases: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        required_trade_columns = {
            "ticker",
            "created_time",
            "count",
            "yes_price",
            "no_price",
            "taker_side",
        }
        missing = required_trade_columns.difference(trades.columns)
        if missing:
            raise ValueError(f"Kalshi trades are missing columns: {', '.join(sorted(missing))}")
        if "ticker" not in markets.columns:
            raise ValueError("Kalshi markets are missing the ticker column")

        prepared = trades.copy()
        prepared["created_time"] = pd.to_datetime(prepared["created_time"], utc=True, errors="coerce")
        if prepared["created_time"].isna().any():
            raise ValueError("Kalshi trades contain invalid created_time values")
        prepared["count"] = pd.to_numeric(prepared["count"], errors="raise")
        yes_side = prepared["taker_side"].astype(str).str.lower().eq("yes")
        taker_price = prepared["yes_price"].where(yes_side, prepared["no_price"])
        prepared["notional_usd"] = prepared["count"] * pd.to_numeric(taker_price, errors="raise") / 100

        known_tickers = set(markets["ticker"].dropna().astype(str))
        prepared = prepared[prepared["ticker"].astype(str).isin(known_tickers)]

        summaries = []
        for release in releases:
            if release.get("release_date_confirmed") is not True:
                continue
            if release.get("release_time_assumed") is True:
                continue

            event_time = pd.to_datetime(release["announcement_datetime"], unit="s", utc=True)
            pre_start = event_time - self.pre_window
            post_end = event_time + self.post_window
            before = prepared[prepared["created_time"].ge(pre_start) & prepared["created_time"].lt(event_time)]
            after = prepared[prepared["created_time"].ge(event_time) & prepared["created_time"].le(post_end)]
            summaries.append(
                {
                    "release": release.get("release") or release.get("name") or "unknown",
                    "event_time_utc": event_time,
                    "pre_window_start_utc": pre_start,
                    "post_window_end_utc": post_end,
                    "pre_trade_count": len(before),
                    "post_trade_count": len(after),
                    "pre_contract_count": before["count"].sum(),
                    "post_contract_count": after["count"].sum(),
                    "pre_notional_usd": before["notional_usd"].sum(),
                    "post_notional_usd": after["notional_usd"].sum(),
                    "pre_unique_markets": before["ticker"].nunique(),
                    "post_unique_markets": after["ticker"].nunique(),
                }
            )
        return summaries

    def _create_figure(self, data: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 6))
        if data.empty:
            ax.text(
                0.5,
                0.5,
                "No confirmed releases in the selected window",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            return fig

        labels = [f"{row.release}\n{row.event_time_utc:%Y-%m-%d}" for row in data.itertuples(index=False)]
        positions = list(range(len(data)))
        width = 0.4
        ax.bar(
            [position - width / 2 for position in positions],
            data["pre_trade_count"],
            width=width,
            label="Before release",
        )
        ax.bar(
            [position + width / 2 for position in positions],
            data["post_trade_count"],
            width=width,
            label="After release",
        )
        ax.set_xticks(positions, labels, rotation=45, ha="right")
        ax.set_ylabel("Trade count")
        ax.set_title("Kalshi activity around confirmed macro releases")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig
