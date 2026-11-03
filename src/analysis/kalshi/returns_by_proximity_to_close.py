"""
Analyze the excess returns of maker trades based on their proximity to the market's closing time.
Also segregates the analysis by event category and position direction (YES vs NO).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.util.events import DEFAULT_EVENTS_DIR, prompt_event_category
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class ReturnsByProximityToCloseAnalysis(Analysis):
    """Analyze the excess returns of maker trades based on their proximity to the market's closing time."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        events_dir: Path | str | None = None,
    ):
        super().__init__(
            name="returns_by_proximity_to_close",
            description="Excess returns of maker trades by proximity to market close",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.events_dir = Path(events_dir or DEFAULT_EVENTS_DIR)
        self.category: str | None = None
        self.start_date = None
        self.end_date = None

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        category = prompt_event_category(self.events_dir)
        self.category = category

        date_range_qry_str, self.start_date, self.end_date = self._get_date_range_query_string()

        self.name = f"maker_returns_by_proximity_to_close_event_category_{category.replace(' ', '_').lower()}"

        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT m.ticker, m.result, m.close_time
                FROM '{self.markets_dir}/*.parquet' m
                INNER JOIN '{self.events_dir}/events_*.parquet' e ON e.event_ticker = m.event_ticker
                WHERE m.status = 'finalized'
                  AND m.result IN ('yes', 'no')
                  AND e.category = '{category}'
                  {date_range_qry_str}
            ),
            maker_yes_positions AS (
                -- Maker bought YES (taker sold YES = taker bought NO)
                SELECT
                    t.yes_price AS price,
                    CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts,
                    'YES' AS maker_side,
                    DATE_DIFF('day', t.created_time, m.close_time) AS days_to_close
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.taker_side = 'no'
            ),
            maker_no_positions AS (
                -- Maker bought NO (taker sold NO = taker bought YES)
                SELECT
                    t.no_price AS price,
                    CASE WHEN m.result = 'no' THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts,
                    'NO' AS maker_side,
                    DATE_DIFF('day', t.created_time, m.close_time) AS days_to_close
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.taker_side = 'yes'
            ),
            all_maker_positions AS (
                SELECT * FROM maker_yes_positions
                UNION ALL
                SELECT * FROM maker_no_positions
            )
            SELECT
                maker_side,
                days_to_close,
                AVG(price) AS avg_price,
                AVG(won) AS win_rate,
                AVG(price) / 100.0 AS avg_expected_win_rate,
                AVG(won) - AVG(price) / 100.0 AS avg_excess_return,
                VAR_POP(won - price / 100.0) AS var_excess,
                COUNT(*) AS n_trades,
                SUM(contracts) AS contracts,
                SUM(contracts * price / 100.0) AS volume_usd
            FROM all_maker_positions
            WHERE 1=1
              AND days_to_close BETWEEN 0 AND 30
            GROUP BY maker_side, days_to_close
            ORDER BY maker_side, days_to_close
            """
        ).df()

        # Pivot to compare YES vs NO at each price
        df_yes = df[df["maker_side"] == "YES"].copy()
        df_no = df[df["maker_side"] == "NO"].copy()

        comparison = pd.merge(
            df_yes[["days_to_close", "win_rate", "avg_excess_return", "n_trades", "contracts", "volume_usd"]].rename(
                columns={
                    "win_rate": "yes_win_rate",
                    "avg_excess_return": "yes_excess",
                    "n_trades": "yes_n",
                    "contracts": "yes_contracts",
                    "volume_usd": "yes_volume",
                }
            ),
            df_no[["days_to_close", "win_rate", "avg_excess_return", "n_trades", "contracts", "volume_usd"]].rename(
                columns={
                    "win_rate": "no_win_rate",
                    "avg_excess_return": "no_excess",
                    "n_trades": "no_n",
                    "contracts": "no_contracts",
                    "volume_usd": "no_volume",
                }
            ),
            on="days_to_close",
            how="outer",
        )
        comparison = comparison.sort_values("days_to_close")
        comparison["diff"] = comparison["no_excess"] - comparison["yes_excess"]

        fig = self._create_figure(comparison)
        chart = self._create_chart(comparison)

        return AnalysisOutput(figure=fig, data=comparison, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(
            df["days_to_close"],
            df["yes_excess"] * 100,
            color="#2ecc71",
            linewidth=1.5,
            label="Maker bought YES",
            alpha=0.8,
        )
        ax.plot(
            df["days_to_close"],
            df["no_excess"] * 100,
            color="#e74c3c",
            linewidth=1.5,
            label="Maker bought NO",
            alpha=0.8,
        )
        ax.fill_between(df["days_to_close"], df["yes_excess"] * 100, alpha=0.2, color="#2ecc71")
        ax.fill_between(df["days_to_close"], df["no_excess"] * 100, alpha=0.2, color="#e74c3c")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Days to Close")
        ax.set_ylabel("Excess Return (pp)")

        category_label = self.category

        if self.start_date and self.end_date:
            ax.set_title(
                f"Maker Excess Returns by Proximity to Close\nEvent Category: {category_label}, {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}"
            )
        elif self.start_date:
            ax.set_title(
                f"Maker Excess Returns by Proximity to Close\nEvent Category: {category_label}, from {self.start_date.strftime('%Y-%m-%d')}"
            )
        elif self.end_date:
            ax.set_title(
                f"Maker Excess Returns by Proximity to Close\nEvent Category: {category_label}, up to {self.end_date.strftime('%Y-%m-%d')}"
            )
        else:
            ax.set_title("Maker Excess Returns by Proximity to Close")

        ax.set_xlim(0, 30)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = [
            {
                "days_to_close": int(row["days_to_close"]),
                "Maker bought YES": round(row["yes_excess"] * 100, 2) if pd.notna(row["yes_excess"]) else None,
                "Maker bought NO": round(row["no_excess"] * 100, 2) if pd.notna(row["no_excess"]) else None,
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="days_to_close",
            yKeys=["Maker bought YES", "Maker bought NO"],
            title="Maker Excess Returns by Proximity to Close",
            yUnit=UnitType.PERCENT,
            xLabel="Days to Close",
            yLabel="Excess Return (pp)",
        )
