"""Analyze maker-taker gap evolution over time.

Computes quarterly maker and taker excess returns to assess whether
the bias changes as the platform matures or with retail influxes.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class MakerTakerGapOverTimeAnalysis(Analysis):
    """Analyze maker-taker gap evolution over time."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="maker_taker_gap_over_time",
            description="Quarterly maker-taker excess returns over time",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
            ),
            all_positions AS (
                SELECT
                    'taker' AS role,
                    DATE_TRUNC('quarter', t.created_time) AS quarter,
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                    CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker

                UNION ALL

                SELECT
                    'maker' AS role,
                    DATE_TRUNC('quarter', t.created_time) AS quarter,
                    CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                    CASE WHEN t.taker_side != m.result THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            )
            SELECT
                role,
                quarter,
                AVG(won - price / 100.0) AS excess_return,
                VAR_POP(won - price / 100.0) AS var_excess,
                COUNT(*) AS n_trades,
                SUM(contracts * price / 100.0) AS volume_usd
            FROM all_positions
            GROUP BY role, quarter
            HAVING COUNT(*) >= 1000
            ORDER BY quarter, role
            """
        ).df()

        # Calculate standard errors and confidence intervals
        df["se"] = np.sqrt(df["var_excess"] / df["n_trades"])
        df["ci_lower"] = df["excess_return"] - 1.96 * df["se"]
        df["ci_upper"] = df["excess_return"] + 1.96 * df["se"]

        # Pivot for easier plotting
        taker_df = df[df["role"] == "taker"].copy()
        maker_df = df[df["role"] == "maker"].copy()

        # Merge to compute gap
        merged = taker_df.merge(
            maker_df,
            on="quarter",
            suffixes=("_taker", "_maker")
        )
        merged["gap"] = (merged["excess_return_maker"] - merged["excess_return_taker"]) * 100

        # Prepare output dataframe
        output_df = merged[["quarter", "excess_return_taker", "excess_return_maker", "gap",
                            "n_trades_taker", "volume_usd_taker"]].copy()
        output_df.columns = ["quarter", "taker_return", "maker_return", "gap_pp", "n_trades", "volume_usd"]
        output_df["taker_return"] = output_df["taker_return"] * 100
        output_df["maker_return"] = output_df["maker_return"] * 100

        fig = self._create_figure(merged)
        chart = self._create_chart(merged)

        return AnalysisOutput(figure=fig, data=output_df, chart=chart)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Convert to pandas timestamps for formatting
        df = df.copy()
        df["quarter"] = pd.to_datetime(df["quarter"])
        quarters = df["quarter"].values
        x = np.arange(len(quarters))
        quarter_labels = [
            f"{pd.Timestamp(q).year} Q{(pd.Timestamp(q).month-1)//3+1}"
            for q in quarters
        ]

        # Plot returns
        ax1.plot(
            x,
            df["excess_return_taker"] * 100,
            color="#e74c3c",
            linewidth=2,
            label="Taker Return",
            marker="o",
            markersize=4,
        )
        ax1.plot(
            x,
            df["excess_return_maker"] * 100,
            color="#2ecc71",
            linewidth=2,
            label="Maker Return",
            marker="o",
            markersize=4,
        )
        ax1.fill_between(x, df["excess_return_taker"] * 100, alpha=0.2, color="#e74c3c")
        ax1.fill_between(x, df["excess_return_maker"] * 100, alpha=0.2, color="#2ecc71")
        ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

        # Mark key events - 2024 Q4 = election + legal victory
        election_idx = None
        for i, q in enumerate(quarters):
            ts = pd.Timestamp(q)
            if ts.year == 2024 and ts.month == 10:
                election_idx = i
                break

        if election_idx is not None:
            ax1.axvline(x=election_idx, color="blue", linestyle=":", linewidth=1.5, alpha=0.7)
            ax1.annotate(
                "2024 Election\n& Legal Victory",
                xy=(election_idx, ax1.get_ylim()[1] * 0.8),
                fontsize=9,
                ha="center",
                color="blue",
            )

        ax1.set_xlabel("Quarter")
        ax1.set_ylabel("Excess Return (pp)")
        ax1.set_title("Maker-Taker Gap Over Time")
        ax1.set_xticks(x)
        ax1.set_xticklabels(quarter_labels, rotation=45, ha="right")
        ax1.legend(loc="upper left")

        # Add volume on secondary axis
        ax2 = ax1.twinx()
        ax2.bar(x, df["volume_usd_taker"] / 1e9, alpha=0.15, color="gray", width=0.8)
        ax2.set_ylabel("Volume ($B)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        df = df.copy()
        df["quarter"] = pd.to_datetime(df["quarter"])

        chart_data = [
            {
                "quarter": f"{pd.Timestamp(row['quarter']).year} Q{(pd.Timestamp(row['quarter']).month-1)//3+1}",
                "Taker Return": round(row["excess_return_taker"] * 100, 2),
                "Maker Return": round(row["excess_return_maker"] * 100, 2),
            }
            for _, row in df.iterrows()
        ]

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="quarter",
            yKeys=["Taker Return", "Maker Return"],
            title="Maker-Taker Returns Over Time",
            yUnit=UnitType.PERCENT,
            xLabel="Quarter",
            yLabel="Excess Return (pp)",
        )
