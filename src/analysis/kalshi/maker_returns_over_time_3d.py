from __future__ import annotations

import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from src.analysis.kalshi.util.events import DEFAULT_EVENTS_DIR, prompt_event_category
from src.common.analysis import Analysis, AnalysisOutput
from src.common.util.prompts import prompt_choice

pd.set_option("display.max_rows", 100)


class MakerReturnsOverTime3DAnalysis(Analysis):
    """Analyze maker returns by position direction (YES vs NO) and event category over time.

    Shows an interactive 3D bar chart: price (x) by week (y) by maker excess return (z),
    with a 3-week rolling average line per price level.
    """

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        events_dir: Path | str | None = None,
    ):
        super().__init__(
            name="3d_maker_returns_over_time",
            description="Maker excess returns by position direction (YES vs NO) and event category over time",
        )
        self.output_type = "3D"

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

        yes_no = prompt_choice("Maker side to plot (yes or no): ", ["yes", "no"])
        self.yes_no = yes_no

        date_range_qry_str, self.start_date, self.end_date = self._get_date_range_query_string()

        self.name = f"{self.name}_{category.replace(' ', '_').lower()}_{yes_no}"

        # Date span to plot: days with at least 100 resolved markets in this category
        df_market_dates = con.execute(
            f"""
            WITH resolved_markets as(
                SELECT DATE(m.close_time) as date, count(*) as n_markets
                FROM '{self.markets_dir}/*.parquet' m
                INNER JOIN '{self.events_dir}/events_*.parquet' e ON e.event_ticker = m.event_ticker
                WHERE 1=1
                AND m.status = 'finalized'
                AND m.volume >= 100
                AND e.category = '{category}'
                {date_range_qry_str}
                GROUP BY DATE(m.close_time)
                HAVING count(*) >= 100
                ORDER BY date ASC
            )
            SELECT MIN(date) as min,
                    MAX(date) as max
            FROM resolved_markets
            """
        ).df()

        min_date = df_market_dates["min"][0]
        max_date = df_market_dates["max"][0]

        if pd.isna(min_date):
            print(f"No days with at least 100 resolved '{category}' markets in this range, nothing to plot.")
            return AnalysisOutput()

        self.start_date = min_date
        self.end_date = max_date

        delta = max_date - min_date

        weeks = math.ceil(delta.days / 7) + 1

        markets_date_range = pd.date_range(start=min_date, periods=weeks, freq="7D")

        values = []  # Initialize an empty array to store the values

        for day, next_day in zip(markets_date_range, markets_date_range[1:]):
            print(f"Processing: {day.strftime('%Y-%m-%d')} thru: {next_day.strftime('%Y-%m-%d')}")

            df = con.execute(
                f"""
                WITH resolved_markets AS (
                    SELECT m.ticker, m.result
                    FROM '{self.markets_dir}/*.parquet' m
                    INNER JOIN '{self.events_dir}/events_*.parquet' e ON e.event_ticker = m.event_ticker
                    WHERE m.status = 'finalized'
                      AND m.result IN ('yes', 'no')
                      AND e.category = '{category}'
                      AND m.close_time >= '{day.strftime("%Y-%m-%d")}'
                      AND m.close_time < '{next_day.strftime("%Y-%m-%d")}'
                ),
                maker_yes_positions AS (
                    -- Maker bought YES (taker sold YES = taker bought NO)
                    SELECT
                        t.yes_price AS price,
                        CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END AS won,
                        t.count AS contracts,
                        'YES' AS maker_side
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
                        'NO' AS maker_side
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
                    price,
                    AVG(won) AS win_rate,
                    price / 100.0 AS expected_win_rate,
                    AVG(won) - price / 100.0 AS excess_return,
                    VAR_POP(won - price / 100.0) AS var_excess,
                    COUNT(*) AS n_trades,
                    SUM(contracts) AS contracts,
                    SUM(contracts * price / 100.0) AS volume_usd
                FROM all_maker_positions
                WHERE price BETWEEN 1 AND 99
                GROUP BY maker_side, price
                ORDER BY maker_side, price
                """
            ).df()

            if self.yes_no == "yes":
                df_result = df[df["maker_side"] == "YES"]
            else:
                df_result = df[df["maker_side"] == "NO"]

            df_result.set_index("price", inplace=True)
            df_result = df_result.reindex(range(1, 100), fill_value=0)
            values.append(df_result["excess_return"].tolist())

        values = np.array(values)  # Convert the list of lists to a 2D NumPy array

        print("Preparing figure...")

        fig = self._create_figure(z_values=values, y_values=markets_date_range)

        return AnalysisOutput(figure=fig, data=None, chart=None)

    def _create_figure(self, z_values: np.ndarray, y_values: np.ndarray) -> plt.Figure:
        fig = plt.figure(figsize=(50, 50))
        ax = fig.add_subplot(111, projection="3d")

        xlabels = np.array(range(1, 100))  # Prices from 1 to 99
        xpos = np.arange(xlabels.shape[0])

        ylabels = [
            f"{date.strftime('%Y-%m-%d')} - {next_date.strftime('%m-%d')}"
            for date, next_date in zip(y_values, y_values[1:])
        ]
        ypos = np.arange(y_values[:-1].shape[0])

        xpos_grid, ypos_grid = np.meshgrid(xpos, ypos, copy=False)

        zpos = z_values.ravel()

        dx = 0.5
        dy = dx * (len(ylabels) / len(xlabels))  # Adjust dy based on the ratio of ylabels to xlabels
        dz = zpos

        x = xpos_grid.ravel()

        xlabels = [label if (i + 1) % 5 == 0 else "" for i, label in enumerate(xlabels)]

        ax.xaxis.set_ticks(xpos + dx / 2.0)
        ax.xaxis.set_ticklabels(xlabels)

        y = ypos_grid.ravel()

        ax.yaxis.set_ticks(ypos + dy / 2.0)
        ax.yaxis.set_ticklabels(ylabels)

        ax.set_zlim(0, dz.max() * 1.1)  # Set z-axis limit slightly above the max value

        values = (dz - dz.min()) / np.float64(dz.max() - dz.min())

        bar_colors = cm.plasma(values)

        ax.set_box_aspect((2, 2, 1))

        bars = ax.bar3d(x, y, dz * 0, dx, dy, dz, shade=True, color=bar_colors, alpha=0.8)

        ax.set_xlabel("Price (cents)")
        ax.set_ylabel("Date")
        ax.set_zlabel("Excess Return (pp)")

        category_label = self.category

        ax.set_title(
            f"Maker Excess Returns by Position Direction ({self.yes_no.upper()})\nEvent Category: {category_label}, {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}"
        )

        z_average_final = []
        lcs = []

        for x in xpos:
            ys = ypos

            zs = z_values[:, x]

            z_average = pd.Series(zs).rolling(window=3).mean().to_numpy()

            points = np.array([np.full_like(ys, x), ys, z_average]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = Normalize(zpos.min(), zpos.max())
            lc = Line3DCollection(segments, cmap="viridis", norm=norm, linewidths=2)
            lc.set_array(z_average)  # Sets the values used for colormapping

            ax.add_collection(lc)

            lcs.append(lc)
            z_average_final.append(z_average[-1])

        ax.scatter(
            xpos,
            ypos[-1],
            z_average_final,
            c=z_average_final,
            cmap="viridis",
            norm=norm,
            s=50,
            label="3-Point Rolling Average",
        )
        ax.plot(xpos, ypos[-1], z_average_final, color="black", linewidth=1.5)

        def toggle_bars(event):
            bars.set_visible(not bars.get_visible())
            plt.draw()

        def toggle_lines(event):
            for lc in lcs:
                lc.set_visible(not lc.get_visible())
            plt.draw()

        ax_btn_bars = fig.add_axes((0.7, 0.05, 0.1, 0.075))
        btn_bars = Button(ax_btn_bars, "Show/Hide Bars")
        btn_bars.on_clicked(toggle_bars)

        ax_btn_lines = fig.add_axes((0.7, 0.15, 0.1, 0.075))
        btn_lines = Button(ax_btn_lines, "Show/Hide Moving Average")
        btn_lines.on_clicked(toggle_lines)

        plt.show()

        return fig
