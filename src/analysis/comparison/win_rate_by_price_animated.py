"""Animated side-by-side calibration comparison between Kalshi and Polymarket."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

from src.common.analysis import Analysis, AnalysisOutput

# Bucket size for block-to-timestamp approximation (10800 blocks ~ 6 hours at 2 sec/block)
BLOCK_BUCKET_SIZE = 10800


class WinRateByPriceAnimatedAnalysis(Analysis):
    """Animated side-by-side visualization of calibration evolution on both platforms."""

    def save(
        self,
        output_dir: Path | str,
        formats: list[str] | None = None,
        dpi: int = 100,
    ) -> dict[str, Path]:
        """Save with GIF as default format for animated output."""
        if formats is None:
            formats = ["gif", "csv"]
        return super().save(output_dir, formats, dpi)

    def __init__(
        self,
        kalshi_trades_dir: Path | str | None = None,
        kalshi_markets_dir: Path | str | None = None,
        polymarket_trades_dir: Path | str | None = None,
        polymarket_legacy_trades_dir: Path | str | None = None,
        polymarket_markets_dir: Path | str | None = None,
        polymarket_blocks_dir: Path | str | None = None,
        collateral_lookup_path: Path | str | None = None,
    ):
        super().__init__(
            name="win_rate_by_price_animated",
            description="Animated side-by-side calibration comparison between platforms",
        )
        base_dir = Path(__file__).parent.parent.parent.parent

        # Kalshi paths
        self.kalshi_trades_dir = Path(kalshi_trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.kalshi_markets_dir = Path(kalshi_markets_dir or base_dir / "data" / "kalshi" / "markets")

        # Polymarket paths
        self.polymarket_trades_dir = Path(polymarket_trades_dir or base_dir / "data" / "polymarket" / "trades")
        self.polymarket_legacy_trades_dir = Path(
            polymarket_legacy_trades_dir or base_dir / "data" / "polymarket" / "legacy_trades"
        )
        self.polymarket_markets_dir = Path(polymarket_markets_dir or base_dir / "data" / "polymarket" / "markets")
        self.polymarket_blocks_dir = Path(polymarket_blocks_dir or base_dir / "data" / "polymarket" / "blocks")
        self.collateral_lookup_path = Path(
            collateral_lookup_path or base_dir / "data" / "polymarket" / "fpmm_collateral_lookup.json"
        )

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return animated output."""
        with self.progress("Loading Kalshi aggregates"):
            kalshi_agg = self._load_kalshi_aggregates()

        with self.progress("Loading Polymarket aggregates"):
            polymarket_agg = self._load_polymarket_aggregates()

        with self.progress("Computing cumulative data"):
            kalshi_cumulative = self._compute_cumulative(kalshi_agg)
            poly_cumulative = self._compute_cumulative(polymarket_agg)

        # Get all weeks from both platforms
        all_weeks = sorted(set(kalshi_cumulative.keys()) | set(poly_cumulative.keys()))

        # Filter to days with enough data, then take every 2nd day
        valid_weeks = [
            w
            for w in all_weeks
            if kalshi_cumulative.get(w, {}).get("total", 0) >= 1000
            or poly_cumulative.get(w, {}).get("total", 0) >= 1000
        ]
        valid_weeks = valid_weeks[::2]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7))

        (poly_line,) = ax.plot([], [], color="#3B82F6", linewidth=2, label="Polymarket (Actual)")
        (kalshi_line,) = ax.plot([], [], color="#10B981", linewidth=2, label="Kalshi (Actual)")
        ax.plot([0, 100], [0, 100], linestyle="--", color="#6B7280", linewidth=1.5, label="Implied")

        info_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel("Contract Price (cents)")
        ax.set_ylabel("Actual Win Rate (%)")
        ax.set_title("Actual Win Rate vs Contract Price")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 101, 10))
        ax.set_yticks(range(0, 101, 10))
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Add pause frames
        pause_frames = 10
        total_frames = len(valid_weeks) + pause_frames

        # Pre-sort weeks for each platform for binary search
        kalshi_weeks = sorted(kalshi_cumulative.keys())
        poly_weeks = sorted(poly_cumulative.keys())

        def get_latest_data(cumulative: dict, sorted_weeks: list, target_week) -> dict:
            """Get cumulative data for the most recent week <= target_week."""
            # Find the latest week that's <= target_week
            latest = None
            for w in sorted_weeks:
                if w <= target_week:
                    latest = w
                else:
                    break
            return cumulative.get(latest, {}) if latest else {}

        def animate(frame_idx: int) -> tuple:
            week_idx = min(frame_idx, len(valid_weeks) - 1)
            week = valid_weeks[week_idx]

            # Kalshi data - get most recent available
            k_data = get_latest_data(kalshi_cumulative, kalshi_weeks, week)
            k_total = k_data.get("total", 0)
            if k_total >= 100:
                prices = sorted(k_data["by_price"].keys())
                win_rates = [100.0 * k_data["by_price"][p]["wins"] / k_data["by_price"][p]["total"] for p in prices]
                kalshi_line.set_data(prices, win_rates)
            else:
                kalshi_line.set_data([], [])

            # Polymarket data - get most recent available
            p_data = get_latest_data(poly_cumulative, poly_weeks, week)
            p_total = p_data.get("total", 0)
            if p_total >= 100:
                prices = sorted(p_data["by_price"].keys())
                win_rates = [100.0 * p_data["by_price"][p]["wins"] / p_data["by_price"][p]["total"] for p in prices]
                poly_line.set_data(prices, win_rates)
            else:
                poly_line.set_data([], [])

            info_text.set_text(week.strftime("%Y-%m-%d"))

            return poly_line, kalshi_line, info_text

        anim = FuncAnimation(
            fig,
            animate,
            frames=total_frames,
            interval=10,
            blit=False,
            repeat=False,
        )

        # Build output data from final week
        output_rows = []
        if valid_weeks:
            final_week = valid_weeks[-1]
            for platform, data in [("kalshi", kalshi_cumulative), ("polymarket", poly_cumulative)]:
                week_data = data.get(final_week, {}).get("by_price", {})
                for price, vals in week_data.items():
                    output_rows.append(
                        {
                            "platform": platform,
                            "price": price,
                            "total": vals["total"],
                            "wins": vals["wins"],
                            "win_rate": 100.0 * vals["wins"] / vals["total"],
                        }
                    )

        output_df = pd.DataFrame(output_rows)

        return AnalysisOutput(
            figure=anim,
            data=output_df,
            metadata={"total_weeks": len(valid_weeks)},
        )

    def _compute_cumulative(self, df: pd.DataFrame) -> dict:
        """Pre-compute cumulative totals/wins by price for each week."""
        if df.empty:
            return {}

        # Normalize timezone
        df = df.copy()
        df["week"] = pd.to_datetime(df["week"])
        if df["week"].dt.tz is not None:
            df["week"] = df["week"].dt.tz_convert(None)

        weeks = sorted(df["week"].unique())
        cumulative: dict = {}
        running_totals: dict[int, dict] = {}  # price -> {total, wins}

        for week in weeks:
            week_data = df[df["week"] == week]
            for _, row in week_data.iterrows():
                price = int(row["price"])
                if price not in running_totals:
                    running_totals[price] = {"total": 0, "wins": 0}
                running_totals[price]["total"] += row["total"]
                running_totals[price]["wins"] += row["wins"]

            # Store snapshot for this week
            cumulative[week] = {
                "total": sum(v["total"] for v in running_totals.values()),
                "by_price": {p: dict(v) for p, v in running_totals.items()},
            }

        return cumulative

    def _load_kalshi_aggregates(self) -> pd.DataFrame:
        """Load Kalshi trades pre-aggregated by week and price."""
        con = duckdb.connect()

        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result
                FROM '{self.kalshi_markets_dir}/*.parquet'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
            ),
            all_positions AS (
                -- Taker side
                SELECT
                    DATE_TRUNC('day', t.created_time) AS week,
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
                FROM '{self.kalshi_trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker

                UNION ALL

                -- Maker side (counterparty)
                SELECT
                    DATE_TRUNC('day', t.created_time) AS week,
                    CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                    CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
                FROM '{self.kalshi_trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            )
            SELECT week, price, COUNT(*) AS total, SUM(won) AS wins
            FROM all_positions
            WHERE price >= 1 AND price <= 99
            GROUP BY week, price
            ORDER BY week, price
            """
        ).df()

        return df

    def _load_polymarket_aggregates(self) -> pd.DataFrame:
        """Load Polymarket trades pre-aggregated by week and price, including legacy trades."""
        con = duckdb.connect()

        # Build CTF token_id -> won mapping
        markets_df = con.execute(
            f"""
            SELECT id, clob_token_ids, outcome_prices, market_maker_address
            FROM '{self.polymarket_markets_dir}/*.parquet'
            WHERE closed = true
            """
        ).df()

        token_won: dict[str, bool] = {}
        fpmm_resolution: dict[str, int] = {}

        for _, row in markets_df.iterrows():
            try:
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                if not prices or len(prices) != 2:
                    continue
                p0, p1 = float(prices[0]), float(prices[1])

                winning_outcome = None
                if p0 > 0.99 and p1 < 0.01:
                    winning_outcome = 0
                elif p0 < 0.01 and p1 > 0.99:
                    winning_outcome = 1
                else:
                    continue

                token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
                if token_ids and len(token_ids) == 2:
                    token_won[token_ids[0]] = winning_outcome == 0
                    token_won[token_ids[1]] = winning_outcome == 1

                fpmm_addr = row.get("market_maker_address")
                if fpmm_addr:
                    fpmm_resolution[fpmm_addr.lower()] = winning_outcome

            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Register CTF token mapping
        con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN)")
        con.executemany("INSERT INTO token_resolution VALUES (?, ?)", list(token_won.items()))

        # Filter FPMM to USDC markets only
        if self.collateral_lookup_path.exists():
            with open(self.collateral_lookup_path) as f:
                collateral_lookup = json.load(f)
            usdc_markets = {
                addr.lower() for addr, info in collateral_lookup.items() if info["collateral_symbol"] == "USDC"
            }
            fpmm_resolution = {k: v for k, v in fpmm_resolution.items() if k in usdc_markets}

        con.execute("CREATE TABLE fpmm_resolution (fpmm_address VARCHAR, winning_outcome BIGINT)")
        if fpmm_resolution:
            con.executemany("INSERT INTO fpmm_resolution VALUES (?, ?)", list(fpmm_resolution.items()))

        # Create blocks lookup table
        con.execute(
            f"""
            CREATE TABLE blocks AS
            SELECT
                block_number // {BLOCK_BUCKET_SIZE} AS bucket,
                FIRST(timestamp::TIMESTAMP) AS timestamp
            FROM '{self.polymarket_blocks_dir}/*.parquet'
            GROUP BY block_number // {BLOCK_BUCKET_SIZE}
            """
        )

        # CTF trades query
        ctf_trades_query = f"""
            -- Buyer side
            SELECT
                DATE_TRUNC('day', b.timestamp) AS week,
                CASE
                    WHEN t.maker_asset_id = '0' THEN ROUND(100.0 * t.maker_amount / t.taker_amount)
                    ELSE ROUND(100.0 * t.taker_amount / t.maker_amount)
                END AS price,
                CASE WHEN tr.won THEN 1 ELSE 0 END AS won
            FROM '{self.polymarket_trades_dir}/*.parquet' t
            INNER JOIN token_resolution tr ON (
                CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END = tr.token_id
            )
            JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
            WHERE t.taker_amount > 0 AND t.maker_amount > 0

            UNION ALL

            -- Seller side (counterparty)
            SELECT
                DATE_TRUNC('day', b.timestamp) AS week,
                CASE
                    WHEN t.maker_asset_id = '0' THEN ROUND(100.0 - 100.0 * t.maker_amount / t.taker_amount)
                    ELSE ROUND(100.0 - 100.0 * t.taker_amount / t.maker_amount)
                END AS price,
                CASE WHEN NOT tr.won THEN 1 ELSE 0 END AS won
            FROM '{self.polymarket_trades_dir}/*.parquet' t
            INNER JOIN token_resolution tr ON (
                CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END = tr.token_id
            )
            JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
        """

        # Legacy FPMM trades query
        legacy_trades_query = ""
        if fpmm_resolution and self.polymarket_legacy_trades_dir.exists():
            legacy_trades_query = f"""
                UNION ALL

                -- Legacy buyer side
                SELECT
                    DATE_TRUNC('day', b.timestamp) AS week,
                    ROUND(100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price,
                    CASE WHEN t.outcome_index = r.winning_outcome THEN 1 ELSE 0 END AS won
                FROM '{self.polymarket_legacy_trades_dir}/*.parquet' t
                INNER JOIN fpmm_resolution r ON LOWER(t.fpmm_address) = r.fpmm_address
                JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
                WHERE t.outcome_tokens::DOUBLE > 0

                UNION ALL

                -- Legacy seller side (counterparty)
                SELECT
                    DATE_TRUNC('day', b.timestamp) AS week,
                    ROUND(100.0 - 100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price,
                    CASE WHEN t.outcome_index != r.winning_outcome THEN 1 ELSE 0 END AS won
                FROM '{self.polymarket_legacy_trades_dir}/*.parquet' t
                INNER JOIN fpmm_resolution r ON LOWER(t.fpmm_address) = r.fpmm_address
                JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
                WHERE t.outcome_tokens::DOUBLE > 0
            """

        df = con.execute(
            f"""
            WITH trade_positions AS (
                {ctf_trades_query}
                {legacy_trades_query}
            )
            SELECT week, price, COUNT(*) AS total, SUM(won) AS wins
            FROM trade_positions
            WHERE price >= 1 AND price <= 99
            GROUP BY week, price
            ORDER BY week, price
            """
        ).df()

        return df
