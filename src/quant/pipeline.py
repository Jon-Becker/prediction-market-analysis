"""Data loading pipeline optimized for 33GB+ parquet datasets.

Uses DuckDB for zero-copy reads and lazy SQL evaluation over parquet files,
avoiding the need to load the entire dataset into memory.
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

from src.quant.config import PipelineConfig

logger = logging.getLogger(__name__)


class DataPipeline:
    """Efficient data loading and preprocessing for large parquet datasets."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.con = duckdb.connect()
        # Set memory limit to avoid OOM on large datasets
        self.con.execute("SET memory_limit = '4GB'")
        self.con.execute("SET threads = 4")

    def close(self):
        self.con.close()

    def load_trades_with_outcomes(self) -> pd.DataFrame:
        """Load trades joined with market outcomes for resolved markets.

        Returns only trades from finalized markets with known results,
        enriched with outcome information needed for feature engineering.
        """
        trades_glob = f"{self.config.trades_dir}/*.parquet"
        markets_glob = f"{self.config.markets_dir}/*.parquet"

        query = f"""
            WITH resolved AS (
                SELECT ticker, result
                FROM '{markets_glob}'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
            )
            SELECT
                t.trade_id,
                t.ticker,
                t.count,
                t.yes_price,
                t.no_price,
                t.taker_side,
                t.created_time,
                m.result,
                -- derived columns
                CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                     ELSE t.no_price END AS taker_price,
                CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS taker_won,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                                ELSE t.no_price END) / 100.0 AS taker_notional
            FROM '{trades_glob}' t
            INNER JOIN resolved m ON t.ticker = m.ticker
            ORDER BY t.created_time
        """
        logger.info("Loading trades with outcomes (this may take a few minutes for large datasets)...")
        df = self.con.execute(query).df()
        logger.info("Loaded %s trades across %s tickers", f"{len(df):,}", f"{df['ticker'].nunique():,}")
        return df

    def load_ticker_trades(self, ticker: str) -> pd.DataFrame:
        """Load all trades for a single ticker, ordered by time."""
        trades_glob = f"{self.config.trades_dir}/*.parquet"
        markets_glob = f"{self.config.markets_dir}/*.parquet"

        query = f"""
            WITH resolved AS (
                SELECT ticker, result
                FROM '{markets_glob}'
                WHERE ticker = '{ticker}'
                  AND status = 'finalized'
                  AND result IN ('yes', 'no')
            )
            SELECT
                t.trade_id, t.ticker, t.count, t.yes_price, t.no_price,
                t.taker_side, t.created_time, m.result,
                CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                     ELSE t.no_price END AS taker_price,
                CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS taker_won,
                t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                                ELSE t.no_price END) / 100.0 AS taker_notional
            FROM '{trades_glob}' t
            INNER JOIN resolved m ON t.ticker = m.ticker
            WHERE t.ticker = '{ticker}'
            ORDER BY t.created_time
        """
        return self.con.execute(query).df()

    def get_active_tickers(self) -> pd.DataFrame:
        """Get tickers with sufficient trade count for modeling."""
        trades_glob = f"{self.config.trades_dir}/*.parquet"
        markets_glob = f"{self.config.markets_dir}/*.parquet"

        query = f"""
            WITH resolved AS (
                SELECT ticker, result
                FROM '{markets_glob}'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
            )
            SELECT
                t.ticker,
                m.result,
                COUNT(*) AS trade_count,
                MIN(t.created_time) AS first_trade,
                MAX(t.created_time) AS last_trade
            FROM '{trades_glob}' t
            INNER JOIN resolved m ON t.ticker = m.ticker
            GROUP BY t.ticker, m.result
            HAVING COUNT(*) >= {self.config.min_trades_per_ticker}
            ORDER BY trade_count DESC
        """
        return self.con.execute(query).df()

    def compute_ticker_features_sql(self) -> pd.DataFrame:
        """Compute per-ticker aggregated features using pure SQL for speed.

        This is the fast path: computes summary statistics per ticker
        entirely inside DuckDB without loading raw trades into pandas.
        """
        trades_glob = f"{self.config.trades_dir}/*.parquet"
        markets_glob = f"{self.config.markets_dir}/*.parquet"

        query = f"""
            WITH resolved AS (
                SELECT ticker, result
                FROM '{markets_glob}'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
            ),
            trades AS (
                SELECT
                    t.ticker,
                    t.yes_price,
                    t.no_price,
                    t.count,
                    t.taker_side,
                    t.created_time,
                    m.result,
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                         ELSE t.no_price END AS taker_price,
                    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS taker_won,
                    t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                                    ELSE t.no_price END) / 100.0 AS taker_notional
                FROM '{trades_glob}' t
                INNER JOIN resolved m ON t.ticker = m.ticker
            )
            SELECT
                ticker,
                result,

                -- Price features
                AVG(yes_price) AS mean_yes_price,
                STDDEV(yes_price) AS std_yes_price,
                MEDIAN(yes_price) AS median_yes_price,
                MIN(yes_price) AS min_yes_price,
                MAX(yes_price) AS max_yes_price,
                MAX(yes_price) - MIN(yes_price) AS price_range,

                -- Volume features
                COUNT(*) AS trade_count,
                SUM(count) AS total_contracts,
                AVG(count) AS mean_contracts_per_trade,
                MEDIAN(count) AS median_contracts_per_trade,
                SUM(taker_notional) AS total_notional,

                -- VWAP
                SUM(yes_price * count) * 1.0 / NULLIF(SUM(count), 0) AS vwap,

                -- Imbalance features
                SUM(CASE WHEN taker_side = 'yes' THEN count ELSE 0 END) * 1.0
                    / NULLIF(SUM(count), 0) AS yes_volume_share,
                SUM(CASE WHEN taker_side = 'yes' THEN 1 ELSE 0 END) * 1.0
                    / COUNT(*) AS yes_trade_share,
                SUM(CASE WHEN taker_side = 'yes' THEN count ELSE 0 END)
                    - SUM(CASE WHEN taker_side = 'no' THEN count ELSE 0 END)
                    AS net_volume_imbalance,

                -- Time features
                MIN(created_time) AS first_trade_time,
                MAX(created_time) AS last_trade_time,
                EXTRACT(EPOCH FROM (MAX(created_time) - MIN(created_time))) AS duration_seconds,

                -- Outcome
                CASE WHEN result = 'yes' THEN 1 ELSE 0 END AS outcome_yes

            FROM trades
            GROUP BY ticker, result
            HAVING COUNT(*) >= {self.config.min_trades_per_ticker}
            ORDER BY trade_count DESC
        """
        logger.info("Computing per-ticker features via SQL...")
        df = self.con.execute(query).df()
        logger.info("Computed features for %s tickers", f"{len(df):,}")
        return df
