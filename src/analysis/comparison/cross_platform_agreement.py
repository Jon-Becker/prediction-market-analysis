"""Analyze price agreement between Kalshi and Polymarket for matched markets."""

from __future__ import annotations

from pathlib import Path
import json

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType


class CrossPlatformAgreementAnalysis(Analysis):
    """Compare prices between Kalshi and Polymarket for matched markets."""

    def __init__(
        self,
        kalshi_trades_dir: Path | str | None = None,
        kalshi_markets_dir: Path | str | None = None,
        polymarket_trades_dir: Path | str | None = None,
        polymarket_markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="cross_platform_agreement",
            description="Price agreement analysis between Kalshi and Polymarket",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.kalshi_trades_dir = Path(
            kalshi_trades_dir or base_dir / "data" / "kalshi" / "trades"
        )
        self.kalshi_markets_dir = Path(
            kalshi_markets_dir or base_dir / "data" / "kalshi" / "markets"
        )
        self.polymarket_trades_dir = Path(
            polymarket_trades_dir or base_dir / "data" / "polymarket" / "trades"
        )
        self.polymarket_markets_dir = Path(
            polymarket_markets_dir or base_dir / "data" / "polymarket" / "markets"
        )

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Get Kalshi election markets (2024 US election related)
        kalshi_election = con.execute(
            f"""
            SELECT 
                ticker,
                title,
                market_type,
                COALESCE((yes_bid + yes_ask) / 2, last_price, 0.5) as avg_price,
                volume as trade_volume,
                result
            FROM '{self.kalshi_markets_dir}/*.parquet'
            WHERE (title ILIKE '%2024%' OR title ILIKE '%election%' OR title ILIKE '%trump%' OR title ILIKE '%harris%')
              AND status = 'finalized'
              AND result IS NOT NULL
            ORDER BY volume DESC
        """
        ).fetch_df()

        # Get Polymarket election markets (skip if corrupted)
        try:
            polymarket_election = con.execute(
                f"""
                SELECT 
                    title,
                    COALESCE(last_price, 0.5) as avg_price,
                    volume as trade_volume
                FROM '{self.polymarket_markets_dir}/*.parquet'
                WHERE (title ILIKE '%2024%' OR title ILIKE '%election%' OR title ILIKE '%trump%' OR title ILIKE '%harris%')
                  AND volume > 0
                ORDER BY volume DESC
            """
            ).fetch_df()
        except Exception as e:
            print(f"Warning: Could not load Polymarket data: {e}")
            # Return early with Kalshi-only results
            return AnalysisOutput(
                metadata={
                    "summary": "Polymarket data unavailable; Kalshi-only analysis provided.",
                    "kalshi_count": len(kalshi_election),
                    "note": "Cross-platform comparison requires both datasets.",
                }
            )

        # Match markets by title similarity (simple approach: exact/fuzzy match on keywords)
        matched_pairs = []
        for _, kalshi_row in kalshi_election.iterrows():
            for _, pm_row in polymarket_election.iterrows():
                # Simple fuzzy matching on keywords
                kalshi_title = kalshi_row["title"].lower()
                pm_title = pm_row["title"].lower()

                if (
                    ("trump" in kalshi_title and "trump" in pm_title)
                    or ("harris" in kalshi_title and "harris" in pm_title)
                    or ("senate" in kalshi_title and "senate" in pm_title)
                ):
                    matched_pairs.append(
                        {
                            "kalshi_ticker": kalshi_row["ticker"],
                            "kalshi_title": kalshi_row["title"],
                            "kalshi_price": kalshi_row["avg_price"],
                            "kalshi_trades": kalshi_row["trades"],
                            "polymarket_slug": pm_row["slug"],
                            "polymarket_title": pm_row["title"],
                            "polymarket_price": pm_row["avg_price"],
                            "polymarket_trades": pm_row["trades"],
                            "price_divergence": abs(
                                kalshi_row["avg_price"] - pm_row["avg_price"]
                            ),
                        }
                    )

        if not matched_pairs:
            # No matches found; return summary
            return AnalysisOutput(
                metadata={
                    "summary": "Insufficient matched markets between platforms for 2024 election.",
                    "kalshi_count": len(kalshi_election),
                    "polymarket_count": len(polymarket_election),
                }
            )

        matched_df = pd.DataFrame(matched_pairs)

        # Compute statistics
        correlation = (
            pearsonr(matched_df["kalshi_price"], matched_df["polymarket_price"])[0]
            if len(matched_df) > 1
            else np.nan
        )
        mean_divergence = matched_df["price_divergence"].mean()
        median_divergence = matched_df["price_divergence"].median()
        max_divergence = matched_df["price_divergence"].max()

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Cross-Platform Price Agreement: Kalshi vs Polymarket (2024 Election)",
                     fontsize=14, fontweight="bold")

        # Scatter plot
        axes[0, 0].scatter(
            matched_df["kalshi_price"],
            matched_df["polymarket_price"],
            alpha=0.6,
            s=matched_df["kalshi_trades"] / 1000,  # Size by volume
        )
        axes[0, 0].plot([0, 1], [0, 1], "r--", label="Perfect agreement", linewidth=2)
        axes[0, 0].set_xlabel("Kalshi Price")
        axes[0, 0].set_ylabel("Polymarket Price")
        axes[0, 0].set_title(f"Price Correlation: {correlation:.3f}")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Divergence histogram
        axes[0, 1].hist(
            matched_df["price_divergence"], bins=20, alpha=0.7, edgecolor="black"
        )
        axes[0, 1].axvline(mean_divergence, color="r", linestyle="--", label=f"Mean: {mean_divergence:.4f}")
        axes[0, 1].set_xlabel("Price Divergence (|Kalshi - Polymarket|)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Price Divergences")
        axes[0, 1].legend()

        # Summary table
        axes[1, 0].axis("off")
        summary_text = f"""
        CROSS-PLATFORM AGREEMENT SUMMARY
        
        Matched Markets: {len(matched_df)}
        
        Correlation (Pearson): {correlation:.4f}
        Mean Divergence: {mean_divergence:.4f} ({mean_divergence*100:.2f}%)
        Median Divergence: {median_divergence:.4f}
        Max Divergence: {max_divergence:.4f}
        
        Market Quality:
        - Kalshi Avg Trades: {matched_df['kalshi_trades'].mean():.0f}
        - Polymarket Avg Trades: {matched_df['polymarket_trades'].mean():.0f}
        """
        axes[1, 0].text(0.1, 0.5, summary_text, fontfamily="monospace", fontsize=10)

        # Top divergences
        top_divergences = matched_df.nlargest(5, "price_divergence")
        axes[1, 1].axis("off")
        divergence_text = "LARGEST PRICE DIVERGENCES\n\n"
        for idx, row in top_divergences.iterrows():
            divergence_text += f"{row['kalshi_ticker']}\n"
            divergence_text += f"  K: {row['kalshi_price']:.3f} | PM: {row['polymarket_price']:.3f} | Δ: {row['price_divergence']:.4f}\n"
        axes[1, 1].text(0.05, 0.95, divergence_text, fontfamily="monospace", fontsize=9, verticalalignment="top")

        plt.tight_layout()
        output_path = Path(__file__).parent.parent.parent.parent / "output" / "cross_platform_agreement.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save detailed results to JSON
        json_output = {
            "correlation": float(correlation),
            "mean_divergence": float(mean_divergence),
            "median_divergence": float(median_divergence),
            "max_divergence": float(max_divergence),
            "matched_count": len(matched_df),
            "top_divergences": top_divergences.to_dict("records"),
        }

        json_path = Path(__file__).parent.parent.parent.parent / "output" / "cross_platform_agreement.json"
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2, default=str)

        return AnalysisOutput(
            metadata={
                "correlation": float(correlation),
                "mean_divergence": float(mean_divergence),
                "matched_markets": len(matched_df),
                "output_chart": str(output_path),
                "output_json": str(json_path),
            }
        )
