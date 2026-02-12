"""Analyze distribution of market types by volume with hierarchical groupings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import squarify
from matplotlib.patches import Patch

from src.analysis.kalshi.util.categories import GROUP_COLORS, get_group, get_hierarchy
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class MarketTypesAnalysis(Analysis):
    """Analyze market type distribution by volume on Kalshi."""

    def __init__(
        self,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="market_types",
            description="Market type distribution by notional volume",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        # Get raw categories with market count
        df_raw = con.execute(
            f"""
            WITH categorized AS (
                SELECT
                    CASE
                        WHEN event_ticker IS NULL OR event_ticker = '' THEN 'independent'
                        WHEN regexp_extract(event_ticker, '^([A-Z0-9]+)', 1) = '' THEN 'independent'
                        ELSE regexp_extract(event_ticker, '^([A-Z0-9]+)', 1)
                    END AS category,
                    COALESCE(volume, 0) AS volume
                FROM '{self.markets_dir}/*.parquet'
            )
            SELECT
                category,
                SUM(volume) AS total_volume,
                COUNT(*) AS market_count
            FROM categorized
            GROUP BY category
            ORDER BY total_volume DESC
            """
        ).df()

        # Apply grouping
        df_raw["group"] = df_raw["category"].apply(get_group)
        df_grouped = (
            df_raw.groupby("group")
            .agg(
                total_volume=("total_volume", "sum"),
                market_count=("market_count", "sum"),
            )
            .reset_index()
        )
        df_grouped = df_grouped.sort_values("total_volume", ascending=False)

        fig = self._create_figure(df_raw, df_grouped)
        chart = self._create_chart(df_raw)

        return AnalysisOutput(figure=fig, data=df_grouped, chart=chart)

    def _build_hierarchy_json(self, df_raw: pd.DataFrame, min_pct: float = 0.01) -> list[dict[str, Any]]:
        """Build hierarchical JSON structure for treemap.

        Args:
            df_raw: DataFrame with category and total_volume columns
            min_pct: Minimum percentage of parent volume to include (default 1%)
                     Items below this threshold are excluded entirely

        Returns:
            List of dicts with name, value, and optional children
        """
        # Apply hierarchy mapping
        hierarchies = df_raw["category"].apply(get_hierarchy)
        df_raw = df_raw.copy()
        df_raw["group"] = hierarchies.apply(lambda x: x[0])
        df_raw["mid_category"] = hierarchies.apply(lambda x: x[1])
        df_raw["subcategory"] = hierarchies.apply(lambda x: x[2])

        # Build tree structure: group -> mid_category -> subcategory
        result = []

        # Get groups sorted by volume
        group_totals = df_raw.groupby("group")["total_volume"].sum().sort_values(ascending=False)

        for group_name, group_vol in group_totals.items():
            df_group = df_raw[df_raw["group"] == group_name]

            # Get mid-categories for this group
            mid_totals = df_group.groupby("mid_category")["total_volume"].sum().sort_values(ascending=False)

            children = []
            for mid_name, mid_vol in mid_totals.items():
                # Skip if below threshold
                if mid_vol / group_vol < min_pct:
                    continue

                df_mid = df_group[df_group["mid_category"] == mid_name]

                # Get subcategories for this mid-category
                sub_totals = df_mid.groupby("subcategory")["total_volume"].sum().sort_values(ascending=False)

                sub_children = []
                for sub_name, sub_vol in sub_totals.items():
                    # Skip if below threshold
                    if sub_vol / mid_vol < min_pct:
                        continue
                    sub_children.append({"name": sub_name, "value": int(sub_vol)})

                if sub_children:
                    # Only add children if there's more than one, or if the single child
                    # is meaningfully different from the parent
                    if len(sub_children) > 1 or (len(sub_children) == 1 and sub_children[0]["name"] != mid_name):
                        children.append(
                            {
                                "name": mid_name,
                                "value": int(mid_vol),
                                "children": sub_children,
                            }
                        )
                    else:
                        children.append({"name": mid_name, "value": int(mid_vol)})
                else:
                    children.append({"name": mid_name, "value": int(mid_vol)})

            if children:
                result.append(
                    {
                        "name": group_name,
                        "value": int(group_vol),
                        "children": children,
                    }
                )
            else:
                result.append({"name": group_name, "value": int(group_vol)})

        return result

    def _create_figure(self, df_raw: pd.DataFrame, df_grouped: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib treemap figure."""
        top_n_per_group = 8

        # Build treemap data
        treemap_data = []
        for group_name in df_grouped["group"].tolist():
            df_group = df_raw[df_raw["group"] == group_name].copy()
            df_group = df_group.sort_values("total_volume", ascending=False)

            # Take top N categories, group rest as "Other"
            if len(df_group) > top_n_per_group:
                top_cats = df_group.head(top_n_per_group)
                other_vol = df_group.iloc[top_n_per_group:]["total_volume"].sum()
                for _, row in top_cats.iterrows():
                    treemap_data.append(
                        {
                            "group": group_name,
                            "category": row["category"],
                            "volume": row["total_volume"],
                        }
                    )
                if other_vol > 0:
                    treemap_data.append(
                        {
                            "group": group_name,
                            "category": f"{group_name} Other",
                            "volume": other_vol,
                        }
                    )
            else:
                for _, row in df_group.iterrows():
                    treemap_data.append(
                        {
                            "group": group_name,
                            "category": row["category"],
                            "volume": row["total_volume"],
                        }
                    )

        df_treemap = pd.DataFrame(treemap_data)
        df_treemap = df_treemap.sort_values("volume", ascending=False)

        # Create colors for each category based on group
        def get_shade(base_color: str, idx: int, total: int) -> tuple:
            """Lighten color based on index."""
            rgb = mcolors.to_rgb(base_color)
            factor = 0.3 + 0.7 * (1 - idx / max(total, 1))
            return tuple(min(1, c * factor + (1 - factor) * 0.9) for c in rgb)

        # Compute shade index per group
        group_counts = df_treemap.groupby("group").cumcount()
        group_totals = df_treemap.groupby("group")["group"].transform("count")

        colors = []
        for _, row in df_treemap.iterrows():
            base = GROUP_COLORS.get(row["group"], "#888888")
            idx = group_counts[row.name]
            total = group_totals[row.name]
            colors.append(get_shade(base, idx, total))

        fig, ax = plt.subplots(figsize=(16, 10))

        sizes = df_treemap["volume"].tolist()

        # Compute rectangles to determine label visibility
        norm_x, norm_y = 100, 100
        rects = squarify.normalize_sizes(sizes, norm_x, norm_y)
        rects = squarify.squarify(rects, 0, 0, norm_x, norm_y)

        # Only show labels for rectangles large enough to fit text
        labels = []
        for rect, (_, row) in zip(rects, df_treemap.iterrows()):
            area = rect["dx"] * rect["dy"]
            min_dim = min(rect["dx"], rect["dy"])
            if area > 50 and min_dim > 5:
                if row["volume"] > 0.5e9:
                    labels.append(f"{row['category']}\n{row['volume'] / 1e9:.1f}B")
                else:
                    labels.append(row["category"])
            else:
                labels.append("")

        squarify.plot(
            sizes=sizes,
            label=labels,
            color=colors,
            alpha=0.9,
            ax=ax,
            text_kwargs={"fontsize": 7},
            pad=False,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.axis("off")
        ax.set_title("Market Types by Volume", fontsize=14, fontweight="bold")

        # Add legend for groups
        legend_elements = [
            Patch(facecolor=GROUP_COLORS[g], label=g) for g in df_grouped["group"].tolist() if g in GROUP_COLORS
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=9,
        )

        plt.tight_layout()
        return fig

    def _create_chart(self, df_raw: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        hierarchy_data = self._build_hierarchy_json(df_raw, min_pct=0.01)

        return ChartConfig(
            type=ChartType.TREEMAP,
            data=hierarchy_data,
            nameKey="name",
            valueKey="value",
            childrenKey="children",
            title="Notional Volume by Market Category",
            caption="Total Notional Volume on Kalshi by Market Category and Subcategory. Click to zoom in.",
            yUnit=UnitType.DOLLARS,
        )
