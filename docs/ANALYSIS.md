# Writing Analysis Scripts

Analysis scripts live in `src/analysis/{kalshi,polymarket}/` and extend the `Analysis` base class.

## Running Analyses

```bash
make analyze
```

This opens an interactive menu to select which analysis to run. You can run all analyses or select a specific one. Output files (PNG, PDF, CSV, JSON) are saved to `output/`.

## Basic Template

```python
"""Brief description of what this analysis does."""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.common.analysis import Analysis


class MyAnalysis(Analysis):
    name = "my_analysis"
    description = "Brief description of what this analysis does"

    def run(self) -> tuple[Figure, dict]:
        base_dir = Path(__file__).parent.parent.parent.parent
        kalshi_trades = base_dir / "data" / "kalshi" / "trades"
        kalshi_markets = base_dir / "data" / "kalshi" / "markets"

        con = duckdb.connect()
        df = con.execute(
            f"""
            SELECT yes_price, count, taker_side
            FROM '{kalshi_trades}/*.parquet'
            WHERE yes_price BETWEEN 1 AND 99
            LIMIT 1000
            """
        ).df()

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df["yes_price"], df["count"])
        ax.set_xlabel("Price (cents)")
        ax.set_ylabel("Count")
        ax.set_title("My Analysis")
        plt.tight_layout()

        # Return figure and data dict (for CSV/JSON export)
        return fig, df.to_dict(orient="records")
```

## Common Query Patterns

### Join trades with market outcomes (Kalshi)

```sql
WITH resolved_markets AS (
    SELECT ticker, result
    FROM '{kalshi_markets}/*.parquet'
    WHERE status = 'finalized'
      AND result IN ('yes', 'no')
)
SELECT
    t.yes_price,
    t.count,
    t.taker_side,
    m.result,
    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS taker_won
FROM '{kalshi_trades}/*.parquet' t
INNER JOIN resolved_markets m ON t.ticker = m.ticker
```

### Analyze both taker and maker positions

```sql
WITH all_positions AS (
    -- Taker positions
    SELECT
        CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END AS price,
        count,
        'taker' AS role
    FROM '{kalshi_trades}/*.parquet'

    UNION ALL

    -- Maker positions (counterparty)
    SELECT
        CASE WHEN taker_side = 'yes' THEN no_price ELSE yes_price END AS price,
        count,
        'maker' AS role
    FROM '{kalshi_trades}/*.parquet'
)
SELECT price, role, SUM(count) AS total_contracts
FROM all_positions
GROUP BY price, role
ORDER BY price
```

### Extract category from event_ticker

```sql
SELECT
    CASE
        WHEN event_ticker IS NULL OR event_ticker = '' THEN 'independent'
        ELSE regexp_extract(event_ticker, '^([A-Z0-9]+)', 1)
    END AS category,
    COUNT(*) AS market_count
FROM '{kalshi_markets}/*.parquet'
GROUP BY category
```

## Using the Categories Utility

For grouping Kalshi markets into high-level categories (Sports, Politics, Crypto, etc.):

```python
from src.analysis.kalshi.util.categories import get_group, get_hierarchy, GROUP_COLORS

# Get high-level group
group = get_group("NFLGAME")  # Returns "Sports"

# Get full hierarchy (group, category, subcategory)
hierarchy = get_hierarchy("NFLGAME")  # Returns ("Sports", "NFL", "Games")

# Use predefined colors for consistent visualizations
color = GROUP_COLORS["Sports"]  # Returns "#1f77b4"
```

## Progress Indicator

For long-running operations, use the `progress()` context manager to show a spinner:

```python
def run(self) -> AnalysisOutput:
    with self.progress("Loading trades data"):
        df = con.execute("SELECT * FROM large_table").df()

    with self.progress("Computing aggregations"):
        # expensive computation
        result = df.groupby(...).agg(...)
```

## Output Conventions

The `Analysis.save()` method handles output automatically:
- PNG at 300 DPI for presentations
- PDF for papers
- CSV/JSON for raw data

All outputs are saved to `output/` with the analysis name as the filename.

## Dependencies

Scripts have access to these libraries (see `pyproject.toml`):

- `duckdb` - SQL queries on Parquet files
- `pandas` - DataFrames
- `matplotlib` - Plotting
- `scipy` - Statistical functions
- `brokenaxes` - Plots with broken axes
- `squarify` - Treemap visualizations
