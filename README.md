# Prediction Market Analysis

A framework for analyzing Kalshi prediction market data. Includes tools for data collection, storage, and running analysis scripts that generate figures and statistics.

## Setup

Requires Python 3.9+. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Running Analyses

The data is stored as compressed chunks (`data.zip.*`). The analysis framework handles extraction and cleanup automatically.

### Run all analyses

```bash
make analysis
```

This will:
1. Reassemble and extract the data archive
2. Run all scripts in `research/analysis/` in parallel
3. Clean up the extracted data when complete

### Run a single analysis

```bash
make analyze <script_name>
```

For example:

```bash
make analyze mispricing_by_price
make analyze total_volume_by_price.py  # .py extension is optional
```

### Manual commands

You can also run the CLI directly:

```bash
uv run main.py setup      # Extract data
uv run main.py analysis   # Run all analyses
uv run main.py analysis mispricing_by_price  # Run single analysis
uv run main.py teardown   # Clean up data
```

## Data Schemas

Data is stored as Parquet files. When extracted, the directory structure is:

```
data/
  markets/
    markets_0_10000.parquet
    markets_10000_20000.parquet
    ...
  trades/
    <TICKER>_trades.parquet
    ...
```

### Markets Schema

Each row represents a prediction market contract.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Unique market identifier (e.g., `PRES-2024-DJT`) |
| `event_ticker` | string | Parent event identifier, used for categorization |
| `market_type` | string | Market type (typically `binary`) |
| `title` | string | Human-readable market title |
| `yes_sub_title` | string | Label for the "Yes" outcome |
| `no_sub_title` | string | Label for the "No" outcome |
| `status` | string | Market status: `open`, `closed`, `finalized` |
| `yes_bid` | int (nullable) | Best bid price for Yes contracts (cents, 1-99) |
| `yes_ask` | int (nullable) | Best ask price for Yes contracts (cents, 1-99) |
| `no_bid` | int (nullable) | Best bid price for No contracts (cents, 1-99) |
| `no_ask` | int (nullable) | Best ask price for No contracts (cents, 1-99) |
| `last_price` | int (nullable) | Last traded price (cents, 1-99) |
| `volume` | int | Total contracts traded |
| `volume_24h` | int | Contracts traded in last 24 hours |
| `open_interest` | int | Outstanding contracts |
| `result` | string | Market outcome: `yes`, `no`, or empty if unresolved |
| `created_time` | datetime | When the market was created |
| `open_time` | datetime (nullable) | When trading opened |
| `close_time` | datetime (nullable) | When trading closed |
| `_fetched_at` | datetime | When this record was fetched |

### Trades Schema

Each row represents a single trade execution.

| Column | Type | Description |
|--------|------|-------------|
| `trade_id` | string | Unique trade identifier |
| `ticker` | string | Market ticker this trade belongs to |
| `count` | int | Number of contracts traded |
| `yes_price` | int | Yes contract price (cents, 1-99) |
| `no_price` | int | No contract price (cents, 1-99), always `100 - yes_price` |
| `taker_side` | string | Which side the taker bought: `yes` or `no` |
| `created_time` | datetime | When the trade occurred |
| `_fetched_at` | datetime | When this record was fetched |

**Note on prices:** Prices are in cents. A `yes_price` of 65 means the contract costs $0.65 and pays $1.00 if the outcome is "Yes" (implied probability: 65%). The `no_price` is always `100 - yes_price`.

## Writing Analysis Scripts

Analysis scripts live in `research/analysis/` and output to `research/fig/`.

### Basic template

```python
#!/usr/bin/env python3
"""Brief description of what this analysis does."""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt


def main():
    # Standard path setup
    base_dir = Path(__file__).parent.parent.parent
    trades_dir = base_dir / "data" / "trades"
    markets_dir = base_dir / "data" / "markets"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Connect to DuckDB (in-memory)
    con = duckdb.connect()

    # Query parquet files directly with glob patterns
    df = con.execute(
        f"""
        SELECT
            yes_price,
            count,
            taker_side
        FROM '{trades_dir}/*.parquet'
        WHERE yes_price BETWEEN 1 AND 99
        LIMIT 1000
        """
    ).df()

    # Save data output
    df.to_csv(fig_dir / "my_analysis.csv", index=False)

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["yes_price"], df["count"])
    ax.set_xlabel("Price (cents)")
    ax.set_ylabel("Count")
    ax.set_title("My Analysis")

    plt.tight_layout()
    fig.savefig(fig_dir / "my_analysis.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "my_analysis.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
```

### Common query patterns

**Join trades with market outcomes:**

```sql
WITH resolved_markets AS (
    SELECT ticker, result
    FROM '{markets_dir}/*.parquet'
    WHERE status = 'finalized'
      AND result IN ('yes', 'no')
)
SELECT
    t.yes_price,
    t.count,
    t.taker_side,
    m.result,
    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS taker_won
FROM '{trades_dir}/*.parquet' t
INNER JOIN resolved_markets m ON t.ticker = m.ticker
```

**Analyze both taker and maker positions:**

```sql
WITH all_positions AS (
    -- Taker positions
    SELECT
        CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END AS price,
        count,
        'taker' AS role
    FROM '{trades_dir}/*.parquet'

    UNION ALL

    -- Maker positions (counterparty)
    SELECT
        CASE WHEN taker_side = 'yes' THEN no_price ELSE yes_price END AS price,
        count,
        'maker' AS role
    FROM '{trades_dir}/*.parquet'
)
SELECT price, role, SUM(count) AS total_contracts
FROM all_positions
GROUP BY price, role
ORDER BY price
```

**Extract category from event_ticker:**

```sql
SELECT
    CASE
        WHEN event_ticker IS NULL OR event_ticker = '' THEN 'independent'
        ELSE regexp_extract(event_ticker, '^([A-Z0-9]+)', 1)
    END AS category,
    COUNT(*) AS market_count
FROM '{markets_dir}/*.parquet'
GROUP BY category
```

### Using the categories utility

For grouping markets into high-level categories (Sports, Politics, Crypto, etc.):

```python
from research.analysis.util.categories import get_group, get_hierarchy, GROUP_COLORS

# Get high-level group
group = get_group("NFLGAME")  # Returns "Sports"

# Get full hierarchy (group, category, subcategory)
hierarchy = get_hierarchy("NFLGAME")  # Returns ("Sports", "NFL", "Games")

# Use predefined colors for consistent visualizations
color = GROUP_COLORS["Sports"]  # Returns "#1f77b4"
```

### Output conventions

- Save CSV/JSON for raw data: `fig_dir / "analysis_name.csv"`
- Save PNG at 300 DPI for presentations: `fig_dir / "analysis_name.png"`
- Save PDF for papers: `fig_dir / "analysis_name.pdf"`
- Print a completion message: `print(f"Outputs saved to {fig_dir}")`

### Dependencies available

Scripts have access to these libraries (see `pyproject.toml`):

- `duckdb` - SQL queries on Parquet files
- `pandas` - DataFrames
- `matplotlib` - Plotting
- `scipy` - Statistical functions
- `brokenaxes` - Plots with broken axes
- `squarify` - Treemap visualizations
