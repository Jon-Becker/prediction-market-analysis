# Prediction Market Analysis

> This dataset was collected for and supports the analysis in [The Microstructure of Wealth Transfer in Prediction Markets](https://jbecker.dev/research/prediction-market-microstructure).

A framework for analyzing prediction market data from Kalshi and Polymarket. Includes tools for data collection, storage, and running analysis scripts that generate figures and statistics.

## Setup

Requires Python 3.9+. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Data Collection

Collect market and trade data from prediction market APIs:

```bash
make collect
```

This opens an interactive menu:

```
Data Collection
========================================
  1. Kalshi - Markets
  2. Kalshi - Trades
  3. Polymarket - Markets
  4. Polymarket - Trades (API)
  5. Polymarket - Trades (Blockchain)
  6. Exit
```

Data is saved to `data/kalshi/` and `data/polymarket/` directories. Progress is saved automatically, so you can interrupt and resume collection.

### Polymarket Trade Sources

- **Trades (API)**: Fetches from Polymarket's public data API. Fast but only provides recent trades (limited historical depth).
- **Trades (Blockchain)**: Fetches `OrderFilled` events directly from the Polygon blockchain. Complete historical data from block 15,000,000 onwards, but slower due to RPC rate limits.

### Packaging Data

To compress the data directory into chunks for storage/distribution:

```bash
make package
```

This will:
1. Compress `data/` into `data.zip`
2. Split into 1GB chunks (`data.zip.000`, `data.zip.001`, etc.) if needed
3. Delete the `data/` directory

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
uv run main.py package    # Compress and split data
```

## Data Schemas

Data is stored as Parquet files. When extracted, the directory structure is:

```
data/
├── kalshi/
│   ├── markets/
│   │   ├── markets_0_10000.parquet
│   │   └── ...
│   └── trades/
│       ├── trades_0_10000.parquet
│       └── ...
└── polymarket/
    ├── markets/
    │   ├── markets_0_10000.parquet
    │   └── ...
    └── trades/
        ├── trades_0_10000.parquet
        └── ...
```

### Kalshi Markets Schema

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

### Kalshi Trades Schema

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

**Note on Kalshi prices:** Prices are in cents. A `yes_price` of 65 means the contract costs $0.65 and pays $1.00 if the outcome is "Yes" (implied probability: 65%). The `no_price` is always `100 - yes_price`.

### Polymarket Markets Schema

Each row represents a prediction market.

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Market ID |
| `condition_id` | string | Condition ID (hex hash) |
| `question` | string | Market question |
| `slug` | string | URL slug |
| `outcomes` | string | JSON string of outcome names |
| `outcome_prices` | string | JSON string of outcome prices |
| `volume` | float | Total volume in USD |
| `liquidity` | float | Current liquidity in USD |
| `active` | bool | Is market active |
| `closed` | bool | Is market closed |
| `end_date` | datetime (nullable) | When market ends |
| `created_at` | datetime (nullable) | When market was created |
| `_fetched_at` | datetime | When this record was fetched |

### Polymarket Trades Schema (API)

Each row represents a single trade from the public data API.

| Column | Type | Description |
|--------|------|-------------|
| `condition_id` | string | Market condition ID |
| `asset` | string | Asset/token ID |
| `side` | string | Trade side: `BUY` or `SELL` |
| `size` | float | Number of shares traded |
| `price` | float | Price (0-1 decimal) |
| `timestamp` | int | Unix timestamp |
| `outcome` | string | Outcome name |
| `outcome_index` | int | Outcome index (0 or 1) |
| `transaction_hash` | string | Blockchain transaction hash |
| `_fetched_at` | datetime | When this record was fetched |

### Polymarket Trades Schema (Blockchain)

Each row represents an `OrderFilled` event from the Polygon blockchain.

| Column | Type | Description |
|--------|------|-------------|
| `block_number` | int | Polygon block number |
| `transaction_hash` | string | Blockchain transaction hash |
| `log_index` | int | Log index within transaction |
| `order_hash` | string | Unique order identifier |
| `maker` | string | Address of limit order placer |
| `taker` | string | Address that filled the order |
| `maker_asset_id` | int | Asset ID maker provided (0=USDC) |
| `taker_asset_id` | int | Asset ID taker provided |
| `maker_amount` | int | Amount maker gave (6 decimals) |
| `taker_amount` | int | Amount taker gave (6 decimals) |
| `fee` | int | Trading fee (6 decimals) |
| `_fetched_at` | datetime | When this record was fetched |
| `_contract` | string | Contract name (CTF Exchange or NegRisk) |

**Note on Polymarket prices:** Prices are decimals between 0 and 1. A price of 0.65 means the contract costs $0.65 and pays $1.00 if the outcome wins (implied probability: 65%).

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
    kalshi_trades = base_dir / "data" / "kalshi" / "trades"
    kalshi_markets = base_dir / "data" / "kalshi" / "markets"
    polymarket_trades = base_dir / "data" / "polymarket" / "trades"
    polymarket_markets = base_dir / "data" / "polymarket" / "markets"
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
        FROM '{kalshi_trades}/*.parquet'
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

**Join trades with market outcomes (Kalshi):**

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

**Analyze both taker and maker positions:**

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

**Extract category from event_ticker:**

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
