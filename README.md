# Prediction Market Analysis

> This dataset was collected for and supports the analysis in [The Microstructure of Wealth Transfer in Prediction Markets](https://jbecker.dev/research/prediction-market-microstructure).

A framework for analyzing prediction market data from Kalshi and Polymarket. Includes tools for data collection, storage, and running analysis scripts that generate figures and statistics.

## Setup

Requires Python 3.9+. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

### Downloading the Dataset

To download and extract the pre-collected dataset:

```bash
make setup
```

This downloads `data.tar.zst` from Google Cloud Storage and extracts it to `data/`.

## Data Collection

Collect market and trade data from prediction market APIs:

```bash
make index
```

This opens an interactive menu to select which indexer to run. Data is saved to `data/kalshi/` and `data/polymarket/` directories. Progress is saved automatically, so you can interrupt and resume collection.

### Polymarket Trade Sources

- **Trades (API)**: Fetches from Polymarket's public data API. Fast but only provides recent trades (limited historical depth).
- **Trades (Blockchain)**: Fetches `OrderFilled` events directly from the Polygon blockchain. Complete historical data from block 15,000,000 onwards, but slower due to RPC rate limits.

### Packaging Data

To compress the data directory for storage/distribution:

```bash
make package
```

This creates a zstd-compressed tar archive (`data.tar.zst`) and removes the `data/` directory.

## Running Analyses

```bash
make analyze
```

This opens an interactive menu to select which analysis to run. You can run all analyses or select a specific one. Output files (PNG, PDF, CSV, JSON) are saved to `output/`.

See [docs/ANALYSIS.md](docs/ANALYSIS.md) for writing custom analysis scripts.

## Project Structure

```
├── src/
│   ├── analysis/           # Analysis scripts
│   │   ├── kalshi/         # Kalshi-specific analyses
│   │   └── polymarket/     # Polymarket-specific analyses
│   ├── indexers/           # Data collection indexers
│   │   ├── kalshi/         # Kalshi API client and indexers
│   │   └── polymarket/     # Polymarket API/blockchain indexers
│   └── common/             # Shared utilities and interfaces
├── data/                   # Data directory (extracted from data.tar.zst)
│   ├── kalshi/
│   │   ├── markets/
│   │   └── trades/
│   └── polymarket/
│       ├── blocks/
│       ├── markets/
│       └── trades/
├── docs/                   # Documentation
└── output/                 # Analysis outputs (figures, CSVs)
```

## Documentation

- [Contributing](CONTRIBUTING.md) - Guidelines for contributors
- [Data Schemas](docs/SCHEMAS.md) - Parquet file schemas for markets and trades
- [Writing Analyses](docs/ANALYSIS.md) - Guide for writing custom analysis scripts
