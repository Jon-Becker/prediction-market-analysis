# Prediction Market Analysis

This repository contains the data, analysis code, and research paper for studying the longshot bias in prediction markets using trade data from Kalshi, a CFTC-regulated prediction market platform.

## Research Overview

**Paper Title:** *The Longshot Bias in Prediction Markets: Evidence from 68 Million Trades*

**Author:** Jonathan Becker ([jonathan@jbecker.dev](mailto:jonathan@jbecker.dev))

This study analyzes 67.8 million trades worth $8.6 billion from Kalshi to investigate market calibration and the longshot bias. Key findings include:

- Contracts priced below 10 cents resolve favorably 30–40% less often than their price-implied probability, confirming a systematic longshot bias
- The bias diminishes monotonically with price and reverses slightly for near-certain outcomes above 95 cents
- Large trades (above $5,000) achieve positive excess returns, while small trades consistently underperform
- Contrarian traders who buy after price declines significantly outperform momentum traders
- Market efficiency improves as resolution approaches, with excess win rate improving from -1.8% at 3–7 days to -0.4% within the final hour

## Dataset

The dataset includes:
- **67.8 million trades** from Kalshi
- **$8.6 billion** in total trading volume
- Data stored in Parquet format for efficient analysis

The data is split into chunks (`data.zip.aa`, `data.zip.ab`, etc.) due to file size constraints.

## Installation

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Jon-Becker/prediction-market-analysis.git
   cd prediction-market-analysis
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Set up the data directory (reassembles and extracts the data):
   ```bash
   uv run main.py setup
   # or
   make setup
   ```

## Usage

### Commands

| Command | Description |
|---------|-------------|
| `uv run main.py setup` | Reassemble data.zip from chunks and extract to `data/` |
| `uv run main.py teardown` | Clean up the data directory and data.zip |
| `uv run main.py analysis` | Run all analysis scripts |
| `uv run main.py backfill` | Backfill market data from Kalshi API |
| `uv run main.py backfill-trades` | Backfill trade data from Kalshi API |

### Makefile Shortcuts

```bash
make setup           # Set up data directory
make teardown        # Clean up data directory
make analysis        # Run all analysis scripts
make backfill        # Backfill market data
make backfill-trades # Backfill trade data
```

### Fetching Sample Markets

Run without arguments to see sample open markets:
```bash
uv run main.py
```

## Project Structure

```
├── main.py                 # CLI entry point
├── src/
│   ├── __init__.py
│   ├── kalshi.py          # Kalshi API client
│   ├── database.py        # Parquet data storage
│   ├── analysis.py        # Analysis utilities
│   ├── backfill.py        # Market data backfill
│   └── backfill_trades.py # Trade data backfill
├── research/
│   ├── main.tex           # Research paper (LaTeX)
│   ├── main.pdf           # Compiled paper
│   ├── sections/          # Paper sections
│   ├── analysis/          # Analysis scripts (Python)
│   └── fig/               # Generated figures and data
├── data.zip.*             # Compressed dataset chunks
├── pyproject.toml         # Project configuration
├── Makefile               # Command shortcuts
└── uv.lock                # Dependency lock file
```

## Analysis Scripts

The `research/analysis/` directory contains scripts for various analyses:

- **Calibration**: `win_rate_by_price.py`, `upset_frequency.py`
- **Trading Behavior**: `trade_size_vs_win_rate.py`, `contrarian_vs_momentum.py`
- **Market Dynamics**: `bid_ask_spread_dynamics.py`, `price_convergence_to_resolution.py`
- **Volume Analysis**: `total_volume_by_price.py`, `volume_acceleration.py`
- **Temporal Patterns**: `intraday_weekday_patterns.py`, `early_vs_late_trader_returns.py`

Each script generates CSV data and PDF/PNG visualizations in `research/fig/`.

## Dependencies

Key dependencies (managed via `uv`):

- `kalshi-python` - Kalshi API client
- `pandas` - Data manipulation
- `duckdb` - Fast analytical queries
- `pyarrow` - Parquet file support
- `matplotlib` - Visualization
- `scipy` - Statistical analysis
- `httpx` - HTTP client

## License

See repository for license information.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{becker2024longshot,
  title={The Longshot Bias in Prediction Markets: Evidence from 68 Million Trades},
  author={Becker, Jonathan},
  year={2024}
}
```
