# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python framework for analyzing prediction market data from Polymarket and Kalshi. Provides data collection indexers, Parquet-based storage, and an extensible analysis framework that generates figures and statistics.

## Commands

```bash
# Install dependencies (requires uv)
uv sync

# Run linter and format checker
make lint

# Auto-fix lint issues and format
make format

# Run all tests
make test

# Run a single test file
uv run pytest tests/test_compile.py -v

# Run a single test by name
uv run pytest tests/test_analysis_run.py -v -k "test_analysis_run[WinRateByPriceAnalysis]"

# Skip slow tests (animated analyses)
uv run pytest tests/ -v -m "not slow"

# Run interactive analysis menu
make analyze

# Run a specific analysis by name
uv run main.py analyze win_rate_by_price

# Run interactive indexer menu
make index

# Download the pre-collected dataset (36GiB)
make setup
```

## Architecture

### Plugin Discovery System

Both `Analysis` and `Indexer` use a dynamic class discovery pattern. The `Analysis.load()` and `Indexer.load()` class methods scan their respective directories (`src/analysis/`, `src/indexers/`) for Python files, import them, and find all concrete subclasses. Any `.py` file (not starting with `_`) containing a subclass is automatically discovered — no registration required.

### Analysis Framework

Analysis scripts subclass `Analysis` (from `src/common/analysis.py`) and implement `run() -> AnalysisOutput`. The `AnalysisOutput` dataclass bundles three optional outputs:
- `figure`: matplotlib `Figure` or `FuncAnimation` for visualizations
- `data`: pandas `DataFrame` for tabular output
- `chart`: `ChartConfig` (from `src/common/interfaces/chart.py`) for web-renderable JSON chart configs

The `save()` method handles export to PNG/PDF/SVG/GIF/CSV/JSON based on which outputs are present. Analyses accept data directory paths as constructor parameters with defaults pointing to `data/`, allowing tests to inject fixture directories.

### Indexer Framework

Indexers subclass `Indexer` (from `src/common/indexer.py`) and implement `run()`. They use `ParquetStorage` for chunked writes and `httpx` with retry logic (`src/common/client.py`) for API calls. Progress is saved automatically via chunked Parquet files, so collection can be interrupted and resumed.

### Data Layer

All data is stored as Parquet files under `data/`. DuckDB is used for SQL queries directly against Parquet files (glob patterns like `'{dir}/*.parquet'`). See `docs/SCHEMAS.md` for column definitions. Key patterns:
- Kalshi prices are in cents (1-99), `no_price = 100 - yes_price`
- Polymarket CTF prices are decimals (0-1)
- Trades have both taker and maker sides (counterparty is inferred)

### Test Structure

Tests use session-scoped pytest fixtures (`tests/conftest.py`) that create temporary Parquet files with synthetic data. Analysis tests inject fixture directories through constructor params. The `@pytest.mark.slow` marker is used for animated analysis tests.

## Configuration

- Python 3.9+ target (set in `.python-version` and `pyproject.toml`)
- Ruff for linting and formatting (line length 120, rules: E, W, F, I, B, C4, UP)
- CI runs lint + tests on every push and PR (`.github/workflows/ci.yml`)
- Environment variables in `.env`: `POLYGON_RPC` (for Polymarket blockchain indexing), `POLYMARKET_START_BLOCK`
