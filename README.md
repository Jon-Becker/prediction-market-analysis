# Prediction Market Analysis

A robust framework for analyzing prediction market data, featuring the largest publicly available dataset of Polymarket and Kalshi market and trade data. This project provides tools for data collection, storage, and extensibile analysis scripts.

## key Features

-   **Data Collection**: Indexers for **Polymarket** (Polygon blockchain) and **Kalshi** (API).
-   **Analysis Framework**: Extensible Python scripts to generate figures and statistics.
-   **Storage**: Efficient Parquet-based storage with automatic progress saving.
-   **Modern CLI**: Interactive menus and command-line arguments via `typer` and `questionary`.
-   **Containerized**: Docker support for consistent execution environments.
-   **Type Safe**: Fully typed codebase with `mypy` and `ruff` linting.

---

## Quick Start

### Option 1: Docker (Recommended)

Run the analysis environment without installing Python dependencies locally.

1.  **Build the image**:
    ```bash
    docker build -t prediction-market-analysis .
    ```

2.  **Run the container**:
    ```bash
    docker run -it -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output prediction-market-analysis
    ```

### Option 2: Local Setup (uv)

We use [uv](https://github.com/astral-sh/uv) for fast dependency management.

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

2.  **Download dataset** (Optional, 36GiB compressed):
    ```bash
    make setup
    ```

### Option 3: Local Setup (pip)

Standard installation using `pip`.

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The project uses a unified CLI entry point `main.py`.

### 1. Run Data Collection (`index`)

Collect market and trade data from APIs and blockchain. This command opens an interactive menu to select specific indexers.

```bash
uv run main.py index
# OR
python main.py index
```

Data is saved to `data/kalshi/` and `data/polymarket/`. Progress is saved automatically.

### 2. Run Analysis (`analyze`)

Generate figures and statistics.

**Interactive Mode:**
Select an analysis from the list.
```bash
uv run main.py analyze
```

**Direct Mode:**
Run a specific analysis by name.
```bash
uv run main.py analyze --name win_rate_by_price
```

Output files (PNG, PDF, CSV, JSON) are saved to `output/`.

### 3. Package Data (`package`)

Compress the data directory for storage or distribution.

```bash
uv run main.py package
```
Creates `data.tar.zst`.

---

## Project Structure

```
├── src/
│   ├── analysis/           # Analysis scripts (add your own here)
│   ├── indexers/           # Data collectors for Kalshi/Polymarket
│   ├── common/             # Shared utilities
│   └── main.py             # CLI entry point
├── data/                   # Data directory (parquet files)
├── output/                 # Generated results
├── tests/                  # Pytest suite
├── Dockerfile              # Container definition
├── pyproject.toml          # Project configuration (dependencies, tools)
└── requirements.txt        # Frozen dependencies
```

## Development

### Code Quality
This project uses `ruff` for linting and formatting. Setup pre-commit hooks to ensure quality:

```bash
uv run pre-commit install
```

### Testing
Run the test suite with `pytest`:

```bash
uv run pytest
```

## Documentation

-   [Data Schemas](docs/SCHEMAS.md) - Parquet file schemas.
-   [Writing Analyses](docs/ANALYSIS.md) - Guide for custom scripts.

## Contributing

Contributions are welcome! Please open a pull request with detailed information on changes. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Research & Citations

-   Becker, J. (2026). _The Microstructure of Wealth Transfer in Prediction Markets_. Jbecker. https://jbecker.dev/research/prediction-market-microstructure

If you use this dataset, please cite the above paper. Reach out via [email](mailto:jonathan@jbecker.dev) or [Twitter](https://x.com/BeckerrJon) if you have questions!
