import sys

from src import KalshiClient
from src.analysis import (
    cleanup_data_directory,
    generate_data_directory,
    reassemble_data_zip,
    run_analysis,
)
from src.backfill import backfill
from src.backfill_trades import backfill_trades


def main():
    client = KalshiClient()

    if len(sys.argv) < 2:
        print("No ticker provided. Fetching sample markets...\n")
        markets = client.list_markets()
        print("Sample open markets:")
        for market in markets[:5]:
            print(f"  - {market.ticker}: {market.title}")
        print("\nUsage: uv run main.py <market_ticker>")
        sys.exit(0)

    command = sys.argv[1]

    if command == "backfill":
        backfill()
        sys.exit(0)

    if command == "backfill-trades":
        backfill_trades()
        sys.exit(0)

    if command == "analysis":
        run_analysis()
        sys.exit(0)

    if command == "setup":
        reassemble_data_zip()
        generate_data_directory()
        sys.exit(0)

    if command == "teardown":
        cleanup_data_directory()
        sys.exit(0)


if __name__ == "__main__":
    main()
