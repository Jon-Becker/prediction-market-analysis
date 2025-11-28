import glob
import os
import sys

from src import KalshiClient
from src.analysis import cleanup_data_directory, generate_data_directory, run_analysis


def reassemble_data_zip():
    """Reassemble data.zip from chunks if it doesn't exist."""
    if os.path.exists("data.zip"):
        return

    chunks = sorted(glob.glob("data.zip.*"))
    if not chunks:
        print("No data.zip chunks found.")
        return

    print("Reassembling data.zip from chunks...")
    with open("data.zip", "wb") as output:
        for chunk in chunks:
            with open(chunk, "rb") as f:
                output.write(f.read())
    print("data.zip reassembled.")


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
