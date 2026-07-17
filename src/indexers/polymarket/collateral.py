"""Indexer for the Polymarket FPMM collateral token lookup."""

import concurrent.futures
import json
from pathlib import Path
from typing import Optional

import duckdb
from tqdm import tqdm
from web3 import Web3

from src.common.indexer import Indexer
from src.indexers.polymarket.blockchain import PolygonClient

# Minimal ABI for FixedProductMarketMaker.collateralToken()
COLLATERAL_TOKEN_ABI = {
    "constant": True,
    "inputs": [],
    "name": "collateralToken",
    "outputs": [{"name": "", "type": "address"}],
    "stateMutability": "view",
    "type": "function",
}

# Minimal ABI for ERC20.symbol()
ERC20_SYMBOL_ABI = {
    "constant": True,
    "inputs": [],
    "name": "symbol",
    "outputs": [{"name": "", "type": "string"}],
    "stateMutability": "view",
    "type": "function",
}

LEGACY_TRADES_DIR = Path("data/polymarket/legacy_trades")
LOOKUP_FILE = Path("data/polymarket/fpmm_collateral_lookup.json")


class PolymarketCollateralIndexer(Indexer):
    """Builds the FPMM address -> collateral token lookup from legacy trade data."""

    def __init__(self, max_workers: int = 10):
        super().__init__(
            name="polymarket_collateral",
            description="Builds the FPMM collateral token lookup from Polymarket legacy trades",
        )
        self._max_workers = max_workers
        self._symbol_cache: dict[str, str] = {}

    def _fetch_collateral(self, client: PolygonClient, fpmm_address: str) -> Optional[dict]:
        """Resolve the collateral token address and symbol for an FPMM contract."""
        try:
            fpmm = client.w3.eth.contract(
                address=Web3.to_checksum_address(fpmm_address),
                abi=[COLLATERAL_TOKEN_ABI],
            )
            collateral_address = fpmm.functions.collateralToken().call()
        except Exception as e:
            tqdm.write(f"Error fetching collateral token for {fpmm_address}: {e}")
            return None

        symbol = self._symbol_cache.get(collateral_address)
        if symbol is None:
            try:
                erc20 = client.w3.eth.contract(
                    address=Web3.to_checksum_address(collateral_address),
                    abi=[ERC20_SYMBOL_ABI],
                )
                symbol = erc20.functions.symbol().call()
            except Exception as e:
                tqdm.write(f"Error fetching symbol for {collateral_address}: {e}")
                symbol = "UNKNOWN"
            self._symbol_cache[collateral_address] = symbol

        return {"collateral_address": collateral_address, "collateral_symbol": symbol}

    def run(self) -> None:
        """Build the FPMM collateral lookup for all distinct FPMM addresses in the legacy trades data.

        The lookup file itself acts as the cursor: known addresses are skipped, and the
        file is rewritten after each parallel batch so interrupts lose at most one batch.
        """
        if not list(LEGACY_TRADES_DIR.glob("trades_*.parquet")):
            print(f"No legacy trades found in {LEGACY_TRADES_DIR}, nothing to do")
            return

        LOOKUP_FILE.parent.mkdir(parents=True, exist_ok=True)

        lookup: dict[str, dict] = {}
        if LOOKUP_FILE.exists():
            try:
                lookup = json.loads(LOOKUP_FILE.read_text())
            except ValueError:
                lookup = {}

        rows = duckdb.sql(f"SELECT DISTINCT fpmm_address FROM '{LEGACY_TRADES_DIR}/trades_*.parquet'").fetchall()
        addresses = [row[0] for row in rows]
        to_process = [address for address in addresses if address not in lookup]
        print(f"Found {len(addresses)} FPMM addresses ({len(to_process)} new)")

        if not to_process:
            print("Nothing to process")
            return

        client = PolygonClient()
        resolved = 0
        pbar = tqdm(total=len(to_process), desc="Resolving collateral", unit=" fpmm")

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                for batch_start in range(0, len(to_process), self._max_workers):
                    batch = to_process[batch_start : batch_start + self._max_workers]
                    futures = {executor.submit(self._fetch_collateral, client, address): address for address in batch}

                    for future in concurrent.futures.as_completed(futures):
                        info = future.result()
                        if info is not None:
                            lookup[futures[future]] = info
                            resolved += 1
                        pbar.update(1)

                    # Persist after each batch so interrupts lose at most one batch
                    LOOKUP_FILE.write_text(json.dumps(lookup, indent=2))

        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved.")
        finally:
            pbar.close()

        print(f"\nCollateral lookup complete: {resolved} new addresses resolved ({len(lookup)} total)")
