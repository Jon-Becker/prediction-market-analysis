from collections.abc import Generator
from typing import Optional, Union

import httpx

from src.common.client import retry_request
from src.indexers.polymarket.models import Market, Trade

GAMMA_API_URL = "https://gamma-api.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"


class PolymarketClient:
    def __init__(
        self,
        gamma_url: str = GAMMA_API_URL,
        data_url: str = DATA_API_URL,
    ):
        self.gamma_url = gamma_url
        self.data_url = data_url
        self.client = httpx.Client(timeout=30.0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def close(self):
        self.client.close()

    @retry_request()
    def _get(self, url: str, params: Optional[dict] = None) -> Union[dict, list]:
        """Make a GET request with retry/backoff."""
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_markets(self, limit: int = 500, offset: int = 0, **kwargs) -> list[Market]:
        """Fetch markets from Gamma API."""
        params = {"limit": limit, "offset": offset, **kwargs}
        data = self._get(f"{self.gamma_url}/markets", params=params)
        if isinstance(data, list):
            return [Market.from_dict(m) for m in data]
        return [Market.from_dict(m) for m in data.get("markets", data)]

    def iter_markets(self, limit: int = 500, offset: int = 0) -> Generator[tuple[list[Market], int], None, None]:
        """Iterate through all markets using offset pagination.

        Yields:
            Tuple of (markets, next_offset) where next_offset is -1 when done.
        """
        current_offset = offset

        while True:
            markets = self.get_markets(limit=limit, offset=current_offset)

            if not markets:
                yield [], -1
                break

            next_offset = current_offset + len(markets)
            yield markets, next_offset

            if len(markets) < limit:
                break

            current_offset = next_offset

    def get_trades(self, limit: int = 500, offset: int = 0) -> list[Trade]:
        """Fetch trades from Data API.

        Note: The Polymarket data API does not support filtering by market.
        All trades are returned globally.

        Args:
            limit: Max trades to fetch (max 500)
            offset: Pagination offset
        """
        params = {"limit": min(limit, 500), "offset": offset}
        data = self._get(f"{self.data_url}/trades", params=params)
        if isinstance(data, list):
            return [Trade.from_dict(t) for t in data]
        return [Trade.from_dict(t) for t in data.get("trades", data)]

    def iter_trades(self, limit: int = 500, offset: int = 0) -> Generator[tuple[list[Trade], int], None, None]:
        """Iterate through all trades using offset pagination.

        Note: The Polymarket data API does not support filtering by market.

        Yields:
            Tuple of (trades, next_offset) where next_offset is -1 when done.
        """
        current_offset = offset

        while True:
            trades = self.get_trades(limit=limit, offset=current_offset)

            if not trades:
                yield [], -1
                break

            next_offset = current_offset + len(trades)
            yield trades, next_offset

            if len(trades) < limit:
                break

            current_offset = next_offset
