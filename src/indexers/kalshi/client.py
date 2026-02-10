from collections.abc import Generator
from typing import Optional

import httpx

from src.common.client import retry_request
from src.indexers.kalshi.models import Market, Trade

KALSHI_API_HOST = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiClient:
    def __init__(self, host: str = KALSHI_API_HOST):
        self.host = host
        self.client = httpx.Client(base_url=host, timeout=30.0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def close(self):
        self.client.close()

    @retry_request()
    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """Make a GET request with retry/backoff."""
        response = self.client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def get_market(self, ticker: str) -> Market:
        data = self._get(f"/markets/{ticker}")
        return Market.from_dict(data["market"])

    def get_market_trades(
        self,
        ticker: str,
        limit: int = 1000,
        verbose: bool = True,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> list[Trade]:
        all_trades = []
        cursor = None

        while True:
            params = {"ticker": ticker, "limit": limit}
            if cursor:
                params["cursor"] = cursor
            if min_ts is not None:
                params["min_ts"] = min_ts
            if max_ts is not None:
                params["max_ts"] = max_ts

            data = self._get("/markets/trades", params=params)

            trades = [Trade.from_dict(t) for t in data.get("trades", [])]
            if trades:
                all_trades.extend(trades)
                if verbose:
                    print(f"Fetched {len(trades)} trades (total: {len(all_trades)})")

            cursor = data.get("cursor")
            if not cursor:
                break

        return all_trades

    def list_markets(self, limit: int = 20, **kwargs) -> list[Market]:
        params = {"limit": limit, **kwargs}
        data = self._get("/markets", params=params)
        return [Market.from_dict(m) for m in data.get("markets", [])]

    def list_all_markets(self, limit: int = 200) -> list[Market]:
        all_markets = []
        cursor = None

        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor

            data = self._get("/markets", params=params)

            markets = [Market.from_dict(m) for m in data.get("markets", [])]
            if markets:
                all_markets.extend(markets)
                print(f"Fetched {len(markets)} markets (total: {len(all_markets)})")

            cursor = data.get("cursor")
            if not cursor:
                break

        return all_markets

    def iter_markets(
        self,
        limit: int = 200,
        cursor: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
    ) -> Generator[tuple[list[Market], Optional[str]], None, None]:
        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
            if min_close_ts is not None:
                params["min_close_ts"] = min_close_ts
            if max_close_ts is not None:
                params["max_close_ts"] = max_close_ts

            data = self._get("/markets", params=params)

            markets = [Market.from_dict(m) for m in data.get("markets", [])]
            cursor = data.get("cursor")

            yield markets, cursor

            if not cursor:
                break

    def get_recent_trades(self, limit: int = 100) -> list[Trade]:
        data = self._get("/markets/trades", params={"limit": limit})
        return [Trade.from_dict(t) for t in data.get("trades", [])]
