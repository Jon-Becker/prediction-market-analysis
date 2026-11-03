from collections.abc import Generator
from typing import Optional

from src.common.client import HttpClient
from src.indexers.kalshi.models import Event, Market, Trade

KALSHI_API_HOST = "https://external-api.kalshi.com/trade-api/v2"


class KalshiClient:
    def __init__(self, host: str = KALSHI_API_HOST):
        self.host = host
        self.http = HttpClient(base_url=host, rate_limit=20)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.http.close()

    def close(self):
        self.http.close()

    def get_market(self, ticker: str) -> Market:
        data = self.http.get(f"/markets/{ticker}")
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

            data = self.http.get("/markets/trades", params=params)

            trades = [Trade.from_dict(t) for t in data.get("trades", [])]
            if trades:
                all_trades.extend(trades)
                if verbose:
                    print(f"Fetched {len(trades)} trades (total: {len(all_trades)})")

            cursor = data.get("cursor")
            if not cursor:
                break

        return all_trades

    def get_historical_trades(
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

            data = self.http.get("/historical/trades", params=params)

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
        data = self.http.get("/markets", params=params)
        return [Market.from_dict(m) for m in data.get("markets", [])]

    def list_events(self, limit: int = 200, **kwargs) -> list[Event]:
        params = {"limit": limit, **kwargs}
        data = self.http.get("/events", params=params)
        return [Event.from_dict(e) for e in data.get("events", [])]

    def list_all_markets(self, limit: int = 200) -> list[Market]:
        all_markets = []
        cursor = None

        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor

            data = self.http.get("/markets", params=params)

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
        # One of: unopened, open, paused, closed, settled
        status: Optional[str] = None,
        # Multivariate (combo) events: "only" returns only them, "exclude" leaves them out
        mve_filter: Optional[str] = None,
    ) -> Generator[tuple[list[Market], Optional[str]], None, None]:
        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
            if min_close_ts is not None:
                params["min_close_ts"] = min_close_ts
            if max_close_ts is not None:
                params["max_close_ts"] = max_close_ts
            if status is not None:
                params["status"] = status
            if mve_filter is not None:
                params["mve_filter"] = mve_filter

            data = self.http.get("/markets", params=params)

            markets = [Market.from_dict(m) for m in data.get("markets", [])]
            cursor = data.get("cursor")

            yield markets, cursor

            if not cursor:
                break

    def iter_historical_markets(
        self,
        limit: int = 1000,
        cursor: Optional[str] = None,
        # Multivariate (combo) events: "only" returns only them, "exclude" leaves them out
        mve_filter: Optional[str] = "exclude",
    ) -> Generator[tuple[list[Market], Optional[str]], None, None]:
        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
            if mve_filter is not None:
                params["mve_filter"] = mve_filter

            data = self.http.get("/historical/markets", params=params)

            markets = [Market.from_dict(m) for m in data.get("markets", [])]
            cursor = data.get("cursor")

            yield markets, cursor

            if not cursor:
                break

    def iter_event_nested_markets(
        self,
        limit: int = 200,
        cursor: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        # One of: unopened, open, paused, closed, settled
        status: Optional[str] = None,
    ) -> Generator[tuple[list[Market], Optional[str]], None, None]:
        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
            if min_close_ts is not None:
                params["min_close_ts"] = min_close_ts
            if max_close_ts is not None:
                params["max_close_ts"] = max_close_ts
            if status is not None:
                params["status"] = status

            # Return each event's markets nested inside it
            params["with_nested_markets"] = "true"

            data = self.http.get("/events", params=params)

            markets = [
                Market.from_dict(m | {"event_category": e.get("category", "")})
                for e in data.get("events", [])
                for m in e.get("markets", [])
            ]

            cursor = data.get("cursor")

            yield markets, cursor

            if not cursor:
                break

    def get_recent_trades(self, limit: int = 100) -> list[Trade]:
        data = self.http.get("/markets/trades", params={"limit": limit})
        return [Trade.from_dict(t) for t in data.get("trades", [])]
