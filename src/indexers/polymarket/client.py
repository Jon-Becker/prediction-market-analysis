from collections.abc import Generator
from typing import Optional, Union

from src.common.client import HttpClient
from src.indexers.polymarket.models import DataApiTrade, Event, Market, OrderBookSnapshot, PricePoint

GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"

# The Data API `/trades` and `/activity` endpoints return only the most recent
# ~3 years when `start` is omitted or 0; any positive epoch retrieves from that
# point, so `start=1` means full history.
FULL_HISTORY_START = 1

# The Data API hard-caps `offset` at 10,000; deeper scans must narrow the
# `start`/`end` timestamp window instead of paging further.
MAX_DATA_API_OFFSET = 10000


class PolymarketClient:
    def __init__(
        self,
        gamma_url: str = GAMMA_API_URL,
        clob_url: str = CLOB_API_URL,
        data_api_url: str = DATA_API_URL,
    ):
        self.gamma_url = gamma_url
        self.clob_url = clob_url
        self.data_api_url = data_api_url
        self.http = HttpClient(rate_limit=10)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.http.close()

    def close(self):
        self.http.close()

    # -- Gamma API (offset pagination) ----------------------------------------

    def get_markets(self, limit: int = 500, offset: int = 0, **kwargs) -> list[Market]:
        params = {"limit": limit, "offset": offset, **kwargs}
        data: Union[dict, list] = self.http.get(f"{self.gamma_url}/markets", params=params)
        if isinstance(data, list):
            return [Market.from_dict(m) for m in data]
        return [Market.from_dict(m) for m in data.get("markets", data)]

    def iter_markets(self, limit: int = 500, offset: int = 0) -> Generator[tuple[list[Market], int], None, None]:
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

    def get_events(self, limit: int = 500, offset: int = 0, **kwargs) -> list[Event]:
        params = {"limit": limit, "offset": offset, **kwargs}
        data: Union[dict, list] = self.http.get(f"{self.gamma_url}/events", params=params)
        if isinstance(data, list):
            return [Event.from_dict(e) for e in data]
        return [Event.from_dict(e) for e in data.get("events", data)]

    def iter_events(self, limit: int = 500, offset: int = 0) -> Generator[tuple[list[Event], int], None, None]:
        current_offset = offset

        while True:
            events = self.get_events(limit=limit, offset=current_offset)

            if not events:
                yield [], -1
                break

            next_offset = current_offset + len(events)
            yield events, next_offset

            if len(events) < limit:
                break

            current_offset = next_offset

    # -- Gamma API (keyset pagination) -----------------------------------------

    def get_markets_keyset(
        self, limit: int = 100, after_cursor: Optional[str] = None, **kwargs
    ) -> tuple[list[Market], Optional[str]]:
        """Fetch one page of `/markets/keyset`; returns (markets, next_cursor).

        `next_cursor` is None on the final page. `limit` is capped at 100 by the API.
        """
        params = {"limit": limit, **kwargs}
        if after_cursor:
            params["after_cursor"] = after_cursor
        data = self.http.get(f"{self.gamma_url}/markets/keyset", params=params)
        markets = [Market.from_dict(m) for m in data.get("markets", [])]
        return markets, data.get("next_cursor")

    def iter_markets_keyset(
        self, limit: int = 100, after_cursor: Optional[str] = None, **kwargs
    ) -> Generator[tuple[list[Market], Optional[str]], None, None]:
        cursor = after_cursor

        while True:
            markets, cursor = self.get_markets_keyset(limit=limit, after_cursor=cursor, **kwargs)
            yield markets, cursor

            if not cursor:
                break

    def get_events_keyset(
        self, limit: int = 500, after_cursor: Optional[str] = None, **kwargs
    ) -> tuple[list[Event], Optional[str]]:
        """Fetch one page of `/events/keyset`; returns (events, next_cursor).

        `next_cursor` is None on the final page. `limit` is capped at 500 by the API.
        """
        params = {"limit": limit, **kwargs}
        if after_cursor:
            params["after_cursor"] = after_cursor
        data = self.http.get(f"{self.gamma_url}/events/keyset", params=params)
        events = [Event.from_dict(e) for e in data.get("events", [])]
        return events, data.get("next_cursor")

    def iter_events_keyset(
        self, limit: int = 500, after_cursor: Optional[str] = None, **kwargs
    ) -> Generator[tuple[list[Event], Optional[str]], None, None]:
        cursor = after_cursor

        while True:
            events, cursor = self.get_events_keyset(limit=limit, after_cursor=cursor, **kwargs)
            yield events, cursor

            if not cursor:
                break

    # -- Data API --------------------------------------------------------------

    def _get_data_trades_page(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 1000,
        offset: int = 0,
        taker_only: bool = True,
        **kwargs,
    ) -> tuple[list[DataApiTrade], int]:
        """Fetch one `/trades` page; returns (parsed trades, raw row count).

        The raw count includes malformed rows that were skipped during parsing,
        so pagination can detect a short page correctly.
        """
        params = {"limit": limit, "offset": offset, "takerOnly": taker_only, **kwargs}
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        data = self.http.get(f"{self.data_api_url}/trades", params=params)
        rows = data if isinstance(data, list) else data.get("trades") or []
        trades = []
        for row in rows:
            try:
                trades.append(DataApiTrade.from_dict(row))
            except (AttributeError, TypeError, ValueError):
                continue
        return trades, len(rows)

    def get_data_trades(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 1000,
        offset: int = 0,
        taker_only: bool = True,
        **kwargs,
    ) -> list[DataApiTrade]:
        """Fetch one page of the market-wide trade tape from the Data API `/trades`.

        `start`/`end` are unix seconds. When `start` is omitted the API returns
        only the most recent ~3 years; pass `start=FULL_HISTORY_START` (1) for
        full history. `takerOnly` is always sent explicitly: True yields one row
        per fill (the taker side), False adds maker-side rows, which
        double-count a fill when aggregating volume. Malformed rows are skipped.
        """
        trades, _ = self._get_data_trades_page(
            start=start, end=end, limit=limit, offset=offset, taker_only=taker_only, **kwargs
        )
        return trades

    def get_data_trades_window(
        self,
        start: int,
        end: int,
        limit: int = 1000,
        taker_only: bool = True,
    ) -> tuple[list[DataApiTrade], bool]:
        """Fetch every `/trades` row in the [start, end] window by paging offsets.

        Returns (trades, truncated). `truncated` is True when the window still
        had rows beyond the API's 10,000 offset cap — callers must split the
        window into smaller timestamp ranges to recover the missing rows.
        """
        trades: list[DataApiTrade] = []
        offset = 0
        while True:
            page, raw_count = self._get_data_trades_page(
                start=start, end=end, limit=limit, offset=offset, taker_only=taker_only
            )
            trades.extend(page)
            if raw_count < limit:
                return trades, False
            offset += limit
            if offset > MAX_DATA_API_OFFSET:
                return trades, True

    # -- CLOB API --------------------------------------------------------------

    def get_price_history(
        self,
        token_id: str,
        interval: Optional[str] = "max",
        fidelity: Optional[int] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> list[PricePoint]:
        """Fetch the price time series for a CLOB token from `/prices-history`.

        `interval` is one of max, all, 1m, 1w, 1d, 6h, 1h; `fidelity` is the
        resolution in minutes; `start_ts`/`end_ts` are unix seconds. Pass
        `interval=None` when querying by explicit timestamp bounds.
        """
        params: dict = {"market": token_id}
        if interval:
            params["interval"] = interval
        if fidelity is not None:
            params["fidelity"] = fidelity
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts
        data = self.http.get(f"{self.clob_url}/prices-history", params=params)
        history = data.get("history") or [] if isinstance(data, dict) else []
        return [PricePoint.from_dict(token_id, point) for point in history]

    def get_order_book(self, token_id: str) -> OrderBookSnapshot:
        data = self.http.get(f"{self.clob_url}/book", params={"token_id": token_id})
        return OrderBookSnapshot.from_dict(data)

    def get_midpoint(self, token_id: str) -> float:
        data = self.http.get(f"{self.clob_url}/midpoint", params={"token_id": token_id})
        return float(data.get("mid", 0) or 0)

    def get_spread(self, token_id: str) -> float:
        data = self.http.get(f"{self.clob_url}/spread", params={"token_id": token_id})
        return float(data.get("spread", 0) or 0)

    def get_price(self, token_id: str, side: str) -> float:
        """Fetch the best price for a token; `side` is BUY (best bid) or SELL (best ask)."""
        data = self.http.get(f"{self.clob_url}/price", params={"token_id": token_id, "side": side})
        return float(data.get("price", 0) or 0)
