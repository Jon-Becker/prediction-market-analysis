"""Unit tests for PolymarketClient and its Gamma/CLOB models (no network)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import src.indexers.polymarket.client as client_module
from src.indexers.polymarket.client import FULL_HISTORY_START, PolymarketClient
from src.indexers.polymarket.models import DataApiTrade, Event, Market, OrderBookSnapshot, PricePoint

GAMMA_MARKET = {
    "id": "500000",
    "conditionId": "0x" + "ab" * 32,
    "question": "Will it rain tomorrow?",
    "slug": "will-it-rain-tomorrow",
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0.65", "0.35"]',
    "clobTokenIds": '["111", "222"]',
    "volume": "1234.5",
    "liquidity": "678.9",
    "active": True,
    "closed": False,
    "endDate": "2026-08-01T00:00:00Z",
    "createdAt": "2026-07-01T12:30:00Z",
    "marketMakerAddress": "0x" + "cd" * 20,
}

GAMMA_EVENT = {
    "id": "2891",
    "slug": "nba-finals-2026",
    "title": "NBA Finals 2026",
    "category": "Sports",
    "tags": [{"id": "1", "label": "Sports", "slug": "sports"}, {"id": "2", "label": "NBA", "slug": "nba"}],
    "markets": [{"id": "500000"}, {"id": "500001"}],
    "volume": 1335.05,
    "liquidity": 42.0,
    "active": True,
    "closed": True,
    "startDate": "2026-06-04T00:00:00Z",
    "endDate": "2026-06-22T00:00:00Z",
    "createdAt": "2026-05-27T14:40:02.074Z",
}

DATA_API_TRADE = {
    "proxyWallet": "0x" + "ef" * 20,
    "side": "BUY",
    "asset": "111",
    "conditionId": "0x" + "ab" * 32,
    "size": "10.5",
    "price": "0.65",
    "timestamp": 1752700000,
    "transactionHash": "0x" + "12" * 32,
    "title": "Will it rain tomorrow?",
    "slug": "will-it-rain-tomorrow",
    "eventSlug": "weather-week",
    "outcome": "Yes",
    "outcomeIndex": 0,
    "name": "trader",
    "pseudonym": "Quiet-Fox",
}

ORDER_BOOK = {
    "market": "0x" + "ab" * 32,
    "asset_id": "111",
    "timestamp": "1784254990511",
    "hash": "720dff71476f22d3",
    "bids": [{"price": "0.45", "size": "100"}, {"price": "0.44", "size": "50.5"}],
    "asks": [{"price": "0.46", "size": "150"}],
    "min_order_size": "5",
    "tick_size": "0.001",
    "neg_risk": True,
    "last_trade_price": "0.45",
}


class FakeHttp:
    """Stub for HttpClient that returns canned responses and records calls."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, *, params: dict | None = None):
        self.calls.append((url, params or {}))
        return self.responses.pop(0)

    def close(self):
        pass


def make_client(responses: list) -> PolymarketClient:
    client = PolymarketClient()
    client.http.close()
    client.http = FakeHttp(responses)
    return client


# -- Model parsing -------------------------------------------------------------


def test_market_from_dict():
    market = Market.from_dict(GAMMA_MARKET)

    assert market.id == "500000"
    assert market.condition_id == "0x" + "ab" * 32
    assert market.clob_token_ids == '["111", "222"]'
    assert market.volume == 1234.5
    assert market.liquidity == 678.9
    assert market.active is True
    assert market.closed is False
    assert market.end_date == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert market.created_at == datetime(2026, 7, 1, 12, 30, tzinfo=timezone.utc)
    assert market.market_maker_address == "0x" + "cd" * 20


def test_market_from_dict_defaults():
    market = Market.from_dict({})

    assert market.id == ""
    assert market.outcomes == "[]"
    assert market.volume == 0.0
    assert market.active is False
    assert market.end_date is None
    assert market.market_maker_address is None


def test_event_from_dict():
    event = Event.from_dict(GAMMA_EVENT)

    assert event.id == "2891"
    assert event.slug == "nba-finals-2026"
    assert event.title == "NBA Finals 2026"
    assert event.category == "Sports"
    assert json.loads(event.tags) == ["sports", "nba"]
    assert json.loads(event.market_ids) == ["500000", "500001"]
    assert event.volume == 1335.05
    assert event.liquidity == 42.0
    assert event.active is True
    assert event.closed is True
    assert event.start_date == datetime(2026, 6, 4, tzinfo=timezone.utc)
    assert event.end_date == datetime(2026, 6, 22, tzinfo=timezone.utc)
    assert event.created_at is not None


def test_event_from_dict_defaults():
    event = Event.from_dict({})

    assert event.id == ""
    assert event.category is None
    assert json.loads(event.tags) == []
    assert json.loads(event.market_ids) == []
    assert event.volume == 0.0
    assert event.start_date is None


def test_price_point_from_dict():
    point = PricePoint.from_dict("111", {"t": 1784251816, "p": 0.0125})

    assert point.token_id == "111"
    assert point.timestamp == 1784251816
    assert point.price == 0.0125


def test_order_book_snapshot_from_dict():
    book = OrderBookSnapshot.from_dict(ORDER_BOOK)

    assert book.condition_id == "0x" + "ab" * 32
    assert book.token_id == "111"
    assert book.timestamp == 1784254990511
    assert book.hash == "720dff71476f22d3"
    assert [(level.price, level.size) for level in book.bids] == [(0.45, 100.0), (0.44, 50.5)]
    assert [(level.price, level.size) for level in book.asks] == [(0.46, 150.0)]
    assert book.min_order_size == 5.0
    assert book.tick_size == 0.001
    assert book.neg_risk is True
    assert book.last_trade_price == 0.45


def test_order_book_snapshot_from_dict_empty():
    book = OrderBookSnapshot.from_dict({})

    assert book.condition_id == ""
    assert book.bids == []
    assert book.asks == []
    assert book.neg_risk is False


# -- Gamma offset pagination ----------------------------------------------------


def test_iter_markets_stops_on_short_page():
    client = make_client([[GAMMA_MARKET] * 2, [GAMMA_MARKET]])

    pages = list(client.iter_markets(limit=2))

    assert [(len(markets), offset) for markets, offset in pages] == [(2, 2), (1, 3)]
    assert all(isinstance(m, Market) for m in pages[0][0])
    assert client.http.calls[0] == ("https://gamma-api.polymarket.com/markets", {"limit": 2, "offset": 0})
    assert client.http.calls[1][1] == {"limit": 2, "offset": 2}


def test_iter_markets_empty_first_page_yields_sentinel():
    client = make_client([[]])

    assert list(client.iter_markets(limit=2)) == [([], -1)]


def test_iter_events_matches_iter_markets_semantics():
    client = make_client([[GAMMA_EVENT] * 2, []])

    pages = list(client.iter_events(limit=2))

    assert [(len(events), offset) for events, offset in pages] == [(2, 2), (0, -1)]
    assert all(isinstance(e, Event) for e in pages[0][0])
    assert client.http.calls[0] == ("https://gamma-api.polymarket.com/events", {"limit": 2, "offset": 0})


def test_get_events_passes_filters():
    client = make_client([[GAMMA_EVENT]])

    events = client.get_events(limit=10, offset=5, closed=True)

    assert len(events) == 1
    assert client.http.calls == [
        ("https://gamma-api.polymarket.com/events", {"limit": 10, "offset": 5, "closed": True})
    ]


# -- Gamma keyset pagination ----------------------------------------------------


def test_get_markets_keyset_first_page_omits_cursor():
    client = make_client([{"markets": [GAMMA_MARKET], "next_cursor": "abc123"}])

    markets, cursor = client.get_markets_keyset(limit=100, closed=True)

    assert len(markets) == 1
    assert isinstance(markets[0], Market)
    assert cursor == "abc123"
    url, params = client.http.calls[0]
    assert url == "https://gamma-api.polymarket.com/markets/keyset"
    assert params == {"limit": 100, "closed": True}
    assert "after_cursor" not in params


def test_get_markets_keyset_final_page_returns_none_cursor():
    client = make_client([{"markets": [GAMMA_MARKET]}])

    markets, cursor = client.get_markets_keyset(after_cursor="abc123")

    assert len(markets) == 1
    assert cursor is None
    assert client.http.calls[0][1]["after_cursor"] == "abc123"


def test_iter_markets_keyset_follows_cursor_until_exhausted():
    client = make_client(
        [
            {"markets": [GAMMA_MARKET] * 2, "next_cursor": "page2"},
            {"markets": [GAMMA_MARKET]},
        ]
    )

    pages = list(client.iter_markets_keyset(limit=2))

    assert [(len(markets), cursor) for markets, cursor in pages] == [(2, "page2"), (1, None)]
    assert "after_cursor" not in client.http.calls[0][1]
    assert client.http.calls[1][1]["after_cursor"] == "page2"


def test_iter_events_keyset_resumes_from_cursor():
    client = make_client([{"events": [GAMMA_EVENT]}])

    pages = list(client.iter_events_keyset(after_cursor="resume-here"))

    assert [(len(events), cursor) for events, cursor in pages] == [(1, None)]
    url, params = client.http.calls[0]
    assert url == "https://gamma-api.polymarket.com/events/keyset"
    assert params["after_cursor"] == "resume-here"


# -- Data API trades --------------------------------------------------------------


def test_data_api_trade_from_dict():
    trade = DataApiTrade.from_dict(DATA_API_TRADE)

    assert trade.proxy_wallet == "0x" + "ef" * 20
    assert trade.side == "BUY"
    assert trade.asset == "111"
    assert trade.condition_id == "0x" + "ab" * 32
    assert trade.size == 10.5
    assert trade.price == 0.65
    assert trade.timestamp == 1752700000
    assert trade.transaction_hash == "0x" + "12" * 32
    assert trade.title == "Will it rain tomorrow?"
    assert trade.event_slug == "weather-week"
    assert trade.outcome == "Yes"
    assert trade.outcome_index == 0
    assert trade.pseudonym == "Quiet-Fox"


def test_data_api_trade_from_dict_defaults():
    trade = DataApiTrade.from_dict({})

    assert trade.proxy_wallet == ""
    assert trade.asset == ""
    assert trade.size == 0.0
    assert trade.price == 0.0
    assert trade.timestamp == 0
    assert trade.outcome_index == -1
    assert trade.pseudonym == ""


def test_get_data_trades_default_window_omits_start_and_sends_taker_only():
    client = make_client([[DATA_API_TRADE]])

    trades = client.get_data_trades()

    assert len(trades) == 1
    url, params = client.http.calls[0]
    assert url == "https://data-api.polymarket.com/trades"
    assert params == {"limit": 1000, "offset": 0, "takerOnly": True}
    assert "start" not in params  # API defaults to its ~3-year window
    assert "end" not in params


def test_get_data_trades_full_history_sends_start_1():
    client = make_client([[DATA_API_TRADE]])

    client.get_data_trades(start=FULL_HISTORY_START, end=1752700000, taker_only=False)

    assert client.http.calls[0][1] == {
        "limit": 1000,
        "offset": 0,
        "takerOnly": False,
        "start": 1,
        "end": 1752700000,
    }


def test_get_data_trades_skips_malformed_rows():
    malformed = dict(DATA_API_TRADE, size="not-a-number")
    client = make_client([[DATA_API_TRADE, malformed, "not-a-dict", None]])

    trades = client.get_data_trades()

    assert len(trades) == 1
    assert trades[0].size == 10.5


def test_get_data_trades_window_paginates_offsets_until_short_page():
    client = make_client([[DATA_API_TRADE] * 2, [DATA_API_TRADE] * 2, [DATA_API_TRADE]])

    trades, truncated = client.get_data_trades_window(1, 100, limit=2)

    assert len(trades) == 5
    assert truncated is False
    assert [call[1]["offset"] for call in client.http.calls] == [0, 2, 4]
    for _, params in client.http.calls:
        assert params["start"] == 1
        assert params["end"] == 100
        assert params["takerOnly"] is True


def test_get_data_trades_window_truncates_at_offset_cap(monkeypatch):
    monkeypatch.setattr(client_module, "MAX_DATA_API_OFFSET", 4)
    client = make_client([[DATA_API_TRADE] * 2] * 3)

    trades, truncated = client.get_data_trades_window(1, 100, limit=2)

    assert len(trades) == 6
    assert truncated is True
    assert [call[1]["offset"] for call in client.http.calls] == [0, 2, 4]


def test_get_data_trades_window_counts_raw_rows_for_pagination():
    malformed = dict(DATA_API_TRADE, size="not-a-number")
    client = make_client([[DATA_API_TRADE, malformed], [DATA_API_TRADE]])

    trades, truncated = client.get_data_trades_window(1, 100, limit=2)

    # First page is full in raw rows despite the skipped malformed row,
    # so pagination must continue to the second (short) page.
    assert len(trades) == 2
    assert truncated is False
    assert len(client.http.calls) == 2


# -- CLOB endpoints --------------------------------------------------------------


def test_get_price_history_parses_points_in_order():
    client = make_client([{"history": [{"t": 100, "p": 0.5}, {"t": 160, "p": 0.55}]}])

    points = client.get_price_history("111")

    assert [(p.token_id, p.timestamp, p.price) for p in points] == [("111", 100, 0.5), ("111", 160, 0.55)]
    assert client.http.calls == [("https://clob.polymarket.com/prices-history", {"market": "111", "interval": "max"})]


def test_get_price_history_passes_bounds_without_interval():
    client = make_client([{"history": []}])

    points = client.get_price_history("111", interval=None, fidelity=10, start_ts=100, end_ts=200)

    assert points == []
    assert client.http.calls[0][1] == {"market": "111", "fidelity": 10, "startTs": 100, "endTs": 200}


def test_get_price_history_tolerates_missing_history():
    client = make_client([{}])

    assert client.get_price_history("111") == []


def test_get_order_book():
    client = make_client([ORDER_BOOK])

    book = client.get_order_book("111")

    assert isinstance(book, OrderBookSnapshot)
    assert book.token_id == "111"
    assert client.http.calls == [("https://clob.polymarket.com/book", {"token_id": "111"})]


def test_get_midpoint_spread_and_price():
    client = make_client([{"mid": "0.012"}, {"spread": "0.014"}, {"price": "0.005"}])

    assert client.get_midpoint("111") == 0.012
    assert client.get_spread("111") == 0.014
    assert client.get_price("111", "BUY") == 0.005
    assert client.http.calls == [
        ("https://clob.polymarket.com/midpoint", {"token_id": "111"}),
        ("https://clob.polymarket.com/spread", {"token_id": "111"}),
        ("https://clob.polymarket.com/price", {"token_id": "111", "side": "BUY"}),
    ]
