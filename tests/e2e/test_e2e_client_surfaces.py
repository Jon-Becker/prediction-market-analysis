"""Client-level wire scenarios: CLOB REST parsing, keyset chaining, the
15-day price-history cap, and the no-egress canary."""

import os

import httpx
import pytest


def test_clob_rest_surfaces_parse_string_payloads(admin, fx):
    from src.indexers.polymarket.client import PolymarketClient

    token = fx.all_featured_token_ids()[0]
    with PolymarketClient() as client:
        book = client.get_order_book(token)
        assert book.token_id == token
        assert book.timestamp == int(fx.WS_BOOK_TS_MS)
        assert [(level.price, level.size) for level in book.bids] == [(0.45, 100.5), (0.44, 200.0)]
        assert [(level.price, level.size) for level in book.asks] == [(0.55, 50.0), (0.56, 75.0)]
        assert book.min_order_size == 5.0
        assert book.tick_size == 0.01

        assert client.get_midpoint(token) == 0.5
        assert client.get_spread(token) == 0.1
        assert client.get_price(token, "BUY") == 0.45
        assert client.get_price(token, "SELL") == 0.55


def test_markets_keyset_pages_chain_via_cursor(admin, fx):
    from src.indexers.polymarket.client import PolymarketClient

    with PolymarketClient() as client:
        first_page, cursor = client.get_markets_keyset(limit=100)
        assert len(first_page) == 100
        assert cursor is not None

        second_page, _ = client.get_markets_keyset(limit=100, after_cursor=cursor)
        assert len(second_page) == 100
        assert {m.id for m in first_page}.isdisjoint({m.id for m in second_page})


def test_price_history_rejects_windows_longer_than_15_days(admin, fx):
    from src.indexers.polymarket.client import PolymarketClient

    token = fx.all_featured_token_ids()[0]
    with PolymarketClient() as client:
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            client.get_price_history(
                token,
                interval=None,
                fidelity=60,
                start_ts=fx.MARKETS_BASE_TS,
                end_ts=fx.MARKETS_BASE_TS + 16 * fx.DAY,
            )
    assert excinfo.value.response.status_code == 400


def test_no_external_egress_is_possible():
    if not os.environ.get("E2E_EXPECT_NO_EGRESS"):
        pytest.skip("canary only applies inside the Compose internal network")
    with pytest.raises(httpx.TransportError):
        httpx.get("https://gamma-api.polymarket.com/markets", timeout=5.0)
