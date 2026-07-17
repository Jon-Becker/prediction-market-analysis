"""Price-history scenario: cross-indexer dataflow (markets parquet -> token
discovery), 15-day windowing, boundary dedup, and resume-skip."""

import duckdb


def test_price_history_windows_dedups_and_resumes(seeded_workdir, admin, fx):
    from src.indexers.polymarket.price_history import PolymarketPriceHistoryIndexer

    PolymarketPriceHistoryIndexer(max_workers=4).run()

    # Tokens were discovered from the markets parquet written by the real
    # markets indexer; every request respected the API's 15-day cap (the mock
    # 400s anything longer) and the per-token window math matched.
    requests = admin.events("/clob/prices-history")
    assert len(requests) == 2 * sum(fx.expected_price_requests(i) for i in range(fx.FEATURED_COUNT))
    assert max(r["meta"]["span"] for r in requests) <= fx.PRICE_WINDOW_DAYS * fx.DAY

    rows = duckdb.sql(
        "SELECT token_id, COUNT(*), COUNT(DISTINCT timestamp) "
        "FROM 'data/polymarket/price_history/prices_*.parquet' GROUP BY token_id"
    ).fetchall()
    expected = {
        token: fx.expected_price_rows(i) for i in range(fx.FEATURED_COUNT) for token in fx.featured_token_ids(i)
    }
    # Window-boundary points are served by both adjacent windows; the indexer
    # must persist each timestamp exactly once per token.
    assert {token: total for token, total, _ in rows} == expected
    assert all(total == distinct for _, total, distinct in rows)

    # Deterministic values straight from the wire format.
    sample_token = fx.featured_token_ids(0)[0]
    ts, price = duckdb.sql(
        "SELECT timestamp, price FROM 'data/polymarket/price_history/prices_*.parquet' "
        f"WHERE token_id = '{sample_token}' ORDER BY timestamp LIMIT 1"
    ).fetchone()
    assert ts == fx.featured_created_ts(0)
    assert price == fx.price_at(sample_token, ts)

    # Second run: every token is already persisted, so resume skips all work.
    before = admin.count("/clob/prices-history")
    PolymarketPriceHistoryIndexer(max_workers=4).run()
    assert admin.count("/clob/prices-history") == before
