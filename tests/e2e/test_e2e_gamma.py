"""Gamma REST scenarios: markets backfill, two-phase events walk, wire-level retry."""

import duckdb


def test_markets_backfill_chunks_and_clears_offset_cursor(workdir, admin, fx):
    from src.indexers.polymarket.markets import PolymarketMarketsIndexer

    PolymarketMarketsIndexer().run()

    files = sorted(f.name for f in (workdir / "data/polymarket/markets").glob("markets_*.parquet"))
    assert files == ["markets_0_10000.parquet", "markets_10000_20000.parquet", "markets_20000_25000.parquet"]

    total, distinct, featured = duckdb.sql(
        "SELECT COUNT(*), COUNT(DISTINCT id), COUNT(*) FILTER (WHERE condition_id <> '') "
        "FROM 'data/polymarket/markets/markets_*.parquet'"
    ).fetchone()
    assert total == fx.TOTAL_MARKETS
    assert distinct == fx.TOTAL_MARKETS
    assert featured == fx.FEATURED_COUNT

    # Clean completion removes the offset cursor.
    assert not (workdir / "data/polymarket/.backfill_offset").exists()

    # 50 full pages of 500 plus the final empty page.
    assert admin.count("/gamma/markets") == fx.TOTAL_MARKETS // 500 + 1


def test_events_two_phase_keyset_walk(workdir, admin, fx):
    from src.indexers.polymarket.events import PolymarketEventsIndexer

    PolymarketEventsIndexer().run()

    total, distinct = duckdb.sql(
        "SELECT COUNT(*), COUNT(DISTINCT id) FROM 'data/polymarket/events/events_*.parquet'"
    ).fetchone()
    assert total == fx.TOTAL_EVENTS
    assert distinct == fx.TOTAL_EVENTS

    open_count, closed_count = duckdb.sql(
        "SELECT COUNT(*) FILTER (WHERE NOT closed), COUNT(*) FILTER (WHERE closed) "
        "FROM 'data/polymarket/events/events_*.parquet'"
    ).fetchone()
    assert open_count == fx.OPEN_EVENTS
    assert closed_count == fx.CLOSED_EVENTS

    assert not (workdir / "data/polymarket/.events_backfill_cursor").exists()

    # Both phases walked the keyset endpoint: two pages per closed value.
    requests = admin.events("/gamma/events/keyset")
    assert [r["meta"]["closed"] for r in requests] == ["false", "false", "true", "true"]
    assert [r["meta"]["cursor"] for r in requests] == [0, 500, 0, 500]


def test_transient_500_is_retried_through_real_httpx(workdir, admin, fx):
    from src.indexers.polymarket.events import PolymarketEventsIndexer

    admin.fail_next("/gamma/events/keyset", status=500, times=1)

    PolymarketEventsIndexer().run()

    total = duckdb.sql("SELECT COUNT(*) FROM 'data/polymarket/events/events_*.parquet'").fetchone()[0]
    assert total == fx.TOTAL_EVENTS

    # 4 successful pages plus the one injected failure that tenacity retried.
    assert admin.count("/gamma/events/keyset") == 5
