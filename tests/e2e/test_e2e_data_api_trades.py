"""Data API trade-tape scenario: market scoping, the 10,000 offset cap with
window bisection, malformed-row skipping, and composite-key dedup."""

import duckdb

COMPOSITE_KEY = "(transaction_hash, proxy_wallet, asset, side, timestamp, size, price)"


def test_data_api_backfill_bisects_offset_cap_and_dedups(seeded_workdir, admin, fx):
    from src.indexers.polymarket.data_api_trades import PolymarketDataApiTradesIndexer

    PolymarketDataApiTradesIndexer(start=fx.DATA_START, end=fx.DATA_END).run()

    total, distinct = duckdb.sql(
        f"SELECT COUNT(*), COUNT(DISTINCT {COMPOSITE_KEY}) FROM 'data/polymarket/data_api_trades/trades_*.parquet'"
    ).fetchone()
    # The two malformed fixture rows are skipped at parse; everything else is
    # persisted exactly once despite the full window overflowing the offset cap.
    assert total == fx.DATA_TRADES_COUNT
    assert distinct == fx.DATA_TRADES_COUNT

    assert not (seeded_workdir / "data/polymarket/.data_api_trades_cursor").exists()

    requests = admin.events("/data-api/trades")
    windows = {(r["meta"]["start"], r["meta"]["end"]) for r in requests}
    mid = (fx.DATA_START + fx.DATA_END) // 2
    # The full window trips the offset cap, so the indexer must bisect it.
    assert (fx.DATA_START, fx.DATA_END) in windows
    assert (fx.DATA_START, mid) in windows
    assert (mid + 1, fx.DATA_END) in windows
    # 11 pages to detect truncation, then 7 per non-truncated half.
    assert len(requests) == 25

    # Every request was scoped to the condition IDs from the markets dataset.
    scopes = {r["meta"]["market"] for r in requests}
    assert scopes == {",".join(fx.featured_condition_id(i) for i in range(fx.FEATURED_COUNT))}
    assert max(r["meta"]["offset"] for r in requests) == 10000
