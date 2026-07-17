"""Polygon JSON-RPC scenarios driven through the real web3 provider: CTF
trades, FPMM trades with too-large bisection, block sampling/interpolation,
and condition resolutions."""

import json
from datetime import datetime, timezone

import duckdb
import pandas as pd


def test_ctf_trades_backfill_decodes_real_logs(workdir, admin, fx):
    from src.indexers.polymarket.trades import PolymarketTradesIndexer

    PolymarketTradesIndexer().run()

    params = fx.build_ctf_trade_params()
    df = pd.concat(pd.read_parquet(f) for f in (workdir / "data/polymarket/trades").glob("trades_*.parquet"))

    # The undecodable fixture log is skipped; every valid log decodes exactly once.
    assert len(df) == len(params)
    assert df[["transaction_hash", "log_index", "_contract"]].drop_duplicates().shape[0] == len(params)
    assert set(df["_contract"]) == {"CTF Exchange", "NegRisk CTF Exchange"}

    expected = sorted(
        (block, maker_amount, taker_amount, fee) for _, block, _, _, _, maker_amount, taker_amount, fee in params
    )
    actual = sorted(zip(df["block_number"], df["maker_amount"], df["taker_amount"], df["fee"]))
    assert actual == expected

    # Large asset IDs are persisted as strings to avoid parquet overflow.
    assert df["maker_asset_id"].map(lambda v: isinstance(v, str)).all()
    assert df["taker_asset_id"].map(lambda v: isinstance(v, str)).all()
    assert {a.lower() for a in df["maker"]} == {fx.MAKER_ADDR}
    assert {a.lower() for a in df["taker"]} == {fx.TAKER_ADDR}

    assert not (workdir / "data/polymarket/.backfill_block_cursor").exists()


def test_fpmm_trades_bisect_too_large_log_ranges(workdir, admin, fx):
    from src.indexers.polymarket.fpmm_trades import PolymarketLegacyTradesIndexer

    PolymarketLegacyTradesIndexer(
        from_block=fx.E2E_START_BLOCK, to_block=fx.HEAD_BLOCK, chunk_size=1000, max_workers=4
    ).run()

    params = fx.build_fpmm_trade_params()
    df = pd.concat(pd.read_parquet(f) for f in (workdir / "data/polymarket/legacy_trades").glob("trades_*.parquet"))

    assert len(df) == len(params)
    expected = sorted(
        (block, fpmm, is_buy, str(amount), outcome_index, str(tokens))
        for fpmm, block, _, is_buy, amount, _, outcome_index, tokens in params
    )
    actual = sorted(
        (row.block_number, row.fpmm_address.lower(), row.is_buy, row.amount, row.outcome_index, row.outcome_tokens)
        for row in df.itertuples()
    )
    assert actual == expected

    # The FPMM indexer filters by topic only (no address); the mock rejects
    # topic-only spans wider than TOPIC_ONLY_MAX_SPAN, so each of the three
    # full-size chunks must be bisected once per topic: 2 topics * (3 chunks *
    # 3 requests + 1 short final chunk) = 20 getLogs calls.
    assert admin.count("rpc:eth_getLogs") == 20
    spans = [r["meta"]["span"] for r in admin.events("rpc:eth_getLogs")]
    assert max(spans) == 1000
    assert spans.count(1000) == 6  # each too-large attempt, before its two halves

    assert not (workdir / "data/polymarket/.legacy_backfill_block_cursor").exists()


def test_blocks_indexer_samples_and_interpolates_timestamps(workdir, admin, fx):
    from src.indexers.polymarket.blocks import PolymarketBlocksIndexer

    # Seed the resume point: the blocks indexer derives it from existing
    # filenames, so a prior bucket file makes the run start at E2E_START_BLOCK.
    blocks_dir = workdir / "data/polymarket/blocks"
    blocks_dir.mkdir(parents=True)
    pd.DataFrame([{"block_number": fx.E2E_START_BLOCK - 1, "timestamp": "2023-01-01T00:00:00Z"}]).to_parquet(
        blocks_dir / f"blocks_{fx.E2E_START_BLOCK - 100000}_{fx.E2E_START_BLOCK}.parquet"
    )

    PolymarketBlocksIndexer().run()

    new_file = blocks_dir / f"blocks_{fx.E2E_START_BLOCK}_{fx.HEAD_BLOCK + 1}.parquet"
    assert new_file.exists()
    df = pd.read_parquet(new_file)

    assert list(df["block_number"]) == list(range(fx.E2E_START_BLOCK, fx.HEAD_BLOCK + 1))
    # The chain's block time is exactly linear, so interpolation between the
    # sampled points must reproduce every timestamp exactly.
    expected = [
        datetime.fromtimestamp(fx.block_timestamp(block), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for block in range(fx.E2E_START_BLOCK, fx.HEAD_BLOCK + 1)
    ]
    assert list(df["timestamp"]) == expected

    # Only every 100th block was fetched over RPC.
    assert admin.count("rpc:eth_getBlockByNumber") == (fx.HEAD_BLOCK - fx.E2E_START_BLOCK) // 100 + 1


def test_resolutions_backfill_decodes_payout_arrays(workdir, admin, fx):
    from src.indexers.polymarket.resolutions import PolymarketResolutionsIndexer

    PolymarketResolutionsIndexer(chunk_size=500).run()

    params = fx.build_resolution_params()
    df = pd.concat(pd.read_parquet(f) for f in (workdir / "data/polymarket/resolutions").glob("resolutions_*.parquet"))

    assert len(df) == len(params)
    expected = sorted((block, cid, qid, len(payouts), json.dumps(payouts)) for block, _, cid, qid, payouts in params)
    actual = sorted(
        (row.block_number, row.condition_id, row.question_id, row.outcome_slot_count, row.payout_numerators)
        for row in df.itertuples()
    )
    assert actual == expected
    assert {a.lower() for a in df["oracle"]} == {fx.ORACLE_ADDR}

    duplicates = duckdb.sql(
        "SELECT COUNT(*) - COUNT(DISTINCT (transaction_hash, log_index)) "
        "FROM 'data/polymarket/resolutions/resolutions_*.parquet'"
    ).fetchone()[0]
    assert duplicates == 0

    assert not (workdir / "data/polymarket/.resolutions_block_cursor").exists()
    # ceil(3101 / 500) chunked ranges, each one getLogs call.
    assert admin.count("rpc:eth_getLogs") == 7
