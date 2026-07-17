"""Collateral scenario: legacy-trades parquet -> eth_call lookups with symbol
caching, UNKNOWN fallback on revert, and an idempotent second run."""

import json


def test_collateral_lookup_caches_symbols_and_is_idempotent(workdir, admin, fx):
    from src.indexers.polymarket.collateral import PolymarketCollateralIndexer
    from src.indexers.polymarket.fpmm_trades import PolymarketLegacyTradesIndexer

    # Real cross-indexer dataflow: the collateral indexer reads the distinct
    # FPMM addresses out of the parquet this run produces.
    PolymarketLegacyTradesIndexer(
        from_block=fx.E2E_START_BLOCK, to_block=fx.HEAD_BLOCK, chunk_size=1000, max_workers=4
    ).run()

    before = admin.count("rpc:eth_call")
    PolymarketCollateralIndexer(max_workers=1).run()

    # 3 collateralToken() calls + 2 symbol() calls: USDC is fetched once and
    # cached, the no-symbol token reverts and is cached as UNKNOWN.
    assert admin.count("rpc:eth_call") - before == 5

    lookup_file = workdir / "data/polymarket/fpmm_collateral_lookup.json"
    lookup = {k.lower(): v for k, v in json.loads(lookup_file.read_text()).items()}
    fpmm_a, fpmm_b, fpmm_c = fx.build_fpmm_addresses()
    assert set(lookup) == {fpmm_a, fpmm_b, fpmm_c}
    for fpmm in (fpmm_a, fpmm_b):
        assert lookup[fpmm]["collateral_address"].lower() == fx.TOKEN_USDC
        assert lookup[fpmm]["collateral_symbol"] == "USDC"
    assert lookup[fpmm_c]["collateral_address"].lower() == fx.TOKEN_NO_SYMBOL
    assert lookup[fpmm_c]["collateral_symbol"] == "UNKNOWN"

    # Second run: the lookup file is the cursor, so no address is re-resolved.
    PolymarketCollateralIndexer(max_workers=1).run()
    assert admin.count("rpc:eth_call") - before == 5
