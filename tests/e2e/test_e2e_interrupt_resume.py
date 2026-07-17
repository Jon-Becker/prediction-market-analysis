"""Process-level interrupt/resume: a real subprocess is killed with a real
SIGINT mid-run, then relaunched, and the combined dataset must be complete
and duplicate-free. Covers the offset-cursor (markets), block-cursor (CTF
trades), and composite-cursor (Data API trades) families."""

import shutil

import duckdb

MARKETS_CODE = "from src.indexers.polymarket.markets import PolymarketMarketsIndexer; PolymarketMarketsIndexer().run()"

TRADES_CODE_FIRST = (
    "from src.indexers.polymarket.trades import PolymarketTradesIndexer; "
    "PolymarketTradesIndexer(from_block={start}, to_block={end}, chunk_size=25).run()"
)
# The resume leg omits from_block so the indexer reads the block cursor.
TRADES_CODE_RESUME = (
    "from src.indexers.polymarket.trades import PolymarketTradesIndexer; "
    "PolymarketTradesIndexer(to_block={end}, chunk_size=25).run()"
)

DATA_API_CODE = (
    "from src.indexers.polymarket.data_api_trades import PolymarketDataApiTradesIndexer; "
    "PolymarketDataApiTradesIndexer(start={start}, end={end}).run()"
)


def test_markets_sigint_preserves_offset_cursor_and_resumes(
    workdir, admin, fx, run_indexer, wait_until, interrupt_and_wait
):
    offset_file = workdir / "data/polymarket/.backfill_offset"

    proc = run_indexer(MARKETS_CODE, workdir)
    wait_until(offset_file.exists, message="offset cursor file")
    output = interrupt_and_wait(proc)
    assert proc.returncode == 0
    assert "Interrupted" in output

    # The cursor and the flushed partial chunk both survived the interrupt.
    assert offset_file.exists()
    assert int(offset_file.read_text()) >= 500
    assert list((workdir / "data/polymarket/markets").glob("markets_*.parquet"))

    resume = run_indexer(MARKETS_CODE, workdir)
    resume_output, _ = resume.communicate(timeout=120)
    assert resume.returncode == 0
    assert "Resuming from offset" in resume_output
    assert not offset_file.exists()

    total, distinct = duckdb.sql(
        "SELECT COUNT(*), COUNT(DISTINCT id) FROM 'data/polymarket/markets/markets_*.parquet'"
    ).fetchone()
    assert total == fx.TOTAL_MARKETS
    assert distinct == fx.TOTAL_MARKETS


def test_trades_sigint_preserves_block_cursor_and_resumes(
    workdir, admin, fx, run_indexer, wait_until, interrupt_and_wait
):
    # Stretch each getLogs call so the interrupt deterministically lands
    # inside a fetch, never between a chunk's append and its cursor write.
    admin.set_latency("rpc:eth_getLogs", 0.1)
    end_block = fx.E2E_START_BLOCK + 199
    cursor_file = workdir / "data/polymarket/.backfill_block_cursor"

    proc = run_indexer(TRADES_CODE_FIRST.format(start=fx.E2E_START_BLOCK, end=end_block), workdir)
    wait_until(cursor_file.exists, message="block cursor file")
    output = interrupt_and_wait(proc)
    assert proc.returncode == 0
    assert "Interrupted" in output
    assert cursor_file.exists()

    resume = run_indexer(TRADES_CODE_RESUME.format(end=end_block), workdir)
    resume_output, _ = resume.communicate(timeout=120)
    assert resume.returncode == 0
    assert "Resuming from block" in resume_output
    assert not cursor_file.exists()

    expected = [p for p in fx.build_ctf_trade_params() if p[1] <= end_block]
    total, distinct = duckdb.sql(
        "SELECT COUNT(*), COUNT(DISTINCT (transaction_hash, log_index, _contract)) "
        "FROM 'data/polymarket/trades/trades_*.parquet'"
    ).fetchone()
    assert total == len(expected)
    assert distinct == len(expected)


def test_data_api_sigint_resumes_from_window_cursor(
    workdir, seeded_markets_dir, admin, fx, run_indexer, wait_until, interrupt_and_wait
):
    shutil.copytree(seeded_markets_dir / "data/polymarket/markets", workdir / "data/polymarket/markets")
    cursor_file = workdir / "data/polymarket/.data_api_trades_cursor"
    code = DATA_API_CODE.format(start=fx.DATA_START, end=fx.DATA_END)

    proc = run_indexer(code, workdir)
    # Interrupt inside a market's window walk: the cursor holds
    # "<market index>,<window timestamp>" once the first bisected leaf is done.
    wait_until(
        lambda: cursor_file.exists() and "," in cursor_file.read_text(),
        message="composite window cursor",
    )
    output = interrupt_and_wait(proc)
    assert proc.returncode == 0
    assert "Interrupted" in output
    assert "," in cursor_file.read_text()

    resume = run_indexer(code, workdir)
    resume_output, _ = resume.communicate(timeout=120)
    assert resume.returncode == 0
    assert "Resuming from market 0 at timestamp" in resume_output
    assert not cursor_file.exists()

    total, distinct = duckdb.sql(
        "SELECT COUNT(*), COUNT(DISTINCT (transaction_hash, proxy_wallet, asset, side, timestamp, size, price)) "
        "FROM 'data/polymarket/data_api_trades/trades_*.parquet'"
    ).fetchone()
    assert total == fx.DATA_TRADES_COUNT
    assert distinct == fx.DATA_TRADES_COUNT
