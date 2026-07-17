"""Deterministic fixture corpus shared by the e2e mock service and the e2e tests.

This module is stdlib-only: it is imported both inside the mock container
(next to ``app.py``) and from the test runner (via ``importlib``) so the
tests can derive expected values from the exact data the mock serves.

All timestamps are fixed in the past so runs are reproducible.
"""

import json
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Time bases
# ---------------------------------------------------------------------------

MARKETS_BASE_TS = 1704067200  # 2024-01-01T00:00:00Z, a multiple of 3600
DAY = 86400
HOUR = 3600

# ---------------------------------------------------------------------------
# Gamma markets
# ---------------------------------------------------------------------------

TOTAL_MARKETS = 25000
# Scheduled market lifetime in days for the six "featured" markets (the only
# ones that carry CLOB token IDs and condition IDs). The price-history window
# for each is its lifetime + the indexer's 7-day end padding, so these spans
# force 1, 1, 2, 2, 4 and 4 fifteen-day request windows respectively.
FEATURED_SPANS_DAYS = [5, 5, 20, 20, 40, 40]
FEATURED_COUNT = len(FEATURED_SPANS_DAYS)
END_PADDING_DAYS = 7
PRICE_WINDOW_DAYS = 15


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def featured_condition_id(i):
    return "0x" + format(0xC0FFEE00 + i, "064x")


def featured_token_ids(i):
    base = 10**20
    return [str(base + i * 2 + 1), str(base + i * 2 + 2)]


def featured_created_ts(i):
    return MARKETS_BASE_TS + i * DAY


def featured_end_ts(i):
    return featured_created_ts(i) + FEATURED_SPANS_DAYS[i] * DAY


def all_featured_token_ids():
    return [token for i in range(FEATURED_COUNT) for token in featured_token_ids(i)]


def build_markets():
    markets = []
    for i in range(TOTAL_MARKETS):
        featured = i < FEATURED_COUNT
        if featured:
            created = featured_created_ts(i)
            market = {
                "id": str(i),
                "conditionId": featured_condition_id(i),
                "question": f"Featured question {i}?",
                "slug": f"featured-market-{i}",
                "outcomes": json.dumps(["Yes", "No"]),
                "outcomePrices": json.dumps(["0.4", "0.6"]),
                "clobTokenIds": json.dumps(featured_token_ids(i)),
                "volume": float(10**9 - i),
                "liquidity": 50000.0,
                "active": True,
                "closed": False,
                "createdAt": iso(created),
                "endDate": iso(featured_end_ts(i)),
                "marketMakerAddress": None,
            }
        else:
            market = {
                "id": str(i),
                "conditionId": "",
                "question": f"Question {i}?",
                "slug": f"market-{i}",
                "outcomes": json.dumps(["Yes", "No"]),
                "outcomePrices": json.dumps(["0.5", "0.5"]),
                "clobTokenIds": "[]",
                "volume": float(i % 1000),
                "liquidity": float(i % 500),
                "active": i % 3 != 0,
                "closed": i % 3 == 0,
                "createdAt": iso(MARKETS_BASE_TS),
                "endDate": iso(MARKETS_BASE_TS + 30 * DAY),
                "marketMakerAddress": None,
            }
        markets.append(market)
    return markets


# ---------------------------------------------------------------------------
# Gamma events (keyset pagination, two closed phases)
# ---------------------------------------------------------------------------

OPEN_EVENTS = 700
CLOSED_EVENTS = 700
TOTAL_EVENTS = OPEN_EVENTS + CLOSED_EVENTS


def build_events():
    events = []
    for i in range(TOTAL_EVENTS):
        closed = i >= OPEN_EVENTS
        events.append(
            {
                "id": str(i),
                "slug": f"event-{i}",
                "title": f"Event {i}",
                "category": f"Category {i % 4}",
                "tags": [{"slug": f"tag-{i % 5}"}],
                "markets": [{"id": str(i)}],
                "volume": float(i * 10),
                "liquidity": float(i * 5),
                "active": not closed,
                "closed": closed,
                "startDate": iso(MARKETS_BASE_TS + i * HOUR),
                "endDate": iso(MARKETS_BASE_TS + 30 * DAY),
                "createdAt": iso(MARKETS_BASE_TS),
            }
        )
    return events


# ---------------------------------------------------------------------------
# CLOB price history (hourly grid anchored at midnight UTC)
# ---------------------------------------------------------------------------


def price_at(token_id, ts):
    return ((int(token_id) * 13 + ts // HOUR) % 99 + 1) / 100


def price_points(start_ts, end_ts):
    """All hourly grid points in [start_ts, end_ts], inclusive on both ends.

    Both-end inclusivity means adjacent request windows share their boundary
    point, which is exactly what the real API does and what the indexer's
    seen-timestamp dedup exists for.
    """
    first = ((start_ts + HOUR - 1) // HOUR) * HOUR
    return list(range(first, end_ts + 1, HOUR))


def expected_price_rows(i):
    """Number of distinct price points for one token of featured market i."""
    span = (FEATURED_SPANS_DAYS[i] + END_PADDING_DAYS) * DAY
    return span // HOUR + 1


def expected_price_requests(i):
    """Number of <=15-day windows needed to cover featured market i."""
    span = (FEATURED_SPANS_DAYS[i] + END_PADDING_DAYS) * DAY
    return -(-span // (PRICE_WINDOW_DAYS * DAY))


# ---------------------------------------------------------------------------
# Data API trade tape
# ---------------------------------------------------------------------------

DATA_TRADES_COUNT = 12000  # sized to overflow the 10,000 offset cap once
DATA_START = MARKETS_BASE_TS
DATA_END = DATA_START + DATA_TRADES_COUNT - 1
MALFORMED_TRADE_TIMESTAMPS = [DATA_START + 100, DATA_START + 200]


def build_data_trades():
    """One trade per second across the six featured markets, plus two rows
    that fail model parsing (non-numeric size) to exercise skip-at-parse."""
    trades = []
    for j in range(DATA_TRADES_COUNT):
        market = j % FEATURED_COUNT
        trades.append(
            {
                "proxyWallet": "0x" + format(j % 97, "040x"),
                "side": "BUY" if j % 2 == 0 else "SELL",
                "asset": featured_token_ids(market)[j % 2],
                "conditionId": featured_condition_id(market),
                "size": round(1 + (j % 50) / 10, 1),
                "price": ((j % 99) + 1) / 100,
                "timestamp": DATA_START + j,
                "transactionHash": "0x" + format(j, "064x"),
                "title": f"Featured question {market}?",
                "slug": f"featured-market-{market}",
                "eventSlug": f"event-{market}",
                "outcome": "Yes" if j % 2 == 0 else "No",
                "outcomeIndex": j % 2,
                "name": f"trader-{j % 97}",
                "pseudonym": f"Pseudonym-{j % 97}",
            }
        )
    for ts in MALFORMED_TRADE_TIMESTAMPS:
        trades.append(
            {
                "proxyWallet": "0xmalformed",
                "side": "BUY",
                "asset": featured_token_ids(0)[0],
                "conditionId": featured_condition_id(0),
                "size": "not-a-number",
                "price": 0.5,
                "timestamp": ts,
                "transactionHash": "0xmalformed",
            }
        )
    return trades


MAX_DATA_API_OFFSET = 10000

# ---------------------------------------------------------------------------
# Polygon JSON-RPC: chain shape
# ---------------------------------------------------------------------------

HEAD_BLOCK = 1103100
E2E_START_BLOCK = 1100000  # POLYMARKET_START_BLOCK / CONDITIONAL_TOKENS_START_BLOCK in compose
GENESIS_TS = 1690000000
BLOCK_TIME_SECONDS = 2
# Topic-only getLogs filters (the FPMM indexer's) wider than this return the
# "block range too large" error, forcing the indexer's bisection path.
TOPIC_ONLY_MAX_SPAN = 600


def block_timestamp(n):
    return GENESIS_TS + n * BLOCK_TIME_SECONDS


CTF_EXCHANGE = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
NEGRISK_CTF_EXCHANGE = "0xc5d563a36ae78145c45a50134d48a1215220f80a"
CONDITIONAL_TOKENS = "0x4d97dcd97ec945f40cf65f87097ace5ea0476045"

ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
CONDITION_RESOLUTION_TOPIC = "0xb44d84d3289691f71497564b85d4233648d9dbae8cbdbb4329f301c3a0185894"
FPMM_BUY_TOPIC = "0x4f62630f51608fc8a7603a9391a5101e58bd7c276139366fc107dc3b67c3dcf8"
FPMM_SELL_TOPIC = "0xadcf2a240ed9300d681d9a3f5382b6c1beed1b7e46643e0c7b42cbe6e2d766b4"


def _word(value):
    return format(value, "064x")


def _addr_topic(address):
    return "0x" + "0" * 24 + address[2:].lower()


def _data(*words):
    return "0x" + "".join(_word(w) for w in words)


def _log(address, topics, data, block, log_index, tx_seq):
    return {
        "address": address,
        "topics": topics,
        "data": data,
        "blockNumber": block,
        "logIndex": log_index,
        "transactionIndex": 0,
        "transactionHash": "0x" + format(0xABC0000000 + tx_seq, "064x"),
        "blockHash": "0x" + format(0xB10C000000 + block, "064x"),
        "removed": False,
    }


MAKER_ADDR = "0x" + "11" * 20
TAKER_ADDR = "0x" + "22" * 20
ORACLE_ADDR = "0x" + "33" * 20


def _order_filled_log(
    address, block, log_index, tx_seq, maker_asset_id, taker_asset_id, maker_amount, taker_amount, fee
):
    return _log(
        address,
        [
            ORDER_FILLED_TOPIC,
            "0x" + format(0x0DE40000 + tx_seq, "064x"),  # orderHash
            _addr_topic(MAKER_ADDR),
            _addr_topic(TAKER_ADDR),
        ],
        _data(maker_asset_id, taker_asset_id, maker_amount, taker_amount, fee),
        block,
        log_index,
        tx_seq,
    )


def build_ctf_trade_params():
    """(contract, block, log_index, maker_asset_id, taker_asset_id, maker_amount, taker_amount, fee).

    CTF Exchange fills live only in blocks [E2E_START_BLOCK, E2E_START_BLOCK+23]
    (before the first cursor write of any chunked run) and NegRisk fills avoid
    blocks that are chunk-boundary re-fetch candidates for a chunk size of 25
    ((block - E2E_START_BLOCK) % 25 == 24). Together these make process-kill
    resume deterministic and duplicate-free by construction.
    """
    params = []
    for k in range(30):
        block = E2E_START_BLOCK + (k % 24)
        params.append((CTF_EXCHANGE, block, k // 24, 0, 10**20 + k, (k + 1) * 10**6, 2 * 10**6, 1000 + k))
    for k in range(60):
        block = E2E_START_BLOCK + 25 + k * 51
        if (block - E2E_START_BLOCK) % 25 == 24:
            block += 1
        params.append((NEGRISK_CTF_EXCHANGE, block, 0, 10**20 + k, 0, 3 * 10**6, (k + 1) * 10**6, 2000 + k))
    return params


def build_fpmm_addresses():
    return ["0x" + "aa" * 19 + "01", "0x" + "aa" * 19 + "02", "0x" + "aa" * 19 + "03"]


def build_fpmm_trade_params():
    """(fpmm_address, block, log_index, is_buy, amount, fee_amount, outcome_index, outcome_tokens)."""
    addresses = build_fpmm_addresses()
    params = []
    for k in range(40):
        block = E2E_START_BLOCK + 10 + k * 77
        params.append(
            (
                addresses[k % 3],
                block,
                0,
                k % 2 == 0,
                (k + 1) * 10**6,
                10**4 + k,
                k % 2,
                (k + 1) * 10**18,
            )
        )
    return params


RESOLUTION_PAYOUTS = [[1, 0], [0, 1], [1, 1], [0, 3, 1]]


def build_resolution_params():
    """(block, log_index, condition_id, question_id, payout_numerators)."""
    params = []
    for k in range(12):
        block = E2E_START_BLOCK + 50 + k * 250
        condition_id = featured_condition_id(k) if k < FEATURED_COUNT else "0x" + format(0xD00D00 + k, "064x")
        question_id = "0x" + format(0x9E570000 + k, "064x")
        params.append((block, k % 3, condition_id, question_id, RESOLUTION_PAYOUTS[k % len(RESOLUTION_PAYOUTS)]))
    return params


def build_logs():
    """The full canned log store served by eth_getLogs."""
    logs = []
    tx_seq = 0
    for (
        contract,
        block,
        log_index,
        maker_asset,
        taker_asset,
        maker_amount,
        taker_amount,
        fee,
    ) in build_ctf_trade_params():
        tx_seq += 1
        logs.append(
            _order_filled_log(
                contract, block, log_index, tx_seq, maker_asset, taker_asset, maker_amount, taker_amount, fee
            )
        )
    # One undecodable OrderFilled log (truncated data) that the indexer must skip.
    tx_seq += 1
    logs.append(
        _log(
            CTF_EXCHANGE,
            [ORDER_FILLED_TOPIC, "0x" + _word(0xBAD), _addr_topic(MAKER_ADDR), _addr_topic(TAKER_ADDR)],
            "0x1234",
            E2E_START_BLOCK + 3,
            9,
            tx_seq,
        )
    )
    for fpmm, block, log_index, is_buy, amount, fee_amount, outcome_index, outcome_tokens in build_fpmm_trade_params():
        tx_seq += 1
        logs.append(
            _log(
                fpmm,
                [
                    FPMM_BUY_TOPIC if is_buy else FPMM_SELL_TOPIC,
                    _addr_topic(TAKER_ADDR),
                    "0x" + _word(outcome_index),
                ],
                _data(amount, fee_amount, outcome_tokens),
                block,
                log_index,
                tx_seq,
            )
        )
    for block, log_index, condition_id, question_id, payouts in build_resolution_params():
        tx_seq += 1
        head = [len(payouts), 0x40, len(payouts)] + payouts
        # outcomeSlotCount word, offset word to the dynamic array, then its length + items
        logs.append(
            _log(
                CONDITIONAL_TOKENS,
                [CONDITION_RESOLUTION_TOPIC, condition_id, _addr_topic(ORACLE_ADDR), question_id],
                _data(*head),
                block,
                log_index,
                tx_seq,
            )
        )
    return logs


# ---------------------------------------------------------------------------
# eth_call fixtures (FPMM collateral lookup)
# ---------------------------------------------------------------------------

TOKEN_USDC = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"
TOKEN_NO_SYMBOL = "0x" + "bb" * 19 + "01"  # symbol() reverts -> UNKNOWN fallback

SELECTOR_COLLATERAL_TOKEN = "0xb2016bd4"
SELECTOR_SYMBOL = "0x95d89b41"


def build_collateral_map():
    a, b, c = build_fpmm_addresses()
    return {a: TOKEN_USDC, b: TOKEN_USDC, c: TOKEN_NO_SYMBOL}


def encode_address(address):
    return "0x" + "0" * 24 + address[2:].lower()


def encode_string(value):
    raw = value.encode()
    padded_len = -(-len(raw) // 32) * 32
    return "0x" + _word(0x20) + _word(len(raw)) + raw.hex().ljust(padded_len * 2, "0")


# ---------------------------------------------------------------------------
# CLOB WebSocket market channel script
# ---------------------------------------------------------------------------

WS_BOOK_TS_MS = "1704067200000"
WS_BOOK_BIDS = [{"price": "0.45", "size": "100.5"}, {"price": "0.44", "size": "200"}]
WS_BOOK_ASKS = [{"price": "0.55", "size": "50"}, {"price": "0.56", "size": "75"}]


def ws_book_message(token_id):
    for i in range(FEATURED_COUNT):
        if token_id in featured_token_ids(i):
            condition_id = featured_condition_id(i)
            break
    else:
        condition_id = "0x" + "0" * 64
    return {
        "event_type": "book",
        "asset_id": token_id,
        "market": condition_id,
        "timestamp": WS_BOOK_TS_MS,
        "hash": f"bookhash-{token_id[-4:]}",
        "bids": WS_BOOK_BIDS,
        "asks": WS_BOOK_ASKS,
    }


def ws_price_change_conn1(token_id):
    return {
        "event_type": "price_change",
        "market": featured_condition_id(0),
        "timestamp": "1704067201000",
        "price_changes": [
            {
                "asset_id": token_id,
                "price": "0.46",
                "size": "25",
                "side": "BUY",
                "hash": "pc1",
                "best_bid": "0.46",
                "best_ask": "0.55",
            },
            {
                "asset_id": token_id,
                "price": "0.55",
                "size": "10",
                "side": "SELL",
                "hash": "pc2",
                "best_bid": "0.46",
                "best_ask": "0.55",
            },
        ],
    }


def ws_price_change_conn2(token_id):
    return {
        "event_type": "price_change",
        "market": featured_condition_id(0),
        "timestamp": "1704067202000",
        "price_changes": [
            {
                "asset_id": token_id,
                "price": "0.44",
                "size": "0",
                "side": "BUY",
                "hash": "pc3",
                "best_bid": "0.45",
                "best_ask": "0.55",
            },
            {
                "asset_id": token_id,
                "price": "0.47",
                "size": "40",
                "side": "BUY",
                "hash": "pc4",
                "best_bid": "0.47",
                "best_ask": "0.55",
            },
        ],
    }


def ws_legacy_change_conn2(token_id):
    return {
        "event_type": "price_change",
        "market": featured_condition_id(0),
        "asset_id": token_id,
        "hash": "legacy1",
        "timestamp": "1704067203000",
        "changes": [
            {"price": "0.56", "size": "0", "side": "SELL", "best_bid": "0.47", "best_ask": "0.57"},
        ],
    }


# Rows persisted to the price_changes table per connection, per the script above.
WS_CONN1_DELTA_ROWS = 2
WS_CONN2_DELTA_ROWS = 3
