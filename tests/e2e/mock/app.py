"""Deterministic mock of every external surface the Polymarket stack talks to.

One FastAPI app serves five surfaces on path prefixes:

    /gamma/*     Gamma REST (offset + keyset pagination, closed filtering)
    /clob/*      CLOB REST (prices-history with the 15-day cap, book, prices)
    /data-api/*  Data API REST (market/start/end scoping, 10,000 offset cap)
    /rpc         Polygon JSON-RPC (block/log filtering, ABI-encoded logs,
                 "too large" errors, eth_call selector dispatch)
    /ws/market   CLOB market WebSocket (subscribe, book, price_change,
                 PING/PONG, scripted disconnect/reconnect)

plus /admin/* endpoints for request inspection and fault injection. All data
comes from the deterministic corpus in ``fixtures.py``.
"""

import asyncio
import json

import fixtures as fx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

app = FastAPI()

MARKETS = fx.build_markets()
EVENTS = fx.build_events()
DATA_TRADES = sorted(fx.build_data_trades(), key=lambda t: -t["timestamp"])
LOGS = fx.build_logs()
COLLATERAL = fx.build_collateral_map()
FEATURED_TOKENS = set(fx.all_featured_token_ids())

KEYSET_LIMIT_CAPS = {"markets": 100, "events": 500}
MAX_EVENT_LOG = 20000


def _initial_state():
    return {
        "counts": {},
        "events": [],
        "fail_next": {},
        "latency": {},
        "ws": {"connections": 0, "subscriptions": [], "pings": 0, "invalid_subscribes": 0},
    }


STATE = _initial_state()


async def pre(key, **meta):
    """Latency, request accounting, and one-shot fault injection for one call."""
    await asyncio.sleep(STATE["latency"].get(key, 0))
    STATE["counts"][key] = STATE["counts"].get(key, 0) + 1
    if len(STATE["events"]) < MAX_EVENT_LOG:
        STATE["events"].append({"key": key, "meta": meta})
    entry = STATE["fail_next"].get(key)
    if entry and entry["times"] > 0:
        entry["times"] -= 1
        raise HTTPException(status_code=entry["status"], detail="injected failure")


def _qbool(params, name):
    value = params.get(name)
    return None if value is None else value.lower() == "true"


def _filter_flags(rows, params):
    for flag in ("active", "closed"):
        wanted = _qbool(params, flag)
        if wanted is not None:
            rows = [r for r in rows if r[flag] == wanted]
    return rows


# ---------------------------------------------------------------------------
# Gamma REST
# ---------------------------------------------------------------------------


async def _gamma_offset(kind, rows, request):
    params = request.query_params
    offset = int(params.get("offset", 0))
    limit = int(params.get("limit", 100))
    await pre("/gamma/" + kind, offset=offset, limit=limit)
    rows = _filter_flags(rows, params)
    if params.get("order") == "volume":
        ascending = _qbool(params, "ascending")
        rows = sorted(rows, key=lambda r: r["volume"], reverse=not ascending)
    return rows[offset : offset + limit]


@app.get("/gamma/markets")
async def gamma_markets(request: Request):
    return await _gamma_offset("markets", MARKETS, request)


@app.get("/gamma/events")
async def gamma_events(request: Request):
    return await _gamma_offset("events", EVENTS, request)


async def _gamma_keyset(kind, rows, request):
    params = request.query_params
    cursor = int(params.get("after_cursor") or 0)
    limit = min(int(params.get("limit", 100)), KEYSET_LIMIT_CAPS[kind])
    await pre(f"/gamma/{kind}/keyset", cursor=cursor, limit=limit, closed=params.get("closed"))
    rows = _filter_flags(rows, params)
    page = rows[cursor : cursor + limit]
    next_cursor = str(cursor + limit) if cursor + limit < len(rows) else None
    return {kind: page, "next_cursor": next_cursor}


@app.get("/gamma/markets/keyset")
async def gamma_markets_keyset(request: Request):
    return await _gamma_keyset("markets", MARKETS, request)


@app.get("/gamma/events/keyset")
async def gamma_events_keyset(request: Request):
    return await _gamma_keyset("events", EVENTS, request)


# ---------------------------------------------------------------------------
# CLOB REST
# ---------------------------------------------------------------------------


@app.get("/clob/prices-history")
async def clob_prices_history(request: Request):
    params = request.query_params
    token_id = params.get("market", "")
    start_ts = params.get("startTs")
    end_ts = params.get("endTs")
    span = int(end_ts) - int(start_ts) if start_ts and end_ts else None
    await pre("/clob/prices-history", market=token_id, span=span)
    if start_ts is None or end_ts is None:
        raise HTTPException(status_code=400, detail="startTs and endTs are required by this mock")
    if span > fx.PRICE_WINDOW_DAYS * fx.DAY:
        # Undocumented production behavior: ranges longer than 15 days 400.
        raise HTTPException(status_code=400, detail="invalid request: startTs/endTs range too long")
    if token_id not in FEATURED_TOKENS:
        return {"history": []}
    points = fx.price_points(int(start_ts), int(end_ts))
    return {"history": [{"t": ts, "p": fx.price_at(token_id, ts)} for ts in points]}


@app.get("/clob/book")
async def clob_book(request: Request):
    token_id = request.query_params.get("token_id", "")
    await pre("/clob/book", token_id=token_id)
    book = fx.ws_book_message(token_id)
    return {
        "market": book["market"],
        "asset_id": token_id,
        "timestamp": fx.WS_BOOK_TS_MS,
        "hash": book["hash"],
        "bids": fx.WS_BOOK_BIDS,
        "asks": fx.WS_BOOK_ASKS,
        "min_order_size": "5",
        "tick_size": "0.01",
        "neg_risk": False,
        "last_trade_price": "0.44",
    }


@app.get("/clob/midpoint")
async def clob_midpoint(request: Request):
    await pre("/clob/midpoint")
    return {"mid": "0.5"}


@app.get("/clob/spread")
async def clob_spread(request: Request):
    await pre("/clob/spread")
    return {"spread": "0.1"}


@app.get("/clob/price")
async def clob_price(request: Request):
    side = request.query_params.get("side", "")
    await pre("/clob/price", side=side)
    return {"price": "0.45" if side == "BUY" else "0.55"}


# ---------------------------------------------------------------------------
# Data API REST
# ---------------------------------------------------------------------------


@app.get("/data-api/trades")
async def data_api_trades(request: Request):
    params = request.query_params
    start = int(params.get("start", 0) or 0)
    end = int(params.get("end", 2**62) or 2**62)
    limit = int(params.get("limit", 100))
    offset = int(params.get("offset", 0))
    market = params.get("market")
    await pre("/data-api/trades", market=market, start=start, end=end, limit=limit, offset=offset)
    rows = DATA_TRADES
    if market:
        # start/end are only honored on market-scoped queries, as in production.
        wanted = set(market.split(","))
        rows = [t for t in rows if t["conditionId"] in wanted and start <= t["timestamp"] <= end]
    offset = min(offset, fx.MAX_DATA_API_OFFSET)
    return rows[offset : offset + limit]


# ---------------------------------------------------------------------------
# Polygon JSON-RPC
# ---------------------------------------------------------------------------

ZERO_HASH = "0x" + "00" * 32


def _rpc_block(number):
    return {
        "number": hex(number),
        "hash": "0x" + format(0xB10C000000 + number, "064x"),
        "parentHash": "0x" + format(0xB10C000000 + number - 1, "064x"),
        "nonce": "0x0000000000000000",
        "sha3Uncles": ZERO_HASH,
        "logsBloom": "0x" + "00" * 256,
        "transactionsRoot": ZERO_HASH,
        "stateRoot": ZERO_HASH,
        "receiptsRoot": ZERO_HASH,
        "miner": "0x" + "00" * 20,
        "mixHash": ZERO_HASH,
        "difficulty": "0x1",
        "totalDifficulty": "0x1",
        # 97-byte POA extraData: requires ExtraDataToPOAMiddleware to parse.
        "extraData": "0x" + "22" * 97,
        "size": "0x220",
        "gasLimit": "0x1c9c380",
        "gasUsed": "0x0",
        "timestamp": hex(fx.block_timestamp(number)),
        "baseFeePerGas": "0x8",
        "transactions": [],
        "uncles": [],
    }


def _to_block_number(value):
    if value in (None, "latest", "pending", "safe", "finalized"):
        return fx.HEAD_BLOCK
    if isinstance(value, str):
        return int(value, 16) if value.startswith("0x") else int(value)
    return int(value)


def _topic_matches(log_topics, wanted):
    for i, entry in enumerate(wanted or []):
        if entry is None:
            continue
        options = entry if isinstance(entry, list) else [entry]
        if i >= len(log_topics) or log_topics[i].lower() not in {o.lower() for o in options}:
            return False
    return True


def _format_log(log):
    return {
        "address": log["address"],
        "topics": log["topics"],
        "data": log["data"],
        "blockNumber": hex(log["blockNumber"]),
        "logIndex": hex(log["logIndex"]),
        "transactionIndex": hex(log["transactionIndex"]),
        "transactionHash": log["transactionHash"],
        "blockHash": log["blockHash"],
        "removed": False,
    }


def _eth_get_logs(flt):
    from_block = _to_block_number(flt.get("fromBlock", 0))
    to_block = _to_block_number(flt.get("toBlock", "latest"))
    address = flt.get("address")
    if address is not None and not isinstance(address, list):
        address = [address]
    addresses = {a.lower() for a in address} if address else None
    if addresses is None and to_block - from_block + 1 > fx.TOPIC_ONLY_MAX_SPAN:
        return None, {"code": -32005, "message": "query returned more than 10000 results, block range too large"}
    matched = [
        _format_log(log)
        for log in LOGS
        if from_block <= log["blockNumber"] <= to_block
        and (addresses is None or log["address"].lower() in addresses)
        and _topic_matches(log["topics"], flt.get("topics"))
    ]
    return matched, None


def _eth_call(call):
    to = (call.get("to") or "").lower()
    data = call.get("data") or call.get("input") or ""
    selector = data[:10].lower()
    if selector == fx.SELECTOR_COLLATERAL_TOKEN and to in COLLATERAL:
        return fx.encode_address(COLLATERAL[to]), None
    if selector == fx.SELECTOR_SYMBOL and to == fx.TOKEN_USDC:
        return fx.encode_string("USDC"), None
    return None, {"code": 3, "message": "execution reverted"}


@app.post("/rpc")
async def rpc(request: Request):
    body = await request.json()
    method = body.get("method", "")
    params = body.get("params") or []
    key = "rpc:" + method
    error = None
    if method == "eth_chainId":
        await pre(key)
        result = "0x89"
    elif method == "net_version":
        await pre(key)
        result = "137"
    elif method == "eth_blockNumber":
        await pre(key)
        result = hex(fx.HEAD_BLOCK)
    elif method == "eth_getBlockByNumber":
        number = _to_block_number(params[0] if params else "latest")
        await pre(key, number=number)
        result = _rpc_block(number)
    elif method == "eth_getLogs":
        flt = params[0] if params else {}
        from_block = _to_block_number(flt.get("fromBlock", 0))
        to_block = _to_block_number(flt.get("toBlock", "latest"))
        topics = flt.get("topics") or []
        await pre(
            key,
            from_block=from_block,
            to_block=to_block,
            span=to_block - from_block + 1,
            address=(flt.get("address") or "").lower() if isinstance(flt.get("address"), str) else flt.get("address"),
            topic0=(topics[0] if topics and isinstance(topics[0], str) else None),
        )
        result, error = _eth_get_logs(flt)
    elif method == "eth_call":
        call = params[0] if params else {}
        await pre(key, to=(call.get("to") or "").lower(), selector=(call.get("data") or "")[:10].lower())
        result, error = _eth_call(call)
    else:
        await pre(key)
        result, error = None, {"code": -32601, "message": f"method {method} not supported by mock"}
    response = {"jsonrpc": "2.0", "id": body.get("id", 1)}
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result
    return JSONResponse(response)


# ---------------------------------------------------------------------------
# CLOB market WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws/market")
async def ws_market(ws: WebSocket):
    await ws.accept()
    try:
        raw = await ws.receive_text()
        payload = json.loads(raw)
        assert isinstance(payload.get("assets_ids"), list) and payload.get("type") == "market"
    except WebSocketDisconnect:
        return
    except (AssertionError, TypeError, ValueError):
        STATE["ws"]["invalid_subscribes"] += 1
        await ws.close()
        return
    STATE["ws"]["connections"] += 1
    connection = STATE["ws"]["connections"]
    STATE["ws"]["subscriptions"].append(payload)
    tokens = [str(t) for t in payload["assets_ids"]]
    # The server re-sends a full book snapshot per token on every subscribe.
    await ws.send_text(json.dumps([fx.ws_book_message(t) for t in tokens]))
    if connection == 1:
        # Scripted mid-stream disconnect: the recorder must reconnect and
        # re-subscribe to keep recording.
        await ws.send_text(json.dumps(fx.ws_price_change_conn1(tokens[0])))
        await ws.close()
        return
    await ws.send_text(json.dumps(fx.ws_price_change_conn2(tokens[0])))
    await ws.send_text(json.dumps(fx.ws_legacy_change_conn2(tokens[0])))
    while True:
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=0.1)
        except asyncio.TimeoutError:
            try:
                # No-op heartbeat so the recorder's message loop stays
                # responsive to stop() without arbitrary sleeps in tests.
                await ws.send_text("{}")
            except Exception:
                return
            continue
        except WebSocketDisconnect:
            return
        if raw == "PING":
            STATE["ws"]["pings"] += 1
            await ws.send_text("PONG")


# ---------------------------------------------------------------------------
# Admin / health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/admin/reset")
async def admin_reset():
    global STATE
    STATE = _initial_state()
    return {"status": "reset"}


@app.get("/admin/requests")
async def admin_requests():
    return {"counts": STATE["counts"], "events": STATE["events"]}


@app.get("/admin/ws")
async def admin_ws():
    return STATE["ws"]


@app.post("/admin/fail-next")
async def admin_fail_next(request: Request):
    body = await request.json()
    STATE["fail_next"][body["key"]] = {"status": int(body.get("status", 500)), "times": int(body.get("times", 1))}
    return {"status": "armed"}


@app.post("/admin/latency")
async def admin_latency(request: Request):
    body = await request.json()
    STATE["latency"][body["key"]] = float(body["seconds"])
    return {"status": "set"}
