# Data Schemas

Data is stored as Parquet files in `data/{kalshi,polymarket}/`.

## Kalshi Markets

Each row represents a prediction market contract.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Unique market identifier (e.g., `PRES-2024-DJT`) |
| `event_ticker` | string | Parent event identifier, used for categorization |
| `market_type` | string | Market type (typically `binary`) |
| `title` | string | Human-readable market title |
| `yes_sub_title` | string | Label for the "Yes" outcome |
| `no_sub_title` | string | Label for the "No" outcome |
| `status` | string | Market status: `open`, `closed`, `finalized` |
| `yes_bid` | int (nullable) | Best bid price for Yes contracts (cents, 1-99) |
| `yes_ask` | int (nullable) | Best ask price for Yes contracts (cents, 1-99) |
| `no_bid` | int (nullable) | Best bid price for No contracts (cents, 1-99) |
| `no_ask` | int (nullable) | Best ask price for No contracts (cents, 1-99) |
| `last_price` | int (nullable) | Last traded price (cents, 1-99) |
| `volume` | int | Total contracts traded |
| `volume_24h` | int | Contracts traded in last 24 hours |
| `open_interest` | int | Outstanding contracts |
| `result` | string | Market outcome: `yes`, `no`, or empty if unresolved |
| `created_time` | datetime | When the market was created |
| `open_time` | datetime (nullable) | When trading opened |
| `close_time` | datetime (nullable) | When trading closed |
| `_fetched_at` | datetime | When this record was fetched |

## Kalshi Trades

Each row represents a single trade execution.

| Column | Type | Description |
|--------|------|-------------|
| `trade_id` | string | Unique trade identifier |
| `ticker` | string | Market ticker this trade belongs to |
| `count` | int | Number of contracts traded |
| `yes_price` | int | Yes contract price (cents, 1-99) |
| `no_price` | int | No contract price (cents, 1-99), always `100 - yes_price` |
| `taker_side` | string | Which side the taker bought: `yes` or `no` |
| `created_time` | datetime | When the trade occurred |
| `_fetched_at` | datetime | When this record was fetched |

**Note on Kalshi prices:** Prices are in cents. A `yes_price` of 65 means the contract costs $0.65 and pays $1.00 if the outcome is "Yes" (implied probability: 65%). The `no_price` is always `100 - yes_price`.

## Polymarket Markets

Each row represents a prediction market.

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Market ID |
| `condition_id` | string | Condition ID (hex hash) |
| `question` | string | Market question |
| `slug` | string | URL slug |
| `outcomes` | string | JSON string of outcome names |
| `outcome_prices` | string | JSON string of outcome prices |
| `clob_token_ids` | string | JSON string of CLOB token IDs, one per outcome |
| `volume` | float | Total volume in USD |
| `liquidity` | float | Current liquidity in USD |
| `active` | bool | Is market active |
| `closed` | bool | Is market closed |
| `end_date` | datetime (nullable) | When market ends |
| `created_at` | datetime (nullable) | When market was created |
| `market_maker_address` | string (nullable) | FPMM contract address for legacy markets |
| `_fetched_at` | datetime | When this record was fetched |

## Polymarket Events

Each row represents an event, the top-level question grouping one or more markets. Events are fetched from the Gamma API with keyset pagination in two passes (`closed=false`, then `closed=true`), since Gamma excludes closed events by default. An event that closes mid-backfill can appear in both passes; deduplicate on `id`, keeping the latest `_fetched_at`.

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Event ID |
| `slug` | string | URL slug |
| `title` | string | Event title |
| `category` | string (nullable) | Event category |
| `tags` | string | JSON string of tag slugs, used for categorization |
| `market_ids` | string | JSON string of child market IDs (joins to `markets.id`) |
| `volume` | float | Total volume in USD |
| `liquidity` | float | Current liquidity in USD |
| `active` | bool | Is event active |
| `closed` | bool | Is event closed |
| `start_date` | datetime (nullable) | When event starts |
| `end_date` | datetime (nullable) | When event ends |
| `created_at` | datetime (nullable) | When event was created |
| `_fetched_at` | datetime | When this record was fetched |

## Polymarket Price History

Each row represents one point in the price time series of a single outcome token, fetched from the CLOB `/prices-history` endpoint at hourly fidelity. Both outcome tokens of every market in the markets dataset are covered.

| Column | Type | Description |
|--------|------|-------------|
| `token_id` | string | CLOB token ID (joins to an entry of `clob_token_ids` in Polymarket Markets) |
| `timestamp` | int | Unix timestamp of the price point (seconds) |
| `price` | float | Token price (decimal between 0 and 1, see the note on Polymarket prices below) |
| `_fetched_at` | datetime | When this record was fetched |

## Polymarket Trades

Each row represents an `OrderFilled` event from the Polygon blockchain.

| Column | Type | Description |
|--------|------|-------------|
| `block_number` | int | Polygon block number |
| `transaction_hash` | string | Blockchain transaction hash |
| `log_index` | int | Log index within transaction |
| `order_hash` | string | Unique order identifier |
| `maker` | string | Address of limit order placer |
| `taker` | string | Address that filled the order |
| `maker_asset_id` | int | Asset ID maker provided (0=USDC) |
| `taker_asset_id` | int | Asset ID taker provided |
| `maker_amount` | int | Amount maker gave (6 decimals) |
| `taker_amount` | int | Amount taker gave (6 decimals) |
| `fee` | int | Trading fee (6 decimals) |
| `_fetched_at` | datetime | When this record was fetched |
| `_contract` | string | Contract name (CTF Exchange or NegRisk) |

**Note on Polymarket prices:** Prices are decimals between 0 and 1. A price of 0.65 means the contract costs $0.65 and pays $1.00 if the outcome wins (implied probability: 65%).

## Polymarket Legacy Trades (FPMM)

Each row represents an `FPMMBuy` or `FPMMSell` event from the legacy Fixed Product Market Maker contracts on Polygon. These are trades from before Polymarket migrated to the CTF Exchange (roughly 2020-2022).

| Column | Type | Description |
|--------|------|-------------|
| `block_number` | int | Polygon block number |
| `transaction_hash` | string | Blockchain transaction hash |
| `log_index` | int | Log index within transaction |
| `fpmm_address` | string | FPMM contract (market) address |
| `trader` | string | Buyer or seller address |
| `amount` | string | Investment amount (buy) or return amount (sell) in collateral units (6 decimals for USDC) |
| `fee_amount` | string | Trading fee in collateral units |
| `outcome_index` | int | Index of the outcome traded (0 or 1) |
| `outcome_tokens` | string | Outcome tokens bought or sold (18 decimals) |
| `is_buy` | bool | True for buy, False for sell |
| `timestamp` | int (nullable) | Unix timestamp (if enriched) |
| `_fetched_at` | datetime | When this record was fetched |

**Note on legacy trade amounts:** The `amount`, `fee_amount`, and `outcome_tokens` fields are stored as strings to avoid integer overflow. Collateral amounts use 6 decimals (for USDC markets), while outcome tokens use 18 decimals.

## Polymarket FPMM Collateral Lookup

Located at `data/polymarket/fpmm_collateral_lookup.json`, this file maps FPMM contract addresses to their collateral token information. Used to filter legacy trades to only include USDC-collateralized markets.

```json
{
  "0x1234...": {
    "collateral_address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    "collateral_symbol": "USDC"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `collateral_address` | string | ERC-20 token address used as collateral |
| `collateral_symbol` | string | Token symbol (e.g., `USDC`, `USDT`) |

## Polymarket Resolutions

Each row represents a `ConditionResolution` event from the Gnosis ConditionalTokens contract on Polygon, recording the exact on-chain payout for a resolved market.

| Column | Type | Description |
|--------|------|-------------|
| `block_number` | int | Polygon block number |
| `transaction_hash` | string | Blockchain transaction hash |
| `log_index` | int | Log index within transaction |
| `condition_id` | string | Condition ID (0x-prefixed hex), joins to `condition_id` in Polymarket Markets |
| `oracle` | string | Oracle address that reported the payouts |
| `question_id` | string | Question ID (0x-prefixed hex) |
| `outcome_slot_count` | int | Number of outcome slots for the condition |
| `payout_numerators` | string | JSON string of payout numerators, one per outcome slot |
| `_fetched_at` | datetime | When this record was fetched |

**Note on payout numerators:** `payout_numerators` is stored as a JSON string to avoid integer overflow. The winning outcome is the index of the sole nonzero numerator; conditions with multiple nonzero numerators resolved to split/invalid payouts.

## Polymarket Blocks

Mapping from Polygon block numbers to timestamps.

| Column | Type | Description |
|--------|------|-------------|
| `block_number` | int | Polygon block number |
| `timestamp` | string | ISO 8601 timestamp (e.g., `2024-01-15T12:30:00Z`) |

## Polymarket Orderbook (live recording)

Located in `data/polymarket/orderbook/{books,price_changes}/`.

**Polymarket exposes no historical order-book endpoint — depth data cannot be fetched retroactively.** These tables only cover periods when the live recorder (`uv run main.py record`) was running against the CLOB market WebSocket channel. Prices and sizes are stored as raw strings exactly as sent by the exchange so books can be replayed and verified against the `hash` field.

### Orderbook Book Snapshots (`books/`)

Each row is a full L2 `book` snapshot for one token. The server sends a snapshot on every (re)subscribe and after each trade that affects the book, so a reconnect always re-baselines the state — deltas missed while disconnected are superseded by the next snapshot.

| Column | Type | Description |
|--------|------|-------------|
| `token_id` | string | CLOB token ID (`asset_id` in the message) |
| `condition_id` | string | Condition ID (`market` in the message) |
| `timestamp` | int | Exchange timestamp in unix **milliseconds** |
| `hash` | string | Book integrity hash from the exchange |
| `bids` | string | Raw JSON array of `{price, size}` levels, as sent |
| `asks` | string | Raw JSON array of `{price, size}` levels, as sent |
| `_recorded_at` | datetime | When the message was received locally |

### Orderbook Price Changes (`price_changes/`)

Each row is one changed level from a `price_change` message (order placements/cancellations). To reconstruct the book at time T: take the latest snapshot at or before T and apply subsequent deltas in `timestamp` order.

| Column | Type | Description |
|--------|------|-------------|
| `token_id` | string | CLOB token ID |
| `condition_id` | string | Condition ID |
| `timestamp` | int | Exchange timestamp in unix **milliseconds** |
| `hash` | string | Book hash after this change |
| `side` | string | `BUY` (bid level) or `SELL` (ask level) |
| `price` | string | Price level that changed (raw string) |
| `size` | string | New total size at that level; `"0"` means the level was removed |
| `best_bid` | string | Best bid after this change |
| `best_ask` | string | Best ask after this change |
| `_recorded_at` | datetime | When the message was received locally |
