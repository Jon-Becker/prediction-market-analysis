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
| `volume` | float | Total volume in USD |
| `liquidity` | float | Current liquidity in USD |
| `active` | bool | Is market active |
| `closed` | bool | Is market closed |
| `end_date` | datetime (nullable) | When market ends |
| `created_at` | datetime (nullable) | When market was created |
| `_fetched_at` | datetime | When this record was fetched |

## Polymarket Trades (API)

Each row represents a single trade from the public data API.

| Column | Type | Description |
|--------|------|-------------|
| `condition_id` | string | Market condition ID |
| `asset` | string | Asset/token ID |
| `side` | string | Trade side: `BUY` or `SELL` |
| `size` | float | Number of shares traded |
| `price` | float | Price (0-1 decimal) |
| `timestamp` | int | Unix timestamp |
| `outcome` | string | Outcome name |
| `outcome_index` | int | Outcome index (0 or 1) |
| `transaction_hash` | string | Blockchain transaction hash |
| `_fetched_at` | datetime | When this record was fetched |

## Polymarket Trades (Blockchain)

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
