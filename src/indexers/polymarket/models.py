import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


def parse_time(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    try:
        # Handle ISO format with Z suffix
        val = val.replace("Z", "+00:00")
        return datetime.fromisoformat(val)
    except (ValueError, TypeError):
        return None


@dataclass
class Market:
    id: str
    condition_id: str
    question: str
    slug: str
    outcomes: str  # JSON string of outcomes
    outcome_prices: str  # JSON string of prices
    clob_token_ids: str  # JSON string of token IDs for each outcome
    volume: float
    liquidity: float
    active: bool
    closed: bool
    end_date: Optional[datetime]
    created_at: Optional[datetime]
    market_maker_address: Optional[str] = None  # FPMM address for legacy markets

    @classmethod
    def from_dict(cls, data: dict) -> "Market":
        return cls(
            id=data.get("id", ""),
            condition_id=data.get("conditionId", ""),
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            outcomes=str(data.get("outcomes", "[]")),
            outcome_prices=str(data.get("outcomePrices", "[]")),
            clob_token_ids=str(data.get("clobTokenIds", "[]")),
            volume=float(data.get("volume", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            active=data.get("active", False),
            closed=data.get("closed", False),
            end_date=parse_time(data.get("endDate")),
            created_at=parse_time(data.get("createdAt")),
            market_maker_address=data.get("marketMakerAddress"),
        )


@dataclass
class Event:
    id: str
    slug: str
    title: str
    category: Optional[str]
    tags: str  # JSON string of tag slugs
    market_ids: str  # JSON string of child market IDs
    volume: float
    liquidity: float
    active: bool
    closed: bool
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    created_at: Optional[datetime]

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        return cls(
            id=data.get("id", ""),
            slug=data.get("slug", ""),
            title=data.get("title", ""),
            category=data.get("category"),
            tags=json.dumps([t.get("slug", "") for t in data.get("tags") or []]),
            market_ids=json.dumps([m.get("id", "") for m in data.get("markets") or []]),
            volume=float(data.get("volume", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            active=data.get("active", False),
            closed=data.get("closed", False),
            start_date=parse_time(data.get("startDate")),
            end_date=parse_time(data.get("endDate")),
            created_at=parse_time(data.get("createdAt")),
        )


@dataclass
class PricePoint:
    token_id: str
    timestamp: int  # unix seconds
    price: float

    @classmethod
    def from_dict(cls, token_id: str, data: dict) -> "PricePoint":
        return cls(
            token_id=token_id,
            timestamp=int(data.get("t", 0) or 0),
            price=float(data.get("p", 0) or 0),
        )


@dataclass
class DataApiTrade:
    proxy_wallet: str
    side: str  # taker side: BUY or SELL
    asset: str  # CLOB token ID (decimal string, kept as string to avoid overflow)
    condition_id: str
    size: float  # outcome shares traded
    price: float  # execution price, 0-1 decimal
    timestamp: int  # unix seconds
    transaction_hash: str
    title: str = ""
    slug: str = ""
    event_slug: str = ""
    outcome: str = ""
    outcome_index: int = -1
    name: str = ""
    pseudonym: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "DataApiTrade":
        return cls(
            proxy_wallet=data.get("proxyWallet", ""),
            side=data.get("side", ""),
            asset=str(data.get("asset", "")),
            condition_id=data.get("conditionId", ""),
            size=float(data.get("size", 0) or 0),
            price=float(data.get("price", 0) or 0),
            timestamp=int(data.get("timestamp", 0) or 0),
            transaction_hash=data.get("transactionHash", ""),
            title=data.get("title", ""),
            slug=data.get("slug", ""),
            event_slug=data.get("eventSlug", ""),
            outcome=data.get("outcome", ""),
            outcome_index=int(data.get("outcomeIndex", -1) if data.get("outcomeIndex") is not None else -1),
            name=data.get("name", ""),
            pseudonym=data.get("pseudonym", ""),
        )


@dataclass
class OrderBookLevel:
    price: float
    size: float

    @classmethod
    def from_dict(cls, data: dict) -> "OrderBookLevel":
        return cls(
            price=float(data.get("price", 0) or 0),
            size=float(data.get("size", 0) or 0),
        )


@dataclass
class OrderBookDelta:
    condition_id: str
    token_id: str
    timestamp: int  # unix milliseconds
    hash: str
    side: str  # BUY (bid level) or SELL (ask level)
    price: str  # raw string, preserved for exact replay
    size: str  # raw string; "0" means the level was removed
    best_bid: str
    best_ask: str

    @classmethod
    def list_from_message(cls, data: dict) -> list["OrderBookDelta"]:
        """Parse a market-channel `price_change` message into one delta per changed level.

        Handles both the current shape (`price_changes` with per-change `asset_id`
        and `hash`) and the legacy shape (`changes` with top-level `asset_id`).
        """
        condition_id = data.get("market", "")
        timestamp = int(data.get("timestamp", 0) or 0)
        changes = data.get("price_changes") or data.get("changes") or []
        return [
            cls(
                condition_id=condition_id,
                token_id=str(change.get("asset_id") or data.get("asset_id") or ""),
                timestamp=timestamp,
                hash=change.get("hash") or data.get("hash") or "",
                side=change.get("side", ""),
                price=str(change.get("price", "")),
                size=str(change.get("size", "")),
                best_bid=str(change.get("best_bid", "")),
                best_ask=str(change.get("best_ask", "")),
            )
            for change in changes
        ]


@dataclass
class OrderBookSnapshot:
    condition_id: str  # `market` in the API response
    token_id: str  # `asset_id` in the API response
    timestamp: int  # unix milliseconds
    hash: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    min_order_size: float = 0.0
    tick_size: float = 0.0
    neg_risk: bool = False
    last_trade_price: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "OrderBookSnapshot":
        return cls(
            condition_id=data.get("market", ""),
            token_id=data.get("asset_id", ""),
            timestamp=int(data.get("timestamp", 0) or 0),
            hash=data.get("hash", ""),
            bids=[OrderBookLevel.from_dict(level) for level in data.get("bids") or []],
            asks=[OrderBookLevel.from_dict(level) for level in data.get("asks") or []],
            min_order_size=float(data.get("min_order_size", 0) or 0),
            tick_size=float(data.get("tick_size", 0) or 0),
            neg_risk=data.get("neg_risk", False),
            last_trade_price=float(data.get("last_trade_price", 0) or 0),
        )
