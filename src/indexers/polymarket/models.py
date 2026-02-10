from dataclasses import dataclass
from datetime import datetime
from typing import Optional


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
        def parse_time(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            try:
                # Handle ISO format with Z suffix
                val = val.replace("Z", "+00:00")
                return datetime.fromisoformat(val)
            except (ValueError, TypeError):
                return None

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
class Trade:
    condition_id: str
    asset: str  # Asset/token ID
    side: str  # BUY or SELL
    size: float  # Number of shares
    price: float  # Price (0-1)
    timestamp: int  # Unix timestamp
    outcome: str
    outcome_index: int  # 0 or 1
    transaction_hash: str

    @classmethod
    def from_dict(cls, data: dict) -> "Trade":
        return cls(
            condition_id=data.get("conditionId", data.get("market", "")),
            asset=data.get("asset", ""),
            side=data.get("side", ""),
            size=float(data.get("size", 0) or 0),
            price=float(data.get("price", 0) or 0),
            timestamp=int(data.get("timestamp", 0) or 0),
            outcome=data.get("outcome", ""),
            outcome_index=int(data.get("outcomeIndex", 0) or 0),
            transaction_hash=data.get("transactionHash", ""),
        )
