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
