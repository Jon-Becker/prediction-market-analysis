from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from dateutil.parser import isoparse


def parse_datetime(val: str) -> datetime:
    return isoparse(val)


@dataclass
class Trade:
    trade_id: str
    ticker: str
    count: int
    yes_price: int # Stored as cents
    no_price: int  # Stored as cents
    taker_side: str
    created_time: datetime

    @classmethod
    def from_dict(cls, data: dict) -> "Trade":
        def parse_dollars_to_cents(val: Optional[str]) -> int:
            """Converts string dollar values like '0.5600' to integer cents (56)."""
            if not val:
                return 0
            return int(float(val) * 100)
            
        def parse_float_string_to_int(val: Optional[str]) -> int:
            """Converts string float values like '10.00' to integers (10)."""
            if not val:
                return 0
            return int(float(val))

        return cls(
            trade_id=data["trade_id"],
            ticker=data["ticker"],
            # Map the exact keys from the JSON and convert to standard ints
            count=parse_float_string_to_int(data.get("count_fp")),
            yes_price=parse_dollars_to_cents(data.get("yes_price_dollars")),
            no_price=parse_dollars_to_cents(data.get("no_price_dollars")),
            taker_side=data.get("taker_side", ""),
            created_time=parse_datetime(data["created_time"]),
        )

@dataclass
class Market:
    ticker: str
    event_ticker: str
    market_type: str
    title: str
    yes_sub_title: str
    no_sub_title: str
    status: str
    yes_bid: Optional[int]    # Stored as cents
    yes_ask: Optional[int]    # Stored as cents
    no_bid: Optional[int]     # Stored as cents
    no_ask: Optional[int]     # Stored as cents
    last_price: Optional[int] # Stored as cents
    volume: int
    volume_24h: int
    open_interest: int
    result: str
    created_time: Optional[datetime]
    open_time: Optional[datetime]
    close_time: Optional[datetime]

    @classmethod
    def from_dict(cls, data: dict) -> "Market":
        def parse_time(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            return parse_datetime(val)

        def parse_dollars_to_cents(val: Optional[str]) -> Optional[int]:
            """Converts string dollar values like '0.5600' to integer cents (56)."""
            if not val:
                return None
            return int(float(val) * 100)
            
        def parse_float_string_to_int(val: Optional[str]) -> int:
            """Converts string float values like '10.00' to integers (10)."""
            if not val:
                return 0
            return int(float(val))

        return cls(
            ticker=data["ticker"],
            event_ticker=data["event_ticker"],
            market_type=data.get("market_type", "binary"),
            title=data.get("title", ""),
            yes_sub_title=data.get("yes_sub_title", ""),
            no_sub_title=data.get("no_sub_title", ""),
            status=data["status"],
            
            # Map to the exact keys in the JSON and convert to cents
            yes_bid=parse_dollars_to_cents(data.get("yes_bid_dollars")),
            yes_ask=parse_dollars_to_cents(data.get("yes_ask_dollars")),
            no_bid=parse_dollars_to_cents(data.get("no_bid_dollars")),
            no_ask=parse_dollars_to_cents(data.get("no_ask_dollars")),
            last_price=parse_dollars_to_cents(data.get("last_price_dollars")),
            
            # Map to the _fp (floating point) keys and convert to ints
            volume=parse_float_string_to_int(data.get("volume_fp")),
            volume_24h=parse_float_string_to_int(data.get("volume_24h_fp")),
            open_interest=parse_float_string_to_int(data.get("open_interest_fp")),
            
            result=data.get("result", ""),
            created_time=parse_time(data.get("created_time")),
            open_time=parse_time(data.get("open_time")),
            close_time=parse_time(data.get("close_time")),
        )