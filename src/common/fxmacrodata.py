"""FXMacroData helpers for macro-event prediction market studies.

The helpers intentionally use the public FXMacroData REST API and the Python
standard library so analysis scripts can join scheduled macro releases to
Kalshi or Polymarket data without adding another dependency.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


def fetch_release_calendar(
    currency: str = "usd",
    *,
    limit: int = 100,
    base_url: str = FXMACRODATA_BASE_URL,
    api_key: Optional[str] = None,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    """Fetch scheduled macro releases from FXMacroData.

    Args:
        currency: ISO currency code, for example ``"usd"``.
        limit: Maximum number of events to request.
        base_url: FXMacroData API base URL.
        api_key: Optional API key. If omitted, ``FXMACRODATA_API_KEY`` is used
            when present.
        timeout: Request timeout in seconds.

    Returns:
        A list of release-calendar rows.
    """

    limit_count = max(1, min(int(limit), 100))
    params: dict[str, str] = {"limit": str(limit_count)}
    token = api_key or os.getenv("FXMACRODATA_API_KEY")
    if token:
        params["api_key"] = token

    url = f"{base_url.rstrip('/')}/calendar/{currency.lower()}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "prediction-market-analysis-fxmacrodata/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("FXMacroData calendar response did not contain a list in 'data'")
    return data[:limit_count]


def upcoming_market_events(
    currency: str = "usd",
    *,
    min_tier: int = 1,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return upcoming macro releases suitable for prediction-market joins."""

    now = datetime.now(timezone.utc)
    events = []
    for event in fetch_release_calendar(currency, limit=limit):
        timestamp = event.get("announcement_datetime")
        if timestamp is None:
            continue
        event_time = datetime.fromtimestamp(float(timestamp), timezone.utc)
        if event_time < now:
            continue
        if int(event.get("market_tier") or 99) > min_tier:
            continue
        events.append(event)
    return events
