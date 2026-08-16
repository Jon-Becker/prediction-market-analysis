"""Read-only access to the FXMacroData release calendar."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FXMACRODATA_BASE_URL = "https://api.fxmacrodata.com/v1"


class FXMacroDataError(RuntimeError):
    """Raised when the release calendar cannot be requested or decoded."""


def parse_release_calendar(payload: Any) -> list[dict[str, Any]]:
    """Validate and sort an FXMacroData release-calendar response."""
    if not isinstance(payload, dict):
        raise FXMacroDataError("FXMacroData calendar response must be an object")

    data = payload.get("data")
    if not isinstance(data, list):
        raise FXMacroDataError("FXMacroData calendar response must contain a data list")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise FXMacroDataError(f"FXMacroData calendar row {index} must be an object")
        timestamp = item.get("announcement_datetime")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise FXMacroDataError(f"FXMacroData calendar row {index} must contain a Unix announcement_datetime")
        rows.append(dict(item))

    return sorted(rows, key=lambda row: float(row["announcement_datetime"]))


def load_release_calendar(path: Path | str) -> list[dict[str, Any]]:
    """Load a saved release-calendar response for reproducible analysis."""
    try:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FXMacroDataError("Unable to read the saved FXMacroData calendar") from exc
    return parse_release_calendar(payload)


def fetch_release_calendar(
    currency: str = "USD",
    *,
    limit: int = 100,
    base_url: str = FXMACRODATA_BASE_URL,
    api_key: str | None = None,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    """Fetch scheduled macro releases from the documented v1 endpoint."""
    normalized_currency = currency.strip().upper()
    if len(normalized_currency) not in (3, 4) or not normalized_currency.isalpha():
        raise ValueError("currency must be a three- or four-letter code")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    params = {"limit": str(limit)}
    token = api_key or os.getenv("FXMACRODATA_API_KEY") or os.getenv("FXMD_API_KEY")
    if token:
        params["api_key"] = token

    url = f"{base_url.rstrip('/')}/calendar/{normalized_currency.lower()}?{urlencode(params)}"
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "prediction-market-analysis/1.0"},
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise FXMacroDataError(f"FXMacroData calendar request failed with HTTP {exc.code}") from None
    except (URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError):
        raise FXMacroDataError("FXMacroData calendar request failed") from None

    return parse_release_calendar(payload)[:limit]
