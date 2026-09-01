"""Small REST client for PMXT's hosted, read-only Router catalog."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from src.indexers.pmxt.models import PmxtQuery

DEFAULT_PMXT_API_URL = "https://api.pmxt.dev"
DEFAULT_MAX_RESPONSE_BYTES = 5_000_000


class PmxtRouterError(RuntimeError):
    """Raised when the hosted PMXT Router cannot serve a sync request."""


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def extract_clusters(payload: Any, *, max_clusters: int | None = None) -> list[dict[str, Any]]:
    """Normalize the documented list/wrapper response forms into cluster records."""

    candidates: Any = payload
    if isinstance(payload, Mapping):
        if "clusterId" in payload or "cluster_id" in payload:
            candidates = [payload]
        else:
            for key in ("clusters", "results", "data"):
                if key in payload:
                    candidates = payload[key]
                    break

    if not isinstance(candidates, list):
        raise PmxtRouterError("PMXT Router returned an unexpected cluster response shape")
    if max_clusters is not None and len(candidates) > max_clusters:
        raise PmxtRouterError(
            f"PMXT Router returned {len(candidates)} clusters, exceeding the requested bound of {max_clusters}"
        )

    clusters: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            raise PmxtRouterError("PMXT Router returned a non-object cluster record")
        clusters.append(dict(item))
    return clusters


class PmxtRouterClient:
    """Authenticated read-only client; it never exposes an order-writing method."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_PMXT_API_URL,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("A PMXT API key is required for Router catalog access")
        normalized_base_url = base_url.rstrip("/")
        if transport is None and normalized_base_url != DEFAULT_PMXT_API_URL:
            raise ValueError("A custom PMXT base URL is permitted only with an injected test transport")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")

        self._max_response_bytes = max_response_bytes
        self._client = httpx.Client(
            base_url=normalized_base_url,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Accept": "application/json",
            },
            timeout=timeout,
            transport=transport,
            trust_env=False,
        )

    def __enter__(self) -> PmxtRouterClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def fetch_market_clusters(self, query: PmxtQuery) -> Any:
        """Fetch hosted identity candidates; no venue credentials or writes are used."""

        try:
            with self._client.stream("GET", "/v0/matched-market-clusters", params=query.as_params()) as response:
                if not response.is_success:
                    if response.status_code == 401:
                        detail = "authentication failed"
                    elif response.status_code == 429:
                        detail = "rate limited"
                    else:
                        detail = f"HTTP {response.status_code}"
                    raise PmxtRouterError(f"PMXT Router request failed: {detail}")

                body = bytearray()
                for chunk in response.iter_bytes():
                    if len(body) + len(chunk) > self._max_response_bytes:
                        raise PmxtRouterError(
                            f"PMXT Router response exceeded the {self._max_response_bytes}-byte safety bound"
                        )
                    body.extend(chunk)
        except httpx.HTTPError as exc:
            raise PmxtRouterError("PMXT Router request failed before a response was received") from exc

        try:
            return json.loads(body, parse_constant=_reject_nonfinite_json_constant)
        except (UnicodeDecodeError, ValueError) as exc:
            raise PmxtRouterError("PMXT Router returned invalid JSON") from exc
