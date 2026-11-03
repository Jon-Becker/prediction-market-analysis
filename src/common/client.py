import logging
import threading
import time
from typing import Any, Optional, Union

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


class HttpClient:
    """HTTP client with rate limiting, retries, and connection pooling.

    Args:
        rate_limit: Max requests per second (0 = unlimited).
        max_retries: Number of retry attempts for transient errors.
        timeout: Request timeout in seconds.
        max_connections: Connection pool size.
        base_url: Optional base URL for all requests.
    """

    def __init__(
        self,
        *,
        rate_limit: float = 10,
        max_retries: int = 5,
        timeout: float = 30.0,
        max_connections: int = 20,
        base_url: str = "",
    ):
        self._rate_limit = rate_limit
        self._max_retries = max_retries
        self._lock = threading.Lock()
        self._last_request: float = 0

        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._client.close()

    def _throttle(self):
        if self._rate_limit <= 0:
            return
        interval = 1.0 / self._rate_limit
        with self._lock:
            now = time.monotonic()
            wait = self._last_request + interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def get(self, url: str, *, params: Optional[dict] = None) -> Union[dict, list]:
        return self._request("GET", url, params=params)

    def post(self, url: str, *, json: Optional[Any] = None) -> Union[dict, list]:
        return self._request("POST", url, json=json)

    def _request(self, method: str, url: str, **kwargs) -> Union[dict, list]:
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _do():
            self._throttle()
            response = self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()

        return _do()
