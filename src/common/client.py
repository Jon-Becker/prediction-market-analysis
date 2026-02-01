import logging

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _is_retryable_error(exc: BaseException) -> bool:
    """Check if an exception should trigger a retry."""
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def retry_request():
    """Decorator for HTTP requests with exponential backoff.

    Retries on:
    - Connection errors
    - Timeouts
    - HTTP 429 (rate limit)
    - HTTP 5xx (server errors)

    Uses exponential backoff starting at 1s, max 60s, up to 5 attempts.
    """
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception(_is_retryable_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
