"""Small REST client for PMXT's hosted, read-only Router catalog."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from src.indexers.pmxt.models import PmxtQuery

DEFAULT_PMXT_API_URL = "https://api.pmxt.dev"
DEFAULT_MAX_RESPONSE_BYTES = 5_000_000
_MAX_RETAINED_HEADER_ITEMS = 32
_MAX_RETAINED_HEADER_VALUE_BYTES = 1_024
_MAX_CREDENTIAL_SCAN_NODES = 100_000
_RESPONSE_HEADER_ALLOWLIST = frozenset(
    {
        "cache-control",
        "content-encoding",
        "content-length",
        "content-type",
        "date",
        "ratelimit-limit",
        "ratelimit-remaining",
        "ratelimit-reset",
        "request-id",
        "retry-after",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "x-request-id",
    }
)
_RATE_LIMIT_CODES = {
    "rate_limit_exceeded": "PER_MINUTE",
    "monthly_quota_exceeded": "MONTHLY_QUOTA",
}


class PmxtRouterError(RuntimeError):
    """Raised when the hosted PMXT Router cannot serve a sync request."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "PMXT_ROUTER_ERROR",
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.evidence = dict(evidence or {})


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bounded_header_value(value: str) -> tuple[str, bool]:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= _MAX_RETAINED_HEADER_VALUE_BYTES:
        return value, False
    prefix = encoded[:_MAX_RETAINED_HEADER_VALUE_BYTES]
    return prefix.decode("utf-8", errors="replace"), True


def _uses_identity_content_encoding(response: httpx.Response) -> bool:
    values = [value.strip().lower() for value in response.headers.get_list("content-encoding")]
    return not any(value and value != "identity" for value in values)


class _JsonObjectPairs(list[tuple[str, Any]]):
    pass


def _credential_scan_status(body: bytes, credential: bytes) -> str:
    if not credential:
        return "ABSENT"
    if credential in body:
        return "PRESENT"
    try:
        credential_text = credential.decode("utf-8")
        decoded = body.decode("utf-8")
    except UnicodeDecodeError:
        return "UNKNOWN_CHARACTER_ENCODING"
    try:
        payload = json.loads(decoded, object_pairs_hook=_JsonObjectPairs)
    except RecursionError:
        return "UNKNOWN_PARSER_LIMIT"
    except ValueError:
        return "UNKNOWN_INVALID_JSON"

    pending = [payload]
    visited = 0
    while pending:
        visited += 1
        if visited > _MAX_CREDENTIAL_SCAN_NODES:
            return "UNKNOWN_SCAN_BOUND"
        value = pending.pop()
        if isinstance(value, str):
            if credential_text in value:
                return "PRESENT"
        elif isinstance(value, _JsonObjectPairs):
            for key, nested in value:
                pending.append(key)
                pending.append(nested)
        elif isinstance(value, list):
            pending.extend(value)
    return "ABSENT"


def _retry_after_evidence(
    values: list[str],
    *,
    credential_withheld: bool,
    value_truncated: bool,
    omitted_for_bound: bool,
) -> dict[str, Any]:
    if credential_withheld:
        return {"kind": "WITHHELD_CREDENTIAL_ECHO"}
    if omitted_for_bound:
        return {"kind": "INCOMPLETE", "reason": "header_item_bound"}
    if value_truncated:
        return {"kind": "INVALID", "reason": "value_truncated"}
    if not values:
        return {"kind": "ABSENT"}
    if len(values) != 1:
        return {"kind": "INVALID", "reason": "multiple_values"}

    raw = values[0].strip()
    if raw.isascii() and raw.isdigit():
        return {"kind": "DELTA_SECONDS", "raw": raw, "seconds": int(raw)}
    try:
        parsed = parsedate_to_datetime(raw)
        if parsed is not None and parsed.tzinfo is not None and parsed.utcoffset() is not None:
            date_utc = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            return {"kind": "HTTP_DATE", "raw": raw, "date_utc": date_utc}
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    return {"kind": "INVALID", "raw": raw}


def _rate_limit_evidence(
    body: bytes,
    *,
    body_complete: bool,
    credential: bytes,
    retry_after_values: list[str],
    retry_after_credential_withheld: bool,
    retry_after_value_truncated: bool,
    retry_after_omitted_for_bound: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "classification": "UNKNOWN_429",
        "retry_policy": "NO_RETRY_THIS_RUN",
        "requests_attempted": 1,
        "retries_performed": 0,
        "retry_after": _retry_after_evidence(
            retry_after_values,
            credential_withheld=retry_after_credential_withheld,
            value_truncated=retry_after_value_truncated,
            omitted_for_bound=retry_after_omitted_for_bound,
        ),
    }
    if not body_complete:
        result["body_parse_status"] = "NOT_PARSED_INCOMPLETE"
        return result

    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError:
        result["body_parse_status"] = "INVALID_UTF8"
        return result
    try:
        payload = json.loads(
            decoded,
            parse_constant=_reject_nonfinite_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except RecursionError:
        result["body_parse_status"] = "PARSER_LIMIT_EXCEEDED"
        return result
    except ValueError:
        result["body_parse_status"] = "INVALID_JSON"
        return result
    if not isinstance(payload, Mapping):
        result["body_parse_status"] = "VALID_JSON_NON_OBJECT"
        return result

    result["body_parse_status"] = "VALID_JSON_OBJECT"
    server_error_code = payload.get("error")
    if isinstance(server_error_code, str) and server_error_code in _RATE_LIMIT_CODES:
        result["server_error_code"] = server_error_code
        result["classification"] = _RATE_LIMIT_CODES[server_error_code]

    quota: dict[str, Any] = {}
    withheld_fields: list[str] = []
    plan = payload.get("plan")
    if isinstance(plan, str) and len(plan.encode("utf-8", errors="replace")) <= 64:
        if credential and credential in plan.encode("utf-8", errors="replace"):
            withheld_fields.append("plan")
        else:
            quota["plan"] = plan
    for name in ("limit", "used"):
        value = payload.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            quota[name] = value
    window = payload.get("window")
    if window == "1 minute":
        quota["window"] = window
    if quota:
        result["quota"] = quota
    if withheld_fields:
        result["credential_fields_withheld"] = withheld_fields
    return result


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
        self._credential_bytes = api_key.strip().encode("utf-8")
        self._client = httpx.Client(
            base_url=normalized_base_url,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
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
        self._credential_bytes = b""
        if "Authorization" in self._client.headers:
            del self._client.headers["Authorization"]
        self._client.cookies.clear()
        self._client.close()

    def _response_headers(self, response: httpx.Response) -> tuple[dict[str, Any], list[list[str]], dict[str, Any]]:
        grouped: dict[str, list[str]] = {}
        items: list[list[str]] = []
        omitted_for_credential = 0
        omitted_for_bound = 0
        truncated_values = 0
        retry_after_credential_withheld = False
        retry_after_value_truncated = False
        retry_after_omitted_for_bound = False

        for raw_name, raw_value in response.headers.multi_items():
            name = raw_name.lower()
            if name not in _RESPONSE_HEADER_ALLOWLIST:
                continue
            value_bytes = raw_value.encode("utf-8", errors="replace")
            if self._credential_bytes and self._credential_bytes in value_bytes:
                omitted_for_credential += 1
                if name == "retry-after":
                    retry_after_credential_withheld = True
                continue
            if len(items) >= _MAX_RETAINED_HEADER_ITEMS:
                omitted_for_bound += 1
                if name == "retry-after":
                    retry_after_omitted_for_bound = True
                continue
            value, was_truncated = _bounded_header_value(raw_value)
            truncated_values += int(was_truncated)
            if name == "retry-after" and was_truncated:
                retry_after_value_truncated = True
            items.append([name, value])
            grouped.setdefault(name, []).append(value)

        headers: dict[str, Any] = {
            name: values[0] if len(values) == 1 else values for name, values in grouped.items()
        }
        metadata = {
            "retention": "EXPLICIT_ALLOWLIST",
            "max_items": _MAX_RETAINED_HEADER_ITEMS,
            "max_value_bytes": _MAX_RETAINED_HEADER_VALUE_BYTES,
            "omitted_for_credential_echo": omitted_for_credential,
            "omitted_for_item_bound": omitted_for_bound,
            "truncated_values": truncated_values,
            "retry_after_credential_withheld": retry_after_credential_withheld,
            "retry_after_value_truncated": retry_after_value_truncated,
            "retry_after_omitted_for_bound": retry_after_omitted_for_bound,
        }
        return headers, items, metadata

    def _response_evidence(
        self,
        *,
        response: httpx.Response,
        query: PmxtQuery,
        body: bytes,
        body_complete: bool,
        observed_byte_size_lower_bound: int,
    ) -> dict[str, Any]:
        headers, header_items, header_metadata = self._response_headers(response)
        credential_scan_status = _credential_scan_status(body, self._credential_bytes)
        if not _uses_identity_content_encoding(response):
            body_evidence: dict[str, Any] = {
                "capture_status": "WITHHELD_UNEXPECTED_CONTENT_ENCODING",
                "retained_byte_size": len(body),
                "observed_byte_size_lower_bound": observed_byte_size_lower_bound,
                "complete": body_complete,
                "max_response_bytes": self._max_response_bytes,
                "representation": "HTTP_ENTITY_BYTES_ACCEPT_ENCODING_IDENTITY",
                "credential_scan_status": (
                    "PRESENT" if credential_scan_status == "PRESENT" else "UNKNOWN_UNEXPECTED_CONTENT_ENCODING"
                ),
            }
        elif not body_complete:
            body_evidence = {
                "capture_status": "WITHHELD_INCOMPLETE_RESPONSE",
                "retained_byte_size": len(body),
                "observed_byte_size_lower_bound": observed_byte_size_lower_bound,
                "complete": False,
                "max_response_bytes": self._max_response_bytes,
                "representation": "HTTP_ENTITY_BYTES_ACCEPT_ENCODING_IDENTITY",
                "credential_scan_status": (
                    "PRESENT" if credential_scan_status == "PRESENT" else "UNKNOWN_INCOMPLETE"
                ),
            }
        elif credential_scan_status == "PRESENT":
            body_evidence = {
                "capture_status": "WITHHELD_API_KEY_ECHO",
                "observed_byte_size": len(body),
                "complete": True,
                "max_response_bytes": self._max_response_bytes,
                "representation": "HTTP_ENTITY_BYTES_ACCEPT_ENCODING_IDENTITY",
                "credential_scan_status": "PRESENT",
            }
        elif credential_scan_status.startswith("UNKNOWN_"):
            body_evidence = {
                "capture_status": "WITHHELD_CREDENTIAL_SCAN_INDETERMINATE",
                "observed_byte_size": len(body),
                "complete": True,
                "max_response_bytes": self._max_response_bytes,
                "representation": "HTTP_ENTITY_BYTES_ACCEPT_ENCODING_IDENTITY",
                "credential_scan_status": credential_scan_status,
            }
        else:
            body_evidence = {
                "capture_status": "EXACT_COMPLETE",
                "raw_body_base64": base64.b64encode(body).decode("ascii"),
                "raw_body_sha256": hashlib.sha256(body).hexdigest(),
                "raw_body_byte_size": len(body),
                "complete": True,
                "max_response_bytes": self._max_response_bytes,
                "representation": "HTTP_ENTITY_BYTES_ACCEPT_ENCODING_IDENTITY",
                "credential_scan_status": credential_scan_status,
            }

        evidence: dict[str, Any] = {
            "schema_version": 1,
            "request": {
                "method": "GET",
                "path": "/v0/matched-market-clusters",
                "query": query.as_params(),
                "accept_encoding": "identity",
                "requests_attempted": 1,
                "retries_performed": 0,
                "retry_policy": "NO_RETRY_THIS_RUN",
            },
            "response": {
                "status_code": response.status_code,
                "response_headers": headers,
                "response_header_items": header_items,
                "response_header_capture": header_metadata,
                "body": body_evidence,
            },
        }
        if response.status_code == 429:
            retry_after_values = [value for name, value in header_items if name == "retry-after"]
            evidence["response"]["rate_limit"] = _rate_limit_evidence(
                body,
                body_complete=body_complete,
                credential=self._credential_bytes,
                retry_after_values=retry_after_values,
                retry_after_credential_withheld=header_metadata["retry_after_credential_withheld"],
                retry_after_value_truncated=header_metadata["retry_after_value_truncated"],
                retry_after_omitted_for_bound=header_metadata["retry_after_omitted_for_bound"],
            )
        return evidence

    def fetch_market_clusters(self, query: PmxtQuery) -> Any:
        """Fetch hosted identity candidates; no venue credentials or writes are used."""

        try:
            with self._client.stream("GET", "/v0/matched-market-clusters", params=query.as_params()) as response:
                body = bytearray()
                body_complete = True
                observed_byte_size_lower_bound = 0
                try:
                    for chunk in response.iter_raw():
                        observed_byte_size_lower_bound += len(chunk)
                        remaining = self._max_response_bytes - len(body)
                        if len(chunk) > remaining:
                            body.extend(chunk[:remaining])
                            body_complete = False
                            break
                        body.extend(chunk)
                except httpx.HTTPError as exc:
                    evidence = self._response_evidence(
                        response=response,
                        query=query,
                        body=bytes(body),
                        body_complete=False,
                        observed_byte_size_lower_bound=observed_byte_size_lower_bound,
                    )
                    raise PmxtRouterError(
                        "PMXT Router response stream ended before the response was complete",
                        reason_code="PMXT_RESPONSE_STREAM_INCOMPLETE",
                        evidence=evidence,
                    ) from exc

                evidence = self._response_evidence(
                    response=response,
                    query=query,
                    body=bytes(body),
                    body_complete=body_complete,
                    observed_byte_size_lower_bound=observed_byte_size_lower_bound,
                )
                if not body_complete:
                    raise PmxtRouterError(
                        f"PMXT Router response exceeded the {self._max_response_bytes}-byte safety bound",
                        reason_code="PMXT_RESPONSE_TOO_LARGE",
                        evidence=evidence,
                    )

                if not _uses_identity_content_encoding(response):
                    raise PmxtRouterError(
                        "PMXT Router returned an unexpected content encoding",
                        reason_code="PMXT_UNEXPECTED_CONTENT_ENCODING",
                        evidence=evidence,
                    )

                body_capture_status = evidence["response"]["body"]["capture_status"]
                if body_capture_status == "WITHHELD_API_KEY_ECHO":
                    raise PmxtRouterError(
                        "PMXT Router response contained credential material",
                        reason_code="PMXT_RESPONSE_CREDENTIAL_ECHO",
                        evidence=evidence,
                    )
                if body_capture_status == "WITHHELD_CREDENTIAL_SCAN_INDETERMINATE":
                    raise PmxtRouterError(
                        "PMXT Router response could not pass the credential safety scan",
                        reason_code="PMXT_RESPONSE_CREDENTIAL_SCAN_INDETERMINATE",
                        evidence=evidence,
                    )

                if not response.is_success:
                    if response.status_code == 401:
                        detail = "authentication failed"
                        reason_code = "PMXT_HTTP_AUTHENTICATION_FAILED"
                    elif response.status_code == 429:
                        detail = "rate limited"
                        reason_code = "PMXT_HTTP_RATE_LIMITED"
                    else:
                        detail = f"HTTP {response.status_code}"
                        reason_code = "PMXT_HTTP_ERROR"
                    raise PmxtRouterError(
                        f"PMXT Router request failed: {detail}",
                        reason_code=reason_code,
                        evidence=evidence,
                    )
        except httpx.HTTPError as exc:
            raise PmxtRouterError(
                "PMXT Router request failed before a response was received",
                reason_code="PMXT_TRANSPORT_ERROR",
            ) from exc
        finally:
            self._client.cookies.clear()

        try:
            return json.loads(body, parse_constant=_reject_nonfinite_json_constant)
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise PmxtRouterError(
                "PMXT Router returned invalid JSON",
                reason_code="PMXT_INVALID_JSON",
                evidence=evidence,
            ) from exc
