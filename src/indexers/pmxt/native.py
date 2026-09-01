"""GET-only venue-native evidence capture for PMXT mapping proposals.

PMXT catalog identifiers are deliberately never used as venue identifiers here.
The client reverse-resolves explicit ticker/slug/condition hints, retains the
complete public venue responses, and exposes no trading methods.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import httpx

from src.indexers.pmxt.candidates import utc_now_iso
from src.indexers.pmxt.semantic import normalized_material_edge_cases

DEFAULT_KALSHI_API_URL = "https://external-api.kalshi.com/trade-api/v2"
DEFAULT_POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
DEFAULT_POLYMARKET_CLOB_URL = "https://clob.polymarket.com"
DEFAULT_MAX_NATIVE_RESPONSE_BYTES = 5_000_000
DEFAULT_KALSHI_FEE_SCHEDULE_URL = "https://kalshi.com/docs/kalshi-fee-schedule.pdf"
DEFAULT_MAX_FEE_SCHEDULE_BYTES = 5_000_000
REVIEWED_KALSHI_FEE_SCHEDULE_BINDINGS: dict[str, dict[str, Any]] = {}
_SUPPORTED_KALSHI_TAKER_FEE_TYPES = frozenset({"quadratic", "quadratic_with_maker_fees"})
_EVIDENCE_RESPONSE_HEADER_ALLOWLIST = frozenset(
    {
        "age",
        "cache-control",
        "content-encoding",
        "content-length",
        "content-type",
        "date",
        "etag",
        "expires",
        "last-modified",
        "request-id",
        "x-kalshi-request-id",
        "x-request-id",
    }
)

_UUID_RE = re.compile(r"[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}")
_KALSHI_MARKET_TICKER_RE = re.compile(r"[A-Z0-9]+(?:-[A-Z0-9]+){2,}")
_KALSHI_MARKET_URL_RE = re.compile(
    r"/markets/([^/?#]+)(?:[/?#]|$)",
    flags=re.IGNORECASE,
)


class NativeEvidenceError(RuntimeError):
    """A fail-closed native resolution or evidence-capture error."""

    def __init__(self, reason_code: str, message: str, *, evidence: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.evidence = dict(evidence or {})


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-safe evidence value with deterministic encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _iso_from_timestamp(value: Any, *, preserve_offset: bool = False) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000
        try:
            return datetime.fromtimestamp(number, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except (OSError, OverflowError, ValueError):
            return None
    text = _string(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    if preserve_offset:
        return parsed.isoformat().replace("+00:00", "Z")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _earliest(values: Sequence[str]) -> str:
    return min(values, key=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")))


def _latest(values: Sequence[str]) -> str:
    return max(values, key=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")))


def _candidate_side(candidate: Mapping[str, Any], side: str) -> dict[str, Any]:
    if side not in {"a", "b"}:
        raise ValueError("side must be 'a' or 'b'")
    return {
        "venue": _string(candidate.get(f"venue_{side}")),
        "pmxt_market_id": _string(candidate.get(f"pmxt_market_id_{side}")),
        "title": _string(candidate.get(f"title_{side}")),
        "slug": _string(candidate.get(f"slug_{side}")),
        "url": _string(candidate.get(f"url_{side}")),
        "event_id": _string(candidate.get(f"event_id_{side}")),
        "contract_address": _string(candidate.get(f"contract_address_{side}")),
        "source_metadata": _mapping(candidate.get(f"source_metadata_{side}")),
        "outcomes": _list_value(candidate.get(f"outcomes_{side}")),
    }


def _collect_named_hints(value: Any, keys: set[str]) -> set[str]:
    hints: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.lower() in keys:
                hint = _string(item)
                if hint:
                    hints.add(hint)
            if isinstance(item, (Mapping, list, tuple)):
                hints.update(_collect_named_hints(item, keys))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            hints.update(_collect_named_hints(item, keys))
    return hints


def _kalshi_ticker_hint(side: Mapping[str, Any]) -> str:
    keys = {
        "ticker",
        "market_ticker",
        "marketticker",
        "native_market_id",
        "nativemarketid",
        "venue_market_id",
        "venuemarketid",
        "original_id",
        "originalid",
    }
    hints = _collect_named_hints(side.get("source_metadata"), keys)
    hints.update(_collect_named_hints(side.get("outcomes"), keys))
    slug = _string(side.get("slug"))
    if slug:
        hints.add(slug)

    url = _string(side.get("url"))
    if url:
        match = _KALSHI_MARKET_URL_RE.search(url)
        if match:
            hints.add(match.group(1))

    pmxt_market_id = (_string(side.get("pmxt_market_id")) or "").upper()
    normalized: set[str] = set()
    for hint in hints:
        candidate = hint.upper()
        aliases_pmxt_id = candidate == pmxt_market_id or candidate.startswith(("PMXT_", "PMXT-"))
        is_uuid = _UUID_RE.fullmatch(candidate) is not None
        is_market_ticker = len(candidate) <= 200 and _KALSHI_MARKET_TICKER_RE.fullmatch(candidate) is not None
        if aliases_pmxt_id or is_uuid or not is_market_ticker:
            raise NativeEvidenceError(
                "NATIVE_ID_UNRESOLVED",
                "Kalshi venue hint is not an unambiguous native market ticker",
            )
        normalized.add(candidate)

    if not normalized:
        raise NativeEvidenceError(
            "NATIVE_ID_UNRESOLVED",
            "Kalshi ticker could not be reverse-resolved from PMXT venue hints",
        )
    if len(normalized) != 1:
        raise NativeEvidenceError(
            "NATIVE_ID_AMBIGUOUS",
            "Kalshi venue hints resolve to more than one ticker",
            evidence={"hint_count": len(normalized)},
        )
    return next(iter(normalized))


def _polymarket_lookup(side: Mapping[str, Any]) -> tuple[dict[str, str], str, str]:
    slug = _string(side.get("slug"))
    condition_id = _string(side.get("contract_address"))
    pmxt_market_id = _string(side.get("pmxt_market_id"))
    if slug:
        if slug == pmxt_market_id or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) is None:
            raise NativeEvidenceError(
                "NATIVE_ID_UNRESOLVED",
                "Polymarket slug hint is invalid or aliases the PMXT catalog ID",
            )
        return {"slug": slug}, "slug", slug
    if condition_id:
        if condition_id == pmxt_market_id or re.fullmatch(r"0x[0-9A-Fa-f]{64}", condition_id) is None:
            raise NativeEvidenceError(
                "NATIVE_ID_UNRESOLVED",
                "Polymarket condition hint is invalid or aliases the PMXT catalog ID",
            )
        return {"condition_ids": condition_id}, "condition_id", condition_id
    raise NativeEvidenceError(
        "NATIVE_ID_UNRESOLVED",
        "Polymarket market could not be reverse-resolved without a slug or condition hint",
    )


def _request_window(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize a bounded sequence without promoting receipt time to source time."""

    if not requests:
        return {
            "request_wall_utc": None,
            "request_monotonic_ns": None,
            "response_wall_utc": None,
            "response_monotonic_ns": None,
            "rtt_ms": None,
        }
    request_wall = [str(item["request_wall_utc"]) for item in requests if item.get("request_wall_utc")]
    response_wall = [str(item["response_wall_utc"]) for item in requests if item.get("response_wall_utc")]
    request_mono = [item.get("request_monotonic_ns") for item in requests]
    response_mono = [item.get("response_monotonic_ns") for item in requests]
    if (
        len(request_wall) != len(requests)
        or len(response_wall) != len(requests)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in request_mono + response_mono)
    ):
        return {
            "request_wall_utc": None,
            "request_monotonic_ns": None,
            "response_wall_utc": None,
            "response_monotonic_ns": None,
            "rtt_ms": None,
        }
    start_ns = min(request_mono)
    end_ns = max(response_mono)
    return {
        "request_wall_utc": _earliest(request_wall),
        "request_monotonic_ns": start_ns,
        "response_wall_utc": _latest(response_wall),
        "response_monotonic_ns": end_ns,
        "rtt_ms": (end_ns - start_ns) / 1_000_000,
    }


def _fee_change_timestamp(record: Mapping[str, Any]) -> str | None:
    return _iso_from_timestamp(record.get("scheduled_ts"))


def _same_fee_value(left: Any, right: Any) -> bool:
    left_number = _float(left)
    right_number = _float(right)
    return left_number is not None and right_number is not None and left_number == right_number


def _latest_matching_change(
    changes: Sequence[Mapping[str, Any]],
    *,
    observed_at: str | None,
    fee_type_key: str,
    fee_multiplier_key: str,
    fee_type: str | None,
    fee_multiplier: Any,
) -> str | None:
    observed = _iso_from_timestamp(observed_at)
    if observed is None:
        return None
    matches: list[str] = []
    for change in changes:
        scheduled = _fee_change_timestamp(change)
        if scheduled is None or scheduled > observed:
            continue
        change_type = _string(change.get(fee_type_key))
        change_multiplier = change.get(fee_multiplier_key)
        if fee_type is None and fee_multiplier is None:
            if change_type is None and change_multiplier is None:
                matches.append(scheduled)
        elif change_type == fee_type and _same_fee_value(change_multiplier, fee_multiplier):
            matches.append(scheduled)
    return _latest(matches) if matches else None


def _validated_fee_formula_binding(binding: Any) -> dict[str, Any] | None:
    if not isinstance(binding, Mapping):
        return None
    result = dict(binding)
    supported_types = _list_value(result.get("supported_taker_fee_types"))
    coefficient = _float(result.get("taker_base_coefficient"))
    trade_quantum = _float(result.get("trade_fee_rounding_quantum"))
    balance_precision = _float(result.get("balance_precision_upper_bound"))
    effective_date = _string(result.get("document_effective_date"))
    try:
        parsed_effective_date = datetime.strptime(effective_date or "", "%Y-%m-%d")
    except ValueError:
        parsed_effective_date = None
    if (
        _string(result.get("binding_id")) is None
        or effective_date is None
        or parsed_effective_date is None
        or coefficient is None
        or coefficient <= 0
        or trade_quantum is None
        or trade_quantum <= 0
        or balance_precision is None
        or balance_precision <= trade_quantum
        or not supported_types
        or any(value not in _SUPPORTED_KALSHI_TAKER_FEE_TYPES for value in supported_types)
    ):
        return None
    result["supported_taker_fee_types"] = sorted({str(value) for value in supported_types})
    result["taker_base_coefficient"] = str(result["taker_base_coefficient"])
    result["trade_fee_rounding_quantum"] = str(result["trade_fee_rounding_quantum"])
    result["balance_precision_upper_bound"] = str(result["balance_precision_upper_bound"])
    return result


class NativeEvidenceClient:
    """Public GET-only metadata and depth client for Kalshi and Polymarket."""

    def __init__(
        self,
        *,
        kalshi_base_url: str = DEFAULT_KALSHI_API_URL,
        gamma_base_url: str = DEFAULT_POLYMARKET_GAMMA_URL,
        clob_base_url: str = DEFAULT_POLYMARKET_CLOB_URL,
        timeout: float = 20.0,
        kalshi_transport: httpx.BaseTransport | None = None,
        gamma_transport: httpx.BaseTransport | None = None,
        clob_transport: httpx.BaseTransport | None = None,
        kalshi_fee_schedule_url: str = DEFAULT_KALSHI_FEE_SCHEDULE_URL,
        kalshi_fee_schedule_transport: httpx.BaseTransport | None = None,
        kalshi_fee_schedule_bindings: Mapping[str, Mapping[str, Any]] | None = None,
        max_response_bytes: int = DEFAULT_MAX_NATIVE_RESPONSE_BYTES,
        max_fee_schedule_bytes: int = DEFAULT_MAX_FEE_SCHEDULE_BYTES,
    ) -> None:
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        if (
            isinstance(max_fee_schedule_bytes, bool)
            or not isinstance(max_fee_schedule_bytes, int)
            or max_fee_schedule_bytes <= 0
        ):
            raise ValueError("max_fee_schedule_bytes must be a positive integer")
        fee_schedule_url = httpx.URL(kalshi_fee_schedule_url)
        if (
            fee_schedule_url.scheme != "https"
            or fee_schedule_url.host != "kalshi.com"
            or fee_schedule_url.path != "/docs/kalshi-fee-schedule.pdf"
            or fee_schedule_url.query
            or fee_schedule_url.fragment
        ):
            raise ValueError("kalshi_fee_schedule_url must be the canonical Kalshi HTTPS PDF URL")
        self._max_response_bytes = max_response_bytes
        self._max_fee_schedule_bytes = max_fee_schedule_bytes
        self._kalshi_fee_schedule_url = str(fee_schedule_url)
        raw_bindings = (
            REVIEWED_KALSHI_FEE_SCHEDULE_BINDINGS
            if kalshi_fee_schedule_bindings is None
            else kalshi_fee_schedule_bindings
        )
        self._kalshi_fee_schedule_bindings = {
            str(digest).lower(): dict(binding) for digest, binding in raw_bindings.items()
        }
        self._kalshi_fee_schedule_attempted = False
        self._kalshi_fee_schedule_capture: dict[str, Any] | None = None
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "prediction-market-analysis/pmxt-monitor",
        }
        self._kalshi = httpx.Client(
            base_url=kalshi_base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=kalshi_transport,
            trust_env=False,
        )
        self._gamma = httpx.Client(
            base_url=gamma_base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=gamma_transport,
            trust_env=False,
        )
        self._clob = httpx.Client(
            base_url=clob_base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=clob_transport,
            trust_env=False,
        )
        self._fee_schedule = httpx.Client(
            base_url="https://kalshi.com",
            headers={
                "Accept": "application/pdf",
                "Accept-Encoding": "identity",
                "User-Agent": "prediction-market-analysis/pmxt-monitor",
            },
            timeout=timeout,
            follow_redirects=False,
            transport=kalshi_fee_schedule_transport,
            trust_env=False,
        )

    def __enter__(self) -> NativeEvidenceClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._kalshi.close()
        self._gamma.close()
        self._clob.close()
        self._fee_schedule.close()

    def _get_response_bytes(
        self,
        client: httpx.Client,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        max_response_bytes: int | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        response_limit = self._max_response_bytes if max_response_bytes is None else max_response_bytes
        request_wall_utc = utc_now_iso()
        request_monotonic_ns = time.monotonic_ns()
        status_code: int | None = None
        response_headers: httpx.Headers | None = None
        response_url: str | None = None
        body = bytearray()

        def acquisition_evidence(*, body_complete: bool) -> dict[str, Any]:
            response_wall_utc = utc_now_iso()
            response_monotonic_ns = time.monotonic_ns()
            request_id = None
            request_id_header = None
            if response_headers is not None:
                for header_name in ("x-request-id", "request-id", "x-kalshi-request-id"):
                    header_value = _string(response_headers.get(header_name))
                    if header_value:
                        request_id = header_value
                        request_id_header = header_name
                        break
            retained_header_items = (
                [
                    [key.lower(), value]
                    for key, value in response_headers.multi_items()
                    if key.lower() in _EVIDENCE_RESPONSE_HEADER_ALLOWLIST
                ]
                if response_headers is not None
                else None
            )
            retained_headers = dict(retained_header_items) if retained_header_items is not None else None
            return {
                "method": "GET",
                "path": path,
                "final_url": response_url,
                "params": dict(params or {}),
                "request_wall_utc": request_wall_utc,
                "request_monotonic_ns": request_monotonic_ns,
                "response_wall_utc": response_wall_utc,
                "response_monotonic_ns": response_monotonic_ns,
                "rtt_ms": (response_monotonic_ns - request_monotonic_ns) / 1_000_000,
                "http_date": _string(response_headers.get("date")) if response_headers is not None else None,
                "age_header": _string(response_headers.get("age")) if response_headers is not None else None,
                "cache_control": (
                    _string(response_headers.get("cache-control")) if response_headers is not None else None
                ),
                "request_id": request_id,
                "request_id_header": request_id_header,
                "raw_body_hash": hashlib.sha256(bytes(body)).hexdigest() if response_headers is not None else None,
                "raw_body_base64": (
                    base64.b64encode(bytes(body)).decode("ascii") if response_headers is not None else None
                ),
                "raw_body_representation": "HTTP_ENTITY_BYTES_ACCEPT_ENCODING_IDENTITY",
                "request_accept_encoding": "identity",
                "response_headers": retained_headers,
                "response_header_items": retained_header_items,
                "response_header_retention": "EXPLICIT_ALLOWLIST",
                "body_complete": body_complete,
                "freshness_basis": "LOCAL_RECEIPT_BOUNDED",
                # Compatibility aliases; neither is a venue source timestamp.
                "requested_at": request_wall_utc,
                "received_at": response_wall_utc,
                "status_code": status_code,
            }

        try:
            with client.stream("GET", path, params=params) as response:
                status_code = response.status_code
                response_headers = response.headers
                response_url = str(response.url)
                for chunk in response.iter_raw():
                    if len(body) + len(chunk) > response_limit:
                        request_evidence = acquisition_evidence(body_complete=False)
                        raise NativeEvidenceError(
                            "NATIVE_RESPONSE_TOO_LARGE",
                            f"Native GET {path} exceeded the response safety bound",
                            evidence={
                                **request_evidence,
                                "max_response_bytes": response_limit,
                            },
                        )
                    body.extend(chunk)
        except httpx.HTTPError as exc:
            request_evidence = acquisition_evidence(body_complete=False)
            raise NativeEvidenceError(
                "NATIVE_REQUEST_ERROR",
                f"Native GET {path} failed before a response was received",
                evidence=request_evidence,
            ) from exc
        finally:
            # Public endpoints can attempt to create ambient client state via
            # Set-Cookie. Cookies are neither evidence nor permitted inputs to
            # a later native request, including after a failed response.
            client.cookies.clear()
        request_evidence = acquisition_evidence(body_complete=True)
        content_encoding = _string(request_evidence.get("response_headers", {}).get("content-encoding")) or "identity"
        if content_encoding.lower() != "identity":
            raise NativeEvidenceError(
                "NATIVE_RESPONSE_CONTENT_ENCODING_UNEXPECTED",
                f"Native GET {path} ignored the identity content-encoding requirement",
                evidence=request_evidence,
            )
        return bytes(body), request_evidence

    def _get_json(
        self,
        client: httpx.Client,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        body, request_evidence = self._get_response_bytes(client, path, params=params)
        try:
            payload = json.loads(body, parse_constant=_reject_nonfinite_json_constant)
        except (UnicodeDecodeError, ValueError) as exc:
            raise NativeEvidenceError(
                "NATIVE_INVALID_JSON",
                f"Native GET {path} returned invalid JSON",
                evidence=request_evidence,
            ) from exc
        status_code = request_evidence.get("status_code")
        if isinstance(status_code, bool) or not isinstance(status_code, int) or not 200 <= status_code < 300:
            raise NativeEvidenceError(
                "NATIVE_HTTP_ERROR",
                f"Native GET {path} returned HTTP {status_code}",
                evidence={**request_evidence, "response": payload},
            )
        return payload, request_evidence

    def _capture_kalshi_official_fee_schedule(self) -> dict[str, Any]:
        """Capture the canonical PDF once and bind only a manually reviewed hash."""

        if self._kalshi_fee_schedule_attempted:
            assert self._kalshi_fee_schedule_capture is not None
            return copy.deepcopy(self._kalshi_fee_schedule_capture)
        self._kalshi_fee_schedule_attempted = True
        try:
            body, request = self._get_response_bytes(
                self._fee_schedule,
                "/docs/kalshi-fee-schedule.pdf",
                max_response_bytes=self._max_fee_schedule_bytes,
            )
        except NativeEvidenceError as exc:
            self._kalshi_fee_schedule_capture = {
                "schema_version": 1,
                "source": "kalshi_official_fee_schedule",
                "source_url": self._kalshi_fee_schedule_url,
                "status": "FAIL_CLOSED",
                "reason_codes": [exc.reason_code],
                "error": str(exc),
                "request": exc.evidence,
                "raw_body_sha256": exc.evidence.get("raw_body_hash"),
                "byte_size": None,
                "formula_binding": None,
                "live_eligible": False,
            }
            return copy.deepcopy(self._kalshi_fee_schedule_capture)

        reasons: list[str] = []
        status_code = request.get("status_code")
        if status_code != 200:
            reasons.append("KALSHI_OFFICIAL_FEE_SCHEDULE_HTTP_STATUS_INVALID")
        final_url = _string(request.get("final_url"))
        if final_url is None or httpx.URL(final_url) != httpx.URL(self._kalshi_fee_schedule_url):
            reasons.append("KALSHI_OFFICIAL_FEE_SCHEDULE_FINAL_URL_INVALID")
        response_headers = _mapping(request.get("response_headers"))
        content_type = (_string(response_headers.get("content-type")) or "").split(";", 1)[0].lower()
        if content_type != "application/pdf":
            reasons.append("KALSHI_OFFICIAL_FEE_SCHEDULE_CONTENT_TYPE_INVALID")
        if not body.startswith(b"%PDF-"):
            reasons.append("KALSHI_OFFICIAL_FEE_SCHEDULE_MAGIC_INVALID")
        digest = hashlib.sha256(body).hexdigest()
        binding = _validated_fee_formula_binding(self._kalshi_fee_schedule_bindings.get(digest))
        if binding is None:
            reasons.append("KALSHI_OFFICIAL_FEE_SCHEDULE_UNRECOGNIZED")
        self._kalshi_fee_schedule_capture = {
            "schema_version": 1,
            "source": "kalshi_official_fee_schedule",
            "source_url": self._kalshi_fee_schedule_url,
            "final_url": final_url,
            "status": "REVIEWED" if not reasons else "FAIL_CLOSED",
            "reason_codes": sorted(set(reasons)),
            "request": request,
            "raw_body_sha256": digest,
            "byte_size": len(body),
            "formula_binding": binding,
            "live_eligible": False,
        }
        return copy.deepcopy(self._kalshi_fee_schedule_capture)

    def supporting_evidence(self) -> dict[str, Any] | None:
        """Return already-captured run-level evidence without causing network I/O."""

        if self._kalshi_fee_schedule_capture is None:
            return None
        return {
            "schema_version": 1,
            "kalshi_official_fee_schedule": copy.deepcopy(self._kalshi_fee_schedule_capture),
            "live_eligible": False,
        }

    def fetch_metadata(self, candidate: Mapping[str, Any], side: str) -> dict[str, Any]:
        """Reverse-resolve and capture one candidate side's native rules."""

        candidate_side = _candidate_side(candidate, side)
        venue = candidate_side["venue"]
        if venue == "kalshi":
            return self._fetch_kalshi_metadata(candidate, side, candidate_side)
        if venue == "polymarket":
            return self._fetch_polymarket_metadata(candidate, side, candidate_side)
        raise NativeEvidenceError("UNSUPPORTED_VENUE", f"Unsupported native venue: {venue!r}")

    def _fetch_kalshi_metadata(
        self,
        candidate: Mapping[str, Any],
        side: str,
        candidate_side: Mapping[str, Any],
    ) -> dict[str, Any]:
        ticker = _kalshi_ticker_hint(candidate_side)
        requests: list[dict[str, Any]] = []
        raw_response: dict[str, Any] = {}

        def capture(name: str, path: str) -> Any:
            try:
                payload, request = self._get_json(self._kalshi, path)
            except NativeEvidenceError as exc:
                raise NativeEvidenceError(
                    exc.reason_code,
                    str(exc),
                    evidence={
                        **exc.evidence,
                        "completed_requests": requests,
                        "partial_raw_response": dict(raw_response),
                        "metadata_capture_stage": name,
                    },
                ) from exc
            requests.append(request)
            raw_response[name] = payload
            return payload

        market_payload = capture("market", f"/markets/{ticker}")
        market = _mapping(_mapping(market_payload).get("market"))
        if not market:
            market = _mapping(market_payload)
        native_ticker = _string(market.get("ticker"))
        if not native_ticker or native_ticker.upper() != ticker:
            raise NativeEvidenceError(
                "NATIVE_ID_MISMATCH",
                "Kalshi response ticker did not match the reverse-resolved ticker",
                evidence={
                    "requested_ticker": ticker,
                    "returned_ticker": native_ticker,
                    "request": requests[0],
                    "raw_response": market_payload,
                },
            )

        event_ticker = _string(market.get("event_ticker"))
        if event_ticker is None:
            raise NativeEvidenceError(
                "NATIVE_ID_UNRESOLVED",
                "Kalshi market response omitted its native event ticker",
                evidence={"completed_requests": requests, "partial_raw_response": dict(raw_response)},
            )
        event_metadata_payload = capture("event_metadata", f"/events/{event_ticker}/metadata")
        event_metadata = _mapping(event_metadata_payload)
        event_payload = capture("event", f"/events/{event_ticker}")
        event = _mapping(_mapping(event_payload).get("event"))
        if not event:
            event = _mapping(event_payload)
        returned_event_ticker = _string(event.get("event_ticker"))
        if returned_event_ticker != event_ticker:
            raise NativeEvidenceError(
                "NATIVE_ID_MISMATCH",
                "Kalshi event response did not match the market event ticker",
                evidence={"completed_requests": requests, "partial_raw_response": dict(raw_response)},
            )
        series_ticker_from_market = _string(market.get("series_ticker"))
        series_ticker_from_event = _string(event.get("series_ticker"))
        if (
            series_ticker_from_market
            and series_ticker_from_event
            and series_ticker_from_market != series_ticker_from_event
        ):
            raise NativeEvidenceError(
                "NATIVE_ID_MISMATCH",
                "Kalshi market and event responses disagreed on the native series ticker",
                evidence={"completed_requests": requests, "partial_raw_response": dict(raw_response)},
            )
        series_ticker = series_ticker_from_event or series_ticker_from_market
        if series_ticker is None:
            raise NativeEvidenceError(
                "NATIVE_ID_UNRESOLVED",
                "Kalshi event response omitted its native series ticker",
                evidence={"completed_requests": requests, "partial_raw_response": dict(raw_response)},
            )
        series_payload = capture("series", f"/series/{series_ticker}")
        series = _mapping(_mapping(series_payload).get("series"))
        if not series:
            series = _mapping(series_payload)
        if _string(series.get("ticker")) != series_ticker:
            raise NativeEvidenceError(
                "NATIVE_ID_MISMATCH",
                "Kalshi series response did not match the event series ticker",
                evidence={"completed_requests": requests, "partial_raw_response": dict(raw_response)},
            )

        settlement_sources: list[Any] = []
        for source_record in (
            *_list_value(event_metadata.get("settlement_sources")),
            *_list_value(event.get("settlement_sources")),
            *_list_value(series.get("settlement_sources")),
        ):
            if source_record not in settlement_sources:
                settlement_sources.append(source_record)
        source_names = sorted(
            name for item in settlement_sources if (name := _string(_mapping(item).get("name"))) is not None
        )
        source_urls = sorted(
            url for item in settlement_sources if (url := _string(_mapping(item).get("url"))) is not None
        )
        rules = {
            "rules_primary": _string(market.get("rules_primary")),
            "rules_secondary": _string(market.get("rules_secondary")),
            "open_time": _iso_from_timestamp(market.get("open_time"), preserve_offset=True),
            "close_time": _iso_from_timestamp(market.get("close_time"), preserve_offset=True),
            "expiration_time": _iso_from_timestamp(market.get("expiration_time"), preserve_offset=True),
            "expected_expiration_time": _iso_from_timestamp(
                market.get("expected_expiration_time"), preserve_offset=True
            ),
            "latest_expiration_time": _iso_from_timestamp(market.get("latest_expiration_time"), preserve_offset=True),
            "settlement_timer_seconds": market.get("settlement_timer_seconds"),
            "settlement_sources": settlement_sources,
            "status": _string(market.get("status")),
            "fractional_trading_enabled": market.get("fractional_trading_enabled"),
            "can_close_early": market.get("can_close_early"),
            "early_close_condition": _string(market.get("early_close_condition")),
            "void_cancel_rules": _string(market.get("void_cancel_rules")),
            "strike_type": _string(market.get("strike_type")),
            "floor_strike": market.get("floor_strike"),
            "cap_strike": market.get("cap_strike"),
            "functional_strike": market.get("functional_strike"),
            "custom_strike": market.get("custom_strike"),
            "market_type": _string(market.get("market_type")),
            "notional_value_dollars": _string(market.get("notional_value_dollars")),
            "price_level_structure": _string(market.get("price_level_structure")),
            "price_ranges": _list_value(market.get("price_ranges")),
            "fee_waiver_expiration_time": _iso_from_timestamp(
                market.get("fee_waiver_expiration_time"), preserve_offset=True
            ),
            "mve_collection_ticker": _string(market.get("mve_collection_ticker")),
            "mve_selected_legs": _list_value(market.get("mve_selected_legs")),
            "event_mutually_exclusive": event.get("mutually_exclusive"),
            "event_collateral_return_type": _string(event.get("collateral_return_type")),
            "event_last_updated_ts": _iso_from_timestamp(event.get("last_updated_ts"), preserve_offset=True),
            "series_contract_url": _string(series.get("contract_url")),
            "series_contract_terms_url": _string(series.get("contract_terms_url")),
            "series_last_updated_ts": _iso_from_timestamp(series.get("last_updated_ts"), preserve_offset=True),
            "outcome_polarity": {"YES": "YES", "NO": "NO"},
        }
        resolution_parts = [
            value for value in (_string(market.get("rules_primary")), _string(market.get("rules_secondary"))) if value
        ]
        material_edge_cases = normalized_material_edge_cases("kalshi", rules)
        market_status = (_string(market.get("status")) or "").lower() or None
        fractional_trading_enabled = market.get("fractional_trading_enabled")
        book_eligible = market_status == "active"
        return {
            "schema_version": 1,
            "candidate_id": candidate.get("candidate_id"),
            "side": side,
            "venue": "kalshi",
            "pmxt_market_id": candidate_side.get("pmxt_market_id"),
            "native_market_id": native_ticker,
            "native_event_id": event_ticker,
            "native_series_id": series_ticker,
            "native_outcome_ids": {"YES": native_ticker, "NO": native_ticker},
            "native_outcome_labels": {
                "YES": _string(market.get("yes_sub_title")) or "YES",
                "NO": _string(market.get("no_sub_title")) or "NO",
            },
            "outcome_polarity": {"YES": "YES", "NO": "NO"},
            "market_status": market_status,
            "book_eligible": book_eligible,
            "minimum_order_size": 1.0 if fractional_trading_enabled is False else None,
            "size_increment": 1.0 if fractional_trading_enabled is False else None,
            "proposition": _string(market.get("title")),
            "open_time": rules["open_time"],
            "close_time": rules["close_time"],
            "expiration_time": rules["expiration_time"],
            "expected_expiration_time": rules["expected_expiration_time"],
            "latest_expiration_time": rules["latest_expiration_time"],
            "settlement_authority": " | ".join(source_names) if source_names else None,
            "resolution_source": " | ".join(source_urls) if source_urls else None,
            "resolution_criteria": "\n".join(resolution_parts) if resolution_parts else None,
            "void_cancel": rules["void_cancel_rules"],
            "material_edge_cases": material_edge_cases,
            "settlement_delay_seconds": _float(market.get("settlement_timer_seconds")),
            "market_type": rules["market_type"],
            "notional_value_dollars": rules["notional_value_dollars"],
            "fee_waiver_expiration_time": rules["fee_waiver_expiration_time"],
            "mve_collection_ticker": rules["mve_collection_ticker"],
            "mve_selected_legs": rules["mve_selected_legs"],
            "event_mutually_exclusive": rules["event_mutually_exclusive"],
            "event_collateral_return_type": rules["event_collateral_return_type"],
            "clarification_or_revision_timestamps": {
                "market_updated_time": _iso_from_timestamp(market.get("updated_time"), preserve_offset=True),
                "event_last_updated_ts": rules["event_last_updated_ts"],
                "series_last_updated_ts": rules["series_last_updated_ts"],
            },
            "normalized_rules": rules,
            "rule_hash": canonical_json_sha256(rules),
            "raw_sha256": canonical_json_sha256(raw_response),
            "requested_at": _earliest([request["requested_at"] for request in requests]),
            "received_at": _latest([request["received_at"] for request in requests]),
            "requests": requests,
            "raw_response": raw_response,
            "status": "RESOLVED",
            "live_eligible": False,
        }

    def _fetch_polymarket_metadata(
        self,
        candidate: Mapping[str, Any],
        side: str,
        candidate_side: Mapping[str, Any],
    ) -> dict[str, Any]:
        params, lookup_kind, lookup_value = _polymarket_lookup(candidate_side)
        payload, request = self._get_json(self._gamma, "/markets", params=params)
        records: Any = payload
        if isinstance(payload, Mapping):
            records = payload.get("markets", payload.get("data", payload.get("results", [])))
        if not isinstance(records, list):
            raise NativeEvidenceError(
                "NATIVE_RESPONSE_SHAPE",
                "Polymarket Gamma returned a non-list market response",
                evidence={"request": request, "raw_response": payload},
            )
        matches = [dict(record) for record in records if isinstance(record, Mapping)]
        if lookup_kind == "slug":
            matches = [record for record in matches if _string(record.get("slug")) == lookup_value]
        else:
            matches = [
                record
                for record in matches
                if (_string(record.get("conditionId")) or _string(record.get("condition_id"))) == lookup_value
            ]
        if len(matches) != 1:
            raise NativeEvidenceError(
                "NATIVE_ID_UNRESOLVED" if not matches else "NATIVE_ID_AMBIGUOUS",
                "Polymarket reverse lookup did not return exactly one native market",
                evidence={
                    "lookup_kind": lookup_kind,
                    "match_count": len(matches),
                    "request": request,
                    "raw_response": payload,
                },
            )
        market = matches[0]
        native_market_id = _string(market.get("id"))
        condition_id = _string(market.get("conditionId")) or _string(market.get("condition_id"))
        if not native_market_id or not condition_id:
            raise NativeEvidenceError(
                "NATIVE_ID_UNRESOLVED",
                "Polymarket Gamma response omitted its native market or condition identifier",
                evidence={"request": request, "raw_response": payload},
            )
        hinted_condition_id = _string(candidate_side.get("contract_address"))
        if hinted_condition_id and condition_id != hinted_condition_id:
            raise NativeEvidenceError(
                "NATIVE_ID_MISMATCH",
                "Polymarket slug lookup returned a different condition ID than the PMXT venue hint",
                evidence={
                    "hinted_condition_id": hinted_condition_id,
                    "returned_condition_id": condition_id,
                    "request": request,
                    "raw_response": payload,
                },
            )

        labels = [_string(value) for value in _list_value(market.get("outcomes"))]
        token_ids = [_string(value) for value in _list_value(market.get("clobTokenIds"))]
        if len(labels) != len(token_ids):
            token_ids = []
        native_outcomes = {
            label.upper(): token
            for label, token in zip(labels, token_ids)
            if label is not None and token is not None and label.upper() in {"YES", "NO"}
        }
        if set(native_outcomes) == {"YES", "NO"} and len(set(native_outcomes.values())) != 2:
            raise NativeEvidenceError(
                "NATIVE_OUTCOME_ID_AMBIGUOUS",
                "Polymarket YES and NO outcomes resolved to the same CLOB token ID",
                evidence={"request": request, "raw_response": payload},
            )
        polarity = {label.upper(): label.upper() for label in labels if label and label.upper() in {"YES", "NO"}}
        native_outcome_labels = {
            label.upper(): label for label in labels if label is not None and label.upper() in {"YES", "NO"}
        }
        parent_events = [_mapping(item) for item in _list_value(market.get("events")) if isinstance(item, Mapping)]
        native_event_ids = sorted(
            {event_id for event in parent_events if (event_id := _string(event.get("id"))) is not None}
        )
        rules = {
            "question": _string(market.get("question")),
            "description": _string(market.get("description")),
            "resolution_source": _string(market.get("resolutionSource")),
            "resolved_by": _string(market.get("resolvedBy")),
            "start_date": _iso_from_timestamp(
                market.get("startDate") or market.get("startDateIso"), preserve_offset=True
            ),
            "end_date": _iso_from_timestamp(market.get("endDate") or market.get("endDateIso"), preserve_offset=True),
            "event_start_time": _iso_from_timestamp(market.get("eventStartTime"), preserve_offset=True),
            "uma_end_date": _iso_from_timestamp(market.get("umaEndDate"), preserve_offset=True),
            "uma_resolution_status": market.get("umaResolutionStatus"),
            "neg_risk": market.get("negRisk"),
            "active": market.get("active"),
            "closed": market.get("closed"),
            "archived": market.get("archived"),
            "accepting_orders": market.get("acceptingOrders"),
            "void_cancel_rules": _string(market.get("voidCancelRules")) or _string(market.get("void_cancel_rules")),
            "settlement_delay_seconds": _float(market.get("settlementDelaySeconds")),
            "fees_enabled": market.get("feesEnabled"),
            "created_at": _iso_from_timestamp(market.get("createdAt"), preserve_offset=True),
            "updated_at": _iso_from_timestamp(market.get("updatedAt"), preserve_offset=True),
            "parent_events": parent_events,
            "outcome_polarity": polarity,
        }
        material_edge_cases = normalized_material_edge_cases("polymarket", rules)
        raw_response = {"markets": payload, "selected_market": market}
        book_eligible = (
            market.get("active") is True
            and market.get("closed") is False
            and market.get("archived") is False
            and market.get("acceptingOrders") is True
        )
        market_status = "active" if book_eligible else "not_accepting_orders"
        return {
            "schema_version": 1,
            "candidate_id": candidate.get("candidate_id"),
            "side": side,
            "venue": "polymarket",
            "pmxt_market_id": candidate_side.get("pmxt_market_id"),
            "native_market_id": native_market_id,
            "native_event_id": native_event_ids[0] if len(native_event_ids) == 1 else None,
            "native_condition_id": condition_id,
            "native_outcome_ids": native_outcomes,
            "native_outcome_labels": native_outcome_labels,
            "outcome_polarity": polarity,
            "market_status": market_status,
            "book_eligible": book_eligible,
            "minimum_order_size": None,
            "size_increment": None,
            "proposition": rules["question"],
            "open_time": rules["start_date"],
            "close_time": rules["end_date"],
            "expiration_time": rules["uma_end_date"],
            "settlement_authority": rules["resolved_by"],
            "resolution_source": rules["resolution_source"],
            "resolution_criteria": rules["description"],
            "void_cancel": rules["void_cancel_rules"],
            "material_edge_cases": material_edge_cases,
            "settlement_delay_seconds": rules["settlement_delay_seconds"],
            "fees_enabled": rules["fees_enabled"],
            "market_type": "binary" if set(native_outcomes) == {"YES", "NO"} else None,
            "negative_risk": rules["neg_risk"],
            "clarification_or_revision_timestamps": {
                "created_at": rules["created_at"],
                "updated_at": rules["updated_at"],
                "uma_end_date": rules["uma_end_date"],
            },
            "normalized_rules": rules,
            "rule_hash": canonical_json_sha256(rules),
            "raw_sha256": canonical_json_sha256(raw_response),
            "requested_at": request["requested_at"],
            "received_at": request["received_at"],
            "requests": [request],
            "raw_response": raw_response,
            "status": "RESOLVED",
            "live_eligible": False,
        }

    def _fetch_kalshi_fee_evidence(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Capture the public, versioned taker-fee schedule for one Kalshi market."""

        market_ticker = _string(metadata.get("native_market_id"))
        event_ticker = _string(metadata.get("native_event_id"))
        series_ticker = _string(metadata.get("native_series_id"))
        if not market_ticker or not event_ticker or not series_ticker:
            return {
                "status": "FAIL_CLOSED",
                "venue": "kalshi",
                "model": None,
                "liquidity_role": "TAKER",
                "reason_codes": ["KALSHI_FEE_IDENTIFIERS_INCOMPLETE"],
                "live_eligible": False,
            }

        official_schedule = self._capture_kalshi_official_fee_schedule()
        formula_binding = _validated_fee_formula_binding(official_schedule.get("formula_binding"))
        requests: list[dict[str, Any]] = []
        raw_response: dict[str, Any] = {}
        official_request = official_schedule.get("request")
        if isinstance(official_request, Mapping) and official_request.get("method") == "GET":
            requests.append(dict(official_request))
        official_summary = {
            key: copy.deepcopy(official_schedule.get(key))
            for key in (
                "source_url",
                "final_url",
                "status",
                "reason_codes",
                "raw_body_sha256",
                "byte_size",
                "formula_binding",
            )
        }
        raw_response["official_fee_schedule_reference"] = official_summary

        def capture(name: str, path: str, params: Mapping[str, Any] | None = None) -> Any:
            try:
                payload, request = self._get_json(self._kalshi, path, params=params)
            except NativeEvidenceError as exc:
                raise NativeEvidenceError(
                    exc.reason_code,
                    str(exc),
                    evidence={
                        **exc.evidence,
                        "completed_requests": requests,
                        "partial_raw_response": dict(raw_response),
                        "fee_capture_stage": name,
                    },
                ) from exc
            requests.append(request)
            raw_response[name] = payload
            return payload

        event_payload = capture("event", f"/events/{event_ticker}")
        series_payload = capture("series", f"/series/{series_ticker}")
        series_changes_payload = capture(
            "series_fee_changes",
            "/series/fee_changes",
            {"series_ticker": series_ticker, "show_historical": "true"},
        )
        event_changes_payload = capture(
            "event_fee_changes",
            "/events/fee_changes",
            {"event_ticker": event_ticker, "limit": 1000},
        )

        event = _mapping(_mapping(event_payload).get("event"))
        if not event:
            event = _mapping(event_payload)
        series = _mapping(_mapping(series_payload).get("series"))
        if not series:
            series = _mapping(series_payload)
        series_changes = [
            _mapping(item)
            for item in _list_value(_mapping(series_changes_payload).get("series_fee_change_arr"))
            if isinstance(item, Mapping)
        ]
        event_changes_wrapper = _mapping(event_changes_payload)
        event_changes = [
            _mapping(item)
            for item in _list_value(event_changes_wrapper.get("event_fee_changes"))
            if isinstance(item, Mapping)
        ]

        reason_codes: list[str] = [str(value) for value in _list_value(official_schedule.get("reason_codes"))]
        if official_schedule.get("status") != "REVIEWED" or formula_binding is None:
            reason_codes.append("KALSHI_OFFICIAL_FEE_SCHEDULE_NOT_REVIEWED")
        returned_event = _string(event.get("event_ticker"))
        returned_series_from_event = _string(event.get("series_ticker"))
        returned_series = _string(series.get("ticker"))
        if returned_event != event_ticker:
            reason_codes.append("KALSHI_EVENT_ID_MISMATCH")
        if returned_series_from_event != series_ticker or returned_series != series_ticker:
            reason_codes.append("KALSHI_SERIES_ID_MISMATCH")
        if _string(event_changes_wrapper.get("cursor")):
            reason_codes.append("KALSHI_EVENT_FEE_HISTORY_PAGINATED")

        for change in series_changes:
            if _string(change.get("series_ticker")) != series_ticker:
                reason_codes.append("KALSHI_SERIES_FEE_CHANGE_ID_MISMATCH")
            if _fee_change_timestamp(change) is None:
                reason_codes.append("KALSHI_SERIES_FEE_CHANGE_TIMESTAMP_INVALID")
        for change in event_changes:
            if _string(change.get("event_ticker")) != event_ticker:
                reason_codes.append("KALSHI_EVENT_FEE_CHANGE_ID_MISMATCH")
            if _string(change.get("series_ticker")) not in {None, series_ticker}:
                reason_codes.append("KALSHI_EVENT_FEE_CHANGE_SERIES_MISMATCH")
            if _fee_change_timestamp(change) is None:
                reason_codes.append("KALSHI_EVENT_FEE_CHANGE_TIMESTAMP_INVALID")

        series_fee_type = _string(series.get("fee_type"))
        series_multiplier = _float(series.get("fee_multiplier"))
        event_override_type = _string(event.get("fee_type_override"))
        event_override_raw = event.get("fee_multiplier_override")
        event_override_multiplier = _float(event_override_raw)
        if series_fee_type is None or series_multiplier is None or series_multiplier < 0:
            reason_codes.append("KALSHI_SERIES_FEE_TUPLE_INVALID")
        override_is_clear = event_override_type is None and event_override_raw is None
        override_is_complete = event_override_type is not None and event_override_multiplier is not None
        if not override_is_clear and not override_is_complete:
            reason_codes.append("KALSHI_EVENT_FEE_OVERRIDE_PARTIAL")

        effective_type = event_override_type if override_is_complete else series_fee_type
        effective_multiplier = event_override_multiplier if override_is_complete else series_multiplier
        supported_fee_types = (
            set(_list_value(formula_binding.get("supported_taker_fee_types"))) if formula_binding else set()
        )
        if formula_binding is not None and effective_type not in supported_fee_types:
            reason_codes.append("KALSHI_FEE_TYPE_UNSUPPORTED")
        if effective_multiplier is None or effective_multiplier < 0:
            reason_codes.append("KALSHI_FEE_MULTIPLIER_INVALID")

        if _string(metadata.get("market_type")) != "binary":
            reason_codes.append("KALSHI_MARKET_TYPE_UNSUPPORTED")
        if _float(metadata.get("notional_value_dollars")) != 1.0:
            reason_codes.append("KALSHI_NOTIONAL_VALUE_UNSUPPORTED")
        if _string(metadata.get("mve_collection_ticker")) or _list_value(metadata.get("mve_selected_legs")):
            reason_codes.append("KALSHI_MULTIVARIATE_MARKET_UNSUPPORTED")

        window = _request_window(requests)
        observed_at = _string(window.get("response_wall_utc"))
        document_effective_from = f"{formula_binding['document_effective_date']}T00:00:00Z" if formula_binding else None
        if document_effective_from is not None and observed_at is not None and document_effective_from > observed_at:
            reason_codes.append("KALSHI_OFFICIAL_FEE_SCHEDULE_NOT_YET_EFFECTIVE")
        series_effective_from = _latest_matching_change(
            series_changes,
            observed_at=observed_at,
            fee_type_key="fee_type",
            fee_multiplier_key="fee_multiplier",
            fee_type=series_fee_type,
            fee_multiplier=series_multiplier,
        )
        event_effective_from = _latest_matching_change(
            event_changes,
            observed_at=observed_at,
            fee_type_key="fee_type_override",
            fee_multiplier_key="fee_multiplier_override",
            fee_type=event_override_type if override_is_complete else None,
            fee_multiplier=event_override_multiplier if override_is_complete else None,
        )
        if series_effective_from is None:
            reason_codes.append("KALSHI_SERIES_FEE_EFFECTIVE_TIMESTAMP_UNRESOLVED")
        if override_is_complete and event_effective_from is None:
            reason_codes.append("KALSHI_EVENT_FEE_EFFECTIVE_TIMESTAMP_UNRESOLVED")
        effective_candidates = [
            value for value in (document_effective_from, series_effective_from, event_effective_from) if value
        ]
        effective_from = _latest(effective_candidates) if effective_candidates else None

        schedule = {
            "market_ticker": market_ticker,
            "event_ticker": event_ticker,
            "series_ticker": series_ticker,
            "series_current": {
                "fee_type": series_fee_type,
                "fee_multiplier": series_multiplier,
                "last_updated_ts": _iso_from_timestamp(series.get("last_updated_ts")),
            },
            "event_current": {
                "fee_type_override": event_override_type,
                "fee_multiplier_override": event_override_multiplier if override_is_complete else None,
                "last_updated_ts": _iso_from_timestamp(event.get("last_updated_ts")),
            },
            "effective": {
                "fee_type": effective_type,
                "fee_multiplier": effective_multiplier,
                "taker_base_coefficient": (formula_binding.get("taker_base_coefficient") if formula_binding else None),
                "trade_fee_rounding_quantum": (
                    formula_binding.get("trade_fee_rounding_quantum") if formula_binding else None
                ),
                "balance_precision_upper_bound": (
                    formula_binding.get("balance_precision_upper_bound") if formula_binding else None
                ),
            },
            "official_fee_schedule": official_summary,
            "series_fee_changes": series_changes,
            "event_fee_changes": event_changes,
        }
        return {
            "status": "VALID" if not reason_codes else "FAIL_CLOSED",
            "venue": "kalshi",
            "model": (
                "KALSHI_QUADRATIC_TAKER"
                if formula_binding is not None and effective_type in supported_fee_types
                else None
            ),
            "liquidity_role": "TAKER",
            "passive_fills_assumed": False,
            "calculation_class": "CONSERVATIVE_UPPER_BOUND",
            "exact_fee_claimed": False,
            "account_fee_accumulator_inputs": "NOT_AVAILABLE",
            "account_class_inputs": "NOT_AVAILABLE",
            "coefficient": formula_binding.get("taker_base_coefficient") if formula_binding else None,
            "trade_fee_rounding_quantum": (
                formula_binding.get("trade_fee_rounding_quantum") if formula_binding else None
            ),
            "balance_precision_upper_bound": (
                formula_binding.get("balance_precision_upper_bound") if formula_binding else None
            ),
            "official_fee_schedule_sha256": official_schedule.get("raw_body_sha256"),
            "formula_binding_id": formula_binding.get("binding_id") if formula_binding else None,
            "schedule_observed_at": observed_at,
            "effective_from": effective_from,
            "effective_basis": "VENUE_SCHEDULED_CHANGE" if effective_from else "UNRESOLVED",
            "schedule_sha256": canonical_json_sha256(schedule),
            "raw_sha256": canonical_json_sha256(raw_response),
            "reason_codes": sorted(set(reason_codes)),
            "schedule": schedule,
            "requests": requests,
            "raw_response": raw_response,
            "capture_window": window,
            "live_eligible": False,
        }

    @staticmethod
    def _finalize_kalshi_fee_window(
        fee_evidence: Mapping[str, Any],
        book_request: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = dict(fee_evidence)
        requests = [*_list_value(result.get("requests")), dict(book_request)]
        result["capture_window"] = _request_window(requests)
        reasons = list(_list_value(result.get("reason_codes")))
        window_start = _iso_from_timestamp(result["capture_window"].get("request_wall_utc"))
        window_end = _iso_from_timestamp(result["capture_window"].get("response_wall_utc"))
        schedule = _mapping(result.get("schedule"))
        all_changes = [
            *_list_value(schedule.get("series_fee_changes")),
            *_list_value(schedule.get("event_fee_changes")),
        ]
        if window_start is None or window_end is None:
            reasons.append("KALSHI_FEE_CAPTURE_WINDOW_INVALID")
        else:
            for raw_change in all_changes:
                scheduled = _fee_change_timestamp(_mapping(raw_change))
                if scheduled is not None and window_start <= scheduled <= window_end:
                    reasons.append("KALSHI_FEE_CHANGE_CROSSED_CAPTURE_WINDOW")
                    break
        result["reason_codes"] = sorted({str(reason) for reason in reasons})
        result["status"] = "VALID" if not result["reason_codes"] else "FAIL_CLOSED"
        return result

    def fetch_book(self, metadata: Mapping[str, Any], *, depth: int = 100) -> dict[str, Any]:
        """Capture aggressive-fill depth for one already verified native market."""

        if not 1 <= depth <= 100:
            raise ValueError("depth must be between 1 and 100")
        venue = metadata.get("venue")
        if venue == "kalshi":
            return self._fetch_kalshi_book(metadata, depth=depth)
        if venue == "polymarket":
            return self._fetch_polymarket_book(metadata, depth=depth)
        raise NativeEvidenceError("UNSUPPORTED_VENUE", f"Unsupported native venue: {venue!r}")

    @staticmethod
    def _levels(value: Any, *, cents: bool = False) -> list[dict[str, float]]:
        levels: list[dict[str, float]] = []
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return levels
        for raw_level in value:
            price: float | None = None
            size: float | None = None
            if isinstance(raw_level, Mapping):
                price = _float(raw_level.get("price"))
                size = _float(raw_level.get("size"))
            elif isinstance(raw_level, Sequence) and not isinstance(raw_level, (str, bytes)) and len(raw_level) >= 2:
                price = _float(raw_level[0])
                size = _float(raw_level[1])
            if price is None or size is None:
                continue
            if cents:
                price /= 100
            levels.append({"price": price, "size": size})
        return levels

    def _fetch_kalshi_book(self, metadata: Mapping[str, Any], *, depth: int) -> dict[str, Any]:
        ticker = _string(metadata.get("native_market_id"))
        if not ticker:
            raise NativeEvidenceError("NATIVE_ID_UNRESOLVED", "Kalshi book request requires a native ticker")
        fee_evidence = self._fetch_kalshi_fee_evidence(metadata)
        try:
            payload, request = self._get_json(self._kalshi, f"/markets/{ticker}/orderbook", params={"depth": depth})
        except NativeEvidenceError as exc:
            raise NativeEvidenceError(
                exc.reason_code,
                str(exc),
                evidence={
                    **exc.evidence,
                    "completed_requests": _list_value(fee_evidence.get("requests")),
                    "partial_raw_response": {"fee_evidence": fee_evidence.get("raw_response")},
                },
            ) from exc
        fee_evidence = self._finalize_kalshi_fee_window(fee_evidence, request)
        wrapper = _mapping(payload)
        book = _mapping(wrapper.get("orderbook_fp"))
        cents = False
        if not book:
            book = _mapping(wrapper.get("orderbook"))
            cents = True
        yes_bids = self._levels(book.get("yes_dollars") if not cents else book.get("yes"), cents=cents)
        no_bids = self._levels(book.get("no_dollars") if not cents else book.get("no"), cents=cents)
        yes_bids.sort(key=lambda level: level["price"], reverse=True)
        no_bids.sort(key=lambda level: level["price"], reverse=True)
        depth_truncated = len(yes_bids) > depth or len(no_bids) > depth
        yes_bids = yes_bids[:depth]
        no_bids = no_bids[:depth]
        yes_asks = sorted(
            ({"price": round(1 - level["price"], 10), "size": level["size"]} for level in no_bids),
            key=lambda level: level["price"],
        )
        no_asks = sorted(
            ({"price": round(1 - level["price"], 10), "size": level["size"]} for level in yes_bids),
            key=lambda level: level["price"],
        )
        raw_response = {"orderbook": payload}
        return {
            "schema_version": 1,
            "candidate_id": metadata.get("candidate_id"),
            "venue": "kalshi",
            "native_market_id": ticker,
            "market_status": metadata.get("market_status"),
            "book_eligible": metadata.get("book_eligible") is True,
            "minimum_order_size": metadata.get("minimum_order_size"),
            "size_increment": metadata.get("size_increment"),
            "request_started_at": request["request_wall_utc"],
            "received_at": request["response_wall_utc"],
            "request_monotonic_ns": request["request_monotonic_ns"],
            "response_monotonic_ns": request["response_monotonic_ns"],
            "rtt_ms": request["rtt_ms"],
            "source_timestamp": None,
            "as_of": None,
            "timestamp_basis": "no_venue_source_timestamp",
            "freshness_basis": "LOCAL_RECEIPT_BOUNDED",
            "normalized_depth_limit": depth,
            "normalized_depth_truncated": depth_truncated,
            "raw_sha256": canonical_json_sha256(raw_response),
            "sides": {
                "YES": {"bids": yes_bids, "asks": yes_asks},
                "NO": {"bids": no_bids, "asks": no_asks},
            },
            "fee_evidence": fee_evidence,
            "requests": [request],
            "raw_response": raw_response,
            "live_eligible": False,
        }

    def _fetch_polymarket_book(self, metadata: Mapping[str, Any], *, depth: int) -> dict[str, Any]:
        outcome_ids = _mapping(metadata.get("native_outcome_ids"))
        if set(outcome_ids) != {"YES", "NO"} or not all(_string(value) for value in outcome_ids.values()):
            raise NativeEvidenceError(
                "NATIVE_OUTCOME_ID_UNRESOLVED",
                "Polymarket CLOB books require unique native YES and NO token IDs",
            )
        if len({_string(value) for value in outcome_ids.values()}) != 2:
            raise NativeEvidenceError(
                "NATIVE_OUTCOME_ID_AMBIGUOUS",
                "Polymarket YES and NO outcomes must use distinct CLOB token IDs",
            )
        raw_books: dict[str, Any] = {}
        raw_fee_rates: dict[str, Any] = {}
        requests: list[dict[str, Any]] = []
        normalized_sides: dict[str, Any] = {}
        source_timestamps: dict[str, str | None] = {}
        minimum_order_sizes: dict[str, float | None] = {}
        tick_sizes: dict[str, float | None] = {}
        base_fee_bps: dict[str, int | None] = {}
        fee_reason_codes: list[str] = []
        depth_truncated = False
        condition_id = _string(metadata.get("native_condition_id"))
        for polarity in ("YES", "NO"):
            token_id = str(outcome_ids[polarity])
            try:
                fee_payload, fee_request = self._get_json(self._clob, "/fee-rate", params={"token_id": token_id})
            except NativeEvidenceError as exc:
                raise NativeEvidenceError(
                    exc.reason_code,
                    str(exc),
                    evidence={
                        **exc.evidence,
                        "completed_requests": requests,
                        "partial_raw_response": {"books": raw_books, "fee_rates": raw_fee_rates},
                        "polarity": polarity,
                        "fee_capture_stage": "polymarket_token_fee_rate",
                    },
                ) from exc
            raw_fee_rates[polarity] = fee_payload
            requests.append(fee_request)
            raw_base_fee = _mapping(fee_payload).get("base_fee")
            if isinstance(raw_base_fee, bool) or not isinstance(raw_base_fee, int) or raw_base_fee < 0:
                base_fee_bps[polarity] = None
                fee_reason_codes.append(f"POLYMARKET_{polarity}_BASE_FEE_INVALID")
            else:
                base_fee_bps[polarity] = raw_base_fee
            try:
                payload, request = self._get_json(self._clob, "/book", params={"token_id": token_id})
            except NativeEvidenceError as exc:
                raise NativeEvidenceError(
                    exc.reason_code,
                    str(exc),
                    evidence={
                        **exc.evidence,
                        "completed_requests": requests,
                        "partial_raw_response": {"books": raw_books, "fee_rates": raw_fee_rates},
                    },
                ) from exc
            raw = _mapping(payload)
            if _string(raw.get("asset_id")) != token_id:
                raise NativeEvidenceError(
                    "NATIVE_ID_MISMATCH",
                    "Polymarket CLOB asset_id was missing or did not match the requested outcome token",
                    evidence={
                        "polarity": polarity,
                        "request": request,
                        "partial_raw_response": {"books": {**raw_books, polarity: payload}},
                    },
                )
            if not condition_id or _string(raw.get("market")) != condition_id:
                raise NativeEvidenceError(
                    "NATIVE_ID_MISMATCH",
                    "Polymarket CLOB market was missing or did not match the resolved condition ID",
                    evidence={
                        "polarity": polarity,
                        "request": request,
                        "partial_raw_response": {"books": {**raw_books, polarity: payload}},
                    },
                )
            bids = self._levels(raw.get("bids"))
            asks = self._levels(raw.get("asks"))
            bids.sort(key=lambda level: level["price"], reverse=True)
            asks.sort(key=lambda level: level["price"])
            depth_truncated = depth_truncated or len(bids) > depth or len(asks) > depth
            bids = bids[:depth]
            asks = asks[:depth]
            normalized_sides[polarity] = {"bids": bids, "asks": asks}
            source_timestamps[polarity] = _iso_from_timestamp(raw.get("timestamp"))
            minimum_order_sizes[polarity] = _float(raw.get("min_order_size"))
            tick_sizes[polarity] = _float(raw.get("tick_size"))
            raw_books[polarity] = payload
            requests.append(request)
        known_source_times = [value for value in source_timestamps.values() if value is not None]
        known_minimums = [value for value in minimum_order_sizes.values() if value is not None]
        fees_enabled = metadata.get("fees_enabled")
        fee_reason_codes.extend(
            [
                "POLYMARKET_CONDITION_FEE_PARAMETERS_NOT_CAPTURED",
                "POLYMARKET_FEE_EFFECTIVE_TIMESTAMP_UNAVAILABLE",
                "POLYMARKET_TOKEN_BASE_FEE_IS_ESTIMATE_ONLY",
            ]
        )
        if not isinstance(fees_enabled, bool):
            fee_reason_codes.append("POLYMARKET_FEES_ENABLED_UNKNOWN")
        if fees_enabled is False and any((value or 0) != 0 for value in base_fee_bps.values()):
            fee_reason_codes.append("POLYMARKET_FEE_FLAG_RATE_CONTRADICTION")
        fee_schedule = {
            "native_condition_id": condition_id,
            "native_outcome_ids": outcome_ids,
            "fees_enabled": fees_enabled,
            "base_fee_bps_estimate_by_outcome": base_fee_bps,
            "condition_fee_details": None,
        }
        fee_requests = [request for request in requests if request.get("path") == "/fee-rate"]
        fee_evidence = {
            "status": "FAIL_CLOSED",
            "venue": "polymarket",
            "model": "POLYMARKET_FEE_ESTIMATE_ONLY",
            "liquidity_role": "TAKER",
            "passive_fills_assumed": False,
            "estimate_formula": "shares * (base_fee_bps / 10000) * price * (1 - price)",
            "estimate_only": True,
            "rounding_quantum": "0.00001",
            "rounding_basis": "CONSERVATIVE_CEILING",
            "schedule_observed_at": _request_window(fee_requests).get("response_wall_utc"),
            "effective_from": None,
            "effective_basis": "UNAVAILABLE",
            "schedule_sha256": canonical_json_sha256(fee_schedule),
            "raw_sha256": canonical_json_sha256(raw_fee_rates),
            "reason_codes": sorted(set(fee_reason_codes)),
            "schedule": fee_schedule,
            "requests": fee_requests,
            "raw_response": raw_fee_rates,
            "capture_window": _request_window(fee_requests),
            "live_eligible": False,
        }
        raw_response = {"books": raw_books, "fee_rates": raw_fee_rates}
        request_window = _request_window(requests)
        return {
            "schema_version": 1,
            "candidate_id": metadata.get("candidate_id"),
            "venue": "polymarket",
            "native_market_id": metadata.get("native_market_id"),
            "native_condition_id": condition_id,
            "native_outcome_ids": outcome_ids,
            "market_status": metadata.get("market_status"),
            "book_eligible": metadata.get("book_eligible") is True,
            "minimum_order_size": max(known_minimums) if len(known_minimums) == 2 else None,
            "size_increment": None,
            "minimum_order_sizes": minimum_order_sizes,
            "tick_sizes": tick_sizes,
            "request_started_at": request_window["request_wall_utc"],
            "received_at": request_window["response_wall_utc"],
            "request_monotonic_ns": request_window["request_monotonic_ns"],
            "response_monotonic_ns": request_window["response_monotonic_ns"],
            "rtt_ms": request_window["rtt_ms"],
            "source_timestamp": _earliest(known_source_times) if len(known_source_times) == 2 else None,
            "source_timestamps": source_timestamps,
            "as_of": _earliest(known_source_times) if len(known_source_times) == 2 else None,
            "timestamp_basis": "venue_source_timestamp",
            "freshness_basis": "VENUE_SOURCE_TIMESTAMP",
            "normalized_depth_limit": depth,
            "normalized_depth_truncated": depth_truncated,
            "raw_sha256": canonical_json_sha256(raw_response),
            "sides": normalized_sides,
            "fee_evidence": fee_evidence,
            "requests": requests,
            "raw_response": raw_response,
            "live_eligible": False,
        }
