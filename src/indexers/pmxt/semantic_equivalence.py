"""Quarantined legacy parser for PMXT-proposed venue-native metadata.

The historical capture is immutable and production-disabled.  Only parser
fixtures using the module-private fixture permit and exact ``httpx.MockTransport``
instances may construct its writer or client.  Real transport and capture entry
points fail before a client or artifact directory is created.  This module has
no book, trade, account, credential, signing, order, fill, PnL, retry, or
automation surface.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

PROTOCOL_ID = "pmxt-venue-native-semantic-equivalence-v1"
SCHEMA_VERSION = "1.0.0"
DEFAULT_KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_POLYMARKET_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_MAX_RESPONSE_BYTES = 20_000_000
DEFAULT_TIMEOUT_SECONDS = 20.0
EXPECTED_PAIR_COUNT = 25
MAX_REQUESTS_PER_PAIR = 6

LEGACY_CAPTURE_QUARANTINED = True
PRODUCTION_READY = False
REAL_NETWORK_CAPTURE_ENABLED = False
_QUARANTINE_MESSAGE = (
    "legacy semantic-equivalence capture is quarantined; real network and capture entrypoints are disabled"
)
_FIXTURE_PARSER_PERMIT = object()

DEFAULT_CANDIDATE_PATH = Path("data/pmxt/runs/20260829T214011875577Z_33495fd8/candidates.jsonl")
DEFAULT_CANDIDATE_SHA256 = "088b881abfbfebc8d4f8b0d20c9d4c1521356ea9ed2724bd34105a6c6cec206e"
DEFAULT_PROTOCOL_PATH = Path("results/semantic_equivalence_v1/protocol_v1.json")
DEFAULT_SCHEMA_PATH = Path("results/semantic_equivalence_v1/venue_record_schema_v1.json")
DEFAULT_FREEZE_PATH = Path("results/semantic_equivalence_v1/implementation_freeze.json")
DEFAULT_OUTPUT_PATH = Path("data/pmxt/semantic_equivalence/runs/capture_001")
SOURCE_PATH = Path("src/indexers/pmxt/semantic_equivalence.py")
TEST_PATH = Path("tests/test_pmxt_semantic_equivalence.py")

_URL_RE = re.compile(r"https?://[^\s<>\]\[(){}\"']+", re.IGNORECASE)
_TIMEZONE_RE = re.compile(r"\b(?:UTC|GMT|ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT)\b")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,254}$")
_KALSHI_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9._:-]{0,199}$")
_POLYMARKET_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
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
_KEYWORD_GROUPS = {
    "void": ("void",),
    "cancel": ("cancel", "cancelled", "canceled"),
    "refund": ("refund",),
    "indeterminate": ("indeterminate", "ambiguous", "no winner", "other"),
    "early_close": ("close early", "early close", "immediately resolve", "resolve immediately"),
}


class SemanticCaptureError(RuntimeError):
    """Raised when the frozen capture contract cannot be followed safely."""


def _require_fixture_parser_permit(permit: object | None) -> None:
    if permit is not _FIXTURE_PARSER_PERMIT:
        raise SemanticCaptureError(_QUARANTINE_MESSAGE)


class _DuplicateKey(ValueError):
    pass


@dataclass(frozen=True)
class ResponseEvidence:
    ordinal: int
    candidate_id: str
    venue: str
    resource: str
    status: str
    status_code: int | None
    request_path: str
    request_params: Mapping[str, Any]
    request_started_utc_ns: int
    response_completed_utc_ns: int
    duration_monotonic_ns: int
    body_path: str | None
    body_bytes: int | None
    body_sha256: str | None
    headers_path: str | None
    headers_sha256: str | None
    request_metadata_path: str
    parsed: Any
    error: str | None

    def reference(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "candidate_id": self.candidate_id,
            "venue": self.venue,
            "resource": self.resource,
            "status": self.status,
            "status_code": self.status_code,
            "request_path": self.request_path,
            "request_params": dict(self.request_params),
            "request_started_utc_ns": self.request_started_utc_ns,
            "response_completed_utc_ns": self.response_completed_utc_ns,
            "duration_monotonic_ns": self.duration_monotonic_ns,
            "body_path": self.body_path,
            "body_bytes": self.body_bytes,
            "body_sha256": self.body_sha256,
            "headers_path": self.headers_path,
            "headers_sha256": self.headers_sha256,
            "request_metadata_path": self.request_metadata_path,
            "error": self.error,
        }


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes) -> Any:
    return json.loads(
        payload.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
        return list(decoded) if isinstance(decoded, list) else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("latin-1").strip()
        return text or None
    text = str(value).strip()
    return text or None


def _time_value(source: str, field: str, raw: Any) -> dict[str, Any]:
    normalized: str | None = None
    timezone_explicit = False
    text = _text(raw)
    if text is not None:
        candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.tzinfo is not None and parsed.utcoffset() is not None:
            timezone_explicit = True
            normalized = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "source": source,
        "field": field,
        "raw": raw,
        "normalized_utc": normalized,
        "timezone_explicit": timezone_explicit,
    }


def _present_time_values(source: str, record: Mapping[str, Any], fields: Iterable[str]) -> list[dict[str, Any]]:
    return [
        _time_value(source, field, record[field]) for field in fields if field in record and record[field] is not None
    ]


def _urls(*values: Any) -> list[str]:
    found: set[str] = set()
    for value in values:
        if isinstance(value, str):
            found.update(match.rstrip(".,;:") for match in _URL_RE.findall(value))
        elif isinstance(value, Mapping):
            found.update(_urls(*value.values()))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            found.update(_urls(*value))
    return sorted(found)


def _timezone_mentions(*values: Any) -> list[str]:
    mentions: set[str] = set()
    for value in values:
        if isinstance(value, str):
            mentions.update(_TIMEZONE_RE.findall(value))
        elif isinstance(value, Mapping):
            mentions.update(_timezone_mentions(*value.values()))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            mentions.update(_timezone_mentions(*value))
    return sorted(mentions)


def _keyword_matches(texts: Iterable[str | None]) -> dict[str, list[str]]:
    paragraphs: list[str] = []
    for text in texts:
        if text:
            paragraphs.extend(part.strip() for part in re.split(r"[\r\n]+", text) if part.strip())
    matches: dict[str, list[str]] = {}
    for group, keywords in _KEYWORD_GROUPS.items():
        values = sorted({part for part in paragraphs if any(keyword in part.casefold() for keyword in keywords)})
        matches[group] = values
    return matches


def _revision_times(records: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    created: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    clarification: list[dict[str, Any]] = []
    revision: list[dict[str, Any]] = []
    for source, record in records:
        for field, value in record.items():
            lowered = field.casefold()
            if value is None or not isinstance(value, (str, int, float)):
                continue
            destination: list[dict[str, Any]] | None = None
            if "clarif" in lowered:
                destination = clarification
            elif "revis" in lowered:
                destination = revision
            elif "creat" in lowered and ("time" in lowered or "date" in lowered or lowered.endswith("at")):
                destination = created
            elif "updat" in lowered and ("time" in lowered or "date" in lowered or lowered.endswith("at")):
                destination = updated
            if destination is not None:
                destination.append(_time_value(source, field, value))
    return {
        "created": created,
        "updated": updated,
        "clarification": clarification,
        "revision": revision,
    }


def _candidate_side(candidate: Mapping[str, Any], venue: str) -> tuple[str, dict[str, Any]]:
    matches: list[tuple[str, dict[str, Any]]] = []
    for side in ("a", "b"):
        if _text(candidate.get(f"venue_{side}")) == venue:
            matches.append(
                (
                    side,
                    {
                        "venue": venue,
                        "pmxt_market_id": _text(candidate.get(f"pmxt_market_id_{side}")),
                        "slug": _text(candidate.get(f"slug_{side}")),
                        "url": _text(candidate.get(f"url_{side}")),
                        "condition_id": _text(candidate.get(f"contract_address_{side}")),
                        "title": _text(candidate.get(f"title_{side}")),
                        "description": _text(candidate.get(f"description_{side}")),
                        "event_id": _text(candidate.get(f"event_id_{side}")),
                        "outcomes": _sequence(candidate.get(f"outcomes_{side}")),
                    },
                )
            )
    if len(matches) != 1:
        raise SemanticCaptureError(f"candidate does not contain exactly one {venue} side")
    return matches[0]


def _candidate_event_slug(side: Mapping[str, Any]) -> str | None:
    url = _text(side.get("url"))
    if not url:
        return None
    match = re.search(r"/event/([a-z0-9]+(?:-[a-z0-9]+)*)(?:[/?#]|$)", url)
    return match.group(1) if match else None


class CaptureWriter:
    """Fixture-only exclusive writer retained for legacy parser regression tests."""

    def __init__(self, final_path: Path, *, _fixture_permit: object | None = None) -> None:
        _require_fixture_parser_permit(_fixture_permit)
        self.final_path = final_path
        self.staging_path = final_path.with_name(f"{final_path.name}.inprogress")
        if self.final_path.exists() or self.staging_path.exists():
            raise SemanticCaptureError("capture output or in-progress path already exists")
        self.staging_path.mkdir(parents=True, exist_ok=False)
        self.ordinal = 0
        self.request_records: list[dict[str, Any]] = []

    def write(self, relative_path: str, payload: bytes) -> None:
        _write_exclusive(self.staging_path / relative_path, payload)

    def next_request_ordinal(self) -> int:
        self.ordinal += 1
        return self.ordinal

    def seal(self) -> None:
        if self.final_path.exists():
            raise SemanticCaptureError("final capture path appeared before seal")
        self.staging_path.rename(self.final_path)


class PublicMetadataClient:
    """Fixture-only parser client; exact mock transports are mandatory."""

    def __init__(
        self,
        *,
        writer: CaptureWriter,
        kalshi_base_url: str = DEFAULT_KALSHI_BASE_URL,
        polymarket_base_url: str = DEFAULT_POLYMARKET_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        kalshi_transport: httpx.BaseTransport | None = None,
        polymarket_transport: httpx.BaseTransport | None = None,
        utc_now_ns: Callable[[], int] = time.time_ns,
        monotonic_now_ns: Callable[[], int] = time.perf_counter_ns,
        _fixture_permit: object | None = None,
    ) -> None:
        _require_fixture_parser_permit(_fixture_permit)
        if type(kalshi_transport) is not httpx.MockTransport or type(polymarket_transport) is not httpx.MockTransport:
            raise SemanticCaptureError("legacy parser fixtures require exact httpx.MockTransport instances")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.writer = writer
        self.max_response_bytes = max_response_bytes
        self.utc_now_ns = utc_now_ns
        self.monotonic_now_ns = monotonic_now_ns
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "prediction-market-analysis/semantic-equivalence-v1",
        }
        self._clients = {
            "kalshi": httpx.Client(
                base_url=kalshi_base_url.rstrip("/"),
                headers=headers,
                timeout=timeout_seconds,
                transport=kalshi_transport,
                trust_env=False,
            ),
            "polymarket": httpx.Client(
                base_url=polymarket_base_url.rstrip("/"),
                headers=headers,
                timeout=timeout_seconds,
                transport=polymarket_transport,
                trust_env=False,
            ),
        }

    def close(self) -> None:
        for client in self._clients.values():
            client.close()

    def __enter__(self) -> PublicMetadataClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get(
        self,
        *,
        candidate_id: str,
        venue: str,
        resource: str,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> ResponseEvidence:
        if venue not in self._clients:
            raise SemanticCaptureError("unsupported venue")
        ordinal = self.writer.next_request_ordinal()
        safe_candidate = candidate_id.replace(":", "_")
        stem = f"{ordinal:04d}_{safe_candidate}_{venue}_{resource}"
        request_path = f"requests/{stem}.request.json"
        body_path = f"requests/{stem}.body.bin"
        headers_path = f"requests/{stem}.response_headers.json"
        request_params = dict(params or {})
        started_utc = self.utc_now_ns()
        started_monotonic = self.monotonic_now_ns()
        status_code: int | None = None
        body_bytes: bytes | None = None
        header_bytes: bytes | None = None
        parsed: Any = None
        error: str | None = None
        http_version: str | None = None
        response_url: str | None = None
        try:
            with self._clients[venue].stream("GET", path, params=request_params) as response:
                status_code = response.status_code
                response_url = str(response.url)
                http_version = _text(response.extensions.get("http_version"))
                body = bytearray()
                if response.is_stream_consumed:
                    buffered = response.content
                    if len(buffered) > self.max_response_bytes:
                        raise SemanticCaptureError("response exceeded frozen byte bound")
                    body.extend(buffered)
                else:
                    for chunk in response.iter_raw():
                        if len(body) + len(chunk) > self.max_response_bytes:
                            raise SemanticCaptureError("response exceeded frozen byte bound")
                        body.extend(chunk)
                body_bytes = bytes(body)
                raw_headers = [
                    {
                        "name_base64": base64.b64encode(name).decode("ascii"),
                        "value_base64": base64.b64encode(value).decode("ascii"),
                        "name_latin1": name.decode("latin-1"),
                        "value_latin1": value.decode("latin-1"),
                    }
                    for name, value in response.headers.raw
                    if name.decode("latin-1").lower() in _EVIDENCE_RESPONSE_HEADER_ALLOWLIST
                ]
                header_document = {
                    "schema_version": SCHEMA_VERSION,
                    "http_version": http_version,
                    "status_code": status_code,
                    "raw_header_pairs_in_order": raw_headers,
                }
                header_bytes = _canonical_json_bytes(header_document)
                content_encoding = response.headers.get("content-encoding")
                if content_encoding not in (None, "", "identity"):
                    error = f"unsupported content-encoding for strict JSON parsing: {content_encoding}"
                elif 200 <= status_code < 300:
                    try:
                        parsed = _strict_json_bytes(body_bytes)
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, _DuplicateKey) as exc:
                        error = f"invalid strict JSON: {type(exc).__name__}"
        except (httpx.HTTPError, OSError, SemanticCaptureError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            # Public responses may try to set a cookie.  It is not evidence and
            # must never become state on a later request from this client.
            self._clients[venue].cookies.clear()
        completed_utc = self.utc_now_ns()
        completed_monotonic = self.monotonic_now_ns()

        body_sha256 = _sha256_bytes(body_bytes) if body_bytes is not None else None
        headers_sha256 = _sha256_bytes(header_bytes) if header_bytes is not None else None
        if body_bytes is not None:
            self.writer.write(body_path, body_bytes)
        if header_bytes is not None:
            self.writer.write(headers_path, header_bytes)
        status = "OK" if error is None and status_code is not None and 200 <= status_code < 300 else "ERROR"
        request_document = {
            "schema_version": SCHEMA_VERSION,
            "ordinal": ordinal,
            "candidate_id": candidate_id,
            "venue": venue,
            "resource": resource,
            "method": "GET",
            "request_path": path,
            "request_params": request_params,
            "response_url": response_url,
            "request_started_utc_ns": started_utc,
            "response_completed_utc_ns": completed_utc,
            "duration_monotonic_ns": completed_monotonic - started_monotonic,
            "status": status,
            "status_code": status_code,
            "http_version": http_version,
            "body_path": body_path if body_bytes is not None else None,
            "body_bytes": len(body_bytes) if body_bytes is not None else None,
            "body_sha256": body_sha256,
            "headers_path": headers_path if header_bytes is not None else None,
            "headers_sha256": headers_sha256,
            "max_response_bytes": self.max_response_bytes,
            "retry_count": 0,
            "authorization_header_sent": False,
            "cookie_header_sent": False,
            "error": error,
        }
        request_bytes = _canonical_json_bytes(request_document)
        self.writer.write(request_path, request_bytes)
        self.writer.request_records.append(request_document)
        return ResponseEvidence(
            ordinal=ordinal,
            candidate_id=candidate_id,
            venue=venue,
            resource=resource,
            status=status,
            status_code=status_code,
            request_path=path,
            request_params=request_params,
            request_started_utc_ns=started_utc,
            response_completed_utc_ns=completed_utc,
            duration_monotonic_ns=completed_monotonic - started_monotonic,
            body_path=body_path if body_bytes is not None else None,
            body_bytes=len(body_bytes) if body_bytes is not None else None,
            body_sha256=body_sha256,
            headers_path=headers_path if header_bytes is not None else None,
            headers_sha256=headers_sha256,
            request_metadata_path=request_path,
            parsed=parsed,
            error=error,
        )


def _empty_venue_record(candidate_id: str, venue: str, side: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "venue": venue,
        "candidate_side": side,
        "status": "PENDING",
        "live_eligible": False,
        "identifiers": {
            "event": [],
            "market": [],
            "series": [],
            "condition": [],
            "question": [],
            "outcome": [],
            "token": [],
        },
        "exact_text": {
            "event_titles": [],
            "market_titles": [],
            "questions": [],
            "subtitles": [],
            "outcome_labels": [],
        },
        "polarity": {
            "explicit": False,
            "yes": None,
            "no": None,
            "source_fields": [],
        },
        "market_structure": {
            "market_type": None,
            "binary": None,
            "multivariate": None,
            "mutually_exclusive": None,
            "negative_risk": None,
            "raw_flags": {},
            "classification_basis": [],
        },
        "times": {
            "opening": [],
            "closing": [],
            "expiration": [],
            "earliest_resolution": [],
            "original_timezone_text": [],
        },
        "rules": {
            "primary": None,
            "secondary": None,
            "additional": [],
            "source_fields": [],
        },
        "resolution": {
            "source_text": [],
            "source_urls": [],
            "settlement_authority": [],
            "oracle_process": {},
            "raw_fields": {},
        },
        "early_close": {
            "can_close_early": None,
            "conditions": [],
            "rule_text_matches": [],
            "raw_fields": {},
        },
        "exceptional_outcomes": {
            "void": [],
            "cancel": [],
            "refund": [],
            "indeterminate": [],
            "raw_fields": {},
        },
        "revisions": {
            "created": [],
            "updated": [],
            "clarification": [],
            "revision": [],
        },
        "fees": {"raw_fields": {}},
        "response_evidence": [],
        "immutable_raw_hash": None,
        "normalized_rule_hash": None,
        "missing_requirements": [],
        "issues": [],
    }


def _id(source: str, field: str, value: Any) -> dict[str, Any]:
    return {"source": source, "field": field, "value": value}


def _response_set_hash(references: Sequence[Mapping[str, Any]]) -> str | None:
    bound = [
        {
            "ordinal": item.get("ordinal"),
            "resource": item.get("resource"),
            "body_sha256": item.get("body_sha256"),
            "headers_sha256": item.get("headers_sha256"),
        }
        for item in references
        if item.get("body_sha256") is not None and item.get("headers_sha256") is not None
    ]
    return _sha256_bytes(_canonical_json_bytes(bound)) if bound else None


def _finalize_venue_record(record: dict[str, Any]) -> dict[str, Any]:
    record["normalized_rule_hash"] = _sha256_bytes(_canonical_json_bytes(record["rules"]))
    record["immutable_raw_hash"] = _response_set_hash(record["response_evidence"])
    missing: list[str] = []
    identifiers = record["identifiers"]
    if not identifiers["event"]:
        missing.append("event_id")
    if not identifiers["market"]:
        missing.append("market_id")
    if not identifiers["series"]:
        missing.append("series_id")
    outcome_or_token_ids = [*identifiers["outcome"], *identifiers["token"]]
    if not any(item.get("value") is not None for item in outcome_or_token_ids):
        missing.append("outcome_or_token_ids")
    exact = record["exact_text"]
    if not (exact["questions"] or exact["market_titles"]):
        missing.append("exact_question_or_title")
    if not exact["outcome_labels"]:
        missing.append("outcome_labels")
    if record["polarity"]["explicit"] is not True:
        missing.append("explicit_yes_no_polarity")
    structure = record["market_structure"]
    if structure["binary"] is None or structure["multivariate"] is None:
        missing.append("market_structure")
    times = record["times"]
    for name in ("opening", "closing", "expiration", "earliest_resolution"):
        if not times[name]:
            missing.append(f"{name}_time")
    if not record["rules"]["primary"]:
        missing.append("primary_rules")
    if record["rules"]["secondary"] is None:
        missing.append("secondary_rules")
    if not record["resolution"]["source_urls"]:
        missing.append("resolution_source_urls")
    if not record["resolution"]["settlement_authority"]:
        missing.append("settlement_authority")
    if record["early_close"]["can_close_early"] is None and not record["early_close"]["conditions"]:
        missing.append("early_close_provisions")
    exceptional = record["exceptional_outcomes"]
    if not any(exceptional[name] for name in ("void", "cancel", "refund", "indeterminate")):
        missing.append("void_cancel_refund_indeterminate_behavior")
    revisions = record["revisions"]
    if not (revisions["clarification"] or revisions["revision"] or revisions["updated"]):
        missing.append("clarification_or_revision_timestamps")
    if any(item.get("status") != "OK" for item in record["response_evidence"]):
        missing.append("successful_raw_response_for_every_requested_resource")
    if record["immutable_raw_hash"] is None:
        missing.append("immutable_raw_hash")
    record["missing_requirements"] = sorted(set(missing))
    record["status"] = "COMPLETE" if not missing else "PARTIAL"
    return record


def _kalshi_record(
    candidate: Mapping[str, Any],
    *,
    side_name: str,
    side: Mapping[str, Any],
    client: PublicMetadataClient,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    record = _empty_venue_record(candidate_id, "kalshi", side_name)
    ticker = (_text(side.get("slug")) or "").upper()
    if not _KALSHI_TICKER_RE.fullmatch(ticker):
        record["issues"].append("KALSHI_TICKER_UNRESOLVED_FROM_CANDIDATE_SLUG")
        return _finalize_venue_record(record)

    market_response = client.get(
        candidate_id=candidate_id,
        venue="kalshi",
        resource="market",
        path=f"/markets/{quote(ticker, safe='')}",
    )
    responses = [market_response]
    market_payload = _mapping(market_response.parsed)
    market = _mapping(market_payload.get("market")) or market_payload
    if _text(market.get("ticker")) != ticker:
        record["issues"].append("KALSHI_MARKET_TICKER_MISMATCH_OR_MISSING")
        record["response_evidence"] = [item.reference() for item in responses]
        return _finalize_venue_record(record)

    event_ticker = _text(market.get("event_ticker"))
    series_ticker = _text(market.get("series_ticker"))
    event: dict[str, Any] = {}
    event_metadata: dict[str, Any] = {}
    series: dict[str, Any] = {}
    if event_ticker:
        event_response = client.get(
            candidate_id=candidate_id,
            venue="kalshi",
            resource="event",
            path=f"/events/{quote(event_ticker, safe='')}",
            params={"with_nested_markets": "true"},
        )
        responses.append(event_response)
        event_payload = _mapping(event_response.parsed)
        event = _mapping(event_payload.get("event")) or event_payload
        metadata_response = client.get(
            candidate_id=candidate_id,
            venue="kalshi",
            resource="event_metadata",
            path=f"/events/{quote(event_ticker, safe='')}/metadata",
        )
        responses.append(metadata_response)
        metadata_payload = _mapping(metadata_response.parsed)
        event_metadata = _mapping(metadata_payload.get("metadata")) or metadata_payload
    else:
        record["issues"].append("KALSHI_EVENT_TICKER_MISSING")
    if not series_ticker:
        series_ticker = _text(event.get("series_ticker"))
    if series_ticker:
        series_response = client.get(
            candidate_id=candidate_id,
            venue="kalshi",
            resource="series",
            path=f"/series/{quote(series_ticker, safe='')}",
        )
        responses.append(series_response)
        series_payload = _mapping(series_response.parsed)
        series = _mapping(series_payload.get("series")) or series_payload
    else:
        record["issues"].append("KALSHI_SERIES_TICKER_MISSING")

    record["identifiers"]["market"].append(_id("market", "ticker", ticker))
    if event_ticker:
        record["identifiers"]["event"].append(_id("market", "event_ticker", event_ticker))
    if series_ticker:
        record["identifiers"]["series"].append(_id("market_or_event", "series_ticker", series_ticker))
    yes_label = _text(market.get("yes_sub_title")) or _text(market.get("yes_title")) or "Yes"
    no_label = _text(market.get("no_sub_title")) or _text(market.get("no_title")) or "No"
    record["identifiers"]["outcome"] = [
        _id("market", "contract_side", "YES"),
        _id("market", "contract_side", "NO"),
    ]
    record["exact_text"]["market_titles"] = [value for value in (_text(market.get("title")),) if value is not None]
    record["exact_text"]["event_titles"] = [value for value in (_text(event.get("title")),) if value is not None]
    record["exact_text"]["subtitles"] = [
        value for value in (_text(market.get("subtitle")), _text(event.get("sub_title"))) if value is not None
    ]
    record["exact_text"]["questions"] = [value for value in (_text(market.get("title")),) if value is not None]
    record["exact_text"]["outcome_labels"] = [
        {"contract_side": "YES", "label": yes_label, "source": "market.yes_sub_title_or_contract_side"},
        {"contract_side": "NO", "label": no_label, "source": "market.no_sub_title_or_contract_side"},
    ]
    record["polarity"] = {
        "explicit": True,
        "yes": {"contract_side": "YES", "label": yes_label, "native_outcome_id": "YES"},
        "no": {"contract_side": "NO", "label": no_label, "native_outcome_id": "NO"},
        "source_fields": ["market.yes_sub_title", "market.no_sub_title", "Kalshi YES/NO contract sides"],
    }
    nested_markets = _sequence(event.get("markets"))
    record["market_structure"] = {
        "market_type": _text(market.get("market_type")) or _text(market.get("strike_type")),
        "binary": True,
        "multivariate": len(nested_markets) > 1,
        "mutually_exclusive": event.get("mutually_exclusive"),
        "negative_risk": None,
        "raw_flags": {
            "strike_type": market.get("strike_type"),
            "mutually_exclusive": event.get("mutually_exclusive"),
            "event_market_count": len(nested_markets),
        },
        "classification_basis": ["Kalshi YES/NO contract sides", "event.markets count"],
    }
    primary = _text(market.get("rules_primary"))
    secondary = _text(market.get("rules_secondary"))
    record["rules"] = {
        "primary": primary,
        "secondary": secondary,
        "additional": [],
        "source_fields": ["market.rules_primary", "market.rules_secondary"],
    }
    record["times"]["opening"] = _present_time_values("market", market, ("open_time",))
    record["times"]["closing"] = _present_time_values("market", market, ("close_time",))
    record["times"]["expiration"] = _present_time_values(
        "market",
        market,
        ("expiration_time", "expected_expiration_time", "latest_expiration_time"),
    )
    record["times"]["earliest_resolution"] = _present_time_values(
        "market", market, ("earliest_resolution_time", "earliest_expiration_time")
    )
    record["times"]["original_timezone_text"] = _timezone_mentions(primary, secondary)
    settlement_sources = _sequence(event_metadata.get("settlement_sources"))
    if not settlement_sources:
        settlement_sources = _sequence(series.get("settlement_sources"))
    source_text = [value for item in settlement_sources if (value := _text(_mapping(item).get("name"))) is not None]
    source_urls = _urls(settlement_sources, primary, secondary)
    authority = [
        {"source": "settlement_sources.name", "value": value, "claim_scope": "venue_resolution_source"}
        for value in source_text
    ]
    record["resolution"] = {
        "source_text": source_text,
        "source_urls": source_urls,
        "settlement_authority": authority,
        "oracle_process": {},
        "raw_fields": {
            "settlement_sources": settlement_sources,
            "settlement_timer_seconds": market.get("settlement_timer_seconds"),
        },
    }
    keyword_matches = _keyword_matches((primary, secondary))
    record["early_close"] = {
        "can_close_early": market.get("can_close_early"),
        "conditions": [value for value in (_text(market.get("early_close_condition")),) if value is not None],
        "rule_text_matches": keyword_matches["early_close"],
        "raw_fields": {
            "can_close_early": market.get("can_close_early"),
            "early_close_condition": market.get("early_close_condition"),
        },
    }
    record["exceptional_outcomes"] = {
        "void": keyword_matches["void"],
        "cancel": keyword_matches["cancel"],
        "refund": keyword_matches["refund"],
        "indeterminate": keyword_matches["indeterminate"],
        "raw_fields": {
            "void_cancel_rules": market.get("void_cancel_rules"),
            "result": market.get("result"),
        },
    }
    record["revisions"] = _revision_times(
        (("market", market), ("event", event), ("event_metadata", event_metadata), ("series", series))
    )
    record["fees"] = {
        "raw_fields": {
            "market_fee_type": market.get("fee_type"),
            "market_fee_multiplier": market.get("fee_multiplier"),
            "series_fee_type": series.get("fee_type"),
            "series_fee_multiplier": series.get("fee_multiplier"),
        }
    }
    record["response_evidence"] = [item.reference() for item in responses]
    record["issues"].extend(
        f"{item.resource.upper()}_REQUEST_{item.status}" for item in responses if item.status != "OK"
    )
    return _finalize_venue_record(record)


def _polymarket_record(
    candidate: Mapping[str, Any],
    *,
    side_name: str,
    side: Mapping[str, Any],
    client: PublicMetadataClient,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    record = _empty_venue_record(candidate_id, "polymarket", side_name)
    slug = _text(side.get("slug")) or ""
    if not _POLYMARKET_SLUG_RE.fullmatch(slug):
        record["issues"].append("POLYMARKET_SLUG_UNRESOLVED")
        return _finalize_venue_record(record)
    market_response = client.get(
        candidate_id=candidate_id,
        venue="polymarket",
        resource="market",
        path="/markets",
        params={"slug": slug},
    )
    responses = [market_response]
    payload = market_response.parsed
    records = payload if isinstance(payload, list) else _mapping(payload).get("markets", [])
    matches = [
        dict(item) for item in _sequence(records) if isinstance(item, Mapping) and _text(item.get("slug")) == slug
    ]
    if len(matches) != 1:
        record["issues"].append("POLYMARKET_MARKET_LOOKUP_NOT_UNIQUE")
        record["response_evidence"] = [item.reference() for item in responses]
        return _finalize_venue_record(record)
    market = matches[0]
    hinted_condition = _text(side.get("condition_id"))
    condition_id = _text(market.get("conditionId")) or _text(market.get("condition_id"))
    if hinted_condition and condition_id != hinted_condition:
        record["issues"].append("POLYMARKET_CONDITION_ID_MISMATCH")

    nested_events = [dict(item) for item in _sequence(market.get("events")) if isinstance(item, Mapping)]
    event_slug = _candidate_event_slug(side) or (
        _text(nested_events[0].get("slug")) if len(nested_events) == 1 else None
    )
    event: dict[str, Any] = nested_events[0] if len(nested_events) == 1 else {}
    if event_slug:
        event_response = client.get(
            candidate_id=candidate_id,
            venue="polymarket",
            resource="event",
            path="/events",
            params={"slug": event_slug},
        )
        responses.append(event_response)
        event_payload = event_response.parsed
        event_records = event_payload if isinstance(event_payload, list) else _mapping(event_payload).get("events", [])
        event_matches = [
            dict(item)
            for item in _sequence(event_records)
            if isinstance(item, Mapping) and _text(item.get("slug")) == event_slug
        ]
        if len(event_matches) == 1:
            event = event_matches[0]
        else:
            record["issues"].append("POLYMARKET_EVENT_LOOKUP_NOT_UNIQUE")
    else:
        record["issues"].append("POLYMARKET_EVENT_SLUG_MISSING")

    series_records = [dict(item) for item in _sequence(event.get("series")) if isinstance(item, Mapping)]
    market_id = _text(market.get("id"))
    event_id = _text(event.get("id"))
    question_id = _text(market.get("questionID")) or _text(market.get("questionId"))
    if market_id:
        record["identifiers"]["market"].append(_id("market", "id", market_id))
    if event_id:
        record["identifiers"]["event"].append(_id("event", "id", event_id))
    for series_item in series_records:
        value = _text(series_item.get("id")) or _text(series_item.get("slug"))
        if value:
            record["identifiers"]["series"].append(_id("event.series", "id_or_slug", value))
    if condition_id:
        record["identifiers"]["condition"].append(_id("market", "conditionId", condition_id))
    if question_id:
        record["identifiers"]["question"].append(_id("market", "questionID", question_id))
    labels = [_text(value) for value in _sequence(market.get("outcomes"))]
    tokens = [_text(value) for value in _sequence(market.get("clobTokenIds"))]
    valid_labels = [value for value in labels if value is not None]
    for index, label in enumerate(labels):
        if label is not None:
            record["identifiers"]["outcome"].append(_id("market.outcomes", str(index), label))
    if len(tokens) == len(labels):
        for index, token in enumerate(tokens):
            if token is not None:
                record["identifiers"]["token"].append(
                    {"source": "market.clobTokenIds", "field": str(index), "value": token, "label": labels[index]}
                )
    record["exact_text"]["market_titles"] = [value for value in (_text(market.get("title")),) if value is not None]
    record["exact_text"]["event_titles"] = [value for value in (_text(event.get("title")),) if value is not None]
    record["exact_text"]["questions"] = [value for value in (_text(market.get("question")),) if value is not None]
    record["exact_text"]["subtitles"] = []
    record["exact_text"]["outcome_labels"] = [
        {"position": index, "label": label, "source": "market.outcomes"} for index, label in enumerate(valid_labels)
    ]
    yes_index = next((index for index, label in enumerate(labels) if label and label.casefold() == "yes"), None)
    no_index = next((index for index, label in enumerate(labels) if label and label.casefold() == "no"), None)
    explicit_polarity = yes_index is not None and no_index is not None and yes_index != no_index
    record["polarity"] = {
        "explicit": explicit_polarity,
        "yes": (
            {
                "label": labels[yes_index],
                "position": yes_index,
                "native_outcome_id": tokens[yes_index] if len(tokens) == len(labels) else None,
            }
            if yes_index is not None
            else None
        ),
        "no": (
            {
                "label": labels[no_index],
                "position": no_index,
                "native_outcome_id": tokens[no_index] if len(tokens) == len(labels) else None,
            }
            if no_index is not None
            else None
        ),
        "source_fields": ["market.outcomes", "market.clobTokenIds"],
    }
    negative_risk = market.get("negRisk") if "negRisk" in market else event.get("negRisk")
    event_markets = _sequence(event.get("markets"))
    record["market_structure"] = {
        "market_type": _text(market.get("marketType")) or _text(market.get("feeType")),
        "binary": len(labels) == 2,
        "multivariate": len(event_markets) > 1 or negative_risk is True,
        "mutually_exclusive": event.get("mutuallyExclusive"),
        "negative_risk": negative_risk,
        "raw_flags": {
            "market_negRisk": market.get("negRisk"),
            "event_negRisk": event.get("negRisk"),
            "event_enableNegRisk": event.get("enableNegRisk"),
            "event_market_count": len(event_markets),
            "market_comboStatus": market.get("comboStatus"),
        },
        "classification_basis": ["market.outcomes count", "event.markets count", "native negative-risk flags"],
    }
    primary = _text(market.get("description"))
    secondary = _text(event.get("description"))
    record["rules"] = {
        "primary": primary,
        "secondary": secondary,
        "additional": [
            {"source": "market", "field": field, "value": market.get(field)}
            for field in ("resolutionSource", "umaResolutionStatuses")
            if field in market
        ],
        "source_fields": ["market.description", "event.description", "market.resolutionSource"],
    }
    record["times"]["opening"] = _present_time_values(
        "market", market, ("startDate", "startDateIso", "acceptingOrdersTimestamp")
    ) + _present_time_values("event", event, ("startDate", "startTime"))
    record["times"]["closing"] = _present_time_values("market", market, ("endDate", "endDateIso")) + (
        _present_time_values("event", event, ("endDate",))
    )
    record["times"]["expiration"] = _present_time_values("market", market, ("umaEndDate",))
    record["times"]["earliest_resolution"] = _present_time_values(
        "market", market, ("earliestResolutionTime", "earliest_resolution_time")
    )
    record["times"]["original_timezone_text"] = _timezone_mentions(primary, secondary)
    resolution_fields = {
        "market_resolutionSource": market.get("resolutionSource"),
        "event_resolutionSource": event.get("resolutionSource"),
    }
    source_text = [value for value in (_text(item) for item in resolution_fields.values()) if value]
    source_urls = _urls(primary, secondary, resolution_fields)
    authority_fields = {
        "market_resolvedBy": market.get("resolvedBy"),
        "market_submitted_by": market.get("submitted_by"),
    }
    authority = [
        {"source": field, "value": value, "claim_scope": "native_oracle_identifier"}
        for field, raw in authority_fields.items()
        if (value := _text(raw)) is not None
    ]
    oracle_fields = {
        field: market.get(field)
        for field in (
            "questionID",
            "resolvedBy",
            "umaBond",
            "umaReward",
            "umaResolutionStatuses",
            "umaEndDate",
        )
        if field in market
    }
    record["resolution"] = {
        "source_text": source_text,
        "source_urls": source_urls,
        "settlement_authority": authority,
        "oracle_process": {
            "classification": "UMA_METADATA_FIELDS_PRESENT"
            if any(key.startswith("uma") for key in oracle_fields)
            else None,
            "raw_fields": oracle_fields,
        },
        "raw_fields": resolution_fields,
    }
    keyword_matches = _keyword_matches((primary, secondary))
    record["early_close"] = {
        "can_close_early": market.get("canCloseEarly"),
        "conditions": [],
        "rule_text_matches": keyword_matches["early_close"],
        "raw_fields": {"canCloseEarly": market.get("canCloseEarly")},
    }
    record["exceptional_outcomes"] = {
        "void": keyword_matches["void"],
        "cancel": keyword_matches["cancel"],
        "refund": keyword_matches["refund"],
        "indeterminate": keyword_matches["indeterminate"],
        "raw_fields": {
            "market_voidCancelRules": market.get("voidCancelRules") or market.get("void_cancel_rules"),
            "event_voidCancelRules": event.get("voidCancelRules") or event.get("void_cancel_rules"),
        },
    }
    revision_records: list[tuple[str, Mapping[str, Any]]] = [("market", market), ("event", event)]
    revision_records.extend((f"event.series[{index}]", item) for index, item in enumerate(series_records))
    record["revisions"] = _revision_times(revision_records)
    record["fees"] = {
        "raw_fields": {
            "feeType": market.get("feeType"),
            "feeSchedule": market.get("feeSchedule"),
            "feesEnabled": market.get("feesEnabled"),
            "makerBaseFee": market.get("makerBaseFee"),
            "takerBaseFee": market.get("takerBaseFee"),
        }
    }
    record["response_evidence"] = [item.reference() for item in responses]
    record["issues"].extend(
        f"{item.resource.upper()}_REQUEST_{item.status}" for item in responses if item.status != "OK"
    )
    return _finalize_venue_record(record)


def capture_pair(candidate: Mapping[str, Any], client: PublicMetadataClient) -> dict[str, Any]:
    candidate_id = _text(candidate.get("candidate_id"))
    if candidate_id is None or _SAFE_ID_RE.fullmatch(candidate_id) is None:
        raise SemanticCaptureError("candidate ID is missing or invalid")
    kalshi_side_name, kalshi_side = _candidate_side(candidate, "kalshi")
    polymarket_side_name, polymarket_side = _candidate_side(candidate, "polymarket")
    kalshi = _kalshi_record(candidate, side_name=kalshi_side_name, side=kalshi_side, client=client)
    polymarket = _polymarket_record(candidate, side_name=polymarket_side_name, side=polymarket_side, client=client)
    missing = {
        "kalshi": kalshi["missing_requirements"],
        "polymarket": polymarket["missing_requirements"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "venue_native_semantic_equivalence_pair",
        "pair_id": candidate_id,
        "source_candidate": {
            "snapshot_id": candidate.get("snapshot_id"),
            "cluster_id": candidate.get("cluster_id"),
            "candidate_id": candidate_id,
            "relation": candidate.get("relation"),
            "relation_confidence": candidate.get("relation_confidence"),
            "candidate_line_sha256": _sha256_bytes(_canonical_json_bytes(candidate)),
        },
        "venues": {"kalshi": kalshi, "polymarket": polymarket},
        "coverage": {
            "complete": not missing["kalshi"] and not missing["polymarket"],
            "missing_by_venue": missing,
        },
        "semantic_decision": "PENDING_SEMANTIC_REVIEW",
        "live_eligible": False,
    }


def _load_candidates(path: Path, *, expected_sha256: str, expected_count: int) -> tuple[bytes, list[dict[str, Any]]]:
    payload = path.read_bytes()
    if _sha256_bytes(payload) != expected_sha256:
        raise SemanticCaptureError("candidate input hash mismatch")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = _strict_json_bytes(line)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, _DuplicateKey) as exc:
            raise SemanticCaptureError(f"candidate line {line_number} is invalid strict JSON") from exc
        if not isinstance(value, dict):
            raise SemanticCaptureError(f"candidate line {line_number} is not an object")
        rows.append(value)
    if len(rows) != expected_count:
        raise SemanticCaptureError("candidate count does not match frozen protocol")
    candidate_ids = [row.get("candidate_id") for row in rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise SemanticCaptureError("candidate IDs are duplicated")
    return payload, rows


def _authenticate_freeze(repository_root: Path, freeze_path: Path, expected_sha256: str) -> dict[str, Any]:
    if _sha256_path(freeze_path) != expected_sha256:
        raise SemanticCaptureError("implementation freeze hash mismatch")
    document = _strict_json_bytes(freeze_path.read_bytes())
    if not isinstance(document, dict) or document.get("protocol_id") != PROTOCOL_ID:
        raise SemanticCaptureError("implementation freeze identity mismatch")
    for binding in _sequence(document.get("files")):
        item = _mapping(binding)
        relative = _text(item.get("path"))
        expected = _text(item.get("sha256"))
        if not relative or not expected or _sha256_path(repository_root / relative) != expected:
            raise SemanticCaptureError(f"frozen file mismatch: {relative}")
    return document


def _artifact_binding(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def run_capture(
    *,
    repository_root: Path,
    expected_freeze_sha256: str,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    candidate_path: Path = DEFAULT_CANDIDATE_PATH,
    candidate_sha256: str = DEFAULT_CANDIDATE_SHA256,
    expected_pair_count: int = EXPECTED_PAIR_COUNT,
    network_authorized: bool,
    kalshi_transport: httpx.BaseTransport | None = None,
    polymarket_transport: httpx.BaseTransport | None = None,
    utc_now_ns: Callable[[], int] = time.time_ns,
    monotonic_now_ns: Callable[[], int] = time.perf_counter_ns,
) -> Path:
    """Reject every legacy capture attempt before transport or artifact creation."""

    raise SemanticCaptureError(_QUARANTINE_MESSAGE)

    # The frozen implementation below remains as historical reference only.
    # The unconditional quarantine above has no fixture bypass.
    if network_authorized is not True:  # pragma: no cover
        raise SemanticCaptureError("explicit public-metadata network authorization is required")
    root = repository_root.resolve(strict=True)
    absolute_candidate_path = root / candidate_path
    absolute_output_path = root / output_path
    if (
        absolute_output_path.exists()
        or absolute_output_path.with_name(f"{absolute_output_path.name}.inprogress").exists()
    ):
        raise SemanticCaptureError("single-use output path already exists")
    _authenticate_freeze(root, root / DEFAULT_FREEZE_PATH, expected_freeze_sha256)
    candidate_bytes, candidates = _load_candidates(
        absolute_candidate_path,
        expected_sha256=candidate_sha256,
        expected_count=expected_pair_count,
    )
    writer = CaptureWriter(absolute_output_path)
    try:
        writer.write("input_candidates.jsonl", candidate_bytes)
        pairs: list[dict[str, Any]] = []
        with PublicMetadataClient(
            writer=writer,
            kalshi_transport=kalshi_transport,
            polymarket_transport=polymarket_transport,
            utc_now_ns=utc_now_ns,
            monotonic_now_ns=monotonic_now_ns,
        ) as client:
            for candidate in candidates:
                start_ordinal = writer.ordinal
                pair = capture_pair(candidate, client)
                if writer.ordinal - start_ordinal > MAX_REQUESTS_PER_PAIR:
                    raise SemanticCaptureError("per-pair request bound exceeded")
                pairs.append(pair)
        pair_bytes = b"".join(_canonical_json_bytes(pair) for pair in pairs)
        writer.write("venue_native_pairs.jsonl", pair_bytes)
        total_venue_records = expected_pair_count * 2
        complete_venue_records = sum(
            venue["status"] == "COMPLETE" for pair in pairs for venue in pair["venues"].values()
        )
        complete_pairs = sum(pair["coverage"]["complete"] for pair in pairs)
        failed_requests = sum(record["status"] != "OK" for record in writer.request_records)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE" if complete_pairs == expected_pair_count else "PARTIAL_NATIVE_METADATA_COVERAGE",
            "input_pair_count": expected_pair_count,
            "venue_record_count": total_venue_records,
            "complete_venue_record_count": complete_venue_records,
            "complete_pair_count": complete_pairs,
            "request_count": len(writer.request_records),
            "failed_request_count": failed_requests,
            "retry_count": 0,
            "network_hosts": ["api.elections.kalshi.com", "gamma-api.polymarket.com"],
            "credentials_read": False,
            "environment_files_read": False,
            "books_or_trades_requested": False,
            "orders_submitted": 0,
            "economics_computed": False,
            "live_eligible": False,
        }
        writer.write("capture_summary.json", _canonical_json_bytes(summary))
        files = [
            _artifact_binding(path, writer.staging_path)
            for path in sorted(writer.staging_path.rglob("*"))
            if path.is_file()
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "run_id": output_path.name,
            "status": summary["status"],
            "repository_head": _git_head(root),
            "implementation_freeze_sha256": expected_freeze_sha256,
            "input_candidate_path": candidate_path.as_posix(),
            "input_candidate_sha256": candidate_sha256,
            "input_pair_count": expected_pair_count,
            "request_count": len(writer.request_records),
            "maximum_request_count": expected_pair_count * MAX_REQUESTS_PER_PAIR,
            "failed_request_count": failed_requests,
            "retry_count": 0,
            "raw_body_semantics": "HTTP entity-body bytes before JSON decoding; Accept-Encoding identity required",
            "response_headers_semantics": (
                "ordered allowlisted provenance-header name/value byte pairs retained as base64 and latin-1"
            ),
            "normalization_semantics": "missing native fields remain null or explicit missing requirements; no cross-venue inference",
            "authority": {
                "research_only": True,
                "public_metadata_get_only": True,
                "credential_accessed": False,
                "environment_file_accessed": False,
                "account_accessed": False,
                "books_or_trades_requested": False,
                "orders_submitted": 0,
                "fills_inferred": False,
                "economics_computed": False,
                "live_eligible": False,
            },
            "files": files,
            "sealed_at_utc_ns": utc_now_ns(),
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        writer.write("manifest.json", manifest_bytes)
        writer.write("manifest.sha256", f"{_sha256_bytes(manifest_bytes)}  manifest.json\n".encode("ascii"))
        writer.seal()
    except Exception as exc:
        failure_path = writer.staging_path / "failure.json"
        if not failure_path.exists():
            _write_exclusive(
                failure_path,
                _canonical_json_bytes(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "CAPTURE_FAILED_NO_RETRY",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retry_count": 0,
                        "orders_submitted": 0,
                    }
                ),
            )
        raise
    return absolute_output_path


def _git_head(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise SemanticCaptureError("repository HEAD unavailable")
    head = completed.stdout.decode("ascii", errors="strict").strip()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise SemanticCaptureError("repository HEAD invalid")
    return head


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(_QUARANTINE_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LEGACY_CAPTURE_QUARANTINED",
    "PRODUCTION_READY",
    "REAL_NETWORK_CAPTURE_ENABLED",
    "SemanticCaptureError",
]
