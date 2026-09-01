"""PROTOTYPE ONLY: offline experiments for one PMXT Router request.

This isolated module is not production-ready and is not imported or exported
by the PMXT sync, monitor, native-continuation, or rule-evidence workflows.  Its
permit and receipt checks are local assertions only: they do not grant network
or live authorization, independently establish billing or zero-dollar status,
control an HTTP transport at most once, or prove crash-safe state transitions.

The module deliberately has no HTTP client, credential loader, environment
access, or market-data reader.  Its current behavior is retained solely as a
testable prototype; callers must not treat a returned claim or receipt as an
operational go/no-go decision or as evidence that a request occurred exactly
once.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

PROTOTYPE_ONLY = True
PRODUCTION_READY = False

PERMIT_SCHEMA_VERSION = 1
PROTOCOL_ID = "pmxt-router-at-most-once-v1"
OFFICIAL_PMXT_ORIGIN = "https://api.pmxt.dev"
OFFICIAL_PMXT_PATH = "/v0/matched-market-clusters"
CONTROL_NAMESPACE = "pmxt-router-at-most-once-v1"

MAXIMUM_PERMIT_BYTES = 65_536
MAXIMUM_RESPONSE_BYTES_CEILING = 5_000_000
MAXIMUM_FANOUT = 25

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_REASON_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_QUERY_KEYS = {
    "relation",
    "minConfidence",
    "minVenues",
    "includeRawMatches",
    "sort",
    "venues",
    "limit",
    "offset",
}
_BOUND_KEYS = {
    "maximum_response_bytes",
    "maximum_clusters",
    "maximum_candidates",
    "maximum_markets_per_cluster",
    "maximum_raw_matches_per_cluster",
}
_REQUEST_KEYS = {
    "method",
    "origin",
    "path",
    "query",
    "max_attempts",
    "retries",
    "redirects",
    "timeout_seconds",
    "bounds",
}
_ZERO_SPEND_KEYS = {
    "provider",
    "authority",
    "provider_authoritative",
    "evidence_id",
    "evidence_sha256",
    "account_scope_sha256",
    "observed_at_utc",
    "valid_until_utc",
    "plan",
    "currency",
    "incremental_charge_usd",
    "credits_required",
    "credits_remaining",
    "overage_enabled",
}
_AUTHORITY_KEYS = {
    "purpose",
    "network_read_authorized",
    "maximum_pmxt_requests",
    "orders_authorized",
    "venue_account_access_authorized",
    "credential_persistence_authorized",
    "live_execution_authorized",
}
_PERMIT_KEYS = {
    "schema_version",
    "protocol_id",
    "issued_at_utc",
    "expires_at_utc",
    "control_root",
    "request",
    "zero_spend_evidence",
    "authority",
}
_TERMINAL_STATUSES = {
    "COMPLETED",
    "RESPONSE_REJECTED",
    "TRANSPORT_FAILED",
    "ABORTED_BEFORE_TRANSPORT",
}


class AcquisitionControlError(RuntimeError):
    """Base class for local PMXT acquisition-control failures."""


class PermitValidationError(AcquisitionControlError, ValueError):
    """The independently authenticated permit is invalid or out of date."""


class AcquisitionStateError(AcquisitionControlError):
    """The fixed control path is already claimed, inconsistent, or unsafe."""


class ReceiptValidationError(AcquisitionControlError, ValueError):
    """A proposed immutable receipt violates the acquisition contract."""


@dataclass(frozen=True)
class AuthenticatedAcquisitionPermit:
    """Validated permit bytes plus the fields needed by the local gate."""

    path: Path
    raw_bytes: bytes
    permit_sha256: str
    canonical_sha256: str
    request_sha256: str
    issued_at_utc: datetime
    expires_at_utc: datetime
    zero_spend_valid_until_utc: datetime
    control_root: Path
    query_items: tuple[tuple[str, str], ...]
    maximum_response_bytes: int
    maximum_clusters: int
    maximum_candidates: int
    maximum_markets_per_cluster: int
    maximum_raw_matches_per_cluster: int
    document: Mapping[str, Any]

    def query_params(self) -> dict[str, str]:
        """Return a fresh, credential-free copy of the bound query."""

        return dict(self.query_items)


@dataclass(frozen=True)
class AcquisitionControlPaths:
    """The only control paths allowed for one permit digest."""

    directory: Path
    claim: Path
    http_receipt: Path
    terminal_receipt: Path


@dataclass(frozen=True)
class AcquisitionClaim:
    """A durable at-most-once claim returned only after file fsync."""

    permit_sha256: str
    permit_canonical_sha256: str
    request_sha256: str
    claimed_at_utc: datetime
    expires_at_utc: datetime
    zero_spend_valid_until_utc: datetime
    maximum_response_bytes: int
    paths: AcquisitionControlPaths
    claim_sha256: str
    claim_evidence_sha256: str


@dataclass(frozen=True)
class ImmutableReceipt:
    """Digest identity for one exclusively created receipt."""

    kind: str
    path: Path
    sha256: str
    evidence_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact, deterministic UTF-8 JSON without a trailing newline."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not finite, canonical JSON") from exc


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value independently of object-key order and whitespace."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def evidence_sha256(kind: str, value: Any) -> str:
    """Return a domain-separated stable hash for an evidence document."""

    if not isinstance(kind, str) or _EVIDENCE_ID_RE.fullmatch(kind) is None:
        raise ValueError("evidence kind must be a bounded safe identifier")
    domain = f"{PROTOCOL_ID}:{kind}\n".encode("ascii")
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_json(raw: bytes, *, label: str, error_type: type[AcquisitionControlError]) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise error_type(f"{label} is not strict JSON") from exc


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PermitValidationError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise PermitValidationError(f"{label} fields changed")


def _valid_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PermitValidationError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _utc_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value or value.strip() != value or not value.endswith("Z"):
        raise PermitValidationError(f"{label} must be an explicit UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PermitValidationError(f"{label} is not a valid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PermitValidationError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _runtime_utc(value: datetime | None, label: str) -> datetime:
    resolved = datetime.now(timezone.utc) if value is None else value
    if not isinstance(resolved, datetime) or resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    return resolved.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _positive_int(value: Any, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise PermitValidationError(f"{label} must be an integer between 1 and {maximum}")
    return value


def _query_int(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise PermitValidationError(f"{label} must be a canonical decimal query string")
    if value != str(int(value)):
        raise PermitValidationError(f"{label} must not contain leading zeros")
    return int(value)


def _validate_query(query: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    _exact_keys(query, _QUERY_KEYS, "request.query")
    if any(not isinstance(value, str) for value in query.values()):
        raise PermitValidationError("every request.query value must be a string")
    if query.get("relation") != "identity":
        raise PermitValidationError("request.query must use identity relation")
    if query.get("minVenues") != "2":
        raise PermitValidationError("request.query minVenues must be exactly 2")
    if query.get("includeRawMatches") != "true":
        raise PermitValidationError("request.query must include raw matches")
    if query.get("sort") != "volume":
        raise PermitValidationError("request.query sort must be volume")
    if query.get("venues") != "kalshi,polymarket":
        raise PermitValidationError("request.query venues must be exactly kalshi,polymarket")
    if _query_int(query.get("offset"), "request.query.offset") != 0:
        raise PermitValidationError("request.query offset must be zero")
    limit = _query_int(query.get("limit"), "request.query.limit")
    if not 1 <= limit <= MAXIMUM_FANOUT:
        raise PermitValidationError("request.query limit must be between 1 and 25")
    confidence_text = query.get("minConfidence")
    try:
        confidence = float(confidence_text)
    except (TypeError, ValueError) as exc:
        raise PermitValidationError("request.query minConfidence must be numeric") from exc
    if not math.isfinite(confidence) or not 0.80 <= confidence <= 1.0:
        raise PermitValidationError("request.query minConfidence must be between 0.80 and 1")
    return tuple(sorted((key, str(value)) for key, value in query.items()))


def _validate_permit(document: Mapping[str, Any], *, now_utc: datetime) -> dict[str, Any]:
    _exact_keys(document, _PERMIT_KEYS, "permit")
    schema_version = document.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != PERMIT_SCHEMA_VERSION:
        raise PermitValidationError("permit schema_version changed")
    if document.get("protocol_id") != PROTOCOL_ID:
        raise PermitValidationError("permit protocol_id changed")

    issued_at = _utc_datetime(document.get("issued_at_utc"), "issued_at_utc")
    expires_at = _utc_datetime(document.get("expires_at_utc"), "expires_at_utc")
    if issued_at >= expires_at:
        raise PermitValidationError("permit expiry must be after issuance")
    if now_utc < issued_at:
        raise PermitValidationError("permit is not valid yet")
    if now_utc >= expires_at:
        raise PermitValidationError("permit has expired")

    control_root_value = document.get("control_root")
    if (
        not isinstance(control_root_value, str)
        or not control_root_value
        or control_root_value.strip() != control_root_value
    ):
        raise PermitValidationError("control_root must be a non-empty absolute path")
    control_root_input = Path(control_root_value)
    if not control_root_input.is_absolute() or ".." in control_root_input.parts:
        raise PermitValidationError("control_root must be an absolute non-traversing path")
    control_root = control_root_input.resolve(strict=False)
    if control_root == Path(control_root.anchor):
        raise PermitValidationError("control_root must not be a filesystem root")

    request = _mapping(document.get("request"), "request")
    _exact_keys(request, _REQUEST_KEYS, "request")
    if request.get("method") != "GET":
        raise PermitValidationError("request method must be GET")
    if request.get("origin") != OFFICIAL_PMXT_ORIGIN or request.get("path") != OFFICIAL_PMXT_PATH:
        raise PermitValidationError("request must target the official PMXT Router path")
    if request.get("max_attempts") != 1 or isinstance(request.get("max_attempts"), bool):
        raise PermitValidationError("request max_attempts must be exactly one")
    if request.get("retries") != 0 or isinstance(request.get("retries"), bool):
        raise PermitValidationError("request retries must be zero")
    if request.get("redirects") is not False:
        raise PermitValidationError("request redirects must be false")
    timeout = request.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise PermitValidationError("request timeout_seconds must be numeric")
    if not math.isfinite(float(timeout)) or not 0 < float(timeout) <= 60:
        raise PermitValidationError("request timeout_seconds must be in (0, 60]")

    query = _mapping(request.get("query"), "request.query")
    query_items = _validate_query(query)
    query_limit = _query_int(query["limit"], "request.query.limit")

    bounds = _mapping(request.get("bounds"), "request.bounds")
    _exact_keys(bounds, _BOUND_KEYS, "request.bounds")
    maximum_response_bytes = _positive_int(
        bounds.get("maximum_response_bytes"),
        "request.bounds.maximum_response_bytes",
        maximum=MAXIMUM_RESPONSE_BYTES_CEILING,
    )
    maximum_clusters = _positive_int(
        bounds.get("maximum_clusters"), "request.bounds.maximum_clusters", maximum=MAXIMUM_FANOUT
    )
    if maximum_clusters != query_limit:
        raise PermitValidationError("maximum_clusters must equal the bound query limit")
    maximum_candidates = _positive_int(
        bounds.get("maximum_candidates"), "request.bounds.maximum_candidates", maximum=MAXIMUM_FANOUT
    )
    maximum_markets_per_cluster = _positive_int(
        bounds.get("maximum_markets_per_cluster"),
        "request.bounds.maximum_markets_per_cluster",
        maximum=MAXIMUM_FANOUT,
    )
    maximum_raw_matches_per_cluster = _positive_int(
        bounds.get("maximum_raw_matches_per_cluster"),
        "request.bounds.maximum_raw_matches_per_cluster",
        maximum=MAXIMUM_FANOUT,
    )

    zero_spend = _mapping(document.get("zero_spend_evidence"), "zero_spend_evidence")
    _exact_keys(zero_spend, _ZERO_SPEND_KEYS, "zero_spend_evidence")
    if (
        zero_spend.get("provider") != "PMXT"
        or zero_spend.get("authority") != "PMXT_PROVIDER_BILLING"
        or zero_spend.get("provider_authoritative") is not True
    ):
        raise PermitValidationError("zero-spend evidence must be explicitly provider-authoritative")
    evidence_id = zero_spend.get("evidence_id")
    if not isinstance(evidence_id, str) or _EVIDENCE_ID_RE.fullmatch(evidence_id) is None:
        raise PermitValidationError("zero-spend evidence_id is invalid")
    _valid_sha256(zero_spend.get("evidence_sha256"), "zero_spend_evidence.evidence_sha256")
    _valid_sha256(zero_spend.get("account_scope_sha256"), "zero_spend_evidence.account_scope_sha256")
    observed_at = _utc_datetime(zero_spend.get("observed_at_utc"), "zero_spend_evidence.observed_at_utc")
    valid_until = _utc_datetime(zero_spend.get("valid_until_utc"), "zero_spend_evidence.valid_until_utc")
    if observed_at > issued_at:
        raise PermitValidationError("zero-spend evidence must predate permit issuance")
    if valid_until < expires_at or now_utc >= valid_until:
        raise PermitValidationError("zero-spend evidence is not valid for the complete permit window")
    if (
        zero_spend.get("plan") != "FREE"
        or zero_spend.get("currency") != "USD"
        or zero_spend.get("incremental_charge_usd") != "0.00"
        or zero_spend.get("overage_enabled") is not False
    ):
        raise PermitValidationError("provider evidence does not establish a zero-dollar request")
    if zero_spend.get("credits_required") != 1 or isinstance(zero_spend.get("credits_required"), bool):
        raise PermitValidationError("exactly one included credit must be required")
    credits_remaining = zero_spend.get("credits_remaining")
    if isinstance(credits_remaining, bool) or not isinstance(credits_remaining, int) or credits_remaining < 1:
        raise PermitValidationError("provider evidence must show at least one included credit remaining")

    authority = _mapping(document.get("authority"), "authority")
    _exact_keys(authority, _AUTHORITY_KEYS, "authority")
    if authority != {
        "purpose": "RESEARCH_ONLY_DISCOVERY",
        "network_read_authorized": True,
        "maximum_pmxt_requests": 1,
        "orders_authorized": False,
        "venue_account_access_authorized": False,
        "credential_persistence_authorized": False,
        "live_execution_authorized": False,
    }:
        raise PermitValidationError("authority boundary changed")

    return {
        "issued_at": issued_at,
        "expires_at": expires_at,
        "valid_until": valid_until,
        "control_root": control_root,
        "query_items": query_items,
        "maximum_response_bytes": maximum_response_bytes,
        "maximum_clusters": maximum_clusters,
        "maximum_candidates": maximum_candidates,
        "maximum_markets_per_cluster": maximum_markets_per_cluster,
        "maximum_raw_matches_per_cluster": maximum_raw_matches_per_cluster,
        "request_sha256": evidence_sha256("request", request),
    }


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _verify_permit_source(permit: AuthenticatedAcquisitionPermit) -> None:
    if permit.path.is_symlink():
        raise PermitValidationError("permit became a symlink after authentication")
    try:
        current = permit.path.read_bytes()
    except OSError as exc:
        raise PermitValidationError("permit became unreadable after authentication") from exc
    if current != permit.raw_bytes or hashlib.sha256(current).hexdigest() != permit.permit_sha256:
        raise PermitValidationError("permit bytes changed after authentication")


def load_acquisition_permit(
    path: Path | str,
    *,
    expected_sha256: str,
    now_utc: datetime | None = None,
) -> AuthenticatedAcquisitionPermit:
    """Authenticate exact permit bytes, then strictly validate every field."""

    expected = _valid_sha256(expected_sha256, "expected_sha256")
    source = Path(path)
    if source.is_symlink():
        raise PermitValidationError("permit must not be a symlink")
    try:
        resolved = source.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise PermitValidationError("permit is missing or unreadable") from exc
    if not resolved.is_file():
        raise PermitValidationError("permit must be a regular file")
    if stat.st_size <= 0 or stat.st_size > MAXIMUM_PERMIT_BYTES:
        raise PermitValidationError("permit byte size is outside the fixed bound")
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise PermitValidationError("permit is unreadable") from exc
    if len(raw) != stat.st_size or len(raw) > MAXIMUM_PERMIT_BYTES:
        raise PermitValidationError("permit changed while it was being authenticated")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise PermitValidationError("permit SHA-256 mismatch")
    parsed = _strict_json(raw, label="permit", error_type=PermitValidationError)
    if not isinstance(parsed, dict):
        raise PermitValidationError("permit must be a JSON object")
    runtime_now = _runtime_utc(now_utc, "now_utc")
    validated = _validate_permit(parsed, now_utc=runtime_now)
    return AuthenticatedAcquisitionPermit(
        path=resolved,
        raw_bytes=raw,
        permit_sha256=observed,
        canonical_sha256=canonical_json_sha256(parsed),
        request_sha256=validated["request_sha256"],
        issued_at_utc=validated["issued_at"],
        expires_at_utc=validated["expires_at"],
        zero_spend_valid_until_utc=validated["valid_until"],
        control_root=validated["control_root"],
        query_items=validated["query_items"],
        maximum_response_bytes=validated["maximum_response_bytes"],
        maximum_clusters=validated["maximum_clusters"],
        maximum_candidates=validated["maximum_candidates"],
        maximum_markets_per_cluster=validated["maximum_markets_per_cluster"],
        maximum_raw_matches_per_cluster=validated["maximum_raw_matches_per_cluster"],
        document=_freeze_json(parsed),
    )


def _control_namespace(control_root: Path | str, *, create: bool) -> Path:
    root_input = Path(control_root)
    if root_input.is_symlink():
        raise AcquisitionStateError("control_root must not be a symlink")
    try:
        root = root_input.resolve(strict=True)
    except OSError as exc:
        raise AcquisitionStateError("control_root must be a pre-existing directory") from exc
    if not root.is_dir():
        raise AcquisitionStateError("control_root must be a pre-existing directory")
    namespace_input = root / CONTROL_NAMESPACE
    if create:
        try:
            namespace_input.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise AcquisitionStateError("control namespace could not be created") from exc
    if namespace_input.is_symlink():
        raise AcquisitionStateError("control namespace must not be a symlink")
    if not namespace_input.exists() and not create:
        return namespace_input
    try:
        namespace = namespace_input.resolve(strict=True)
    except OSError as exc:
        raise AcquisitionStateError("control namespace is missing or inaccessible") from exc
    if namespace.parent != root or not namespace.is_dir():
        raise AcquisitionStateError("control namespace escaped the fixed control root")
    return namespace


def control_paths(control_root: Path | str, permit_sha256: str) -> AcquisitionControlPaths:
    """Return the deterministic digest-keyed paths without creating a claim."""

    digest = _valid_sha256(permit_sha256, "permit_sha256")
    namespace = _control_namespace(control_root, create=False)
    directory = namespace / digest
    return AcquisitionControlPaths(
        directory=directory,
        claim=directory / "claim.json",
        http_receipt=directory / "http_receipt.json",
        terminal_receipt=directory / "terminal_receipt.json",
    )


def _flush_and_fsync(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_durable(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            written = handle.write(payload)
            if written != len(payload):
                raise OSError("short evidence write")
            _flush_and_fsync(handle)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise AcquisitionStateError(f"refusing to overwrite immutable {path.name}") from exc
    except OSError as exc:
        # Never unlink here.  A partial exclusive file is crash residue and must
        # continue to block a retry under the same permit digest.
        raise AcquisitionStateError(f"durable exclusive write failed for {path.name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _sealed_document(kind: str, document: Mapping[str, Any]) -> tuple[dict[str, Any], bytes, str]:
    sealed = dict(document)
    digest = evidence_sha256(kind, sealed)
    sealed["evidence_sha256"] = digest
    return sealed, canonical_json_bytes(sealed) + b"\n", digest


def claim_acquisition(
    permit: AuthenticatedAcquisitionPermit,
    *,
    claimed_at_utc: datetime | None = None,
) -> AcquisitionClaim:
    """Durably reserve the permit digest exactly once before transport."""

    claimed_at = _runtime_utc(claimed_at_utc, "claimed_at_utc")
    if claimed_at < permit.issued_at_utc:
        raise PermitValidationError("permit is not valid at claim time")
    if claimed_at >= permit.expires_at_utc or claimed_at >= permit.zero_spend_valid_until_utc:
        raise PermitValidationError("permit or zero-spend evidence expired before claim")

    # Close the load-to-claim TOCTOU window before creating any control state.
    _verify_permit_source(permit)

    namespace = _control_namespace(permit.control_root, create=True)
    paths = AcquisitionControlPaths(
        directory=namespace / permit.permit_sha256,
        claim=namespace / permit.permit_sha256 / "claim.json",
        http_receipt=namespace / permit.permit_sha256 / "http_receipt.json",
        terminal_receipt=namespace / permit.permit_sha256 / "terminal_receipt.json",
    )
    try:
        paths.directory.mkdir(mode=0o700)
        _fsync_directory(namespace)
    except FileExistsError as exc:
        raise AcquisitionStateError("permit digest is already claimed or has crash residue") from exc
    except OSError as exc:
        raise AcquisitionStateError("exclusive permit claim could not be reserved") from exc

    claim_document = {
        "schema_version": 1,
        "kind": "PMXT_ACQUISITION_CLAIM",
        "protocol_id": PROTOCOL_ID,
        "permit_sha256": permit.permit_sha256,
        "permit_canonical_sha256": permit.canonical_sha256,
        "request_sha256": permit.request_sha256,
        "claimed_at_utc": _format_utc(claimed_at),
        "at_most_once": True,
        "file_fsync_required_before_return": True,
        "credential_material_persisted": False,
        "orders_authorized": False,
        "live_execution_authorized": False,
    }
    _, claim_bytes, claim_evidence = _sealed_document("claim", claim_document)
    _write_exclusive_durable(paths.claim, claim_bytes)
    claim_hash = hashlib.sha256(claim_bytes).hexdigest()

    # Residual receipts racing with claim creation invalidate the run before a
    # claim object is returned to any transport-capable caller.
    if paths.http_receipt.exists() or paths.terminal_receipt.exists():
        raise AcquisitionStateError("receipt residue exists at a newly claimed control path")
    return AcquisitionClaim(
        permit_sha256=permit.permit_sha256,
        permit_canonical_sha256=permit.canonical_sha256,
        request_sha256=permit.request_sha256,
        claimed_at_utc=claimed_at,
        expires_at_utc=permit.expires_at_utc,
        zero_spend_valid_until_utc=permit.zero_spend_valid_until_utc,
        maximum_response_bytes=permit.maximum_response_bytes,
        paths=paths,
        claim_sha256=claim_hash,
        claim_evidence_sha256=claim_evidence,
    )


def _load_sealed(path: Path, *, expected_kind: str, error_label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise AcquisitionStateError(f"{error_label} must not be a symlink")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AcquisitionStateError(f"{error_label} is missing or unreadable") from exc
    parsed = _strict_json(raw, label=error_label, error_type=AcquisitionStateError)
    if not isinstance(parsed, dict) or parsed.get("kind") != expected_kind:
        raise AcquisitionStateError(f"{error_label} identity changed")
    sealed_digest = parsed.get("evidence_sha256")
    if not isinstance(sealed_digest, str) or _SHA256_RE.fullmatch(sealed_digest) is None:
        raise AcquisitionStateError(f"{error_label} evidence hash is invalid")
    unsealed = dict(parsed)
    del unsealed["evidence_sha256"]
    hash_kind = {
        "PMXT_ACQUISITION_CLAIM": "claim",
        "PMXT_HTTP_RECEIPT": "http-receipt",
        "PMXT_TERMINAL_RECEIPT": "terminal-receipt",
    }[expected_kind]
    if evidence_sha256(hash_kind, unsealed) != sealed_digest:
        raise AcquisitionStateError(f"{error_label} evidence hash mismatch")
    if canonical_json_bytes(parsed) + b"\n" != raw:
        raise AcquisitionStateError(f"{error_label} is not canonical immutable evidence")
    return parsed, raw


def _verify_claim(claim: AcquisitionClaim) -> None:
    parsed, raw = _load_sealed(
        claim.paths.claim,
        expected_kind="PMXT_ACQUISITION_CLAIM",
        error_label="claim",
    )
    if hashlib.sha256(raw).hexdigest() != claim.claim_sha256:
        raise AcquisitionStateError("claim bytes changed")
    if (
        parsed.get("permit_sha256") != claim.permit_sha256
        or parsed.get("permit_canonical_sha256") != claim.permit_canonical_sha256
        or parsed.get("request_sha256") != claim.request_sha256
        or parsed.get("evidence_sha256") != claim.claim_evidence_sha256
    ):
        raise AcquisitionStateError("claim binding changed")


def write_http_receipt(
    claim: AcquisitionClaim,
    *,
    request_started_at_utc: datetime,
    response_received_at_utc: datetime,
    status_code: int,
    response_byte_count: int,
    response_sha256: str,
) -> ImmutableReceipt:
    """Write one metadata-only HTTP receipt; response or auth bytes are absent."""

    _verify_claim(claim)
    if claim.paths.terminal_receipt.exists() or claim.paths.terminal_receipt.is_symlink():
        raise AcquisitionStateError("terminal receipt already exists")
    started = _runtime_utc(request_started_at_utc, "request_started_at_utc")
    received = _runtime_utc(response_received_at_utc, "response_received_at_utc")
    if started < claim.claimed_at_utc or received < started:
        raise ReceiptValidationError("HTTP receipt timestamps are out of order")
    if started >= claim.expires_at_utc or started >= claim.zero_spend_valid_until_utc:
        raise ReceiptValidationError("transport started after the permit or zero-spend evidence expired")
    if isinstance(status_code, bool) or not isinstance(status_code, int) or not 200 <= status_code <= 599:
        raise ReceiptValidationError("status_code must be an integer between 200 and 599")
    if (
        isinstance(response_byte_count, bool)
        or not isinstance(response_byte_count, int)
        or not 0 <= response_byte_count <= claim.maximum_response_bytes
    ):
        raise ReceiptValidationError("response byte count exceeds the permit bound")
    if not isinstance(response_sha256, str) or _SHA256_RE.fullmatch(response_sha256) is None:
        raise ReceiptValidationError("response_sha256 must be lowercase hexadecimal")
    if response_byte_count == 0 and response_sha256 != hashlib.sha256(b"").hexdigest():
        raise ReceiptValidationError("an empty response must use the SHA-256 of empty bytes")

    receipt_document = {
        "schema_version": 1,
        "kind": "PMXT_HTTP_RECEIPT",
        "protocol_id": PROTOCOL_ID,
        "permit_sha256": claim.permit_sha256,
        "claim_sha256": claim.claim_sha256,
        "request_sha256": claim.request_sha256,
        "method": "GET",
        "origin": OFFICIAL_PMXT_ORIGIN,
        "path": OFFICIAL_PMXT_PATH,
        "attempt_number": 1,
        "redirect_count": 0,
        "request_started_at_utc": _format_utc(started),
        "response_received_at_utc": _format_utc(received),
        "status_code": status_code,
        "response_byte_count": response_byte_count,
        "response_sha256": response_sha256,
        "response_body_persisted": False,
        "request_headers_persisted": False,
        "credential_material_persisted": False,
    }
    _, payload, receipt_evidence = _sealed_document("http-receipt", receipt_document)
    _write_exclusive_durable(claim.paths.http_receipt, payload)
    return ImmutableReceipt(
        kind="HTTP",
        path=claim.paths.http_receipt,
        sha256=hashlib.sha256(payload).hexdigest(),
        evidence_sha256=receipt_evidence,
    )


def write_terminal_receipt(
    claim: AcquisitionClaim,
    *,
    finished_at_utc: datetime,
    status: str,
    reason_code: str,
    transport_attempted: bool,
) -> ImmutableReceipt:
    """Seal one terminal outcome without permitting a second acquisition."""

    _verify_claim(claim)
    finished = _runtime_utc(finished_at_utc, "finished_at_utc")
    if finished < claim.claimed_at_utc:
        raise ReceiptValidationError("terminal receipt predates the claim")
    if status not in _TERMINAL_STATUSES:
        raise ReceiptValidationError("terminal status is not allowed")
    if not isinstance(reason_code, str) or _REASON_CODE_RE.fullmatch(reason_code) is None:
        raise ReceiptValidationError("reason_code must be a bounded uppercase identifier")
    if not isinstance(transport_attempted, bool):
        raise ReceiptValidationError("transport_attempted must be boolean")

    http_document: dict[str, Any] | None = None
    http_raw: bytes | None = None
    if claim.paths.http_receipt.exists() or claim.paths.http_receipt.is_symlink():
        http_document, http_raw = _load_sealed(
            claim.paths.http_receipt,
            expected_kind="PMXT_HTTP_RECEIPT",
            error_label="HTTP receipt",
        )
        if (
            http_document.get("permit_sha256") != claim.permit_sha256
            or http_document.get("claim_sha256") != claim.claim_sha256
            or http_document.get("request_sha256") != claim.request_sha256
        ):
            raise AcquisitionStateError("HTTP receipt binding changed")

    has_http = http_document is not None and http_raw is not None
    if status in {"COMPLETED", "RESPONSE_REJECTED"} and not has_http:
        raise ReceiptValidationError(f"{status} requires an immutable HTTP receipt")
    if status in {"TRANSPORT_FAILED", "ABORTED_BEFORE_TRANSPORT"} and has_http:
        raise ReceiptValidationError(f"{status} cannot follow an HTTP receipt")
    if status == "ABORTED_BEFORE_TRANSPORT" and transport_attempted:
        raise ReceiptValidationError("pre-transport abort cannot claim a transport attempt")
    if status == "TRANSPORT_FAILED" and not transport_attempted:
        raise ReceiptValidationError("transport failure must record one attempted transport")
    if has_http and not transport_attempted:
        raise ReceiptValidationError("an HTTP receipt proves transport was attempted")
    if has_http:
        received = _utc_datetime(http_document["response_received_at_utc"], "response_received_at_utc")
        if finished < received:
            raise ReceiptValidationError("terminal receipt predates the HTTP receipt")
        if status == "COMPLETED" and not 200 <= int(http_document["status_code"]) < 300:
            raise ReceiptValidationError("COMPLETED requires a successful HTTP status")

    terminal_document = {
        "schema_version": 1,
        "kind": "PMXT_TERMINAL_RECEIPT",
        "protocol_id": PROTOCOL_ID,
        "permit_sha256": claim.permit_sha256,
        "claim_sha256": claim.claim_sha256,
        "request_sha256": claim.request_sha256,
        "finished_at_utc": _format_utc(finished),
        "status": status,
        "reason_code": reason_code,
        "transport_attempted": transport_attempted,
        "http_receipt_present": has_http,
        "http_receipt_sha256": hashlib.sha256(http_raw).hexdigest() if http_raw is not None else None,
        "credential_material_persisted": False,
        "orders_submitted": False,
        "claim_released": False,
    }
    _, payload, terminal_evidence = _sealed_document("terminal-receipt", terminal_document)
    _write_exclusive_durable(claim.paths.terminal_receipt, payload)
    return ImmutableReceipt(
        kind="TERMINAL",
        path=claim.paths.terminal_receipt,
        sha256=hashlib.sha256(payload).hexdigest(),
        evidence_sha256=terminal_evidence,
    )
