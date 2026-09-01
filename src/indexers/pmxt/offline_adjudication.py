"""Immutable, offline-only adjudication of one hash-bound PMXT monitor run.

This module has no network, credential, book, account, order, or economics
surface.  It validates an already-reviewed decision packet against immutable
monitor artifacts, then writes a new append-only adjudication run.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .semantic import _metadata_provenance_reasons

PROTOCOL_ID = "pmxt-offline-semantic-adjudication-v1"
SCHEMA_VERSION = 1
DEFAULT_PROTOCOL_PATH = Path("results/semantic_adjudication_v1/protocol_v1.json")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REASON_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_DECISIONS = {"REJECTED", "NEEDS_REVIEW"}
_SOURCE_ROW_KEYS = {
    "candidate": "candidates",
    "automated_decision": "semantic_decisions",
    "kalshi_metadata": "raw_native_metadata",
    "polymarket_metadata": "raw_native_metadata",
}


class OfflineAdjudicationError(RuntimeError):
    """Raised before any unsafe or unverifiable adjudication is emitted."""


class _DuplicateKey(ValueError):
    pass


@dataclass(frozen=True)
class _JsonlRow:
    line: int
    raw: bytes
    sha256: str
    value: Mapping[str, Any]


@dataclass(frozen=True)
class _ArtifactRows:
    path: Path
    binding: Mapping[str, Any]
    rows: tuple[_JsonlRow, ...]


@dataclass(frozen=True)
class _InputSnapshot:
    path: Path
    payload: bytes
    binding: Mapping[str, Any]


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, _DuplicateKey) as exc:
        raise OfflineAdjudicationError(f"{label} is not strict UTF-8 JSON") from exc


def _canonical_json_value_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise OfflineAdjudicationError("value is not canonical JSON") from exc
    return text.encode("utf-8")


def _canonical_json_bytes(value: Any) -> bytes:
    return _canonical_json_value_bytes(value) + b"\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OfflineAdjudicationError(f"{label} must be a lowercase SHA-256")
    return value


def _relative_parts(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        raise OfflineAdjudicationError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or path.drive or any(part in {"", ".", ".."} for part in path.parts):
        raise OfflineAdjudicationError(f"{label} must be a contained relative path")
    return path.parts


def _is_reparse_status(status: os.stat_result) -> bool:
    attributes = int(getattr(status, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag)


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OfflineAdjudicationError(f"cannot inspect path component: {path}") from exc


def _reject_absolute_reparse_chain(path: Path, *, label: str) -> os.stat_result:
    if not path.is_absolute():
        raise OfflineAdjudicationError(f"{label} must be absolute")
    current = Path(path.anchor)
    status = _lstat_or_none(current)
    if status is None:
        raise OfflineAdjudicationError(f"{label} is missing")
    if _is_reparse_status(status):
        raise OfflineAdjudicationError(f"{label} must not traverse a symlink or reparse point")
    for part in path.parts[1:]:
        current /= part
        status = _lstat_or_none(current)
        if status is None:
            raise OfflineAdjudicationError(f"{label} is missing")
        if _is_reparse_status(status):
            raise OfflineAdjudicationError(f"{label} must not traverse a symlink or reparse point")
    return status


def _reject_reparse_chain(
    root: Path,
    parts: Sequence[str],
    *,
    label: str,
    require_all: bool,
) -> Path:
    current = root
    root_status = _lstat_or_none(root)
    if root_status is None:
        raise OfflineAdjudicationError("repository root is missing")
    if _is_reparse_status(root_status):
        raise OfflineAdjudicationError("repository root must not be a symlink or reparse point")
    for index, part in enumerate(parts):
        current /= part
        status = _lstat_or_none(current)
        if status is None:
            if require_all:
                raise OfflineAdjudicationError(f"{label} is missing")
            return root.joinpath(*parts)
        if _is_reparse_status(status):
            raise OfflineAdjudicationError(f"{label} must not traverse a symlink or reparse point")
        if index < len(parts) - 1 and not stat.S_ISDIR(status.st_mode):
            raise OfflineAdjudicationError(f"{label} has a non-directory path component")
    return current


def _contained_file(root: Path, relative: Any, *, label: str) -> Path:
    parts = _relative_parts(relative, label=label)
    path = _reject_reparse_chain(root, parts, label=label, require_all=True)
    status = _lstat_or_none(path)
    if status is None or not stat.S_ISREG(status.st_mode):
        raise OfflineAdjudicationError(f"{label} is not a regular file")
    return path


def _contained_output(root: Path, relative: Any, *, run_id: str) -> Path:
    parts = _relative_parts(relative, label="output path")
    expected = ("data", "pmxt", "semantic_adjudication", "runs", run_id)
    expected_text = "/".join(expected)
    if relative != expected_text or parts != expected:
        raise OfflineAdjudicationError(f"output path must be exactly data/pmxt/semantic_adjudication/runs/{run_id}")
    return _reject_reparse_chain(
        root,
        parts,
        label="output path",
        require_all=False,
    )


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    if left.st_dev != right.st_dev:
        return False
    if (left.st_ino or right.st_ino) and left.st_ino != right.st_ino:
        return False
    return (
        left.st_mode,
        left.st_size,
        getattr(left, "st_mtime_ns", None),
        getattr(left, "st_ctime_ns", None),
    ) == (
        right.st_mode,
        right.st_size,
        getattr(right, "st_mtime_ns", None),
        getattr(right, "st_ctime_ns", None),
    )


def _read_input(path: Path, *, root: Path, label: str) -> _InputSnapshot:
    """Read one input once; validate and bind the bytes from that same open handle."""

    try:
        relative_parts = path.relative_to(root).parts
    except ValueError as exc:
        raise OfflineAdjudicationError(f"{label} is outside the repository") from exc
    _reject_reparse_chain(root, relative_parts, label=label, require_all=True)
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OfflineAdjudicationError(f"cannot open {label} without following links") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OfflineAdjudicationError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not _same_file(before, after) or before.st_size != after.st_size:
        raise OfflineAdjudicationError(f"{label} changed while it was read")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise OfflineAdjudicationError(f"{label} size changed while it was read")
    path_status = _lstat_or_none(path)
    if path_status is None or _is_reparse_status(path_status) or not _same_file(after, path_status):
        raise OfflineAdjudicationError(f"{label} path changed while it was read")
    binding = {
        "path": path.relative_to(root).as_posix(),
        "byte_size": len(payload),
        "sha256": _sha256_bytes(payload),
    }
    return _InputSnapshot(path=path, payload=payload, binding=binding)


def _load_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    value = _strict_json_bytes(payload, label=label)
    if not isinstance(value, Mapping):
        raise OfflineAdjudicationError(f"{label} must be a JSON object")
    return value


def _load_jsonl(payload: bytes, *, label: str) -> tuple[_JsonlRow, ...]:
    if payload and not payload.endswith(b"\n"):
        raise OfflineAdjudicationError(f"{label} must end with an LF terminator")
    rows: list[_JsonlRow] = []
    for line_number, raw_line in enumerate(payload.splitlines(keepends=True), start=1):
        if not raw_line.strip():
            raise OfflineAdjudicationError(f"{label} contains a blank row")
        value = _strict_json_bytes(raw_line, label=f"{label} line {line_number}")
        if not isinstance(value, Mapping):
            raise OfflineAdjudicationError(f"{label} line {line_number} must be a JSON object")
        rows.append(
            _JsonlRow(
                line=line_number,
                raw=raw_line,
                sha256=_sha256_bytes(raw_line),
                value=value,
            )
        )
    return tuple(rows)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OfflineAdjudicationError(f"{label} must be an object")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise OfflineAdjudicationError(f"{label} must be an array")
    return value


def _json_pointer(document: Any, pointer: Any, *, label: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise OfflineAdjudicationError(f"{label} must be a non-root JSON pointer")
    current = document
    for encoded in pointer[1:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise OfflineAdjudicationError(f"{label} does not resolve")
            current = current[token]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            if not token.isdigit() or int(token) >= len(current):
                raise OfflineAdjudicationError(f"{label} does not resolve")
            current = current[int(token)]
        else:
            raise OfflineAdjudicationError(f"{label} does not resolve")
    return current


def _is_substantive_clause(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    return isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)) and bool(value)


_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_SUPPORTED_SCHEMA_KEYWORDS = {
    "$defs",
    "$id",
    "$ref",
    "$schema",
    "additionalProperties",
    "allOf",
    "const",
    "description",
    "else",
    "enum",
    "if",
    "items",
    "maxItems",
    "minItems",
    "minLength",
    "minimum",
    "pattern",
    "properties",
    "required",
    "then",
    "title",
    "type",
    "uniqueItems",
}
_JSON_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality that does not confuse booleans with the integers zero/one."""

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(left_item, right_item) for left_item, right_item in zip(left, right)
        )
    return left == right


def _resolve_schema_reference(root_schema: Mapping[str, Any], reference: Any) -> Mapping[str, Any]:
    if not isinstance(reference, str) or not reference.startswith("#/"):
        raise OfflineAdjudicationError("decision schema supports only local JSON-pointer $ref values")
    target = _json_pointer(root_schema, reference[1:], label="decision schema $ref")
    if not isinstance(target, Mapping):
        raise OfflineAdjudicationError("decision schema $ref must resolve to a schema object")
    return target


def _validate_schema_shape(
    schema: Any,
    *,
    root_schema: Mapping[str, Any],
    location: str,
) -> None:
    if not isinstance(schema, Mapping):
        raise OfflineAdjudicationError(f"decision schema at {location} must be an object")
    unsupported = sorted(set(schema) - _SUPPORTED_SCHEMA_KEYWORDS)
    if unsupported:
        raise OfflineAdjudicationError(f"decision schema uses unsupported Draft 2020-12 keyword: {unsupported[0]}")
    for annotation in ("$id", "$schema", "title", "description"):
        if annotation in schema and not isinstance(schema[annotation], str):
            raise OfflineAdjudicationError(f"decision schema {annotation} at {location} must be a string")
    if "$ref" in schema:
        _resolve_schema_reference(root_schema, schema["$ref"])
    if "type" in schema and schema["type"] not in _JSON_TYPES:
        raise OfflineAdjudicationError(f"decision schema type at {location} is unsupported")
    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or not choices:
            raise OfflineAdjudicationError(f"decision schema enum at {location} must be non-empty")
        if any(_json_equal(choice, earlier) for index, choice in enumerate(choices) for earlier in choices[:index]):
            raise OfflineAdjudicationError(f"decision schema enum at {location} has duplicates")
    for keyword in ("minLength", "minItems", "maxItems"):
        if keyword in schema:
            limit = schema[keyword]
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise OfflineAdjudicationError(
                    f"decision schema {keyword} at {location} must be a non-negative integer"
                )
    if "minimum" in schema:
        minimum = schema["minimum"]
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not math.isfinite(float(minimum)):
            raise OfflineAdjudicationError(f"decision schema minimum at {location} must be finite")
    if "pattern" in schema:
        pattern = schema["pattern"]
        if not isinstance(pattern, str):
            raise OfflineAdjudicationError(f"decision schema pattern at {location} must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise OfflineAdjudicationError(f"decision schema pattern at {location} is invalid") from exc
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise OfflineAdjudicationError(f"decision schema uniqueItems at {location} must be boolean")
    if "required" in schema:
        required = schema["required"]
        if (
            not isinstance(required, list)
            or any(not isinstance(item, str) for item in required)
            or len(set(required)) != len(required)
        ):
            raise OfflineAdjudicationError(f"decision schema required at {location} must contain unique strings")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise OfflineAdjudicationError(f"decision schema additionalProperties at {location} must be boolean")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping) or any(not isinstance(key, str) for key in properties):
            raise OfflineAdjudicationError(f"decision schema properties at {location} must be an object")
        for key, child in properties.items():
            _validate_schema_shape(
                child,
                root_schema=root_schema,
                location=f"{location}/properties/{key}",
            )
    definitions = schema.get("$defs")
    if definitions is not None:
        if not isinstance(definitions, Mapping) or any(not isinstance(key, str) for key in definitions):
            raise OfflineAdjudicationError(f"decision schema $defs at {location} must be an object")
        for key, child in definitions.items():
            _validate_schema_shape(
                child,
                root_schema=root_schema,
                location=f"{location}/$defs/{key}",
            )
    if "items" in schema:
        _validate_schema_shape(
            schema["items"],
            root_schema=root_schema,
            location=f"{location}/items",
        )
    if "allOf" in schema:
        clauses = schema["allOf"]
        if not isinstance(clauses, list) or not clauses:
            raise OfflineAdjudicationError(f"decision schema allOf at {location} must be non-empty")
        for index, child in enumerate(clauses):
            _validate_schema_shape(
                child,
                root_schema=root_schema,
                location=f"{location}/allOf/{index}",
            )
    if ("then" in schema or "else" in schema) and "if" not in schema:
        raise OfflineAdjudicationError(f"decision schema then/else at {location} requires if")
    for keyword in ("if", "then", "else"):
        if keyword in schema:
            _validate_schema_shape(
                schema[keyword],
                root_schema=root_schema,
                location=f"{location}/{keyword}",
            )


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "boolean":
        return isinstance(value, bool)
    return value is None


def _schema_validation_errors(
    value: Any,
    schema: Mapping[str, Any],
    *,
    root_schema: Mapping[str, Any],
    location: str = "$",
) -> list[str]:
    errors: list[str] = []
    if "$ref" in schema:
        errors.extend(
            _schema_validation_errors(
                value,
                _resolve_schema_reference(root_schema, schema["$ref"]),
                root_schema=root_schema,
                location=location,
            )
        )
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _json_type_matches(value, expected_type):
        return [f"{location} must have type {expected_type}"]
    if "const" in schema and not _json_equal(value, schema["const"]):
        errors.append(f"{location} does not equal const")
    if "enum" in schema and not any(_json_equal(value, choice) for choice in schema["enum"]):
        errors.append(f"{location} is not in enum")
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(f"{location}/{key} is required")
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            if schema.get("additionalProperties") is False:
                extras = sorted(set(value) - set(properties))
                if extras:
                    errors.append(f"{location} has additional property {extras[0]}")
            for key, child_schema in properties.items():
                if key in value:
                    errors.extend(
                        _schema_validation_errors(
                            value[key],
                            child_schema,
                            root_schema=root_schema,
                            location=f"{location}/{key}",
                        )
                    )
    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            errors.append(f"{location} is shorter than minLength")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            errors.append(f"{location} does not match pattern")
    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            errors.append(f"{location} has fewer than minItems")
        if isinstance(maximum_items, int) and len(value) > maximum_items:
            errors.append(f"{location} has more than maxItems")
        if schema.get("uniqueItems") is True and any(
            _json_equal(item, earlier) for index, item in enumerate(value) for earlier in value[:index]
        ):
            errors.append(f"{location} does not have uniqueItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                errors.extend(
                    _schema_validation_errors(
                        item,
                        item_schema,
                        root_schema=root_schema,
                        location=f"{location}/{index}",
                    )
                )
    if (
        "minimum" in schema
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value < schema["minimum"]
    ):
        errors.append(f"{location} is below minimum")
    for child_schema in schema.get("allOf", []):
        errors.extend(
            _schema_validation_errors(
                value,
                child_schema,
                root_schema=root_schema,
                location=location,
            )
        )
    if_schema = schema.get("if")
    if isinstance(if_schema, Mapping):
        branch = (
            "then"
            if not _schema_validation_errors(
                value,
                if_schema,
                root_schema=root_schema,
                location=location,
            )
            else "else"
        )
        branch_schema = schema.get(branch)
        if isinstance(branch_schema, Mapping):
            errors.extend(
                _schema_validation_errors(
                    value,
                    branch_schema,
                    root_schema=root_schema,
                    location=location,
                )
            )
    return errors


def _validate_decision_schema(schema: Mapping[str, Any]) -> None:
    if schema.get("$schema") != _DRAFT_2020_12:
        raise OfflineAdjudicationError("decision schema must declare Draft 2020-12")
    _validate_schema_shape(schema, root_schema=schema, location="#")


def _validate_protocol(protocol: Mapping[str, Any]) -> tuple[int, int, int]:
    if protocol.get("schema_version") != SCHEMA_VERSION or protocol.get("protocol_id") != PROTOCOL_ID:
        raise OfflineAdjudicationError("protocol identity is invalid")
    authority = _require_mapping(protocol.get("authority"), label="protocol authority")
    required_authority = {
        "offline_only": True,
        "network_requests": 0,
        "credentials_read": False,
        "books_requested": False,
        "orders_submitted": 0,
        "economics_computed": False,
        "live_eligible": False,
    }
    if any(not _json_equal(authority.get(key), value) for key, value in required_authority.items()):
        raise OfflineAdjudicationError("protocol authority widens the offline-only boundary")
    counts = _require_mapping(protocol.get("expected_counts"), label="expected counts")
    total = counts.get("total")
    rejected = counts.get("rejected")
    needs_review = counts.get("needs_review")
    for name, value in (("total", total), ("rejected", rejected), ("needs_review", needs_review)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OfflineAdjudicationError(f"expected {name} count must be a non-negative integer")
    assert isinstance(total, int) and isinstance(rejected, int) and isinstance(needs_review, int)
    if total <= 0 or rejected + needs_review != total:
        raise OfflineAdjudicationError("expected adjudication counts do not sum to a positive total")
    return total, rejected, needs_review


def _validate_adjudication_provenance(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(protocol.get("adjudication_provenance"), label="adjudication provenance")
    if set(value) != {"method", "version", "review_completed_at_utc"}:
        raise OfflineAdjudicationError("adjudication provenance fields are invalid")
    if value.get("method") != "offline_native_rule_clause_review":
        raise OfflineAdjudicationError("adjudication provenance method is invalid")
    version = value.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise OfflineAdjudicationError("adjudication provenance version is invalid")
    completed = value.get("review_completed_at_utc")
    match = (
        re.fullmatch(
            r"(?P<seconds>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(?P<fraction>\d{1,9}))?Z",
            completed,
        )
        if isinstance(completed, str)
        else None
    )
    if match is None:
        raise OfflineAdjudicationError("adjudication review timestamp must be strict UTC ending Z")
    fraction = match.group("fraction")
    parseable = match.group("seconds")
    if fraction is not None:
        parseable += "." + (fraction + "000000")[:6]
    try:
        parsed = datetime.fromisoformat(parseable + "+00:00")
    except ValueError as exc:
        raise OfflineAdjudicationError("adjudication review timestamp must be strict UTC ending Z") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise OfflineAdjudicationError("adjudication review timestamp must be strict UTC ending Z")
    return dict(value)


def _validate_source_artifact(
    root: Path,
    run_dir: Path,
    manifest: Mapping[str, Any],
    protocol_artifacts: Mapping[str, Any],
    name: str,
) -> _ArtifactRows:
    manifest_artifacts = _require_mapping(manifest.get("artifacts"), label="source manifest artifacts")
    manifest_entry = _require_mapping(manifest_artifacts.get(name), label=f"source manifest artifact {name}")
    protocol_entry = _require_mapping(protocol_artifacts.get(name), label=f"protocol artifact {name}")
    manifest_parts = _relative_parts(manifest_entry.get("path"), label=f"source manifest artifact path {name}")
    expected_repo_path = run_dir.joinpath(*manifest_parts).relative_to(root).as_posix()
    if protocol_entry.get("path") != expected_repo_path:
        raise OfflineAdjudicationError(f"protocol path for {name} does not match the source manifest")
    expected_sha = _require_sha256(protocol_entry.get("sha256"), label=f"protocol {name} SHA-256")
    manifest_sha = _require_sha256(manifest_entry.get("sha256"), label=f"manifest {name} SHA-256")
    if expected_sha != manifest_sha:
        raise OfflineAdjudicationError(f"protocol and manifest hashes disagree for {name}")
    path = _contained_file(root, expected_repo_path, label=f"source artifact {name}")
    snapshot = _read_input(path, root=root, label=f"source artifact {name}")
    declared_size = manifest_entry.get("byte_size")
    if isinstance(declared_size, bool) or not isinstance(declared_size, int) or declared_size != len(snapshot.payload):
        raise OfflineAdjudicationError(f"source artifact size mismatch for {name}")
    if snapshot.binding["sha256"] != expected_sha:
        raise OfflineAdjudicationError(f"source artifact hash mismatch for {name}")
    return _ArtifactRows(
        path=path,
        binding=snapshot.binding,
        rows=_load_jsonl(snapshot.payload, label=name),
    )


def _validate_sources(
    root: Path,
    protocol: Mapping[str, Any],
    expected_total: int,
) -> tuple[
    dict[str, _ArtifactRows],
    Mapping[str, Any],
    _InputSnapshot,
    Mapping[str, Any],
]:
    source = _require_mapping(protocol.get("source_monitor_run"), label="source monitor run")
    source_run_id = source.get("run_id")
    if not isinstance(source_run_id, str) or _RUN_ID_RE.fullmatch(source_run_id) is None:
        raise OfflineAdjudicationError("source run ID is invalid")
    manifest_spec = _require_mapping(source.get("manifest"), label="source manifest binding")
    manifest_path = _contained_file(root, manifest_spec.get("path"), label="source manifest")
    manifest_snapshot = _read_input(manifest_path, root=root, label="source manifest")
    expected_manifest_sha = _require_sha256(manifest_spec.get("sha256"), label="source manifest SHA-256")
    if manifest_snapshot.binding["sha256"] != expected_manifest_sha:
        raise OfflineAdjudicationError("source manifest hash mismatch")
    manifest = _load_json_object(manifest_snapshot.payload, label="source manifest")
    if (
        manifest.get("run_id") != source_run_id
        or manifest.get("live_eligible") is not False
        or manifest.get("no_order_actions") is not True
    ):
        raise OfflineAdjudicationError("source manifest does not preserve the research-only boundary")
    run_dir = manifest_path.parent
    protocol_artifacts = _require_mapping(source.get("artifacts"), label="protocol source artifacts")
    if set(protocol_artifacts) != {"candidates", "raw_native_metadata", "semantic_decisions"}:
        raise OfflineAdjudicationError("protocol must bind exactly the three adjudication source artifacts")
    artifacts = {
        name: _validate_source_artifact(root, run_dir, manifest, protocol_artifacts, name)
        for name in ("candidates", "raw_native_metadata", "semantic_decisions")
    }
    if len(artifacts["candidates"].rows) != expected_total:
        raise OfflineAdjudicationError("candidate count does not match the protocol")
    if len(artifacts["semantic_decisions"].rows) != expected_total:
        raise OfflineAdjudicationError("automated-decision count does not match the protocol")
    if len(artifacts["raw_native_metadata"].rows) != expected_total * 2:
        raise OfflineAdjudicationError("native-metadata count does not equal two rows per candidate")
    manifest_counts = _require_mapping(manifest.get("counts"), label="source manifest counts")
    for name, artifact in artifacts.items():
        if manifest_counts.get(name) != len(artifact.rows):
            raise OfflineAdjudicationError(f"source manifest row count mismatch for {name}")
    return artifacts, manifest, manifest_snapshot, source


def _index_sources(
    artifacts: Mapping[str, _ArtifactRows],
) -> tuple[
    list[str],
    dict[str, _JsonlRow],
    dict[str, _JsonlRow],
    dict[tuple[str, str], _JsonlRow],
]:
    candidate_order: list[str] = []
    candidates: dict[str, _JsonlRow] = {}
    for row in artifacts["candidates"].rows:
        candidate_id = row.value.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidates:
            raise OfflineAdjudicationError("candidate IDs must be non-empty and unique")
        if (
            row.value.get("relation") != "identity"
            or row.value.get("live_eligible") is not False
            or {row.value.get("venue_a"), row.value.get("venue_b")} != {"kalshi", "polymarket"}
        ):
            raise OfflineAdjudicationError("candidate row is outside the identity-only safety envelope")
        candidates[candidate_id] = row
        candidate_order.append(candidate_id)

    decisions: dict[str, _JsonlRow] = {}
    for row in artifacts["semantic_decisions"].rows:
        candidate_id = row.value.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in decisions or candidate_id not in candidates:
            raise OfflineAdjudicationError("automated-decision IDs do not match unique source candidates")
        if row.value.get("status") != "NEEDS_REVIEW" or row.value.get("live_eligible") is not False:
            raise OfflineAdjudicationError("source automated decisions must remain NEEDS_REVIEW and live-ineligible")
        decisions[candidate_id] = row
    if list(decisions) != candidate_order:
        raise OfflineAdjudicationError("automated decisions are not in source candidate order")

    metadata: dict[tuple[str, str], _JsonlRow] = {}
    for row in artifacts["raw_native_metadata"].rows:
        candidate_id = row.value.get("candidate_id")
        venue = row.value.get("venue")
        if candidate_id not in candidates or venue not in {"kalshi", "polymarket"}:
            raise OfflineAdjudicationError("native metadata is not bound to a source candidate and venue")
        key = (str(candidate_id), str(venue))
        if key in metadata:
            raise OfflineAdjudicationError("native metadata contains a duplicate candidate/venue row")
        candidate = candidates[str(candidate_id)].value
        side = row.value.get("side")
        if side not in {"a", "b"} or candidate.get(f"venue_{side}") != venue:
            raise OfflineAdjudicationError("native metadata side does not match its source candidate")
        if row.value.get("status") != "RESOLVED" or row.value.get("live_eligible") is not False:
            raise OfflineAdjudicationError("native metadata must be resolved and live-ineligible")
        raw_hash = _require_sha256(row.value.get("raw_sha256"), label="native raw SHA-256")
        rule_hash = _require_sha256(row.value.get("rule_hash"), label="native rule SHA-256")
        if _sha256_bytes(_canonical_json_value_bytes(row.value.get("raw_response"))) != raw_hash:
            raise OfflineAdjudicationError("native raw-response hash mismatch")
        if _sha256_bytes(_canonical_json_value_bytes(row.value.get("normalized_rules"))) != rule_hash:
            raise OfflineAdjudicationError("native normalized-rule hash mismatch")
        metadata[key] = row
    if len(metadata) != len(candidate_order) * 2:
        raise OfflineAdjudicationError("native metadata does not contain exactly two venues per candidate")
    for candidate_id in candidate_order:
        provenance_reasons = _metadata_provenance_reasons(
            candidates[candidate_id].value,
            {
                "kalshi": metadata[(candidate_id, "kalshi")].value,
                "polymarket": metadata[(candidate_id, "polymarket")].value,
            },
        )
        if provenance_reasons:
            raise OfflineAdjudicationError(
                "native metadata provenance validation failed: " + ",".join(sorted(set(provenance_reasons)))
            )
    return candidate_order, candidates, decisions, metadata


def _validate_source_row_reference(
    reference: Any,
    *,
    label: str,
    expected_artifact: str,
    expected_row: _JsonlRow,
) -> None:
    value = _require_mapping(reference, label=label)
    if set(value) - {"artifact", "line", "row_sha256", "raw_sha256", "rule_hash"}:
        raise OfflineAdjudicationError(f"{label} contains unsupported fields")
    if value.get("artifact") != expected_artifact or value.get("line") != expected_row.line:
        raise OfflineAdjudicationError(f"{label} does not identify the expected source row")
    if value.get("row_sha256") != expected_row.sha256:
        raise OfflineAdjudicationError(f"{label} row SHA-256 mismatch")
    if expected_artifact == "raw_native_metadata":
        if value.get("raw_sha256") != expected_row.value.get("raw_sha256"):
            raise OfflineAdjudicationError(f"{label} native raw SHA-256 mismatch")
        if value.get("rule_hash") != expected_row.value.get("rule_hash"):
            raise OfflineAdjudicationError(f"{label} native rule SHA-256 mismatch")


def _validate_pointer_reference(
    reference: Any,
    *,
    label: str,
    source_rows: Mapping[str, _JsonlRow],
    require_clause_hash: bool,
) -> Any:
    value = _require_mapping(reference, label=label)
    allowed = {"source_row", "json_pointer", "clause_sha256"}
    if set(value) - allowed:
        raise OfflineAdjudicationError(f"{label} contains unsupported fields")
    source_key = value.get("source_row")
    if source_key not in source_rows:
        raise OfflineAdjudicationError(f"{label} names an unsupported source row")
    pointer = value.get("json_pointer")
    resolved = _json_pointer(source_rows[str(source_key)].value, pointer, label=f"{label} pointer")
    if require_clause_hash:
        if source_key not in {"kalshi_metadata", "polymarket_metadata"}:
            raise OfflineAdjudicationError(f"{label} must cite venue-native metadata")
        if not isinstance(pointer, str) or not pointer.startswith("/normalized_rules/"):
            raise OfflineAdjudicationError(f"{label} must cite an explicit normalized native-rule clause")
        if not _is_substantive_clause(resolved):
            raise OfflineAdjudicationError(f"{label} must resolve to a non-empty two-sided clause")
        clause_hash = _require_sha256(value.get("clause_sha256"), label=f"{label} clause SHA-256")
        if clause_hash != _sha256_bytes(_canonical_json_value_bytes(resolved)):
            raise OfflineAdjudicationError(f"{label} clause SHA-256 mismatch")
    elif "clause_sha256" in value:
        raise OfflineAdjudicationError(f"{label} may not carry an unvalidated clause hash")
    return resolved


def _validate_text_list(value: Any, *, label: str, allow_empty: bool = False) -> list[str]:
    items = _require_list(value, label=label)
    if (not allow_empty and not items) or any(not isinstance(item, str) or not item.strip() for item in items):
        raise OfflineAdjudicationError(f"{label} must contain non-empty strings")
    if len(set(items)) != len(items):
        raise OfflineAdjudicationError(f"{label} must not contain duplicates")
    return items


def _validate_evidence_gaps(
    gaps: Sequence[Any],
    *,
    pointer_rows: Mapping[str, _JsonlRow],
) -> None:
    for index, gap in enumerate(gaps):
        item = _require_mapping(gap, label=f"evidence gap {index}")
        if set(item) != {"axis", "summary", "evidence"}:
            raise OfflineAdjudicationError("evidence gap fields are invalid")
        for field in ("axis", "summary"):
            if not isinstance(item.get(field), str) or not str(item[field]).strip():
                raise OfflineAdjudicationError("evidence gap axis and summary must be non-empty")
        refs = _require_list(item.get("evidence"), label=f"evidence gap {index} references")
        if not refs:
            raise OfflineAdjudicationError("evidence gap must cite at least one source pointer")
        for ref_index, reference in enumerate(refs):
            _validate_pointer_reference(
                reference,
                label=f"evidence gap {index} reference {ref_index}",
                source_rows=pointer_rows,
                require_clause_hash=False,
            )


def _validate_review_row(
    row: _JsonlRow,
    *,
    expected_candidate_id: str,
    candidate: _JsonlRow,
    automated_decision: _JsonlRow,
    kalshi_metadata: _JsonlRow,
    polymarket_metadata: _JsonlRow,
) -> str:
    value = row.value
    required_keys = {
        "schema_version",
        "record_type",
        "candidate_id",
        "decision",
        "review_group",
        "review_summary",
        "reason_codes",
        "source_rows",
        "automated_decision_evidence",
        "material_mismatches",
        "evidence_gaps",
        "reviewed",
        "semantic_verified",
        "profitability_evaluation_eligible",
        "network_requests",
        "orders_submitted",
        "economics_computed",
        "live_eligible",
    }
    if set(value) != required_keys:
        raise OfflineAdjudicationError("review row fields do not match the v1 schema")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("record_type") != "pmxt_offline_adjudication":
        raise OfflineAdjudicationError("review row schema identity is invalid")
    if value.get("candidate_id") != expected_candidate_id:
        raise OfflineAdjudicationError("review rows are not in source candidate order")
    decision = value.get("decision")
    if decision not in _DECISIONS:
        raise OfflineAdjudicationError("review decision must be REJECTED or NEEDS_REVIEW")
    for field in ("review_group", "review_summary"):
        if not isinstance(value.get(field), str) or not str(value[field]).strip():
            raise OfflineAdjudicationError(f"{field} must be non-empty")
    reason_codes = _validate_text_list(value.get("reason_codes"), label="reason codes")
    if any(_REASON_RE.fullmatch(code) is None for code in reason_codes):
        raise OfflineAdjudicationError("reason codes must use stable uppercase identifiers")
    safety = {
        "reviewed": True,
        "semantic_verified": False,
        "profitability_evaluation_eligible": False,
        "network_requests": 0,
        "orders_submitted": 0,
        "economics_computed": False,
        "live_eligible": False,
    }
    if any(not _json_equal(value.get(key), expected) for key, expected in safety.items()):
        raise OfflineAdjudicationError("review row widens the offline, non-executable safety boundary")

    row_references = _require_mapping(value.get("source_rows"), label="source rows")
    if set(row_references) != set(_SOURCE_ROW_KEYS):
        raise OfflineAdjudicationError("review row must bind all four source rows")
    expected_rows = {
        "candidate": candidate,
        "automated_decision": automated_decision,
        "kalshi_metadata": kalshi_metadata,
        "polymarket_metadata": polymarket_metadata,
    }
    for source_key, expected_row in expected_rows.items():
        _validate_source_row_reference(
            row_references[source_key],
            label=f"source row {source_key}",
            expected_artifact=_SOURCE_ROW_KEYS[source_key],
            expected_row=expected_row,
        )

    pointer_rows = dict(expected_rows)
    automated_refs = _require_list(value.get("automated_decision_evidence"), label="automated decision evidence")
    observed_automated_pointers: set[str] = set()
    for index, reference in enumerate(automated_refs):
        ref = _require_mapping(reference, label=f"automated evidence {index}")
        if ref.get("source_row") != "automated_decision":
            raise OfflineAdjudicationError("automated decision evidence must cite the automated-decision row")
        _validate_pointer_reference(
            reference,
            label=f"automated evidence {index}",
            source_rows=pointer_rows,
            require_clause_hash=False,
        )
        pointer = ref.get("json_pointer")
        if isinstance(pointer, str):
            observed_automated_pointers.add(pointer)
    if not {"/status", "/reason_codes"} <= observed_automated_pointers:
        raise OfflineAdjudicationError("review row must cite source automated status and reason codes")

    mismatches = _require_list(value.get("material_mismatches"), label="material mismatches")
    gaps = _require_list(value.get("evidence_gaps"), label="evidence gaps")
    if decision == "REJECTED":
        if not mismatches:
            raise OfflineAdjudicationError("REJECTED requires a dispositive material mismatch")
        for index, mismatch in enumerate(mismatches):
            item = _require_mapping(mismatch, label=f"material mismatch {index}")
            if set(item) != {"axis", "summary", "kalshi_clause", "polymarket_clause"}:
                raise OfflineAdjudicationError("material mismatch fields are invalid")
            for field in ("axis", "summary"):
                if not isinstance(item.get(field), str) or not str(item[field]).strip():
                    raise OfflineAdjudicationError("material mismatch axis and summary must be non-empty")
            kalshi_ref = _require_mapping(item.get("kalshi_clause"), label="Kalshi mismatch clause")
            polymarket_ref = _require_mapping(item.get("polymarket_clause"), label="Polymarket mismatch clause")
            if kalshi_ref.get("source_row") != "kalshi_metadata":
                raise OfflineAdjudicationError("REJECTED must cite an explicit Kalshi clause")
            if polymarket_ref.get("source_row") != "polymarket_metadata":
                raise OfflineAdjudicationError("REJECTED must cite an explicit Polymarket clause")
            _validate_pointer_reference(
                kalshi_ref,
                label=f"material mismatch {index} Kalshi clause",
                source_rows=pointer_rows,
                require_clause_hash=True,
            )
            _validate_pointer_reference(
                polymarket_ref,
                label=f"material mismatch {index} Polymarket clause",
                source_rows=pointer_rows,
                require_clause_hash=True,
            )
        _validate_evidence_gaps(gaps, pointer_rows=pointer_rows)
    else:
        if mismatches or not gaps:
            raise OfflineAdjudicationError("NEEDS_REVIEW requires evidence gaps and no asserted material mismatch")
        _validate_evidence_gaps(gaps, pointer_rows=pointer_rows)
    return str(decision)


def _validate_review_packet(
    root: Path,
    protocol: Mapping[str, Any],
    *,
    expected_total: int,
    expected_rejected: int,
    expected_needs_review: int,
    candidate_order: Sequence[str],
    candidates: Mapping[str, _JsonlRow],
    automated_decisions: Mapping[str, _JsonlRow],
    metadata: Mapping[tuple[str, str], _JsonlRow],
) -> tuple[_InputSnapshot, tuple[_JsonlRow, ...], _InputSnapshot]:
    schema_spec = _require_mapping(protocol.get("decision_schema"), label="decision schema binding")
    schema_path = _contained_file(root, schema_spec.get("path"), label="decision schema")
    schema_snapshot = _read_input(schema_path, root=root, label="decision schema")
    if schema_snapshot.binding["sha256"] != _require_sha256(schema_spec.get("sha256"), label="decision schema SHA-256"):
        raise OfflineAdjudicationError("decision schema hash mismatch")
    schema = _load_json_object(schema_snapshot.payload, label="decision schema")
    if schema.get("$id") != "urn:prediction-market-analysis:pmxt:offline-adjudication:1":
        raise OfflineAdjudicationError("decision schema identity mismatch")
    _validate_decision_schema(schema)

    review_spec = _require_mapping(protocol.get("review_input"), label="review input binding")
    review_path = _contained_file(root, review_spec.get("path"), label="review input")
    review_snapshot = _read_input(review_path, root=root, label="review input")
    if review_snapshot.binding["sha256"] != _require_sha256(review_spec.get("sha256"), label="review input SHA-256"):
        raise OfflineAdjudicationError("review input hash mismatch")
    review_rows = _load_jsonl(review_snapshot.payload, label="review input")
    if len(review_rows) != expected_total or len(candidate_order) != expected_total:
        raise OfflineAdjudicationError("review input count does not match the protocol")
    observed_counts = {"REJECTED": 0, "NEEDS_REVIEW": 0}
    for index, row in enumerate(review_rows):
        schema_errors = _schema_validation_errors(
            row.value,
            schema,
            root_schema=schema,
        )
        if schema_errors:
            raise OfflineAdjudicationError(
                f"review input line {row.line} does not satisfy decision schema: " + "; ".join(schema_errors[:5])
            )
        candidate_id = candidate_order[index]
        decision = _validate_review_row(
            row,
            expected_candidate_id=candidate_id,
            candidate=candidates[candidate_id],
            automated_decision=automated_decisions[candidate_id],
            kalshi_metadata=metadata[(candidate_id, "kalshi")],
            polymarket_metadata=metadata[(candidate_id, "polymarket")],
        )
        observed_counts[decision] += 1
    if observed_counts != {"REJECTED": expected_rejected, "NEEDS_REVIEW": expected_needs_review}:
        raise OfflineAdjudicationError("review input decision counts do not match the protocol")
    return review_snapshot, review_rows, schema_snapshot


def _ensure_real_directories(root: Path, parts: Sequence[str]) -> Path:
    current = root
    for part in parts:
        current /= part
        status = _lstat_or_none(current)
        if status is None:
            try:
                os.mkdir(current)
            except FileExistsError:
                pass
            status = _lstat_or_none(current)
        if status is None or _is_reparse_status(status) or not stat.S_ISDIR(status.st_mode):
            raise OfflineAdjudicationError("output path must not traverse a symlink or reparse point")
    return current


def _write_exclusive(path: Path, payload: bytes) -> None:
    parent_status = _lstat_or_none(path.parent)
    if parent_status is None or _is_reparse_status(parent_status) or not stat.S_ISDIR(parent_status.st_mode):
        raise OfflineAdjudicationError("output parent changed before exclusive write")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while sealing adjudication artifact")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically finalize a directory without replacing any destination."""

    if _lstat_or_none(destination) is not None:
        raise OfflineAdjudicationError("final adjudication output already exists")
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise OfflineAdjudicationError("final adjudication output already exists") from exc
        except OSError as exc:
            raise OfflineAdjudicationError("exclusive adjudication finalization failed") from exc
        return

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise OfflineAdjudicationError("atomic no-replace finalization is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin":
        renamex_np = getattr(library, "renamex_np", None)
        if renamex_np is None:
            raise OfflineAdjudicationError("atomic no-replace finalization is unavailable")
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
    else:
        raise OfflineAdjudicationError("atomic no-replace finalization is unavailable")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise OfflineAdjudicationError("final adjudication output already exists")
    raise OfflineAdjudicationError(f"exclusive adjudication finalization failed: errno {error_number}")


def run_offline_adjudication(
    *,
    repository_root: Path,
    expected_protocol_sha256: str,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> Path:
    """Validate all inputs first, then exclusively seal one offline run."""

    expected_protocol_sha256 = _require_sha256(expected_protocol_sha256, label="expected protocol SHA-256")
    root_candidate = Path(os.path.abspath(repository_root))
    root_candidate_status = _reject_absolute_reparse_chain(root_candidate, label="repository root")
    try:
        root = root_candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise OfflineAdjudicationError("repository root is missing") from exc
    if not stat.S_ISDIR(root_candidate_status.st_mode):
        raise OfflineAdjudicationError("repository root must be a real directory")
    protocol_file = _contained_file(root, protocol_path.as_posix(), label="protocol")
    protocol_snapshot = _read_input(protocol_file, root=root, label="protocol")
    if protocol_snapshot.binding["sha256"] != expected_protocol_sha256:
        raise OfflineAdjudicationError("protocol hash mismatch")
    protocol = _load_json_object(protocol_snapshot.payload, label="protocol")
    expected_total, expected_rejected, expected_needs_review = _validate_protocol(protocol)
    adjudication_provenance = _validate_adjudication_provenance(protocol)
    artifacts, source_manifest, source_manifest_snapshot, source_spec = _validate_sources(
        root, protocol, expected_total
    )
    candidate_order, candidates, automated_decisions, metadata = _index_sources(artifacts)
    review_snapshot, review_rows, schema_snapshot = _validate_review_packet(
        root,
        protocol,
        expected_total=expected_total,
        expected_rejected=expected_rejected,
        expected_needs_review=expected_needs_review,
        candidate_order=candidate_order,
        candidates=candidates,
        automated_decisions=automated_decisions,
        metadata=metadata,
    )

    output_spec = _require_mapping(protocol.get("output"), label="output")
    run_id = output_spec.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise OfflineAdjudicationError("output run ID is invalid")
    final_path = _contained_output(root, output_spec.get("path"), run_id=run_id)
    staging_path = final_path.with_name(f"{run_id}.inprogress")
    staging_parts = staging_path.relative_to(root).parts
    _reject_reparse_chain(
        root,
        staging_parts,
        label="in-progress output path",
        require_all=False,
    )
    if _lstat_or_none(final_path) is not None or _lstat_or_none(staging_path) is not None:
        raise OfflineAdjudicationError("adjudication output or in-progress path already exists")

    source_bindings = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "pmxt_offline_adjudication_source_bindings",
        "adjudication_provenance": adjudication_provenance,
        "protocol": dict(protocol_snapshot.binding),
        "decision_schema": dict(schema_snapshot.binding),
        "review_input": dict(review_snapshot.binding),
        "source_monitor_run": {
            "run_id": source_spec.get("run_id"),
            "manifest": dict(source_manifest_snapshot.binding),
            "manifest_status": source_manifest.get("status"),
            "artifacts": {name: dict(artifact.binding) for name, artifact in artifacts.items()},
        },
        "row_sha256_semantics": "exact source JSONL line bytes including the LF terminator",
        "network_requests": 0,
        "orders_submitted": 0,
        "economics_computed": False,
        "live_eligible": False,
    }
    decisions_content = b"".join(_canonical_json_bytes(row.value) for row in review_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "status": "OFFLINE_ADJUDICATION_COMPLETE_NO_VERIFIED_CANDIDATES",
        "source_run_id": source_spec.get("run_id"),
        "adjudication_provenance": adjudication_provenance,
        "counts": {
            "total": expected_total,
            "rejected": expected_rejected,
            "needs_review": expected_needs_review,
            "verified_equivalent": 0,
        },
        "network_requests": 0,
        "credentials_read": False,
        "books_requested": False,
        "orders_submitted": 0,
        "economics_computed": False,
        "profitability_established": False,
        "live_eligible": False,
    }
    contents = {
        "source_bindings": _canonical_json_bytes(source_bindings),
        "decisions": decisions_content,
        "summary": _canonical_json_bytes(summary),
    }
    filenames = {
        "source_bindings": "source_bindings.json",
        "decisions": "decisions.jsonl",
        "summary": "summary.json",
    }
    artifact_bindings = {
        name: {
            "path": filenames[name],
            "byte_size": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        for name, payload in contents.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "status": summary["status"],
        "source_run_id": source_spec.get("run_id"),
        "source_manifest_sha256": source_bindings["source_monitor_run"]["manifest"]["sha256"],
        "adjudication_provenance": adjudication_provenance,
        "counts": summary["counts"],
        "artifacts": artifact_bindings,
        "authority": {
            "offline_only": True,
            "network_requests": 0,
            "credentials_read": False,
            "books_requested": False,
            "orders_submitted": 0,
            "economics_computed": False,
            "profitability_established": False,
            "live_eligible": False,
        },
    }
    manifest_content = _canonical_json_bytes(manifest)
    sidecar_content = f"{_sha256_bytes(manifest_content)}  manifest.json\n".encode("ascii")

    parent_parts = final_path.parent.relative_to(root).parts
    _ensure_real_directories(root, parent_parts)
    if _lstat_or_none(final_path) is not None or _lstat_or_none(staging_path) is not None:
        raise OfflineAdjudicationError("adjudication output or in-progress path already exists")
    try:
        os.mkdir(staging_path)
    except FileExistsError as exc:
        raise OfflineAdjudicationError("adjudication in-progress path already exists") from exc
    staging_status = _lstat_or_none(staging_path)
    if staging_status is None or _is_reparse_status(staging_status) or not stat.S_ISDIR(staging_status.st_mode):
        raise OfflineAdjudicationError("in-progress output is not a real directory")
    try:
        for name in ("source_bindings", "decisions", "summary"):
            _write_exclusive(staging_path / filenames[name], contents[name])
        _write_exclusive(staging_path / "manifest.json", manifest_content)
        _write_exclusive(staging_path / "manifest.sha256", sidecar_content)
        _sync_directory(staging_path)
        _reject_reparse_chain(
            root,
            parent_parts,
            label="output parent",
            require_all=True,
        )
        _rename_directory_no_replace(staging_path, final_path)
        _sync_directory(final_path.parent)
    except BaseException:
        failure_path = staging_path / "failure.json"
        failed_status = _lstat_or_none(staging_path)
        if (
            failed_status is not None
            and not _is_reparse_status(failed_status)
            and stat.S_ISDIR(failed_status.st_mode)
            and _lstat_or_none(failure_path) is None
        ):
            try:
                _write_exclusive(
                    failure_path,
                    _canonical_json_bytes(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "status": "ADJUDICATION_WRITE_FAILED_NO_RETRY",
                            "network_requests": 0,
                            "orders_submitted": 0,
                            "economics_computed": False,
                            "live_eligible": False,
                        }
                    ),
                )
                _sync_directory(staging_path)
            except (OSError, OfflineAdjudicationError):
                pass
        raise
    return final_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal one immutable offline-only PMXT adjudication")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL_PATH)
    args = parser.parse_args(argv)
    output = run_offline_adjudication(
        repository_root=args.repository_root,
        expected_protocol_sha256=args.expected_protocol_sha256,
        protocol_path=args.protocol_path,
    )
    manifest_snapshot = _read_input(output / "manifest.json", root=output, label="sealed manifest")
    manifest = _load_json_object(manifest_snapshot.payload, label="sealed manifest")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_id": manifest["run_id"],
                "rejected": manifest["counts"]["rejected"],
                "needs_review": manifest["counts"]["needs_review"],
                "orders_submitted": 0,
                "live_eligible": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["OfflineAdjudicationError", "run_offline_adjudication"]
