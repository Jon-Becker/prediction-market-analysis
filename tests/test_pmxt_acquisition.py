"""Offline tests for the isolated PMXT acquisition-control prototype."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import types
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import src.common.indexer as indexer_discovery_module
import src.indexers.pmxt.acquisition as acquisition_module
from src.common.indexer import Indexer
from src.indexers.pmxt.acquisition import (
    CONTROL_NAMESPACE,
    AcquisitionStateError,
    PermitValidationError,
    ReceiptValidationError,
    canonical_json_bytes,
    canonical_json_sha256,
    claim_acquisition,
    control_paths,
    evidence_sha256,
    load_acquisition_permit,
    write_http_receipt,
    write_terminal_receipt,
)

UTC = timezone.utc
NOW = datetime(2026, 8, 29, 18, 0, tzinfo=UTC)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE = "src.indexers.pmxt.acquisition"


def _static_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left)
        right = _static_string(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            part = _static_string(value)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    return None


def _module_name(path: Path) -> str:
    relative = path.relative_to(REPOSITORY_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_from_import(path: Path, node: ast.ImportFrom) -> str:
    module = node.module or ""
    if node.level == 0:
        return module
    current = _module_name(path)
    package_parts = current.split(".") if path.name == "__init__.py" else current.split(".")[:-1]
    remove = node.level - 1
    if remove > len(package_parts):
        return module
    prefix = package_parts[: len(package_parts) - remove]
    return ".".join([*prefix, *module.split(".")]) if module else ".".join(prefix)


def _targets_acquisition(module: str) -> bool:
    normalized = module.lstrip(".")
    return normalized == TARGET_MODULE or normalized == "indexers.pmxt.acquisition"


def _production_import_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
                if _targets_acquisition(alias.name):
                    violations.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_from_import(path, node)
            if node.level == 0 and node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        import_module_aliases.add(alias.asname or alias.name)
            if _targets_acquisition(resolved):
                violations.append(f"line {node.lineno}: from {resolved} import ...")
            elif resolved in {"src.indexers.pmxt", "indexers.pmxt"} and any(
                alias.name == "acquisition" for alias in node.names
            ):
                violations.append(f"line {node.lineno}: package import of acquisition")
        elif isinstance(node, ast.Call) and node.args:
            is_dynamic_import = (
                isinstance(node.func, ast.Name) and node.func.id in ({"__import__"} | import_module_aliases)
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in importlib_aliases
            )
            if not is_dynamic_import:
                continue
            module = _static_string(node.args[0])
            if module is None:
                continue
            if module.startswith("."):
                package = _static_string(node.args[1]) if len(node.args) > 1 else None
                if package is None:
                    package = next(
                        (_static_string(keyword.value) for keyword in node.keywords if keyword.arg == "package"),
                        None,
                    )
                if package == "src.indexers.pmxt" and module == ".acquisition":
                    violations.append(f"line {node.lineno}: dynamic relative import")
            elif _targets_acquisition(module):
                violations.append(f"line {node.lineno}: dynamic import {module}")
    return violations


def test_acquisition_module_is_explicitly_prototype_only() -> None:
    assert acquisition_module.PROTOTYPE_ONLY is True
    assert acquisition_module.PRODUCTION_READY is False
    assert acquisition_module.__doc__ is not None
    assert "PROTOTYPE ONLY" in acquisition_module.__doc__
    assert "do not grant network" in acquisition_module.__doc__
    assert "control an HTTP transport at most once" in acquisition_module.__doc__


def test_production_code_does_not_import_or_export_acquisition_prototype() -> None:
    paths = [REPOSITORY_ROOT / "main.py"]
    paths.extend(
        path
        for path in (REPOSITORY_ROOT / "src").rglob("*.py")
        if path.resolve() != Path(acquisition_module.__file__).resolve()
    )
    violations = {
        str(path.relative_to(REPOSITORY_ROOT)): found
        for path in paths
        if (found := _production_import_violations(path))
    }
    assert violations == {}

    init_path = REPOSITORY_ROOT / "src" / "indexers" / "pmxt" / "__init__.py"
    init_tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    exported: set[str] = set()
    for node in ast.walk(init_tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                    exported.update(
                        item.value
                        for item in value.elts
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    )
    assert "acquisition" not in exported
    assert TARGET_MODULE not in exported


def test_generic_indexer_discovery_does_not_import_quarantined_pmxt_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def record_import(module_name: str) -> types.SimpleNamespace:
        imported.append(module_name)
        return types.SimpleNamespace()

    monkeypatch.setattr(indexer_discovery_module.importlib, "import_module", record_import)
    assert Indexer.load(REPOSITORY_ROOT / "src" / "indexers") == []
    assert "src.indexers.pmxt.market_clusters" in imported
    assert "src.indexers.pmxt.acquisition" not in imported
    assert "src.indexers.pmxt.semantic_equivalence" not in imported


def _permit_document(control_root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": "pmxt-router-at-most-once-v1",
        "issued_at_utc": "2026-08-29T17:59:00Z",
        "expires_at_utc": "2026-08-29T18:10:00Z",
        "control_root": str(control_root.resolve()),
        "request": {
            "method": "GET",
            "origin": "https://api.pmxt.dev",
            "path": "/v0/matched-market-clusters",
            "query": {
                "relation": "identity",
                "minConfidence": "0.8",
                "minVenues": "2",
                "includeRawMatches": "true",
                "sort": "volume",
                "venues": "kalshi,polymarket",
                "limit": "25",
                "offset": "0",
            },
            "max_attempts": 1,
            "retries": 0,
            "redirects": False,
            "timeout_seconds": 20,
            "bounds": {
                "maximum_response_bytes": 5_000_000,
                "maximum_clusters": 25,
                "maximum_candidates": 25,
                "maximum_markets_per_cluster": 25,
                "maximum_raw_matches_per_cluster": 25,
            },
        },
        "zero_spend_evidence": {
            "provider": "PMXT",
            "authority": "PMXT_PROVIDER_BILLING",
            "provider_authoritative": True,
            "evidence_id": "usage-snapshot-20260829T175800Z",
            "evidence_sha256": "1" * 64,
            "account_scope_sha256": "2" * 64,
            "observed_at_utc": "2026-08-29T17:58:00Z",
            "valid_until_utc": "2026-08-29T18:15:00Z",
            "plan": "FREE",
            "currency": "USD",
            "incremental_charge_usd": "0.00",
            "credits_required": 1,
            "credits_remaining": 24_999,
            "overage_enabled": False,
        },
        "authority": {
            "purpose": "RESEARCH_ONLY_DISCOVERY",
            "network_read_authorized": True,
            "maximum_pmxt_requests": 1,
            "orders_authorized": False,
            "venue_account_access_authorized": False,
            "credential_persistence_authorized": False,
            "live_execution_authorized": False,
        },
    }


def _write_permit(path: Path, document: dict[str, Any], *, pretty: bool = True) -> str:
    if pretty:
        raw = (json.dumps(document, indent=2, sort_keys=False) + "\n").encode("utf-8")
    else:
        raw = canonical_json_bytes(document)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _loaded_permit(tmp_path: Path):
    control_root = tmp_path / "control"
    control_root.mkdir()
    path = tmp_path / "permit.json"
    digest = _write_permit(path, _permit_document(control_root))
    return load_acquisition_permit(path, expected_sha256=digest, now_utc=NOW)


def test_loader_authenticates_raw_bytes_and_exposes_stable_hashes(tmp_path: Path) -> None:
    document = _permit_document(tmp_path / "control")
    path = tmp_path / "permit.json"
    digest = _write_permit(path, document)

    permit = load_acquisition_permit(path, expected_sha256=digest, now_utc=NOW)

    assert permit.permit_sha256 == digest
    assert permit.canonical_sha256 == canonical_json_sha256(document)
    assert permit.request_sha256 == evidence_sha256("request", document["request"])
    assert permit.query_params() == document["request"]["query"]
    assert permit.maximum_response_bytes == 5_000_000
    assert permit.maximum_clusters == permit.maximum_candidates == 25

    reordered = {key: document[key] for key in reversed(document)}
    assert canonical_json_sha256(reordered) == permit.canonical_sha256
    assert evidence_sha256("request", reordered) != permit.request_sha256


def test_wrong_digest_and_non_strict_json_fail_before_schema_use(tmp_path: Path) -> None:
    path = tmp_path / "permit.json"
    digest = _write_permit(path, _permit_document(tmp_path / "control"))
    with pytest.raises(PermitValidationError, match="SHA-256 mismatch"):
        load_acquisition_permit(path, expected_sha256="0" * 64, now_utc=NOW)

    duplicate = b'{"schema_version":1,"schema_version":1}\n'
    path.write_bytes(duplicate)
    with pytest.raises(PermitValidationError, match="strict JSON"):
        load_acquisition_permit(
            path,
            expected_sha256=hashlib.sha256(duplicate).hexdigest(),
            now_utc=NOW,
        )
    assert digest != hashlib.sha256(duplicate).hexdigest()


@pytest.mark.parametrize(
    "case",
    [
        "extra_top_level",
        "relative_control_root",
        "post",
        "foreign_origin",
        "non_identity",
        "nonzero_offset",
        "limit_over_25",
        "retry",
        "redirect",
        "two_attempts",
        "byte_bound_too_large",
        "fanout_too_large",
        "missing_zero_spend_field",
        "self_asserted_billing",
        "paid_plan",
        "nonzero_charge",
        "no_credits",
        "orders_authorized",
    ],
)
def test_malformed_or_weakened_permit_fails_closed(tmp_path: Path, case: str) -> None:
    document = _permit_document(tmp_path / "control")
    if case == "extra_top_level":
        document["extra"] = True
    elif case == "relative_control_root":
        document["control_root"] = "relative/control"
    elif case == "post":
        document["request"]["method"] = "POST"
    elif case == "foreign_origin":
        document["request"]["origin"] = "https://example.test"
    elif case == "non_identity":
        document["request"]["query"]["relation"] = "overlap"
    elif case == "nonzero_offset":
        document["request"]["query"]["offset"] = "1"
    elif case == "limit_over_25":
        document["request"]["query"]["limit"] = "26"
        document["request"]["bounds"]["maximum_clusters"] = 26
    elif case == "retry":
        document["request"]["retries"] = 1
    elif case == "redirect":
        document["request"]["redirects"] = True
    elif case == "two_attempts":
        document["request"]["max_attempts"] = 2
    elif case == "byte_bound_too_large":
        document["request"]["bounds"]["maximum_response_bytes"] = 5_000_001
    elif case == "fanout_too_large":
        document["request"]["bounds"]["maximum_candidates"] = 26
    elif case == "missing_zero_spend_field":
        del document["zero_spend_evidence"]["evidence_sha256"]
    elif case == "self_asserted_billing":
        document["zero_spend_evidence"]["provider_authoritative"] = False
    elif case == "paid_plan":
        document["zero_spend_evidence"]["plan"] = "PRO"
    elif case == "nonzero_charge":
        document["zero_spend_evidence"]["incremental_charge_usd"] = "0.01"
    elif case == "no_credits":
        document["zero_spend_evidence"]["credits_remaining"] = 0
    elif case == "orders_authorized":
        document["authority"]["orders_authorized"] = True

    path = tmp_path / f"{case}.json"
    digest = _write_permit(path, document)
    with pytest.raises(PermitValidationError):
        load_acquisition_permit(path, expected_sha256=digest, now_utc=NOW)


def test_stale_permit_or_provider_evidence_cannot_be_loaded_or_claimed(tmp_path: Path) -> None:
    path = tmp_path / "permit.json"
    control_root = tmp_path / "control"
    control_root.mkdir()
    digest = _write_permit(path, _permit_document(control_root))
    with pytest.raises(PermitValidationError, match="expired"):
        load_acquisition_permit(path, expected_sha256=digest, now_utc=NOW + timedelta(minutes=11))

    permit = load_acquisition_permit(path, expected_sha256=digest, now_utc=NOW)
    with pytest.raises(PermitValidationError, match="expired before claim"):
        claim_acquisition(permit, claimed_at_utc=NOW + timedelta(minutes=11))
    assert not (control_root / CONTROL_NAMESPACE).exists()


def test_claim_has_fixed_digest_path_and_returns_only_after_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    permit = _loaded_permit(tmp_path)
    control_root = permit.control_root
    calls: list[str] = []
    real_flush = acquisition_module._flush_and_fsync

    def recording_flush(handle: Any) -> None:
        real_flush(handle)
        calls.append("fsynced")

    monkeypatch.setattr(acquisition_module, "_flush_and_fsync", recording_flush)
    claim = claim_acquisition(permit, claimed_at_utc=NOW)

    expected_directory = control_root.resolve() / CONTROL_NAMESPACE / permit.permit_sha256
    assert calls == ["fsynced"]
    assert claim.paths.directory == expected_directory
    assert claim.paths.claim == expected_directory / "claim.json"
    assert hashlib.sha256(claim.paths.claim.read_bytes()).hexdigest() == claim.claim_sha256
    assert not claim.paths.http_receipt.exists()
    assert not claim.paths.terminal_receipt.exists()


def test_control_paths_are_deterministic_before_namespace_creation(tmp_path: Path) -> None:
    permit = _loaded_permit(tmp_path)
    paths = control_paths(permit.control_root, permit.permit_sha256)

    expected = permit.control_root / CONTROL_NAMESPACE / permit.permit_sha256
    assert paths.directory == expected
    assert paths.claim == expected / "claim.json"
    assert paths.http_receipt == expected / "http_receipt.json"
    assert paths.terminal_receipt == expected / "terminal_receipt.json"
    assert not (permit.control_root / CONTROL_NAMESPACE).exists()


def test_permit_mutation_between_load_and_claim_fails_before_control_state(tmp_path: Path) -> None:
    permit = _loaded_permit(tmp_path)
    permit.path.write_bytes(permit.raw_bytes + b" ")

    with pytest.raises(PermitValidationError, match="changed after authentication"):
        claim_acquisition(permit, claimed_at_utc=NOW)
    assert not (permit.control_root / CONTROL_NAMESPACE).exists()


def test_concurrent_claim_allows_exactly_one_winner(tmp_path: Path) -> None:
    permit = _loaded_permit(tmp_path)
    control_root = permit.control_root

    def attempt(_: int) -> str:
        try:
            claim_acquisition(permit, claimed_at_utc=NOW)
        except AcquisitionStateError:
            return "rejected"
        return "claimed"

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(attempt, range(24)))

    assert results.count("claimed") == 1
    assert results.count("rejected") == 23
    claim_files = list((control_root / CONTROL_NAMESPACE).glob("*/claim.json"))
    assert len(claim_files) == 1


def test_duplicate_and_crash_residue_permanently_reject_rerun(tmp_path: Path) -> None:
    permit = _loaded_permit(tmp_path)
    claim = claim_acquisition(permit, claimed_at_utc=NOW)
    before = claim.paths.claim.read_bytes()

    # Simulate an arbitrary later caller failure.  The gate exposes no release
    # or cleanup primitive, so the durable claim remains the at-most-once fact.
    with pytest.raises(RuntimeError, match="later failure"):
        raise RuntimeError("simulated later failure")
    with pytest.raises(AcquisitionStateError, match="already claimed|crash residue"):
        claim_acquisition(permit, claimed_at_utc=NOW + timedelta(seconds=1))
    assert claim.paths.claim.read_bytes() == before

    # The digest-keyed directory itself is sufficient crash residue, even when
    # a process died before it could write claim.json.
    residue_document = _permit_document(tmp_path / "residue-control")
    residue_root = Path(residue_document["control_root"])
    residue_root.mkdir()
    residue_path = tmp_path / "residue-permit.json"
    residue_digest = _write_permit(residue_path, residue_document)
    residue_permit = load_acquisition_permit(residue_path, expected_sha256=residue_digest, now_utc=NOW)
    namespace = residue_permit.control_root / CONTROL_NAMESPACE
    namespace.mkdir()
    (namespace / residue_permit.permit_sha256).mkdir()
    with pytest.raises(AcquisitionStateError, match="already claimed|crash residue"):
        claim_acquisition(residue_permit, claimed_at_utc=NOW)


def test_http_and_terminal_receipts_are_hash_bound_write_once_and_auth_free(tmp_path: Path) -> None:
    permit = _loaded_permit(tmp_path)
    claim = claim_acquisition(permit, claimed_at_utc=NOW)
    body = b'{"clusters":[]}'

    http_receipt = write_http_receipt(
        claim,
        request_started_at_utc=NOW + timedelta(seconds=1),
        response_received_at_utc=NOW + timedelta(seconds=2),
        status_code=200,
        response_byte_count=len(body),
        response_sha256=hashlib.sha256(body).hexdigest(),
    )
    http_before = http_receipt.path.read_bytes()
    with pytest.raises(AcquisitionStateError, match="overwrite"):
        write_http_receipt(
            claim,
            request_started_at_utc=NOW + timedelta(seconds=1),
            response_received_at_utc=NOW + timedelta(seconds=3),
            status_code=200,
            response_byte_count=len(body),
            response_sha256=hashlib.sha256(body).hexdigest(),
        )
    assert http_receipt.path.read_bytes() == http_before

    terminal = write_terminal_receipt(
        claim,
        finished_at_utc=NOW + timedelta(seconds=4),
        status="COMPLETED",
        reason_code="HTTP_CAPTURE_COMPLETE",
        transport_attempted=True,
    )
    terminal_before = terminal.path.read_bytes()
    with pytest.raises(AcquisitionStateError, match="overwrite"):
        write_terminal_receipt(
            claim,
            finished_at_utc=NOW + timedelta(seconds=5),
            status="COMPLETED",
            reason_code="SECOND_WRITE",
            transport_attempted=True,
        )
    assert terminal.path.read_bytes() == terminal_before

    parsed_terminal = json.loads(terminal_before)
    assert parsed_terminal["http_receipt_sha256"] == hashlib.sha256(http_before).hexdigest()
    assert http_receipt.sha256 == hashlib.sha256(http_before).hexdigest()
    assert terminal.sha256 == hashlib.sha256(terminal_before).hexdigest()

    combined = b"\n".join(path.read_bytes() for path in sorted(claim.paths.directory.iterdir())).lower()
    for forbidden in (b"authorization", b"bearer ", b"api_key", b"password", b"cookie", b"private_key"):
        assert forbidden not in combined
    assert body not in combined


def test_receipt_bounds_and_terminal_state_rules_fail_closed(tmp_path: Path) -> None:
    permit = _loaded_permit(tmp_path)
    claim = claim_acquisition(permit, claimed_at_utc=NOW)

    with pytest.raises(ReceiptValidationError, match="exceeds"):
        write_http_receipt(
            claim,
            request_started_at_utc=NOW + timedelta(seconds=1),
            response_received_at_utc=NOW + timedelta(seconds=2),
            status_code=200,
            response_byte_count=permit.maximum_response_bytes + 1,
            response_sha256="3" * 64,
        )
    with pytest.raises(ReceiptValidationError, match="requires an immutable HTTP receipt"):
        write_terminal_receipt(
            claim,
            finished_at_utc=NOW + timedelta(seconds=3),
            status="COMPLETED",
            reason_code="NO_HTTP_RECEIPT",
            transport_attempted=True,
        )

    terminal = write_terminal_receipt(
        claim,
        finished_at_utc=NOW + timedelta(seconds=4),
        status="ABORTED_BEFORE_TRANSPORT",
        reason_code="LOCAL_PREFLIGHT_FAILED",
        transport_attempted=False,
    )
    assert json.loads(terminal.path.read_text(encoding="utf-8"))["claim_released"] is False
    with pytest.raises(AcquisitionStateError, match="terminal receipt already exists"):
        write_http_receipt(
            claim,
            request_started_at_utc=NOW + timedelta(seconds=5),
            response_received_at_utc=NOW + timedelta(seconds=6),
            status_code=200,
            response_byte_count=0,
            response_sha256=hashlib.sha256(b"").hexdigest(),
        )


def test_claim_and_receipt_tampering_is_detected(tmp_path: Path) -> None:
    permit = _loaded_permit(tmp_path)
    claim = claim_acquisition(permit, claimed_at_utc=NOW)
    claim_document = json.loads(claim.paths.claim.read_text(encoding="utf-8"))
    claim_document["orders_authorized"] = True
    claim.paths.claim.write_bytes(canonical_json_bytes(claim_document) + b"\n")

    with pytest.raises(AcquisitionStateError, match="hash mismatch|bytes changed"):
        write_terminal_receipt(
            claim,
            finished_at_utc=NOW + timedelta(seconds=1),
            status="ABORTED_BEFORE_TRANSPORT",
            reason_code="LOCAL_FAILURE",
            transport_attempted=False,
        )


def test_permit_document_is_deeply_read_only(tmp_path: Path) -> None:
    permit = _loaded_permit(tmp_path)
    with pytest.raises(TypeError):
        permit.document["protocol_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        permit.document["request"]["query"]["limit"] = "1"  # type: ignore[index]
    query = permit.query_params()
    query["limit"] = "1"
    assert permit.query_params()["limit"] == "25"


def test_provider_evidence_window_must_cover_permit(tmp_path: Path) -> None:
    document = copy.deepcopy(_permit_document(tmp_path / "control"))
    document["zero_spend_evidence"]["valid_until_utc"] = "2026-08-29T18:05:00Z"
    path = tmp_path / "permit.json"
    digest = _write_permit(path, document)
    with pytest.raises(PermitValidationError, match="complete permit window"):
        load_acquisition_permit(path, expected_sha256=digest, now_utc=NOW)
