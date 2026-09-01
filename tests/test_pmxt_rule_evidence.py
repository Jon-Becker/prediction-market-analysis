"""Fixture-first tests for the bounded native rule-evidence capture."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
from pathlib import Path
from typing import Any, BinaryIO, Callable

import httpx
import pytest

import src.indexers.pmxt.rule_evidence as rule_evidence_module
from src.indexers.pmxt.rule_evidence import (
    RuleEvidenceCaptureError,
    run_rule_evidence_capture,
)

SAFETY = {
    "books_requested": False,
    "credentials_read": False,
    "economics_computed": False,
    "fees_requested": False,
    "live_eligible": False,
    "orders_submitted": 0,
    "pmxt_requests": 0,
    "positions_requested": False,
    "semantic_promotions": 0,
}

_TRANSPORT_LOCAL = threading.local()


def _offline_transport_constructor(**kwargs: Any) -> httpx.BaseTransport:
    factory = getattr(_TRANSPORT_LOCAL, "factory", None)
    if factory is None:
        raise AssertionError("offline test transport was not installed for this thread")
    return factory(**kwargs)


@pytest.fixture(autouse=True)
def _forbid_real_http_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rule_evidence_module.httpx, "HTTPTransport", _offline_transport_constructor)


MLB_IDS = [
    "pmxt_candidate_8d5742f402659dabd733",
    "pmxt_candidate_721ba66cd3a45a1b0671",
    "pmxt_candidate_fd848ee9c295396e2f3a",
    "pmxt_candidate_57ae4edbdd1c3c89ac81",
]
NFL_IDS = [
    "pmxt_candidate_690e0d3c9bf11cf77133",
    "pmxt_candidate_47e1a797ca569a3e085d",
    "pmxt_candidate_5f2429d1c418b0b9e5f3",
    "pmxt_candidate_7bb4f3faaa25fc57c4ee",
    "pmxt_candidate_cf808058738e89a67c1a",
]
NOMINEE_IDS = [
    "pmxt_candidate_68f36067448490a9ed48",
    "pmxt_candidate_093b3d59fde18b10a289",
    "pmxt_candidate_154820cbf06559e5b7c1",
]
GOVERNOR_IDS = [
    "pmxt_candidate_8756452d08adab8a6bdb",
    "pmxt_candidate_b8aef6caa557ab7b4434",
]
REVIEW_IDS = MLB_IDS + NFL_IDS + NOMINEE_IDS + GOVERNOR_IDS

SPORT_GAPS = ["terminal_no_winner_treatment"]
NOMINEE_GAPS = [
    "deadline_role",
    "external_terms",
    "terminal_rules",
    "resolution_authority",
    "group_set_and_other",
]
GOVERNOR_GAPS = [
    "accelerated_trigger_and_source_set",
    "deadline_role",
    "terminal_and_void_cancel",
    "resolution_authority_and_fallback",
]


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, value: Any) -> bytes:
    payload = _canonical(value)
    _write(path, payload)
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> tuple[bytes, list[dict[str, Any]]]:
    encoded = [_canonical(row) for row in rows]
    payload = b"".join(encoded)
    _write(path, payload)
    references = [{"line": line, "row_sha256": _sha256(raw)} for line, raw in enumerate(encoded, start=1)]
    return payload, references


def _binding(root: Path, path: Path, payload: bytes | None = None) -> dict[str, Any]:
    value = path.read_bytes() if payload is None else payload
    return {
        "path": path.relative_to(root).as_posix(),
        "byte_size": len(value),
        "sha256": _sha256(value),
    }


def _group(candidate_id: str) -> tuple[str, str, str, list[str]]:
    if candidate_id in MLB_IDS:
        return "TITLE", "179312", "polymarket-event-179312", SPORT_GAPS
    if candidate_id in NFL_IDS:
        return "TITLE", "202857", "polymarket-event-202857", SPORT_GAPS
    if candidate_id in NOMINEE_IDS:
        return "PRESNOM", "31875", "polymarket-event-31875", NOMINEE_GAPS
    event_id = "59234" if candidate_id == GOVERNOR_IDS[0] else "57096"
    resource_id = f"polymarket-event-{event_id}"
    return "GOV", event_id, resource_id, GOVERNOR_GAPS


def _make_repository(tmp_path: Path) -> tuple[Path, dict[str, Any], Path]:
    root = tmp_path / "repo"
    implementation_source = Path(rule_evidence_module.__file__).read_bytes()
    implementation_path = root / "src/indexers/pmxt/rule_evidence.py"
    _write(implementation_path, implementation_source)

    candidates = [{"candidate_id": candidate_id} for candidate_id in REVIEW_IDS]
    semantics = [{"candidate_id": candidate_id} for candidate_id in REVIEW_IDS]
    metadata: list[dict[str, Any]] = []
    candidate_details: dict[str, dict[str, Any]] = {}
    for index, candidate_id in enumerate(REVIEW_IDS, start=1):
        series, event_id, gamma_resource, gaps = _group(candidate_id)
        contract_url = f"https://assets.kalshi.com/contract_terms/{series}.pdf"
        kalshi_raw = {"candidate_id": candidate_id, "venue": "kalshi"}
        kalshi_rules = {"series_contract_terms_url": contract_url}
        polymarket_raw = {"candidate_id": candidate_id, "venue": "polymarket"}
        polymarket_rules = {"description": "fixture"}
        kalshi = {
            "candidate_id": candidate_id,
            "venue": "kalshi",
            "live_eligible": False,
            "native_market_id": f"K-MARKET-{index}",
            "native_event_id": f"K-EVENT-{index}",
            "native_series_id": series,
            "raw_sha256": _sha256(_canonical(kalshi_raw)[:-1]),
            "rule_hash": _sha256(_canonical(kalshi_rules)[:-1]),
            "normalized_rules": kalshi_rules,
        }
        polymarket = {
            "candidate_id": candidate_id,
            "venue": "polymarket",
            "live_eligible": False,
            "native_market_id": str(700000 + index),
            "native_event_id": event_id,
            "raw_sha256": _sha256(_canonical(polymarket_raw)[:-1]),
            "rule_hash": _sha256(_canonical(polymarket_rules)[:-1]),
            "normalized_rules": polymarket_rules,
        }
        metadata.extend((kalshi, polymarket))
        candidate_details[candidate_id] = {
            "kalshi": kalshi,
            "polymarket": polymarket,
            "contract_url": contract_url,
            "gamma_resource": gamma_resource,
            "gap_axes": list(gaps),
        }

    monitor_dir = root / "data/pmxt/runs/monitor_fixture"
    candidate_path = monitor_dir / "candidates.jsonl"
    metadata_path = monitor_dir / "raw_native_metadata.jsonl"
    semantic_path = monitor_dir / "semantic_decisions.jsonl"
    candidate_payload, candidate_refs = _write_jsonl(candidate_path, candidates)
    metadata_payload, metadata_refs = _write_jsonl(metadata_path, metadata)
    semantic_payload, semantic_refs = _write_jsonl(semantic_path, semantics)
    monitor_artifacts = {
        "candidates": _binding(root, candidate_path, candidate_payload),
        "raw_native_metadata": _binding(root, metadata_path, metadata_payload),
        "semantic_decisions": _binding(root, semantic_path, semantic_payload),
    }
    monitor_manifest_artifacts = {
        name: {**binding, "path": Path(binding["path"]).name} for name, binding in monitor_artifacts.items()
    }
    monitor_manifest_path = monitor_dir / "manifest.json"
    monitor_manifest_payload = _write_json(
        monitor_manifest_path,
        {
            "run_id": "monitor_fixture",
            "artifacts": monitor_manifest_artifacts,
            "counts": {"native_book_attempts": 0, "pmxt_network_requests": 0},
            "live_eligible": False,
            "no_order_actions": True,
        },
    )

    decisions: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(REVIEW_IDS):
        details = candidate_details[candidate_id]
        kalshi = details["kalshi"]
        polymarket = details["polymarket"]
        kalshi_ref = {
            **metadata_refs[index * 2],
            "raw_sha256": kalshi["raw_sha256"],
            "rule_hash": kalshi["rule_hash"],
        }
        polymarket_ref = {
            **metadata_refs[index * 2 + 1],
            "raw_sha256": polymarket["raw_sha256"],
            "rule_hash": polymarket["rule_hash"],
        }
        decisions.append(
            {
                "candidate_id": candidate_id,
                "decision": "NEEDS_REVIEW",
                "semantic_verified": False,
                "live_eligible": False,
                "orders_submitted": 0,
                "evidence_gaps": [{"axis": axis} for axis in details["gap_axes"]],
                "source_rows": {
                    "candidate": candidate_refs[index],
                    "automated_decision": semantic_refs[index],
                    "kalshi_metadata": kalshi_ref,
                    "polymarket_metadata": polymarket_ref,
                },
            }
        )
    decisions.extend(
        {
            "candidate_id": f"pmxt_candidate_deadbeef{index:012x}",
            "decision": "REJECTED",
        }
        for index in range(11)
    )
    adjudication_dir = root / "data/pmxt/semantic_adjudication/runs/adjudication_fixture"
    decision_path = adjudication_dir / "decisions.jsonl"
    source_bindings_path = adjudication_dir / "source_bindings.json"
    summary_path = adjudication_dir / "summary.json"
    decision_payload, decision_refs = _write_jsonl(decision_path, decisions)
    source_bindings_payload = _write_json(
        source_bindings_path,
        {
            "source_monitor_run": {
                "run_id": "monitor_fixture",
                "manifest": {"sha256": _sha256(monitor_manifest_payload)},
            }
        },
    )
    summary_payload = _write_json(
        summary_path,
        {"counts": {"needs_review": 14, "verified_equivalent": 0}},
    )
    adjudication_artifacts = {
        "decisions": _binding(root, decision_path, decision_payload),
        "source_bindings": _binding(root, source_bindings_path, source_bindings_payload),
        "summary": _binding(root, summary_path, summary_payload),
    }
    adjudication_manifest_artifacts = {
        name: {**binding, "path": Path(binding["path"]).name} for name, binding in adjudication_artifacts.items()
    }
    adjudication_manifest_path = adjudication_dir / "manifest.json"
    adjudication_manifest_payload = _write_json(
        adjudication_manifest_path,
        {
            "protocol_id": "pmxt-offline-semantic-adjudication-v1",
            "run_id": "adjudication_fixture",
            "source_run_id": "monitor_fixture",
            "source_manifest_sha256": _sha256(monitor_manifest_payload),
            "counts": {"total": 25, "rejected": 11, "needs_review": 14},
            "authority": {
                "network_requests": 0,
                "orders_submitted": 0,
                "live_eligible": False,
                "economics_computed": False,
            },
            "artifacts": adjudication_manifest_artifacts,
        },
    )

    candidate_bindings = []
    for index, candidate_id in enumerate(REVIEW_IDS):
        details = candidate_details[candidate_id]
        kalshi = details["kalshi"]
        polymarket = details["polymarket"]
        candidate_bindings.append(
            {
                "candidate_id": candidate_id,
                "adjudication_decision_row": decision_refs[index],
                "candidate_row": candidate_refs[index],
                "kalshi_metadata_row": {
                    **metadata_refs[index * 2],
                    "raw_sha256": kalshi["raw_sha256"],
                    "rule_hash": kalshi["rule_hash"],
                },
                "polymarket_metadata_row": {
                    **metadata_refs[index * 2 + 1],
                    "raw_sha256": polymarket["raw_sha256"],
                    "rule_hash": polymarket["rule_hash"],
                },
                "automated_decision_row": semantic_refs[index],
                "gap_axes": details["gap_axes"],
                "resource_ids": [
                    {
                        "TITLE": "kalshi-title-terms",
                        "PRESNOM": "kalshi-presnom-terms",
                        "GOV": "kalshi-gov-terms",
                    }[kalshi["native_series_id"]],
                    details["gamma_resource"],
                ],
                "native_ids": {
                    "kalshi_market_id": kalshi["native_market_id"],
                    "kalshi_event_id": kalshi["native_event_id"],
                    "kalshi_series_id": kalshi["native_series_id"],
                    "polymarket_market_id": polymarket["native_market_id"],
                    "polymarket_event_id": polymarket["native_event_id"],
                },
                "kalshi_contract_terms_url": details["contract_url"],
            }
        )

    resource_groups = [
        (1, "kalshi-title-terms", "kalshi", "pdf", "TITLE", MLB_IDS + NFL_IDS, None),
        (2, "kalshi-presnom-terms", "kalshi", "pdf", "PRESNOM", NOMINEE_IDS, None),
        (3, "kalshi-gov-terms", "kalshi", "pdf", "GOV", GOVERNOR_IDS, None),
        (4, "polymarket-event-179312", "polymarket", "json", None, MLB_IDS, "179312"),
        (5, "polymarket-event-202857", "polymarket", "json", None, NFL_IDS, "202857"),
        (6, "polymarket-event-31875", "polymarket", "json", None, NOMINEE_IDS, "31875"),
        (7, "polymarket-event-59234", "polymarket", "json", None, [GOVERNOR_IDS[0]], "59234"),
        (8, "polymarket-event-57096", "polymarket", "json", None, [GOVERNOR_IDS[1]], "57096"),
    ]
    resources = []
    details_by_id = candidate_details
    for ordinal, resource_id, venue, kind, series, candidate_ids, event_id in resource_groups:
        url = (
            f"https://assets.kalshi.com/contract_terms/{series}.pdf"
            if kind == "pdf"
            else f"https://gamma-api.polymarket.com/events/{event_id}"
        )
        resources.append(
            {
                "ordinal": ordinal,
                "resource_id": resource_id,
                "venue": venue,
                "kind": kind,
                "url": url,
                "expected_media_type": "application/pdf" if kind == "pdf" else "application/json",
                "expected_event_id": event_id,
                "candidate_ids": list(candidate_ids),
                "expected_market_ids": []
                if kind == "pdf"
                else [details_by_id[candidate_id]["polymarket"]["native_market_id"] for candidate_id in candidate_ids],
                "gap_axes": sorted(
                    {axis for candidate_id in candidate_ids for axis in details_by_id[candidate_id]["gap_axes"]}
                ),
            }
        )

    schema_path = root / "results/rule_evidence_v1/resource_record_schema_v1.json"
    schema_payload = _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )
    protocol = {
        "schema_version": 1,
        "protocol_id": "pmxt-native-rule-evidence-capture-v1",
        "authority": dict(SAFETY),
        "runtime_dependencies": {"httpcore": "1.0.9", "httpx": "0.28.1"},
        "limits": {
            "inactivity_timeout_seconds": 20,
            "max_resource_bytes": 5_000_000,
            "resource_elapsed_checkpoint_seconds": 30,
            "run_elapsed_checkpoint_seconds": 240,
            "max_total_bytes": 40_000_000,
            "request_count": 8,
            "retries": 0,
            "sequential": True,
        },
        "implementation": {
            "identifier": "pmxt-rule-evidence-capture-v1",
            **_binding(root, implementation_path, implementation_source),
        },
        "resource_record_schema": _binding(root, schema_path, schema_payload),
        "source_adjudication_run": {
            "run_id": "adjudication_fixture",
            "manifest": _binding(root, adjudication_manifest_path, adjudication_manifest_payload),
            "artifacts": adjudication_artifacts,
        },
        "source_monitor_run": {
            "run_id": "monitor_fixture",
            "manifest": _binding(root, monitor_manifest_path, monitor_manifest_payload),
            "artifacts": monitor_artifacts,
        },
        "candidate_bindings": candidate_bindings,
        "resources": resources,
        "output": {
            "run_id": "fixture_rule_evidence",
            "path": "data/pmxt/rule_evidence/runs/fixture_rule_evidence",
        },
    }
    protocol_path = root / "results/rule_evidence_v1/protocol_fixture.json"
    _write_json(protocol_path, protocol)
    return root, protocol, protocol_path


def _rewrite_protocol(protocol_path: Path, protocol: dict[str, Any]) -> str:
    payload = _write_json(protocol_path, protocol)
    return _sha256(payload)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _RecordingTransportFactory:
    def __init__(
        self,
        protocol: dict[str, Any],
        responder: Callable[[httpx.Request, dict[str, Any], int], httpx.Response] | None = None,
    ) -> None:
        self.protocol = protocol
        self.responder = responder
        self.factory_calls: list[dict[str, Any]] = []
        self.requests: list[httpx.Request] = []
        self.resources = {resource["url"]: resource for resource in protocol["resources"]}

    def __call__(self, **kwargs: Any) -> httpx.BaseTransport:
        self.factory_calls.append(dict(kwargs))
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        resource = self.resources[str(request.url)]
        if self.responder is not None:
            return self.responder(request, resource, len(self.requests))
        return _valid_response(request, resource)


def _valid_body(resource: dict[str, Any]) -> bytes:
    if resource["kind"] == "pdf":
        return f"%PDF-1.7\nfixture:{resource['resource_id']}\n%%EOF\n".encode()
    return json.dumps(
        {
            "id": resource["expected_event_id"],
            "markets": [{"id": market_id} for market_id in resource["expected_market_ids"]],
        },
        separators=(",", ":"),
    ).encode()


def _valid_response(request: httpx.Request, resource: dict[str, Any]) -> httpx.Response:
    body = _valid_body(resource)
    headers = [
        ("Content-Type", resource["expected_media_type"] + "; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("X-Request-ID", f"request-{resource['ordinal']}"),
        ("ETag", f'"etag-{resource["ordinal"]}"'),
        ("Set-Cookie", "must-not-be-retained=1"),
        ("X-Untrusted", "must-not-be-retained"),
    ]
    return httpx.Response(200, headers=headers, stream=httpx.ByteStream(body), request=request)


def _run(
    root: Path,
    protocol: dict[str, Any],
    protocol_path: Path,
    factory: Callable[..., httpx.BaseTransport],
    *,
    network_authorized: bool = True,
) -> Path:
    protocol_hash = _rewrite_protocol(protocol_path, protocol)
    previous = getattr(_TRANSPORT_LOCAL, "factory", None)
    _TRANSPORT_LOCAL.factory = factory
    try:
        return run_rule_evidence_capture(
            repository_root=root,
            protocol_path=protocol_path.relative_to(root),
            expected_protocol_sha256=protocol_hash,
            network_authorized=network_authorized,
        )
    finally:
        if previous is None:
            delattr(_TRANSPORT_LOCAL, "factory")
        else:
            _TRANSPORT_LOCAL.factory = previous


def _assert_safety(document: dict[str, Any]) -> None:
    assert {key: document[key] for key in SAFETY} == SAFETY


def test_exact_mapping_transport_artifacts_manifest_and_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    transport_factory = _RecordingTransportFactory(protocol)
    original_client = httpx.Client
    client_kwargs: list[dict[str, Any]] = []

    def client_spy(*args: Any, **kwargs: Any) -> httpx.Client:
        client_kwargs.append(dict(kwargs))
        return original_client(*args, **kwargs)

    monkeypatch.setattr(rule_evidence_module.httpx, "Client", client_spy)

    output = _run(root, protocol, protocol_path, transport_factory)

    assert [str(request.url) for request in transport_factory.requests] == [
        resource["url"] for resource in protocol["resources"]
    ]
    assert len(transport_factory.requests) == len(transport_factory.factory_calls) == 8
    assert transport_factory.factory_calls == [{"retries": 0, "trust_env": False}] * 8
    assert all(kwargs["trust_env"] is False for kwargs in client_kwargs)
    assert all(kwargs["follow_redirects"] is False for kwargs in client_kwargs)
    for request, resource in zip(transport_factory.requests, protocol["resources"]):
        assert request.method == "GET"
        assert request.url.scheme == "https"
        assert request.url.host in {"assets.kalshi.com", "gamma-api.polymarket.com"}
        assert request.url.port is None and not request.url.query
        assert request.headers["accept"] == resource["expected_media_type"]
        assert request.headers["accept-encoding"] == "identity"
        assert not ({"authorization", "cookie", "proxy-authorization"} & set(request.headers))

    plan = _read_json(output / "capture_plan.json")
    assert [resource["resource_id"] for resource in plan["resources"]] == [
        "kalshi-title-terms",
        "kalshi-presnom-terms",
        "kalshi-gov-terms",
        "polymarket-event-179312",
        "polymarket-event-202857",
        "polymarket-event-31875",
        "polymarket-event-59234",
        "polymarket-event-57096",
    ]
    assert [len(resource["candidate_ids"]) for resource in plan["resources"]] == [9, 3, 2, 4, 5, 3, 1, 1]
    assert plan["transport"]["hard_wall_clock_deadline_enforced"] is False
    assert "do not preempt blocking DNS" in plan["transport"]["elapsed_checkpoint_semantics"]
    candidate_rows = _read_jsonl(output / "candidate_gap_index.jsonl")
    assert len(candidate_rows) == 14
    assert {row["candidate_id"] for row in candidate_rows} == set(REVIEW_IDS)
    assert all(len(row["resource_ids"]) == 2 for row in candidate_rows)
    records = _read_jsonl(output / "resource_records.jsonl")
    assert len(records) == 8
    for record, resource in zip(records, protocol["resources"]):
        raw_path = output / record["raw"]["path"]
        assert raw_path.read_bytes() == _valid_body(resource)
        assert record["raw"]["sha256"] == _sha256(raw_path.read_bytes())
        assert record["content_length_agrees"] is True
        assert record["raw_byte_semantics"] == (
            "HTTP response content octets after transfer framing, before content decoding"
        )
        headers = _read_json(output / record["headers"]["path"])["raw_header_pairs_in_order"]
        names = [header["name"] for header in headers]
        assert "x-request-id" in names and "etag" in names
        assert "set-cookie" not in names and "x-untrusted" not in names

    manifest_payload = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_payload)
    source_bindings = _read_json(output / "source_bindings.json")
    assert manifest["counts"] == {"candidates": 14, "resources": 8, "request_attempts": 8}
    assert source_bindings["runtime_dependencies"] == {"httpcore": "1.0.9", "httpx": "0.28.1"}
    assert (output / "manifest.sha256").read_text(encoding="ascii") == (f"{_sha256(manifest_payload)}  manifest.json\n")
    for artifact in manifest["artifacts"].values():
        payload = (output / artifact["path"]).read_bytes()
        assert artifact["byte_size"] == len(payload)
        assert artifact["sha256"] == _sha256(payload)

    json_documents = [
        plan,
        manifest,
        source_bindings,
        _read_json(output / "summary.json"),
        *candidate_rows,
        *records,
    ]
    json_documents.extend(_read_json(path) for path in (output / "headers").glob("*.json"))
    json_documents.extend(_read_json(path) for path in (output / "requests").glob("*.json"))
    for document in json_documents:
        _assert_safety(document)


def test_network_authorization_false_has_zero_transport_and_zero_artifacts(tmp_path: Path) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)

    assert "transport_factory" not in inspect.signature(run_rule_evidence_capture).parameters
    with pytest.raises(RuleEvidenceCaptureError, match="not explicitly authorized"):
        _run(root, protocol, protocol_path, factory, network_authorized=False)

    assert factory.requests == [] and factory.factory_calls == []
    assert not (root / "data/pmxt/rule_evidence").exists()


@pytest.mark.parametrize("target", ["final", "staging"])
def test_preexisting_output_blocks_before_transport(tmp_path: Path, target: str) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)
    final = root / protocol["output"]["path"]
    existing = final if target == "final" else final.with_name(final.name + ".inprogress")
    existing.mkdir(parents=True)
    marker = existing / "owner.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(RuleEvidenceCaptureError, match="already exists"):
        _run(root, protocol, protocol_path, factory)

    assert factory.requests == [] and marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("tamper", ["source", "implementation_binding", "implementation_file"])
def test_source_and_implementation_tamper_fail_before_transport(tmp_path: Path, tamper: str) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)
    if tamper == "source":
        source_path = root / protocol["source_monitor_run"]["artifacts"]["candidates"]["path"]
        source_path.write_bytes(source_path.read_bytes() + b"\n")
    elif tamper == "implementation_binding":
        protocol["implementation"]["sha256"] = "0" * 64
    else:
        implementation_path = root / protocol["implementation"]["path"]
        implementation_path.write_bytes(implementation_path.read_bytes() + b"# tampered\n")
        protocol["implementation"] = {
            "identifier": "pmxt-rule-evidence-capture-v1",
            **_binding(root, implementation_path),
        }

    with pytest.raises(RuleEvidenceCaptureError):
        _run(root, protocol, protocol_path, factory)

    assert factory.requests == []
    assert not (root / "data/pmxt/rule_evidence").exists()


@pytest.mark.parametrize(
    ("url", "match"),
    [
        ("http://assets.kalshi.com/contract_terms/TITLE.pdf", "frozen source"),
        ("https://example.com/contract_terms/TITLE.pdf", "frozen source"),
        ("https://assets.kalshi.com/contract_terms/OTHER.pdf", "frozen source"),
        ("https://assets.kalshi.com/contract_terms/TITLE.pdf?x=1", "frozen source"),
        ("https://assets.kalshi.com:443/contract_terms/TITLE.pdf", "frozen source"),
    ],
)
def test_scheme_host_path_query_and_explicit_port_are_frozen(
    tmp_path: Path,
    url: str,
    match: str,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    protocol["resources"][0]["url"] = url
    factory = _RecordingTransportFactory(protocol)

    with pytest.raises(RuleEvidenceCaptureError, match=match):
        _run(root, protocol, protocol_path, factory)

    assert factory.requests == []


@pytest.mark.parametrize("header", ["Authorization", "Cookie", "Proxy-Authorization"])
def test_forbidden_request_headers_fail_before_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    header: str,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)
    original_build = httpx.Client.build_request

    def build_with_secret(client: httpx.Client, *args: Any, **kwargs: Any) -> httpx.Request:
        request = original_build(client, *args, **kwargs)
        request.headers[header] = "secret-must-not-leave-process"
        return request

    monkeypatch.setattr(rule_evidence_module.httpx.Client, "build_request", build_with_secret)

    with pytest.raises(RuleEvidenceCaptureError, match="forbidden credential"):
        _run(root, protocol, protocol_path, factory)

    assert factory.requests == []
    staging = root / (protocol["output"]["path"] + ".inprogress")
    assert _read_json(staging / "failure.json")["retry_permitted"] is False


def test_response_cookie_never_reaches_next_one_shot_client(tmp_path: Path) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)

    _run(root, protocol, protocol_path, factory)

    assert len(factory.requests) == 8
    assert all("cookie" not in request.headers for request in factory.requests)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("redirect", "HTTP 200"),
        ("final_url", "final response URL"),
        ("content_type", "content type"),
        ("content_encoding", "content encoding"),
        ("pdf_magic", "PDF magic"),
        ("length_mismatch", "Content-Length does not equal"),
    ],
)
def test_redirect_url_media_encoding_pdf_and_length_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    match: str,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)

    if case == "final_url":
        original_send = rule_evidence_module.httpx.Client.send

        def send_with_wrong_final_url(
            client: httpx.Client,
            request: httpx.Request,
            *args: Any,
            **kwargs: Any,
        ) -> httpx.Response:
            response = original_send(client, request, *args, **kwargs)
            response._request = httpx.Request("GET", "https://assets.kalshi.com/contract_terms/GOV.pdf")
            return response

        monkeypatch.setattr(rule_evidence_module.httpx.Client, "send", send_with_wrong_final_url)

    def responder(request: httpx.Request, resource: dict[str, Any], _: int) -> httpx.Response:
        body = _valid_body(resource)
        if case == "redirect":
            return httpx.Response(
                302,
                headers={"Content-Type": "application/pdf", "Location": "https://example.com"},
                stream=httpx.ByteStream(body),
                request=request,
            )
        if case == "final_url":
            return _valid_response(request, resource)
        if case == "content_type":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/plain"},
                stream=httpx.ByteStream(body),
                request=request,
            )
        if case == "content_encoding":
            return httpx.Response(
                200,
                headers={"Content-Type": "application/pdf", "Content-Encoding": "gzip"},
                stream=httpx.ByteStream(body),
                request=request,
            )
        if case == "pdf_magic":
            return httpx.Response(
                200,
                headers={"Content-Type": "application/pdf"},
                stream=httpx.ByteStream(b"not a PDF"),
                request=request,
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "application/pdf", "Content-Length": str(len(body) + 1)},
            stream=httpx.ByteStream(body),
            request=request,
        )

    factory = _RecordingTransportFactory(protocol, responder)

    with pytest.raises(RuleEvidenceCaptureError, match=match):
        _run(root, protocol, protocol_path, factory)

    assert len(factory.requests) == 1
    staging = root / (protocol["output"]["path"] + ".inprogress")
    receipt = _read_json(next((staging / "requests").glob("*.receipt.json")))
    failure = _read_json(staging / "failure.json")
    assert receipt["status"] == "FAILED_CLOSED"
    assert receipt["headers"] is not None
    assert receipt["raw"] is not None
    assert receipt["body_capture_status"] == "FULL"
    assert receipt["body_not_captured_reason"] is None
    assert failure["failed_receipt"] == {
        "path": next((staging / "requests").glob("*.receipt.json")).relative_to(staging).as_posix(),
        "byte_size": next((staging / "requests").glob("*.receipt.json")).stat().st_size,
        "sha256": _sha256(next((staging / "requests").glob("*.receipt.json")).read_bytes()),
    }
    retained_names = {
        item["name"] for item in _read_json(staging / receipt["headers"]["path"])["raw_header_pairs_in_order"]
    }
    assert "set-cookie" not in retained_names
    raw_payload = (staging / receipt["raw"]["path"]).read_bytes()
    assert receipt["raw"] == {
        "path": receipt["raw"]["path"],
        "byte_size": len(raw_payload),
        "sha256": _sha256(raw_payload),
    }


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b'{"id":"179312","id":"179312","markets":[]}', "strict UTF-8 JSON"),
        (b'{"id":"179312","markets":[],"x":NaN}', "strict UTF-8 JSON"),
        (b'{"id":"wrong","markets":[]}', "event ID mismatch"),
        (b'{"id":"179312","markets":[]}', "missing bound markets"),
        (b'{"id":"179312","markets":[{"id":true}]}', "invalid market ID"),
    ],
)
def test_strict_gamma_json_event_id_and_market_membership(
    tmp_path: Path,
    payload: bytes,
    match: str,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)

    def responder(request: httpx.Request, resource: dict[str, Any], attempt: int) -> httpx.Response:
        if attempt != 4:
            return _valid_response(request, resource)
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            stream=httpx.ByteStream(payload),
            request=request,
        )

    factory = _RecordingTransportFactory(protocol, responder)

    with pytest.raises(RuleEvidenceCaptureError, match=match):
        _run(root, protocol, protocol_path, factory)

    assert len(factory.requests) == 4


def test_declared_and_streamed_resource_size_limits_fail_closed(tmp_path: Path) -> None:
    for declared in (True, False):
        case_root = tmp_path / ("declared" if declared else "streamed")
        root, protocol, protocol_path = _make_repository(case_root)

        def responder(
            request: httpx.Request,
            resource: dict[str, Any],
            _: int,
            declared_case: bool = declared,
        ) -> httpx.Response:
            if declared_case:
                return httpx.Response(
                    200,
                    headers={
                        "Content-Type": "application/pdf",
                        "Content-Length": str(rule_evidence_module.MAX_RESOURCE_BYTES + 1),
                    },
                    request=request,
                )
            body = b"%PDF-" + b"x" * rule_evidence_module.MAX_RESOURCE_BYTES
            return httpx.Response(
                200,
                headers={"Content-Type": "application/pdf"},
                stream=httpx.ByteStream(body),
                request=request,
            )

        factory = _RecordingTransportFactory(protocol, responder)
        with pytest.raises(RuleEvidenceCaptureError, match="resource.*bound|declared response size"):
            _run(root, protocol, protocol_path, factory)
        assert len(factory.requests) == 1
        staging = root / (protocol["output"]["path"] + ".inprogress")
        receipt = _read_json(next((staging / "requests").glob("*.receipt.json")))
        assert receipt["headers"] is not None
        if declared:
            assert receipt["raw"] is None
            assert receipt["body_capture_status"] == "NOT_CAPTURED"
            assert receipt["body_not_captured_reason"] == "declared_content_length_exceeds_frozen_bound"
        else:
            assert receipt["raw"] is not None
            assert receipt["body_capture_status"] == "PARTIAL"
            assert (staging / receipt["raw"]["path"]).stat().st_size <= rule_evidence_module.MAX_RESOURCE_BYTES


def test_cumulative_limit_stops_second_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    monkeypatch.setattr(rule_evidence_module, "MAX_RESOURCE_BYTES", 10)
    monkeypatch.setattr(rule_evidence_module, "MAX_TOTAL_BYTES", 10)
    protocol["limits"]["max_resource_bytes"] = 10
    protocol["limits"]["max_total_bytes"] = 10

    def responder(request: httpx.Request, resource: dict[str, Any], _: int) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/pdf"},
            stream=httpx.ByteStream(b"%PDF-x"),
            request=request,
        )

    factory = _RecordingTransportFactory(protocol, responder)
    with pytest.raises(RuleEvidenceCaptureError, match="cumulative|declared response size"):
        _run(root, protocol, protocol_path, factory)
    assert len(factory.requests) == 2


def test_stop_first_failure_leaves_intent_receipt_and_failure_residue(tmp_path: Path) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)

    def responder(request: httpx.Request, resource: dict[str, Any], attempt: int) -> httpx.Response:
        if attempt == 2:
            return httpx.Response(503, request=request)
        return _valid_response(request, resource)

    factory = _RecordingTransportFactory(protocol, responder)
    with pytest.raises(RuleEvidenceCaptureError, match="503"):
        _run(root, protocol, protocol_path, factory)

    assert len(factory.requests) == 2
    staging = root / (protocol["output"]["path"] + ".inprogress")
    assert staging.is_dir()
    assert not (root / protocol["output"]["path"]).exists()
    intents = sorted((staging / "requests").glob("*.intent.json"))
    receipts = sorted((staging / "requests").glob("*.receipt.json"))
    assert len(intents) == len(receipts) == 2
    assert [_read_json(path)["status"] for path in receipts] == ["VALIDATED", "FAILED_CLOSED"]
    failure = _read_json(staging / "failure.json")
    assert failure["failed_resource_ordinal"] == 2
    assert failure["request_attempts_at_most"] == 2
    assert failure["retry_permitted"] is False
    _assert_safety(failure)


def test_concurrent_staging_claim_loses_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)
    staging = root / (protocol["output"]["path"] + ".inprogress")
    original_mkdir = rule_evidence_module.os.mkdir

    def racing_mkdir(path: os.PathLike[str] | str, mode: int = 0o777) -> None:
        candidate = Path(path)
        if candidate == staging:
            original_mkdir(path, mode)
            (candidate / "other-owner").write_text("preserve", encoding="utf-8")
            raise FileExistsError(str(path))
        original_mkdir(path, mode)

    monkeypatch.setattr(rule_evidence_module.os, "mkdir", racing_mkdir)
    with pytest.raises(RuleEvidenceCaptureError, match="in-progress.*already exists"):
        _run(root, protocol, protocol_path, factory)

    assert factory.requests == []
    assert (staging / "other-owner").read_text(encoding="utf-8") == "preserve"


def test_write_failure_leaves_permanent_failure_residue_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)
    original_write = rule_evidence_module._write_durable
    failed = False

    def failing_write(path: Path, payload: bytes) -> dict[str, Any]:
        nonlocal failed
        if path.name == "capture_plan.json" and not failed:
            failed = True
            raise OSError("injected write failure")
        return original_write(path, payload)

    monkeypatch.setattr(rule_evidence_module, "_write_durable", failing_write)
    with pytest.raises(RuleEvidenceCaptureError, match="injected write failure"):
        _run(root, protocol, protocol_path, factory)

    staging = root / (protocol["output"]["path"] + ".inprogress")
    assert factory.requests == []
    assert _read_json(staging / "failure.json")["status"] == "FAILED_CLOSED_STAGING_RETAINED"


def test_fsync_failure_after_intent_leaves_ambiguous_attempt_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    original_fsync = rule_evidence_module.os.fsync
    armed = False
    failed = False

    def failing_fsync(descriptor: int) -> None:
        nonlocal failed
        if armed and not failed:
            failed = True
            raise OSError("injected fsync failure")
        original_fsync(descriptor)

    def responder(request: httpx.Request, resource: dict[str, Any], _: int) -> httpx.Response:
        nonlocal armed
        armed = True
        return _valid_response(request, resource)

    monkeypatch.setattr(rule_evidence_module.os, "fsync", failing_fsync)
    factory = _RecordingTransportFactory(protocol, responder)
    with pytest.raises(RuleEvidenceCaptureError, match="injected fsync failure"):
        _run(root, protocol, protocol_path, factory)

    staging = root / (protocol["output"]["path"] + ".inprogress")
    assert len(factory.requests) == 1
    assert len(list((staging / "requests").glob("*.intent.json"))) == 1
    assert (staging / "failure.json").exists()


def test_rename_failure_never_exposes_final_and_retains_complete_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)

    def fail_rename(source: Path, destination: Path) -> None:
        raise OSError(f"injected rename failure: {source} -> {destination}")

    monkeypatch.setattr(rule_evidence_module, "_rename_directory_no_replace", fail_rename)
    with pytest.raises(RuleEvidenceCaptureError, match="injected rename failure"):
        _run(root, protocol, protocol_path, factory)

    final = root / protocol["output"]["path"]
    staging = final.with_name(final.name + ".inprogress")
    assert len(factory.requests) == 8
    assert not final.exists()
    assert (staging / "manifest.json").exists()
    assert (staging / "manifest.sha256").exists()
    assert (staging / "failure.json").exists()


def test_invalid_http_transport_constructor_is_one_attempt_without_retry(tmp_path: Path) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    calls: list[dict[str, Any]] = []

    def bad_factory(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return object()

    with pytest.raises(RuleEvidenceCaptureError, match="BaseTransport"):
        _run(root, protocol, protocol_path, bad_factory)

    assert calls == [{"retries": 0, "trust_env": False}]
    staging = root / (protocol["output"]["path"] + ".inprogress")
    assert len(list((staging / "requests").glob("*.intent.json"))) == 1
    assert _read_json(staging / "failure.json")["request_attempts_at_most"] == 1


@pytest.mark.parametrize(("dependency", "value"), [("httpx", "0.28.2"), ("httpcore", "1.0.10")])
def test_runtime_version_mismatch_fails_before_transport_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
    value: str,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)
    monkeypatch.setattr(getattr(rule_evidence_module, dependency), "__version__", value)

    with pytest.raises(RuleEvidenceCaptureError, match="runtime dependency version mismatch"):
        _run(root, protocol, protocol_path, factory)

    assert factory.requests == [] and factory.factory_calls == []
    assert not (root / "data/pmxt/rule_evidence").exists()


def test_bound_monitor_manifest_is_opened_once_and_records_are_not_reopened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)
    monitor_manifest = (root / protocol["source_monitor_run"]["manifest"]["path"]).resolve()
    original_read_input = rule_evidence_module._read_input
    original_read_bytes = Path.read_bytes
    manifest_opens = 0

    def read_input_spy(path: Path, *, root: Path, label: str) -> Any:
        nonlocal manifest_opens
        if Path(path).resolve() == monitor_manifest:
            manifest_opens += 1
        return original_read_input(path, root=root, label=label)

    def forbid_records_reopen(path: Path) -> bytes:
        if path.name == "resource_records.jsonl":
            raise AssertionError("resource records must be hashed through their original write handle")
        return original_read_bytes(path)

    monkeypatch.setattr(rule_evidence_module, "_read_input", read_input_spy)
    monkeypatch.setattr(Path, "read_bytes", forbid_records_reopen)

    output = _run(root, protocol, protocol_path, factory)

    assert output.is_dir()
    assert manifest_opens == 1


@pytest.mark.parametrize(("directory_name", "expected_requests"), [("requests", 0), ("raw", 1)])
def test_output_subdirectory_replacement_fails_closed_without_next_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory_name: str,
    expected_requests: int,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)
    original_open = rule_evidence_module._open_exclusive
    replaced = False

    def replace_parent_then_open(path: Path) -> BinaryIO:
        nonlocal replaced
        if not replaced and path.parent.name == directory_name:
            replaced = True
            moved = path.parent.with_name(f"{directory_name}.replaced")
            path.parent.rename(moved)
            path.parent.mkdir()
        return original_open(path)

    monkeypatch.setattr(rule_evidence_module, "_open_exclusive", replace_parent_then_open)
    with pytest.raises(RuleEvidenceCaptureError, match="artifact parent|identity changed"):
        _run(root, protocol, protocol_path, factory)

    final = root / protocol["output"]["path"]
    staging = final.with_name(final.name + ".inprogress")
    assert len(factory.requests) == expected_requests
    assert not final.exists()
    assert staging.is_dir() and (staging / "failure.json").exists()


def test_final_destination_race_never_overwrites_competing_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)
    original_rename = rule_evidence_module._rename_directory_no_replace
    marker_payload = b"competing owner"

    def race_final_destination(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "owner.bin").write_bytes(marker_payload)
        original_rename(source, destination)

    monkeypatch.setattr(rule_evidence_module, "_rename_directory_no_replace", race_final_destination)
    with pytest.raises(RuleEvidenceCaptureError, match="already exists"):
        _run(root, protocol, protocol_path, factory)

    final = root / protocol["output"]["path"]
    staging = final.with_name(final.name + ".inprogress")
    assert len(factory.requests) == 8
    assert (final / "owner.bin").read_bytes() == marker_payload
    assert staging.is_dir() and (staging / "failure.json").exists()


@pytest.mark.parametrize(("artifact_kind", "expected_requests"), [("source_bindings", 0), ("raw", 1)])
def test_same_size_artifact_mutation_is_detected_before_next_request_or_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
    expected_requests: int,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    factory = _RecordingTransportFactory(protocol)

    def mutate_same_size(path: Path) -> None:
        with path.open("r+b") as handle:
            first = handle.read(1)
            handle.seek(0)
            handle.write(b"!" if first != b"!" else b"?")
            handle.flush()
            os.fsync(handle.fileno())

    if artifact_kind == "source_bindings":
        original_write = rule_evidence_module._write_durable
        mutated = False

        def mutate_after_write(path: Path, payload: bytes) -> dict[str, Any]:
            nonlocal mutated
            metadata = original_write(path, payload)
            if not mutated and path.name == "source_bindings.json":
                mutated = True
                mutate_same_size(path)
            return metadata

        monkeypatch.setattr(rule_evidence_module, "_write_durable", mutate_after_write)
    else:
        original_capture = rule_evidence_module._capture_resource
        mutated = False

        def mutate_after_capture(**kwargs: Any) -> Any:
            nonlocal mutated
            record, artifacts = original_capture(**kwargs)
            if not mutated:
                mutated = True
                mutate_same_size(Path(kwargs["staging"]) / artifacts["raw"]["path"])
            return record, artifacts

        monkeypatch.setattr(rule_evidence_module, "_capture_resource", mutate_after_capture)

    with pytest.raises(RuleEvidenceCaptureError, match="no longer matches its binding"):
        _run(root, protocol, protocol_path, factory)

    final = root / protocol["output"]["path"]
    staging = final.with_name(final.name + ".inprogress")
    assert len(factory.requests) == expected_requests
    assert not final.exists()
    assert staging.is_dir() and (staging / "failure.json").exists()


def test_monotonic_resource_checkpoint_preserves_headers_and_closes_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, protocol, protocol_path = _make_repository(tmp_path)
    clock = {"now": 0}

    class TrackingStream(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self) -> Any:
            yield b"%PDF-delayed"

        def close(self) -> None:
            self.closed = True

    stream = TrackingStream()

    def monotonic() -> int:
        return clock["now"]

    def responder(request: httpx.Request, resource: dict[str, Any], _: int) -> httpx.Response:
        clock["now"] = 31_000_000_000
        return httpx.Response(
            200,
            headers={"Content-Type": resource["expected_media_type"]},
            stream=stream,
            request=request,
        )

    monkeypatch.setattr(rule_evidence_module, "_monotonic_ns", monotonic)
    factory = _RecordingTransportFactory(protocol, responder)
    with pytest.raises(RuleEvidenceCaptureError, match="monotonic elapsed checkpoint threshold exceeded"):
        _run(root, protocol, protocol_path, factory)

    staging = root / (protocol["output"]["path"] + ".inprogress")
    receipt = _read_json(next((staging / "requests").glob("*.receipt.json")))
    assert len(factory.requests) == 1
    assert stream.closed is True
    assert receipt["headers"] is not None and receipt["raw"] is None
    assert receipt["body_capture_status"] == "NOT_CAPTURED"
    assert receipt["body_not_captured_reason"] == "monotonic_elapsed_checkpoint_exceeded_before_body"
    assert not (root / protocol["output"]["path"]).exists()


def test_capture_has_no_unbound_local_helper_or_public_transport_seam() -> None:
    source = Path(rule_evidence_module.__file__).read_text(encoding="utf-8")
    signature = inspect.signature(run_rule_evidence_capture)

    assert "offline_adjudication" not in source
    assert "transport_factory" not in signature.parameters
