from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.indexers.pmxt import rejection_taxonomy as taxonomy

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {item.relative_to(path).as_posix(): item.read_bytes() for item in sorted(path.rglob("*")) if item.is_file()}


def _copy_authoritative_sources(destination: Path) -> None:
    for relative, _digest, _sidecar in taxonomy._AUTHORITATIVE_MANIFESTS.values():
        source_dir = (REPOSITORY_ROOT / relative).parent
        target_dir = (destination / relative).parent
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, target_dir)


def test_exact_repository_derivation_is_hash_bound_and_non_executable(tmp_path: Path) -> None:
    output_root = tmp_path / "rejection_taxonomy"
    run_dir = taxonomy.derive_rejection_taxonomy(
        repository_root=REPOSITORY_ROOT,
        output_root=output_root,
        run_id="fixture_taxonomy_v1",
    )

    assert run_dir == output_root / "runs" / "fixture_taxonomy_v1"
    assert {path.name for path in run_dir.iterdir()} == {
        "taxonomy_rows.jsonl",
        "taxonomy_summary.json",
        "source_bindings.json",
        "unresolved_caveats.jsonl",
        "manifest.json",
        "manifest.sha256",
    }
    rows = _jsonl(run_dir / "taxonomy_rows.jsonl")
    assert len(rows) == 25
    assert len({row["candidate_id"] for row in rows}) == 25
    assert all(row["terminal_decision"] == "REJECTED" for row in rows)
    assert all(row["live_eligible"] is False for row in rows)
    assert all(row["books_requested"] is False for row in rows)
    assert all(row["economics_computed"] is False for row in rows)
    assert all(row["orders_submitted"] == 0 for row in rows)
    assert all(row["profitability_established"] is False for row in rows)
    assert all(row["profitability_evaluation_eligible"] is False for row in rows)
    assert all(row["pmxt_discovery"]["catalog_semantic_fields_authoritative"] is False for row in rows)
    assert all(row["pmxt_discovery"]["catalog_semantic_fields_can_hard_reject"] is False for row in rows)
    assert all(row["semantic_evidence_boundary"]["stage"] == "POST_NATIVE_PRE_BOOK" for row in rows)

    summary = _json(run_dir / "taxonomy_summary.json")
    assert summary["counts"] == {"total": 25, "rejected": 25, "verified_equivalent": 0, "needs_review": 0}
    assert summary["counts_by_taxonomy_id"] == {
        "BRAZIL_SETTLEMENT_SOURCE_FALLBACK": 1,
        "GOP_NOMINEE_STRICT_CLOSE_WITH_UNRESOLVED_AXES": 3,
        "GOVERNOR_TRIGGER_DEADLINE_AND_FALLBACK": 2,
        "HOUSE_CONTROL_DETERMINATION_PREDICATE": 2,
        "ISRAEL_PM_ALTERNATE_ELECTION_AND_CUTOFF": 2,
        "MLB_AWARD_MULTIPLE_WINNER_TIEBREAK": 2,
        "PRESIDENTIAL_CALL_VS_INAUGURATION": 4,
        "SPORTS_TERMINAL_NO_WINNER_PAYOUT_MISMATCH": 9,
    }
    assert summary["stage_transition_counts"] == {
        "NEEDS_REVIEW -> NEEDS_REVIEW -> REJECTED": 14,
        "NEEDS_REVIEW -> REJECTED -> REJECTED": 11,
    }
    assert summary["overlapping_reason_code_incidence"] == taxonomy._EXPECTED_REASON_INCIDENCE
    assert summary["pair_family_counts"] == taxonomy._EXPECTED_PAIR_FAMILIES
    assert summary["terminal_result"] == "NO_VERIFIED_CANDIDATES"

    source_bindings = _json(run_dir / "source_bindings.json")
    assert source_bindings["source_manifest_count"] == 4
    assert source_bindings["all_listed_source_artifacts_hash_validated"] is True
    assert set(source_bindings["authoritative_runs"]) == {
        "monitor",
        "adjudication",
        "rule_evidence",
        "rule_review",
    }
    assert source_bindings["legacy_capture"]["read_or_ingested"] is False
    for source in source_bindings["authoritative_runs"].values():
        assert source["all_manifest_listed_artifacts_hash_validated"] is True
        manifest = _json(REPOSITORY_ROOT / source["manifest"]["path"])
        assert set(source["manifest_artifacts"]) == set(manifest["artifacts"])

    manifest_bytes = (run_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    for binding in manifest["artifacts"].values():
        payload = (run_dir / binding["path"]).read_bytes()
        assert len(payload) == binding["byte_size"]
        assert hashlib.sha256(payload).hexdigest() == binding["sha256"]
    assert (run_dir / "manifest.sha256").read_text(encoding="ascii") == (
        f"{hashlib.sha256(manifest_bytes).hexdigest()}  manifest.json\n"
    )
    output_payload = b"".join(path.read_bytes() for path in run_dir.iterdir())
    assert b'"raw_response"' not in output_payload
    assert b'"normalized_rules"' not in output_payload


@pytest.mark.parametrize("existing_suffix", ["", ".inprogress"])
def test_preexisting_final_or_staging_preserves_the_whole_tree(tmp_path: Path, existing_suffix: str) -> None:
    output_root = tmp_path / "rejection_taxonomy"
    existing = output_root / "runs" / f"immutable_run{existing_suffix}"
    existing.mkdir(parents=True)
    (existing / "sentinel.bin").write_bytes(b"preserve-me")
    (existing / "nested").mkdir()
    (existing / "nested" / "also.bin").write_bytes(b"unchanged")
    before = _tree_bytes(output_root)

    with pytest.raises(taxonomy.RejectionTaxonomyError, match="already exists"):
        taxonomy.derive_rejection_taxonomy(
            repository_root=REPOSITORY_ROOT,
            output_root=output_root,
            run_id="immutable_run",
        )

    assert _tree_bytes(output_root) == before


def test_source_tamper_fails_before_any_output(tmp_path: Path) -> None:
    repository_copy = tmp_path / "repository"
    repository_copy.mkdir()
    _copy_authoritative_sources(repository_copy)
    candidate_path = repository_copy / taxonomy._AUTHORITATIVE_MANIFESTS["monitor"][0]
    candidate_path = candidate_path.parent / "candidates.jsonl"
    candidate_path.write_bytes(candidate_path.read_bytes() + b" ")
    output_root = tmp_path / "output"

    with pytest.raises(taxonomy.RejectionTaxonomyError, match="disagrees with its manifest"):
        taxonomy.derive_rejection_taxonomy(
            repository_root=repository_copy,
            output_root=output_root,
            run_id="tampered_source",
        )

    assert not output_root.exists()


def test_unknown_taxonomy_signature_fails_before_any_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(taxonomy, "_TAXONOMY_BY_SIGNATURE", {})
    output_root = tmp_path / "output"

    with pytest.raises(taxonomy.RejectionTaxonomyError, match="unknown rejection taxonomy signature"):
        taxonomy.derive_rejection_taxonomy(
            repository_root=REPOSITORY_ROOT,
            output_root=output_root,
            run_id="unknown_taxonomy",
        )

    assert not output_root.exists()


def test_source_is_revalidated_before_atomic_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository_copy = tmp_path / "repository"
    repository_copy.mkdir()
    _copy_authoritative_sources(repository_copy)
    candidate_path = repository_copy / taxonomy._AUTHORITATIVE_MANIFESTS["monitor"][0]
    candidate_path = candidate_path.parent / "candidates.jsonl"
    original_write = taxonomy._write_exclusive
    changed = False

    def write_then_change_source(path: Path, payload: bytes) -> dict[str, object]:
        nonlocal changed
        result = dict(original_write(path, payload))
        if path.name == "taxonomy_rows.jsonl" and not changed:
            candidate_path.write_bytes(candidate_path.read_bytes() + b" ")
            changed = True
        return result

    monkeypatch.setattr(taxonomy, "_write_exclusive", write_then_change_source)
    output_root = tmp_path / "output"
    with pytest.raises(taxonomy.RejectionTaxonomyError, match="source changed after validation"):
        taxonomy.derive_rejection_taxonomy(
            repository_root=repository_copy,
            output_root=output_root,
            run_id="source_race",
        )

    assert not (output_root / "runs" / "source_race").exists()
    staging = output_root / "runs" / "source_race.inprogress"
    assert staging.is_dir()
    failure = _json(staging / "failure.json")
    assert failure["status"] == "FAILED_CLOSED_STAGING_RETAINED"
    assert failure["retry_permitted"] is False


def test_atomic_no_replace_preserves_competing_final_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "output"
    original_rename = taxonomy._rename_no_replace

    def inject_competing_final(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "sentinel.bin").write_bytes(b"winner")
        original_rename(source, destination)

    monkeypatch.setattr(taxonomy, "_rename_no_replace", inject_competing_final)
    with pytest.raises(taxonomy.RejectionTaxonomyError, match="already exists"):
        taxonomy.derive_rejection_taxonomy(
            repository_root=REPOSITORY_ROOT,
            output_root=output_root,
            run_id="competing_final",
        )

    final = output_root / "runs" / "competing_final"
    assert _tree_bytes(final) == {"sentinel.bin": b"winner"}
    staging = output_root / "runs" / "competing_final.inprogress"
    assert _json(staging / "failure.json")["retry_permitted"] is False


def test_module_has_no_environment_credential_or_network_import_surface() -> None:
    source_path = REPOSITORY_ROOT / "src/indexers/pmxt/rejection_taxonomy.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(
        {"aiohttp", "dotenv", "http", "httpx", "requests", "socket", "urllib", "websocket", "websockets"}
    )
    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "PMXT_API_KEY" not in source
