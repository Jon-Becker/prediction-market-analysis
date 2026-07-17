"""Tests for the Polymarket events indexer: pagination, resume, interrupt, schema (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.indexers.polymarket.models import Event

EVENT_FIELDS = [
    "id",
    "slug",
    "title",
    "category",
    "tags",
    "market_ids",
    "volume",
    "liquidity",
    "active",
    "closed",
    "start_date",
    "end_date",
    "created_at",
]


def make_event(i: int, closed: bool = False) -> Event:
    return Event.from_dict(
        {
            "id": str(i),
            "slug": f"event-{i}",
            "title": f"Event {i}",
            "category": "Sports",
            "tags": [{"slug": "sports"}],
            "markets": [{"id": str(i * 10)}],
            "volume": 10.0,
            "liquidity": 5.0,
            "active": not closed,
            "closed": closed,
            "startDate": "2026-06-01T00:00:00Z",
            "endDate": "2026-06-30T00:00:00Z",
            "createdAt": "2026-05-01T00:00:00Z",
        }
    )


class FakeClient:
    """Stub for PolymarketClient serving canned keyset pages.

    `responses` maps (closed, after_cursor) -> (events, next_cursor);
    `interrupt_at` raises `exc` (KeyboardInterrupt by default) before serving that key.
    """

    def __init__(
        self,
        responses: dict,
        interrupt_at: tuple | None = None,
        exc: type[BaseException] = KeyboardInterrupt,
    ):
        self.responses = responses
        self.calls: list[dict] = []
        self._interrupt_at = interrupt_at
        self._exc = exc
        self.was_closed = False

    def iter_events_keyset(self, limit: int = 500, after_cursor: str | None = None, **kwargs):
        closed = kwargs.get("closed")
        cursor = after_cursor
        while True:
            if self._interrupt_at is not None and (closed, cursor) == self._interrupt_at:
                raise self._exc
            self.calls.append({"limit": limit, "after_cursor": cursor, "closed": closed})
            events, cursor = self.responses[(closed, cursor)]
            yield events, cursor
            if not cursor:
                break

    def close(self):
        self.was_closed = True


@pytest.fixture()
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point DATA_DIR and CURSOR_FILE at temp directories."""
    data_dir = tmp_path / "events"
    cursor_file = tmp_path / ".events_backfill_cursor"
    import src.indexers.polymarket.events as mod

    monkeypatch.setattr(mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(mod, "CURSOR_FILE", cursor_file)
    return data_dir, cursor_file


def run_indexer(client: FakeClient) -> None:
    from src.indexers.polymarket.events import PolymarketEventsIndexer

    with patch("src.indexers.polymarket.events.PolymarketClient", return_value=client):
        PolymarketEventsIndexer().run()


def read_all(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("events_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def test_full_backfill_paginates_both_phases(isolated_dirs, monkeypatch: pytest.MonkeyPatch):
    """Both closed=False and closed=True passes run, following keyset cursors."""
    import src.indexers.polymarket.events as mod

    monkeypatch.setattr(mod, "CHUNK_SIZE", 4)
    data_dir, cursor_file = isolated_dirs

    client = FakeClient(
        {
            (False, None): ([make_event(1), make_event(2)], "o2"),
            (False, "o2"): ([make_event(3)], None),
            (True, None): ([make_event(4, closed=True), make_event(5, closed=True)], None),
        }
    )
    run_indexer(client)

    assert client.calls == [
        {"limit": 500, "after_cursor": None, "closed": False},
        {"limit": 500, "after_cursor": "o2", "closed": False},
        {"limit": 500, "after_cursor": None, "closed": True},
    ]

    files = sorted(data_dir.glob("events_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    assert [len(pd.read_parquet(f)) for f in files] == [4, 1], "Full chunk saved mid-run, remainder flushed at end"
    assert sorted(read_all(data_dir)["id"]) == ["1", "2", "3", "4", "5"]

    assert not cursor_file.exists(), "Cursor file should be deleted after successful completion"
    assert client.was_closed


def test_parquet_schema_matches_event_model(isolated_dirs):
    """Chunk columns are exactly the Event fields plus _fetched_at."""
    data_dir, _ = isolated_dirs

    client = FakeClient(
        {
            (False, None): ([make_event(1)], None),
            (True, None): ([], None),
        }
    )
    run_indexer(client)

    df = read_all(data_dir)
    assert list(df.columns) == EVENT_FIELDS + ["_fetched_at"]
    assert json.loads(df.iloc[0]["tags"]) == ["sports"]
    assert json.loads(df.iloc[0]["market_ids"]) == ["10"]
    assert df.iloc[0]["_fetched_at"] is not None


def test_resume_starts_from_saved_phase_and_cursor(isolated_dirs):
    """A saved cursor skips completed phases and resumes mid-phase."""
    _, cursor_file = isolated_dirs
    cursor_file.write_text(json.dumps({"phase": "closed", "cursor": "c2"}))

    client = FakeClient({(True, "c2"): ([make_event(6, closed=True)], None)})
    run_indexer(client)

    assert client.calls == [{"limit": 500, "after_cursor": "c2", "closed": True}]
    assert not cursor_file.exists(), "Cursor file should be deleted after successful completion"


def test_interrupt_then_resume_completes(isolated_dirs):
    """Interrupt a backfill with Ctrl+C, then resume and finish the job."""
    data_dir, cursor_file = isolated_dirs

    # Run 1: interrupt when the open phase requests the page at cursor "o2"
    client1 = FakeClient(
        {(False, None): ([make_event(1), make_event(2)], "o2")},
        interrupt_at=(False, "o2"),
    )
    run_indexer(client1)

    assert cursor_file.exists(), "Cursor file should survive Ctrl+C"
    assert json.loads(cursor_file.read_text()) == {"phase": "open", "cursor": "o2"}
    assert sorted(read_all(data_dir)["id"]) == ["1", "2"], "Buffered events flushed on interrupt"

    # Run 2: resume from the saved cursor, no interrupt
    client2 = FakeClient(
        {
            (False, "o2"): ([make_event(3)], None),
            (True, None): ([make_event(4, closed=True)], None),
        }
    )
    run_indexer(client2)

    assert client2.calls[0] == {"limit": 500, "after_cursor": "o2", "closed": False}
    assert sorted(read_all(data_dir)["id"]) == ["1", "2", "3", "4"], "Resume fetches only the remaining pages"
    assert not cursor_file.exists(), "Cursor file should be deleted after successful completion"


def test_crash_flushes_buffer_and_preserves_cursor(isolated_dirs):
    """A non-interrupt crash still flushes buffered events, so the saved cursor never points past unsaved data."""
    data_dir, cursor_file = isolated_dirs

    client = FakeClient(
        {(False, None): ([make_event(1), make_event(2)], "o2")},
        interrupt_at=(False, "o2"),
        exc=RuntimeError,
    )
    with pytest.raises(RuntimeError):
        run_indexer(client)

    assert json.loads(cursor_file.read_text()) == {"phase": "open", "cursor": "o2"}
    assert sorted(read_all(data_dir)["id"]) == ["1", "2"], "Buffered events flushed on crash"


def test_interrupt_between_phases_resumes_at_closed_phase(isolated_dirs):
    """After the open phase completes, resume skips straight to the closed phase."""
    data_dir, cursor_file = isolated_dirs

    client1 = FakeClient(
        {(False, None): ([make_event(1)], None)},
        interrupt_at=(True, None),
    )
    run_indexer(client1)

    assert json.loads(cursor_file.read_text()) == {"phase": "closed", "cursor": None}

    client2 = FakeClient({(True, None): ([make_event(2, closed=True)], None)})
    run_indexer(client2)

    assert all(call["closed"] is True for call in client2.calls), "Open phase should not be re-crawled"
    assert sorted(read_all(data_dir)["id"]) == ["1", "2"]
    assert not cursor_file.exists()
