"""Indexer for Polymarket events data."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.indexer import Indexer
from src.indexers.polymarket.client import PolymarketClient

DATA_DIR = Path("data/polymarket/events")
CURSOR_FILE = Path("data/polymarket/.events_backfill_cursor")
CHUNK_SIZE = 10000
PAGE_LIMIT = 500

# Gamma excludes closed events unless closed=true is requested, so full coverage
# needs one pass per closed value. The open pass runs first so events that close
# mid-crawl are still picked up by the closed pass.
PHASES = [("open", False), ("closed", True)]


class PolymarketEventsIndexer(Indexer):
    """Fetches and stores Polymarket events data."""

    def __init__(self):
        super().__init__(
            name="polymarket_events",
            description="Backfills Polymarket events data to parquet files",
        )

    def run(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        client = PolymarketClient()

        resume_phase = None
        resume_cursor = None
        if CURSOR_FILE.exists():
            try:
                state = json.loads(CURSOR_FILE.read_text())
                resume_phase = state.get("phase")
                resume_cursor = state.get("cursor")
                print(f"Resuming from {resume_phase} phase")
            except (ValueError, TypeError):
                pass

        phase_names = [name for name, _ in PHASES]
        start_idx = phase_names.index(resume_phase) if resume_phase in phase_names else 0

        buffer = []
        total = 0
        total_saved = 0
        interrupted = False

        def get_next_chunk_idx():
            existing = list(DATA_DIR.glob("events_*.parquet"))
            indices = []
            for f in existing:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    try:
                        indices.append(int(parts[1]))
                    except ValueError:
                        pass
            return max(indices) + CHUNK_SIZE if indices else 0

        def save_chunk(records):
            nonlocal total_saved
            if not records:
                return
            chunk_idx = get_next_chunk_idx()
            chunk_path = DATA_DIR / f"events_{chunk_idx}_{chunk_idx + CHUNK_SIZE}.parquet"
            pd.DataFrame(records).to_parquet(chunk_path)
            total_saved += len(records)
            print(f"Saved {len(records)} events to {chunk_path.name}")

        try:
            for idx in range(start_idx, len(PHASES)):
                phase, closed = PHASES[idx]
                after_cursor = resume_cursor if phase == resume_phase else None
                print(f"Fetching {phase} events (closed={closed})")

                for events, next_cursor in client.iter_events_keyset(
                    limit=PAGE_LIMIT, after_cursor=after_cursor, closed=closed
                ):
                    if events:
                        fetched_at = datetime.utcnow()
                        for event in events:
                            record = asdict(event)
                            record["_fetched_at"] = fetched_at
                            buffer.append(record)

                        total += len(events)
                        print(f"Fetched {len(events)} {phase} events (total: {total})")

                        while len(buffer) >= CHUNK_SIZE:
                            save_chunk(buffer[:CHUNK_SIZE])
                            buffer = buffer[CHUNK_SIZE:]

                    if next_cursor:
                        CURSOR_FILE.write_text(json.dumps({"phase": phase, "cursor": next_cursor}))

                if idx + 1 < len(PHASES):
                    CURSOR_FILE.write_text(json.dumps({"phase": PHASES[idx + 1][0], "cursor": None}))
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted. Progress saved.")

        # Save remaining events
        save_chunk(buffer)

        # Only clean up cursor on successful completion
        if not interrupted and CURSOR_FILE.exists():
            CURSOR_FILE.unlink()

        client.close()
        print(f"\nBackfill complete: {total} events fetched")
