"""Helpers for analyses that group Kalshi markets by their API-supplied event category.

Categories live in the shared events dataset (data/kalshi/events), which covers
both recent and historical markets. Join it to markets on event_ticker.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.common.util.prompts import prompt_choice

BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
DEFAULT_EVENTS_DIR = BASE_DIR / "data" / "kalshi" / "events"


def list_event_categories(events_dir: Path) -> list[str]:
    """All non-empty event categories in the events dataset, sorted."""
    rows = duckdb.sql(f"""
        SELECT DISTINCT category
        FROM '{events_dir}/events_*.parquet'
        WHERE category IS NOT NULL AND category <> ''
        ORDER BY category
    """).fetchall()
    return [row[0] for row in rows]


def prompt_event_category(events_dir: Path) -> str:
    """Ask the user to pick one of the event categories in the events dataset."""
    return prompt_choice(
        "Enter event category to analyze (e.g., 'Politics', 'Sports'): ",
        list_event_categories(events_dir),
    )
