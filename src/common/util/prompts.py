"""Interactive prompts shared by indexers and analyses."""

from __future__ import annotations

from datetime import datetime, timezone


def prompt_date(prompt: str, utc: bool = False) -> datetime | None:
    """Ask for a YYYY-MM-DD date until a valid one is entered. Blank input returns None."""
    while True:
        date_str = input(prompt).strip()
        if not date_str:
            return None
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD (e.g., 2026-03-01)")
            continue
        return date.replace(tzinfo=timezone.utc) if utc else date


def prompt_date_range(utc: bool = False) -> tuple[datetime | None, datetime | None]:
    """Ask for an optional start and end date, re-asking if start is after end."""
    while True:
        start = prompt_date("Start date (YYYY-MM-DD), leave blank for no start date: ", utc=utc)
        end = prompt_date("End date (YYYY-MM-DD), leave blank for no end date: ", utc=utc)
        if start and end and start > end:
            print("Start date cannot be after end date. Please try again.")
            continue
        return start, end


def prompt_choice(prompt: str, choices: list[str]) -> str:
    """Ask until the input exactly matches one of `choices`."""
    while True:
        choice = input(prompt).strip()
        if choice in choices:
            return choice
        print(f"Invalid selection. Please enter one of the following: {', '.join(choices)}")
