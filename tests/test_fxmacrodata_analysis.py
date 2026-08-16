from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from urllib.error import URLError

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.analysis.kalshi.macro_release_activity import MacroReleaseActivityAnalysis
from src.common.fxmacrodata import (
    FXMacroDataError,
    fetch_release_calendar,
    parse_release_calendar,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_parse_release_calendar_sorts_rows():
    rows = parse_release_calendar(
        {
            "data": [
                {"release": "later", "announcement_datetime": 200},
                {"release": "earlier", "announcement_datetime": 100},
            ]
        }
    )

    assert [row["release"] for row in rows] == ["earlier", "later"]


@pytest.mark.parametrize(
    "payload",
    [None, [], {}, {"data": ["not-an-object"]}, {"data": [{"release": "missing time"}]}],
)
def test_parse_release_calendar_rejects_malformed_payload(payload):
    with pytest.raises(FXMacroDataError):
        parse_release_calendar(payload)


def test_fetch_release_calendar_uses_canonical_route_and_caps_rows(monkeypatch):
    captured = {}
    payload = {
        "data": [
            {"release": "first", "announcement_datetime": 100},
            {"release": "second", "announcement_datetime": 200},
        ]
    }

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse(payload)

    monkeypatch.setattr("src.common.fxmacrodata.urlopen", fake_urlopen)
    rows = fetch_release_calendar(
        "USD",
        limit=1,
        api_key="placeholder-token",
        timeout=7,
    )

    assert rows == [payload["data"][0]]
    assert captured == {
        "url": "https://api.fxmacrodata.com/v1/calendar/usd?limit=1&api_key=placeholder-token",
        "timeout": 7,
    }


def test_fetch_release_calendar_redacts_transport_errors(monkeypatch):
    def fail(request, timeout):
        raise URLError("provider details")

    monkeypatch.setattr("src.common.fxmacrodata.urlopen", fail)

    with pytest.raises(FXMacroDataError, match="calendar request failed") as raised:
        fetch_release_calendar("USD", api_key="placeholder-token")

    assert "placeholder-token" not in str(raised.value)


def test_macro_release_activity_uses_exact_window_boundaries():
    event_time = datetime(2026, 8, 14, 12, tzinfo=timezone.utc)
    offsets = [-61, -60, -1, 0, 1, 60, 61]
    trades = pd.DataFrame(
        [
            {
                "ticker": "MKT-A",
                "created_time": event_time + timedelta(minutes=offset),
                "count": 2,
                "yes_price": 60,
                "no_price": 40,
                "taker_side": "yes",
            }
            for offset in offsets
        ]
    )
    markets = pd.DataFrame([{"ticker": "MKT-A"}])
    releases = [
        {
            "release": "inflation",
            "announcement_datetime": int(event_time.timestamp()),
            "release_date_confirmed": True,
            "release_time_assumed": False,
        }
    ]

    output = MacroReleaseActivityAnalysis(
        trades=trades,
        markets=markets,
        releases=releases,
        pre_window_minutes=60,
        post_window_minutes=60,
    ).run()

    assert output.data is not None
    row = output.data.iloc[0]
    assert row["pre_trade_count"] == 2
    assert row["post_trade_count"] == 3
    assert row["pre_contract_count"] == 4
    assert row["post_contract_count"] == 6
    assert row["pre_notional_usd"] == pytest.approx(2.4)
    assert row["post_notional_usd"] == pytest.approx(3.6)
    plt.close(output.figure)


def test_macro_release_activity_excludes_unconfirmed_and_assumed_rows():
    event_time = datetime(2026, 8, 14, 12, tzinfo=timezone.utc)
    trades = pd.DataFrame(
        [
            {
                "ticker": "MKT-A",
                "created_time": event_time,
                "count": 1,
                "yes_price": 50,
                "no_price": 50,
                "taker_side": "yes",
            }
        ]
    )
    markets = pd.DataFrame([{"ticker": "MKT-A"}])
    releases = [
        {
            "release": "unconfirmed",
            "announcement_datetime": int(event_time.timestamp()),
            "release_date_confirmed": False,
        },
        {
            "release": "assumed",
            "announcement_datetime": int(event_time.timestamp()),
            "release_date_confirmed": True,
            "release_time_assumed": True,
        },
    ]

    output = MacroReleaseActivityAnalysis(
        trades=trades,
        markets=markets,
        releases=releases,
    ).run()

    assert output.data is not None
    assert output.data.empty
    plt.close(output.figure)


def test_release_calendar_fixture_is_valid_json(release_calendar_path):
    payload = json.loads(release_calendar_path.read_text(encoding="utf-8"))
    assert len(parse_release_calendar(payload)) == 1
