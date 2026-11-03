"""Test that every analysis run() produces valid output."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from src.common.analysis import Analysis, AnalysisOutput

_ALL_ANALYSES = Analysis.load()
_STATIC_ANALYSES = [c for c in _ALL_ANALYSES if c.__name__ != "WinRateByPriceAnimatedAnalysis"]
_ANIMATED_ANALYSES = [c for c in _ALL_ANALYSES if c.__name__ == "WinRateByPriceAnimatedAnalysis"]


def _build_kwargs(cls: type[Analysis], fixture_dirs: dict[str, Path]) -> dict[str, Path]:
    """Map constructor params to fixture paths based on platform module."""
    sig = inspect.signature(cls.__init__)
    params = [p for p in sig.parameters if p != "self"]

    module = cls.__module__
    is_kalshi = ".kalshi." in module
    is_polymarket = ".polymarket." in module

    kwargs: dict[str, Path] = {}
    for param in params:
        # Direct match — comparison module params use explicit platform prefixes
        if param in fixture_dirs:
            kwargs[param] = fixture_dirs[param]
        elif is_kalshi and param == "trades_dir":
            kwargs[param] = fixture_dirs["kalshi_trades_dir"]
        elif is_kalshi and param == "markets_dir":
            kwargs[param] = fixture_dirs["kalshi_markets_dir"]
        elif is_kalshi and param == "events_dir":
            kwargs[param] = fixture_dirs["kalshi_events_dir"]
        elif is_polymarket and param == "trades_dir":
            kwargs[param] = fixture_dirs["polymarket_trades_dir"]
        elif is_polymarket and param == "legacy_trades_dir":
            kwargs[param] = fixture_dirs["polymarket_legacy_trades_dir"]
        elif is_polymarket and param == "markets_dir":
            kwargs[param] = fixture_dirs["polymarket_markets_dir"]
        elif is_polymarket and param == "blocks_dir":
            kwargs[param] = fixture_dirs["polymarket_blocks_dir"]

    return kwargs


def _fake_input(prompt: str) -> str:
    """Answer the interactive prompts some Kalshi analyses ask before running."""
    prompt = prompt.lower()
    if "category" in prompt:
        return "Sports"
    if "yes or no" in prompt:
        return "yes"
    return ""  # Dates: leave blank for no filter


@pytest.fixture(autouse=True)
def _answer_prompts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("builtins.input", _fake_input)


@pytest.mark.parametrize("cls", _STATIC_ANALYSES, ids=lambda c: c.__name__)
def test_analysis_run(cls: type[Analysis], all_fixture_dirs: dict[str, Path]):
    """Every non-animated analysis run() returns valid AnalysisOutput."""
    kwargs = _build_kwargs(cls, all_fixture_dirs)
    instance = cls(**kwargs)
    output = instance.run()

    assert isinstance(output, AnalysisOutput)

    if output.data is not None:
        assert isinstance(output.data, pd.DataFrame)

    if output.figure is not None:
        assert isinstance(output.figure, Figure)

    if output.chart is not None:
        json_str = output.chart.to_json()
        parsed = json.loads(json_str)
        assert "type" in parsed
        assert "data" in parsed

    # Close figure to prevent memory leaks
    if isinstance(output.figure, Figure):
        plt.close(output.figure)


@pytest.mark.slow
@pytest.mark.parametrize("cls", _ANIMATED_ANALYSES, ids=lambda c: c.__name__)
def test_animated_analysis_run(cls: type[Analysis], all_fixture_dirs: dict[str, Path]):
    """Animated analysis run() returns valid AnalysisOutput with FuncAnimation."""
    kwargs = _build_kwargs(cls, all_fixture_dirs)
    instance = cls(**kwargs)
    output = instance.run()

    assert isinstance(output, AnalysisOutput)
    assert isinstance(output.figure, FuncAnimation)

    if output.data is not None:
        assert isinstance(output.data, pd.DataFrame)
