"""Test Analysis.save() in isolation with mock outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType


class _StubAnalysis(Analysis):
    """Stub analysis whose run() returns a pre-built AnalysisOutput."""

    def __init__(self, output: AnalysisOutput):
        super().__init__(name="stub_analysis", description="Stub for testing save()")
        self._output = output

    def run(self) -> AnalysisOutput:
        return self._output


def _make_figure() -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    return fig


def _make_dataframe() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})


def _make_chart() -> ChartConfig:
    return ChartConfig(
        type=ChartType.LINE,
        data=[{"x": 1, "y": 1}, {"x": 2, "y": 4}, {"x": 3, "y": 9}],
        xKey="x",
        yKeys=["y"],
        title="Test Chart",
    )


def _make_animation() -> FuncAnimation:
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [])

    def animate(frame):
        line.set_data([0, frame], [0, frame])
        return (line,)

    return FuncAnimation(fig, animate, frames=3, interval=50, blit=True)


class TestSaveStaticFigureWithData:
    """Static figure + DataFrame + ChartConfig saves png, pdf, csv, json."""

    def test_saves_all_formats(self, tmp_path: Path):
        output = AnalysisOutput(figure=_make_figure(), data=_make_dataframe(), chart=_make_chart())
        saved = _StubAnalysis(output).save(tmp_path, formats=["png", "pdf", "csv", "json"])

        assert set(saved.keys()) == {"png", "pdf", "csv", "json"}
        for path in saved.values():
            assert path.exists()
            assert path.stat().st_size > 0

    def test_csv_roundtrips(self, tmp_path: Path):
        df = _make_dataframe()
        output = AnalysisOutput(data=df, figure=_make_figure())
        saved = _StubAnalysis(output).save(tmp_path, formats=["csv"])

        loaded = pd.read_csv(saved["csv"])
        pd.testing.assert_frame_equal(loaded, df)

    def test_excel_roundtrips(self, tmp_path: Path):
        df = _make_dataframe()
        output = AnalysisOutput(data=df, figure=_make_figure())
        saved = _StubAnalysis(output).save(tmp_path, formats=["xlsx"])

        loaded = pd.read_excel(saved["xlsx"])
        pd.testing.assert_frame_equal(loaded, df)

    def test_json_has_required_keys(self, tmp_path: Path):
        output = AnalysisOutput(figure=_make_figure(), chart=_make_chart())
        saved = _StubAnalysis(output).save(tmp_path, formats=["json"])

        parsed = json.loads(saved["json"].read_text())
        assert "type" in parsed
        assert "data" in parsed


class TestSaveAnimation:
    """FuncAnimation + DataFrame saves gif and csv."""

    def test_saves_gif_and_csv(self, tmp_path: Path):
        output = AnalysisOutput(figure=_make_animation(), data=_make_dataframe())
        saved = _StubAnalysis(output).save(tmp_path, formats=["gif", "csv"])

        assert "gif" in saved
        assert "csv" in saved
        assert saved["gif"].exists()
        assert saved["csv"].exists()


class TestSaveFormatSkipping:
    """Incompatible format/figure combos are silently skipped."""

    def test_static_figure_skips_gif(self, tmp_path: Path):
        output = AnalysisOutput(figure=_make_figure())
        saved = _StubAnalysis(output).save(tmp_path, formats=["gif"])

        assert "gif" not in saved

    def test_animation_skips_png_pdf(self, tmp_path: Path):
        output = AnalysisOutput(figure=_make_animation())
        saved = _StubAnalysis(output).save(tmp_path, formats=["png", "pdf"])

        assert "png" not in saved
        assert "pdf" not in saved


class TestSaveEdgeCases:
    """Edge cases: data-only, empty output, defaults, figure closure, dir creation."""

    def test_data_only(self, tmp_path: Path):
        output = AnalysisOutput(data=_make_dataframe())
        saved = _StubAnalysis(output).save(tmp_path, formats=["csv"])

        assert "csv" in saved
        assert saved["csv"].exists()

    def test_empty_output(self, tmp_path: Path):
        output = AnalysisOutput()
        saved = _StubAnalysis(output).save(tmp_path)

        assert saved == {}

    def test_default_formats(self, tmp_path: Path):
        output = AnalysisOutput(figure=_make_figure(), data=_make_dataframe())
        saved = _StubAnalysis(output).save(tmp_path)

        assert set(saved.keys()) == {"png", "pdf", "csv"}

    def test_figure_closed_after_save(self, tmp_path: Path):
        fig = _make_figure()
        output = AnalysisOutput(figure=fig)
        _StubAnalysis(output).save(tmp_path, formats=["png"])

        assert fig.number not in plt.get_fignums()

    def test_creates_output_dir(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        output = AnalysisOutput(figure=_make_figure())
        saved = _StubAnalysis(output).save(nested, formats=["png"])

        assert nested.exists()
        assert "png" in saved
