"""
Base class for analyses that generate figures and data outputs.

Usage:
    from common.analysis import Analysis, AnalysisOutput

    class MyAnalysis(Analysis):
        def run(self) -> AnalysisOutput:
            # Generate your figure and data
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})

            return AnalysisOutput(
                figure=fig,
                data=df,
                chart=line_chart(df.to_dict("records"), x="x", y="y"),
            )

    analysis = MyAnalysis("my_analysis", "A simple quadratic plot")
    analysis.save("output/")
"""

from __future__ import annotations

import importlib
import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from tqdm import tqdm

if TYPE_CHECKING:
    from src.common.interfaces.chart import ChartConfig


@dataclass
class AnalysisOutput:
    """Output from an analysis run."""

    figure: Figure | FuncAnimation | None = None
    data: pd.DataFrame | None = None
    chart: ChartConfig | None = None
    metadata: dict | None = None


class Analysis(ABC):
    """Base class for generating analysis outputs.

    Subclasses implement `run()` to produce figures and data.
    The `save()` method handles exporting to multiple formats.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @contextmanager
    def progress(self, description: str) -> Generator[None, None, None]:
        """Show a progress spinner while executing a block of code.

        Usage:
            with self.progress("Loading data"):
                df = con.execute("SELECT * FROM large_table").df()

        Args:
            description: Text to display alongside the spinner.
        """
        with tqdm(
            total=None,
            desc=description,
            bar_format="{desc}: {elapsed}",
            file=sys.stderr,
            leave=False,
        ) as pbar:
            yield
            pbar.update()

    @abstractmethod
    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs.

        Returns:
            AnalysisOutput containing figure, data, and optional chart config.
        """
        pass

    def save(
        self,
        output_dir: Path | str,
        formats: list[str] | None = None,
        dpi: int = 300,
    ) -> dict[str, Path]:
        """Run the analysis and save outputs to the specified directory.

        Args:
            output_dir: Directory to save outputs.
            formats: List of formats to save. Defaults to ["png", "pdf", "csv"].
                     Supported: png, pdf, svg, gif, csv, json.
            dpi: Resolution for raster formats (default: 300).

        Returns:
            Dict mapping format to saved file path.
        """
        if formats is None:
            formats = ["png", "pdf", "csv"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output = self.run()
        saved: dict[str, Path] = {}

        # Save figure formats
        if output.figure is not None:
            fig_formats = [f for f in formats if f in ("png", "pdf", "svg", "gif")]
            for fmt in fig_formats:
                path = output_dir / f"{self.name}.{fmt}"
                if fmt == "gif" and isinstance(output.figure, FuncAnimation):
                    output.figure.save(path, writer="pillow", dpi=dpi)
                elif isinstance(output.figure, Figure):
                    output.figure.savefig(path, dpi=dpi, bbox_inches="tight")
                saved[fmt] = path

            # Close figure to free memory
            if isinstance(output.figure, Figure):
                plt.close(output.figure)

        # Save CSV
        if output.data is not None and "csv" in formats:
            path = output_dir / f"{self.name}.csv"
            output.data.to_csv(path, index=False)
            saved["csv"] = path

        # Save JSON chart config
        if output.chart is not None and "json" in formats:
            path = output_dir / f"{self.name}.json"
            path.write_text(output.chart.to_json())
            saved["json"] = path

        return saved

    @classmethod
    def load(cls, analysis_dir: Path | str = "src/analysis") -> list[type[Analysis]]:
        """Scan directory for Analysis subclass implementations.

        Args:
            analysis_dir: Directory to scan for analysis modules.

        Returns:
            List of Analysis subclass types found.
        """
        analysis_dir = Path(analysis_dir)
        if not analysis_dir.exists():
            return []

        analyses: list[type[Analysis]] = []

        for py_file in analysis_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue

            relative_path = py_file.relative_to(analysis_dir)
            module_parts = relative_path.with_suffix("").parts
            module_name = "src.analysis." + ".".join(module_parts)
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, cls) and obj is not cls and not inspect.isabstract(obj):
                    analyses.append(obj)

        return analyses
