"""3D maker returns over time, run on the historical Kalshi dataset."""

from __future__ import annotations

from pathlib import Path

# Import the module, not the class: Analysis.load() registers every Analysis class
# found in a module's namespace, so importing the class would list it twice.
from src.analysis.kalshi import maker_returns_over_time_3d as base
from src.analysis.kalshi.util.events import DEFAULT_EVENTS_DIR


class MakerReturnsOverTime3DHistoricalAnalysis(base.MakerReturnsOverTime3DAnalysis):
    """Same analysis as MakerReturnsOverTime3DAnalysis, reading the historical markets and trades."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        events_dir: Path | str | None = None,
    ):
        base_dir = Path(__file__).parent.parent.parent.parent
        super().__init__(
            trades_dir=trades_dir or base_dir / "data" / "kalshi" / "historical" / "trades",
            markets_dir=markets_dir or base_dir / "data" / "kalshi" / "historical" / "markets",
            events_dir=events_dir or DEFAULT_EVENTS_DIR,
        )
        self.name = "3d_maker_returns_over_time_historical"
        self.description = (
            "Maker excess returns by position direction (YES vs NO) and event category over time (historical)"
        )
