"""
Python types for generating valid ResearchChart configurations.

Usage:
    from common.chart import ChartConfig, ChartType, UnitType, Series

    config = ChartConfig(
        type=ChartType.LINE,
        data=[{"x": 1, "y": 10}, {"x": 2, "y": 20}],
        xKey="x",
        yKeys=["y"],
        title="My Chart",
        yUnit=UnitType.DOLLARS,
    )

    # Export to dict for JSON serialization
    json_config = config.to_dict()
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    STACKED_BAR = "stacked-bar"
    STACKED_BAR_100 = "stacked-bar-100"
    AREA = "area"
    STACKED_AREA_100 = "stacked-area-100"
    PIE = "pie"
    SCATTER = "scatter"
    TREEMAP = "treemap"
    HEATMAP = "heatmap"


class UnitType(str, Enum):
    DOLLARS = "dollars"
    PERCENT = "percent"
    BYTES = "bytes"
    ETH = "eth"
    BTC = "btc"
    CENTS = "cents"
    NUMBER = "number"


class ScaleType(str, Enum):
    LINEAR = "linear"
    LOG = "log"


@dataclass
class Series:
    """A named data series for scatter charts."""

    name: str
    data: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "data": self.data}


@dataclass
class ChartConfig:
    """
    Configuration for a ResearchChart component.

    Attributes:
        type: The chart type (line, bar, pie, etc.)
        data: Array of data points as dicts
        series: Named series for scatter charts
        xKey: Key for x-axis values (default: "x")
        yKeys: Keys for y-axis values (default: ["y"])
        yKey: Single y-key for heatmaps
        zKey: Key for z-axis (bubble size in scatter)
        title: Chart title
        height: Chart height in pixels (default: 300)
        stacked: Whether to stack bar/area charts
        nameKey: Key for names in pie/treemap (default: "name")
        valueKey: Key for values in pie/treemap (default: "value")
        childrenKey: Key for children in treemap (default: "children")
        xScale: X-axis scale type (linear or log)
        yScale: Y-axis scale type (linear or log)
        yUnit: Unit type for formatting y values
        strokeDasharrays: Dash patterns per series (e.g., "5 5" for dashed)
        caption: Caption text below chart
        colors: Custom colors per series key
        xLabel: X-axis label
        yLabel: Y-axis label
    """

    type: ChartType
    data: list[dict[str, Any]]
    series: list[Series] | None = None
    xKey: str | None = None
    yKeys: list[str] | None = None
    yKey: str | None = None
    zKey: str | None = None
    title: str | None = None
    height: int | None = None
    stacked: bool | None = None
    nameKey: str | None = None
    valueKey: str | None = None
    childrenKey: str | None = None
    xScale: ScaleType | None = None
    yScale: ScaleType | None = None
    yUnit: UnitType | None = None
    strokeDasharrays: list[str | None] | None = None
    caption: str | None = None
    colors: dict[str, str] | None = None
    xLabel: str | None = None
    yLabel: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization, omitting None values."""
        result: dict[str, Any] = {"type": self.type.value, "data": self.data}

        if self.series is not None:
            result["series"] = [s.to_dict() for s in self.series]
        if self.xKey is not None:
            result["xKey"] = self.xKey
        if self.yKeys is not None:
            result["yKeys"] = self.yKeys
        if self.yKey is not None:
            result["yKey"] = self.yKey
        if self.zKey is not None:
            result["zKey"] = self.zKey
        if self.title is not None:
            result["title"] = self.title
        if self.height is not None:
            result["height"] = self.height
        if self.stacked is not None:
            result["stacked"] = self.stacked
        if self.nameKey is not None:
            result["nameKey"] = self.nameKey
        if self.valueKey is not None:
            result["valueKey"] = self.valueKey
        if self.childrenKey is not None:
            result["childrenKey"] = self.childrenKey
        if self.xScale is not None:
            result["xScale"] = self.xScale.value
        if self.yScale is not None:
            result["yScale"] = self.yScale.value
        if self.yUnit is not None:
            result["yUnit"] = self.yUnit.value
        if self.strokeDasharrays is not None:
            result["strokeDasharrays"] = self.strokeDasharrays
        if self.caption is not None:
            result["caption"] = self.caption
        if self.colors is not None:
            result["colors"] = self.colors
        if self.xLabel is not None:
            result["xLabel"] = self.xLabel
        if self.yLabel is not None:
            result["yLabel"] = self.yLabel

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def line_chart(
    data: list[dict[str, Any]],
    x: str = "x",
    y: list[str] | str = "y",
    **kwargs: Any,
) -> ChartConfig:
    """Create a line chart configuration."""
    yKeys = [y] if isinstance(y, str) else y
    return ChartConfig(type=ChartType.LINE, data=data, xKey=x, yKeys=yKeys, **kwargs)


def bar_chart(
    data: list[dict[str, Any]],
    x: str = "x",
    y: list[str] | str = "y",
    stacked: bool = False,
    **kwargs: Any,
) -> ChartConfig:
    """Create a bar chart configuration."""
    yKeys = [y] if isinstance(y, str) else y
    chart_type = ChartType.STACKED_BAR if stacked else ChartType.BAR
    return ChartConfig(type=chart_type, data=data, xKey=x, yKeys=yKeys, **kwargs)


def area_chart(
    data: list[dict[str, Any]],
    x: str = "x",
    y: list[str] | str = "y",
    stacked: bool = False,
    **kwargs: Any,
) -> ChartConfig:
    """Create an area chart configuration."""
    yKeys = [y] if isinstance(y, str) else y
    return ChartConfig(type=ChartType.AREA, data=data, xKey=x, yKeys=yKeys, stacked=stacked, **kwargs)


def pie_chart(
    data: list[dict[str, Any]],
    name: str = "name",
    value: str = "value",
    **kwargs: Any,
) -> ChartConfig:
    """Create a pie chart configuration."""
    return ChartConfig(type=ChartType.PIE, data=data, nameKey=name, valueKey=value, **kwargs)


def scatter_chart(
    data: list[dict[str, Any]],
    x: str = "x",
    y: str = "y",
    z: str | None = None,
    series: list[Series] | None = None,
    **kwargs: Any,
) -> ChartConfig:
    """Create a scatter chart configuration."""
    return ChartConfig(
        type=ChartType.SCATTER,
        data=data,
        xKey=x,
        yKeys=[y],
        zKey=z,
        series=series,
        **kwargs,
    )


def heatmap(
    data: list[dict[str, Any]],
    x: str = "x",
    y: str = "y",
    value: str = "value",
    **kwargs: Any,
) -> ChartConfig:
    """Create a heatmap configuration."""
    return ChartConfig(type=ChartType.HEATMAP, data=data, xKey=x, yKey=y, valueKey=value, **kwargs)


def treemap(
    data: list[dict[str, Any]],
    name: str = "name",
    value: str = "value",
    children: str = "children",
    **kwargs: Any,
) -> ChartConfig:
    """Create a treemap configuration."""
    return ChartConfig(
        type=ChartType.TREEMAP,
        data=data,
        nameKey=name,
        valueKey=value,
        childrenKey=children,
        **kwargs,
    )
