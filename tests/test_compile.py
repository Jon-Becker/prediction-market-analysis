"""Verify all source modules compile, import cleanly, and can be instantiated."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.common.analysis import Analysis
from src.common.indexer import Indexer

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"


def _discover_modules() -> list[str]:
    """Collect dotted module names for every .py file under src/."""
    modules = []
    for py_file in sorted(SRC_DIR.rglob("*.py")):
        if py_file.name.startswith("_"):
            continue
        relative = py_file.relative_to(ROOT_DIR)
        module_name = ".".join(relative.with_suffix("").parts)
        modules.append(module_name)
    return modules


ALL_MODULES = _discover_modules()


# -- Import tests ------------------------------------------------------------


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports(module_name: str):
    """Every source module should import without errors."""
    importlib.import_module(module_name)


# -- Analysis tests -----------------------------------------------------------


def test_analysis_discovery():
    """Analysis.load() should find at least one concrete analysis."""
    analyses = Analysis.load()
    assert len(analyses) > 0, "No analyses discovered in src/analysis/"


@pytest.mark.parametrize("cls", Analysis.load(), ids=lambda c: c.__name__)
def test_analysis_instantiation(cls: type[Analysis]):
    """Every discovered analysis should instantiate with default arguments."""
    instance = cls()
    assert isinstance(instance.name, str) and instance.name
    assert isinstance(instance.description, str) and instance.description


# -- Indexer tests ------------------------------------------------------------


def test_indexer_discovery():
    """Indexer.load() should find at least one concrete indexer."""
    indexers = Indexer.load()
    assert len(indexers) > 0, "No indexers discovered in src/indexers/"


@pytest.mark.parametrize("cls", Indexer.load(), ids=lambda c: c.__name__)
def test_indexer_instantiation(cls: type[Indexer]):
    """Every discovered indexer should instantiate with default arguments."""
    instance = cls()
    assert isinstance(instance.name, str) and instance.name
    assert isinstance(instance.description, str) and instance.description
