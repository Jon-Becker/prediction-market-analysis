"""
Base class for data indexers that fetch and store market data.

Usage:
    from src.common.indexer import Indexer

    class MyIndexer(Indexer):
        def run(self) -> None:
            # Fetch and store data
            pass

    indexer = MyIndexer("my_indexer", "Fetches data from source")
    indexer.run()
"""

from __future__ import annotations

import ast
import importlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path


def _declares_direct_indexer_subclass(py_file: Path) -> bool:
    """Return whether a module statically declares an ``Indexer`` subclass.

    Discovery must not import arbitrary helper modules merely because they live
    below ``src/indexers``.  Besides reducing import side effects, this keeps
    explicitly quarantined research helpers outside the interactive production
    import graph.
    """

    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    except (OSError, SyntaxError, UnicodeError):
        return False

    indexer_names = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "src.common.indexer"
        for alias in node.names
        if alias.name == "Indexer"
    }
    if not indexer_names:
        return False
    return any(
        isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id in indexer_names for base in node.bases)
        for node in tree.body
    )


class Indexer(ABC):
    """Base class for data indexers.

    Subclasses implement `run()` to fetch and store data.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self) -> None:
        """Execute the indexer to fetch and store data."""
        pass

    @classmethod
    def load(cls, indexer_dir: Path | str = "src/indexers") -> list[type[Indexer]]:
        """Scan directory for Indexer subclass implementations.

        Args:
            indexer_dir: Directory to scan for indexer modules.

        Returns:
            List of Indexer subclass types found.
        """
        indexer_dir = Path(indexer_dir)
        if not indexer_dir.exists():
            return []

        indexers: list[type[Indexer]] = []

        for py_file in indexer_dir.glob("**/*.py"):
            if py_file.name.startswith("_") or not _declares_direct_indexer_subclass(py_file):
                continue

            relative_path = py_file.relative_to(indexer_dir)
            module_parts = relative_path.with_suffix("").parts
            module_name = "src.indexers." + ".".join(module_parts)
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, cls) and obj is not cls and not inspect.isabstract(obj):
                    indexers.append(obj)

        return indexers
