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

import importlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Type, Union


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
    def load(cls, indexer_dir: Union[Path, str] = "src/indexers") -> List[Type["Indexer"]]:
        """Scan directory for Indexer subclass implementations.

        Args:
            indexer_dir: Directory to scan for indexer modules.

        Returns:
            List of Indexer subclass types found.
        """
        indexer_dir = Path(indexer_dir)
        if not indexer_dir.exists():
            return []

        indexers: List[Type["Indexer"]] = []

        for py_file in indexer_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue

            relative_path = py_file.relative_to(indexer_dir)
            module_parts = relative_path.with_suffix("").parts
            module_name = "src.indexers." + ".".join(module_parts)
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, cls)
                    and obj is not cls
                    and not inspect.isabstract(obj)
                ):
                    indexers.append(obj)

        return indexers
