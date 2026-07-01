from typing import TYPE_CHECKING

from .types import StructType

if TYPE_CHECKING:
    from .dataframe import DataFrame
    from .session import SparkSession

PrimitiveType = bool | float | int | str
OptionalPrimitiveType = PrimitiveType | None


class DataStreamWriter:  # noqa: D101
    def __init__(self, dataframe: "DataFrame") -> None:  # noqa: D107
        self.dataframe = dataframe

    def toTable(self, table_name: str) -> None:  # noqa: D102
        # Should we register the dataframe or create a table from the contents?
        raise NotImplementedError


class DataStreamReader:  # noqa: D101
    def __init__(self, session: "SparkSession") -> None:  # noqa: D107
        self.session = session

    def load(  # noqa: D102
        self,
        path: str | None = None,
        format: str | None = None,
        schema: StructType | str | None = None,
        **options: OptionalPrimitiveType,
    ) -> "DataFrame":
        raise NotImplementedError


__all__ = ["DataStreamReader", "DataStreamWriter"]
