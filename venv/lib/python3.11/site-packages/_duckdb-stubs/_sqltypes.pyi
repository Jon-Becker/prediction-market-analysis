import duckdb
import typing
from ._typing import StrIntoPyType, PyTypeIds

__all__: list[str] = [
    "BIGINT",
    "BIT",
    "BLOB",
    "BOOLEAN",
    "DATE",
    "DOUBLE",
    "FLOAT",
    "HUGEINT",
    "INTEGER",
    "INTERVAL",
    "SMALLINT",
    "SQLNULL",
    "TIME",
    "TIMESTAMP",
    "TIMESTAMP_MS",
    "TIMESTAMP_NS",
    "TIMESTAMP_S",
    "TIMESTAMP_TZ",
    "TIME_TZ",
    "TINYINT",
    "UBIGINT",
    "UHUGEINT",
    "UINTEGER",
    "USMALLINT",
    "UTINYINT",
    "UUID",
    "VARCHAR",
    "VARIANT",
    "DuckDBPyType",
]

class DuckDBPyType:
    def __eq__(self, other: object) -> bool: ...
    def __getattr__(self, name: str) -> DuckDBPyType: ...
    def __getitem__(self, name: str) -> DuckDBPyType: ...
    def __hash__(self) -> int: ...
    @typing.overload
    def __init__(self, type_str: StrIntoPyType, connection: duckdb.DuckDBPyConnection) -> None: ...
    @typing.overload
    def __init__(self, obj: object) -> None: ...
    @property
    def children(self) -> list[tuple[str, DuckDBPyType | int | list[str]]]: ...
    @property
    def id(self) -> PyTypeIds: ...

BIGINT: DuckDBPyType  # value = BIGINT
BIT: DuckDBPyType  # value = BIT
BLOB: DuckDBPyType  # value = BLOB
BOOLEAN: DuckDBPyType  # value = BOOLEAN
DATE: DuckDBPyType  # value = DATE
DOUBLE: DuckDBPyType  # value = DOUBLE
FLOAT: DuckDBPyType  # value = FLOAT
HUGEINT: DuckDBPyType  # value = HUGEINT
INTEGER: DuckDBPyType  # value = INTEGER
INTERVAL: DuckDBPyType  # value = INTERVAL
SMALLINT: DuckDBPyType  # value = SMALLINT
SQLNULL: DuckDBPyType  # value = "NULL"
TIME: DuckDBPyType  # value = TIME
TIME_NS: DuckDBPyType  # value = TIME_NS
TIMESTAMP: DuckDBPyType  # value = TIMESTAMP
TIMESTAMP_MS: DuckDBPyType  # value = TIMESTAMP_MS
TIMESTAMP_NS: DuckDBPyType  # value = TIMESTAMP_NS
TIMESTAMP_S: DuckDBPyType  # value = TIMESTAMP_S
TIMESTAMP_TZ: DuckDBPyType  # value = TIMESTAMP WITH TIME ZONE
TIME_TZ: DuckDBPyType  # value = TIME WITH TIME ZONE
TINYINT: DuckDBPyType  # value = TINYINT
UBIGINT: DuckDBPyType  # value = UBIGINT
UHUGEINT: DuckDBPyType  # value = UHUGEINT
UINTEGER: DuckDBPyType  # value = UINTEGER
USMALLINT: DuckDBPyType  # value = USMALLINT
UTINYINT: DuckDBPyType  # value = UTINYINT
UUID: DuckDBPyType  # value = UUID
VARCHAR: DuckDBPyType  # value = VARCHAR
VARIANT: DuckDBPyType  # value = VARIANT
