from __future__ import annotations

from typing import TypeAlias, TYPE_CHECKING, Protocol, Any, TypeVar, Generic, Literal
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from uuid import UUID
from collections.abc import Mapping, Iterator, Sequence, Callable

if TYPE_CHECKING:
    import pyarrow as pa
    from ._expression import Expression
    from ._sqltypes import DuckDBPyType

# Numpy protocols

_S_co = TypeVar("_S_co", bound=tuple[Any, ...], covariant=True)
_D_co = TypeVar("_D_co", covariant=True)

class NPProtocol(Protocol):
    """Base Protocol for numpy objects."""
    @property
    def dtype(self) -> Any: ...
    @property
    def ndim(self) -> int: ...
    def __array__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __array_wrap__(self, *args: Any, **kwargs: Any) -> Any: ...
    @property
    def __array_interface__(self) -> dict[str, Any]: ...
    @property
    def __array_priority__(self) -> float: ...

class NPScalarTypeLike(NPProtocol, Protocol):
    @property
    def itemsize(self) -> int: ...

class NPArrayLike(NPProtocol, Generic[_S_co, _D_co], Protocol):
    """`numpy.ndarray` Protocol.

    This is needed to accept numpy arrays as literals in expressions, without emitting type checker errors about unknown symbol if the user doesn't have `numpy` installed.

    Note:
        Using `np.typing.NDArray` is still the best option for return types.
    """
    def __len__(self) -> int: ...
    def __contains__(self, value: object, /) -> bool: ...
    def __iter__(self) -> Iterator[_D_co]: ...
    def __array_finalize__(self, *args: Any, **kwargs: Any) -> None: ...
    def __getitem__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __setitem__(self, *args: Any, **kwargs: Any) -> None: ...
    @property
    def shape(self) -> _S_co: ...
    @property
    def size(self) -> int: ...

# Expression and values conversions

NumericLiteral: TypeAlias = int | float
"""Python objects that can be converted to a numerical `Expression` or `DuckDBPyType`."""
TemporalLiteral: TypeAlias = date | datetime | time | timedelta
BlobLiteral: TypeAlias = bytes | bytearray
"""Python objects that can be converted to a `BLOB` `ConstantExpression` or `DuckDBPyType`.

Note:
    `bytes` can also be converted to a `BITSTRING`.
"""
ScalarLiteral: TypeAlias = NumericLiteral | BlobLiteral | str | bool
NonNestedLiteral: TypeAlias = ScalarLiteral | TemporalLiteral | UUID | Decimal | memoryview

# NOTE:
# Using `Sequence` and `Mapping` instead of `list | tuple` and `dict` would make the covariance of the element types work.
# Thus, this would allow to avoid the use of `Any` for them.
# However, this would also be incorrect at runtime, since only the 3 aforementioned containers types are accepted.
NestedLiteral: TypeAlias = list[Any] | tuple[Any, ...] | dict[Any, Any] | NPArrayLike[Any, Any]
"""Containers types that can be converted to a nested `ConstantExpression` (e.g. to `ARRAY` or `STRUCT`).

Those types can be arbitrarily nested, as long as their leaf values are `PythonLiteral`."""

PythonLiteral: TypeAlias = NonNestedLiteral | NestedLiteral | None
"""Python objects that can be converted to a `ConstantExpression`."""

IntoExprColumn: TypeAlias = Expression | str
"""Types that are, or can be used as a `ColumnExpression`."""

IntoExpr: TypeAlias = IntoExprColumn | PythonLiteral
"""Any type that can be converted to an `Expression` (or is already one).

See Also:
    https://duckdb.org/docs/stable/clients/python/conversion
"""

IntoValues: TypeAlias = list[PythonLiteral] | tuple[Expression, ...] | Expression | NPArrayLike[Any, Any]
"""Types that can be converted to a table of values."""

# PyType conversions

Builtins: TypeAlias = Literal[
    "bigint",
    "bit",
    "bignum",
    "blob",
    "boolean",
    "date",
    "double",
    "float",
    "geometry",
    "hugeint",
    "integer",
    "interval",
    "smallint",
    "null",
    "time with time zone",
    "time",
    "time_ns",
    "timestamp_ms",
    "timestamp_ns",
    "timestamp_s",
    "timestamp with time zone",
    "timestamp",
    "tinyint",
    "ubigint",
    "uhugeint",
    "uinteger",
    "usmallint",
    "utinyint",
    "uuid",
    "varchar",
    "variant",
]
"""Literals `str` that can be converted into `DuckDBPyType` instances.

Note:
    Passing the same values in uppercase is also accepted. 
    We use lowercase here to be able to reuse this `Literal` in the `DTypeIdentifiers` `Literal`.
"""

NestedIds: TypeAlias = Literal["list", "struct", "array", "enum", "map", "decimal", "union"]
"""Identifiers for nested types in `DuckDBPyType.id`."""

PyTypeIds: TypeAlias = Builtins | NestedIds
"""All possible identifiers for `DuckDBPyType.id`."""

StrIntoPyType: TypeAlias = Builtins | Literal["json"] | str
"""Any `str` that can be converted into a `DuckDBPyType`.

The `DuckDBPyType` not present in the literal values are the composed ones, like `STRUCT` or `DECIMAL`.

Note:
    A `StrEnum` will be handled the same way as a `str`."""

# NOTE:
# the `dict` and `list` types are `Any` due to the same limitation mentioned in `NestedLiteral`.
IntoPyType: TypeAlias = (
    DuckDBPyType
    | StrIntoPyType
    | type[NPScalarTypeLike]
    | type[ScalarLiteral]
    | type[list[Any]]
    | type[dict[Any, Any]]
    | dict[Any, Any]
)
"""All types that can be converted to a `DuckDBPyType`.

They can be arbitrarily nested as long as their leaf values are convertible to `DuckDBPyType`.

See Also:
    https://duckdb.org/docs/stable/clients/python/types
"""

# NOTE: here we keep the covariance "hack" and warn the user in the docstring,
# because otherwise we can just resort to `Any` for the `dict` and `list` types.
IntoFields: TypeAlias = Mapping[str, IntoPyType] | Sequence[IntoPyType]
"""Types that can be converted either into: 

- a nested `DuckDBPyType` (e.g. `STRUCT` or `UNION`)
- a schema for file reads

Warning:
    Only `dict` and `list` containers are accepted at runtime. 
    We use `Mapping` and `Sequence` here to satisfy the covariance of the element types.
"""

# Files related

# NOTE: ideally HiveTypes should also be accepted as a Mapping[str, StrIntoPyType].
ColumnsTypes: TypeAlias = Mapping[str, StrIntoPyType]
HiveTypes: TypeAlias = dict[str, StrIntoPyType]
ParquetFieldIdsType: TypeAlias = Mapping[str, int | ParquetFieldIdsType]

_Auto: TypeAlias = Literal["auto"]
ParquetFieldsOptions: TypeAlias = _Auto | ParquetFieldIdsType
"""Types accepted for the `field_ids` parameter in parquet writing methods."""

CsvEncoding: TypeAlias = Literal["utf-8", "utf-16", "latin-1"] | str
"""Encoding options.

All available options not in the literal values can be seen here:
    https://duckdb.org/docs/stable/core_extensions/encodings
"""
JsonFormat: TypeAlias = _Auto | Literal["unstructured", "newline_delimited", "array"]
JsonRecordOptions: TypeAlias = _Auto | Literal["true", "false"]

# compression kinds

_CompressionOptions: TypeAlias = Literal["gzip", "zstd"]
"""Generally available compression options."""
_None: TypeAlias = Literal["none"]
CsvCompression: TypeAlias = _Auto | _None | _CompressionOptions
JsonCompression: TypeAlias = Literal["auto_detect"] | _None | _CompressionOptions
ParquetCompression: TypeAlias = Literal["uncompressed", "brotli", "snappy", "lz4", "lz4_raw"] | _CompressionOptions

# Other

JoinType: TypeAlias = Literal["inner", "left", "right", "outer", "semi", "anti"]
"""Types of join accepted by `DuckDBPyRelation.join` method."""

ProfilerFormat: TypeAlias = Literal["json", "query_tree", "query_tree_optimizer", "no_output", "html", "graphviz"]
"""Formats available in `get_profiling_information` method/function."""
# TODO: this should be a `Protocol` just like `NPArrayLike`.
ArrowUDF: TypeAlias = Callable[..., pa.Table | pa.Array | pa.ChunkedArray]
"""Type accepted for Python UDFs that return Arrow data."""
