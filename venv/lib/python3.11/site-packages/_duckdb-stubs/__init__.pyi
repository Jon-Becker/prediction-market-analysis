import os
import pathlib
import typing
from typing_extensions import Self
from ._expression import Expression
from ._enums import (
    CSVLineTerminator,
    StatementType,
    ExpectedResultType,
    ExplainType,
    PythonExceptionHandling,
    RenderMode,
    token_type,
)

if typing.TYPE_CHECKING:
    import fsspec
    import numpy as np
    import polars
    import pandas
    import pyarrow.lib
    from builtins import list as lst
    from collections.abc import Callable, Iterable, Sequence, Mapping
    from ._typing import (
        ParquetFieldsOptions,
        IntoExpr,
        IntoExprColumn,
        PythonLiteral,
        IntoValues,
        IntoPyType,
        IntoFields,
        StrIntoPyType,
        JoinType,
        JsonCompression,
        JsonFormat,
        JsonRecordOptions,
        CsvEncoding,
        CsvCompression,
        HiveTypes,
        ColumnsTypes,
        ProfilerFormat,
        ParquetCompression,
        ArrowUDF,
    )
    from ._enums import ExplainTypeLiteral, RenderModeLiteral
    from duckdb import sqltypes, func

__all__: lst[str] = [
    "BinderException",
    "CSVLineTerminator",
    "CaseExpression",
    "CatalogException",
    "CoalesceOperator",
    "ColumnExpression",
    "ConnectionException",
    "ConstantExpression",
    "ConstraintException",
    "ConversionException",
    "DataError",
    "DatabaseError",
    "DefaultExpression",
    "DependencyException",
    "DuckDBPyConnection",
    "DuckDBPyRelation",
    "Error",
    "ExpectedResultType",
    "ExplainType",
    "Expression",
    "FatalException",
    "FunctionExpression",
    "HTTPException",
    "IOException",
    "IntegrityError",
    "InternalError",
    "InternalException",
    "InterruptException",
    "InvalidInputException",
    "InvalidTypeException",
    "LambdaExpression",
    "NotImplementedException",
    "NotSupportedError",
    "OperationalError",
    "OutOfMemoryException",
    "OutOfRangeException",
    "ParserException",
    "PermissionException",
    "ProgrammingError",
    "PythonExceptionHandling",
    "RenderMode",
    "SQLExpression",
    "SequenceException",
    "SerializationException",
    "StarExpression",
    "Statement",
    "StatementType",
    "SyntaxException",
    "TransactionException",
    "TypeMismatchException",
    "Warning",
    "aggregate",
    "alias",
    "apilevel",
    "append",
    "array_type",
    "arrow",
    "begin",
    "checkpoint",
    "close",
    "commit",
    "connect",
    "create_function",
    "cursor",
    "decimal_type",
    "default_connection",
    "description",
    "df",
    "disable_profiling",
    "distinct",
    "dtype",
    "duplicate",
    "enable_profiling",
    "enum_type",
    "execute",
    "executemany",
    "extract_statements",
    "to_arrow_reader",
    "to_arrow_table",
    "fetch_arrow_table",
    "fetch_df",
    "fetch_df_chunk",
    "fetch_record_batch",
    "fetchall",
    "fetchdf",
    "fetchmany",
    "fetchnumpy",
    "fetchone",
    "filesystem_is_registered",
    "filter",
    "from_arrow",
    "from_csv_auto",
    "from_df",
    "from_parquet",
    "from_query",
    "get_profiling_information",
    "get_table_names",
    "install_extension",
    "interrupt",
    "limit",
    "list_filesystems",
    "list_type",
    "load_extension",
    "map_type",
    "order",
    "paramstyle",
    "pl",
    "project",
    "query",
    "query_df",
    "query_progress",
    "read_csv",
    "read_json",
    "read_parquet",
    "register",
    "register_filesystem",
    "remove_function",
    "rollback",
    "row_type",
    "rowcount",
    "set_default_connection",
    "sql",
    "sqltype",
    "string_type",
    "struct_type",
    "table",
    "table_function",
    "tf",
    "threadsafety",
    "token_type",
    "tokenize",
    "torch",
    "type",
    "union_type",
    "unregister",
    "unregister_filesystem",
    "values",
    "view",
    "write_csv",
]

class BinderException(ProgrammingError): ...
class CatalogException(ProgrammingError): ...
class ConnectionException(OperationalError): ...
class ConstraintException(IntegrityError): ...
class ConversionException(DataError): ...
class DataError(DatabaseError): ...
class DatabaseError(Error): ...
class DependencyException(DatabaseError): ...

class DuckDBPyConnection:
    def __del__(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def append(self, table_name: str, df: pandas.DataFrame, *, by_name: bool = False) -> DuckDBPyConnection: ...
    def array_type(self, type: IntoPyType, size: typing.SupportsInt) -> sqltypes.DuckDBPyType: ...
    def arrow(self, rows_per_batch: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader:
        """Alias of to_arrow_reader(). We recommend using to_arrow_reader() instead."""
        ...
    def to_arrow_reader(self, batch_size: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader: ...
    def to_arrow_table(self, batch_size: typing.SupportsInt = 1000000) -> pyarrow.lib.Table: ...
    def begin(self) -> DuckDBPyConnection: ...
    def checkpoint(self) -> DuckDBPyConnection: ...
    def close(self) -> None: ...
    def commit(self) -> DuckDBPyConnection: ...
    @typing.overload
    def create_function(
        self,
        name: str,
        function: Callable[..., PythonLiteral],
        parameters: lst[IntoPyType] | None = None,
        return_type: IntoPyType | None = None,
        *,
        type: func.PythonUDFType = func.PythonUDFType.NATIVE,
        null_handling: func.FunctionNullHandling = ...,
        exception_handling: PythonExceptionHandling = ...,
        side_effects: bool = False,
    ) -> DuckDBPyConnection: ...
    @typing.overload
    def create_function(
        self,
        name: str,
        function: ArrowUDF,
        parameters: lst[IntoPyType] | None = None,
        return_type: IntoPyType | None = None,
        *,
        type: func.PythonUDFType = func.PythonUDFType.ARROW,
        null_handling: func.FunctionNullHandling = ...,
        exception_handling: PythonExceptionHandling = ...,
        side_effects: bool = False,
    ) -> DuckDBPyConnection: ...
    def cursor(self) -> DuckDBPyConnection: ...
    def decimal_type(self, width: typing.SupportsInt, scale: typing.SupportsInt) -> sqltypes.DuckDBPyType: ...
    def df(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def dtype(self, type_str: StrIntoPyType) -> sqltypes.DuckDBPyType: ...
    def duplicate(self) -> DuckDBPyConnection: ...
    def enum_type(self, name: str, type: sqltypes.DuckDBPyType, values: lst[typing.Any]) -> sqltypes.DuckDBPyType: ...
    def execute(self, query: Statement | str, parameters: object = None) -> DuckDBPyConnection: ...
    def executemany(self, query: Statement | str, parameters: object = None) -> DuckDBPyConnection: ...
    def extract_statements(self, query: str) -> lst[Statement]: ...
    def fetch_arrow_table(self, rows_per_batch: typing.SupportsInt = 1000000) -> pyarrow.lib.Table:
        """Deprecated: use to_arrow_table() instead."""
        ...
    def fetch_df(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def fetch_df_chunk(
        self, vectors_per_chunk: typing.SupportsInt = 1, *, date_as_object: bool = False
    ) -> pandas.DataFrame: ...
    def fetch_record_batch(self, rows_per_batch: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader:
        """Deprecated: use to_arrow_reader() instead."""
        ...
    def fetchall(self) -> lst[tuple[typing.Any, ...]]: ...
    def fetchdf(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def fetchmany(self, size: typing.SupportsInt = 1) -> lst[tuple[typing.Any, ...]]: ...
    def fetchnumpy(self) -> dict[str, np.typing.NDArray[typing.Any] | pandas.Categorical]: ...
    def fetchone(self) -> tuple[typing.Any, ...] | None: ...
    def filesystem_is_registered(self, name: str) -> bool: ...
    def from_arrow(self, arrow_object: object) -> DuckDBPyRelation: ...
    def from_csv_auto(
        self,
        path_or_buffer: str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str],
        header: bool | int | None = None,
        compression: CsvCompression | None = None,
        sep: str | None = None,
        delimiter: str | None = None,
        files_to_sniff: int | None = None,
        comment: str | None = None,
        thousands: str | None = None,
        dtype: IntoFields | None = None,
        na_values: str | lst[str] | None = None,
        skiprows: int | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        encoding: CsvEncoding | None = None,
        parallel: bool | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        sample_size: int | None = None,
        auto_detect: bool | int | None = None,
        all_varchar: bool | None = None,
        normalize_names: bool | None = None,
        null_padding: bool | None = None,
        names: lst[str] | None = None,
        lineterminator: CSVLineTerminator | None = None,
        columns: ColumnsTypes | None = None,
        auto_type_candidates: lst[StrIntoPyType] | None = None,
        max_line_size: int | None = None,
        ignore_errors: bool | None = None,
        store_rejects: bool | None = None,
        rejects_table: str | None = None,
        rejects_scan: str | None = None,
        rejects_limit: int | None = None,
        force_not_null: lst[str] | None = None,
        buffer_size: int | None = None,
        decimal: str | None = None,
        allow_quoted_nulls: bool | None = None,
        filename: bool | str | None = None,
        hive_partitioning: bool | None = None,
        union_by_name: bool | None = None,
        hive_types: HiveTypes | None = None,
        hive_types_autocast: bool | None = None,
        strict_mode: bool | None = None,
    ) -> DuckDBPyRelation: ...
    def from_df(self, df: pandas.DataFrame) -> DuckDBPyRelation: ...
    def from_parquet(
        self,
        path_or_buffer: str
        | bytes
        | os.PathLike[str]
        | os.PathLike[bytes]
        | typing.IO[bytes]
        | typing.IO[str]
        | Sequence[str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str]],
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: ParquetCompression | None = None,
    ) -> DuckDBPyRelation: ...
    def from_query(self, query: str, *, alias: str = "", params: object = None) -> DuckDBPyRelation: ...
    def get_table_names(self, query: str, *, qualified: bool = False) -> set[str]: ...
    def install_extension(
        self,
        extension: str,
        *,
        force_install: bool = False,
        repository: str | None = None,
        repository_url: str | None = None,
        version: str | None = None,
    ) -> None: ...
    def get_profiling_information(self, format: ProfilerFormat = "json") -> str: ...
    def enable_profiling(self) -> None: ...
    def disable_profiling(self) -> None: ...
    def interrupt(self) -> None: ...
    def list_filesystems(self) -> lst[str]: ...
    def list_type(self, type: IntoPyType) -> sqltypes.DuckDBPyType: ...
    def load_extension(self, extension: str) -> None: ...
    def map_type(self, key: IntoPyType, value: IntoPyType) -> sqltypes.DuckDBPyType: ...
    @typing.overload
    def pl(
        self, rows_per_batch: typing.SupportsInt = 1000000, *, lazy: typing.Literal[False] = ...
    ) -> polars.DataFrame: ...
    @typing.overload
    def pl(self, rows_per_batch: typing.SupportsInt = 1000000, *, lazy: typing.Literal[True]) -> polars.LazyFrame: ...
    @typing.overload
    def pl(
        self, rows_per_batch: typing.SupportsInt = 1000000, *, lazy: bool = False
    ) -> polars.DataFrame | polars.LazyFrame: ...
    def query(self, query: str, *, alias: str = "", params: object = None) -> DuckDBPyRelation: ...
    def query_progress(self) -> float: ...
    def read_csv(
        self,
        path_or_buffer: str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str],
        header: bool | int | None = None,
        compression: CsvCompression | None = None,
        sep: str | None = None,
        delimiter: str | None = None,
        files_to_sniff: int | None = None,
        comment: str | None = None,
        thousands: str | None = None,
        dtype: IntoFields | None = None,
        na_values: str | lst[str] | None = None,
        skiprows: int | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        encoding: CsvEncoding | None = None,
        parallel: bool | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        sample_size: int | None = None,
        auto_detect: bool | int | None = None,
        all_varchar: bool | None = None,
        normalize_names: bool | None = None,
        null_padding: bool | None = None,
        names: lst[str] | None = None,
        lineterminator: CSVLineTerminator | None = None,
        columns: ColumnsTypes | None = None,
        auto_type_candidates: lst[StrIntoPyType] | None = None,
        max_line_size: int | None = None,
        ignore_errors: bool | None = None,
        store_rejects: bool | None = None,
        rejects_table: str | None = None,
        rejects_scan: str | None = None,
        rejects_limit: int | None = None,
        force_not_null: lst[str] | None = None,
        buffer_size: int | None = None,
        decimal: str | None = None,
        allow_quoted_nulls: bool | None = None,
        filename: bool | str | None = None,
        hive_partitioning: bool | None = None,
        union_by_name: bool | None = None,
        hive_types: HiveTypes | None = None,
        hive_types_autocast: bool | None = None,
        strict_mode: bool | None = None,
    ) -> DuckDBPyRelation: ...
    def read_json(
        self,
        path_or_buffer: str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str],
        *,
        columns: ColumnsTypes | None = None,
        sample_size: int | None = None,
        maximum_depth: int | None = None,
        records: JsonRecordOptions | None = None,
        format: JsonFormat | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        compression: JsonCompression | None = None,
        maximum_object_size: int | None = None,
        ignore_errors: bool | None = None,
        convert_strings_to_integers: bool | None = None,
        field_appearance_threshold: float | None = None,
        map_inference_threshold: int | None = None,
        maximum_sample_files: int | None = None,
        filename: bool | str | None = None,
        hive_partitioning: bool | None = None,
        union_by_name: bool | None = None,
        hive_types: HiveTypes | None = None,
        hive_types_autocast: bool | None = None,
    ) -> DuckDBPyRelation: ...
    def read_parquet(
        self,
        path_or_buffer: str
        | bytes
        | os.PathLike[str]
        | os.PathLike[bytes]
        | typing.IO[bytes]
        | typing.IO[str]
        | Sequence[str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str]],
        binary_as_string: bool = False,
        *,
        file_row_number: bool = False,
        filename: bool = False,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        compression: ParquetCompression | None = None,
    ) -> DuckDBPyRelation: ...
    def register(self, view_name: str, python_object: object) -> DuckDBPyConnection: ...
    def register_filesystem(self, filesystem: fsspec.AbstractFileSystem) -> None: ...
    def remove_function(self, name: str) -> DuckDBPyConnection: ...
    def rollback(self) -> DuckDBPyConnection: ...
    def row_type(self, fields: IntoFields) -> sqltypes.DuckDBPyType: ...
    def sql(self, query: Statement | str, *, alias: str = "", params: object = None) -> DuckDBPyRelation: ...
    def sqltype(self, type_str: str) -> sqltypes.DuckDBPyType: ...
    def string_type(self, collation: str = "") -> sqltypes.DuckDBPyType: ...
    def struct_type(self, fields: IntoFields) -> sqltypes.DuckDBPyType: ...
    def table(self, table_name: str) -> DuckDBPyRelation: ...
    def table_function(self, name: str, parameters: object = None) -> DuckDBPyRelation: ...
    def tf(self) -> dict[str, typing.Any]: ...
    def torch(self) -> dict[str, typing.Any]: ...
    def type(self, type_str: str) -> sqltypes.DuckDBPyType: ...
    def union_type(self, members: IntoFields) -> sqltypes.DuckDBPyType: ...
    def unregister(self, view_name: str) -> DuckDBPyConnection: ...
    def unregister_filesystem(self, name: str) -> None: ...
    def values(self, *args: IntoValues) -> DuckDBPyRelation: ...
    def view(self, view_name: str) -> DuckDBPyRelation: ...
    @property
    def description(self) -> lst[tuple[str, sqltypes.DuckDBPyType, None, None, None, None, None]]: ...
    @property
    def rowcount(self) -> int: ...

class DuckDBPyRelation:
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> typing.Any: ...
    def __contains__(self, name: str) -> bool: ...
    def __getattr__(self, name: str) -> DuckDBPyRelation: ...
    def __getitem__(self, name: str) -> DuckDBPyRelation: ...
    def __len__(self) -> int: ...
    def aggregate(self, aggr_expr: str | Iterable[IntoExpr], group_expr: IntoExpr = "") -> DuckDBPyRelation: ...
    def any_value(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def apply(
        self,
        function_name: str,
        function_aggr: str,
        group_expr: str = "",
        function_parameter: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def arg_max(
        self, arg_column: str, value_column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def arg_min(
        self, arg_column: str, value_column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def arrow(self, batch_size: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader:
        """Alias of to_arrow_reader(). We recommend using to_arrow_reader() instead."""
        ...
    def to_arrow_reader(self, batch_size: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader: ...
    def to_arrow_table(self, batch_size: typing.SupportsInt = 1000000) -> pyarrow.lib.Table: ...
    def avg(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def bit_and(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def bit_or(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def bit_xor(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def bitstring_agg(
        self,
        expression: str,
        min: int | None = None,
        max: int | None = None,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def bool_and(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def bool_or(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def close(self) -> None: ...
    def count(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def create(self, table_name: str) -> None: ...
    def create_view(self, view_name: str, replace: bool = True) -> DuckDBPyRelation: ...
    def cross(self, other_rel: Self) -> DuckDBPyRelation: ...
    def cume_dist(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    def dense_rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    def describe(self) -> DuckDBPyRelation: ...
    def df(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def distinct(self) -> DuckDBPyRelation: ...
    def except_(self, other_rel: Self) -> DuckDBPyRelation: ...
    def execute(self) -> DuckDBPyRelation: ...
    def explain(self, type: ExplainType | ExplainTypeLiteral = ExplainType.STANDARD) -> str: ...
    def favg(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def fetch_arrow_reader(self, batch_size: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader:
        """Deprecated: use to_arrow_reader() instead."""
        ...
    def fetch_arrow_table(self, batch_size: typing.SupportsInt = 1000000) -> pyarrow.lib.Table:
        """Deprecated: use to_arrow_table() instead."""
        ...
    def fetch_df_chunk(
        self, vectors_per_chunk: typing.SupportsInt = 1, *, date_as_object: bool = False
    ) -> pandas.DataFrame: ...
    def fetch_record_batch(self, rows_per_batch: typing.SupportsInt = 1000000) -> pyarrow.lib.RecordBatchReader:
        """Deprecated: use to_arrow_reader() instead."""
        ...
    def fetchall(self) -> lst[tuple[typing.Any, ...]]: ...
    def fetchdf(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def fetchmany(self, size: typing.SupportsInt = 1) -> lst[tuple[typing.Any, ...]]: ...
    def fetchnumpy(self) -> dict[str, np.typing.NDArray[typing.Any] | pandas.Categorical]: ...
    def fetchone(self) -> tuple[typing.Any, ...] | None: ...
    def filter(self, filter_expr: IntoExprColumn) -> DuckDBPyRelation: ...
    def first(self, expression: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation: ...
    def first_value(self, expression: str, window_spec: str = "", projected_columns: str = "") -> DuckDBPyRelation: ...
    def fsum(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def geomean(self, expression: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation: ...
    def histogram(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def insert(self, values: lst[object]) -> None: ...
    def insert_into(self, table_name: str) -> None: ...
    def intersect(self, other_rel: Self) -> DuckDBPyRelation: ...
    def join(self, other_rel: Self, condition: IntoExprColumn, how: JoinType = "inner") -> DuckDBPyRelation: ...
    def lag(
        self,
        expression: str,
        window_spec: str,
        offset: typing.SupportsInt = 1,
        default_value: str = "NULL",
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def last(self, expression: str, groups: str = "", projected_columns: str = "") -> DuckDBPyRelation: ...
    def last_value(self, expression: str, window_spec: str = "", projected_columns: str = "") -> DuckDBPyRelation: ...
    def lead(
        self,
        expression: str,
        window_spec: str,
        offset: typing.SupportsInt = 1,
        default_value: str = "NULL",
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def limit(self, n: typing.SupportsInt, offset: typing.SupportsInt = 0) -> DuckDBPyRelation: ...
    def list(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def map(
        self, map_function: Callable[..., typing.Any], *, schema: dict[str, sqltypes.DuckDBPyType] | None = None
    ) -> DuckDBPyRelation: ...
    def max(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def mean(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def median(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def min(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def mode(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def n_tile(
        self, window_spec: str, num_buckets: typing.SupportsInt, projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def nth_value(
        self,
        expression: str,
        window_spec: str,
        offset: typing.SupportsInt,
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def order(self, order_expr: str) -> DuckDBPyRelation: ...
    def percent_rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    @typing.overload
    def pl(
        self, batch_size: typing.SupportsInt = 1000000, *, lazy: typing.Literal[False] = ...
    ) -> polars.DataFrame: ...
    @typing.overload
    def pl(self, batch_size: typing.SupportsInt = 1000000, *, lazy: typing.Literal[True]) -> polars.LazyFrame: ...
    @typing.overload
    def pl(
        self, batch_size: typing.SupportsInt = 1000000, *, lazy: bool = False
    ) -> polars.DataFrame | polars.LazyFrame: ...
    def product(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def project(self, *args: IntoExpr, groups: str = "") -> DuckDBPyRelation: ...
    def quantile(
        self,
        expression: str,
        q: float | lst[float] = 0.5,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def quantile_cont(
        self,
        expression: str,
        q: float | lst[float] = 0.5,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def quantile_disc(
        self,
        expression: str,
        q: float | lst[float] = 0.5,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> DuckDBPyRelation: ...
    def query(self, virtual_table_name: str, sql_query: str) -> DuckDBPyRelation: ...
    def rank(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    def rank_dense(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    def row_number(self, window_spec: str, projected_columns: str = "") -> DuckDBPyRelation: ...
    def select(self, *args: IntoExpr, groups: str = "") -> DuckDBPyRelation: ...
    def select_dtypes(self, types: lst[sqltypes.DuckDBPyType | StrIntoPyType]) -> DuckDBPyRelation: ...
    def select_types(self, types: lst[sqltypes.DuckDBPyType | StrIntoPyType]) -> DuckDBPyRelation: ...
    def set_alias(self, alias: str) -> DuckDBPyRelation: ...
    def show(
        self,
        *,
        max_width: typing.SupportsInt | None = None,
        max_rows: typing.SupportsInt | None = None,
        max_col_width: typing.SupportsInt | None = None,
        null_value: str | None = None,
        render_mode: RenderMode | RenderModeLiteral | None = None,
    ) -> None: ...
    def sort(self, *args: IntoExpr) -> DuckDBPyRelation: ...
    def sql_query(self) -> str: ...
    def std(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def stddev(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def stddev_pop(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def stddev_samp(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def string_agg(
        self, expression: str, sep: str = ",", groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def sum(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def tf(self) -> dict[str, typing.Any]: ...
    def to_csv(
        self,
        file_name: str,
        *,
        sep: str | None = None,
        na_rep: str | None = None,
        header: bool | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        quoting: str | int | None = None,
        encoding: CsvEncoding | None = None,
        compression: CsvCompression | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: lst[str] | None = None,
        write_partition_columns: bool | None = None,
    ) -> None: ...
    def to_df(self, *, date_as_object: bool = False) -> pandas.DataFrame: ...
    def to_parquet(
        self,
        file_name: str,
        *,
        compression: ParquetCompression | None = None,
        field_ids: ParquetFieldsOptions | None = None,
        row_group_size_bytes: int | str | None = None,
        row_group_size: int | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: lst[str] | None = None,
        write_partition_columns: bool | None = None,
        append: bool | None = None,
        filename_pattern: str | None = None,
        file_size_bytes: str | int | None = None,
    ) -> None: ...
    def to_table(self, table_name: str) -> None: ...
    def to_view(self, view_name: str, replace: bool = True) -> DuckDBPyRelation: ...
    def torch(self) -> dict[str, typing.Any]: ...
    def union(self, union_rel: Self) -> DuckDBPyRelation: ...
    def unique(self, unique_aggr: str) -> DuckDBPyRelation: ...
    def update(self, set: Mapping[str, IntoExpr], *, condition: IntoExpr = None) -> None: ...
    def value_counts(self, expression: str, groups: str = "") -> DuckDBPyRelation: ...
    def var(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def var_pop(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def var_samp(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def variance(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> DuckDBPyRelation: ...
    def write_csv(
        self,
        file_name: str,
        *,
        sep: str | None = None,
        na_rep: str | None = None,
        header: bool | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        quoting: str | int | None = None,
        encoding: CsvEncoding | None = None,
        compression: CsvCompression | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: lst[str] | None = None,
        write_partition_columns: bool | None = None,
    ) -> None: ...
    def write_parquet(
        self,
        file_name: str,
        *,
        compression: ParquetCompression | None = None,
        field_ids: ParquetFieldsOptions | None = None,
        row_group_size_bytes: str | int | None = None,
        row_group_size: int | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: lst[str] | None = None,
        write_partition_columns: bool | None = None,
        append: bool | None = None,
        filename_pattern: str | None = None,
        file_size_bytes: str | int | None = None,
    ) -> None: ...
    @property
    def alias(self) -> str: ...
    @property
    def columns(self) -> lst[str]: ...
    @property
    def description(self) -> lst[tuple[str, sqltypes.DuckDBPyType, None, None, None, None, None]]: ...
    @property
    def dtypes(self) -> lst[sqltypes.DuckDBPyType]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def type(self) -> str: ...
    @property
    def types(self) -> lst[sqltypes.DuckDBPyType]: ...

class Error(Exception): ...
class FatalException(DatabaseError): ...

class HTTPException(IOException):
    status_code: int
    body: str
    reason: str
    headers: dict[str, str]

class IOException(OperationalError): ...
class IntegrityError(DatabaseError): ...
class InternalError(DatabaseError): ...
class InternalException(InternalError): ...
class InterruptException(DatabaseError): ...
class InvalidInputException(ProgrammingError): ...
class InvalidTypeException(ProgrammingError): ...
class NotImplementedException(NotSupportedError): ...
class NotSupportedError(DatabaseError): ...
class OperationalError(DatabaseError): ...
class OutOfMemoryException(OperationalError): ...
class OutOfRangeException(DataError): ...
class ParserException(ProgrammingError): ...
class PermissionException(DatabaseError): ...
class ProgrammingError(DatabaseError): ...
class SequenceException(DatabaseError): ...
class SerializationException(OperationalError): ...

class Statement:
    @property
    def expected_result_type(self) -> lst[StatementType]: ...
    @property
    def named_parameters(self) -> set[str]: ...
    @property
    def query(self) -> str: ...
    @property
    def type(self) -> StatementType: ...

class SyntaxException(ProgrammingError): ...
class TransactionException(OperationalError): ...
class TypeMismatchException(DataError): ...
class Warning(Exception): ...

def CaseExpression(condition: IntoExpr, value: IntoExpr) -> Expression: ...
def CoalesceOperator(*args: IntoExpr) -> Expression: ...
def ColumnExpression(*args: str) -> Expression: ...
def ConstantExpression(value: PythonLiteral) -> Expression: ...
def DefaultExpression() -> Expression: ...
def FunctionExpression(function_name: str, *args: IntoExpr) -> Expression: ...
def LambdaExpression(lhs: IntoExprColumn | tuple[IntoExprColumn, ...], rhs: IntoExpr) -> Expression: ...
def SQLExpression(expression: str) -> Expression: ...
def StarExpression(*, exclude: Iterable[IntoExprColumn] | None = None) -> Expression: ...
def aggregate(
    df: pandas.DataFrame,
    aggr_expr: str | Iterable[IntoExpr],
    group_expr: str = "",
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def alias(df: pandas.DataFrame, alias: str, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def append(
    table_name: str, df: pandas.DataFrame, *, by_name: bool = False, connection: DuckDBPyConnection | None = None
) -> DuckDBPyConnection: ...
def array_type(
    type: IntoPyType, size: typing.SupportsInt, *, connection: DuckDBPyConnection | None = None
) -> sqltypes.DuckDBPyType: ...
@typing.overload
def arrow(
    rows_per_batch: typing.SupportsInt = 1000000, *, connection: DuckDBPyConnection | None = None
) -> pyarrow.lib.RecordBatchReader:
    """Alias of to_arrow_reader(). We recommend using to_arrow_reader() instead."""
    ...

@typing.overload
def arrow(arrow_object: typing.Any, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def to_arrow_reader(
    batch_size: typing.SupportsInt = 1000000, *, connection: DuckDBPyConnection | None = None
) -> pyarrow.lib.RecordBatchReader: ...
def to_arrow_table(
    batch_size: typing.SupportsInt = 1000000, *, connection: DuckDBPyConnection | None = None
) -> pyarrow.lib.Table: ...
def begin(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def checkpoint(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def close(*, connection: DuckDBPyConnection | None = None) -> None: ...
def commit(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def connect(
    database: str | pathlib.Path = ":memory:",
    read_only: bool = False,
    config: dict[str, str | bool | int | float | lst[str]] | None = None,
) -> DuckDBPyConnection: ...
@typing.overload
def create_function(
    name: str,
    function: Callable[..., PythonLiteral],
    parameters: lst[IntoPyType] | None = None,
    return_type: IntoPyType | None = None,
    *,
    type: func.PythonUDFType = func.PythonUDFType.NATIVE,
    null_handling: func.FunctionNullHandling = ...,
    exception_handling: PythonExceptionHandling = ...,
    side_effects: bool = False,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyConnection: ...
@typing.overload
def create_function(
    name: str,
    function: ArrowUDF,
    parameters: lst[IntoPyType] | None = None,
    return_type: IntoPyType | None = None,
    *,
    type: func.PythonUDFType = func.PythonUDFType.ARROW,
    null_handling: func.FunctionNullHandling = ...,
    exception_handling: PythonExceptionHandling = ...,
    side_effects: bool = False,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyConnection: ...
def cursor(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def decimal_type(
    width: typing.SupportsInt, scale: typing.SupportsInt, *, connection: DuckDBPyConnection | None = None
) -> sqltypes.DuckDBPyType: ...
def default_connection() -> DuckDBPyConnection: ...
def description(
    *, connection: DuckDBPyConnection | None = None
) -> lst[tuple[str, sqltypes.DuckDBPyType, None, None, None, None, None]] | None: ...
@typing.overload
def df(*, date_as_object: bool = False, connection: DuckDBPyConnection | None = None) -> pandas.DataFrame: ...
@typing.overload
def df(df: pandas.DataFrame, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def distinct(df: pandas.DataFrame, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def dtype(type_str: StrIntoPyType, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def duplicate(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def enum_type(
    name: str,
    type: sqltypes.DuckDBPyType,
    values: lst[typing.Any],
    *,
    connection: DuckDBPyConnection | None = None,
) -> sqltypes.DuckDBPyType: ...
def execute(
    query: Statement | str,
    parameters: object = None,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyConnection: ...
def executemany(
    query: Statement | str,
    parameters: object = None,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyConnection: ...
def extract_statements(query: str, *, connection: DuckDBPyConnection | None = None) -> lst[Statement]: ...
def fetch_arrow_table(
    rows_per_batch: typing.SupportsInt = 1000000, *, connection: DuckDBPyConnection | None = None
) -> pyarrow.lib.Table:
    """Deprecated: use to_arrow_table() instead."""
    ...

def fetch_df(*, date_as_object: bool = False, connection: DuckDBPyConnection | None = None) -> pandas.DataFrame: ...
def fetch_df_chunk(
    vectors_per_chunk: typing.SupportsInt = 1,
    *,
    date_as_object: bool = False,
    connection: DuckDBPyConnection | None = None,
) -> pandas.DataFrame: ...
def fetch_record_batch(
    rows_per_batch: typing.SupportsInt = 1000000, *, connection: DuckDBPyConnection | None = None
) -> pyarrow.lib.RecordBatchReader:
    """Deprecated: use to_arrow_reader() instead."""
    ...

def fetchall(*, connection: DuckDBPyConnection | None = None) -> lst[tuple[typing.Any, ...]]: ...
def fetchdf(*, date_as_object: bool = False, connection: DuckDBPyConnection | None = None) -> pandas.DataFrame: ...
def fetchmany(
    size: typing.SupportsInt = 1, *, connection: DuckDBPyConnection | None = None
) -> lst[tuple[typing.Any, ...]]: ...
def fetchnumpy(
    *, connection: DuckDBPyConnection | None = None
) -> dict[str, np.typing.NDArray[typing.Any] | pandas.Categorical]: ...
def fetchone(*, connection: DuckDBPyConnection | None = None) -> tuple[typing.Any, ...] | None: ...
def filesystem_is_registered(name: str, *, connection: DuckDBPyConnection | None = None) -> bool: ...
def filter(
    df: pandas.DataFrame,
    filter_expr: IntoExprColumn,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def from_arrow(
    arrow_object: object,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def from_csv_auto(
    path_or_buffer: str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str],
    header: bool | int | None = None,
    compression: CsvCompression | None = None,
    sep: str | None = None,
    delimiter: str | None = None,
    files_to_sniff: int | None = None,
    comment: str | None = None,
    thousands: str | None = None,
    dtype: IntoFields | None = None,
    na_values: str | lst[str] | None = None,
    skiprows: int | None = None,
    quotechar: str | None = None,
    escapechar: str | None = None,
    encoding: CsvEncoding | None = None,
    parallel: bool | None = None,
    date_format: str | None = None,
    timestamp_format: str | None = None,
    sample_size: int | None = None,
    auto_detect: bool | int | None = None,
    all_varchar: bool | None = None,
    normalize_names: bool | None = None,
    null_padding: bool | None = None,
    names: lst[str] | None = None,
    lineterminator: CSVLineTerminator | None = None,
    columns: ColumnsTypes | None = None,
    auto_type_candidates: lst[StrIntoPyType] | None = None,
    max_line_size: int | None = None,
    ignore_errors: bool | None = None,
    store_rejects: bool | None = None,
    rejects_table: str | None = None,
    rejects_scan: str | None = None,
    rejects_limit: int | None = None,
    force_not_null: lst[str] | None = None,
    buffer_size: int | None = None,
    decimal: str | None = None,
    allow_quoted_nulls: bool | None = None,
    filename: bool | str | None = None,
    hive_partitioning: bool | None = None,
    union_by_name: bool | None = None,
    hive_types: HiveTypes | None = None,
    hive_types_autocast: bool | None = None,
    strict_mode: bool | None = None,
) -> DuckDBPyRelation: ...
def from_df(df: pandas.DataFrame, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def from_parquet(
    path_or_buffer: str
    | bytes
    | os.PathLike[str]
    | os.PathLike[bytes]
    | typing.IO[bytes]
    | typing.IO[str]
    | Sequence[str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str]],
    binary_as_string: bool = False,
    *,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: ParquetCompression | None = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def from_query(
    query: Statement | str,
    *,
    alias: str = "",
    params: object = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def get_table_names(
    query: str, *, qualified: bool = False, connection: DuckDBPyConnection | None = None
) -> set[str]: ...
def install_extension(
    extension: str,
    *,
    force_install: bool = False,
    repository: str | None = None,
    repository_url: str | None = None,
    version: str | None = None,
    connection: DuckDBPyConnection | None = None,
) -> None: ...
def interrupt(*, connection: DuckDBPyConnection | None = None) -> None: ...
def limit(
    df: pandas.DataFrame,
    n: typing.SupportsInt,
    offset: typing.SupportsInt = 0,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def get_profiling_information(
    *, connection: DuckDBPyConnection | None = None, format: ProfilerFormat = "json"
) -> str: ...
def enable_profiling(*, connection: DuckDBPyConnection | None = None) -> None: ...
def disable_profiling(*, connection: DuckDBPyConnection | None = None) -> None: ...
def list_filesystems(*, connection: DuckDBPyConnection | None = None) -> lst[str]: ...
def list_type(type: IntoPyType, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def load_extension(extension: str, *, connection: DuckDBPyConnection | None = None) -> None: ...
def map_type(
    key: IntoPyType, value: IntoPyType, *, connection: DuckDBPyConnection | None = None
) -> sqltypes.DuckDBPyType: ...
def order(
    df: pandas.DataFrame, order_expr: str, *, connection: DuckDBPyConnection | None = None
) -> DuckDBPyRelation: ...
@typing.overload
def pl(
    rows_per_batch: typing.SupportsInt = 1000000,
    *,
    lazy: typing.Literal[False] = ...,
    connection: DuckDBPyConnection | None = None,
) -> polars.DataFrame: ...
@typing.overload
def pl(
    rows_per_batch: typing.SupportsInt = 1000000,
    *,
    lazy: typing.Literal[True],
    connection: DuckDBPyConnection | None = None,
) -> polars.LazyFrame: ...
@typing.overload
def pl(
    rows_per_batch: typing.SupportsInt = 1000000,
    *,
    lazy: bool = False,
    connection: DuckDBPyConnection | None = None,
) -> polars.DataFrame | polars.LazyFrame: ...
def project(
    df: pandas.DataFrame, *args: IntoExpr, groups: str = "", connection: DuckDBPyConnection | None = None
) -> DuckDBPyRelation: ...
def query(
    query: Statement | str,
    *,
    alias: str = "",
    params: object = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def query_df(
    df: pandas.DataFrame,
    virtual_table_name: str,
    sql_query: str,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def query_progress(*, connection: DuckDBPyConnection | None = None) -> float: ...
def read_csv(
    path_or_buffer: str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str],
    header: bool | int | None = None,
    compression: CsvCompression | None = None,
    sep: str | None = None,
    delimiter: str | None = None,
    files_to_sniff: int | None = None,
    comment: str | None = None,
    thousands: str | None = None,
    dtype: IntoFields | None = None,
    na_values: str | lst[str] | None = None,
    skiprows: int | None = None,
    quotechar: str | None = None,
    escapechar: str | None = None,
    encoding: CsvEncoding | None = None,
    parallel: bool | None = None,
    date_format: str | None = None,
    timestamp_format: str | None = None,
    sample_size: int | None = None,
    auto_detect: bool | int | None = None,
    all_varchar: bool | None = None,
    normalize_names: bool | None = None,
    null_padding: bool | None = None,
    names: lst[str] | None = None,
    lineterminator: CSVLineTerminator | None = None,
    columns: ColumnsTypes | None = None,
    auto_type_candidates: lst[StrIntoPyType] | None = None,
    max_line_size: int | None = None,
    ignore_errors: bool | None = None,
    store_rejects: bool | None = None,
    rejects_table: str | None = None,
    rejects_scan: str | None = None,
    rejects_limit: int | None = None,
    force_not_null: lst[str] | None = None,
    buffer_size: int | None = None,
    decimal: str | None = None,
    allow_quoted_nulls: bool | None = None,
    filename: bool | str | None = None,
    hive_partitioning: bool | None = None,
    union_by_name: bool | None = None,
    hive_types: HiveTypes | None = None,
    hive_types_autocast: bool | None = None,
    strict_mode: bool | None = None,
) -> DuckDBPyRelation: ...
def read_json(
    path_or_buffer: str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str],
    *,
    columns: ColumnsTypes | None = None,
    sample_size: int | None = None,
    maximum_depth: int | None = None,
    records: JsonRecordOptions | None = None,
    format: JsonFormat | None = None,
    date_format: str | None = None,
    timestamp_format: str | None = None,
    compression: JsonCompression | None = None,
    maximum_object_size: int | None = None,
    ignore_errors: bool | None = None,
    convert_strings_to_integers: bool | None = None,
    field_appearance_threshold: float | None = None,
    map_inference_threshold: int | None = None,
    maximum_sample_files: int | None = None,
    filename: bool | str | None = None,
    hive_partitioning: bool | None = None,
    union_by_name: bool | None = None,
    hive_types: HiveTypes | None = None,
    hive_types_autocast: bool | None = None,
) -> DuckDBPyRelation: ...
def read_parquet(
    path_or_buffer: str
    | bytes
    | os.PathLike[str]
    | os.PathLike[bytes]
    | typing.IO[bytes]
    | typing.IO[str]
    | Sequence[str | bytes | os.PathLike[str] | os.PathLike[bytes] | typing.IO[bytes] | typing.IO[str]],
    binary_as_string: bool = False,
    *,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: ParquetCompression | None = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def register(
    view_name: str,
    python_object: object,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyConnection: ...
def register_filesystem(
    filesystem: fsspec.AbstractFileSystem, *, connection: DuckDBPyConnection | None = None
) -> None: ...
def remove_function(name: str, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def rollback(*, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def row_type(fields: IntoFields, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def rowcount(*, connection: DuckDBPyConnection | None = None) -> int: ...
def set_default_connection(connection: DuckDBPyConnection) -> None: ...
def sql(
    query: Statement | str,
    *,
    alias: str = "",
    params: object = None,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def sqltype(type_str: str, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def string_type(collation: str = "", *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def struct_type(fields: IntoFields, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def table(table_name: str, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def table_function(
    name: str,
    parameters: object = None,
    *,
    connection: DuckDBPyConnection | None = None,
) -> DuckDBPyRelation: ...
def tf(*, connection: DuckDBPyConnection | None = None) -> dict[str, typing.Any]: ...
def tokenize(query: str) -> lst[tuple[int, token_type]]: ...
def torch(*, connection: DuckDBPyConnection | None = None) -> dict[str, typing.Any]: ...
def type(type_str: str, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def union_type(members: IntoFields, *, connection: DuckDBPyConnection | None = None) -> sqltypes.DuckDBPyType: ...
def unregister(view_name: str, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyConnection: ...
def unregister_filesystem(name: str, *, connection: DuckDBPyConnection | None = None) -> None: ...
def values(*args: IntoValues, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def view(view_name: str, *, connection: DuckDBPyConnection | None = None) -> DuckDBPyRelation: ...
def write_csv(
    df: pandas.DataFrame,
    filename: str,
    *,
    sep: str | None = None,
    na_rep: str | None = None,
    header: bool | None = None,
    quotechar: str | None = None,
    escapechar: str | None = None,
    date_format: str | None = None,
    timestamp_format: str | None = None,
    quoting: str | int | None = None,
    encoding: CsvEncoding | None = None,
    compression: CsvCompression | None = None,
    overwrite: bool | None = None,
    per_thread_output: bool | None = None,
    use_tmp_file: bool | None = None,
    partition_by: lst[str] | None = None,
    write_partition_columns: bool | None = None,
) -> None: ...

__formatted_python_version__: str
__git_revision__: str
__interactive__: bool
__jupyter__: bool
__standard_vector_size__: int
__version__: str
_clean_default_connection: typing.Any  # value = <capsule object>
apilevel: str
paramstyle: str
threadsafety: int
