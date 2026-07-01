from typing import TYPE_CHECKING, cast

from ..errors import PySparkNotImplementedError, PySparkTypeError
from ..exception import ContributionsAcceptedError
from .types import StructType

PrimitiveType = bool | float | int | str
OptionalPrimitiveType = PrimitiveType | None

if TYPE_CHECKING:
    from duckdb.experimental.spark.sql.dataframe import DataFrame
    from duckdb.experimental.spark.sql.session import SparkSession


class DataFrameWriter:  # noqa: D101
    def __init__(self, dataframe: "DataFrame") -> None:  # noqa: D107
        self.dataframe = dataframe

    def saveAsTable(self, table_name: str) -> None:  # noqa: D102
        relation = self.dataframe.relation
        relation.create(table_name)

    def parquet(  # noqa: D102
        self,
        path: str,
        mode: str | None = None,
        partitionBy: str | list[str] | None = None,
        compression: str | None = None,
    ) -> None:
        relation = self.dataframe.relation
        if mode:
            raise NotImplementedError
        if partitionBy:
            raise NotImplementedError

        relation.write_parquet(path, compression=compression)

    def csv(  # noqa: D102
        self,
        path: str,
        mode: str | None = None,
        compression: str | None = None,
        sep: str | None = None,
        quote: str | None = None,
        escape: str | None = None,
        header: bool | str | None = None,
        nullValue: str | None = None,
        escapeQuotes: bool | str | None = None,
        quoteAll: bool | str | None = None,
        dateFormat: str | None = None,
        timestampFormat: str | None = None,
        ignoreLeadingWhiteSpace: bool | str | None = None,
        ignoreTrailingWhiteSpace: bool | str | None = None,
        charToEscapeQuoteEscaping: str | None = None,
        encoding: str | None = None,
        emptyValue: str | None = None,
        lineSep: str | None = None,
    ) -> None:
        if mode not in (None, "overwrite"):
            raise NotImplementedError
        if escapeQuotes:
            raise NotImplementedError
        if ignoreLeadingWhiteSpace:
            raise NotImplementedError
        if ignoreTrailingWhiteSpace:
            raise NotImplementedError
        if charToEscapeQuoteEscaping:
            raise NotImplementedError
        if emptyValue:
            raise NotImplementedError
        if lineSep:
            raise NotImplementedError
        relation = self.dataframe.relation
        relation.write_csv(
            path,
            sep=sep,
            na_rep=nullValue,
            quotechar=quote,
            compression=compression,
            escapechar=escape,
            header=header if isinstance(header, bool) else header == "True",
            encoding=encoding,
            quoting=quoteAll,
            date_format=dateFormat,
            timestamp_format=timestampFormat,
        )


class DataFrameReader:  # noqa: D101
    def __init__(self, session: "SparkSession") -> None:  # noqa: D107
        self.session = session

    def load(  # noqa: D102
        self,
        path: str | list[str] | None = None,
        format: str | None = None,
        schema: StructType | str | None = None,
        **options: OptionalPrimitiveType,
    ) -> "DataFrame":
        from duckdb.experimental.spark.sql.dataframe import DataFrame

        if not isinstance(path, str):
            raise TypeError
        if options:
            raise ContributionsAcceptedError

        rel = None
        if format:
            format = format.lower()
            if format == "csv" or format == "tsv":
                rel = self.session.conn.read_csv(path)
            elif format == "json":
                rel = self.session.conn.read_json(path)
            elif format == "parquet":
                rel = self.session.conn.read_parquet(path)
            else:
                raise ContributionsAcceptedError
        else:
            rel = self.session.conn.sql(f"select * from {path}")
        df = DataFrame(rel, self.session)
        if schema:
            if not isinstance(schema, StructType):
                raise ContributionsAcceptedError
            schema = cast("StructType", schema)
            types, names = schema.extract_types_and_names()
            df = df._cast_types(types)
            df = df.toDF(names)
        return df

    def csv(  # noqa: D102
        self,
        path: str | list[str],
        schema: StructType | str | None = None,
        sep: str | None = None,
        encoding: str | None = None,
        quote: str | None = None,
        escape: str | None = None,
        comment: str | None = None,
        header: bool | str | None = None,
        inferSchema: bool | str | None = None,
        ignoreLeadingWhiteSpace: bool | str | None = None,
        ignoreTrailingWhiteSpace: bool | str | None = None,
        nullValue: str | None = None,
        nanValue: str | None = None,
        positiveInf: str | None = None,
        negativeInf: str | None = None,
        dateFormat: str | None = None,
        timestampFormat: str | None = None,
        maxColumns: int | str | None = None,
        maxCharsPerColumn: int | str | None = None,
        maxMalformedLogPerPartition: int | str | None = None,
        mode: str | None = None,
        columnNameOfCorruptRecord: str | None = None,
        multiLine: bool | str | None = None,
        charToEscapeQuoteEscaping: str | None = None,
        samplingRatio: float | str | None = None,
        enforceSchema: bool | str | None = None,
        emptyValue: str | None = None,
        locale: str | None = None,
        lineSep: str | None = None,
        pathGlobFilter: bool | str | None = None,
        recursiveFileLookup: bool | str | None = None,
        modifiedBefore: bool | str | None = None,
        modifiedAfter: bool | str | None = None,
        unescapedQuoteHandling: str | None = None,
    ) -> "DataFrame":
        if not isinstance(path, str):
            raise NotImplementedError
        if schema and not isinstance(schema, StructType):
            raise ContributionsAcceptedError
        if comment:
            raise ContributionsAcceptedError
        if inferSchema:
            raise ContributionsAcceptedError
        if ignoreLeadingWhiteSpace:
            raise ContributionsAcceptedError
        if ignoreTrailingWhiteSpace:
            raise ContributionsAcceptedError
        if nanValue:
            raise ConnectionAbortedError
        if positiveInf:
            raise ConnectionAbortedError
        if negativeInf:
            raise ConnectionAbortedError
        if negativeInf:
            raise ConnectionAbortedError
        if maxColumns:
            raise ContributionsAcceptedError
        if maxCharsPerColumn:
            raise ContributionsAcceptedError
        if maxMalformedLogPerPartition:
            raise ContributionsAcceptedError
        if mode:
            raise ContributionsAcceptedError
        if columnNameOfCorruptRecord:
            raise ContributionsAcceptedError
        if multiLine:
            raise ContributionsAcceptedError
        if charToEscapeQuoteEscaping:
            raise ContributionsAcceptedError
        if samplingRatio:
            raise ContributionsAcceptedError
        if enforceSchema:
            raise ContributionsAcceptedError
        if emptyValue:
            raise ContributionsAcceptedError
        if locale:
            raise ContributionsAcceptedError
        if pathGlobFilter:
            raise ContributionsAcceptedError
        if recursiveFileLookup:
            raise ContributionsAcceptedError
        if modifiedBefore:
            raise ContributionsAcceptedError
        if modifiedAfter:
            raise ContributionsAcceptedError
        if unescapedQuoteHandling:
            raise ContributionsAcceptedError
        if lineSep:
            # We have support for custom newline, just needs to be ported to 'read_csv'
            raise NotImplementedError

        dtype = None
        names = None
        if schema:
            schema = cast("StructType", schema)
            dtype, names = schema.extract_types_and_names()

        rel = self.session.conn.read_csv(
            path,
            header=header if isinstance(header, bool) else header == "True",
            sep=sep,
            dtype=dtype,
            na_values=nullValue,
            quotechar=quote,
            escapechar=escape,
            encoding=encoding,
            date_format=dateFormat,
            timestamp_format=timestampFormat,
        )
        from ..sql.dataframe import DataFrame

        df = DataFrame(rel, self.session)
        if names:
            df = df.toDF(*names)
        return df

    def parquet(self, *paths: str, **options: "OptionalPrimitiveType") -> "DataFrame":  # noqa: D102
        input = list(paths)
        if len(input) != 1:
            msg = "Only single paths are supported for now"
            raise NotImplementedError(msg)
        option_amount = len(options.keys())
        if option_amount != 0:
            msg = "Options are not supported"
            raise ContributionsAcceptedError(msg)
        path = input[0]
        rel = self.session.conn.read_parquet(path)
        from ..sql.dataframe import DataFrame

        df = DataFrame(rel, self.session)
        return df

    def json(
        self,
        path: str | list[str],
        schema: StructType | str | None = None,
        primitivesAsString: bool | str | None = None,
        prefersDecimal: bool | str | None = None,
        allowComments: bool | str | None = None,
        allowUnquotedFieldNames: bool | str | None = None,
        allowSingleQuotes: bool | str | None = None,
        allowNumericLeadingZero: bool | str | None = None,
        allowBackslashEscapingAnyCharacter: bool | str | None = None,
        mode: str | None = None,
        columnNameOfCorruptRecord: str | None = None,
        dateFormat: str | None = None,
        timestampFormat: str | None = None,
        multiLine: bool | str | None = None,
        allowUnquotedControlChars: bool | str | None = None,
        lineSep: str | None = None,
        samplingRatio: float | str | None = None,
        dropFieldIfAllNull: bool | str | None = None,
        encoding: str | None = None,
        locale: str | None = None,
        pathGlobFilter: bool | str | None = None,
        recursiveFileLookup: bool | str | None = None,
        modifiedBefore: bool | str | None = None,
        modifiedAfter: bool | str | None = None,
        allowNonNumericNumbers: bool | str | None = None,
    ) -> "DataFrame":
        """Loads JSON files and returns the results as a :class:`DataFrame`.

        `JSON Lines <http://jsonlines.org/>`_ (newline-delimited JSON) is supported by default.
        For JSON (one record per file), set the ``multiLine`` parameter to ``true``.

        If the ``schema`` parameter is not specified, this function goes
        through the input once to determine the input schema.

        .. versionadded:: 1.4.0

        .. versionchanged:: 3.4.0
            Supports Spark Connect.

        Parameters
        ----------
        path : str, list or :class:`RDD`
            string represents path to the JSON dataset, or a list of paths,
            or RDD of Strings storing JSON objects.
        schema : :class:`pyspark.sql.types.StructType` or str, optional
            an optional :class:`pyspark.sql.types.StructType` for the input schema or
            a DDL-formatted string (For example ``col0 INT, col1 DOUBLE``).

        Other Parameters
        ----------------
        Extra options
            For the extra options, refer to
            `Data Source Option <https://spark.apache.org/docs/latest/sql-data-sources-json.html#data-source-option>`_
            for the version you use.

            .. # noqa

        Examples:
        --------
        Write a DataFrame into a JSON file and read it back.

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     # Write a DataFrame into a JSON file
        ...     spark.createDataFrame([{"age": 100, "name": "Hyukjin Kwon"}]).write.mode(
        ...         "overwrite"
        ...     ).format("json").save(d)
        ...
        ...     # Read the JSON file as a DataFrame.
        ...     spark.read.json(d).show()
        +---+------------+
        |age|        name|
        +---+------------+
        |100|Hyukjin Kwon|
        +---+------------+
        """
        if schema is not None:
            msg = "The 'schema' option is not supported"
            raise ContributionsAcceptedError(msg)
        if primitivesAsString is not None:
            msg = "The 'primitivesAsString' option is not supported"
            raise ContributionsAcceptedError(msg)
        if prefersDecimal is not None:
            msg = "The 'prefersDecimal' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowComments is not None:
            msg = "The 'allowComments' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowUnquotedFieldNames is not None:
            msg = "The 'allowUnquotedFieldNames' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowSingleQuotes is not None:
            msg = "The 'allowSingleQuotes' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowNumericLeadingZero is not None:
            msg = "The 'allowNumericLeadingZero' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowBackslashEscapingAnyCharacter is not None:
            msg = "The 'allowBackslashEscapingAnyCharacter' option is not supported"
            raise ContributionsAcceptedError(msg)
        if mode is not None:
            msg = "The 'mode' option is not supported"
            raise ContributionsAcceptedError(msg)
        if columnNameOfCorruptRecord is not None:
            msg = "The 'columnNameOfCorruptRecord' option is not supported"
            raise ContributionsAcceptedError(msg)
        if dateFormat is not None:
            msg = "The 'dateFormat' option is not supported"
            raise ContributionsAcceptedError(msg)
        if timestampFormat is not None:
            msg = "The 'timestampFormat' option is not supported"
            raise ContributionsAcceptedError(msg)
        if multiLine is not None:
            msg = "The 'multiLine' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowUnquotedControlChars is not None:
            msg = "The 'allowUnquotedControlChars' option is not supported"
            raise ContributionsAcceptedError(msg)
        if lineSep is not None:
            msg = "The 'lineSep' option is not supported"
            raise ContributionsAcceptedError(msg)
        if samplingRatio is not None:
            msg = "The 'samplingRatio' option is not supported"
            raise ContributionsAcceptedError(msg)
        if dropFieldIfAllNull is not None:
            msg = "The 'dropFieldIfAllNull' option is not supported"
            raise ContributionsAcceptedError(msg)
        if encoding is not None:
            msg = "The 'encoding' option is not supported"
            raise ContributionsAcceptedError(msg)
        if locale is not None:
            msg = "The 'locale' option is not supported"
            raise ContributionsAcceptedError(msg)
        if pathGlobFilter is not None:
            msg = "The 'pathGlobFilter' option is not supported"
            raise ContributionsAcceptedError(msg)
        if recursiveFileLookup is not None:
            msg = "The 'recursiveFileLookup' option is not supported"
            raise ContributionsAcceptedError(msg)
        if modifiedBefore is not None:
            msg = "The 'modifiedBefore' option is not supported"
            raise ContributionsAcceptedError(msg)
        if modifiedAfter is not None:
            msg = "The 'modifiedAfter' option is not supported"
            raise ContributionsAcceptedError(msg)
        if allowNonNumericNumbers is not None:
            msg = "The 'allowNonNumericNumbers' option is not supported"
            raise ContributionsAcceptedError(msg)

        if isinstance(path, str):
            path = [path]
        if isinstance(path, list):
            if len(path) == 1:
                rel = self.session.conn.read_json(path[0])
                from .dataframe import DataFrame

                df = DataFrame(rel, self.session)
                return df
            raise PySparkNotImplementedError(message="Only a single path is supported for now")
        else:
            raise PySparkTypeError(
                error_class="NOT_STR_OR_LIST_OF_RDD",
                message_parameters={
                    "arg_name": "path",
                    "arg_type": type(path).__name__,
                },
            )


__all__ = ["DataFrameReader", "DataFrameWriter"]
