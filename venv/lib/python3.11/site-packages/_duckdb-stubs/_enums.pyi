from typing import ClassVar, Protocol, SupportsInt, Literal, TypeAlias

class CppEnum(Protocol):
    """Base Enum-like Protocol class in C++ code.

    Correspond to `py::enum_` in Pybind11.

    Note:
        This is marked as a `Protocol` to specify that an `isinstance` check against this class won't work, as this is a typing-only construct.
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: SupportsInt) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CSVLineTerminator(CppEnum):
    CARRIAGE_RETURN_LINE_FEED: ClassVar[CSVLineTerminator]  # value = <CSVLineTerminator.CARRIAGE_RETURN_LINE_FEED: 1>
    LINE_FEED: ClassVar[CSVLineTerminator]  # value = <CSVLineTerminator.LINE_FEED: 0>
    __members__: ClassVar[
        dict[str, CSVLineTerminator]
    ]  # value = {'LINE_FEED': <CSVLineTerminator.LINE_FEED: 0>, 'CARRIAGE_RETURN_LINE_FEED': <CSVLineTerminator.CARRIAGE_RETURN_LINE_FEED: 1>}  # noqa: E501

class ExpectedResultType(CppEnum):
    CHANGED_ROWS: ClassVar[ExpectedResultType]  # value = <ExpectedResultType.CHANGED_ROWS: 1>
    NOTHING: ClassVar[ExpectedResultType]  # value = <ExpectedResultType.NOTHING: 2>
    QUERY_RESULT: ClassVar[ExpectedResultType]  # value = <ExpectedResultType.QUERY_RESULT: 0>
    __members__: ClassVar[
        dict[str, ExpectedResultType]
    ]  # value = {'QUERY_RESULT': <ExpectedResultType.QUERY_RESULT: 0>, 'CHANGED_ROWS': <ExpectedResultType.CHANGED_ROWS: 1>, 'NOTHING': <ExpectedResultType.NOTHING: 2>}  # noqa: E501

class ExplainType(CppEnum):
    ANALYZE: ClassVar[ExplainType]  # value = <ExplainType.ANALYZE: 1>
    STANDARD: ClassVar[ExplainType]  # value = <ExplainType.STANDARD: 0>
    __members__: ClassVar[
        dict[str, ExplainType]
    ]  # value = {'STANDARD': <ExplainType.STANDARD: 0>, 'ANALYZE': <ExplainType.ANALYZE: 1>}

ExplainTypeLiteral: TypeAlias = Literal["analyze", "standard", "ANALYZE", "STANDARD"]

class PythonExceptionHandling(CppEnum):
    DEFAULT: ClassVar[PythonExceptionHandling]  # value = <PythonExceptionHandling.DEFAULT: 0>
    RETURN_NULL: ClassVar[PythonExceptionHandling]  # value = <PythonExceptionHandling.RETURN_NULL: 1>
    __members__: ClassVar[
        dict[str, PythonExceptionHandling]
    ]  # value = {'DEFAULT': <PythonExceptionHandling.DEFAULT: 0>, 'RETURN_NULL': <PythonExceptionHandling.RETURN_NULL: 1>}  # noqa: E501

class RenderMode(CppEnum):
    COLUMNS: ClassVar[RenderMode]  # value = <RenderMode.COLUMNS: 1>
    ROWS: ClassVar[RenderMode]  # value = <RenderMode.ROWS: 0>
    __members__: ClassVar[
        dict[str, RenderMode]
    ]  # value = {'ROWS': <RenderMode.ROWS: 0>, 'COLUMNS': <RenderMode.COLUMNS: 1>}

RenderModeLiteral: TypeAlias = Literal["columns", "rows", "COLUMNS", "ROWS"]

class StatementType(CppEnum):
    ALTER: ClassVar[StatementType]  # value = <StatementType.ALTER: 8>
    ANALYZE: ClassVar[StatementType]  # value = <StatementType.ANALYZE: 11>
    ATTACH: ClassVar[StatementType]  # value = <StatementType.ATTACH: 25>
    CALL: ClassVar[StatementType]  # value = <StatementType.CALL: 19>
    COPY_DATABASE: ClassVar[StatementType]  # value = <StatementType.COPY_DATABASE: 28>
    COPY: ClassVar[StatementType]  # value = <StatementType.COPY: 10>
    CREATE_FUNC: ClassVar[StatementType]  # value = <StatementType.CREATE_FUNC: 13>
    CREATE: ClassVar[StatementType]  # value = <StatementType.CREATE: 4>
    DELETE: ClassVar[StatementType]  # value = <StatementType.DELETE: 5>
    DETACH: ClassVar[StatementType]  # value = <StatementType.DETACH: 26>
    DROP: ClassVar[StatementType]  # value = <StatementType.DROP: 15>
    EXECUTE: ClassVar[StatementType]  # value = <StatementType.EXECUTE: 7>
    EXPLAIN: ClassVar[StatementType]  # value = <StatementType.EXPLAIN: 14>
    EXPORT: ClassVar[StatementType]  # value = <StatementType.EXPORT: 16>
    EXTENSION: ClassVar[StatementType]  # value = <StatementType.EXTENSION: 23>
    INSERT: ClassVar[StatementType]  # value = <StatementType.INSERT: 2>
    INVALID: ClassVar[StatementType]  # value = <StatementType.INVALID: 0>
    LOAD: ClassVar[StatementType]  # value = <StatementType.LOAD: 21>
    LOGICAL_PLAN: ClassVar[StatementType]  # value = <StatementType.LOGICAL_PLAN: 24>
    MERGE_INTO: ClassVar[StatementType]  # value = <StatementType.MERGE_INTO: 30>
    MULTI: ClassVar[StatementType]  # value = <StatementType.MULTI: 27>
    PRAGMA: ClassVar[StatementType]  # value = <StatementType.PRAGMA: 17>
    PREPARE: ClassVar[StatementType]  # value = <StatementType.PREPARE: 6>
    RELATION: ClassVar[StatementType]  # value = <StatementType.RELATION: 22>
    SELECT: ClassVar[StatementType]  # value = <StatementType.SELECT: 1>
    SET: ClassVar[StatementType]  # value = <StatementType.SET: 20>
    TRANSACTION: ClassVar[StatementType]  # value = <StatementType.TRANSACTION: 9>
    UPDATE: ClassVar[StatementType]  # value = <StatementType.UPDATE: 3>
    VACUUM: ClassVar[StatementType]  # value = <StatementType.VACUUM: 18>
    VARIABLE_SET: ClassVar[StatementType]  # value = <StatementType.VARIABLE_SET: 12>
    __members__: ClassVar[
        dict[str, StatementType]
    ]  # value = {'INVALID': <StatementType.INVALID: 0>, 'SELECT': <StatementType.SELECT: 1>, 'INSERT': <StatementType.INSERT: 2>, 'UPDATE': <StatementType.UPDATE: 3>, 'CREATE': <StatementType.CREATE: 4>, 'DELETE': <StatementType.DELETE: 5>, 'PREPARE': <StatementType.PREPARE: 6>, 'EXECUTE': <StatementType.EXECUTE: 7>, 'ALTER': <StatementType.ALTER: 8>, 'TRANSACTION': <StatementType.TRANSACTION: 9>, 'COPY': <StatementType.COPY: 10>, 'ANALYZE': <StatementType.ANALYZE: 11>, 'VARIABLE_SET': <StatementType.VARIABLE_SET: 12>, 'CREATE_FUNC': <StatementType.CREATE_FUNC: 13>, 'EXPLAIN': <StatementType.EXPLAIN: 14>, 'DROP': <StatementType.DROP: 15>, 'EXPORT': <StatementType.EXPORT: 16>, 'PRAGMA': <StatementType.PRAGMA: 17>, 'VACUUM': <StatementType.VACUUM: 18>, 'CALL': <StatementType.CALL: 19>, 'SET': <StatementType.SET: 20>, 'LOAD': <StatementType.LOAD: 21>, 'RELATION': <StatementType.RELATION: 22>, 'EXTENSION': <StatementType.EXTENSION: 23>, 'LOGICAL_PLAN': <StatementType.LOGICAL_PLAN: 24>, 'ATTACH': <StatementType.ATTACH: 25>, 'DETACH': <StatementType.DETACH: 26>, 'MULTI': <StatementType.MULTI: 27>, 'COPY_DATABASE': <StatementType.COPY_DATABASE: 28>, 'MERGE_INTO': <StatementType.MERGE_INTO: 30>}  # noqa: E501

class token_type(CppEnum):
    __members__: ClassVar[
        dict[str, token_type]
    ]  # value = {'identifier': <token_type.identifier: 0>, 'numeric_const': <token_type.numeric_const: 1>, 'string_const': <token_type.string_const: 2>, 'operator': <token_type.operator: 3>, 'keyword': <token_type.keyword: 4>, 'comment': <token_type.comment: 5>}  # noqa: E501
    comment: ClassVar[token_type]  # value = <token_type.comment: 5>
    identifier: ClassVar[token_type]  # value = <token_type.identifier: 0>
    keyword: ClassVar[token_type]  # value = <token_type.keyword: 4>
    numeric_const: ClassVar[token_type]  # value = <token_type.numeric_const: 1>
    operator: ClassVar[token_type]  # value = <token_type.operator: 3>
    string_const: ClassVar[token_type]  # value = <token_type.string_const: 2>
