# https://sparkbyexamples.com/pyspark/pyspark-udf-user-defined-function/
from typing import TYPE_CHECKING, Any, Optional, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

from .types import DataType

if TYPE_CHECKING:
    from .session import SparkSession

DataTypeOrString = DataType | str
UserDefinedFunctionLike = TypeVar("UserDefinedFunctionLike")


class UDFRegistration:  # noqa: D101
    def __init__(self, sparkSession: "SparkSession") -> None:  # noqa: D107
        self.sparkSession = sparkSession

    def register(  # noqa: D102
        self,
        name: str,
        f: "Callable[..., Any] | UserDefinedFunctionLike",
        returnType: Optional["DataTypeOrString"] = None,
    ) -> "UserDefinedFunctionLike":
        self.sparkSession.conn.create_function(name, f, return_type=returnType)

    def registerJavaFunction(  # noqa: D102
        self,
        name: str,
        javaClassName: str,
        returnType: Optional["DataTypeOrString"] = None,
    ) -> None:
        raise NotImplementedError

    def registerJavaUDAF(self, name: str, javaClassName: str) -> None:  # noqa: D102
        raise NotImplementedError


__all__ = ["UDFRegistration"]
