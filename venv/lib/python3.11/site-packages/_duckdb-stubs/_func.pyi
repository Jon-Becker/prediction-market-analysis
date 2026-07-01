from typing import ClassVar
from ._enums import CppEnum

__all__: list[str] = ["ARROW", "DEFAULT", "NATIVE", "SPECIAL", "FunctionNullHandling", "PythonUDFType"]

class FunctionNullHandling(CppEnum):
    DEFAULT: ClassVar[FunctionNullHandling]  # value = <FunctionNullHandling.DEFAULT: 0>
    SPECIAL: ClassVar[FunctionNullHandling]  # value = <FunctionNullHandling.SPECIAL: 1>
    __members__: ClassVar[
        dict[str, FunctionNullHandling]
    ]  # value = {'DEFAULT': <FunctionNullHandling.DEFAULT: 0>, 'SPECIAL': <FunctionNullHandling.SPECIAL: 1>}

class PythonUDFType(CppEnum):
    ARROW: ClassVar[PythonUDFType]  # value = <PythonUDFType.ARROW: 1>
    NATIVE: ClassVar[PythonUDFType]  # value = <PythonUDFType.NATIVE: 0>
    __members__: ClassVar[
        dict[str, PythonUDFType]
    ]  # value = {'NATIVE': <PythonUDFType.NATIVE: 0>, 'ARROW': <PythonUDFType.ARROW: 1>}

ARROW: PythonUDFType  # value = <PythonUDFType.ARROW: 1>
DEFAULT: FunctionNullHandling  # value = <FunctionNullHandling.DEFAULT: 0>
NATIVE: PythonUDFType  # value = <PythonUDFType.NATIVE: 0>
SPECIAL: FunctionNullHandling  # value = <FunctionNullHandling.SPECIAL: 1>
