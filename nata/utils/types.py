# -*- coding: utf-8 -*-
# import sys
from pathlib import Path
from typing import Tuple
from typing import Union

import dask.array as da
import numpy as np

# "Protocol" and "runtime_checkable" are builtin for 3.8+
# otherwise use "typing_extension" package
# if sys.version_info >= (3, 8):
#     from typing import Protocol
#     from typing import runtime_checkable
#     from typing import TypedDict
# else:
#     from typing_extensions import Protocol
#     from typing_extensions import runtime_checkable
#     from typing_extensions import TypedDict

#: Scalars and numbers
Number = Union[float, int]

#: Type which can be supplied to `numpy.array` and the resulting output is an
#: array
Array = Union[np.ndarray, da.core.Array]

#: Type for basic indexing
BasicIndex = Union[int, slice]

#: Type for basic indexing
BasicIndexing = Union[BasicIndex, Tuple[BasicIndex, ...]]

#: Type for file location
FileLocation = Union[Path, str]
