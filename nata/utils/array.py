# -*- coding: utf-8 -*-
from typing import Tuple
from typing import Union

import numpy as np


def expand_ellipsis(key, dimensions: int) -> Tuple[Union[int, slice], ...]:
    _None = object()

    key = np.index_exp[key]

    if all(k is not Ellipsis for k in key):
        return key

    first_key = key[0] if key[0] is not Ellipsis else _None
    last_key = key[-1] if key[-1] is not Ellipsis else _None

    unexpended_dimensions = dimensions
    unexpended_dimensions -= 1 if first_key is not _None else 0
    unexpended_dimensions -= 1 if last_key is not _None else 0

    expanded_key = [slice(None) for _ in range(unexpended_dimensions)]

    if first_key is not _None:
        expanded_key.insert(0, first_key)

    if last_key is not _None:
        expanded_key.append(last_key)

    return tuple(expanded_key)
