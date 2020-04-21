# -*- coding: utf-8 -*-
from typing import Tuple
from typing import Union

import numpy as np


def expand_ellipsis(key, dimensions: int) -> Tuple[Union[int, slice], ...]:
    key = np.index_exp[key]

    if all(k is not Ellipsis for k in key):
        return key

    if key.count(Ellipsis) > 1:
        raise KeyError("Only one Ellipse '...' is allowed!")

    key = list(key)
    cleaned_key = list(
        filter(lambda k: (k is not None) and (k is not Ellipsis), key)
    )

    ellipsis_expanded = [
        slice(None) for _ in range(dimensions - len(cleaned_key))
    ]

    index_of_ellipse = key.index(Ellipsis)
    expanded_key = (
        key[:index_of_ellipse]
        + ellipsis_expanded
        + key[(index_of_ellipse + 1) :]
    )

    return tuple(expanded_key)
