# -*- coding: utf-8 -*-
from numpy import ndarray


def array_format(data: ndarray):
    if data.ndim == 0:
        return str(data)
    elif len(data) < 4:
        return "[" + ", ".join(str(i) for i in data) + "]"
    else:
        return (
            "["
            + ", ".join(str(i) for i in data[:2])
            + ", ... "
            + str(data[-1])
            + "]"
        )
