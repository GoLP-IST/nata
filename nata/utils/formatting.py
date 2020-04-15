# -*- coding: utf-8 -*-
import re

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


def make_as_identifier(s: str):
    # Remove leading characters until we find a letter or underscore
    s = re.sub("^[^a-zA-Z]+", "", s)

    # replace spaces by underscore
    s = s.replace(" ", "_")

    # Remove invalid characters
    s = re.sub("[^0-9a-zA-Z_]", "", s)

    return s


def make_identifiable(s: str) -> str:
    # Remove leading characters until we find a letter or underscore
    s = re.sub("^[^a-zA-Z]+", "", s)

    # replace spaces by underscore
    s = s.replace(" ", "_")

    # Remove invalid characters
    s = re.sub("[^0-9a-zA-Z_]", "", s)

    return s
