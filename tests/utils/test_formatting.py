# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nata.utils.formatting import array_format
from nata.utils.formatting import make_as_identifier


@pytest.mark.parametrize(
    "input_, expected",
    [
        (np.array(1), "1"),
        (np.array(None), "None"),
        (np.array([1]), "[1]"),
        (np.array([None]), "[None]"),
        (np.array([1, 2, 3]), "[1, 2, 3]"),
        (np.array([1, 2, 3, 4]), "[1, 2, ... 4]"),
    ],
    ids=(
        "zero-dim",
        "None, zero-dim",
        "single-element",
        "None, single-element",
        "shape==(3,)",
        "shape==(4,)",
    ),
)
def test_array_format(input_, expected):
    assert array_format(input_) == expected


@pytest.mark.parametrize(
    "actual, expected",
    [
        (" abc", "abc"),
        ("_abc", "abc"),
        ("ab c", "ab_c"),
        ("ab c d", "ab_c_d"),
        ("$abc", "abc"),
    ],
)
def test_make_as_identifier(actual, expected):
    assert make_as_identifier(actual) == expected
