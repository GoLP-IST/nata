# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nata.utils.formatting import array_format


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
