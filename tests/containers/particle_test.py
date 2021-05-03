# -*- coding: utf-8 -*-
from operator import eq
from operator import ne
from textwrap import dedent
from typing import Callable

import numpy as np
import pytest

from nata.containers.axis import Axis
from nata.containers.particle import Quantity


def test_Quantity():
    quant = Quantity.from_array(123)

    assert quant.name == "unnamed"
    assert quant.label == "unlabeled"
    assert quant.unit == ""
    assert not quant.axes
    assert hash(quant.time) == hash(
        Axis.from_array(0, name="time", label="time", unit="")
    )
    assert quant.dtype == int
    np.testing.assert_array_equal(quant, 123)


def test_Quantity_raise_non_0d():
    with pytest.raises(ValueError, match="only 0d data is supported"):
        Quantity.from_array([123])


def test_Quantity_raise_when_fields_represent():
    with pytest.raises(ValueError, match="only unstructured data types supported"):
        Quantity.from_array(np.array(0, dtype=[("f1", int)]))


@pytest.mark.skip
def test_Quantity_getitem():
    pass


@pytest.mark.parametrize(
    "left, right, operation",
    [
        # general
        (Quantity.from_array(1), Quantity.from_array(1), eq),
        # content
        (Quantity.from_array(1), Quantity.from_array(2), eq),
        # name
        (Quantity.from_array(1), Quantity.from_array(1, name="some"), ne),
        # label
        (Quantity.from_array(1), Quantity.from_array(1, label="some"), ne),
        # unit
        (Quantity.from_array(1), Quantity.from_array(1, unit="some"), ne),
        # time value
        (Quantity.from_array(1, time=0), Quantity.from_array(1, time=1), eq),
        # time name
        (
            Quantity.from_array(1, time=Axis.from_array(0)),
            Quantity.from_array(1, time=Axis.from_array(0, name="some")),
            ne,
        ),
        # time label
        (
            Quantity.from_array(1, time=Axis.from_array(0)),
            Quantity.from_array(1, time=Axis.from_array(0, label="some")),
            ne,
        ),
        # time unit
        (
            Quantity.from_array(1, time=Axis.from_array(0)),
            Quantity.from_array(1, time=Axis.from_array(0, unit="some")),
            ne,
        ),
    ],
    ids=(
        "general",
        "content",
        "name",
        "label",
        "unit",
        "time",
        "time.name",
        "time.label",
        "time.unit",
    ),
)
def test_Quantity_hash(left: Quantity, right: Quantity, operation: Callable):
    assert operation(hash(left), hash(right))


def test_Quantity_repr():
    quant = Quantity.from_array(np.array(0, dtype=np.int64))
    assert repr(quant) == "Quantity<0, dtype=int64, time=0.0>"


def test_Quantity_repr_html():
    grid = Quantity.from_array(np.array(0, dtype=np.int64))
    expected_markdown = """
    | **Quantity** | |
    | ---: | :--- |
    | **name**  | unnamed |
    | **label** | unlabeled |
    | **unit**  | '' |
    | **dtype** | int64 |
    | **time**  | 0.0 |

    """

    assert grid._repr_markdown_() == dedent(expected_markdown)
