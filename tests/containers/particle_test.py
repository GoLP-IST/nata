# -*- coding: utf-8 -*-
from operator import eq
from operator import ne
from textwrap import dedent
from typing import Callable

import numpy as np
import pytest

from nata.containers.axis import Axis
from nata.containers.particle import Particle
from nata.containers.particle import ParticleArray
from nata.containers.particle import Quantity
from nata.containers.particle import QuantityArray


def test_Quantity():
    quant = Quantity.from_array(123)

    assert quant.name == "unnamed"
    assert quant.label == "unlabeled"
    assert quant.unit == ""

    assert quant.count == 1

    assert quant.time.name == "time"
    assert quant.time.label == "time"
    assert quant.time.unit == ""
    assert quant.time.to_numpy() == 0.0

    assert quant.shape == ()
    assert quant.ndim == 0
    assert quant.dtype == int

    np.testing.assert_array_equal(quant, 123)


def test_QuantityArray():
    quantArr = QuantityArray.from_array([1, 2, 3])

    assert quantArr.name == "unnamed"
    assert quantArr.label == "unlabeled"
    assert quantArr.unit == ""

    assert quantArr.count == 3

    assert quantArr.time.name == "time"
    assert quantArr.time.label == "time"
    assert quantArr.time.unit == ""
    assert quantArr.time.to_numpy() == 0.0

    assert quantArr.shape == (3,)
    assert quantArr.ndim == 1
    assert quantArr.dtype == int

    np.testing.assert_array_equal(quantArr, [1, 2, 3])


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
    assert repr(quant) == "Quantity<name=unnamed, dtype=int64, time=0.0>"


def test_Quantity_repr_html():
    grid = Quantity.from_array(np.array(0, dtype=np.int64))
    expected_markdown = """
    | **Quantity** | |
    | ---: | :--- |
    | **name**  | unnamed |
    | **label** | unlabeled |
    | **unit**  | '' |
    | **count** | 1 |
    | **dtype** | int64 |
    | **time**  | 0.0 |

    """

    assert grid._repr_markdown_() == dedent(expected_markdown)


def test_Particle():
    prt = Particle.from_array(0.0)
    expected_dtype = np.dtype([("quant1", float)])

    assert prt.name == "unnamed"
    assert prt.label == "unlabeled"

    assert prt.time.name == "time"
    assert prt.time.label == "time"
    assert prt.time.unit == ""
    assert prt.time.to_numpy() == 0.0

    assert prt.count == 1
    assert prt.quantities == (("quant1", "quant_1 label", ""),)
    assert prt.quantity_names == ("quant1",)
    assert prt.quantity_labels == ("quant_1 label",)
    assert prt.quantity_units == ("",)

    assert prt.ndim == 0
    assert prt.shape == ()
    assert prt.dtype == expected_dtype

    np.testing.assert_array_equal(prt, np.array(0.0, expected_dtype))


@pytest.mark.parametrize(
    "arr, kwargs, quantities, expected_arr",
    [
        (
            1.0,
            {},
            (("quant1", "quant_1 label", ""),),
            np.array(1.0, dtype=np.dtype([("quant1", float)])),
        ),
        (
            np.array(1.0, dtype=np.dtype([("x", float)])),
            {},
            (("x", "x", ""),),
            np.array(1.0, dtype=np.dtype([("x", float)])),
        ),
        (
            np.array([1, 2, 3, 4]),
            {
                "quantities": tuple(
                    (f"q{i}", f"q_{i} label", f"q{i} unit") for i in range(4)
                ),
            },
            tuple((f"q{i}", f"q_{i} label", f"q{i} unit") for i in range(4)),
            np.array((1, 2, 3, 4), dtype=np.dtype([(f"q{i}", int) for i in range(4)])),
        ),
    ],
    ids=["number -> 1 quantity", "structured array", "array -> n quantities"],
)
def test_Particle_from_array_various_inputs(arr, kwargs, quantities, expected_arr):
    prt = Particle.from_array(arr, **kwargs)

    assert prt.count == 1
    assert prt.ndim == 0
    assert prt.shape == ()

    assert prt.quantities == quantities

    np.testing.assert_array_equal(prt, expected_arr)


def test_ParticleArray():
    prt = ParticleArray.from_array([0.0, 1.0, 2.0])
    expected_dtype = np.dtype([("quant1", float)])

    assert prt.name == "unnamed"
    assert prt.label == "unlabeled"

    assert prt.time.name == "time"
    assert prt.time.label == "time"
    assert prt.time.unit == ""
    assert prt.time.to_numpy() == 0.0

    assert prt.count == 3
    assert prt.quantities == (("quant1", "quant_1 label", ""),)
    assert prt.quantity_names == ("quant1",)
    assert prt.quantity_labels == ("quant_1 label",)
    assert prt.quantity_units == ("",)

    assert prt.ndim == 1
    assert prt.shape == (3,)
    assert prt.dtype == expected_dtype

    np.testing.assert_array_equal(prt, np.array([0.0, 1.0, 2.0], expected_dtype))
