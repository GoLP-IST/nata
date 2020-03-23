# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis.extra.numpy import arrays
from hypothesis.extra.numpy import basic_indices
from hypothesis.extra.numpy import floating_dtypes
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import text

from nata.axes import Axis
from nata.axes import GridAxis
from nata.axes import IterationAxis
from nata.axes import ParticleQuantity
from nata.axes import TimeAxis
from nata.axes import UnnamedAxis
from nata.utils.formatting import array_format

from .strategies import anyarray
from .strategies import array_and_basic_indices
from .strategies import array_with_two_entries
from .strategies import number
from .strategies import number_or_none

_int_limit = np.iinfo(np.intc)
strategies_for_number_tests = (
    number(include_complex_numbers=False),
    number(include_complex_numbers=False),
    integers(min_value=1, max_value=1_000),
    text(),
    text(),
    text(),
)
sort_along_second_axis = partial(np.sort, axis=1)
strategies_for_array_tests = (
    array_with_two_entries(array_length=1_000).map(sort_along_second_axis),
    integers(min_value=1, max_value=1_000),
    text(),
    text(),
    text(),
)


@given(one_of(number_or_none(), anyarray(min_dims=0, max_dims=0)))
def test_UnnamedAxis_single_value_init(x):
    axis = UnnamedAxis(x)
    asarray = np.array(x)

    assert np.array_equiv(np.array(axis), asarray)
    assert np.array_equiv(axis.data, asarray)
    assert axis.shape == tuple()
    assert axis.ndim == 0
    assert axis.dtype == asarray.dtype
    # should trigger `__repr()__`
    assert str(axis) == f"UnnamedAxis({array_format(asarray)})"

    for a in axis:
        assert a is axis

    assert len(axis) == 0

    with pytest.raises(IndexError, match="too many indices"):
        axis[:]

    with pytest.raises(IndexError, match="too many indices"):
        axis[:] = 42


@given(
    array_and_basic_indices(array_min_dims=1),
    number(include_complex_numbers=False),
)
def test_UnnamedAxis_1d_init(test_case, random_value):
    arr, ind = test_case

    axis = UnnamedAxis(arr)

    assert np.array_equiv(axis, arr)
    assert np.array_equiv(axis.data, arr)
    assert axis.shape == arr.shape
    assert axis.ndim == arr.ndim
    assert axis.dtype == arr.dtype
    # should trigger `__repr()__`
    assert str(axis) == f"UnnamedAxis({array_format(arr)})"

    for actual, expected in zip(axis, arr):
        assert np.array_equiv(actual, expected)

    assert len(axis) == len(arr)
    assert np.array_equiv(axis[ind], arr[ind])
    axis[ind] = random_value
    arr[ind] = random_value
    assert np.array_equiv(axis, arr)


@given(
    one_of(number_or_none(), anyarray(min_dims=0, max_dims=0)),
    one_of(number_or_none(), anyarray(min_dims=0, max_dims=0)),
)
def test_UnnamedAxis_append_single_value_and_single_value(value1, value2):
    axis = UnnamedAxis(value1)
    axis.append(UnnamedAxis(value2))
    arr = np.array([value1, value2])

    assert np.array_equiv(axis, arr)
    assert np.array_equiv(axis.data, arr)
    assert axis.shape == arr.shape
    assert axis.ndim == arr.ndim
    assert axis.dtype == arr.dtype


@given(anyarray(min_dims=1, max_dims=1), anyarray(min_dims=1, max_dims=1))
def test_UnnamedAxis_append_1d_arrays(arr1, arr2):
    axis = UnnamedAxis(arr1)
    axis.append(UnnamedAxis(arr2))
    arr = np.array([d for d in arr1] + [d for d in arr2])

    assert np.array_equiv(axis, arr)
    assert np.array_equiv(axis.data, arr)
    assert axis.shape == arr.shape
    assert axis.ndim == arr.ndim
    assert axis.dtype == arr.dtype
    assert len(axis) == len(arr)


@given(one_of(number_or_none(), anyarray(max_dims=1)), text(), text(), text())
def test_Axis_init(data, name, label, unit):
    axis = Axis(data, name, label, unit)
    data = np.asanyarray(data)
    if data.ndim == 0:
        data = data.reshape((1,))

    assert axis.name == name
    assert axis.label == label
    assert axis.unit == unit

    for actual, expected in zip(axis, data):
        assert np.array_equiv(actual, expected)


@given(one_of(number_or_none(), anyarray(max_dims=1)))
def test_Axis_append(data):
    axis = Axis(data, "", "", "")
    axis.append(Axis(data, "", "", ""))
    np.testing.assert_array_equal(axis, np.array([data, data]))

    with pytest.raises(ValueError, match="can not append"):
        axis.append(Axis(data, "some name", "", ""))


@given(one_of(number_or_none(), anyarray(max_dims=1)), text(), text(), text())
def test_IterationAxis_init(data, name, label, unit):
    axis = IterationAxis(data, name, label, unit)

    assert axis.name == name
    assert axis.label == label
    assert axis.unit == unit


@given(one_of(number_or_none(), anyarray(max_dims=1)))
def test_IterationAxis_default(data):
    axis = IterationAxis(data)

    assert axis.name == "iteration"
    assert axis.label == "iteration"
    assert axis.unit == ""


@given(one_of(number_or_none(), anyarray(max_dims=1)), text(), text(), text())
def test_TimeAxis_init(data, name, label, unit):
    axis = TimeAxis(data, name, label, unit)

    assert axis.name == name
    assert axis.label == label
    assert axis.unit == unit


@given(one_of(number_or_none(), anyarray(max_dims=1)))
def test_TimeAxis_default(data):
    axis = TimeAxis(data)

    assert axis.name == "time"
    assert axis.label == "time"
    assert axis.unit == ""


@given(*strategies_for_number_tests)
def test_GridAxis_init_numbers(num1, num2, length, name, label, unit):
    assume(num1 <= num2)

    gridaxis = GridAxis(
        min=num1,
        max=num2,
        axis_length=length,
        name=name,
        label=label,
        unit=unit,
    )

    assert gridaxis.name == name
    assert gridaxis.label == label
    assert gridaxis.unit == unit

    assert len(gridaxis) == length
    assert gridaxis.shape == (length,)
    assert gridaxis.ndim == 1

    for a in gridaxis:
        assert a is gridaxis


@given(
    number(include_complex_numbers=False),
    number(include_complex_numbers=False),
    integers(min_value=1, max_value=1_000),
)
def test_GridAxis_init_numbers_as_list(num1, num2, length):
    assume(num1 <= num2)

    gridaxis = GridAxis(
        min=[num1],
        max=[num2],
        axis_length=length,
        name="name",
        label="label",
        unit="unit",
    )

    assert gridaxis.ndim == 1
    assert len(gridaxis) == length
    assert gridaxis.shape == (length,)

    np.testing.assert_array_equal(gridaxis, np.linspace(num1, num2, length))


@given(*strategies_for_number_tests)
def test_GridAxis_array_interface_with_singleValue_default(
    num1, num2, length, name, label, unit
):
    assume(num1 < num2)

    gridaxis = GridAxis(
        min=num1,
        max=num2,
        axis_length=length,
        name=name,
        label=label,
        unit=unit,
    )

    np.testing.assert_array_equal(gridaxis, np.linspace(num1, num2, length))


@given(*strategies_for_number_tests)
def test_GridAxis_array_interface_with_singleValue_linear(
    num1, num2, length, name, label, unit
):
    assume(num1 < num2)

    gridaxis = GridAxis(
        min=num1,
        max=num2,
        axis_length=length,
        axis_type="linear",
        name=name,
        label=label,
        unit=unit,
    )

    np.testing.assert_array_equal(gridaxis, np.linspace(num1, num2, length))


@given(*strategies_for_number_tests)
def test_GridAxis_array_interface_with_singleValue_lin(
    num1, num2, length, name, label, unit
):
    assume(num1 < num2)

    gridaxis = GridAxis(
        min=num1,
        max=num2,
        axis_length=length,
        axis_type="lin",
        name=name,
        label=label,
        unit=unit,
    )

    np.testing.assert_array_equal(gridaxis, np.linspace(num1, num2, length))


# RuntimeWarnings can appear from numpy - as using logspace
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(*strategies_for_number_tests)
def test_GridAxis_array_interface_with_singleValue_logarithmic(
    num1, num2, length, name, label, unit
):
    assume(num1 < num2)

    gridaxis = GridAxis(
        min=num1,
        max=num2,
        axis_length=length,
        axis_type="logarithmic",
        name=name,
        label=label,
        unit=unit,
    )

    np.testing.assert_array_equal(
        gridaxis, np.logspace(np.log10(num1), np.log10(num2), length)
    )


# RuntimeWarnings can appear from numpy - as using logspace
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(*strategies_for_number_tests)
def test_GridAxis_array_interface_with_singleValue_log(
    num1, num2, length, name, label, unit
):
    assume(num1 < num2)

    gridaxis = GridAxis(
        min=num1,
        max=num2,
        axis_length=length,
        axis_type="log",
        name=name,
        label=label,
        unit=unit,
    )

    np.testing.assert_array_equal(
        gridaxis, np.logspace(np.log10(num1), np.log10(num2), length)
    )


@given(*strategies_for_array_tests)
@settings(deadline=None)
def test_GridAxis_init_array(data, length, name, label, unit):
    lower = data[:, 0]
    upper = data[:, 1]

    gridaxis = GridAxis(
        min=lower,
        max=upper,
        axis_length=length,
        name=name,
        label=label,
        unit=unit,
    )

    assert gridaxis.name == name
    assert gridaxis.label == label
    assert gridaxis.unit == unit

    assert len(gridaxis) == len(data)
    if len(data) == 1:
        assert gridaxis.shape == (length,)
    else:
        assert gridaxis.shape == (len(data), length)

    for actual, expected in zip(gridaxis, data):
        assert np.array_equal(actual.data, expected)


@given(*strategies_for_array_tests)
@settings(deadline=None)
def test_GridAxis_array_interface_with_array_default(
    data, length, name, label, unit
):
    lower = data[:, 0]
    upper = data[:, 1]

    gridaxis = GridAxis(
        min=lower,
        max=upper,
        axis_length=length,
        name=name,
        label=label,
        unit=unit,
    )

    expected = []
    for l, u in zip(lower, upper):
        expected.append(np.linspace(l, u, length))
    expected = np.array(expected)
    if expected.shape[0] == 1:
        expected = np.squeeze(expected, axis=0)

    np.testing.assert_array_equal(gridaxis, expected)

    # test for individual to ensure correct passing through
    for axis, expected_axis in zip(gridaxis, expected):
        np.testing.assert_array_equal(axis, expected_axis)


@given(*strategies_for_number_tests)
def test_GridAxis_append_with_numbers(num1, num2, length, name, label, unit):
    assume(num1 < num2)
    gridaxis = GridAxis(
        min=num1,
        max=num2,
        axis_length=length,
        name=name,
        label=label,
        unit=unit,
    )
    gridaxis.append(gridaxis)

    assert gridaxis.name == name
    assert gridaxis.label == label
    assert gridaxis.unit == unit

    assert len(gridaxis) == 2
    assert gridaxis.shape == (2, length)
    assert gridaxis.ndim == 2

    expected = []
    for _ in range(2):
        expected.append(np.linspace(num1, num2, length))
    expected = np.array(expected)
    np.testing.assert_array_equal(gridaxis, expected)

    for axis, expected_axis in zip(gridaxis, expected):
        np.testing.assert_array_equal(axis, expected_axis)


@given(*strategies_for_array_tests)
@settings(deadline=None)
def test_GridAxis_append_with_arrays(data, length, name, label, unit):
    lower = data[:, 0]
    upper = data[:, 1]

    gridaxis = GridAxis(
        min=lower,
        max=upper,
        axis_length=length,
        name=name,
        label=label,
        unit=unit,
    )
    gridaxis.append(gridaxis)
    number_of_entries = data.shape[0] * 2

    assert gridaxis.name == name
    assert gridaxis.label == label
    assert gridaxis.unit == unit

    assert len(gridaxis) == number_of_entries
    assert gridaxis.shape == (number_of_entries, length)
    assert gridaxis.ndim == 2

    expected = []
    # initial grid axis
    for l, u in zip(lower, upper):
        expected.append(np.linspace(l, u, length))
    # after append
    for l, u in zip(lower, upper):
        expected.append(np.linspace(l, u, length))
    expected = np.array(expected)

    np.testing.assert_array_equal(gridaxis, expected)
    for axis, expected_axis in zip(gridaxis, expected):
        np.testing.assert_array_equal(axis, expected_axis)


@composite
def strategy_for_gridaxes_access(
    draw, max_axis_length=1_000, number_of_iterations=1_000
):
    """
    Grid axis iterated over time is a 2d array but only min/max are stored.
        * shape of the 2d matrix: (number of iterations) x (axis length)
        * if 'number of iterations == 1' then iteration axes is squeezed
    """
    axis_length = draw(integers(min_value=1, max_value=max_axis_length))
    iterations = draw(integers(min_value=1, max_value=number_of_iterations))

    arr = draw(arrays(dtype=floating_dtypes(), shape=(iterations, 2)))

    assume(not np.any(np.isnan(arr)))
    assume(np.all(np.isfinite(arr)))

    if iterations == 1:
        ind = draw(basic_indices(shape=(axis_length,)))
    else:
        ind = draw(basic_indices(shape=(iterations, axis_length)))

    return arr, ind, axis_length


@given(strategy_for_gridaxes_access())
def test_GridAxis_getitem(data):
    arr, ind, length = data

    gridaxis = GridAxis(
        min=arr[:, 0],
        max=arr[:, 1],
        axis_length=length,
        name="",
        label="",
        unit="",
    )

    full_expected_array = []
    for min_, max_ in arr:
        full_expected_array.append(np.linspace(min_, max_, length))
    full_expected_array = np.array(full_expected_array)

    np.testing.assert_array_equal(gridaxis[ind], full_expected_array[ind])


@given(
    lists(
        number(include_complex_numbers=False),
        min_size=4,
        max_size=4,
        unique=True,
    ),
    integers(min_value=1, max_value=100),
)
def test_GridAxis_append_moving_axis(l, length):
    axis_values = sorted(l)

    axis = GridAxis(
        min=axis_values[0],
        max=axis_values[2],
        axis_length=length,
        name="",
        label="",
        unit="",
    )
    axis.append(
        GridAxis(
            min=axis_values[1],
            max=axis_values[3],
            axis_length=length,
            name="",
            label="",
            unit="",
        )
    )


@given(array_and_basic_indices(array_min_dims=1, array_max_dims=2),)
def test_ParticleQuantity_init_using_arrays(data):
    arr, ind = data

    if arr.ndim == 1:
        length = [len(arr)]
    else:
        length = [arr.shape[1]] * len(arr)

    quant = ParticleQuantity(
        data=arr, dtype=arr.dtype, prt_num=length, name="", label="", unit="",
    )

    if arr.ndim == 2 and len(arr) == 1:
        expected_shape = arr.shape[1:]
        expected_ndim = arr.ndim - 1
    else:
        expected_shape = arr.shape
        expected_ndim = arr.ndim

    assert quant.shape == expected_shape
    assert quant.ndim == expected_ndim
    assert quant.dtype == arr.dtype
    assert quant.data.ndim == 2

    if arr.ndim == 2 and len(arr) == 1:
        ind = ind[-1]
        np.testing.assert_array_equal(quant, np.squeeze(arr, axis=0))
        np.testing.assert_array_equal(quant[ind], np.squeeze(arr, axis=0)[ind])
    else:
        np.testing.assert_array_equal(quant, arr)
        np.testing.assert_array_equal(quant[ind], arr[ind])


@pytest.mark.parametrize(
    ["data", "prt_num", "shape", "ndim", "expected"],
    [
        (42, 1, (1,), 1, np.array([42])),
        ([42], 1, (1,), 1, np.array([42])),
        (42, [1], (1,), 1, np.array([42])),
        ([42], [1], (1,), 1, np.array([42])),
        (np.arange(10), 10, (10,), 1, np.arange(10)),
        (
            [[i] for i in range(10)],
            [1] * 10,
            (10, 1),
            2,
            np.arange(10).reshape((10, 1)),
        ),
        (np.arange(10).reshape((1, 10)), [10], (10,), 1, np.arange(10),),
        (np.arange(10).reshape((1, 10)), 10, (10,), 1, np.arange(10),),
        (
            [np.arange(3), np.arange(2)],
            [3, 2],
            (2, 3),
            2,
            np.ma.array(
                [np.arange(3), np.arange(3)], mask=[[0, 0, 0], [0, 0, 1]]
            ),
        ),
    ],
    ids=(
        "data::shape=(), prt_num::shape=()",
        "data::shape=(1,), prt_num::shape=()",
        "data::shape=(), prt_num::shape=(1,)",
        "data::shape=(1,), prt_num::shape=(1,)",
        "data::shape=(10,), prt_num::shape=()",
        "data::shape=(10,1), prt_num::shape=(10,)",
        "data::shape=(1,10), prt_num::shape=(1,)",
        "data::shape=(1,10), prt_num::shape=()",
        "data::dtype=object, prt_num::shape=(2,)",
    ),
)
def test_ParticleQuantity_init_using_various_simple_example(
    data, prt_num, shape, ndim, expected
):
    data = np.array(data)
    prt_num = np.array(prt_num)

    quant = ParticleQuantity(
        data=data,
        dtype=int,
        prt_num=prt_num,
        name="some name",
        label="some label",
        unit="some unit",
    )

    assert quant.shape == shape
    assert quant.dtype == np.dtype(int)
    assert quant.ndim == ndim
    if quant.data.dtype == object:
        assert quant.data.ndim == 1
    else:
        assert quant.data.ndim == 2

    np.testing.assert_array_equal(quant, expected)


@given(anyarray(min_dims=2, max_dims=2))
def test_ParticleQuantity_iteration(arr):
    particle_array = [arr.shape[1] for _ in range(arr.shape[0])]

    prt_quant = ParticleQuantity(
        data=arr,
        prt_num=particle_array,
        dtype=arr.dtype,
        name="",
        label="",
        unit="",
    )

    for quant, subarray in zip(prt_quant, arr):
        assert quant is not prt_quant
        np.testing.assert_array_equal(quant, subarray)


@pytest.fixture(name="ParticleBackend")
def _dummy_ParticleBackend():
    class ParticleBackend:
        def __init__(self, field_name, data):
            self.field_name = field_name
            self.data = data

        def get_data(self, fields):
            if fields == self.field_name:
                return self.data
            else:
                raise ValueError("Wrong field name requested")

    return ParticleBackend


@given(data=anyarray(min_dims=2, max_dims=2), quantity_name=text())
def test_ParticleQuantity_data_reading(ParticleBackend, data, quantity_name):
    backends = [ParticleBackend(quantity_name, d) for d in data]
    particle_numbers = [data.shape[1]] * len(data)
    quantity = ParticleQuantity(
        data=backends,
        prt_num=particle_numbers,
        dtype=data.dtype,
        name=quantity_name,
        label="",
        unit="",
    )

    if len(data) == 1:
        data = np.squeeze(data, axis=0)

    np.testing.assert_array_equal(quantity, data)
    # mask can be accessed after first array
    assert isinstance(quantity.data, np.ma.MaskedArray)
    assert not np.any(quantity.data.mask)


@given(number(), number())
def test_ParticleQuantity_appending_two_numbers(num1, num2):
    assume(type(num1) == type(num2))

    quant = ParticleQuantity(
        data=num1, prt_num=1, dtype=type(num1), name="", label="", unit=""
    )
    quant.append(
        ParticleQuantity(
            data=num2, prt_num=1, dtype=type(num2), name="", label="", unit=""
        )
    )

    assert quant.shape == (2, 1)
    assert quant.ndim == 2
    assert len(quant) == 2
    np.testing.assert_array_equal(quant, np.array([num1, num2]).reshape((2, 1)))


@given(
    lists(
        integers(min_value=_int_limit.min, max_value=_int_limit.max),
        min_size=1,
    ),
    lists(
        integers(min_value=_int_limit.min, max_value=_int_limit.max),
        min_size=1,
    ),
)
def test_ParticleQuantity_appending_two_lists(l1, l2):
    quant = ParticleQuantity(
        data=l1[0], prt_num=1, dtype=int, name="", label="", unit=""
    )

    for v in l1[1:]:
        quant.append(
            ParticleQuantity(
                data=v, prt_num=1, dtype=int, name="", label="", unit=""
            )
        )

    other = ParticleQuantity(
        data=l2[0], prt_num=1, dtype=int, name="", label="", unit=""
    )

    for v in l2[1:]:
        other.append(
            ParticleQuantity(
                data=v, prt_num=1, dtype=int, name="", label="", unit=""
            )
        )

    quant.append(other)

    total_length = len(l1) + len(l2)
    assert quant.shape == (total_length, 1)
    assert quant.ndim == 2
    assert len(quant) == total_length
    np.testing.assert_array_equal(
        quant, np.array(l1 + l2).reshape((total_length, 1))
    )
