# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import integers
from hypothesis.strategies import one_of
from hypothesis.strategies import text

from nata.axes import Axis
from nata.axes import GridAxis
from nata.axes import IterationAxis
from nata.axes import TimeAxis
from nata.axes import UnnamedAxis
from nata.utils.formatting import array_format

from .strategies import anyarray
from .strategies import array_and_basic_indices
from .strategies import array_with_two_entries
from .strategies import number
from .strategies import number_or_none


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

    # 0-dim -> don't have a size
    with pytest.raises(TypeError, match="unsized object"):
        len(axis)

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


strategies_for_number_tests = (
    number(include_complex_numbers=False),
    number(include_complex_numbers=False),
    integers(min_value=1, max_value=1_000),
    text(),
    text(),
    text(),
)


@given(*strategies_for_number_tests)
def test_GridAxis_init_numbers(num1, num2, length, name, label, unit):
    assume(num1 <= num2)

    gridaxis = GridAxis(
        lower_boundary=num1,
        upper_boundary=num2,
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


@given(*strategies_for_number_tests)
def test_GridAxis_array_interface_with_singleValue_default(
    num1, num2, length, name, label, unit
):
    assume(num1 < num2)

    gridaxis = GridAxis(
        lower_boundary=num1,
        upper_boundary=num2,
        axis_length=length,
        name=name,
        label=label,
        unit=unit,
    )

    np.testing.assert_array_almost_equal(
        gridaxis, np.linspace(num1, num2, length)
    )


@given(*strategies_for_number_tests)
def test_GridAxis_array_interface_with_singleValue_linear(
    num1, num2, length, name, label, unit
):
    assume(num1 < num2)

    gridaxis = GridAxis(
        lower_boundary=num1,
        upper_boundary=num2,
        axis_length=length,
        axis_type="linear",
        name=name,
        label=label,
        unit=unit,
    )

    np.testing.assert_array_almost_equal(
        gridaxis, np.linspace(num1, num2, length)
    )


@given(*strategies_for_number_tests)
def test_GridAxis_array_interface_with_singleValue_lin(
    num1, num2, length, name, label, unit
):
    assume(num1 < num2)

    gridaxis = GridAxis(
        lower_boundary=num1,
        upper_boundary=num2,
        axis_length=length,
        axis_type="lin",
        name=name,
        label=label,
        unit=unit,
    )

    np.testing.assert_array_almost_equal(
        gridaxis, np.linspace(num1, num2, length)
    )


# RuntimeWarnings can appear from numpy - as using logspace
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@given(*strategies_for_number_tests)
def test_GridAxis_array_interface_with_singleValue_logarithmic(
    num1, num2, length, name, label, unit
):
    assume(num1 < num2)

    gridaxis = GridAxis(
        lower_boundary=num1,
        upper_boundary=num2,
        axis_length=length,
        axis_type="logarithmic",
        name=name,
        label=label,
        unit=unit,
    )

    np.testing.assert_array_almost_equal(
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
        lower_boundary=num1,
        upper_boundary=num2,
        axis_length=length,
        axis_type="log",
        name=name,
        label=label,
        unit=unit,
    )

    np.testing.assert_array_almost_equal(
        gridaxis, np.logspace(np.log10(num1), np.log10(num2), length)
    )


sort_along_second_axis = partial(np.sort, axis=1)
strategies_for_array_tests = (
    array_with_two_entries(array_length=1_000).map(sort_along_second_axis),
    integers(min_value=1, max_value=1_000),
    text(),
    text(),
    text(),
)


@given(*strategies_for_array_tests)
@settings(deadline=None)
def test_GridAxis_init_array(data, length, name, label, unit):
    lower = data[:, 0]
    upper = data[:, 1]

    gridaxis = GridAxis(
        lower_boundary=lower,
        upper_boundary=upper,
        axis_length=length,
        name=name,
        label=label,
        unit=unit,
    )

    assert gridaxis.name == name
    assert gridaxis.label == label
    assert gridaxis.unit == unit

    assert len(gridaxis) == len(data)
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
        lower_boundary=lower,
        upper_boundary=upper,
        axis_length=length,
        name=name,
        label=label,
        unit=unit,
    )

    expected = []
    for l, u in zip(lower, upper):
        expected.append(np.linspace(l, u, length))
    expected = np.array(expected)

    np.testing.assert_array_almost_equal(gridaxis, expected)

    # test for individual to ensure correct passing through
    for axis, expected_axis in zip(gridaxis, expected):
        np.testing.assert_array_almost_equal(axis, expected_axis)


@given(*strategies_for_number_tests)
def test_GridAxis_append_with_numbers(num1, num2, length, name, label, unit):
    assume(num1 < num2)
    gridaxis = GridAxis(
        lower_boundary=num1,
        upper_boundary=num2,
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
    np.testing.assert_array_almost_equal(gridaxis, expected)

    for axis, expected_axis in zip(gridaxis, expected):
        np.testing.assert_array_almost_equal(axis, expected_axis)


@given(*strategies_for_array_tests)
@settings(deadline=None)
def test_GridAxis_append_with_arrays(data, length, name, label, unit):
    lower = data[:, 0]
    upper = data[:, 1]

    gridaxis = GridAxis(
        lower_boundary=lower,
        upper_boundary=upper,
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

    np.testing.assert_array_almost_equal(gridaxis, expected)
    for axis, expected_axis in zip(gridaxis, expected):
        np.testing.assert_array_almost_equal(axis, expected_axis)
