# -*- coding: utf-8 -*-
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import one_of
from hypothesis.strategies import text

from nata.axes import Axis
from nata.axes import IterationAxis
from nata.axes import TimeAxis
from nata.axes import UnnamedAxis
from nata.utils.formatting import array_format

from .strategies import anyarray
from .strategies import array_and_basic_indices
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
