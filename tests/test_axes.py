# -*- coding: utf-8 -*-
import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text

from nata.axes import Axis
from nata.axes import GridAxis
from nata.types import AxisType
from nata.types import GridAxisType

from .strategies import anyarray
from .strategies import array_and_basic_indices
from .strategies import bounded_intergers
from .strategies import number


def test_Axis_type_check():
    assert isinstance(Axis, AxisType)


@given(anyarray())
def test_Axis_default_init(arr):
    axis = Axis(arr)

    assert axis.name == "unnamed"
    assert axis.label == ""
    assert axis.unit == ""
    assert axis.shape == (
        arr.shape[1:] if arr.ndim != 0 and len(arr) == 1 else arr.shape
    )
    if arr.ndim != 0 and len(arr) == 1:
        np.testing.assert_array_equal(axis, arr[0, ...])
    else:
        np.testing.assert_array_equal(axis, arr)


@given(base=anyarray(), new=anyarray())
def test_Axis_data_change_broadcastable(base, new):
    try:
        expected = np.broadcast_to(new, base.shape)
    except ValueError:
        assume(False)

    # compensating the dimension consumption
    if base.shape and len(base) == 1:
        axis = Axis(base[np.newaxis])
    else:
        axis = Axis(base)
    np.testing.assert_array_equal(axis, base)

    axis.data = new
    np.testing.assert_array_equal(axis, expected)


@given(base=anyarray(), new=anyarray())
def test_Axis_data_change_not_broadcastable(base, new):
    try:
        np.broadcast_to(new, base.shape)
        assume(False)  # not valid hypothesis test
    except ValueError:
        pass

    axis = Axis(base)

    with pytest.raises(ValueError):
        axis.data = new


@pytest.mark.parametrize(
    "arr, expected_shape",
    [(1, ()), ([1], ()), ([[1]], (1,)), ([[[1]]], (1, 1))],
    ids=["int -> ()", "[int] -> ()", "[[int]] -> (1,)", "[[[int]]] -> (1, 1)"],
)
def test_Axis_shape(arr, expected_shape):
    axis = Axis(arr)
    assert axis.shape == expected_shape


@pytest.mark.parametrize(
    "arr, expected_ndim",
    [(1, 0), ([1], 0), ([[1]], 1), ([[[1]]], 2)],
    ids=["int -> 0", "[int] -> 0", "[[int]] -> 1", "[[[int]]] -> 2"],
)
def test_Axis_ndim(arr, expected_ndim):
    axis = Axis(arr)
    assert axis.ndim == expected_ndim


@pytest.mark.parametrize(
    "arr, expected_axis_dim",
    [(1, 0), ([1], 0), ([[1]], 1), ([[[1]]], 2)],
    ids=["int -> 0", "[int] -> 0", "[[int]] -> 1", "[[[int]]] -> 2"],
)
def test_Axis_axis_dim(arr, expected_axis_dim):
    axis = Axis(arr)
    assert axis.axis_dim == expected_axis_dim


@given(anyarray(min_dims=0, max_dims=1, include_complex_numbers=False))
def test_Axis_array_interface(arr):
    axis = Axis(arr)
    np.testing.assert_array_equal(
        np.array(axis, dtype=float), arr.astype(float)
    )
    np.testing.assert_array_equal(axis, arr)


@given(anyarray(min_dims=0, max_dims=1, include_complex_numbers=False))
def test_Axis_dtype(arr):
    axis = Axis(arr)
    assert axis.dtype == arr.dtype


def test_Axis_name_init():
    axis = Axis(np.empty(10), name="test")
    assert axis.name == "test"


def test_Axis_name_setter():
    axis = Axis(0, name="something")
    assert axis.name == "something"

    axis.name = "something_else"
    assert axis.name == "something_else"

    with pytest.raises(ValueError, match="Invalid name provided!"):
        axis.name = ""


def test_Axis_name_parsing():
    assert Axis(0, name="").name == "unnamed"


def test_Axis_label_init():
    axis = Axis(np.empty(10), label="test")
    assert axis.label == "test"


def test_Axis_label_setter():
    axis = Axis(0, label="something")
    assert axis.label == "something"

    axis.label = "something_else"
    assert axis.label == "something_else"


def test_Axis_unit_init():
    axis = Axis(np.empty(10), unit="test")
    assert axis.unit == "test"


def test_Axis_unit_setter():
    axis = Axis(0, unit="something")
    assert axis.unit == "something"

    axis.unit = "something_else"
    assert axis.unit == "something_else"


def test_Axis_repr():
    axis = Axis(1, name="some_name", label="some_label", unit="some_unit")

    assert (
        repr(axis)
        == "Axis("
        + "name='some_name', "
        + "label='some_label', "
        + "unit='some_unit', "
        + "axis_dim=0, "
        + "len=1"
        + ")"
    )

    axis = Axis([0, 1], name="some_name", label="some_label", unit="some_unit",)
    assert (
        repr(axis)
        == "Axis("
        + "name='some_name', "
        + "label='some_label', "
        + "unit='some_unit', "
        + "axis_dim=0, "
        + "len=2"
        + ")"
    )

    axis = Axis(
        [[0, 1]], name="some_name", label="some_label", unit="some_unit",
    )
    assert (
        repr(axis)
        == "Axis("
        + "name='some_name', "
        + "label='some_label', "
        + "unit='some_unit', "
        + "axis_dim=1, "
        + "len=1"
        + ")"
    )
    axis = Axis(
        [[0, 1], [0, 1]],
        name="some_name",
        label="some_label",
        unit="some_unit",
    )
    assert (
        repr(axis)
        == "Axis("
        + "name='some_name', "
        + "label='some_label', "
        + "unit='some_unit', "
        + "axis_dim=1, "
        + "len=2"
        + ")"
    )


def test_Axis_dimensionality():
    axis = Axis(np.empty(10))
    assert axis.shape == (10,)
    assert axis.ndim == 1
    assert len(axis) == 10

    axis = Axis(np.empty((1, 10)))
    assert axis.shape == (10,)
    assert axis.ndim == 1
    assert len(axis) == 1

    axis = Axis(np.empty((2, 10)))
    assert axis.shape == (2, 10)
    assert axis.ndim == 2
    assert len(axis) == 2


def test_Axis_equivalent():
    axis = Axis(1)

    assert axis.equivalent(Axis(1)) is True
    assert axis.equivalent(object()) is False
    assert axis.equivalent(Axis(np.array(1), name="other")) is False
    assert axis.equivalent(Axis(np.array(1), label="other")) is False
    assert axis.equivalent(Axis(np.array(1), unit="other")) is False
    assert axis.equivalent(Axis(np.array([1]))) is True
    assert axis.equivalent(Axis(np.array([[1]]))) is False


def test_Axis_append():
    axis = Axis(1)
    assert axis.shape == ()

    axis.append(Axis(2))
    assert axis.shape == (2,)
    np.testing.assert_array_equal(axis, [1, 2])

    axis.append(Axis([3, 4]))
    assert axis.shape == (4,)
    np.testing.assert_array_equal(axis, [1, 2, 3, 4])

    axis = Axis(1)
    axis.append(Axis([2, 3]))
    assert axis.shape == (3,)
    np.testing.assert_array_equal(axis, [1, 2, 3])

    axis = Axis([[1, 2]])
    assert axis.shape == (2,)
    axis.append(Axis([[3, 4]]))
    assert axis.shape == (2, 2)
    axis.append(Axis([[5, 6]]))
    assert axis.shape == (3, 2)


def test_Axis_append_wrong_type():
    with pytest.raises(TypeError, match="Can not append"):
        Axis(np.array(1)).append(1)


def test_Axis_append_different_axis():
    with pytest.raises(ValueError, match="Mismatch in attributes"):
        Axis(np.array(1), name="some_name").append(
            Axis(np.array(1), name="different_name")
        )


def test_Axis_len():
    axis = Axis(1)
    assert len(axis) == 1

    axis.append(Axis(2))
    assert len(axis) == 2

    axis.append(Axis([3, 4]))
    assert len(axis) == 4

    axis = Axis([1])
    assert len(axis) == 1

    axis = Axis([1, 2])
    assert len(axis) == 2

    axis = Axis([[1, 2]])
    assert len(axis) == 1


def test_Axis_iterator_single_item():
    axis = Axis(1)

    for ax in axis:
        assert ax is not axis
        np.testing.assert_array_equal(ax, 1)


def test_Axis_iterator_multiple_items_0d():
    name = "some_name"
    label = "some_label"
    unit = "some_unit"

    arrays = [np.array(1), np.array(2), np.array(3)]
    axis = Axis(arrays[0], name=name, label=label, unit=unit)
    axis.append(Axis(arrays[1], name=name, label=label, unit=unit))
    axis.append(Axis(arrays[2], name=name, label=label, unit=unit))

    for ax, arr in zip(axis, arrays):
        assert ax is not axis
        assert ax.name == name
        assert ax.label == label
        assert ax.unit == unit
        np.testing.assert_array_equal(ax, arr)


@given(array_and_basic_indices())
def test_Axis_getitem(arr_and_indexing):
    arr, indexing = arr_and_indexing
    name = "some_name"
    label = "some_label"
    unit = "some_unit"

    if arr.shape and len(arr) == 1:
        axis = Axis(arr[np.newaxis], name=name, label=label, unit=unit)
    else:
        axis = Axis(arr, name=name, label=label, unit=unit)

    subaxis = axis[indexing]
    subarr = arr[indexing]

    assert subaxis is not axis
    assert subaxis.name == name
    assert subaxis.label == label
    assert subaxis.unit == unit
    assert subaxis.ndim == subarr.ndim
    assert subaxis.shape == subarr.shape

    np.testing.assert_array_equal(subaxis, subarr)


@pytest.mark.parametrize("indexing", [object(), "something"])
@given(arr=anyarray(min_dims=0, max_dims=1))
def test_Axis_getitem_raise_when_not_basic_indexing(arr, indexing):
    axis = Axis(arr)

    with pytest.raises(IndexError, match="Only basic indexing is supported!"):
        axis[indexing]


def test_GridAxis_type_check():
    assert isinstance(GridAxis, AxisType)
    assert isinstance(GridAxis, GridAxisType)


@given(
    num=number(include_complex_numbers=False).filter(lambda n: n > 0.0),
    delta=number(include_complex_numbers=False).filter(lambda n: n >= 0.0),
    cells=bounded_intergers(min_value=1, max_value=100),
    label=text(),
    unit=text(),
    axis_type=sampled_from(("linear", "logarithmic", "log", "lin")),
)
def test_GridAxis_from_limits(num, delta, cells, label, unit, axis_type):
    min_ = num
    max_ = num + delta
    gridaxis = GridAxis.from_limits(
        min_,
        max_,
        cells,
        name="some_name",
        label=label,
        unit=unit,
        axis_type=axis_type,
    )

    assert gridaxis.name == "some_name"
    assert gridaxis.label == label
    assert gridaxis.unit == unit

    assert gridaxis.axis_type == axis_type
    assert gridaxis.axis_dim == 1

    if axis_type in ("lin", "linear"):
        np.testing.assert_array_equal(gridaxis, np.linspace(min_, max_, cells))
    else:
        np.testing.assert_array_equal(
            gridaxis, np.logspace(np.log10(min_), np.log10(max_), cells)
        )


@given(
    invalid_str=text().filter(lambda s: s not in GridAxis._supported_axis_types)
)
def test_GridAxis_from_limits_invlid_axis_type_raise(invalid_str):
    with pytest.raises(ValueError, match="Invalid axis type provided"):
        GridAxis.from_limits(
            0.0, 1.0, 10, axis_type=invalid_str,
        )


@given(text())
def test_GridAxis_invalid_axis_type(s):
    with pytest.raises(
        ValueError, match="('lin', 'linear', 'log', 'logarithmic', 'custom')"
    ):
        GridAxis([1, 2], axis_type=s)


def test_GridAxis_iteration():
    gridaxis = GridAxis(np.arange(10), axis_type="custom")
    for i, axis in enumerate(gridaxis):
        assert axis is not gridaxis
        assert isinstance(axis, GridAxis)
        assert axis.axis_type == "custom"
        np.testing.assert_array_equal(i, axis)


@given(array_and_basic_indices())
def test_GridAxis_getitem(arr_and_indexing):
    arr, indexing = arr_and_indexing
    name = "some_name"
    label = "some_label"
    unit = "some_unit"

    axis = GridAxis(arr, name=name, label=label, unit=unit)

    subaxis = axis[indexing]
    subarr = arr[indexing]

    assert subaxis is not axis
    assert subaxis.name == name
    assert subaxis.label == label
    assert subaxis.unit == unit
    assert subaxis.ndim == subarr.ndim
    assert subaxis.shape == subarr.shape

    np.testing.assert_array_equal(subaxis, subarr)


@pytest.mark.parametrize("indexing", [object(), "something"])
@given(arr=anyarray(min_dims=0, max_dims=1))
def test_GridAxis_getitem_raise_when_not_basic_indexing(arr, indexing):
    gridaxis = GridAxis(arr)

    with pytest.raises(IndexError, match="Only basic indexing is supported!"):
        gridaxis[indexing]


def test_GridAxis_axis_type_setter_valid():
    gridaxis = GridAxis([0, 1])
    # default
    assert gridaxis.axis_type == "linear"

    # try to set all possible
    for valid in GridAxis._supported_axis_types:
        gridaxis.axis_type = valid
        assert gridaxis.axis_type == valid


@given(
    invalid_str=text().filter(lambda s: s not in GridAxis._supported_axis_types)
)
def test_GridAxis_axis_type_setter_invalid(invalid_str):
    gridaxis = GridAxis([0, 1])
    with pytest.raises(
        ValueError, match=f"'{invalid_str}' is not supported for axis_type!"
    ):
        gridaxis.axis_type = invalid_str


@given(
    label=text(),
    unit=text(),
    axis_type=sampled_from(("linear", "logarithmic", "log", "lin")),
)
def test_GridAxis_repr(label, unit, axis_type):
    gridaxis = GridAxis(
        1, name="some_name", label=label, unit=unit, axis_type=axis_type
    )
    assert (
        repr(gridaxis)
        == "GridAxis("
        + "name='some_name', "
        + f"label='{label}', "
        + f"unit='{unit}', "
        + "axis_dim=1, "
        + f"axis_type={axis_type}"
        + ")"
    )


def test_GridAxis_equivalent_raise_from_parent_class():
    base = GridAxis([0, 1], name="something")
    assert base.equivalent(GridAxis([0, 1], name="something_else")) is False


def test_GridAxis_equivalent_raise_different_axis_type():
    base = GridAxis([0, 1], axis_type="linear")
    assert base.equivalent(GridAxis([0, 1], axis_type="log")) is False


@given(
    arr=(
        integers(min_value=2, max_value=22).flatmap(
            lambda n: arrays(np.dtype(float), (n, 123)),
        )
    ),
    label=text(),
    unit=text(),
)
def test_GridAxis_append(arr, label, unit):
    gridaxis = GridAxis(arr[0], name="some_name", label=label, unit=unit)

    for i, d in enumerate(arr):
        if i == 0:
            continue
        gridaxis.append(
            GridAxis(arr[i], name="some_name", label=label, unit=unit)
        )

    assert gridaxis.axis_dim == 1
    assert gridaxis.ndim == 2
    np.testing.assert_array_equal(gridaxis, arr)
