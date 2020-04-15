# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nata._axes import Axis
from nata._axes import GridAxis
from nata.types import AxisType


def test_Axis_type_check():
    assert isinstance(Axis, AxisType)


def test_Axis_default_init():
    axis = Axis(np.arange(10))

    assert axis.name == "unnamed"
    assert axis.label == ""
    assert axis.unit == ""
    assert axis.shape == (10,)
    np.testing.assert_array_equal(axis, np.arange(10))


def test_Axis_data_change():
    arr = np.arange(10)
    axis = Axis(arr)
    np.testing.assert_array_equal(axis, arr)

    new = np.random.random_sample(axis.shape)
    axis.data = new
    np.testing.assert_array_equal(axis, new)


def test_Axis_data_shapeMismatch():
    length = 10
    arr = np.arange(length)
    axis = Axis(arr)

    new = np.arange(length + 1)
    with pytest.raises(ValueError, match="Shapes inconsistent"):
        axis.data = new


def test_Axis_name_init():
    axis = Axis(np.empty(10), name="test")
    assert axis.name == "test"


def test_Axis_name_parsing():
    assert Axis(0, name="").name == "unnamed"


def test_Axis_label_init():
    axis = Axis(np.empty(10), label="test")
    assert axis.label == "test"


def test_Axis_unit_init():
    axis = Axis(np.empty(10), unit="test")
    assert axis.unit == "test"


def test_Axis_repr():
    axis = Axis(1, name="some_name", label="some_label", unit="some_unit")
    assert (
        repr(axis)
        == "Axis("
        + "name='some_name', "
        + "label='some_label', "
        + "unit='some_unit', "
        + "data=1"
        + ")"
    )

    axis = Axis([0, 1], name="some_name", label="some_label", unit="some_unit")
    assert (
        repr(axis)
        == "Axis("
        + "name='some_name', "
        + "label='some_label', "
        + "unit='some_unit', "
        + "data=[0, 1]"
        + ")"
    )


def test_Axis_dimensionality_mismatch():
    with pytest.raises(ValueError, match="dimensionality mismatch"):
        Axis(np.empty([]), axis_dim=1)

    with pytest.raises(ValueError, match="dimensionality mismatch"):
        Axis(np.empty((1, 1)), axis_dim=0)


def test_Axis_underlaying_data_dimensionality():
    axis = Axis(np.array(1))
    assert axis._data.shape == (1,)

    axis = Axis(np.array([1]))
    assert axis._data.shape == (1,)

    axis = Axis(np.array([1]), axis_dim=1)
    assert axis._data.shape == (1, 1)


def test_Axis_dimensionality_consist():
    axis = Axis(np.empty(10))
    assert axis.shape == (10,)
    assert axis.ndim == 1

    axis = Axis(np.empty(10), axis_dim=1)
    assert axis.shape == (10,)
    assert axis.ndim == 1

    axis = Axis(np.empty((2, 10)), axis_dim=1)
    assert axis.shape == (2, 10)
    assert axis.ndim == 2


def test_Axis_equivalent():
    axis = Axis(np.array(1))

    assert axis.equivalent(Axis(np.array(1))) is True
    assert axis.equivalent(object()) is False
    assert axis.equivalent(Axis(np.array(1), name="other")) is False
    assert axis.equivalent(Axis(np.array(1), label="other")) is False
    assert axis.equivalent(Axis(np.array(1), unit="other")) is False
    assert axis.equivalent(Axis(np.array([1]), axis_dim=1)) is False


def test_Axis_append():
    axis = Axis(np.array(1))
    axis.append(Axis(np.array(2)))

    np.testing.assert_array_equal(axis, [1, 2])

    axis.append(Axis(np.array([3, 4])))
    np.testing.assert_array_equal(axis, [1, 2, 3, 4])


def test_Axis_append_wrong_type():
    with pytest.raises(TypeError, match="Can not append"):
        Axis(np.array(1)).append(1)


def test_Axis_append_different_axis():
    with pytest.raises(ValueError, match="Mismatch in attributes"):
        Axis(np.array(1), name="some_name").append(
            Axis(np.array(1), name="different_name")
        )


def test_Axis_len():
    axis = Axis(np.array(1))
    assert len(axis) == 1

    axis.append(Axis(np.array(2)))
    assert len(axis) == 2

    axis.append(Axis(np.array([3, 4])))
    assert len(axis) == 4


def test_Axis_iterator_single_item():
    axis = Axis(np.array(1))

    for ax in axis:
        assert ax is axis


def test_Axis_iterator_multiple_items():
    arrays = [np.array(1), np.array(2), np.array(3)]
    axis = Axis(
        arrays[0], name="some_name", label="some_label", unit="some_unit"
    )
    axis.append(
        Axis(arrays[1], name="some_name", label="some_label", unit="some_unit")
    )
    axis.append(
        Axis(arrays[2], name="some_name", label="some_label", unit="some_unit")
    )

    for ax, arr in zip(axis, arrays):
        assert ax is not axis
        assert ax.name == axis.name
        assert ax.label == axis.label
        assert ax.unit == axis.unit
        np.testing.assert_array_equal(ax, [arr])


@pytest.mark.parametrize(
    "init_array, indexing, expected_array",
    [
        (np.arange(10), np.s_[0], np.array(0)),
        (np.arange(10), np.s_[:], np.arange(10)),
        (np.arange(10), np.s_[3:7], np.arange(10)[3:7]),
        (np.arange(10), np.s_[3:7:2], np.arange(10)[3:7:2]),
    ],
    ids=["int", "slice", "range", "range+step"],
)
def test_Axis_getitem_0d(init_array, indexing, expected_array):
    axis = Axis(
        init_array, name="some_name", label="some_label", unit="some_unit"
    )

    subaxis = axis[indexing]

    assert subaxis is not axis
    assert subaxis.name == axis.name
    assert subaxis.label == axis.label
    assert subaxis.unit == axis.unit
    assert subaxis.axis_dim == axis.axis_dim

    np.testing.assert_array_equal(subaxis, expected_array)


@pytest.mark.parametrize(
    "init_array, indexing, expected_array, expected_axis_dim",
    [
        (np.arange(10), np.s_[0], 0, 0),
        (np.arange(10), np.s_[2:2], np.array([]), 1),
        (np.arange(10).reshape((2, 5)), np.s_[0], np.arange(5), 1),
        (
            np.arange(100).reshape((10, 10)),
            np.s_[2:7],
            np.arange(100).reshape((10, 10))[2:7],
            1,
        ),
        (np.arange(10).reshape((2, 5)), np.s_[1, 0], 5, 0),
        (np.arange(10).reshape((2, 5)), np.s_[1, 1:], np.arange(4) + 6, 1),
        (
            np.arange(25).reshape((5, 5)),
            np.s_[1:, 1:],
            np.arange(25).reshape((5, 5))[1:, 1:],
            1,
        ),
    ],
    ids=[
        "len == 1 and (int,)",
        "len == 1 and (slice,)",
        "len != 1 and (int,)",
        "len != 1 and (slice,)",
        "len != 1 and (int, int)",
        "len != 1 and (int, slice)",
        "len != 1 and (slice, slice)",
    ],
)
def test_Axis_getitem_1d(
    init_array, indexing, expected_array, expected_axis_dim
):
    axis = Axis(
        init_array,
        axis_dim=1,
        name="some_name",
        label="some_label",
        unit="some_unit",
    )

    subaxis = axis[indexing]

    assert subaxis is not axis
    assert subaxis.name == axis.name
    assert subaxis.label == axis.label
    assert subaxis.unit == axis.unit
    assert subaxis.axis_dim == expected_axis_dim

    np.testing.assert_array_equal(subaxis, expected_array)


def test_GridAxis_type_check():
    assert isinstance(GridAxis, AxisType)


def test_GridAxis_default_init():
    grid = GridAxis(0.0, 1.0, 10)

    assert grid.name == "unnamed"
    assert grid.label == ""
    assert grid.unit == ""

    assert grid.axis_type == "linear"
    assert grid.shape == (10,)
    assert grid.ndim == 1

    np.testing.assert_array_equal(grid, np.linspace(0, 1, 10))


def test_GridAxis_multidim_init():
    min_values = (0.0, 1.0, 3.0)
    max_values = (10.0, 2.0, 15.0)
    grid = GridAxis(min_values, max_values, 10)

    expected = []
    for min_, max_ in zip(min_values, max_values):
        expected.append(np.linspace(min_, max_, 10))
    expected = np.array(expected)

    np.testing.assert_array_equal(grid, expected)


def test_GridAxis_init_with_data():
    data_arr = np.arange(10)
    gridaxis = GridAxis(
        data=data_arr,
        name="custom_name",
        label="custom_label",
        unit="custom_unit",
    )

    assert gridaxis.name == "custom_name"
    assert gridaxis.label == "custom_label"
    assert gridaxis.unit == "custom_unit"
    assert gridaxis.axis_type == "custom"
    assert gridaxis.grid_cells == 10
    assert len(gridaxis) == 1
    np.testing.assert_array_equal(gridaxis, data_arr)

    data_arr = np.arange(10).reshape((2, 5))
    gridaxis = GridAxis(
        data=data_arr,
        name="custom_name",
        label="custom_label",
        unit="custom_unit",
    )

    assert gridaxis.name == "custom_name"
    assert gridaxis.label == "custom_label"
    assert gridaxis.unit == "custom_unit"
    assert gridaxis.axis_type == "custom"
    assert len(gridaxis) == 2
    np.testing.assert_array_equal(gridaxis, data_arr)


@pytest.mark.parametrize(
    "axis_type, expected",
    [
        (None, "linear"),
        ("lin", "linear"),
        ("linear", "linear"),
        ("log", "logarithmic"),
        ("logarithmic", "logarithmic"),
    ],
    ids=[
        "default",
        "lin->linear",
        "linear->linear",
        "log->logarithmic",
        "logarithmic->logarithmic",
    ],
)
def test_GridAxis_axis_type(axis_type, expected):
    if axis_type is not None:
        assert GridAxis(0.0, 1.0, 10, axis_type=axis_type).axis_type == expected
    else:
        assert GridAxis(0.0, 1.0, 10).axis_type == expected


def test_GridAxis_invalid_axis_type_raise():
    with pytest.raises(ValueError, match="Invalid axis type"):
        GridAxis(0.0, 1.0, 10, axis_type="invlid")


def test_GridAxis_array_interface():
    np.testing.assert_array_equal(
        GridAxis(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10)
    )
    np.testing.assert_array_equal(
        GridAxis(0.1, 1.0, 10, axis_type="log"), np.logspace(-1, 0, 10),
    )


def test_GridAxis_equivalent():
    gridaxis = GridAxis(0.0, 1.0, 10)
    other = GridAxis(0.0, 1.0, 10)
    other.axis_type = "lin"
    assert gridaxis.equivalent(other) is True
    other.axis_type = "log"
    assert gridaxis.equivalent(other) is False
    other.axis_type = "logarithmic"
    assert gridaxis.equivalent(other) is False


def test_GridAxis_append():
    axis = GridAxis(0.0, 1.0, 10)
    expected_axis = np.linspace(0.0, 1.0, 10)
    np.testing.assert_array_equal(axis, expected_axis)

    axis.append(GridAxis(0.0, 1.0, 10))
    np.testing.assert_array_equal(axis, [expected_axis] * 2)

    axis.append(GridAxis([0.0, 0.0], [1.0, 1.0], 10))
    np.testing.assert_array_equal(axis, [expected_axis] * 4)


def test_GridAxis_append_raise_wrong_type():
    with pytest.raises(TypeError, match="Can not append"):
        gridaxis = GridAxis(0.0, 1.0, 10, name="gridaxis")
        gridaxis.append(1)


def test_GridAxis_append_raise_mismatch_attributes():
    with pytest.raises(ValueError, match="Mismatch in attributes"):
        gridaxis = GridAxis(0.0, 1.0, 10, name="gridaxis")
        gridaxis.append(GridAxis(0.0, 1.0, 10, name="different"))


@pytest.mark.parametrize(
    "kwargs, key, expected_grid_cells, expected_shape, expected_array",
    [
        (dict(data=np.arange(10)), np.s_[0], 1, (1,), 0),
        (dict(data=np.arange(10)), np.s_[2:5], 3, (3,), [2, 3, 4]),
        (dict(data=np.arange(10)), np.s_[2:2], 0, (0,), []),
        (
            dict(data=np.arange(21).reshape((3, 7))),
            np.s_[0],
            7,
            (7,),
            [0, 1, 2, 3, 4, 5, 6],
        ),
        (
            dict(data=np.arange(21).reshape((7, 3))),
            np.s_[1:5],
            3,
            (4, 3),
            [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]],
        ),
        (dict(data=np.arange(30).reshape((5, 6))), np.s_[2, 4], 1, (1,), [16],),
        (
            dict(data=np.arange(30).reshape((5, 6))),
            np.s_[2, 1:3],
            2,
            (2,),
            [13, 14],
        ),
        (
            dict(data=np.arange(30).reshape((5, 6))),
            np.s_[2:4, 4],
            1,
            (2, 1),
            [[16], [22]],
        ),
        (
            dict(data=np.arange(30).reshape((5, 6))),
            np.s_[2:4, 1:4],
            3,
            (2, 3),
            [[13, 14, 15], [19, 20, 21]],
        ),
    ],
    ids=[
        "len == 1 and (int,)",
        "len == 1 and (slice,)",
        "len == 1 and (x:x,) -> lead to empty",
        "len != 1 and (int,)",
        "len != 1 and (slice,)",
        "len != 1 and (int, int)",
        "len != 1 and (int, slice)",
        "len != 1 and (slice, int)",
        "len != 1 and (slice, slice)",
    ],
)
def test_GridAxis_getitem(
    kwargs, key, expected_grid_cells, expected_shape, expected_array
):
    gridaxis = GridAxis(
        **kwargs, name="some_name", label="some_label", unit="some_unit"
    )
    subgridaxis = gridaxis[key]

    assert subgridaxis is not gridaxis
    assert subgridaxis.name == gridaxis.name
    assert subgridaxis.label == gridaxis.label
    assert subgridaxis.unit == gridaxis.unit
    assert subgridaxis.axis_type == "custom"
    assert subgridaxis.grid_cells == expected_grid_cells
    assert subgridaxis.shape == expected_shape
    np.testing.assert_array_equal(subgridaxis, expected_array)
