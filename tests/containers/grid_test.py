# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
import pytest

from nata.containers import GridArray


def test_GridArray_from_array_default():
    grid_arr = GridArray.from_array(da.arange(12, dtype=int).reshape((4, 3)))

    assert grid_arr.shape == (4, 3)
    assert grid_arr.ndim == 2
    assert grid_arr.dtype == int

    assert grid_arr.axes[0].name == "axis0"
    assert grid_arr.axes[0].label == "unlabeled"
    assert grid_arr.axes[0].unit == ""
    assert grid_arr.axes[0].shape == (4,)

    assert grid_arr.axes[1].name == "axis1"
    assert grid_arr.axes[1].label == "unlabeled"
    assert grid_arr.axes[1].unit == ""
    assert grid_arr.axes[1].shape == (3,)

    assert grid_arr.time.name == "time"
    assert grid_arr.time.label == "time"
    assert grid_arr.time.unit == ""

    assert grid_arr.name == "unnamed"
    assert grid_arr.label == "unlabeled"
    assert grid_arr.unit == ""


def test_GridArray_from_array_check_name():
    grid_arr = GridArray.from_array([], name="custom_name")
    assert grid_arr.name == "custom_name"


def test_GridArray_from_array_check_label():
    grid_arr = GridArray.from_array([], label="custom label")
    assert grid_arr.label == "custom label"


def test_GridArray_from_array_check_unit():
    grid_arr = GridArray.from_array([], unit="custom unit")
    assert grid_arr.unit == "custom unit"


def test_GridArray_from_array_check_time():
    grid_arr = GridArray.from_array([], time=123)
    np.testing.assert_array_equal(grid_arr.time, 123)
    assert grid_arr.time.name == "time"
    assert grid_arr.time.label == "time"
    assert grid_arr.time.unit == ""


def test_GridArray_from_array_check_axes():
    grid_arr = GridArray.from_array([0, 1], axes=[[0, 1]])
    np.testing.assert_array_equal(grid_arr.axes[0], [0, 1])
    assert grid_arr.axes[0].name == "axis0"
    assert grid_arr.axes[0].label == "unlabeled"
    assert grid_arr.axes[0].unit == ""


def test_GridArray_from_array_raise_invalid_name():
    with pytest.raises(ValueError, match="'name' has to be a valid identifier"):
        GridArray.from_array([], name="invalid name")


def test_GridArray_from_array_raise_invalid_time():
    with pytest.raises(ValueError, match="time axis has to be 0 dimensional"):
        GridArray.from_array([], time=[0, 1, 2])


def test_GridArray_from_array_raise_invalid_axes():
    # invalid number of axes
    with pytest.raises(ValueError, match="mismatches with dimensionality of data"):
        GridArray.from_array([], axes=[0, 1])

    # axes which are not 1D dimensional
    with pytest.raises(ValueError, match="only 1D axis for GridArray are supported"):
        GridArray.from_array([0, 1], axes=[[[0, 1]]])

    # axis mismatch with shape of data
    with pytest.raises(ValueError, match="inconsistency between data and axis shape"):
        GridArray.from_array([0, 1], axes=[[0, 1, 2, 3]])


def test_GridArray_change_name_by_prop():
    grid_arr = GridArray.from_array([])
    assert grid_arr.name == "unnamed"

    grid_arr.name = "new_name"
    assert grid_arr.name == "new_name"

    with pytest.raises(ValueError, match="name has to be an identifier"):
        grid_arr.name = "invalid name"


def test_GridArray_change_label_by_prop():
    grid_arr = GridArray.from_array([])
    assert grid_arr.label == "unlabeled"

    grid_arr.label = "new label"
    assert grid_arr.label == "new label"


def test_GridArray_change_unit_by_prop():
    grid_arr = GridArray.from_array([])
    assert grid_arr.unit == ""

    grid_arr.unit = "new unit"
    assert grid_arr.unit == "new unit"
