# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nata.containers import GridDataset


def test_GridDataset_from_array():
    grid = GridDataset.from_array([[1, 2, 3], [3, 4, 5]])

    # general information
    assert grid.name == "unnamed"
    assert grid.label == "unlabeled"
    assert grid.unit == ""

    # array and grid information
    assert grid.ndim == grid.grid_ndim == 2
    assert grid.shape == grid.grid_shape == (2, 3)
    assert grid.dtype == np.int64

    # axis info
    assert grid.axes[0].name == "axis0"
    assert grid.axes[1].name == "axis1"


def test_GridDataset_from_array_naming():
    grid = GridDataset.from_array(
        [[1, 2, 3], [3, 4, 5]],
        name="my_new_name",
        label="My New Label",
        unit="some unit",
    )

    assert grid.name == "my_new_name"
    assert grid.label == "My New Label"
    assert grid.unit == "some unit"


def test_GridDataset_change_name():
    grid = GridDataset.from_array([[1, 2, 3], [3, 4, 5]], name="old_name")

    assert grid.name == "old_name"
    grid.name = "new_name"
    assert grid.name == "new_name"


def test_GridDataset_raise_invalid_name():
    with pytest.raises(ValueError, match="has to be an identifier"):
        GridDataset.from_array([1, 2], name="some invalid name")

    grid = GridDataset.from_array([1, 2])
    with pytest.raises(ValueError, match="has to be an identifier"):
        grid.name = "some invalid name"


def test_GridDataset_from_array_with_time_axis():
    grid = GridDataset.from_array([[1, 2, 3], [3, 4, 5]], time=[15, 20])

    assert grid.ndim == 2
    assert grid.grid_ndim == 1
    assert grid.shape == (2, 3)
    assert grid.grid_shape == (3,)

    assert grid.axes[0].name == "time"
    assert grid.axes[1].name == "axis0"


def test_GridDataset_from_array_with_iteration_axis():
    grid = GridDataset.from_array([[1, 2, 3], [3, 4, 5]], iteration=[1, 5])

    assert grid.ndim == 2
    assert grid.grid_ndim == 1
    assert grid.shape == (2, 3)
    assert grid.grid_shape == (3,)

    assert grid.axes[0].name == "iteration"
    assert grid.axes[1].name == "axis0"


def test_GridDataset_from_array_time_precedence():
    grid = GridDataset.from_array(
        [[1, 2, 3], [3, 4, 5]],
        iteration=[1, 5],
        time=[2.2, 5.5],
    )

    assert grid.axes[0].name == "time"
    assert grid.axes[1].name == "axis0"

    assert "iteration" in grid.axes

    np.testing.assert_array_equal(grid.axes.time, [2.2, 5.5])
    np.testing.assert_array_equal(grid.axes.iteration, [1, 5])
