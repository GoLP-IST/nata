# -*- coding: utf-8 -*-
import numpy as np

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


def test_GridDataset_from_array_with_time_axis():
    grid = GridDataset.from_array([[1, 2, 3], [3, 4, 5]], time=[15, 20])

    assert grid.ndim == 2
    assert grid.grid_ndim == 1
    assert grid.shape == (2, 3)
    assert grid.grid_shape == (3,)


def test_GridDataset_from_array_with_iteration_axis():
    grid = GridDataset.from_array([[1, 2, 3], [3, 4, 5]], iteration=[1, 5])

    assert grid.ndim == 2
    assert grid.grid_ndim == 1
    assert grid.shape == (2, 3)
    assert grid.grid_shape == (3,)
