# -*- coding: utf-8 -*-
from nata.containers import GridDataset


def test_GridDataset_from_array():
    grid = GridDataset.from_array([[1, 2, 3], [3, 4, 5]])

    assert grid.ndim == grid.grid_ndim == 2
    assert grid.shape == grid.grid_shape == (2, 3)


def test_GridDataset_from_array_with_time_axis():
    grid = GridDataset.from_array([[1, 2, 3], [3, 4, 5]], time=[15, 20])

    assert grid.ndim == 2
    assert grid.grid_ndim == 1
    assert grid.shape == (2, 3)
    assert grid.grid_shape == (3,)
