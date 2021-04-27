# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nata.containers import Axis
from nata.containers import GridArray
from nata.containers import GridDataset

def test_array_transpose_invalid_axes():

    grid = GridArray.from_array(np.arange(7*6).reshape((7, 6)))

    for invalid_axes in [
        [0,],
        [1, 1],
        ["axis0", "axis0"],
    ]:
        with pytest.raises(ValueError, match="invalid transpose axes"):
            grid.transpose(axes=invalid_axes)

    for invalid_axis in [2, -3]:
        with pytest.raises(ValueError, match="invalid axis index"):
            grid.transpose(axes=[invalid_axis])

def test_array_transpose_shape():

    data = np.zeros((7,6,5))
    grid = GridArray.from_array(data)

    for tr_axes in [
        None,
        [0,1,2],
        [0,2,1],
        [2,1,0],
        [1,2,0],
    ]:
        tr_grid = grid.transpose(axes=tr_axes)
        assert tr_grid.shape == np.transpose(data, axes=tr_axes).shape

def test_array_transpose_axes():

    data = np.zeros((7,6,5))
    grid = GridArray.from_array(data)

    tr_grid = grid.transpose(axes=[1,2,0])

    assert tr_grid.axes[0] is grid.axes[1]
    assert tr_grid.axes[1] is grid.axes[2]
    assert tr_grid.axes[2] is grid.axes[0]

def test_array_transpose_data():

    data = np.arange(7*6*5).reshape((7,6,5))
    grid = GridArray.from_array(data)

    tr_grid = grid.transpose(axes=[1,2,0])

    np.testing.assert_array_equal(tr_grid, np.transpose(data, axes=[1,2,0]))

def test_dataset_transpose_shape():

    data = np.zeros((7,6,5))
    grid = GridDataset.from_array(data)

    for tr_axes in [
        None,
        [1,2],
        [2,1],
    ]:
        tr_grid = grid.transpose(axes=tr_axes)
        axes = ([0,] + tr_axes) if tr_axes else [0,2,1] 
        assert tr_grid.shape == np.transpose(data, axes=axes).shape

def test_dataset_transpose_selection():
    grid = GridDataset.from_array(np.arange(12).reshape((4, 3)))

    with pytest.raises(ValueError, match="transpose along the time axis is not supported"):
        grid.transpose(axes=[0, 1])