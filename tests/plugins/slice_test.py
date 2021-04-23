# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nata.containers import Axis
from nata.containers import GridArray
from nata.containers import GridDataset

def test_array_slice_dimensionality():
    grid = GridArray.from_array(np.arange(96).reshape((8, 4, 3)))

    sliced_grid = grid.slice(constant="axis0", value=0)

    assert sliced_grid.ndim == 2
    assert sliced_grid.axes[0].name == grid.axes[1].name
    assert sliced_grid.axes[1].name == grid.axes[2].name

    np.testing.assert_array_equal(sliced_grid, np.arange(12).reshape(4, 3))


def test_array_slice_selection():
    grid = GridArray.from_array(np.arange(100).reshape((10, 10)))

    with pytest.raises(ValueError, match="out of range value"):
        grid.slice(constant="axis0", value=-1)

    with pytest.raises(ValueError, match="could not be found"):
        grid.slice(constant="axis2", value=0)

    with pytest.raises(ValueError, match="invalid axis index"):
        grid.slice(constant=2, value=0)

    with pytest.raises(ValueError, match="invalid axis index"):
        grid.slice(constant=-3, value=0)


def test_array_slice_invalid_ndim():

    with pytest.raises(ValueError, match="0 dimensional GridArrays"):
        GridArray.from_array(1).slice(constant="axis0", value=0)


def test_dataset_slice_dimensionality():
    data = np.arange(96).reshape((8, 4, 3))
    grid = GridDataset.from_array(data)

    sliced_grid = grid.slice(constant="axis0", value=0)

    assert sliced_grid.ndim == 2
    assert sliced_grid.axes[0].name == grid.axes[0].name
    assert sliced_grid.axes[1].name == grid.axes[2].name

    np.testing.assert_array_equal(sliced_grid, data[:,0,:])

def test_dataset_slice_selection():
    grid = GridDataset.from_array(np.arange(12).reshape((4, 3)))

    with pytest.raises(ValueError, match=f"slice along the time axis is not supported"):
        grid.slice(constant=grid.time.name, value=0)

def test_dataset_slice_invalid_ndim():

    with pytest.raises(ValueError, match="0 dimensional GridDatasets"):
        GridDataset.from_array(np.arange(5)).slice(constant="axis0", value=1)