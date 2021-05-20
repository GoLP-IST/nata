# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nata.containers import Axis
from nata.containers import GridArray
from nata.containers import GridDataset


def test_streak_type():
    grid = GridDataset.from_array(np.arange(12).reshape((4, 3)))

    assert isinstance(grid.streak(), GridArray)


def test_streak_shape():
    grid = GridDataset.from_array(np.arange(12).reshape((4, 3)))

    assert grid.streak().shape == grid.shape


def test_streak_axes_shape():

    time = np.arange(3)
    x = np.arange(10)

    grid = GridDataset.from_array(
        np.tile(x, (len(time), 1)),
        axes=[Axis.from_array(time), Axis.from_array([x for time_i in time])],
    )

    stk_grid = grid.streak()

    assert stk_grid.shape == grid.shape
    assert stk_grid.axes[0].shape == time.shape
    assert stk_grid.axes[1].shape == x.shape


def test_streak_invalid_ndim():

    with pytest.raises(ValueError, match="0 dimensional GridDatasets"):
        GridDataset.from_array(np.arange(5)).streak()
