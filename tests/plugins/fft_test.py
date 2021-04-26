# -*- coding: utf-8 -*-
import numpy as np
import numpy.fft as fft
import pytest

from nata.containers import Axis
from nata.containers import GridArray
from nata.containers import GridDataset


def test_array_fft_invalid_ndim():

    with pytest.raises(ValueError, match="0 dimensional GridArrays"):
        GridArray.from_array(1).fft()


def test_array_fft_invalid_axis():

    for invalid_axis in [2, -3]:
        with pytest.raises(ValueError, match="invalid axis index"):
            GridArray.from_array(
                np.arange(100).reshape((10, 10)),
                axes=[Axis(np.arange(10))] * 2,
            ).fft(axes=[invalid_axis])

    for invalid_axis in ["axis2", "abc"]:
        with pytest.raises(ValueError, match="could not be found"):
            GridArray.from_array(
                np.arange(100).reshape((10, 10)),
                axes=[Axis(np.arange(10))] * 2,
            ).fft(axes=[invalid_axis])


def test_array_fft_shape_1d():
    x = np.arange(10)

    for axes in [
        [Axis(x)],
        # [Axis([x] * 2)],
    ]:
        grid = GridArray.from_array(x, axes=axes)

        assert grid.fft().shape == grid.shape
        assert grid.fft().axes[0].shape == grid.axes[0].shape


def test_array_fft_shape_2d():
    x = np.arange(10)
    data = np.arange(100).reshape((10, 10))

    for axes in [
        [Axis(x)] * 2,
        # [Axis([x] * 2)] * 2,
    ]:
        grid = GridArray.from_array(data, axes=axes)

        assert grid.fft().shape == grid.shape
        assert grid.fft().axes[0].shape == grid.axes[0].shape


def test_array_fft_comps_1d():
    x = np.linspace(0, 10 * np.pi, 101)
    grid = GridArray.from_array(np.sin(x), axes=[Axis(x)])

    for comp, fn in zip(
        ["full", "real", "imag", "abs"], [lambda x: x, np.real, np.imag, np.abs]
    ):
        fft_grid = grid.fft(comp=comp)
        fft_data = fft.fftshift(fft.fftn(grid.to_dask()))
        np.testing.assert_array_equal(fft_grid.to_dask(), fn(fft_data))


def test_array_fft_peak_1d():
    x = np.linspace(0, 10 * np.pi, 101)
    grid = GridArray.from_array(np.sin(x), axes=[Axis(x)])
    fft_grid = grid.fft()

    assert (
        fft_grid.to_dask().argmax() == (np.abs(fft_grid.axes[0].to_dask() + 1)).argmin()
    )


def test_dataset_fft_peak_1d():
    time = np.arange(1, 3)
    x = np.linspace(0, 10 * np.pi, 101)

    k_modes = np.arange(len(time)) + 1

    grid = GridDataset.from_array(
        [np.sin(k_i * x) for k_i in k_modes],
        axes=[Axis(time), Axis(np.tile(x, (len(time), 1)))],
    )
    fft_grid = grid.fft()

    for k_i, fft_grid_i in zip(k_modes, fft_grid):
        assert (
            fft_grid_i.to_dask().argmax()
            == (np.abs(fft_grid_i.axes[0].to_dask() + k_i)).argmin()
        )


def test_dataset_fft_selection():
    grid = GridDataset.from_array(np.arange(12).reshape((4, 3)))

    with pytest.raises(ValueError, match="fft along the time axis is not supported"):
        grid.fft(axes=[0])


def test_dataset_fft_invalid_ndim():

    with pytest.raises(ValueError, match="0 dimensional GridDatasets"):
        GridDataset.from_array(np.arange(5)).fft()
