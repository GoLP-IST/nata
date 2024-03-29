# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

from nata.containers import GridArray
from nata.containers import GridDataset
from nata.containers import register_plugin
from nata.containers.grid import stack


def get_slice_axis(
    grid: Union[GridArray, GridDataset],
    constant: Union[str, int],
):
    if isinstance(constant, str):
        try:
            slice_axis = list(ax.name for ax in grid.axes).index(constant)
        except ValueError:
            raise ValueError(f"axis '{constant}' could not be found in '{grid.name}'")
    else:
        if constant >= len(grid.axes) or constant < -len(grid.axes):
            raise ValueError(f"invalid axis index '{constant}'")
        slice_axis = constant

    return slice_axis


@register_plugin(name="slice")
def slice_grid_array(
    grid: GridArray,
    constant: Union[str, int],
    value: float,
) -> GridArray:
    """
    Takes a slice of a `GridArray` at a constant value of a given axis.

    Arguments:
        constant:
            Name or index that defines the axis taken to be constant in the slice.
        value:
            Value of the axis at which the slice is taken.

    Returns:
        Slice of ``grid``.

    Examples:
        Obtain a slice of a two-dimensional array.

        ```pycon
        >>> from nata.containers import GridArray
        >>> from nata.containers import Axis
        >>> import numpy as np
        >>> x = np.arange(5)
        >>> data = np.arange(25).reshape((5, 5))
        >>> grid = GridArray.from_array(data, axes=[Axis(x), Axis(x)])
        >>> grid.slice(constant=0, value=1).to_numpy()
        array([5, 6, 7, 8, 9]) # the second column
        >>> grid.slice(constant=1, value=1).to_numpy()
        array([ 1,  6, 11, 16, 21]) # the second row
        ```
    """

    if grid.ndim < 1:
        raise ValueError("slice is not available for 0 dimensional GridArrays")

    # get slice axis
    slice_axis = get_slice_axis(grid, constant)

    axis = grid.axes[slice_axis]

    if value < np.min(axis.to_dask()) or value >= np.max(axis.to_dask()):
        raise ValueError(f"out of range value for axis '{constant}'")

    # get index of nearest neighbour
    slice_idx = (np.abs(axis.to_dask() - value)).argmin(axis=-1)

    # build data slice
    data_slice = [slice(None)] * len(grid.axes)
    data_slice[slice_axis] = slice_idx

    return GridArray.from_array(
        grid.to_dask()[tuple(data_slice)],
        name=grid.name,
        label=grid.label,
        unit=grid.unit,
        axes=[ax for key, ax in enumerate(grid.axes) if ax is not axis],
        time=grid.time,
    )


@register_plugin(name="slice")
def slice_grid_dataset(
    grid: GridDataset,
    constant: Union[str, int],
    value: float,
) -> GridDataset:
    """
    Takes a slice of a `GridDataset` at a constant value of a given axis.
    Slices are not allowed along the time axis. In other words, `GridDataset`
    slices always preserve the time dependence. To do slices over the time
    axis, consider converting `grid` to a `GridArray` using the `streak()`
    plugin.

    Arguments:
        constant:
            Name or index that defines the axis taken to be constant in the slice.
            Must not refer to the time axis.
        value:
            Value of the axis at which the slice is taken.

    Returns:
        Slice of ``grid``.

    Examples:
        Obtain a slice of a one-dimensional dataset with time dependence.

        ```pycon
        >>> from nata.containers import GridDataset
        >>> from nata.containers import Axis
        >>> import numpy as np
        >>> time, x = np.arange(5), np.tile(np.arange(4), (5, 1))
        >>> data = np.arange(20).reshape((5, 4))
        >>> grid = GridDataset.from_array(data, axes=[Axis(time), Axis(x)])
        >>> grid.to_numpy()
        array([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, 19]])
        >>> grid.slice(constant=1, value=0).to_numpy()
        array([ 0,  4,  8, 12, 16]) # the first column
        ```
    """

    if grid.ndim < 2:
        raise ValueError("slice is not available for 0 dimensional GridDatasets")

    # get slice axis
    slice_axis = get_slice_axis(grid, constant)

    axis = grid.axes[slice_axis]

    if axis is grid.time:
        raise ValueError("slice along the time axis is not supported")

    if np.any(value < np.min(axis.to_dask(), axis=-1)) or np.any(
        value >= np.max(axis.to_dask(), axis=-1)
    ):
        raise ValueError(f"out of range value for axis '{constant}'")

    # apply slice to individual grid arrays and stack them
    return stack(
        [i_grid.slice(constant=slice_axis - 1, value=value) for i_grid in grid]
    )
