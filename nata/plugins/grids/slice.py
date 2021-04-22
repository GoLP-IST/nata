# -*- coding: utf-8 -*-
from typing import Union

import numpy as np
import dask.array as da

from nata.containers import GridArray
from nata.containers import GridDataset
from nata.containers.grid import stack
from nata.plugins.register import register_container_plugin


@register_container_plugin(GridArray, name="slice")
def slice_grid_array(
    grid: GridArray,
    constant: Union[str, int],
    value: float,
) -> GridArray:
    """Takes a slice of a `GridArray` at a constant value of a given axis.

    Parameters
    ----------
    constant: ``str`` or ``int``
        Name or index that defines the axis taken to be constant in the slice.
    comp: ``float``
        Value of the axis at which the slice is taken.

    Returns
    ------
    :class:`nata.containers.GridArray`:
        Slice of ``grid``.

    Examples
    --------
    Obtain a slice of a two-dimensional array.

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

    """

    if grid.ndim < 1:
        raise ValueError("slice is not available for 0 dimensional GridArray")

    if isinstance(constant, str):
        try:
            ax_idx = list(ax.name for ax in grid.axes).index(constant)
        except ValueError:
            raise ValueError(
                f"axis '{constant}' could not be found in GridArray '{grid.name}'"
            )
    else:
        if constant >= len(grid.axes) or constant < -len(grid.axes):
            raise ValueError(f"invalid axis index '{constant}'")
        ax_idx = constant

    axis = grid.axes[ax_idx]

    if value < np.min(axis.to_dask()) or value >= np.max(axis.to_dask()):
        raise ValueError(f"out of range value for axis '{constant}'")

    # get index of nearest neighbour
    idx = (np.abs(axis.to_dask() - value)).argmin(axis=-1)

    # build data slice
    data_slice = [slice(None)] * len(grid.axes)
    data_slice[ax_idx] = idx

    return GridArray.from_array(
        grid.to_dask()[tuple(data_slice)],
        name=grid.name,
        label=grid.label,
        unit=grid.unit,
        axes=[ax for key, ax in enumerate(grid.axes) if ax is not axis],
    )


@register_container_plugin(GridDataset, name="slice")
def slice_grid_dataset(
    grid: GridDataset,
    constant: Union[str, int],
    value: float,
) -> GridDataset:
    """Takes a slice of a `GridDataset` at a constant value of a given axis.
    Slices are not allowed along the time axis. In other words, `GridDataset`
    slices always preserve the time dependence. To do slices over the time
    axis, consider converting `grid` to a `GridArray` using the `streak()`
    plugin.

    Parameters
    ----------
    constant: ``str`` or ``int``
        Name or index that defines the axis taken to be constant in the slice.
        Cannot refer to the time axis.
    comp: ``float``
        Value of the axis at which the slice is taken.

    Returns
    ------
    :class:`nata.containers.GridDataset`:
        Slice of ``grid``.

    Examples
    --------
    Obtain a slice of a one-dimensional dataset with time dependence.

    >>> from nata.containers import GridDataset
    >>> from nata.containers import Axis
    >>> import numpy as np
    >>> time, x = np.arange(5), np.tile(np.arange(4), (5,1))
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

    """

    if grid.ndim < 2:
        raise ValueError("slice is not available for 0 dimensional GridDataset")

    if isinstance(constant, str):
        try:
            ax_idx = list(ax.name for ax in grid.axes).index(constant)
        except ValueError:
            raise ValueError(
                f"axis '{constant}' could not be found in GridDataset '{grid.name}'"
            )
    else:
        if constant >= len(grid.axes) or constant < -len(grid.axes):
            raise ValueError(f"invalid axis index '{constant}'")
        ax_idx = constant

    axis = grid.axes[ax_idx]

    if axis is grid.time:
        raise ValueError(f"slice along the time axis `{axis.name}` is not supported")
    
    if np.any(value < np.min(axis.to_dask(), axis=-1)) or np.any(value >= np.max(axis.to_dask(), axis=-1)):
        raise ValueError(f"out of range value for axis '{constant}'")

    # apply slices to individual grid arrays and stack them
    return stack([i_grid.slice(constant=ax_idx-1, value=value) for i_grid in grid])