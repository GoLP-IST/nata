# -*- coding: utf-8 -*-
from typing import Optional
from typing import Union

import dask.array as da

from nata.containers import GridArray
from nata.containers import GridDataset
from nata.containers.grid import stack
from nata.plugins.register import register_container_plugin


def get_transpose_axes(
    grid: Union[GridArray, GridDataset], axes: list, offset: int = 0
):
    # build transpose axes
    if axes is None:
        # no axes provided, all axes are transpose axes
        tr_axes = list(reversed(range(offset, grid.ndim)))
    else:
        # axes provided, determine transpose axes
        tr_axes = []

        for axis in axes:
            if isinstance(axis, str):
                try:
                    idx = list(ax.name for ax in grid.axes).index(axis)
                except ValueError:
                    raise ValueError(
                        f"axis '{axis}' could not be found in '{grid.name}'"
                    )
            else:
                if axis >= len(grid.axes) or axis < -len(grid.axes):
                    raise ValueError(f"invalid axis index '{axis}'")
                idx = axis

            tr_axes.append(idx)

    return tr_axes


@register_container_plugin(GridArray, name="transpose")
def transpose_grid_array(
    grid: GridArray,
    axes: Optional[list] = None,
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

    # get transpose axes
    tr_axes = get_transpose_axes(grid, axes)

    if len(set(tr_axes)) is not grid.ndim:
        raise ValueError("invalid transpose axes")

    return GridArray.from_array(
        da.transpose(grid.to_dask(), axes=tr_axes),
        name=grid.name,
        label=grid.label,
        unit=grid.unit,
        axes=[grid.axes[axis] for axis in tr_axes],
        time=grid.time,
    )


@register_container_plugin(GridDataset, name="transpose")
def transpose_grid_dataset(
    grid: GridDataset,
    axes: Optional[list] = None,
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
        Must not refer to the time axis.
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

    # get transpose axes
    tr_axes = get_transpose_axes(grid, axes, offset=1)

    if 0 in tr_axes:
        raise ValueError("transpose along the time axis is not supported")

    if len(set(tr_axes)) is not (grid.ndim - 1):
        raise ValueError("invalid transpose axes")

    # apply transpose to individual grid arrays and stack them
    return stack(
        [i_grid.transpose(axes=[idx - 1 for idx in tr_axes]) for i_grid in grid]
    )
