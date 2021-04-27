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
    """Reverses or permutes the axes of a `GridArray`.

    Parameters
    ----------
    axes: ``list``, optional
         List of integers and/or strings that identify the permutation of the
         axes. The i'th axis of the returned `GridArray` will correspond to the
         axis numbered/labeled axes[i] of the input. If not specified, the
         order of the axes is reversed.

    Returns
    ------
    :class:`nata.containers.GridArray`:
        Transpose of ``grid``.

    Examples
    --------
    Transpose a three-dimensional array.

    >>> from nata.containers import GridArray
    >>> import numpy as np
    >>> data = np.arange(96).reshape((8, 4, 3))
    >>> grid = GridArray.from_array(data)
    >>> grid.transpose().shape
    (3, 4, 8)
    >>> grid.transpose(axes=[0,2,1]).shape
    (8, 3, 4)

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
    """Reverses or permutes the axes of a `GridDataset`. Transpose is not
    allowed along the time axis. In other words, `GridDataset` tranposes always
    preserve the time dependence. To do tranposes over the time axis, consider
    converting `grid` to a `GridArray` using the `streak()` plugin.

    Parameters
    ----------
    axes: ``list``, optional
         List of integers and/or strings that identify the permutation of the
         axes. The i'th axis of the returned `GridArray` will correspond to the
         axis numbered/labeled axes[i] of the input. If not specified, the
         order of the axes is reversed. Must not include to the time axis.

    Returns
    ------
    :class:`nata.containers.GridDataset`:
        Transpose of ``grid``.

    Examples
    --------
    Transpose a three-dimensional dataset with time dependence.

    >>> from nata.containers import GridDataset
    >>> import numpy as np
    >>> data = np.arange(8*6*4*2).reshape((8, 6, 4, 2))
    >>> grid = GridDataset.from_array(data)
    >>> grid.transpose().shape
    (8, 2, 4, 6)
    >>> grid.transpose(axes=[1,3,2]).shape
    (8, 6, 2, 4)

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
