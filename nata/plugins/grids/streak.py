# -*- coding: utf-8 -*-
import numpy as np

from nata.containers import GridArray
from nata.containers import GridDataset
from nata.plugins.register import register_container_plugin


@register_container_plugin(GridDataset, name="streak")
def streak_grid_array(
    grid: GridDataset,
) -> GridArray:
    """Converts a `GridDataset` to a `GridArray`. Only `GridDataset` with axes
    that do not change over time are supported.

    Returns
    ------
    :class:`nata.containers.GridArray`:
        Streak of ``grid``.

    Examples
    --------
    Convert a one-dimensional dataset with time dependence to a two-dimensional
    array.

    >>> from nata.containers import GridDataset
    >>> import numpy as np
    >>> data = np.arange(5*7).reshape((5, 7))
    >>> grid = GridDataset.from_array(data)
    >>> stk_grid = grid.streak()
    >>> stk_grid.shape
    (5, 7)
    >>> [axis.shape for axis in stk_grid.axes]
    [(5,), (7,)]

    """

    if grid.ndim < 2:
        raise ValueError("streak is not available for 0 dimensional GridDatasets")

    for axis in grid.axes[1:]:
        for i, axis_i in enumerate(axis):
            if np.any(axis_i.to_dask() != axis[0].to_dask()):
                raise ValueError("invalid axes for streak")

    return GridArray.from_array(
        grid.to_dask(),
        name=grid.name,
        label=grid.label,
        unit=grid.unit,
        axes=[grid.time] + [axis[0] for axis in grid.axes[1:]],
    )
