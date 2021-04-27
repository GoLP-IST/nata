# -*- coding: utf-8 -*-
import numpy as np

from nata.containers import GridArray
from nata.containers import GridDataset
from nata.plugins.register import register_container_plugin


@register_container_plugin(GridDataset, name="streak")
def slice_grid_array(
    grid: GridDataset,
) -> GridArray:

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
