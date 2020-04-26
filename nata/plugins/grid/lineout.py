# -*- coding: utf-8 -*-
import numpy as np

from nata.containers import GridDataset
from nata.plugins.register import register_container_plugin


@register_container_plugin(GridDataset, name="lineout")
def lineout(dataset: GridDataset, fixed: str, value: float,) -> GridDataset:

    if dataset.ndim != 2:
        raise ValueError(
            "Grid lineouts are only supported for two-dimensional grid datasets"
        )

    # get handle for grid axes
    axes = dataset.axes["grid_axes"]

    ax_idx = -1
    # get index based on
    for key, ax in enumerate(axes):
        if ax.name == fixed:
            ax_idx = key
            break
    if ax_idx < 0:
        raise ValueError(
            f"Axis `{fixed}` could not be found in dataset `{dataset}`"
        )

    # build axis values
    axis = axes[ax_idx]

    if value < np.min(axis) or value > np.max(axis):
        raise ValueError(f"Out of range value for `{fixed}`")

    values = np.array(axis)
    idx = (np.abs(values - value)).argmin()

    data = np.array(dataset)

    # get lineout
    if ax_idx == 0:
        lo_data = data[:, idx, :] if len(dataset) > 1 else data[idx, :]
        lo_axis = axes[1]

    elif ax_idx == 1:
        lo_data = data[:, :, idx] if len(dataset) > 1 else data[:, idx]
        lo_axis = axes[0]

    return GridDataset(
        lo_data if len(dataset) > 1 else lo_data[np.newaxis],
        name=dataset.name,
        label=dataset.label,
        unit=dataset.unit,
        grid_axes=[lo_axis],
        time=dataset.axes["time"],
        iteration=dataset.axes["iteration"],
    )
