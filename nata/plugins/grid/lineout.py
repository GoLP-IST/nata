# -*- coding: utf-8 -*-
import numpy as np

from nata.containers import GridDataset
from nata.plugins.register import register_container_plugin


@register_container_plugin(GridDataset, name="lineout")
def lineout(dataset: GridDataset, fixed: str, value: float,) -> GridDataset:

    if dataset.grid_dim != 2:
        raise ValueError(
            "Grid lineouts are only supported for two-dimensional grid datasets"
        )

    ax_idx = -1
    # get index based on
    for key, ax in enumerate(dataset.axes):
        if ax.name == fixed:
            ax_idx = key
            break
    if ax_idx < 0:
        raise ValueError(
            f"Axis `{fixed}` could not be found in dataset `{dataset}`"
        )

    # build axis values
    axis = dataset.axes[ax_idx]

    if value < np.min(axis) or value > np.max(axis):
        raise ValueError(f"Out of range value for `{fixed}`")

    values = np.array(axis)
    idx = (np.abs(values - value)).argmin()

    data = np.array(dataset)

    # get lineout
    if ax_idx == 0:
        l_data = data[:, idx, :] if len(dataset.iteration) > 2 else data[idx, :]
        l_axis = dataset.axes[1]

    elif ax_idx == 1:
        l_data = data[:, :, idx] if len(dataset.iteration) > 2 else data[:, idx]
        l_axis = dataset.axes[0]

    lo = GridDataset.from_array(
        l_data,
        name=dataset.name,
        label=dataset.label,
        unit=dataset.unit,
        axes_names=[l_axis.name],
        axes_min=[np.min(l_axis)],
        axes_max=[np.max(l_axis)],
        axes_labels=[l_axis.label],
        axes_units=[l_axis.unit],
        iteration=np.array(dataset.iteration),
        time=np.array(dataset.time),
        time_unit=dataset.time.unit,
    )

    return lo
