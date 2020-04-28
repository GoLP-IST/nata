# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

from nata.containers import GridDataset
from nata.plugins.register import register_container_plugin


@register_container_plugin(GridDataset, name="lineout")
def lineout_grid_dataset(
    dataset: GridDataset, fixed: Union[str, int], value: float,
) -> GridDataset:
    """Takes a lineout across a two-dimensional, single/multiple iteration\
       :class:`nata.containers.GridDataset`:

        Parameters
        ----------

        fixed: :class:``str`` or :class:``int``
            Selection of the axes along which the taken lineout is constant.

            * if it is a string, then it must match the ``name`` property of an
              existing grid axis in ``dataset``.
            * if it is an integer, then it must match the index of a grid axis
              in ``dataset`` (i.e. `0` or `1`).

        value: scalar
            Value between the minimum and maximum of the axes selected through
            ``fixed`` over which the lineout is taken.

        Returns
        ------
        :class:`nata.containers.GridDataset`:
            One-dimensional :class:`nata.containers.GridDataset`.

        Examples
        --------
        The following example shows how to obtain a lineout from a
        two-dimensional :class:`nata.containers.GridDataset`. Since no axes are
        attributed to the dataset in this example, they are automatically
        generated with no names, and ``fixed`` must be an integer.

        >>> from nata.containers import GridDataset
        >>> import numpy as np
        >>> arr = np.arange(25).reshape((5,5))
        >>> ds = GridDataset(arr[np.newaxis])
        >>> lo = ds.lineout(fixed=0, value=2)
        >>> lo.data
        array([10, 11, 12, 13, 14])

    """
    if len(dataset.grid_shape) != 2:
        raise ValueError(
            "Grid lineouts are only supported for two-dimensional grid datasets"
        )

    # get handle for grid axes
    axes = dataset.axes["grid_axes"]

    if isinstance(fixed, str):
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
    else:
        ax_idx = fixed

    # build axis values
    axis = axes[ax_idx]

    if value < np.min(axis) or value > np.max(axis):
        raise ValueError(f"Out of range value for fixed `{fixed}`")

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
