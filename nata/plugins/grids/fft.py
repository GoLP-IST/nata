# -*- coding: utf-8 -*-
from typing import Optional
from typing import Union

import numpy as np
import numpy.fft as fft

from nata.containers import Axis
from nata.containers import GridArray
from nata.containers import GridDataset
from nata.containers.grid import stack
from nata.plugins.register import register_container_plugin


def get_fft_axes(
    grid: Union[GridArray, GridDataset],
    axes: list,
    offset: int = 0,
):
    # build fft axes
    if axes is None:
        # no axes provided, all axes are fft axes
        fft_axes = [i for i in range(offset, len(grid.axes))]
    else:
        # axes provided, determine fft axes
        fft_axes = []

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

            fft_axes.append(idx)

    return fft_axes


@register_container_plugin(GridArray, name="fft")
def fft_grid_array(
    grid: GridArray,
    axes: Optional[list] = None,
    comp: str = "abs",
) -> GridArray:
    """Computes the Fast Fourier Transform (FFT) of a
    :class:`nata.containers.GridArray` using `numpy's fft module`__.

     .. _fft: https://numpy.org/doc/stable/reference/routines.fft.html
     __ fft_

     The axes over which the FFT is computed are transformed such that
     the zero frequency bins are centered.

     Parameters
     ----------
     axes: ``list``, optional
         List of integers and/or strings that identify the axes over which
         to compute the FFT.
     comp: ``{'abs', 'real', 'imag', 'full'}``, optional
         Defines the component of the FFT selected for output. Available
         values are  ``'abs'`` (default), ``'real'``, ``'imag'`` and
         ``'full'``, which correspond to the absolute value, real component,
         imaginary component and full (complex) result of the FFT,
         respectively.

     Returns
     ------
     :class:`nata.containers.GridArray`:
         Selected FFT component of ``grid``.

     Examples
     --------
     Obtain the FFT of a one-dimensional array.

     >>> from nata.containers import GridArray
     >>> import numpy as np
     >>> x = np.arange(100)
     >>> grid = GridArray.from_array(np.exp(-(x-len(x)/2)**2))
     >>> fft_grid = grid.fft()

     # TODO: add an image here with a plot of the FFT

     Compute the FFT over the first axis of a two-dimensional array.

     >>> from nata.containers import GridArray
     >>> import numpy as np
     >>> x = np.linspace(0, 10*np.pi)
     >>> y = np.linspace(0, 10*np.pi)
     >>> X, Y = np.meshgrid(x, y, indexing="ij")
     >>> grid = GridArray.from_array(np.sin(X) + np.sin(2*Y))
     >>> fft_grid = grid.fft(axes=[0])

     # TODO: add an image here with a plot of the FFT

    """

    if grid.ndim < 1:
        raise ValueError("fft is not available for 0 dimensional GridArrays")

    # build fft axes
    fft_axes = get_fft_axes(grid, axes)

    # build new axes
    new_axes = []

    for idx, axis in enumerate(grid.axes):
        if idx in fft_axes:
            # axis is fft axis, determine its fourier counterpart
            delta = (
                (np.max(axis.to_dask()) - np.min(axis.to_dask()))
                / axis.shape[-1]
                / (2.0 * np.pi)
            )

            axis_data = fft.fftshift(fft.fftfreq(axis.shape[-1], delta))

            new_axes.append(
                Axis(
                    axis_data,
                    name=f"k_{axis.name}",
                    label=f"k_{{{axis.label}}}",
                    unit=f"({axis.unit})^{{-1}}" if axis.unit else "",
                )
            )
        else:
            # axis is not fft axis, stays the same
            new_axes.append(axis)

    # do the data fft
    fft_data = fft.fftn(grid.to_dask(), axes=fft_axes)
    fft_data = fft.fftshift(fft_data, axes=fft_axes)

    # get only selected component
    if comp == "abs":
        fft_data = np.abs(fft_data)
        label = f"|{grid.label}|"
    elif comp == "real":
        fft_data = np.real(fft_data)
        label = f"\\Re({grid.label})"
    elif comp == "imag":
        fft_data = np.imag(fft_data)
        label = f"\\Im({grid.label})"
    elif comp == "full":
        label = f"{grid.label}"
    else:
        raise ValueError(f"invalid fft component '{comp}'")

    # build return grid
    return GridArray.from_array(
        fft_data,
        name=grid.name,
        label=label,
        unit=grid.unit,
        axes=new_axes,
    )


@register_container_plugin(GridDataset, name="fft")
def fft_grid_dataset(
    grid: GridDataset,
    axes: Optional[list] = None,
    comp: str = "abs",
) -> GridDataset:
    """Computes the Fast Fourier Transform (FFT) of a
    :class:`nata.containers.GridDataset` using `numpy's fft module`__. FFTs
    are not allowed along the time axis. In other words, `GridDataset` FFTs
    always preserve the time dependence. To do FFTs over the time axis,
    consider converting `grid` to a `GridArray` using the `streak()` plugin.

     .. _fft: https://numpy.org/doc/stable/reference/routines.fft.html
     __ fft_

     The axes over which the FFT is computed are transformed such that
     the zero frequency bins are centered.

     Parameters
     ----------
     axes: ``list``, optional
         List of integers and/or strings that identify the axes over which
         to compute the FFT. Must not include to the time axis.
     comp: ``{'abs', 'real', 'imag', 'full'}``, optional
         Defines the component of the FFT selected for output. Available
         values are  ``'abs'`` (default), ``'real'``, ``'imag'`` and
         ``'full'``, which correspond to the absolute value, real component,
         imaginary component and full (complex) result of the FFT,
         respectively.

     Returns
     ------
     :class:`nata.containers.GridDataset`:
         Selected FFT component of ``grid``.

     Examples
     --------
     Obtain the FFT of a one-dimensional dataset with time dependence.

     >>> from nata.containers import GridDataset
     >>> import numpy as np
     >>> time = np.arange(3)
     >>> x = np.linspace(0, 10*np.pi, 101)
     >>> grid = GridDataset.from_array(
             [np.sin(x), np.sin(2*x), np.sin(3*x)],
             axes=[
                 Axis(time),
                 Axis(np.tile(x, (3, 1)))
             ]
         )
     >>> fft_grid = grid.fft()

     # TODO: add an image here with a plot of the FFT

    """

    if grid.ndim < 2:
        raise ValueError("fft is not available for 0 dimensional GridDatasets")

    # build fft axes
    fft_axes = get_fft_axes(grid, axes, 1)

    if 0 in fft_axes:
        raise ValueError("fft along the time axis is not supported")

    # apply fft to individual grid arrays and stack them
    return stack(
        [i_grid.fft(axes=[idx - 1 for idx in fft_axes], comp=comp) for i_grid in grid]
    )
