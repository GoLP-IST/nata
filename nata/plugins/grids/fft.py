# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
import numpy.fft as fft

from nata.axes import GridAxis
from nata.containers import GridDataset
from nata.plugins.register import register_container_plugin


@register_container_plugin(GridDataset, name="fft")
def fft_grid_dataset(
    dataset: GridDataset, type: Optional[str] = "abs",
) -> GridDataset:
    """Computes the Fast Fourier Transform (FFT) of a single/multiple\
       iteration :class:`nata.containers.GridDataset` along all grid axes\
       using `numpy's fft module`__.

        .. _fft: https://numpy.org/doc/stable/reference/routines.fft.html
        __ fft_

        Parameters
        ----------
        type: ``{'abs', 'real', 'imag', 'full'}``, optional
            Defines the component of the FFT selected for output. Available
            values are  ``'abs'`` (default), ``'real'``, ``'imag'`` and
            ``'full'``, which correspond to the absolute value, real component,
            imaginary component and full (complex) result of the FFT,
            respectively.

        Returns
        ------
        :class:`nata.containers.GridDataset`:
            Selected FFT component along all grid axes of ``dataset``.

        Examples
        --------
        To obtain the FFT of a :class:`nata.containers.GridDataset`, a simple
        call to the ``fft()`` method is enough. In the following example, we
        compute the FFT of a one-dimensional
        :class:`nata.containers.GridDataset`.

        >>> from nata.containers import GridDataset
        >>> import numpy as np
        >>> x = np.linspace(100)
        >>> arr = np.exp(-(x-len(x)/2)**2)
        >>> ds = GridDataset(arr[np.newaxis])
        >>> ds_fft = ds.fft()

    """

    fft_data = np.array(dataset)
    fft_axes = np.arange(len(dataset.grid_shape)) + 1

    fft_data = fft.fftn(
        fft_data if len(dataset) > 1 else fft_data[np.newaxis], axes=fft_axes
    )
    fft_data = fft.fftshift(fft_data, axes=fft_axes)

    if type == "real":
        fft_data = np.real(fft_data)
        label = f"Re(FFT({dataset.label}))"
        name = f"fftr_{dataset.name}"
    elif type == "imag":
        fft_data = np.imag(fft_data)
        label = f"Im(FFT({dataset.label}))"
        name = f"ffti_{dataset.name}"
    elif type == "abs":
        fft_data = np.abs(fft_data)
        label = f"|FFT({dataset.label})|"
        name = f"ffta_{dataset.name}"
    else:
        label = f"FFT({dataset.label})"
        name = f"fft_{dataset.name}"

    axes = []
    for a in dataset.axes["grid_axes"]:
        delta = [
            (np.max(a_ts) - np.min(a_ts)) / len(np.array(a_ts)) / (2.0 * np.pi)
            for a_ts in a
        ]
        axis_data = fft.fftshift(
            [
                fft.fftfreq(len(np.array(a_ts)), delta[idx])
                for idx, a_ts in enumerate(a)
            ],
            axes=-1,
        )
        axes.append(
            GridAxis(
                axis_data,
                name=f"k_{a.name}",
                label=f"k_{{{a.label}}}",
                unit=f"1/({a.unit})",
            )
        )

    return GridDataset(
        fft_data,
        name=name,
        label=label,
        unit=dataset.unit,
        grid_axes=axes,
        time=dataset.axes["time"],
        iteration=dataset.axes["iteration"],
    )
