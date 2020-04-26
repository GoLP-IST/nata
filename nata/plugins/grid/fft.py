# -*- coding: utf-8 -*-
import numpy as np
import numpy.fft as fft

from nata.axes import GridAxis
from nata.containers import GridDataset
from nata.plugins.register import register_container_plugin


@register_container_plugin(GridDataset, name="fft")
def fft_grid_dataset(dataset: GridDataset, type: str = "abs",) -> GridDataset:

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
