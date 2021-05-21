# -*- coding: utf-8 -*-
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np

from nata.containers import ParticleArray
from nata.containers import register_plugin
from nata.plots import Colorbar
from nata.plots import Figure
from nata.plots import Scale
from nata.plots import Scatter
from nata.plots import Theme
from nata.plots import Ticks
from nata.plots.elements import scale_from_str
from nata.plots.kinds import PlotKind

Numbers = Union[int, float]


def default_plot_kind(data):
    return Scatter()


def is_valid_plot_kind(data, kind):
    return isinstance(kind, Scatter)


@register_plugin(ParticleArray, name="plot")
def plot_particle_array(
    data,
    xrange: Optional[Sequence[Numbers]] = None,
    yrange: Optional[Sequence[Numbers]] = None,
    xscale: Optional[Union[Scale, str]] = None,
    yscale: Optional[Union[Scale, str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xticks: Optional[Union[Ticks, Sequence[Numbers]]] = None,
    yticks: Optional[Union[Ticks, Sequence[Numbers]]] = None,
    title: Optional[str] = None,
    aspect: Optional[str] = None,
    size: Optional[Sequence[Numbers]] = None,
    theme: Optional[Union[Theme, str]] = None,
    kind: Optional[PlotKind] = None,
):

    if kind and not is_valid_plot_kind(data, kind):
        raise ValueError(f"invalid plot kind for {type(data)} `{data.name}`")

    kind = kind if kind else default_plot_kind(data)

    if isinstance(theme, str) or theme is None:
        theme = Theme(name=theme or "light")

    if title is None and not (data.time == None).to_numpy():
        title = f"{data.time.label} = {data.time.to_numpy()}" + (
            f" [{data.time.unit}]" if data.time.unit else ""
        )

    if isinstance(xscale, str):
        xscale = scale_from_str(xscale)

    if isinstance(yscale, str):
        yscale = scale_from_str(yscale)

    if isinstance(xticks, Sequence):
        xticks = Ticks(values=xticks)

    if isinstance(yticks, Sequence):
        yticks = Ticks(values=yticks)

    fig = Figure(
        size=size,
        aspect=aspect,
        theme=theme,
        xrange=xrange,
        yrange=yrange,
        xscale=xscale,
        yscale=yscale,
        xticks=xticks,
        yticks=yticks,
        title=title,
    )

    if isinstance(kind, Scatter):

        # check that color is either:
        # - an array with the right size
        # - an array with size 3/4 representing an RGB/RGBA value
        # - a string
        if kind.color is not None:
            if (
                not (
                    isinstance(kind.color, (Sequence, np.ndarray))
                    and len(kind.color) == data.count
                )
                and not (isinstance(kind.color, Sequence) and len(kind.color) in (3, 4))
                and not isinstance(kind.color, str)
            ):
                raise ValueError("invalid scatter color")

        # when a color that requires a colorbar is provided
        if (
            kind.color is not None
            and isinstance(kind.color, (Sequence, np.ndarray))
            and len(kind.color) == data.count
            and not kind.colorbar
        ):
            kind.colorbar = Colorbar()

        # when a color is not provided and can be computed from quantities
        if kind.color is None and len(data.quantities) > 2:
            kind.color = data[:, data.quantity_names[2]].to_numpy()

            if not kind.colorbar:
                kind.colorbar = Colorbar(
                    label=f"{data.quantity_labels[2]}"
                    + (f" [{data.quantity_units[2]}]" if data.quantity_units[2] else "")
                )

        if isinstance(kind.colorscale, str):
            kind.colorscale = scale_from_str(kind.colorscale)

        if kind.colorbar and isinstance(kind.colorbar.ticks, Sequence):
            kind.colorbar.ticks = Ticks(values=kind.colorbar.ticks)

        fig.scatter(
            data[:, data.quantity_names[0]].to_numpy(),
            data[:, data.quantity_names[1]].to_numpy(),
            **kind.to_dict(),
        )

    fig.xlabel = xlabel or f"{data.quantity_labels[0]}" + (
        f" [{data.quantity_units[0]}]" if data.quantity_units[0] else ""
    )
    fig.ylabel = ylabel or f"{data.quantity_labels[1]}" + (
        f" [{data.quantity_units[1]}]" if data.quantity_units[1] else ""
    )

    fig.close()

    return fig
