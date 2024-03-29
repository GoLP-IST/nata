# -*- coding: utf-8 -*-
from typing import Optional
from typing import Sequence
from typing import Union

from nata.containers import GridArray
from nata.containers import register_plugin
from nata.plots import Colorbar
from nata.plots import Figure
from nata.plots import Image
from nata.plots import Line
from nata.plots import Scale
from nata.plots import Scatter
from nata.plots import Theme
from nata.plots import Ticks
from nata.plots.elements import scale_from_str
from nata.plots.kinds import PlotKind

Numbers = Union[int, float]


def default_plot_kind(data):
    if isinstance(data, GridArray):
        if data.ndim == 1:
            return Line()
        elif data.ndim == 2:
            return Image()
        else:
            raise NotImplementedError


def is_valid_plot_kind(data, kind):
    if isinstance(data, GridArray):
        if data.ndim == 1:
            return isinstance(kind, (Line, Scatter))
        elif data.ndim == 2:
            return isinstance(kind, (Image,))
        else:
            raise NotImplementedError


@register_plugin(GridArray, name="plot")
def plot_data_array(
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

    if data.ndim == 1:
        if isinstance(kind, Line):
            fig.line(data.axes[0].to_numpy(), data.to_numpy(), **kind.to_dict())
        elif isinstance(kind, Scatter):
            fig.scatter(data.axes[0].to_numpy(), data.to_numpy(), **kind.to_dict())

        fig.xlabel = xlabel or f"{data.axes[0].label}" + (
            f" [{data.axes[0].unit}]" if data.axes[0].unit else ""
        )
        fig.ylabel = ylabel or f"{data.label}" + (
            f" [{data.unit}]" if data.unit else ""
        )

    elif data.ndim == 2:

        if not kind.colorbar:
            kind.colorbar = Colorbar(
                label=f"{data.label}" + (f" [{data.unit}]" if data.unit else "")
            )

        if isinstance(kind.colorbar.ticks, Sequence):
            kind.colorbar.ticks = Ticks(values=kind.colorbar.ticks)

        if isinstance(kind, Image):
            fig.image(
                data.axes[0].to_numpy(),
                data.axes[1].to_numpy(),
                data.to_numpy(),
                **kind.to_dict(),
            )

        fig.xlabel = xlabel or f"{data.axes[0].label}" + (
            f" [{data.axes[0].unit}]" if data.axes[0].unit else ""
        )
        fig.ylabel = ylabel or f"{data.axes[1].label}" + (
            f" [{data.axes[1].unit}]" if data.axes[0].unit else ""
        )

    else:
        raise NotImplementedError

    fig.close()

    return fig
