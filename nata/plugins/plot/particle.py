# -*- coding: utf-8 -*-
from typing import Optional
from typing import Sequence
from typing import Union
from warnings import warn

import numpy as np

from nata.containers import ParticleArray
from nata.containers import ParticleDataset
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
from nata.plugins.register import register_container_plugin

Numbers = Union[int, float]


def default_plot_kind(data):
    return Scatter()

def is_valid_plot_kind(data, kind):
    return isinstance(kind, Scatter)

@register_container_plugin(ParticleArray, name="plot")
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
            if not ( 
                isinstance(kind.color, (Sequence, np.ndarray)) and
                len(kind.color) == data.count
            ) and not (
                isinstance(kind.color, Sequence) and 
                len(kind.color) in (3, 4)
            ) and not isinstance(kind.color, str):
                raise ValueError("invalid scatter color")

        # when a color that requires a colorbar is provided
        if (
            kind.color is not None and 
            isinstance(kind.color, (Sequence, np.ndarray)) and
            len(kind.color) == data.count and
            not kind.colorbar
        ):
            kind.colorbar = Colorbar()

        # when a color is not provided and can be computed from quantities
        if (
            kind.color is None and 
            len(data.quantities) > 2
        ):
            kind.color = data[:, data.quantity_names[2]].to_numpy()

            if not kind.colorbar:
                kind.colorbar = Colorbar(
                    label=f"{data.quantity_labels[2]}" + (f" [{data.quantity_units[2]}]" if data.quantity_units[2] else "")
                )

        if isinstance(kind.colorscale, str):
            kind.colorscale = scale_from_str(kind.colorscale)

        if kind.colorbar and isinstance(kind.colorbar.ticks, Sequence):
            kind.colorbar.ticks = Ticks(values=kind.colorbar.ticks)
        
        fig.scatter(data[:, data.quantity_names[0]].to_numpy(), data[:, data.quantity_names[1]].to_numpy(), **kind.to_dict())

    fig.xlabel = xlabel or f"{data.quantity_labels[0]}" + (
        f" [{data.quantity_units[0]}]" if data.quantity_units[0] else ""
    )
    fig.ylabel = ylabel or f"{data.quantity_labels[1]}" + (
        f" [{data.quantity_units[1]}]" if data.quantity_units[1] else ""
    )

    fig.close()

    return fig

@register_container_plugin(ParticleDataset, name="plot")
def plot_dataset(
    data,
    start: int = 0,
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
    try:
        from IPython.display import display
        from ipywidgets import Layout
        from ipywidgets import widgets

        interact = True
    except ImportError:
        interact = False

    if interact:

        if start not in range(len(data)):
            raise ValueError("invalid start index")

        dropdown = widgets.Dropdown(
            options=["Index", data.time.label],
            value="Index",
            disabled=False,
            layout=Layout(max_width="100px"),
            continuous_update=False,
        )

        slider = widgets.SelectionSlider(
            options=[f"{i}" for i in range(len(data))],
            value=f"{start}",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
        )

        def dropdown_change(change):
            if change.old in [data.time.label]:
                options = np.array(slider.options).astype(np.float)
                n = np.argmax(options >= float(slider.value)).item()
            else:
                n = int(slider.value)

            with out.hold_trait_notifications():
                if change.new == data.time.label:
                    slider.options = [f"{i:.2f}" for i in data.time.to_numpy()]
                    slider.value = f"{data.time[n].to_numpy():.2f}"

                else:
                    slider.options = [f"{i:n}" for i in range(len(data))]
                    slider.value = f"{n:d}"

        dropdown.observe(dropdown_change, names=["value"], type="change")

        ui = widgets.HBox([dropdown, slider])

        def update_figure(sel):
            if dropdown.value == data.time.label:
                n = np.argmax(data.time.to_numpy() >= float(sel)).item()
            else:
                n = int(sel)

            fig = data[n].plot(
                xrange=xrange,
                yrange=yrange,
                xscale=xscale,
                yscale=yscale,
                xlabel=xlabel,
                ylabel=ylabel,
                xticks=xticks,
                yticks=yticks,
                title=title,
                aspect=aspect,
                size=size,
                theme=theme,
                kind=kind,
            )

            return fig.show()

        out = widgets.interactive_output(update_figure, {"sel": slider})

        display(ui, out)

    else:
        warn("interactivity unavailable, please run `pip install nata[jupyter]`")
        return data[start].plot(
            xrange=xrange,
            yrange=yrange,
            xscale=xscale,
            yscale=yscale,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
            yticks=yticks,
            title=title,
            aspect=aspect,
            size=size,
            theme=theme,
            kind=kind,
        )