# -*- coding: utf-8 -*-
from typing import Optional
from typing import Sequence
from typing import Union
from warnings import warn

import numpy as np

from nata.containers import GridDataset
from nata.containers import ParticleDataset
from nata.containers import register_plugin
from nata.plots import Scale
from nata.plots import Theme
from nata.plots import Ticks
from nata.plots.kinds import PlotKind

Numbers = Union[int, float]


@register_plugin([GridDataset, ParticleDataset], name="plot")
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
