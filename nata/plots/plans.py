# -*- coding: utf-8 -*-
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from IPython.display import display
from ipywidgets import Layout
from ipywidgets import widgets
from nata.containers import GridDataset
from nata.plots.axes import Axes
from nata.plots.figure import Figure


@dataclass
class PlotPlan:
    dataset: GridDataset = object()
    style: Optional[Dict] = field(default_factory=dict)


@dataclass
class AxesPlan:
    plots: List[PlotPlan] = field(default_factory=list)
    axes: Optional[Axes] = object()
    style: Optional[Dict] = field(default_factory=dict)

    @property
    def datasets(self) -> list:
        ds = []
        for p in self.plots:
            ds.append(p.dataset)
        return ds


@dataclass
class FigurePlan:
    axes: List[AxesPlan] = field(default_factory=list)
    fig: Optional[Figure] = object()
    style: Optional[Dict] = field(default_factory=dict)

    @property
    def datasets(self) -> list:
        ds = []
        for a in self.axes:
            for d in a.datasets:
                ds.append(d)
        return ds

    def __getitem__(self, key: Union[int, slice]):
        axes = []
        for a in self.axes:
            plots = []
            for p in a.plots:
                plots.append(PlotPlan(dataset=p.dataset[key], style=p.style,))
            axes.append(AxesPlan(plots=plots, axes=a.axes, style=a.style,))

        return self.__class__(axes=axes, fig=self.fig, style=self.style,)

    def build(self) -> Figure:
        fig = self.fig if self.fig is not None else Figure(**self.style)

        for a in self.axes:
            axes = a.axes if a.axes is not None else fig.add_axes(style=a.style)

            for p in a.plots:
                plot_type = p.dataset.plot_type()
                plot_data = p.dataset.plot_data()
                axes.add_plot(
                    plot_type=plot_type, data=plot_data, style=p.style
                )

            axes.update()

        fig.close()

        return fig

    def build_interactive(self, n: int = 0):
        # get reference dataset
        d_ref = self.datasets[0]

        time = np.array(d_ref.axes["time"])
        iteration = np.array(d_ref.axes["iteration"])

        dropdown = widgets.Dropdown(
            options=["File Index", "Iteration", "Time"],
            value="File Index",
            disabled=False,
            layout=Layout(max_width="100px"),
            continuous_update=False,
        )

        slider = widgets.SelectionSlider(
            options=[f"{i}" for i in np.arange(len(iteration))],
            value=f"{n}",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
        )

        def dropdown_change(change):

            if change.old in ["Time", "Iteration"]:
                options = np.array(slider.options).astype(np.float)
                n = np.argmax(options >= float(slider.value)).item()
            else:
                n = int(slider.value)

            with out.hold_trait_notifications():
                if change.new == "Time":
                    slider.options = [f"{i:.2f}" for i in time]
                    slider.value = f"{time[n]:.2f}"

                elif change.new == "Iteration":
                    slider.options = [f"{i:d}" for i in iteration]
                    slider.value = f"{iteration[n]:d}"
                else:
                    slider.options = [
                        f"{i:n}" for i in np.arange(len(iteration))
                    ]
                    slider.value = f"{n:d}"

        dropdown.observe(dropdown_change, names=["value"], type="change")

        ui = widgets.HBox([dropdown, slider])

        def update_figure(sel):
            if dropdown.value == "Time":
                n = np.argmax(time >= float(sel)).item()
            elif dropdown.value == "Iteration":
                n = np.argmax(iteration >= int(sel)).item()
            else:
                n = int(sel)

            f_a = []

            for a in self.axes:
                a_p = []
                for p in a.plots:
                    a_p.append(PlotPlan(dataset=p.dataset[n], style=p.style))

                f_a.append(AxesPlan(axes=None, plots=a_p, style=a.style))

            f_p = FigurePlan(fig=None, axes=f_a, style=self.style)

            # build figure
            fig = f_p.build()

            return fig.show()

        out = widgets.interactive_output(update_figure, {"sel": slider})

        display(ui, out)
