# -*- coding: utf-8 -*-
from typing import List
from typing import Optional

import numpy as np

from nata.containers import GridDataset
from nata.plots.axes import Axes
from nata.plots.data import PlotData
from nata.plots.data import PlotDataAxis
from nata.plots.figure import Figure
from nata.plots.helpers import filter_style
from nata.plots.plans import AxesPlan
from nata.plots.plans import FigurePlan
from nata.plots.plans import PlotPlan
from nata.plots.types import DefaultGridPlotTypes
from nata.plugins.register import register_container_plugin
from nata.utils.env import inside_notebook


@register_container_plugin(GridDataset, name="plot_data")
def grid_plot_data(dataset: GridDataset, quants: List[str] = []) -> PlotData:

    a = []

    for ds_a in dataset.axes["grid_axes"]:
        new_a = PlotDataAxis(
            name=ds_a.name,
            label=ds_a.label,
            units=ds_a.unit,
            data=np.array(ds_a),
        )

        a.append(new_a)

    d = PlotData(
        name=dataset.name,
        label=dataset.label,
        units=dataset.unit,
        data=np.array(dataset),
        time=np.array(dataset.axes["time"]),
        time_units=dataset.axes["time"].unit,
        axes=a,
    )

    return d


@register_container_plugin(GridDataset, name="plot_type")
def grid_plot_type(dataset: GridDataset) -> PlotData:
    return DefaultGridPlotTypes[len(dataset.grid_shape)]


@register_container_plugin(GridDataset, name="plot")
def plot_grid_dataset(
    dataset: GridDataset,
    fig: Optional[Figure] = None,
    axes: Optional[Axes] = None,
    style: dict = dict(),
    interactive: bool = True,
    n: int = 0,
):

    p_plan = PlotPlan(
        dataset=dataset, style=filter_style(dataset.plot_type(), style)
    )

    a_plan = AxesPlan(
        axes=axes, plots=[p_plan], style=filter_style(Axes, style)
    )

    f_plan = FigurePlan(
        fig=fig, axes=[a_plan], style=filter_style(Figure, style)
    )

    if len(dataset.axes["iteration"]) > 1 and inside_notebook() and interactive:
        f_plan.build_interactive(n)

    else:
        return f_plan.build()
