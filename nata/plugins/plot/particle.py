# -*- coding: utf-8 -*-
from typing import List
from typing import Optional

import numpy as np

from nata.containers import ParticleDataset
from nata.plots.axes import Axes
from nata.plots.data import PlotData
from nata.plots.data import PlotDataAxis
from nata.plots.figure import Figure
from nata.plots.helpers import filter_style
from nata.plots.plans import AxesPlan
from nata.plots.plans import FigurePlan
from nata.plots.plans import PlotPlan
from nata.plots.types import DefaultParticlePlotType
from nata.plugins.register import register_container_plugin
from nata.utils.env import inside_notebook


@register_container_plugin(ParticleDataset, name="plot_data")
def particle_plot_data(
    dataset: ParticleDataset, quants: List[str] = []
) -> PlotData:
    a = []
    d = []

    for quant in quants:
        q = dataset.quantities[quant]
        new_a = PlotDataAxis(name=q.name, label=q.label, units=q.unit)

        a.append(new_a)
        d.append(np.array(q))

    p_d = PlotData(
        name=dataset.name,
        label=dataset.name,
        units="",
        data=d,
        time=np.array(dataset.axes["time"]),
        time_units=dataset.axes["time"].unit,
        axes=a,
    )

    return p_d


@register_container_plugin(ParticleDataset, name="plot_type")
def particle_plot_type(dataset: ParticleDataset) -> PlotData:
    return DefaultParticlePlotType


@register_container_plugin(ParticleDataset, name="plot")
def plot_particle_dataset(
    dataset: ParticleDataset,
    quants: List[str] = [],
    fig: Optional[Figure] = None,
    axes: Optional[Axes] = None,
    style: dict = dict(),
    interactive: bool = True,
    n: int = 0,
):

    p_plan = PlotPlan(
        dataset=dataset,
        quants=quants,
        style=filter_style(dataset.plot_type(), style),
    )

    a_plan = AxesPlan(
        axes=axes, plots=[p_plan], style=filter_style(Axes, style)
    )

    f_plan = FigurePlan(
        fig=fig, axes=[a_plan], style=filter_style(Figure, style)
    )

    if len(dataset) > 1 and inside_notebook() and interactive:
        f_plan.build_interactive(n)

    else:
        return f_plan.build()
