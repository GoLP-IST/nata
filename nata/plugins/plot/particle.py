# -*- coding: utf-8 -*-
from typing import Optional
from warnings import warn

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
def particle_plot_data(dataset: ParticleDataset) -> PlotData:
    a = []
    d = []

    for quant in dataset.quantities.values():
        new_a = PlotDataAxis(
            name=quant.name, label=quant.label, units=quant.unit
        )

        a.append(new_a)
        d.append(np.array(quant))

    return PlotData(
        name=dataset.name,
        label=dataset.name,
        units="",
        data=d,
        time=np.array(dataset.axes["time"]),
        time_units=dataset.axes["time"].unit,
        axes=a,
    )


@register_container_plugin(ParticleDataset, name="plot_type")
def particle_plot_type(dataset: ParticleDataset) -> PlotData:
    return DefaultParticlePlotType


@register_container_plugin(ParticleDataset, name="plot")
def plot_particle_dataset(
    dataset: ParticleDataset,
    fig: Optional[Figure] = None,
    axes: Optional[Axes] = None,
    style: dict = dict(),
    interactive: bool = True,
    n: int = 0,
):
    """Plots a single/multiple iteration :class:`nata.containers.ParticleDataset`\
       using a :class:`nata.plots.types.ScatterPlot`.

        Parameters
        ----------
        fig: :class:`nata.plots.Figure`, optional
            If provided, the plot is drawn on ``fig``. The plot is drawn on
            ``axes`` if it is a child axes of ``fig``, otherwise a new axes
            is created on ``fig``. If ``fig`` is not provided, a new
            :class:`nata.plots.Figure` is created.

        axes: :class:`nata.plots.Axes`, optional
            If provided, the plot is drawn on ``axes``, which must be an axes
            of ``fig``. If ``axes`` is not provided or is provided without a
            corresponding ``fig``, a new :class:`nata.plots.Axes` is created in
            a new :class:`nata.plots.Figure`.

        style: ``dict``, optional
            Dictionary that takes a mix of style properties of
            :class:`nata.plots.Figure`, :class:`nata.plots.Axes` and any plot
            type (see :class:`nata.plots.types.ScatterPlot`).

        interactive: ``bool``, optional
            Controls wether interactive widgets should be shown with the plot
            to allow for temporal navigation. Only applicable if ``dataset``
            has multiple iterations.

        n: ``int``, optional
            Selects the index of the iteration to be shown initially. Only
            applicable if ``dataset`` has multiple iterations, .

        Returns
        ------
        :class:`nata.plots.Figure` or ``None``:
            Figure with plot built based on ``dataset``. Interactive widgets
            are shown with the figure if ``dataset`` has multiple iterations,
            in which case this method returns  ``None``.

        Examples
        --------
        To get a plot with default style properties in a new figure, simply
        call the ``.plot()`` method. The first two quantities in the dataset
        ``quantities`` dictionary will be represented in the horizontal and
        vertical plot axes, respectively. If a third quantity is available, it
        will be represented in colors.

        >>> from nata.containers import ParticleDataset
        >>> import numpy as np
        >>> arr = np.arange(30).reshape(1,10,3)
        >>> ds = ParticleDataset("path/to/file")
        >>> fig = ds.plot()

        The list of quantities in the dataset can be filtered with the
        :meth:`nata.containers.ParticleDataset.filter` method.

        >>> fig = ds.filter(quantities=["x1", "p1", "ene"]).plot()

    """

    p_plan = PlotPlan(
        dataset=dataset, style=filter_style(dataset.plot_type(), style),
    )

    a_plan = AxesPlan(
        axes=axes, plots=[p_plan], style=filter_style(Axes, style)
    )

    f_plan = FigurePlan(
        fig=fig, axes=[a_plan], style=filter_style(Figure, style)
    )

    if len(dataset) > 1:
        if inside_notebook():
            if interactive:
                f_plan.build_interactive(n)
            else:
                return f_plan[n].build()
        else:
            # TODO: remove last line from warn
            warn(
                f"Plotting only iteration with index n={str(n)}."
                + " Interactive plots of multiple iteration datasets are not"
                + " supported outside notebook environments."
            )
            return f_plan[n].build()

    else:
        return f_plan.build()
