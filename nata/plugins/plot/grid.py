# -*- coding: utf-8 -*-
from typing import Optional
from typing import Union
from warnings import warn

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
def grid_plot_data(dataset: GridDataset) -> PlotData:

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
    style: Optional[dict] = {},
    interactive: Optional[bool] = True,
    n: Optional[int] = 0,
) -> Union[Figure, None]:
    """Plots a single/multiple iteration :class:`nata.containers.GridDataset`\
       using a :class:`nata.plots.types.LinePlot` or\
       :class:`nata.plots.types.ColorPlot` if the dataset is one- or\
       two-dimensional, respectively.

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
            type (see :class:`nata.plots.types.LinePlot` or
            :class:`nata.plots.types.ColorPlot`).

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
        call the ``.plot()`` method of the dataset.

        >>> from nata.containers import GridDataset
        >>> import numpy as np
        >>> arr = np.arange(10)
        >>> ds = GridDataset.from_array(arr)
        >>> fig = ds.plot()

        In case a :class:`nata.plots.Figure` is returned by the method, it can
        be shown by calling the :func:`nata.plots.Figure.show` method.

        >>> fig.show()

        To draw a new plot on ``fig``, we can pass it as an argument to the
        ``.plot()`` method. If ``axes`` is provided, the new plot is drawn on
        the selected axes.

        >>> ds2 = GridDataset.from_array(arr**2)
        >>> fig = ds2.plot(fig=fig, axes=fig.axes[0])


    """

    p_plan = PlotPlan(
        dataset=dataset, style=filter_style(dataset.plot_type(), style)
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
