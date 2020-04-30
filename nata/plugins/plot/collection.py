# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np

from nata.containers import DatasetCollection
from nata.plots.axes import Axes
from nata.plots.figure import Figure
from nata.plots.helpers import filter_style
from nata.plots.plans import AxesPlan
from nata.plots.plans import FigurePlan
from nata.plots.plans import PlotPlan
from nata.plugins.register import register_container_plugin
from nata.utils.env import inside_notebook


@register_container_plugin(DatasetCollection, name="plot")
def plot_collection(
    collection: DatasetCollection,
    order: Optional[list] = list(),
    styles: Optional[dict] = dict(),
    interactive: bool = True,
    n: int = 0,
) -> Figure:

    # check if collection is not empty
    if not collection.store:
        raise ValueError("Collection is empty.")

    # check if time and iteration arrays are equal
    for check in ["iteration", "time"]:
        arr = [
            np.array(dataset.axes[check])
            for dataset in collection.store.values()
        ]
        if not all(np.array_equal(arr[0], i) for i in arr):
            raise ValueError(
                f"Attribute `{check}` is not the same for all datasets "
                + "in the collection."
            )

    f_a = []

    for dataset in collection.store.values():
        i_style = (
            styles[dataset.name] if dataset.name in styles.keys() else None
        )

        p_plan = PlotPlan(
            dataset=dataset, style=filter_style(dataset.plot_type(), i_style),
        )

        a_plan = AxesPlan(
            axes=None, plots=[p_plan], style=filter_style(Axes, i_style)
        )

        f_a.append(a_plan)

    f_plan = FigurePlan(
        fig=None,
        axes=f_a,
        style=filter_style(
            Figure, styles["fig"] if "fig" in styles.keys() else None
        ),
    )

    if len(dataset) > 1 and inside_notebook() and interactive:
        f_plan.build_interactive(n)

    else:
        return f_plan.build()
