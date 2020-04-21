# -*- coding: utf-8 -*-
from typing import List
from typing import Optional

import numpy as np

from IPython.display import display
from ipywidgets import Layout
from ipywidgets import widgets
from nata.containers import GridDataset
from nata.plots import DefaultGridPlotTypes
from nata.plots.axes import Axes
from nata.plots.data import PlotData
from nata.plots.data import PlotDataAxis
from nata.plots.figure import Figure
from nata.plots.helpers import filter_style
from nata.plots.plans import AxesPlan
from nata.plots.plans import FigurePlan
from nata.plots.plans import PlotPlan
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
    return DefaultGridPlotTypes[dataset.ndim]


# @register_container_plugin(ParticleDataset, name="plot_data")
# def particle_plot_data(
#     dataset: ParticleDataset, quants: List[str] = []
# ) -> PlotData:

#     a = []
#     d = []

#     for quant in quants:
#         q = getattr(dataset, quant)
#         new_a = PlotDataAxis(name=q.name, label=q.label, units=q.unit)

#         a.append(new_a)
#         d.append(np.array(q))

#     p_d = PlotData(
#         name=dataset.name,
#         label=dataset.name,
#         units="",
#         data=d,
#         time=np.array(dataset.time),
#         time_units=dataset.time.unit,
#         axes=a,
#     )

#     return p_d

# @register_container_plugin(ParticleDataset, name="plot_type")
# def particle_plot_type(dataset: ParticleDataset) -> PlotData:
#     return DefaultParticlePlotType


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
        build_interactive_tools(f_plan, n)

    else:
        # build figure
        fig = build_figure(f_plan)
        return fig


# @register_container_plugin(ParticleDataset, name="plot")
# def plot_particle_dataset(
#     dataset: ParticleDataset,
#     quants: List[str] = [],
#     fig: Optional[Figure] = None,
#     axes: Optional[Axes] = None,
#     style: dict = dict(),
#     interactive: bool = True,
#     n: int = 0,
# ):

#     p_plan = PlotPlan(
#         dataset=dataset,
#         quants=quants,
#         style=filter_style(dataset.plot_type(), style),
#     )


#     if len(dataset.iteration) > 1 and inside_notebook() and interactive:
#         build_interactive_tools(f_plan, n)

#     else:
#         # build figure
#         fig = build_figure(f_plan)
#         return fig


# @register_container_plugin(DatasetCollection, name="plot")
# def plot_collection(
#     collection: DatasetCollection,
#     order: Optional[list] = list(),
#     styles: Optional[dict] = dict(),
#     quants: Optional[Dict[str, list]] = list(),
#     interactive: bool = True,
#     n: int = 0,
# ) -> Figure:

#     # check if collection is not empty
#     if not collection.store:
#         raise ValueError("Collection is empty.")

#     # check if time and iteration arrays are equal
#     for check in ["iteration", "time"]:
#         arr = [
#             np.array(getattr(dataset, check))
#             for dataset in collection.store.values()
#         ]
#         if not all(np.array_equal(arr[0], i) for i in arr):
#             raise ValueError(
#                 f"Attribute `{check}` is not the same for all datasets "
#                 + "in the collection."
#             )

#     f_a = []

#     for dataset in collection.store.values():
#         i_style = (
#             styles[dataset.name] if dataset.name in styles.keys() else None
#         )

#         if isinstance(dataset, ParticleDataset):
#             if dataset.name not in quants.keys():
#                 raise ValueError("quants not passed!")
#             i_quants = quants[dataset.name]
#         else:
#             i_quants = None

#         p_plan = PlotPlan(
#             dataset=dataset,
#             quants=i_quants,
#             style=filter_style(dataset.plot_type(), i_style),
#         )

#         a_plan = AxesPlan(
#             axes=None, plots=[p_plan], style=filter_style(Axes, i_style)
#         )

#         f_a.append(a_plan)

#     f_plan = FigurePlan(
#         fig=None,
#         axes=f_a,
#         style=filter_style(
#             Figure, styles["fig"] if "fig" in styles.keys() else None
#         ),
#     )

#     if len(dataset.iteration) > 1 and inside_notebook() and interactive:
#         build_interactive_tools(f_plan, n)

#     else:
#         # build figure
#         fig = build_figure(f_plan)
#         return fig


def build_figure(f_plan: FigurePlan) -> Figure:

    fig = f_plan.fig if f_plan.fig is not None else Figure(**f_plan.style)

    for a in f_plan.axes:
        axes = a.axes if a.axes is not None else fig.add_axes(style=a.style)

        for p in a.plots:
            plot_type = p.dataset.plot_type()
            plot_data = p.dataset.plot_data(quants=p.quants)
            axes.add_plot(plot_type=plot_type, data=plot_data, style=p.style)

        axes.update()

    fig.close()

    return fig


def build_interactive_tools(f_plan=FigurePlan, n: int = 0):
    # get reference dataset
    d_ref = f_plan.datasets[0]

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
                slider.options = [f"{i:n}" for i in np.arange(len(iteration))]
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

        for a in f_plan.axes:
            a_p = []
            for p in a.plots:
                a_p.append(
                    PlotPlan(
                        dataset=p.dataset[n], quants=p.quants, style=p.style
                    )
                )

            f_a.append(AxesPlan(axes=None, plots=a_p, style=a.style))

        f_p = FigurePlan(fig=None, axes=f_a, style=f_plan.style)

        # build figure
        fig = build_figure(f_plan=f_p)

        return fig.show()

    out = widgets.interactive_output(update_figure, {"sel": slider})

    display(ui, out)
