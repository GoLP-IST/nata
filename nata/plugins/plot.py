# -*- coding: utf-8 -*-
from typing import Dict
from typing import List
from typing import Optional

import attr
import numpy as np
from attr.validators import instance_of
from attr.validators import optional

from IPython.display import display
from ipywidgets import Layout
from ipywidgets import widgets
from nata.containers import BaseDataset
from nata.containers import DatasetCollection
from nata.containers import GridDataset
from nata.containers import ParticleDataset
from nata.plots import DefaultGridPlotTypes
from nata.plots import DefaultParticlePlotType
from nata.plots.axes import Axes
from nata.plots.data import PlotData
from nata.plots.data import PlotDataAxis
from nata.plots.figure import Figure
from nata.plugins.register import register_container_plugin
from nata.utils.attrs import filter_style
from nata.utils.env import inside_notebook


@register_container_plugin(GridDataset, name="plot_data")
def grid_plot_data(dataset: GridDataset, quants: List[str] = []) -> PlotData:

    a = []

    for ds_a in dataset.axes:
        new_a = PlotDataAxis(
            name=ds_a.name,
            label=ds_a.label,
            units=ds_a.unit,
            type=ds_a.axis_type,
            data=np.array(ds_a),
        )

        a.append(new_a)

    d = PlotData(
        name=dataset.name,
        label=dataset.label,
        units=dataset.unit,
        data=np.array(dataset),
        time=np.array(dataset.time),
        time_units=dataset.time.unit,
        axes=a,
    )

    return d


@register_container_plugin(GridDataset, name="plot_type")
def grid_plot_type(dataset: GridDataset) -> PlotData:
    return DefaultGridPlotTypes[dataset.grid_dim]


@register_container_plugin(ParticleDataset, name="plot_data")
def particle_plot_data(
    dataset: ParticleDataset, quants: List[str] = []
) -> PlotData:

    a = []
    d = []

    for quant in quants:
        q = getattr(dataset, quant)
        new_a = PlotDataAxis(name=q.name, label=q.label, units=q.unit)

        a.append(new_a)
        d.append(np.array(q))

    p_d = PlotData(
        name=dataset.name,
        label=dataset.name,
        units="",
        data=d,
        time=np.array(dataset.time),
        time_units=dataset.time.unit,
        axes=a,
    )

    return p_d


@register_container_plugin(ParticleDataset, name="plot_type")
def particle_plot_type(dataset: ParticleDataset) -> PlotData:
    return DefaultParticlePlotType


@attr.s
class FigurePlanPlot:
    dataset: BaseDataset = attr.ib()
    quants: list = attr.ib(default=None, validator=optional(instance_of(list)))
    # other arguments specific to the type of dataset,
    # necessary to build the PlotData and PlotDataAxis objects
    style: dict = attr.ib(default=dict(), validator=optional(instance_of(dict)))


@attr.s
class FigurePlanAxis:
    plots: List[FigurePlanPlot] = attr.ib()
    axes: Axes = attr.ib(default=None, validator=optional(instance_of(Axes)))
    style: dict = attr.ib(default=dict(), validator=optional(instance_of(dict)))


@attr.s
class FigurePlan:
    axes: List[FigurePlanAxis] = attr.ib()
    fig: Figure = attr.ib(default=None, validator=optional(instance_of(Figure)))
    style: dict = attr.ib(default=dict(), validator=optional(instance_of(dict)))

    @property
    def datasets(self) -> list:
        d = []
        for a in self.axes:
            for p in a.plots:
                d.append(p.dataset)
        return d


@register_container_plugin(GridDataset, name="plot")
def plot_grid_dataset(
    dataset: GridDataset,
    fig: Optional[Figure] = None,
    axes: Optional[Axes] = None,
    style: dict = dict(),
    interactive: bool = True,
    n: int = 0,
):

    p_plan = FigurePlanPlot(
        dataset=dataset, style=filter_style(dataset.plot_type(), style)
    )

    a_plan = FigurePlanAxis(
        axes=axes, plots=[p_plan], style=filter_style(Axes, style)
    )

    f_plan = FigurePlan(
        fig=fig, axes=[a_plan], style=filter_style(Figure, style)
    )

    if len(dataset.iteration) > 1 and inside_notebook() and interactive:
        build_interactive_tools(f_plan, n)

    else:
        # build figure
        fig = build_figure(f_plan)
        return fig


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

    p_plan = FigurePlanPlot(
        dataset=dataset,
        quants=quants,
        style=filter_style(dataset.plot_type(), style),
    )

    a_plan = FigurePlanAxis(
        axes=axes, plots=[p_plan], style=filter_style(Axes, style)
    )

    f_plan = FigurePlan(
        fig=fig, axes=[a_plan], style=filter_style(Figure, style)
    )

    if len(dataset.iteration) > 1 and inside_notebook() and interactive:
        build_interactive_tools(f_plan, n)

    else:
        # build figure
        fig = build_figure(f_plan)
        return fig


@register_container_plugin(DatasetCollection, name="plot")
def plot_collection(
    collection: DatasetCollection,
    order: Optional[list] = list(),
    styles: Optional[dict] = dict(),
    quants: Optional[Dict[str, list]] = list(),
    interactive: bool = True,
    n: int = 0,
) -> Figure:

    # check if collection is not empty
    if not collection.store:
        raise ValueError("Collection is empty.")

    # check if time and iteration arrays are equal
    for check in ["iteration", "time"]:
        arr = [
            np.array(getattr(dataset, check))
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

        if isinstance(dataset, ParticleDataset):
            if dataset.name not in quants.keys():
                raise ValueError("quants not passed!")
            i_quants = quants[dataset.name]
        else:
            i_quants = None

        p_plan = FigurePlanPlot(
            dataset=dataset,
            quants=i_quants,
            style=filter_style(dataset.plot_type(), i_style),
        )

        a_plan = FigurePlanAxis(
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

    if len(dataset.iteration) > 1 and inside_notebook() and interactive:
        build_interactive_tools(f_plan, n)

    else:
        # build figure
        fig = build_figure(f_plan)
        return fig

    # # check if order elements exist in collection
    # for key in order:
    #     if key not in collection.store.keys():
    #         raise ValueError(
    #             f"Order key `{key}` is not a part of the collection."
    #         )

    # if len(order) > 0 and len(order) < len(collection.store):

    #     # get collection keys as list
    #     unused_keys = list(collection.store.keys())

    #     # remove elemets already in order
    #     for key in order:
    #         unused_keys.remove(key)

    #     # add unused keys to order
    #     for key in unused_keys:
    #         order.append(key)

    # elif not order:
    #     order = collection.store.keys()

    # # build figure object
    # fig_kwargs = filter_kwargs(Figure, **kwargs)

    # fig = Figure(**fig_kwargs)

    # for key in order:
    #     # get dataset
    #     dataset = collection.store[key]

    #     # get dataset plot specific kwargs
    #     ds_kwargs = {}
    #     if key in styles:
    #         ds_kwargs = styles[key]

    #     # add new axes
    #     axes_kwargs = filter_kwargs(Axes, **ds_kwargs)
    #     axes = fig.add_axes(**axes_kwargs)

    #     plot_type = DefaultGridPlotTypes[dataset.grid_dim]

    #     # TODO: make this a method of the dataset?
    #     # build plot axes object
    #     plot_axes = []

    #     for ds_axes in dataset.axes:
    #         new_axes = PlotDataAxis(
    #             name=ds_axes.name,
    #             label=ds_axes.label,
    #             units=ds_axes.unit,
    #             type=ds_axes.axis_type,
    #             data=np.array(ds_axes),
    #         )

    #         plot_axes.append(new_axes)

    #     # build data object
    #     data = PlotData(
    #         name=dataset.name,
    #         label=dataset.label,
    #         units=dataset.unit,
    #         data=np.array(dataset),
    #         time=np.array(dataset.time),
    #         time_units=dataset.time.unit,
    #         axes=plot_axes,
    #     )

    #     # build plot
    #     plot_kwargs = filter_kwargs(plot_type, **ds_kwargs)
    #     axes.add_plot(plot_type=plot_type, data=data, **plot_kwargs)

    #     axes.update()

    # fig.close()

    # return fig


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

    time = np.array(d_ref.time)
    iteration = np.array(d_ref.iteration)

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
                    FigurePlanPlot(
                        dataset=p.dataset[n], quants=p.quants, style=p.style
                    )
                )

            f_a.append(FigurePlanAxis(axes=None, plots=a_p, style=a.style))

        f_p = FigurePlan(fig=None, axes=f_a, style=f_plan.style)

        # build figure
        fig = build_figure(f_plan=f_p)

        return fig.show()

    out = widgets.interactive_output(update_figure, {"sel": slider})

    display(ui, out)
