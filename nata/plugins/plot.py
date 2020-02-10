# -*- coding: utf-8 -*-
from typing import Optional

from nata.containers import DatasetCollection
from nata.containers import GridDataset
from nata.plots import Axes
from nata.plots import DefaultGridPlotTypes
from nata.plots import Figure
from nata.plots import PlotData
from nata.plots import PlotDataAxis
from nata.plugins.register import register_container_plugin
from nata.utils.attrs import filter_kwargs
from nata.utils.exceptions import NataInvalidPlot

# from nata.plots.grid import GridPlotObj
# from nata.plots.particle import ParticlePlot1D


@register_container_plugin(GridDataset, name="plot")
def plot_grid_dataset(
    dataset: GridDataset,
    fig: Optional[Figure] = None,
    axes: Optional[Axes] = None,
    **kwargs,
) -> Figure:

    # raise error if dataset has more than one data object
    if len(dataset) != 1:
        raise NataInvalidPlot

    # 1. build figure
    if fig is None:

        fig_kwargs = filter_kwargs(Figure, **kwargs)
        print(fig_kwargs)
        fig = Figure(**fig_kwargs)

        # ignore axes
        axes = None

    # 2. build axes
    if axes is None:

        axes_kwargs = filter_kwargs(Axes, **kwargs)
        axes = fig.add_axes(**axes_kwargs)

    # 3. get default plot type for grids
    # TODO: make this an argument?
    plot_type = DefaultGridPlotTypes[dataset.grid_dim]

    # TODO: make this a method of the dataset?
    # build plot axes object
    plot_axes = []

    for ds_axes in dataset.axes:
        new_axes = PlotDataAxis(
            name=ds_axes.name,
            label=ds_axes.label,
            units=ds_axes.unit,
            type=ds_axes.axis_type,
            min=ds_axes.min[0],
            max=ds_axes.max[0],
            n=ds_axes.length,
        )

        plot_axes.append(new_axes)

    # build data object
    data = PlotData(
        name=dataset.name,
        label=dataset.label,
        units=dataset.unit,
        values=dataset.data[0],
        time=dataset.time.asarray()[0],
        time_units=dataset.time.unit,
        axes=plot_axes,
    )

    # 4. build plot
    plot_kwargs = filter_kwargs(plot_type, **kwargs)
    axes.add_plot(plot_type=plot_type, data=data, **plot_kwargs)

    axes.update()

    fig.close()

    return fig


# @register_container_plugin(ParticleDataset, name="plot")
# def plot_particle_dataset(dataset, sel=None, fig=None, **kwargs):

#     # raise error if dataset has more than one data object
#     if   len(dataset.prt_objs) != 1:
#         raise NataInvalidPlot

#     axes = np.empty(2, dtype=PlotAxis)
#     for i in range(2):
#         idx = np.argwhere(dataset.quantities == sel[i])
#         name = dataset.quantities[idx[0]][0]

#         axes[i] = PlotAxis(
#             name=name,
#             label=dataset.quantities_labels[name],
#             units=dataset.quantities_units[name],
#             xmin=np.min(dataset.data[name]),
#             xmax=np.max(dataset.data[name])
#         )

#     # build data object
#     data = PlotData(
#         name=dataset.name,
#         values=np.array([
#             dataset.data[sel[0]],
#             dataset.data[sel[1]]
#         ]),
#         time=dataset.time,
#         time_units=dataset.time_units
#     )

#     # build figure object is no figure is passed as argument
#     if fig is None:
#         fig_kwargs = filter_kwargs(Figure, **kwargs)
#         fig = Figure(**fig_kwargs)

#     # add plot to figure object
#     plot_kwargs = filter_kwargs(ParticlePlot1D, **kwargs)
#     fig.add_plot(ParticlePlot1D, axes, data, **plot_kwargs)

#     return fig


@register_container_plugin(DatasetCollection, name="plot")
def plot_collection(
    collection: DatasetCollection,
    order: Optional[list] = [],
    styles: Optional[dict] = {},
    # fig: Optional[Figure] = None,
    # axes: Optional[Axes] = None,
    **kwargs,
) -> Figure:

    # check if collection is not empty
    if not collection.store:
        raise ValueError("Collection is empty.")

    # check if order elements exist in collection
    for key in order:
        if key not in collection.store.keys():
            raise ValueError(
                f"Order key `{key}` is not a part of the collection."
            )

    if len(order) > 0 and len(order) < len(collection.store):

        # get collection keys as list
        unused_keys = list(collection.store.keys())

        # remove elemets already in order
        for key in order:
            unused_keys.remove(key)

        # add unused keys to order
        for key in unused_keys:
            order.append(key)

    elif not order:
        order = collection.store.keys()

    # build figure object
    fig_kwargs = filter_kwargs(Figure, **kwargs)

    fig = Figure(**fig_kwargs)

    for key in order:
        # get dataset
        dataset = collection.store[key]

        # get dataset plot specific kwargs
        ds_kwargs = {}
        if key in styles:
            ds_kwargs = styles[key]

        # add new axes
        axes_kwargs = filter_kwargs(Axes, **ds_kwargs)
        axes = fig.add_axes(**axes_kwargs)

        plot_type = DefaultGridPlotTypes[dataset.grid_dim]

        # TODO: make this a method of the dataset?
        # build plot axes object
        plot_axes = []

        for ds_axes in dataset.axes:
            new_axes = PlotDataAxis(
                name=ds_axes.name,
                label=ds_axes.label,
                units=ds_axes.unit,
                type=ds_axes.axis_type,
                min=ds_axes.min[0],
                max=ds_axes.max[0],
                n=ds_axes.length,
            )

            plot_axes.append(new_axes)

        # build data object
        data = PlotData(
            name=dataset.name,
            label=dataset.label,
            units=dataset.unit,
            values=dataset.data[0],
            time=dataset.time.asarray()[0],
            time_units=dataset.time.unit,
            axes=plot_axes,
        )

        # build plot
        plot_kwargs = filter_kwargs(plot_type, **ds_kwargs)
        axes.add_plot(plot_type=plot_type, data=data, **plot_kwargs)

        axes.update()

    fig.close()

    return fig
