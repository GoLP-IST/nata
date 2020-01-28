from typing import Set, Dict

from nata.utils.exceptions import NataInvalidPlot

from nata.containers import DatasetCollection, BaseDataset, \
                            GridDataset, ParticleDataset
from nata.plugins.register import register_container_plugin

from nata.plots.axis import PlotAxis
from nata.plots.data import PlotData
from nata.plots.figure import Figure
from nata.plots.grid import GridPlotObj
from nata.plots.particle import ParticlePlot1D

from nata.utils.attrs import filter_kwargs

import numpy as np

@register_container_plugin(GridDataset, name="plot")
def plot_grid_dataset(dataset, fig=None, **kwargs):

    # raise error if dataset has more than one data object
    if   len(dataset.grid_obj) != 1:
        raise NataInvalidPlot

    # build axes object
    axes = np.empty(dataset.dimension, dtype=PlotAxis)

    for i in range(dataset.dimension):
        axes[i] = PlotAxis(
            name=dataset.axes_names[i],
            label=dataset.axes_labels[i],
            units=dataset.axes_units[i],
            xtype="linear",
            xmin=dataset.axes_min[i],
            xmax=dataset.axes_max[i],
            nx=dataset.shape[i]
        )
    
    # build data object
    data = PlotData(
        name=dataset.name,
        label=dataset.label,
        units=dataset.unit,
        values=dataset.data,
        time=dataset.time,
        time_units=dataset.time_unit
    )

    # build figure object is no figure is passed as argument
    if fig is None:
        fig_kwargs = filter_kwargs(Figure, **kwargs)
        fig = Figure(**fig_kwargs)
        
    # get grid plot object
    obj  = GridPlotObj[dataset.dimension]

    # add plot to figure object
    plot_kwargs = filter_kwargs(obj, **kwargs)
    fig.add_plot(obj, axes, data, **plot_kwargs)

    return fig

@register_container_plugin(ParticleDataset, name="plot")
def plot_particle_dataset(dataset, sel=None, fig=None, **kwargs):

    # raise error if dataset has more than one data object
    if   len(dataset.prt_objs) != 1:
        raise NataInvalidPlot

    axes = np.empty(2, dtype=PlotAxis)
    for i in range(2):
        idx = np.argwhere(dataset.quantities == sel[i])
        name = dataset.quantities[idx[0]][0]
        
        axes[i] = PlotAxis(
            name=name,
            label=dataset.quantities_labels[name],
            units=dataset.quantities_units[name],
            xtype="linear",
            xmin=np.min(dataset.data[name]),
            xmax=np.max(dataset.data[name]),
            nx=10
        )

    # build data object
    data = PlotData(
        name=dataset.name,
        values=np.array([
            dataset.data[sel[0]],
            dataset.data[sel[1]]
        ]),
        time=dataset.time,
        time_units=dataset.time_units
    )

    # build figure object is no figure is passed as argument
    if fig is None:
        fig_kwargs = filter_kwargs(Figure, **kwargs)
        fig = Figure(**fig_kwargs)

    # add plot to figure object
    plot_kwargs = filter_kwargs(ParticlePlot1D, **kwargs)
    fig.add_plot(ParticlePlot1D, axes, data, **plot_kwargs)

    return fig

@register_container_plugin(DatasetCollection, name="plot")
def plot_collection(collection, order=[], styles={}, **kwargs):
    if not collection.store:
        raise ValueError(
            "Collection is empty."
        ) 

    # check if order elements exist in collection
    for key in order:
        if key not in collection.store.keys():
            raise ValueError(
            f"Order key `{key}` is not a part of the collection."
        )

    # build figure object
    fig_kwargs = filter_kwargs(Figure, **kwargs)
    
    fig = Figure(**fig_kwargs)

    num_order = len(order)
    
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


    for key in order:
        # get dataset
        dataset = collection.store[key]

        # get dataset plot specific kwargs
        plt_kwargs = {}
        if key in styles:
            plt_kwargs = styles[key]

        # build plot object
        fig = dataset.plot(fig=fig, **plt_kwargs)
    
    return fig