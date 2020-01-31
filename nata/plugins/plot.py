from typing import Union, Optional

from nata.utils.exceptions import NataInvalidPlot

from nata.containers import DatasetCollection, BaseDataset, \
                            GridDataset, ParticleDataset
from nata.plugins.register import register_container_plugin

from nata.plots import PlotDataAxis, PlotData
from nata.plots import Figure
from nata.plots import Axes
from nata.plots import DefaultGridPlotTypes

# from nata.plots.grid import GridPlotObj
# from nata.plots.particle import ParticlePlot1D

from nata.utils.attrs import filter_kwargs

import numpy as np

@register_container_plugin(GridDataset, name="plot")
def plot_grid_dataset(
    dataset: GridDataset,
    fig: Optional[Figure] = None,
    axes: Optional[Axes] = None,
    **kwargs
) -> Figure:

    # raise error if dataset has more than one data object
    if len(dataset) != 1:
        raise NataInvalidPlot

    # build plot axes object
    plot_axes = []

    for axes in dataset.axes:
        new_axes = PlotDataAxis(
            name=axes.name,
            label=axes.label,
            units=axes.unit,
            type=axes.axis_type,
            min=axes.min,
            max=axes.max,
            n=axes.length
        )

        plot_axes.append(new_axes)
    
    # build data object
    data = PlotData(
        name=dataset.name,
        label=dataset.label,
        units=dataset.unit,
        values=dataset.data,
        time=0., #dataset.time,
        time_units="", #dataset.time.unit,
        axes=plot_axes
    )

    # 1. build figure
    if fig is None:
        
        fig_kwargs = filter_kwargs(Figure, **kwargs)
        fig = Figure(**fig_kwargs)

        # ignore axes
        axes = None
    
    # 2. build axes
    if axes is None:

        axes_kwargs = filter_kwargs(Axes, **kwargs)
        axes = fig.add_axes(**axes_kwargs)
        
    # 3. get default plot object for grids
    # TODO: make this an argument?
    plot = DefaultGridPlotTypes[dataset.grid_dim]

    # 4. build plot
    plot_kwargs = filter_kwargs(plot, **kwargs)
    axes.add_plot(
        plot=plot,
        data=data,
        **plot_kwargs
    )

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

# @register_container_plugin(DatasetCollection, name="plot")
# def plot_collection(collection, order=[], styles={}, **kwargs):
#     if not collection.store:
#         raise ValueError(
#             "Collection is empty."
#         ) 

#     # check if order elements exist in collection
#     for key in order:
#         if key not in collection.store.keys():
#             raise ValueError(
#             f"Order key `{key}` is not a part of the collection."
#         )

#     # build figure object
#     fig_kwargs = filter_kwargs(Figure, **kwargs)
    
#     fig = Figure(**fig_kwargs)

#     num_order = len(order)
    
#     if len(order) > 0 and len(order) < len(collection.store):
        
#         # get collection keys as list
#         unused_keys = list(collection.store.keys())

#         # remove elemets already in order
#         for key in order:
#             unused_keys.remove(key)

#         # add unused keys to order
#         for key in unused_keys:
#             order.append(key)
        
#     elif not order:
#         order = collection.store.keys()

#     for key in order:
#         # get dataset
#         dataset = collection.store[key]

#         # get dataset plot specific kwargs
#         plt_kwargs = {}
#         if key in styles:
#             plt_kwargs = styles[key]

#         # build plot object
#         fig = dataset.plot(fig=fig, **plt_kwargs)
    
#     return fig