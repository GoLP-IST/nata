from nata.utils.exceptions import NataInvalidPlot

from nata.containers import BaseDataset, GridDataset, ParticleDataset
from nata.plugins.register import register_container_plugin

from nata.plots.axis import PlotAxis
from nata.plots.data import PlotData
from nata.plots.figure import Figure
from nata.plots.grid import GridPlot1D, GridPlot2D
from nata.plots.particle import ParticlePlot1D

from nata.utils.attrs import filter_kwargs

import numpy as np

@register_container_plugin(GridDataset, name="plot")
def plot_grid_dataset(dataset, fig=None, **kwargs):
    if   dataset.dimension == 1:
        
        # raise error if dataset has more than one data object
        if   len(dataset.grid_obj) != 1:
            raise NataInvalidPlot

        # build axes object
        axes = np.empty(dataset.dimension, dtype=PlotAxis)
        axes[0] = PlotAxis(
            name=dataset.axes_names[0],
            label=dataset.axes_labels[0],
            units=dataset.axes_units[0],
            xtype="linear",
            xmin=dataset.axes_min[0],
            xmax=dataset.axes_max[0],
            nx=dataset.shape[0]
        )
        
        # build data object
        data = PlotData(
            name=dataset.name,
            label=dataset.label,
            units=dataset.unit,
            values=dataset.data,
            time=dataset.time,
            time_units=dataset.time_units
        )
        
        # build figure object is no figure is passed as argument
        if fig is None:
            fig_kwargs = filter_kwargs(Figure, **kwargs)
            fig = Figure(**fig_kwargs)

        # add plot to figure object
        plot_kwargs = filter_kwargs(GridPlot1D, **kwargs)
        fig.add_plot(GridPlot1D, axes, data, **plot_kwargs)

        # show figure
        fig.show()

        return fig
            

    elif (dataset.dimension == 2):

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
            time_units=dataset.time_units
        )

        # build figure object is no figure is passed as argument
        if fig is None:
            fig_kwargs = filter_kwargs(Figure, **kwargs)
            fig = Figure(**fig_kwargs)

        # add plot to figure object
        plot_kwargs = filter_kwargs(GridPlot2D, **kwargs)
        fig.add_plot(GridPlot2D, axes, data, **plot_kwargs)

        # show figure
        fig.show()

        return fig


@register_container_plugin(ParticleDataset, name="plot")
def plot_particle_dataset(dataset, sel=None, fig=None, show=True, **kwargs):

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
    plot_kwargs = filter_kwargs(GridPlot1D, **kwargs)
    fig.add_plot(ParticlePlot1D, axes, data, **plot_kwargs)

    # show figure
    fig.show()

    return fig

# @register_container_plugin(DatasetCollection, name="plot")
# def plot_collection(collection, show=True, **kwargs):
#     if   isinstance(collection, GridDataset):
#         if   dataset.dimension == 1:
#             if   len(dataset.grid_obj) == 1:
#                 axes = np.empty(dataset.dimension, dtype=PlotAxis)
#                 axes[0] = PlotAxis(
#                     name=dataset.axes_names[0],
#                     label=dataset.axes_labels[0],
#                     units=dataset.axes_units[0],
#                     xtype="linear",
#                     xmin=dataset.axes_min[0],
#                     xmax=dataset.axes_max[0],
#                     nx=dataset.shape[0]
#                 )

#                 data = PlotData(
#                     name=dataset.name,
#                     label=dataset.label,
#                     units=dataset.unit,
#                     data=dataset.data,
#                     time=dataset.time,
#                     time_units=dataset.time_units
#                 )

#                 dataset._p = GridPlot1D(
#                     parent=dataset,
#                     axes=axes,
#                     data=data,
#                     show=show, 
#                     **kwargs
#                 )
#             else:
#                 NotImplementedError("Not yet implemented")
#     else:
#         raise NotImplementedError("Not yet implemented")
