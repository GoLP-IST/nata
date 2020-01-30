import attr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import ceil
from copy import copy

from nata.plots import PlotTypes
from nata.plots.axes import Axes
from nata.plots.data import PlotData

from nata.utils.attrs import filter_kwargs

@attr.s
class Figure: 

    # axes contained in the figure
    _axes: list = attr.ib(init=False, repr=False)

    # backend object
    _plt: attr.ib(init=False, repr=False)

    # backend figure object
    _fig: attr.ib(init=False, repr=False)

    # plotting options
    figsize: tuple = attr.ib(
        default=(9,6),
        validator=attr.validators.instance_of((tuple, np.ndarray))
    )
    facecolor: str = attr.ib(
        default="#ffffff",
        validator=attr.validators.instance_of(str)
    )
    nrows: int = attr.ib(
        default=1
    )
    ncols: int = attr.ib(
        default=1
    )
    fontsize: int = attr.ib(
        default=16
    )
    pad: int = attr.ib(
        default=10
    )

    @property
    def axes(self) -> dict:
        return {axes.index : axes for axes in self._axes}

    # TODO: add metadata to attributes to identify auto state

    def __attrs_post_init__(self):

        # initialize list of axes objects
        self.init_axes()
        
        # initialize plotting backend
        self.init_backend()

        # open figure object
        self.open()

        # set plotting style
        self.set_style(style="default")

    def init_axes(self):
        self._axes = []

    def init_backend(self):
        self._plt = plt
    
    def open(self):
        # TODO: generalize this for arbitrary backend
        self._fig = self._plt.figure(figsize=self.figsize, facecolor=self.facecolor)
    
    def close(self):
        self._plt.close(self._fig)

    def reset(self):
        self.close()
        self.open()

    def set_style(self, style="default"):
        # TODO: allow providing of a general style from arguments
        #       or from a style file

        # fonts
        # self._plt.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
        self._plt.rc('text', usetex=True)
        
        # self._plt.rc('font', size=self.fontsize, serif="Palatino")
        self._plt.rc('axes', titlesize=self.fontsize)
        self._plt.rc('axes', labelsize=self.fontsize)
        self._plt.rc('xtick', labelsize=self.fontsize)
        self._plt.rc('ytick', labelsize=self.fontsize)
        self._plt.rc('legend', fontsize=self.fontsize)
        self._plt.rc('figure', titlesize=self.fontsize)

        # padding
        self._plt.rc('xtick.major', pad=self.pad)
        self._plt.rc('ytick.major', pad=self.pad)
    
    def show(self):
        # TODO: generalize this for arbitrary backend
        dummy = self._plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = self._fig

        self._fig.tight_layout()
        self._plt.show()

    def _repr_html_(self):
        self.show()

    def add_axes(
        self,
        **kwargs
    ):

        new_index = len(self.axes)+1

        if new_index > (self.nrows * self.ncols):
            # increase number of rows
            # TODO: really?
            self.nrows += 1

            for axes in self.axes.values(): 
                axes.redo_plots()

        axes_kwargs = filter_kwargs(Axes, **kwargs)
        axes = Axes(fig=self, index=new_index, **axes_kwargs)
        self._axes.append(axes)

        return axes

    def add_plot(
        self,
        axes: Axes,
        plot: PlotTypes,
        data: PlotData,
        **kwargs
    ):
        p = axes.add_plot(plot=plot, data=data, **kwargs)

    def __mul__(self, other):

        new = copy(self)
        
        for key, axes in new.axes.items():
            if key in other.axes:
                for plot in other.axes[key]._plots:
                    axes.add_plot(
                        plot=plot.__class__,
                        data=plot._data
                    )
    
        return new

    def __add__(self, other):

        new = copy(self)
        # TODO: move this to a costum copy method?
        for axes in new.axes.values():
            axes._fig = new

        new.nrows = ceil((len(new.axes)+len(other.axes))/new.ncols)

        if new.nrows > self.nrows:
            for axes in new.axes.values():
                axes.redo_plots()

        for axes in other.axes.values():

            # this loses properties of the old axes, namely style options
            new_axes = new.add_axes()
            new_axes._plots = copy(axes._plots)

            for plot in new_axes._plots:
                plot._axes = new_axes

            new_axes.redo_plots()
    
        return new