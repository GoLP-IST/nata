from typing import Dict

import attr
import numpy as np
import matplotlib.pyplot as plt

from nata.plots.base import BasePlot
from nata.plots.grid import GridPlot1D, GridPlot2D

@attr.s
class Figure: 

    # plots contained in the figure
    plot_obj: Dict[int, BasePlot] = attr.ib(init=False, factory=dict)

    # backend object
    _plt: attr.ib(init=False)

    # backend figure object
    _fig: attr.ib(init=False)

    # plotting options
    show: bool = attr.ib(
        default=True,
        validator=attr.validators.instance_of(bool)
    )
    figsize: tuple = attr.ib(
        default=(9,6),
        validator=attr.validators.instance_of((tuple, np.ndarray))
    )
    facecolor: str = attr.ib(
        default="#ffffff",
        validator=attr.validators.instance_of(str)
    )
    fontsize: int = attr.ib(
        default=16
    )
    pad: int = attr.ib(
        default=10
    )

    @property
    def plots(self) -> np.ndarray:
        if len(self.plot_obj) == 1:
            return next(iter(self.plot_obj))

        d = np.zeros((len(self.plot_obj),))
        for i, plot in enumerate(self.plot_obj):
            d[i] = plot
        return d
    
    # TODO: add metadata to attributes to identify auto state

    def __attrs_post_init__(self, **kwargs):
        # initialize plotting backend
        self.init_backend()

        # initialize backend figure object
        self.init_fig()

        # set plotting style
        self.set_style(style="default")

    def init_backend(self):
        self._plt = plt
    
    def init_fig(self):
        self._fig = self._plt.figure(figsize=self.figsize, facecolor=self.facecolor)

    def set_style(self, style="default"):
        # TODO: allow providing of a general style from arguments
        #       or from a style file

        # fonts
        self._plt.rc('text', usetex=True)
        
        self._plt.rc('font', size=self.fontsize, serif="Palatino")
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
        if self.show:
            self._plt.show()


    def add_plot(self, plot_type, axes, data, **kwargs):
        p = plot_type(
            fig=self,
            axes=axes,
            data=data,
            **kwargs
        )

    # def update(self):

    #     if self._xlim_auto:
    #         self.xlim = (self._parent.xmin[0], self._parent.xmax[0])
        
    #     if self._ylim_auto:
    #         self.ylim = (self._parent.xmin[1], self._parent.xmax[1])

    #     if self._xlabel_auto:
    #         self._xlabel = self._parent._axes[0].get_label()

    #     if self._ylabel_auto:
    #         self._ylabel = self._parent._axes[1].get_label()

    #     if self._title_auto:
    #         self._title = self._parent.get_title()