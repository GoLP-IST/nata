from typing import ValuesView

import attr
import numpy as np
import matplotlib.pyplot as plt
import math

from nata.plots.base import BasePlot
from nata.plots.grid import GridPlot1D, GridPlot2D

@attr.s
class Figure: 

    # plots contained in the figure
    plot_objs: np.ndarray = attr.ib(init=False, repr=False)

    # backend object
    _plt: attr.ib(init=False)

    # backend figure object
    _fig: attr.ib(init=False)

    # plotting options
    if_show: bool = attr.ib(
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
        if len(self.plot_objs) == 1:
            return next(iter(self.plot_objs))

        return self.plot_objs
    
    # TODO: add metadata to attributes to identify auto state

    def __attrs_post_init__(self, **kwargs):

        self.plot_objs = ()
        
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

    def reset_fig(self):
        self._plt.close(self._fig)
        self.init_fig()

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
        if self.if_show:
            self._fig.tight_layout()
            self._plt.show()


    def add_plot(self, plot_type, axes, data, **kwargs):
        
        fig_pos = 111
        
        # atm, nata uses one columns
        req_num = len(self.plot_objs) + 1
        if req_num > 1:
            # reset figure
            self.reset_fig()
            
            # redefine figure position for existing plots
            nrows = math.ceil(req_num / 2)
            ncols = 2
            offset = nrows * 100 + ncols * 10
            fig_pos = offset + 1

            for plot in self.plot_objs:
                plot.fig_pos = fig_pos
                plot.build_canvas()
                fig_pos += 1

        # build plot
        p = plot_type(
            fig=self,
            fig_pos=fig_pos,
            axes=axes,
            data=data,
            **kwargs
        )

        # store plot in array of figure plots
        self.plot_objs = np.append(self.plot_objs, p)

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