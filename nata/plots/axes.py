from typing import Optional

import attr
from attr.validators import optional, instance_of, and_, in_

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

from copy import copy

# from nata.plots import Figure
from nata.plots import PlotTypes
from nata.plots import PlotData

from nata.utils.attrs import filter_kwargs

@attr.s
class Axes: 

    # plots contained in the axes
    _plots: list = attr.ib(init=False, repr=False)

    # parent figure object
    _fig = attr.ib(repr=False)

    # backend axes object
    _ax: attr.ib(init=False, repr=False)

    # axes index in the parent figure object
    index: int = attr.ib()

    # backend legend object
    _legend: attr.ib(init=False, repr=True, default=None)

    # backend colorbar object
    _cb: attr.ib(init=False, repr=True, default=None)

    # the attributes below are property of axes, 
    # however their validation and default assignment is done
    # by their child plot objects

    # axes limit options
    xlim: tuple = attr.ib(
        default=None,
        validator=optional(instance_of(tuple))
    )
    
    ylim: tuple = attr.ib(
        default=None,
        validator=optional(instance_of(tuple))
    )

    # axes label options
    xlabel: str = attr.ib(
        default=None,
        validator=optional(instance_of(str))
    )
    ylabel: str = attr.ib(
        default=None,
        validator=optional(instance_of(str))
    )
    title: str = attr.ib(
        default=None,
        validator=optional(instance_of(str))
    )

    # axes scale options
    xscale: str = attr.ib(
        default="linear", 
        validator=optional(and_(
            instance_of(str),
            in_(("linear", "log", "symlog"))
        ))
    )
    yscale: str = attr.ib(
        default="linear", 
        validator=optional(and_(
            instance_of(str),
            in_(("linear", "log", "symlog"))
        ))
    )
    aspect: str = attr.ib(
        default="auto", 
        validator=optional(and_(
            instance_of(str),
            in_(("equal", "auto"))
        ))
    )

    # legend options
    legend_show: bool = attr.ib(
        default=True, 
        validator=optional(instance_of(bool))
    )
    legend_loc: str = attr.ib(
        default="best", 
        validator=optional(instance_of(str))
    )
    legend_frameon: bool = attr.ib(
        default=False, 
        validator=optional(instance_of(bool))
    )

    # colorbar options
    cb_show: bool = attr.ib(
        default=True, 
        validator=optional(instance_of(bool))
    )

    @property
    def plots(self) -> list:
        return self._plots
    
    def copy(self):
        new = copy(self)

        for plot in new._plots:
            plot._axes = new

        return new        

    def __attrs_post_init__(self):
        # initialize list of plot objects
        self.init_plots()
        
        # initialize axes backend
        self.init_backend()
    
    def init_plots(self):
        self._plots = []

    def init_backend(self):
        # TODO: generalize this for arbitrary backend
        self._ax = self._fig._fig.add_subplot(
            self._fig.nrows,
            self._fig.ncols,
            self.index
        )

        self._legend = None
        self._cb = None
        
    def clear_backend(self):
        for plot in self._plots:
            plot.clear()

        self.clear_colorbar()

        # TODO: generalize this for arbitrary backend
        self._ax.clear()
        self._ax.remove()
        self._ax = None

    def reset_backend(self):
        self.clear_backend()
        self.init_backend()

    def add_plot(
        self,
        plot: PlotTypes,
        data: PlotData,
        **kwargs
    ):
        plot_kwargs = filter_kwargs(plot, **kwargs)
        p = plot(axes=self, data=data, **kwargs)
        self._plots.append(p)

        return p

    def redo_plots(self):
        
        self.reset_backend()
        
        for plot in self.plots:
            plot.build_canvas()

    def update(self):
        ax = self._ax
        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        # set axes labels
        ax.set_xlabel(self.xlabel, labelpad=self._fig.pad)
        ax.set_ylabel(self.ylabel, labelpad=self._fig.pad)
        
        # set title
        ax.set_title(label=self.title, pad=self._fig.pad)

        # set aspect ratio
        ax.set_aspect(self.aspect)
    
    def legend(self):
        labels = [p.label or "" for p in self._plots]

        if self.legend_show and labels:
            # show legend
            self._legend = self._ax.legend(
                labels = labels,
                loc=self.legend_loc, 
                frameon=self.legend_frameon
            )

    def colorbar(
        self,
        plot: PlotTypes
    ):

        if self.cb_show:
            # show colorbar
            self._cb = self._ax.get_figure().colorbar(plot._h, aspect=30)
            
            # set colorbar title
            self._cb.set_label(
                label=plot.cb_title, 
                labelpad=self._fig.pad
            )
    
    def clear_colorbar(self):
        if self._cb:
            self._cb.remove()