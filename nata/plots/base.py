from abc import ABC, abstractmethod

import attr
import numpy as np

from nata.plots.data import PlotData

@attr.s
class BasePlot(ABC):

    fig = attr.ib()
    fig_pos: int = attr.ib()

    axes: np.ndarray = attr.ib()
    data: PlotData = attr.ib()

    # axes limit options
    xlim: tuple = attr.ib(
        validator=attr.validators.instance_of((tuple, np.ndarray))
    )
    ylim: tuple = attr.ib(
        validator=attr.validators.instance_of((tuple, np.ndarray))
    )

    # axes label options
    xlabel: str = attr.ib(
        validator=attr.validators.instance_of(str)
    )
    ylabel: str = attr.ib(
        validator=attr.validators.instance_of(str)
    )
    title: str = attr.ib(
        validator=attr.validators.instance_of(str)
    )

    # axes scale options
    xscale: str = attr.ib(
        default="linear", 
        validator=attr.validators.instance_of(str)
    )
    yscale: str = attr.ib(
        default="linear", 
        validator=attr.validators.instance_of(str)
    )
    aspect: str = attr.ib(
        default="auto", 
        validator=attr.validators.instance_of(str)
    )

    # backend axes object
    _ax: attr.ib(init=False)

    # decorators for default values
    @xlim.default
    def _default_xlim_proxy(self):
        return self._default_xlim()

    @ylim.default
    def _default_ylim_proxy(self):
        return self._default_ylim()

    @xlabel.default
    def _default_xlabel_proxy(self):
        return self._default_xlabel()

    @ylabel.default
    def _default_ylabel_proxy(self):
        return self._default_ylabel()

    @title.default
    def _default_title_proxy(self):
        return self._default_title()


    def _default_xlim(self):
        return ()

    def _default_ylim(self):
        return ()

    def _default_xlabel(self):
        return ""

    def _default_ylabel(self):
        return ""

    def _default_title(self):
        return ""

    @property
    def _backend_fig(self):
        return self.fig._fig

    @property
    def _backend_colors(self):
        # TODO: make this a function of the backend?
        return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22', '#17becf']