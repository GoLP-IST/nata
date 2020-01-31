import attr
from attr.validators import optional, instance_of

import numpy as np

from nata.plots.plots.base import BasePlot

@attr.s
class LinePlot(BasePlot):

    # line plot options
    ls: str = attr.ib(
        default=None, 
        validator=optional(instance_of(str))
    )
    lw: float = attr.ib(
        default=None, 
        validator=optional(instance_of((int, float)))
    )
    c: str = attr.ib(
        default=None, 
        validator=optional(instance_of(str))
    )
    alpha: float = attr.ib(
        default=None, 
        validator=optional(instance_of((int, float)))
    )
    marker: str = attr.ib(
        default=None, 
        validator=optional(instance_of(str))
    )
    ms: float = attr.ib(
        default=None,
        validator=optional(instance_of((int, float)))
    )

    def _default_xlim(self):
        return (self._data.axes[0].min, self._data.axes[0].max)
    
    def _default_ylim(self):
        return (np.min(self._data.values), np.max(self._data.values))
    
    def _default_xlabel(self):
        return self._data.axes[0].get_label()
    
    def _default_ylabel(self):
        return self._data.get_label()

    def _default_title(self):
        return self._data.get_time_label()

    def __attrs_post_init__(self):
        
        self.build_canvas()

    def build_canvas(self):
        # get plot axes and data
        x = self._data.axes[0].values
        y = np.transpose(self._data.values)
        
        # build plot
        self._axes._ax.plot(x, y, 
            ls=self.ls, 
            lw=self.lw, 
            c=self.c, 
            alpha=self.alpha,
            marker=self.marker,
            ms=self.ms
        )

        self._axes.update()