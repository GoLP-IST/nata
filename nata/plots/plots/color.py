import attr
from attr.validators import optional, instance_of

import numpy as np
import matplotlib.colors as clr

from nata.plots.plots.base import BasePlot

@attr.s
class ColorPlot(BasePlot):

    # color plot options
    vmin: float = attr.ib(
        default=None, 
        validator=optional(instance_of((int, float)))
    )
    vmax: float = attr.ib(
        default=None, 
        validator=optional(instance_of((int, float)))
    )
    cb_draw: bool = attr.ib(
        default=True, 
        validator=optional(instance_of(bool))
    )
    cb_map: str = attr.ib(
        default="rainbow", 
        validator=optional(instance_of(str))
    )
    cb_scale: str = attr.ib(
        default="linear", 
        validator=optional(instance_of(str))
    )
    cb_linthresh: float = attr.ib(
        default=1e-5, 
        validator=optional(instance_of((int, float)))
    )
    cb_title: str = attr.ib(
        default=None, 
        validator=optional(instance_of(str))
    )

    @cb_title.validator
    def cb_title_validator(self, attr, cb_title):
        if cb_title is None:
            self.cb_title = self._data.get_label()

    def _default_xlim(self):
        return (self._data.axes[0].min, self._data.axes[0].max)
    
    def _default_ylim(self):
        return (self._data.axes[1].min, self._data.axes[1].max)
    
    def _default_xlabel(self):
        return self._data.axes[0].get_label()
    
    def _default_ylabel(self):
        return self._data.axes[1].get_label()

    def _default_title(self):
        return self._data.get_time_label()

    def __attrs_post_init__(self):
        
        self.build_canvas()

    def build_canvas(self):
        # get plot axes and data
        x = self._data.axes[0].values
        y = self._data.axes[1].values
        z = np.transpose(self._data.values)
        
        # build color map norm
        if   self.cb_scale == "log":
            # convert values to positive numbers
            z = np.abs(z) + 1e-16

            # adjust min and max values
            # TODO: do this only if vmin was not init
            self.vmin = np.min(z)
            
            # if self.vmax_auto:
            self.vmax = np.max(z)

            # set color map norm
            self.cb_norm = clr.LogNorm(
                vmin=self.vmin, 
                vmax=self.vmax
            )
        elif self.cb_scale == "symlog":
            # set color map norm
            self.cb_norm = clr.SymLogNorm(
                vmin=self.vmin, 
                vmax=self.vmax,
                linthresh=self.cb_linthresh
            )
        else:
            self.cb_norm = clr.Normalize(
                vmin=self.vmin, 
                vmax=self.vmax
            )
            
        # build plot
        c = self._axes._ax.pcolormesh(
            x, 
            y, 
            z,
            cmap=self.cb_map,
            norm=self.cb_norm,
            antialiased=False
            )

        if self.cb_draw:
            # draw colorbar
            self.cb = self._axes._ax.get_figure().colorbar(c, aspect=30)
            
            # set colorbar title
            self.cb.set_label(
                label=self.cb_title, 
                labelpad=self._axes._fig.pad
            )

        self._axes.update()

    def clear(self):
        if self.cb:
            self.cb.remove()