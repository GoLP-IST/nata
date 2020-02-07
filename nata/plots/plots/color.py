# -*- coding: utf-8 -*-
import attr
import matplotlib.colors as clr
import numpy as np
from attr.validators import instance_of
from attr.validators import optional

from nata.plots.plots.base import BasePlot


@attr.s
class ColorPlot(BasePlot):

    # color plot options
    vmin: float = attr.ib(
        default=None, validator=optional(instance_of((int, float)))
    )
    vmax: float = attr.ib(
        default=None, validator=optional(instance_of((int, float)))
    )
    cb_map: str = attr.ib(
        default="rainbow", validator=optional(instance_of(str))
    )
    cb_scale: str = attr.ib(
        default="linear", validator=optional(instance_of(str))
    )
    cb_linthresh: float = attr.ib(
        default=1e-5, validator=optional(instance_of((int, float)))
    )
    cb_title: str = attr.ib(default=None, validator=optional(instance_of(str)))
    interpolation: str = attr.ib(
        default="none", validator=optional(instance_of(str))
    )

    @cb_title.validator
    def cb_title_validator(self, attr, cb_title):
        if cb_title is None:
            self.cb_title = self._data.get_label()

    def _default_xlim(self):
        return (self._data.axes[0].min, self._data.axes[0].max)

    def _default_ylim(self):
        return (self._data.axes[1].min, self._data.axes[1].max)

    def _default_xlabel(self, units=True):
        return self._data.axes[0].get_label(units)

    def _default_ylabel(self, units=True):
        return self._data.axes[1].get_label(units)

    def _default_title(self):
        return self._data.get_time_label()

    def _default_label(self):
        return self._data.get_label(units=False)

    def _xunits(self):
        return f"${self._data.axes[0].units}$"

    def _yunits(self):
        return f"${self._data.axes[1].units}$"

    def build_canvas(self):
        # get plot extent and data
        extent = (
            self._data.axes[0].min,
            self._data.axes[0].max,
            self._data.axes[1].min,
            self._data.axes[1].max,
        )
        z = np.transpose(self._data.values)

        # build color map norm
        if self.cb_scale == "log":
            # convert values to positive numbers
            z = np.abs(z) + 1e-16

            # adjust min and max values
            # TODO: do this only if vmin was not init
            # self.vmin = np.min(z)

            # if self.vmax_auto:
            # self.vmax = np.max(z)

            # set color map norm
            self.cb_norm = clr.LogNorm(vmin=self.vmin, vmax=self.vmax)
        elif self.cb_scale == "symlog":
            # set color map norm
            self.cb_norm = clr.SymLogNorm(
                vmin=self.vmin, vmax=self.vmax, linthresh=self.cb_linthresh
            )
        else:
            self.cb_norm = clr.Normalize(vmin=self.vmin, vmax=self.vmax)

        # build plot
        self._h = self._axes._ax.imshow(
            z,
            extent=extent,
            origin="lower",
            cmap=self.cb_map,
            norm=self.cb_norm,
            interpolation=self.interpolation,
        )

        self._axes.colorbar(plot=self)
