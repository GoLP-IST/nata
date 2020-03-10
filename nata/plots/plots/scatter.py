# -*- coding: utf-8 -*-

import attr
import matplotlib.colors as clr
import numpy as np
from attr.validators import instance_of
from attr.validators import optional

from nata.plots.plots.base import BasePlot


@attr.s
class ScatterPlot(BasePlot):

    # scatter plot options
    s: float = attr.ib(
        default=0.05, validator=optional(instance_of((int, float)))
    )
    c: str = attr.ib(default=None, validator=optional(instance_of(str)))
    marker: str = attr.ib(default=None, validator=optional(instance_of(str)))
    alpha: float = attr.ib(
        default=None, validator=optional(instance_of((int, float)))
    )

    # color plot related options
    has_cb: bool = attr.ib(init=False)

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

    @cb_title.validator
    def cb_title_validator(self, attr, cb_title):
        if cb_title is None and len(self._data.axes) > 2:
            self.cb_title = self._data.axes[2].get_label(units=True)

    def _default_xlim(self):
        return (np.min(self._data.data[0]), np.max(self._data.data[0]))

    def _default_ylim(self):
        return (np.min(self._data.data[1]), np.max(self._data.data[1]))

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

    def __attrs_post_init__(self):
        self.has_cb = (self.c is None) and (len(self._data.axes) > 2)

        if self.has_cb:
            self.c = self._data.data[2]

        super().__attrs_post_init__()

    def build_canvas(self):
        # get plot axes and data
        x = self._data.data[0]
        y = self._data.data[1]

        # build color map norm
        if self.cb_scale == "log":
            self.cb_norm = clr.LogNorm(vmin=self.vmin, vmax=self.vmax)
        elif self.cb_scale == "symlog":
            self.cb_norm = clr.SymLogNorm(
                vmin=self.vmin, vmax=self.vmax, linthresh=self.cb_linthresh
            )
        else:
            self.cb_norm = clr.Normalize(vmin=self.vmin, vmax=self.vmax)

        # build plot
        self._h = self._axes._ax.scatter(
            x,
            y,
            s=self.s,
            c=self.c,
            marker=self.marker,
            alpha=self.alpha,
            label=self.label,
            cmap=self.cb_map,
            norm=self.cb_norm
            # antialiased=self.antialiased,
        )

        if self.has_cb:
            self._axes.colorbar(plot=self)
