# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional
from typing import Union

import matplotlib.colors as clr
import numpy as np

from nata.plots.types import BasePlot


@dataclass
class ScatterPlot(BasePlot):
    s: Optional[Union[int, float]] = 0.05
    c: Optional[str] = None
    marker: Optional[str] = None
    alpha: Optional[Union[int, float]] = None
    vmin: Optional[Union[int, float]] = None
    vmax: Optional[Union[int, float]] = None
    cb_map: Optional[str] = "rainbow"
    cb_scale: Optional[str] = "linear"
    cb_linthresh: Optional[Union[int, float]] = 1e-5
    cb_title: Optional[str] = None

    def __post_init__(self):
        if self.cb_title is None and len(self.data.axes) > 2:
            self.cb_title = self.data.axes[2].get_label(units=True)

        if self.has_cb:
            self.c = self.data.data[2]

        super().__post_init__()

    @property
    def has_cb(self):
        return len(self.data.axes) > 2

    def _default_xlim(self):
        return (np.min(self.data.data[0]), np.max(self.data.data[0]))

    def _default_ylim(self):
        return (np.min(self.data.data[1]), np.max(self.data.data[1]))

    def _default_xlabel(self, units=True):
        return self.data.axes[0].get_label(units)

    def _default_ylabel(self, units=True):
        return self.data.axes[1].get_label(units)

    def _default_title(self):
        return self.data.get_time_label()

    def _default_label(self):
        return self.data.get_label(units=False)

    def _xunits(self):
        return f"${self.data.axes[0].units}$"

    def _yunits(self):
        return f"${self.data.axes[1].units}$"

    def build_canvas(self):
        # get plot axes and data
        x = self.data.data[0]
        y = self.data.data[1]

        # build color map norm
        if self.cb_scale == "log":
            self.cb_norm = clr.LogNorm(vmin=self.vmin, vmax=self.vmax)
        elif self.cb_scale == "symlog":
            self.cb_norm = clr.SymLogNorm(
                vmin=self.vmin,
                vmax=self.vmax,
                linthresh=self.cb_linthresh,
                base=10,
            )
        else:
            self.cb_norm = clr.Normalize(vmin=self.vmin, vmax=self.vmax)

        # build plot
        self.h = self.axes.ax.scatter(
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
            self.axes.init_colorbar(plot=self)
