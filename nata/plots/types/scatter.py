# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional

import matplotlib.colors as clr
import numpy as np

from nata.plots.types import BasePlot


@dataclass
class ScatterPlot(BasePlot):
    """Color plot class.

    Parameters
    ----------
    s: ``float``, optional
        Marker size in in points**2. If not provided, defaults to ``0.1``

    c: ``str``, optional
        Color of the markers. See :mod:`matplotlib.colors` for available
        options.

    marker: ``str``, optional
        Marker style. See :mod:`matplotlib.markers` for available options.

    alpha: ``float``, optional
        Marker alpha value. Must be between ``0`` and ``1``.

    vmin: ``float``, optional
        Minimum of the colorbar axis. If not provided, it is
        inferred from the dataset represented in the plot.

    vmax: ``float``, optional
        Same as ``vmin`` for the maximum of the colorbar axis.

    cb_title: ``str``, optional
        Colorbar title. If not provided, it is inferred from the dataset
        represented in the plot.

    cb_scale: ``{'linear','log', 'symlog'}``, optional
        Scale of the colorbar. If not provided, defaults to ``'linear'``.

    cb_map: ``str``, optional
        Colormap used to represent the data. See
        :func:`matplotlib.pyplot.colormaps` for available options. If not
        provided, defaults to ``rainbow``.

    cb_linthresh: ``float``, optional
        Range within which the colorbar axis is linear. Applicable only when
        ``cb_scale`` is set to ``'symlog'``. If not provided, defaults to
        ``1e-5``.

    Notes
    -----
    All colorbar parameters are only applicable if the dataset represented in
    the plot has a quantity to be represented in color. In this case, ``c`` is
    overriden if set.

    """

    s: Optional[float] = 0.1
    c: Optional[str] = None
    marker: Optional[str] = None
    alpha: Optional[float] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    cb_map: Optional[str] = "rainbow"
    cb_scale: Optional[str] = "linear"
    cb_linthresh: Optional[float] = 1e-5
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
        return f"${self.data.axes[0].units}$" if self.data.axes[0].units else ""

    def _yunits(self):
        return f"${self.data.axes[1].units}$" if self.data.axes[1].units else ""

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
